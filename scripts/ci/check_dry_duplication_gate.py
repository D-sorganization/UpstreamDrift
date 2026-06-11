#!/usr/bin/env python3
"""Block unapproved duplicated logic blocks in production source files."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import logging
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("scripts/config/dry_duplication_gate.json")
DEFAULT_OWNER = "@core"
DEFAULT_ISSUE = "#7315"
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|[-+*/%]")


@dataclass(frozen=True)
class Occurrence:
    """Location of one duplicated logic window."""

    path: Path
    start_line: int
    end_line: int

    def format(self, repo_root: Path) -> str:
        rel = self.path.relative_to(repo_root).as_posix()
        return f"{rel}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class DuplicateFinding:
    """A duplicate fingerprint whose occurrence count exceeds its baseline."""

    fingerprint: str
    occurrence_count: int
    baseline_max_occurrences: int
    sample: tuple[Occurrence, ...]

    def format(self, repo_root: Path) -> str:
        header = (
            f"{self.fingerprint[:12]}: {self.occurrence_count} occurrences "
            f"(baseline max {self.baseline_max_occurrences})"
        )
        locations = "\n".join(
            f"    - {occurrence.format(repo_root)}" for occurrence in self.sample
        )
        return f"{header}\n{locations}"


@dataclass(frozen=True)
class _LogicalLine:
    number: int
    text: str


def _repo_root() -> Path:
    """Return the repository root for this script."""
    return Path(__file__).resolve().parents[2]


def _run_git(args: list[str], repo_root: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout


def _tracked_python_files(repo_root: Path) -> list[Path]:
    output = _run_git(["ls-files", "--", "*.py"], repo_root)
    return [
        repo_root / rel
        for rel in output.splitlines()
        if rel and (repo_root / rel).is_file()
    ]


def _fallback_python_files(repo_root: Path) -> list[Path]:
    return [
        path
        for path in repo_root.rglob("*.py")
        if not any(part.startswith(".") for part in path.relative_to(repo_root).parts)
    ]


def _load_json(repo_root: Path, path: Path) -> dict[str, Any]:
    with (repo_root / path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path.as_posix()} must contain a JSON object")
    return data


def _matches_any(rel_path: str, patterns: list[str]) -> bool:
    rel = PurePosixPath(rel_path)
    return any(
        rel.match(pattern) or fnmatch.fnmatch(rel_path, pattern) for pattern in patterns
    )


def _included(path: Path, repo_root: Path, config: dict[str, Any]) -> bool:
    rel_path = path.relative_to(repo_root).as_posix()
    includes = [str(pattern) for pattern in config.get("include", ["src/**/*.py"])]
    excludes = [str(pattern) for pattern in config.get("exclude", [])]
    return _matches_any(rel_path, includes) and not _matches_any(rel_path, excludes)


def _docstring_line_numbers(source: str) -> set[int]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            start = getattr(first, "lineno", None)
            end = getattr(first, "end_lineno", start)
            if start is not None and end is not None:
                lines.update(range(start, end + 1))
    return lines


def _normalize_line(line: str) -> str:
    tokens: list[str] = []
    try:
        for token in tokenize.generate_tokens(StringIO(line).readline):
            if token.type in {
                tokenize.COMMENT,
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.NEWLINE,
                tokenize.NL,
            }:
                continue
            tokens.append(token.string)
    except tokenize.TokenError:
        return " ".join(line.strip().split())
    return " ".join(tokens)


def _logical_lines(path: Path) -> list[_LogicalLine]:
    source = path.read_text(encoding="utf-8")
    docstring_lines = _docstring_line_numbers(source)
    logical: list[_LogicalLine] = []
    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or line_number in docstring_lines:
            continue
        if stripped.startswith(("import ", "from ")):
            continue
        normalized = _normalize_line(raw_line)
        if normalized:
            logical.append(_LogicalLine(number=line_number, text=normalized))
    return logical


def _iter_statement_bodies(tree: ast.AST) -> list[list[ast.stmt]]:
    bodies: list[list[ast.stmt]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef)):
            continue
        for attribute in ("body", "orelse", "finalbody"):
            body = getattr(node, attribute, None)
            if (
                isinstance(body, list)
                and body
                and all(isinstance(statement, ast.stmt) for statement in body)
            ):
                bodies.append(body)
    return bodies


def _candidate_windows(path: Path, threshold_lines: int) -> list[list[_LogicalLine]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    logical = _logical_lines(path)
    windows: list[list[_LogicalLine]] = []
    for body in _iter_statement_bodies(tree):
        for start_index, start_statement in enumerate(body):
            start_line = getattr(start_statement, "lineno", None)
            if start_line is None:
                continue
            for end_statement in body[start_index:]:
                end_line = getattr(end_statement, "end_lineno", None)
                if end_line is None:
                    continue
                candidate = [
                    line for line in logical if start_line <= line.number <= end_line
                ]
                if len(candidate) >= threshold_lines:
                    windows.append(candidate)
                    break
    return windows


def _window_fingerprint(lines: list[_LogicalLine]) -> str:
    normalized = "\n".join(line.text for line in lines)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _token_count(lines: list[_LogicalLine]) -> int:
    return sum(len(_TOKEN_RE.findall(line.text)) for line in lines)


def _iter_occurrences(
    *,
    repo_root: Path,
    paths: list[Path],
    config: dict[str, Any],
) -> dict[str, list[Occurrence]]:
    threshold_lines = int(config.get("threshold_lines", 6))
    min_tokens = int(config.get("min_tokens_per_window", 12))
    if threshold_lines < 2:
        raise ValueError("threshold_lines must be at least 2")
    if min_tokens < 1:
        raise ValueError("min_tokens_per_window must be positive")

    occurrences: dict[str, set[Occurrence]] = {}
    for path in sorted(paths):
        if not path.is_file() or not _included(path, repo_root, config):
            continue
        for window in _candidate_windows(path, threshold_lines):
            if _token_count(window) < min_tokens:
                continue
            fingerprint = _window_fingerprint(window)
            occurrences.setdefault(fingerprint, set()).add(
                Occurrence(
                    path=path,
                    start_line=window[0].number,
                    end_line=window[-1].number,
                )
            )
    return {
        fingerprint: sorted(
            fingerprint_occurrences,
            key=lambda occurrence: (
                occurrence.path.relative_to(repo_root).as_posix(),
                occurrence.start_line,
            ),
        )
        for fingerprint, fingerprint_occurrences in occurrences.items()
    }


def _baseline_limit(baseline: dict[str, Any], fingerprint: str) -> int:
    entry = baseline.get("entries", {}).get(fingerprint)
    if entry is None:
        return 1
    if isinstance(entry, int):
        return entry
    return int(entry.get("max_occurrences", 1))


def _validate_baseline(baseline: dict[str, Any]) -> list[str]:
    entries = baseline.get("entries", {})
    if not isinstance(entries, dict):
        return ["baseline entries must be a JSON object"]

    default_owner = str(baseline.get("owner", "")).strip()
    default_issue = str(baseline.get("issue", "")).strip()
    default_reason = str(baseline.get("reason", "")).strip()
    errors: list[str] = []
    for fingerprint, raw_entry in entries.items():
        if isinstance(raw_entry, int):
            if (
                raw_entry < 2
                or not default_owner
                or not default_issue.startswith("#")
                or not default_reason
            ):
                errors.append(
                    f"Invalid compact baseline entry for {fingerprint}: "
                    "requires count >= 2 plus top-level owner, #issue, and reason"
                )
            continue
        if not isinstance(raw_entry, dict):
            errors.append(f"Invalid baseline entry for {fingerprint}: {raw_entry}")
            continue
        owner = str(raw_entry.get("owner", default_owner)).strip()
        issue = str(raw_entry.get("issue", default_issue)).strip()
        reason = str(raw_entry.get("reason", default_reason)).strip()
        try:
            max_occurrences = int(raw_entry.get("max_occurrences", 0))
        except (TypeError, ValueError):
            max_occurrences = 0
        if max_occurrences < 2 or not owner or not issue.startswith("#") or not reason:
            errors.append(
                f"Invalid baseline entry for {fingerprint}: "
                "requires max_occurrences >= 2, owner, #issue, and reason"
            )
    return errors


def collect_findings(
    *,
    repo_root: Path,
    paths: list[Path],
    config: dict[str, Any],
    baseline: dict[str, Any],
) -> list[DuplicateFinding]:
    """Return duplicate logic fingerprints that exceed the allowed baseline."""
    if not repo_root.is_dir():
        raise ValueError(f"repo_root must be an existing directory: {repo_root}")

    occurrences_by_fingerprint = _iter_occurrences(
        repo_root=repo_root,
        paths=paths,
        config=config,
    )
    findings: list[DuplicateFinding] = []
    for fingerprint, occurrences in sorted(occurrences_by_fingerprint.items()):
        occurrence_count = len(occurrences)
        baseline_max = _baseline_limit(baseline, fingerprint)
        if occurrence_count <= baseline_max:
            continue
        ordered = tuple(
            sorted(
                occurrences,
                key=lambda occurrence: (
                    occurrence.path.relative_to(repo_root).as_posix(),
                    occurrence.start_line,
                ),
            )[:5]
        )
        findings.append(
            DuplicateFinding(
                fingerprint=fingerprint,
                occurrence_count=occurrence_count,
                baseline_max_occurrences=baseline_max,
                sample=ordered,
            )
        )
    return findings


def build_baseline(
    *,
    repo_root: Path,
    paths: list[Path],
    config: dict[str, Any],
    owner: str = DEFAULT_OWNER,
    issue: str = DEFAULT_ISSUE,
) -> dict[str, Any]:
    """Return a baseline JSON object for current duplicate fingerprints."""
    entries: dict[str, int] = {}
    occurrences_by_fingerprint = _iter_occurrences(
        repo_root=repo_root,
        paths=paths,
        config=config,
    )
    for fingerprint, occurrences in sorted(occurrences_by_fingerprint.items()):
        if len(occurrences) <= 1:
            continue
        entries[fingerprint] = len(occurrences)
    return {
        "version": 1,
        "owner": owner,
        "issue": issue,
        "reason": "Grandfathered by the DRY duplication no-growth ratchet.",
        "description": (
            "Baseline for existing duplicated production logic fingerprints. "
            "The blocking gate fails when any fingerprint grows beyond its "
            "recorded max_occurrences."
        ),
        "entries": entries,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON config relative to the repository root.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Rewrite the configured baseline from current tracked source files.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    repo_root = _repo_root()

    try:
        config = _load_json(repo_root, args.config_path)
        baseline_path = Path(config.get("baseline_path", ""))
        if not baseline_path:
            raise ValueError("config must define baseline_path")
        try:
            paths = _tracked_python_files(repo_root)
        except RuntimeError:
            paths = _fallback_python_files(repo_root)
        if args.write_baseline:
            baseline = build_baseline(repo_root=repo_root, paths=paths, config=config)
            _write_json(repo_root / baseline_path, baseline)
            logger.info(
                "Wrote DRY duplication baseline with %s entries to %s.",
                len(baseline["entries"]),
                baseline_path.as_posix(),
            )
            return 0

        baseline = _load_json(repo_root, baseline_path)
        validation_errors = _validate_baseline(baseline)
        findings = collect_findings(
            repo_root=repo_root,
            paths=paths,
            config=config,
            baseline=baseline,
        )
    except (OSError, RuntimeError, SyntaxError, ValueError) as exc:
        logger.error("DRY duplication gate failed: %s", exc)
        return 2

    if validation_errors:
        logger.error("FAIL: DRY duplication baseline metadata is invalid:\n")
        for error in validation_errors:
            logger.error("  %s", error)
        return 1

    if findings:
        logger.error("FAIL: DRY duplication gate found unapproved duplicated logic:\n")
        for finding in findings:
            logger.error("%s", finding.format(repo_root))
        logger.error(
            "\nExtract the duplicated logic into a shared helper, or add an owned "
            "baseline entry with a linked issue for existing debt only."
        )
        return 1

    logger.info("OK: DRY duplication gate found no unapproved duplicate growth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
