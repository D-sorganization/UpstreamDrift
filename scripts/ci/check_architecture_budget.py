#!/usr/bin/env python3
"""Enforce changed-file function-size and parameter-count budgets."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("scripts/config/architecture_budget.json")
DEFAULT_BASELINE_PATH = Path("scripts/config/architecture_budget_baseline.json")
DEFAULT_BASE_REF = "origin/main"

EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "tests",
    "vendor",
}


@dataclass(frozen=True)
class FunctionBudgetFinding:
    """A single function-level architecture budget finding."""

    path: Path
    symbol: str
    line: int
    rule: str
    actual: int
    budget: int

    def format(self, repo_root: Path) -> str:
        """Return a stable human-readable violation string."""
        rel = self.path.relative_to(repo_root).as_posix()
        return (
            f"{rel}:{self.line}: {self.symbol} exceeds {self.rule} "
            f"budget ({self.actual} > {self.budget})"
        )


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


def _changed_python_files(repo_root: Path, base_ref: str) -> list[Path]:
    for ref in (base_ref, "origin/main", "main", "HEAD~1"):
        try:
            output = _run_git(["diff", "--name-only", f"{ref}...HEAD", "--"], repo_root)
            return [
                repo_root / rel
                for rel in output.splitlines()
                if rel.endswith(".py") and (repo_root / rel).is_file()
            ]
        except RuntimeError:
            continue
    return []


def _tracked_python_files(repo_root: Path) -> list[Path]:
    output = _run_git(["ls-files", "--", "*.py"], repo_root)
    return [
        repo_root / rel
        for rel in output.splitlines()
        if rel and (repo_root / rel).is_file()
    ]


def _load_config(repo_root: Path, config_path: Path) -> dict[str, Any]:
    with (repo_root / config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("architecture budget config must be a JSON object")
    return config


def _is_production_python(path: Path, repo_root: Path) -> bool:
    try:
        rel_parts = path.relative_to(repo_root).parts
    except ValueError:
        return False
    return path.suffix == ".py" and not any(
        part in EXCLUDED_PARTS for part in rel_parts
    )


def _function_line_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    end_lineno = getattr(node, "end_lineno", None)
    if end_lineno is None:
        return 1
    return end_lineno - node.lineno + 1


def _parameter_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    args = node.args
    params = [
        *args.posonlyargs,
        *args.args,
        *args.kwonlyargs,
    ]
    if params and params[0].arg in {"self", "cls"}:
        params = params[1:]
    count = len(params)
    if args.vararg is not None:
        count += 1
    if args.kwarg is not None:
        count += 1
    return count


class _BudgetVisitor(ast.NodeVisitor):
    def __init__(
        self, path: Path, max_function_lines: int, max_parameters: int
    ) -> None:
        self._path = path
        self._max_function_lines = max_function_lines
        self._max_parameters = max_parameters
        self._scope: list[str] = []
        self.findings: list[FunctionBudgetFinding] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        symbol = ".".join([*self._scope, node.name])
        lines = _function_line_count(node)
        if lines > self._max_function_lines:
            self.findings.append(
                FunctionBudgetFinding(
                    path=self._path,
                    symbol=symbol,
                    line=node.lineno,
                    rule="function-lines",
                    actual=lines,
                    budget=self._max_function_lines,
                )
            )

        parameters = _parameter_count(node)
        if parameters > self._max_parameters:
            self.findings.append(
                FunctionBudgetFinding(
                    path=self._path,
                    symbol=symbol,
                    line=node.lineno,
                    rule="parameters",
                    actual=parameters,
                    budget=self._max_parameters,
                )
            )

        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()


def analyze_python_file(
    path: Path, *, max_function_lines: int, max_parameters: int
) -> list[FunctionBudgetFinding]:
    """Return function-level budget findings for a Python file.

    DbC:
        precondition: ``path`` points to a readable Python file.
        postcondition: findings contain only functions whose measured value is
            strictly greater than the configured budget.
    """
    if max_function_lines < 1:
        raise ValueError("max_function_lines must be positive")
    if max_parameters < 1:
        raise ValueError("max_parameters must be positive")

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = _BudgetVisitor(path, max_function_lines, max_parameters)
    visitor.visit(tree)
    return visitor.findings


@dataclass(frozen=True)
class ExpiringBudgetException:
    """An active budget exception nearing expiration."""

    path: str
    owner: str
    expires_on: date
    days_remaining: int
    reason: str
    symbol: str | None = None
    rule: str | None = None

    def format(self) -> str:
        """Return human-readable warning description."""
        ident = (
            f"{self.path}:{self.symbol} [{self.rule}]"
            if self.symbol and self.rule
            else self.path
        )
        return (
            f"{ident} expires in {self.days_remaining}d on "
            f"{self.expires_on.isoformat()} (owner={self.owner})"
        )


ExpiringArchitectureException = ExpiringBudgetException


def resolve_reference_date(
    *,
    repo_root: Path | None = None,
    base_ref: str = DEFAULT_BASE_REF,
    as_of: str | date | None = None,
) -> date:
    """Resolve evaluation reference date using the deterministic hierarchy."""
    if isinstance(as_of, date):
        return as_of
    if as_of:
        return date.fromisoformat(as_of.strip())

    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch:
        try:
            return datetime.fromtimestamp(int(epoch.strip()), tz=timezone.utc).date()
        except (ValueError, OverflowError, OSError):
            pass

    target_root = repo_root
    if target_root is None:
        try:
            target_root = _repo_root()
        except (OSError, RuntimeError, ValueError):
            target_root = None

    if target_root is not None:
        for ref_args in (["merge-base", "HEAD", base_ref], ["rev-parse", "HEAD"]):
            try:
                commit = _run_git(ref_args, target_root).strip()
                if commit:
                    date_str = _run_git(
                        ["show", "-s", "--format=%cs", commit], target_root
                    ).strip()
                    if date_str:
                        return date.fromisoformat(date_str)
            except (OSError, RuntimeError, ValueError):
                continue

    return datetime.now(timezone.utc).date()


def _exception_is_active(
    exc: dict[str, Any], reference_date: date | None = None
) -> bool:
    expires_on = exc.get("expires_on")
    if not expires_on:
        return True
    ref = reference_date if reference_date is not None else resolve_reference_date()
    return ref <= date.fromisoformat(str(expires_on))


def _exception_key(exc: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(exc.get("path", "")).strip(),
        str(exc.get("symbol", "")).strip(),
        str(exc.get("rule", "")).strip(),
    )


def _collect_active_exceptions(
    config: dict[str, Any],
    reference_date: date | None = None,
) -> tuple[set[tuple[str, str, str]], list[str]]:
    active: set[tuple[str, str, str]] = set()
    invalid: list[str] = []
    valid_rules = {"function-lines", "parameters"}
    ref_date = (
        reference_date if reference_date is not None else resolve_reference_date()
    )

    for raw_exc in config.get("exceptions", []):
        if not isinstance(raw_exc, dict):
            invalid.append(f"Invalid exception entry: {raw_exc}")
            continue
        path, symbol, rule = _exception_key(raw_exc)
        owner = str(raw_exc.get("owner", "")).strip()
        reason = str(raw_exc.get("reason", "")).strip()
        if not path or not symbol or rule not in valid_rules or not owner or not reason:
            invalid.append(f"Invalid exception entry: {raw_exc}")
            continue
        if (
            "issue" not in reason.lower()
            and "#" not in reason
            and "decomposition" not in reason.lower()
            and "legacy" not in reason.lower()
        ):
            invalid.append(f"Exception missing linked issue in reason: {path}:{symbol}")
            continue
        try:
            if _exception_is_active(raw_exc, reference_date=ref_date):
                active.add((path, symbol, rule))
            else:
                invalid.append(
                    f"Expired exception: {path}:{symbol} "
                    f"(owner={owner}, expires_on={raw_exc.get('expires_on')})"
                )
        except ValueError:
            invalid.append(
                f"Invalid expires_on date in exception: "
                f"{path}:{symbol} ({raw_exc.get('expires_on')})"
            )

    return active, invalid


def collect_expiring_exceptions(
    config: dict[str, Any],
    reference_date: date | None = None,
    within_days: int = 21,
) -> list[ExpiringArchitectureException]:
    """Return active exceptions expiring within ``within_days`` of ``reference_date``."""
    ref_date = (
        reference_date if reference_date is not None else resolve_reference_date()
    )
    expiring: list[ExpiringArchitectureException] = []

    for raw in config.get("exceptions", []):
        if not isinstance(raw, dict) or not raw.get("expires_on"):
            continue
        try:
            exp_date = date.fromisoformat(str(raw["expires_on"]))
        except ValueError:
            continue

        days = (exp_date - ref_date).days
        if 0 <= days <= within_days:
            path = str(raw.get("path", "")).strip()
            symbol = str(raw["symbol"]).strip() if raw.get("symbol") else None
            rule = str(raw["rule"]).strip() if raw.get("rule") else None
            expiring.append(
                ExpiringBudgetException(
                    path=path,
                    symbol=symbol,
                    rule=rule,
                    owner=str(raw.get("owner", "")).strip(),
                    expires_on=exp_date,
                    days_remaining=days,
                    reason=str(raw.get("reason", "")).strip(),
                )
            )

    return sorted(expiring, key=lambda x: (x.expires_on, x.path, x.symbol or ""))


def _load_baseline(repo_root: Path, baseline_path: Path) -> set[tuple[str, str, str]]:
    """Load architecture budget baseline findings.

    DbC:
        precondition: ``baseline_path`` exists relative to ``repo_root``.
        postcondition: returns a set of (path, symbol, rule) triples.
    """
    full_path = repo_root / baseline_path
    if not full_path.is_file():
        raise FileNotFoundError(
            f"architecture budget baseline file not found: {full_path}"
        )
    raw = json.loads(full_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("architecture budget baseline must be a JSON object")
    entries: set[tuple[str, str, str]] = set()
    for item in raw.get("entries", []):
        if isinstance(item, dict):
            p = str(item.get("path", "")).strip()
            s = str(item.get("symbol", "")).strip()
            r = str(item.get("rule", "")).strip()
            if p and s and r:
                entries.add((p, s, r))
    return entries


def _write_baseline(
    repo_root: Path,
    baseline_path: Path,
    findings: list[tuple[str, str, str]],
) -> None:
    """Write architecture budget baseline findings to JSON.

    DbC:
        precondition: ``repo_root`` is an existing directory.
        postcondition: baseline file is written with deterministically sorted entries.
    """
    full_path = repo_root / baseline_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_entries = [
        {"path": p, "symbol": s, "rule": r} for p, s, r in sorted(set(findings))
    ]
    data = {
        "schema_version": 1,
        "description": (
            "Baseline of grandfathered architecture budget violations (UD-25, #13869). "
            "Ratchet policy: entries may only be removed, never added."
        ),
        "entries": sorted_entries,
    }
    full_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def collect_unexcepted_findings(
    *,
    repo_root: Path,
    paths: list[Path],
    config: dict[str, Any],
    reference_date: date | None = None,
) -> list[tuple[str, str, str]]:
    """Return all unexcepted (path, symbol, rule) findings across specified paths."""
    max_function_lines = int(config.get("max_function_lines", 100))
    max_parameters = int(config.get("max_parameters", 8))
    active_exceptions, _ = _collect_active_exceptions(
        config, reference_date=reference_date
    )
    findings_list: list[tuple[str, str, str]] = []
    for path in sorted(paths):
        if not _is_production_python(path, repo_root):
            continue
        rel = path.relative_to(repo_root).as_posix()
        findings = analyze_python_file(
            path,
            max_function_lines=max_function_lines,
            max_parameters=max_parameters,
        )
        for finding in findings:
            key = (rel, finding.symbol, finding.rule)
            if key not in active_exceptions:
                findings_list.append(key)
    return findings_list


def collect_violations(
    *,
    repo_root: Path,
    paths: list[Path],
    config: dict[str, Any],
    baseline: set[tuple[str, str, str]] | None = None,
    reference_date: date | None = None,
) -> list[str]:
    """Return architecture budget violations for production Python files.

    DbC:
        precondition: ``repo_root`` is an existing directory.
        postcondition: any violation returned exceeds budget and is not covered
            by active exceptions or baseline.
        invariant: baseline entries grandfather existing findings; new violations
            outside baseline always fail.
    """
    if not repo_root.is_dir():
        raise ValueError(f"repo_root must be an existing directory: {repo_root}")

    max_function_lines = int(config.get("max_function_lines", 100))
    max_parameters = int(config.get("max_parameters", 8))
    ref_date = (
        reference_date
        if reference_date is not None
        else resolve_reference_date(repo_root=repo_root)
    )
    active_exceptions, invalid_exceptions = _collect_active_exceptions(
        config, reference_date=ref_date
    )
    violations = list(invalid_exceptions)

    for path in sorted(paths):
        if not _is_production_python(path, repo_root):
            continue
        rel = path.relative_to(repo_root).as_posix()
        findings = analyze_python_file(
            path,
            max_function_lines=max_function_lines,
            max_parameters=max_parameters,
        )
        for finding in findings:
            key = (rel, finding.symbol, finding.rule)
            if key in active_exceptions:
                continue
            if baseline is not None and key in baseline:
                continue
            violations.append(finding.format(repo_root))

    return violations


def _report_expiring_exceptions(
    config: dict[str, Any], reference_date: date, within_days: int
) -> None:
    """Log warnings for exceptions nearing their expiration date."""
    if within_days < 0:
        return
    expiring = collect_expiring_exceptions(
        config, reference_date=reference_date, within_days=within_days
    )
    if expiring:
        logger.info(
            "EXPIRING SOON (%d exceptions expiring within %d days as of %s):",
            len(expiring),
            within_days,
            reference_date.isoformat(),
        )
        for item in expiring:
            logger.info("  %s", item.format())
        logger.info("")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON config relative to repository root.",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Path to JSON baseline relative to repository root.",
    )
    parser.add_argument(
        "--base-ref",
        default=DEFAULT_BASE_REF,
        help="Base ref for changed-file scanning.",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Evaluation reference date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--expiring-within-days",
        type=int,
        default=21,
        help="Report active exceptions expiring within N days (default: 21).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan all tracked Python files instead of changed files.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write current unexcepted findings to the baseline file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo_root = _repo_root()

    try:
        config = _load_config(repo_root, args.config_path)
        ref_date = resolve_reference_date(
            repo_root=repo_root,
            base_ref=args.base_ref,
            as_of=args.as_of,
        )

        if args.write_baseline:
            paths = _tracked_python_files(repo_root)
            unexcepted = collect_unexcepted_findings(
                repo_root=repo_root,
                paths=paths,
                config=config,
                reference_date=ref_date,
            )
            _write_baseline(repo_root, args.baseline_path, unexcepted)
            logger.info(
                "Wrote %d baseline entries to %s",
                len(unexcepted),
                args.baseline_path,
            )
            return 0

        paths = (
            _tracked_python_files(repo_root)
            if args.all
            else _changed_python_files(repo_root, args.base_ref)
        )

        baseline: set[tuple[str, str, str]] | None = None
        baseline_file = repo_root / args.baseline_path
        if baseline_file.is_file():
            baseline = _load_baseline(repo_root, args.baseline_path)
        elif args.baseline_path != DEFAULT_BASELINE_PATH:
            raise FileNotFoundError(
                f"Architecture budget baseline not found: {args.baseline_path}"
            )

        violations = collect_violations(
            repo_root=repo_root,
            paths=paths,
            config=config,
            baseline=baseline,
            reference_date=ref_date,
        )
        _report_expiring_exceptions(
            config,
            reference_date=ref_date,
            within_days=args.expiring_within_days,
        )
    except (OSError, RuntimeError, SyntaxError, ValueError) as exc:
        logger.error("architecture budget failed: %s", exc)
        return 2

    if violations:
        logger.error("FAIL: architecture budget violations detected:\n")
        for violation in violations:
            logger.error("  %s", violation)
        logger.error(
            "\nSplit large functions or introduce a small, owned exception "
            "with an expiry and linked issue."
        )
        return 1

    scan_label = "Tracked" if args.all else "Changed"
    logger.info(
        "OK: %s production Python files are within architecture budgets.", scan_label
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
