#!/usr/bin/env python3
"""Select pytest modules for the CI Standard fast tier (#13578 / UD-CI-013).

Precondition: every changed path is a repository-relative POSIX path.
Postcondition: ``full`` mode means the unit gate keeps collecting the whole
green suite; ``modules`` mode lists tracked ``tests/**/test_*.py`` paths only.
Shared-core or workflow edits, an empty diff, or unmapped source changes fall
back to ``full`` so the fast tier never silently shrinks below full collection.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "scripts/config/pytest_fast_tier_shared_core_globs.json"

# The monolith register (#7131) is derived from every tracked ``src/`` .py/.m
# file, so no stem-based mapping can see drift. Any such change must also run
# its cheap guard test on the PR tier instead of failing later in merge_group.
MONOLITH_REGISTER_TEST = "tests/scripts/test_monolith_register.py"
_MONOLITH_REGISTER_DOC = "docs/development/monolith_refactor_register.md"
_MONOLITH_REGISTER_SCRIPT = "scripts/gen_monolith_register.py"


@dataclass(frozen=True)
class SelectionConfig:
    shared_core_globs: tuple[str, ...]
    workflow_globs: tuple[str, ...]


@dataclass(frozen=True)
class SelectionDecision:
    mode: Literal["full", "modules"]
    modules: tuple[str, ...]
    reason: str


def _normalize_path(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise TypeError("path must be a non-empty string")
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _glob_matches(pattern: str, path: str) -> bool:
    normalized = _normalize_path(path)
    pattern = pattern.replace("\\", "/")
    if pattern.endswith("/**"):
        prefix = pattern[: -len("/**")]
        return normalized == prefix or normalized.startswith(f"{prefix}/")
    return fnmatch.fnmatch(normalized, pattern)


def load_config(config_path: Path | None = None) -> SelectionConfig:
    path = config_path or _CONFIG_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    shared = tuple(str(item) for item in raw.get("shared_core_globs", ()))
    workflow = tuple(str(item) for item in raw.get("workflow_globs", ()))
    if not shared:
        raise ValueError(f"{path} must define shared_core_globs")
    return SelectionConfig(shared_core_globs=shared, workflow_globs=workflow)


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    return any(_glob_matches(pattern, path) for pattern in patterns)


def _tracked_test_modules(
    repo_root: Path, candidates: Iterable[Path]
) -> tuple[str, ...]:
    tracked: list[str] = []
    for candidate in candidates:
        rel = candidate.relative_to(repo_root).as_posix()
        if not rel.startswith("tests/") or not rel.endswith(".py"):
            continue
        if candidate.is_file():
            tracked.append(rel)
    return tuple(sorted(set(tracked)))


def _modules_for_script(rel: str, repo_root: Path) -> tuple[str, ...]:
    stem = Path(rel).stem
    candidates = [
        repo_root / f"tests/ci/test_{stem}.py",
        repo_root / f"tests/unit/scripts/test_{stem}.py",
    ]
    candidates.extend(repo_root.glob(f"tests/**/test_{stem}.py"))
    return _tracked_test_modules(repo_root, candidates)


def _modules_for_source(rel: str, repo_root: Path) -> tuple[str, ...]:
    parts = Path(rel).parts
    if len(parts) < 2 or parts[0] != "src":
        return ()
    stem = Path(rel).stem
    segment_hints = {part for part in parts[1:-1] if part != "python"}
    hits: list[Path] = []
    for candidate in repo_root.glob(f"tests/**/test_{stem}.py"):
        candidate_parts = set(candidate.parts)
        if segment_hints & candidate_parts:
            hits.append(candidate)
    if not hits:
        hits = list(repo_root.glob(f"tests/**/test_{stem}.py"))
    return _tracked_test_modules(repo_root, hits)


def _affects_monolith_register(path: str) -> bool:
    if path in (_MONOLITH_REGISTER_DOC, _MONOLITH_REGISTER_SCRIPT):
        return True
    return path.startswith("src/") and path.endswith((".py", ".m"))


def select_modules(
    *,
    tier: str,
    changed_files: Sequence[str],
    repo_root: Path | None = None,
    config: SelectionConfig | None = None,
) -> SelectionDecision:
    """Return how the unit gate should scope pytest on this run."""
    root = repo_root or _REPO_ROOT
    cfg = config or load_config()
    if tier != "fast":
        return SelectionDecision("full", (), "not_fast_tier")

    cleaned = [_normalize_path(path) for path in changed_files if path.strip()]
    if not cleaned:
        return SelectionDecision("full", (), "empty_diff")

    fallback_patterns = cfg.shared_core_globs + cfg.workflow_globs
    if any(_matches_any(path, fallback_patterns) for path in cleaned):
        return SelectionDecision("full", (), "shared_core_or_workflow")

    modules: set[str] = set()
    for path in cleaned:
        if path.startswith("tests/") and path.endswith(".py"):
            candidate = root / path
            if candidate.is_file():
                modules.add(path)
            continue
        if path.startswith("scripts/") and path.endswith(".py"):
            modules.update(_modules_for_script(path, root))
            continue
        if path.startswith("src/") and path.endswith(".py"):
            modules.update(_modules_for_source(path, root))

    if not modules:
        return SelectionDecision("full", (), "no_mapped_modules")
    if any(_affects_monolith_register(path) for path in cleaned):
        if (root / MONOLITH_REGISTER_TEST).is_file():
            modules.add(MONOLITH_REGISTER_TEST)
    return SelectionDecision("modules", tuple(sorted(modules)), "mapped")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tier", required=True, help="fast or full from changed-paths")
    parser.add_argument(
        "--changed-file",
        action="append",
        default=[],
        help="repository-relative changed path (repeatable)",
    )
    parser.add_argument(
        "--modules-out",
        type=Path,
        help="write selected module paths, one per line",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        help="append mode= and reason= lines for Actions",
    )
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    paths = list(args.changed_file)
    if not paths and not sys.stdin.isatty():
        paths = [line.strip() for line in sys.stdin if line.strip()]

    decision = select_modules(
        tier=args.tier,
        changed_files=paths,
        repo_root=args.repo_root,
        config=load_config(args.config),
    )

    if args.modules_out is not None:
        args.modules_out.write_text(
            "\n".join(decision.modules) + ("\n" if decision.modules else ""),
            encoding="utf-8",
        )
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"mode={decision.mode}\n")
            handle.write(f"reason={decision.reason}\n")

    for module in decision.modules:
        print(module)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
