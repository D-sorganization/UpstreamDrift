#!/usr/bin/env python3
"""Block generated artifacts from being tracked in Git."""

from __future__ import annotations

import argparse
import fnmatch
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

FORBIDDEN_FILES = frozenset(
    {
        "coverage.json",
        "bandit_results.json",
        "reports/bandit.json",
        "temp_id.txt",
        "matlab_quality_report.txt",
        "test.npz",
    }
)
FORBIDDEN_DIR_PREFIXES = (".jules/completist_data/",)
FORBIDDEN_PATTERNS = (
    "*.bak",
    "*.swp",
    "*.orig",
    "*.rej",
    "pr_body_*.md",
    "fix_*.py",
    ".ci_trigger*",
)


def normalize_git_path(path: str | Path) -> str:
    """Return a stable POSIX-style Git path for artifact matching."""
    normalized = Path(path).as_posix().lower()
    if normalized.startswith("./"):
        return normalized[2:]
    return normalized


def is_forbidden_artifact(path: str | Path) -> bool:
    """Return True when a repository path is a generated artifact."""
    normalized = normalize_git_path(path)
    if normalized in FORBIDDEN_FILES:
        return True
    if any(normalized.startswith(prefix) for prefix in FORBIDDEN_DIR_PREFIXES):
        return True

    # Check patterns against the filename part (or the whole path if it matches)
    name = Path(path).name
    return any(
        fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(normalized, pat)
        for pat in FORBIDDEN_PATTERNS
    )


def find_forbidden_artifacts(
    repo_root: Path, tracked_paths: list[str] | None = None
) -> list[Path]:
    """Return tracked generated-artifact paths that must not be committed."""
    paths = (
        tracked_paths if tracked_paths is not None else list_tracked_paths(repo_root)
    )
    return sorted(Path(path) for path in paths if is_forbidden_artifact(path))


def list_tracked_paths(repo_root: Path) -> list[str]:
    """Return paths tracked by Git under repo_root."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail if generated run artifacts are tracked by Git."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root to inspect.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        violations = find_forbidden_artifacts(args.repo_root)
    except RuntimeError as exc:
        logger.error("FAIL: %s", exc)
        return 1

    if violations:
        logger.error("FAIL: generated artifacts are tracked:")
        for violation in violations:
            logger.error("  %s", violation.as_posix())
        logger.error("Remove these paths from Git and keep them ignored.")
        return 1

    logger.info("OK: No forbidden generated artifacts are tracked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
