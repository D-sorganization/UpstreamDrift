#!/usr/bin/env python3
"""Rehydrate and verify tracked Dockerfile and build context inputs before Buildx."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
import sys

logger = logging.getLogger(__name__)

DEFAULT_TARGETS: tuple[str, ...] = (
    "Dockerfile",
    "Dockerfile.modular",
    "Dockerfile.heavy_test",
)


def _is_tracked_at_head(root: Path, rel_path: str) -> bool:
    """Return True if the relative path is tracked in git at HEAD."""
    try:
        result = subprocess.run(
            ["git", "ls-tree", "HEAD", "--", rel_path],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except subprocess.SubprocessError:
        return False


def _differs_from_head(root: Path, rel_path: str) -> bool:
    """Return True if the working-tree file does not match its HEAD blob.

    Existence alone is not proof of a usable build context: a shared or
    partially-restored workspace can leave a tracked Dockerfile present but
    truncated or stale, which Buildx would then consume silently.  ``git diff``
    is used rather than a raw byte comparison so that git's own filters and
    end-of-line normalisation are applied, avoiding false mismatches on
    Windows checkouts.  A git failure is reported as a difference so the
    caller fails closed and restores.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", rel_path],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.SubprocessError as exc:
        logger.warning("Could not compare %s against HEAD: %s", rel_path, exc)
        return True
    return result.returncode != 0


def _checkout_from_head(root: Path, rel_path: str) -> bool:
    """Restore a tracked file from git HEAD."""
    try:
        subprocess.run(
            ["git", "checkout", "HEAD", "--", rel_path],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except subprocess.SubprocessError as exc:
        logger.error("Failed to checkout %s from HEAD: %s", rel_path, exc)
        return False


def rehydrate_tracked_files(
    root: Path,
    targets: list[str] | tuple[str, ...],
    *,
    check_only: bool = False,
) -> list[str]:
    """Verify and optionally restore tracked files from git HEAD.

    Fails closed if any target is not tracked at HEAD or cannot be restored.
    """
    failures: list[str] = []

    for target in targets:
        rel_str = str(Path(target).as_posix())
        if not _is_tracked_at_head(root, rel_str):
            failures.append(f"Target '{rel_str}' is not tracked at HEAD")
            continue

        file_path = root / rel_str
        missing = not file_path.is_file()
        stale = not missing and _differs_from_head(root, rel_str)

        if missing or stale:
            if check_only:
                reason = "is missing on disk" if missing else "differs from HEAD"
                failures.append(f"Tracked file '{rel_str}' {reason}")
                continue

            logger.info(
                "Rehydrating %s tracked file from HEAD: %s",
                "missing" if missing else "stale",
                rel_str,
            )
            restored = _checkout_from_head(root, rel_str)
            if not restored or not file_path.is_file():
                failures.append(f"Failed to restore tracked file '{rel_str}' from HEAD")
                continue
            if _differs_from_head(root, rel_str):
                failures.append(
                    f"Tracked file '{rel_str}' still differs from HEAD after restore"
                )
                continue

        # Verify the restored file is readable.  Content now provably matches the
        # HEAD blob, so a 0-byte file means the tracked blob is itself empty --
        # worth reporting for a build context, not merely logging.
        try:
            size = file_path.stat().st_size
        except OSError as exc:
            failures.append(f"Could not access rehydrated file '{rel_str}': {exc}")
            continue
        if size == 0:
            failures.append(f"Tracked file '{rel_str}' is 0 bytes at HEAD")

    return failures


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Docker context rehydration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root directory (default: current directory).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help="Target file paths relative to root to rehydrate and verify.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify file existence without restoring.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    failures = rehydrate_tracked_files(
        args.root.resolve(),
        args.targets,
        check_only=args.check_only,
    )
    if failures:
        for failure in failures:
            logger.error(failure)
        return 1

    logger.info("All target Dockerfile build context files verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
