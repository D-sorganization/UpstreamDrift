#!/usr/bin/env python3
"""
Check for duplicate files in the repository.
Fails if duplicates of tracked files are found (e.g., quality check scripts).
"""
import hashlib
import sys
from pathlib import Path


def get_file_hash(path: Path) -> str:
    """Calculate the MD5 hash of a file."""
    hasher = hashlib.md5()
    with path.open("rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def find_duplicates(root_dir: Path, tracked_files: set[str]) -> int:
    """Finds duplicates of tracked filenames."""

    # helper: map filename -> list of paths
    files_by_name: dict[str, list[Path]] = {}

    # Walk the tree
    # Walk the tree, excluding common heavy directories
    ignored_dirs = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
    }

    for path in root_dir.rglob("*"):
        # Optimization: Skip ignored directories to avoid traversing deep trees
        # Note: rglob yields everything; we must check parts to skip properly
        # but Path.rglob is a generator. We can't prune the walk easily with rglob.
        # A simple check on path parts is enough for correctness, but os.walk or explicit recursion is better for perf.
        # For now, just filtering is better than nothing if the iterator is fast.
        if any(part in ignored_dirs for part in path.parts):
            continue

        if path.is_file():
            if path.name in tracked_files:
                files_by_name.setdefault(path.name, []).append(path)

    exit_code = 0

    for name, paths in files_by_name.items():
        if len(paths) > 1:
            print(f"ERROR: Duplicate files found for '{name}':")
            # Sort by length to likely find the "canonical" one (shortest path usu. canonical)
            paths.sort(key=lambda p: len(str(p)))
            for p in paths:
                print(f"  - {p}")

            # Allow symlinks? For now, purely duplicate content or name is bad if we expect single source.
            exit_code = 1

    return exit_code


def main() -> None:
    root = Path(".")
    # Files that must NOT be duplicated
    tracked_files = {
        "matlab_quality_check.py",
        # "matlab_quality_config.m", # Todo: consolidating in Phase 2
        "constants.py",
    }

    print(f"Scanning for duplicates of: {tracked_files}")
    sys.exit(find_duplicates(root, tracked_files))


if __name__ == "__main__":
    main()
