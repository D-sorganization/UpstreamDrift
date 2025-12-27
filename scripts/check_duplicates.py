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
    for path in root_dir.rglob("*"):
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


def main():
    root = Path(".")
    # Files that must NOT be duplicated
    tracked_files = {
        "matlab_quality_check.py",
        # "matlab_quality_config.m", # Todo: consolidating in Phase 2
        # "constants.py" # Todo: consolidating in Phase 2
    }

    print(f"Scanning for duplicates of: {tracked_files}")
    sys.exit(find_duplicates(root, tracked_files))


if __name__ == "__main__":
    main()
