#!/usr/bin/env python3
"""
Check for duplicate files in the repository.
Intended to prevent proliferation of identical scripts.
"""

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

# Directories to ignore
IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "venv",
    "env",
    "build",
    "dist",
    "htmlcov",
    ".jules",
    "archive",
    "opensim-models",  # Vendored OpenSim models often have duplicate outputs
    "myo_sim",  # Vendored MyoSuite models
}

# Files that are allowed to be duplicated (e.g., generic names)
ALLOWED_DUPLICATES = {
    "__init__.py",
    "README.md",
    "LICENSE",
    "conftest.py",
    "requirements.txt",
    "CMakeLists.txt",
    "package.xml",
    "setup.py",
}

# Extensions to likely ignore for duplication checks (data files)
IGNORE_EXTENSIONS = {
    ".sto",
    ".mot",
    ".xml",
    ".stl",
    ".vtp",
    ".obj",
    ".osim",
    ".msg",
    ".srv",
}


def get_file_hash(path: Path) -> str:
    """Calculate SHA256 hash of file content."""
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in chunks to avoid memory issues
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def find_duplicates(root_dir: Path) -> int:
    """Find duplicate files by name and content."""
    files_by_name = defaultdict(list)
    files_by_hash = defaultdict(list)

    print(f"Scanning {root_dir}...")

    for path in root_dir.rglob("*"):
        # Skip ignored directories
        if any(part in IGNORE_DIRS for part in path.parts):
            continue

        if path.is_file():
            # Skip ignored extensions
            if path.suffix in IGNORE_EXTENSIONS:
                continue

            # Check by name
            if path.name not in ALLOWED_DUPLICATES:
                files_by_name[path.name].append(path)

            # Check by content (only for scripts mostly)
            if path.suffix in {".py", ".sh", ".m"}:
                try:
                    file_hash = get_file_hash(path)
                    files_by_hash[file_hash].append(path)
                except OSError:
                    pass  # Skip if cannot read

    exit_code = 0

    # Report duplicates by content (exact copies)
    print("\n=== Exact Content Duplicates ===")
    content_dupes = 0
    for file_hash, paths in files_by_hash.items():
        if len(paths) > 1:
            # Filter out empty files
            if paths[0].stat().st_size == 0:
                continue

            print(f"\nHash: {file_hash[:8]}...")
            for p in paths:
                print(f"  {p.relative_to(root_dir)}")
            content_dupes += 1
            exit_code = 1

    if content_dupes == 0:
        print("None found.")

    # Report duplicates by name (potential copies)
    print("\n=== Duplicate Filenames (Potential Copies) ===")
    name_dupes = 0
    for name, paths in files_by_name.items():
        if len(paths) > 1:
            # Special check for matlab_quality_check.py (Issue #121)
            if name == "matlab_quality_check.py":
                print(f"\nFilename: {name} (CRITICAL - Issue #121)")
                for p in paths:
                    print(f"  {p.relative_to(root_dir)}")
                exit_code = 1
            elif len(paths) >= 3:  # Only report if 3+ duplicates to reduce noise
                print(f"\nFilename: {name}")
                for p in paths:
                    print(f"  {p.relative_to(root_dir)}")
                name_dupes += 1

    if name_dupes == 0 and exit_code == 0:
        print("No critical duplicates found.")

    return exit_code


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    sys.exit(find_duplicates(root_dir))
