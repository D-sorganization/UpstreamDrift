import hashlib
import os
import sys
from pathlib import Path


def get_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def find_duplicates(root_dir, extension=".py"):
    """Find files with duplicate content."""
    hashes = {}
    duplicates = []

    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories and virtualenvs
        if ".git" in dirs:
            dirs.remove(".git")
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")
        if ".venv" in dirs:
            dirs.remove(".venv")
        if "venv" in dirs:
            dirs.remove("venv")

        for file in files:
            if not file.endswith(extension):
                continue

            filepath = Path(root) / file
            file_hash = get_file_hash(filepath)

            if file_hash in hashes:
                duplicates.append((filepath, hashes[file_hash]))
            else:
                hashes[file_hash] = filepath

    return duplicates


def find_duplicate_names(root_dir, targets):
    """Find multiple occurrences of specific filenames."""
    counts = {name: [] for name in targets}

    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories
        if ".git" in dirs:
            dirs.remove(".git")

        for file in files:
            if file in counts:
                counts[file].append(Path(root) / file)

    return {k: v for k, v in counts.items() if len(v) > 1}


def main():
    root_dir = Path(".")

    # 1. Check for duplicate content in Python files
    # We might want to restrict this or warn only, as some setup.py or __init__.py might be identical.
    # For now, let's focus on the specific issue requirement: duplicate scripts.

    # 2. Check for specific problematic duplicates
    target_duplicates = ["matlab_quality_check.py"]
    name_dupes = find_duplicate_names(root_dir, target_duplicates)

    if name_dupes:
        print("ERROR: Duplicate script names found:")
        for name, paths in name_dupes.items():
            print(f"  {name}:")
            for p in paths:
                print(f"    {p}")
        sys.exit(1)

    print("No duplicate tracked scripts found.")
    sys.exit(0)


if __name__ == "__main__":
    main()
