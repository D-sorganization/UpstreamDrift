#!/usr/bin/env python3
"""
Apply quick fixes to the repository.

Current fixes:
1. Add missing __init__.py files to Python package directories in specific roots.
"""

import os
from pathlib import Path


def add_missing_init_files():
    """Add missing __init__.py files to package directories."""

    # Target directories to scan for missing __init__.py
    target_roots = ["engines", "shared", "api", "launchers"]

    fixed_count = 0

    print("Scanning for missing __init__.py files...")

    for root_name in target_roots:
        root_path = Path(root_name)
        if not root_path.exists():
            continue

        for dirpath, _dirnames, filenames in os.walk(root_path):
            path_obj = Path(dirpath)

            # Skip hidden directories
            if any(part.startswith(".") for part in path_obj.parts):
                continue

            # Skip directories that are not valid Python identifiers (e.g. contain spaces)
            if not path_obj.name.isidentifier():
                continue

            # Check if directory contains .py files
            has_py_files = any(f.endswith(".py") for f in filenames)

            # Check if __init__.py exists
            has_init = "__init__.py" in filenames

            if has_py_files and not has_init:
                init_path = Path(dirpath) / "__init__.py"
                try:
                    with open(init_path, "w") as f:
                        f.write('"""\nAuto-generated __init__.py\n"""\n')
                    print(f"Fixed: Created {init_path}")
                    fixed_count += 1
                except Exception as e:
                    print(f"Error creating {init_path}: {e}")

    print(f"\nTotal fixed: {fixed_count} missing __init__.py files.")


if __name__ == "__main__":
    add_missing_init_files()
