#!/usr/bin/env python3
"""
Apply quick fixes to the repository.

Current fixes:
1. Add missing __init__.py files to Python package directories in specific roots.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.shared.python.path_utils import get_repo_root
except ImportError:
    # Fallback if src import fails
    def get_repo_root():
        return _REPO_ROOT


def add_missing_init_files():
    """Add missing __init__.py files to package directories."""
    repo_root = get_repo_root()

    # Target directories to scan for missing __init__.py (relative to repo root)
    target_roots = [
        "src/engines",
        "src/shared",
        "src/api",
        "src/launchers",
        "shared",  # Check root shared if it exists
    ]

    fixed_count = 0

    print("Scanning for missing __init__.py files...")

    for root_rel in target_roots:
        root_path = repo_root / root_rel
        if not root_path.exists():
            continue

        for dirpath, _dirnames, filenames in os.walk(root_path):
            path_obj = Path(dirpath)

            # Skip hidden directories
            if any(part.startswith(".") for part in path_obj.parts):
                continue

            # Skip directories that are not valid Python identifiers (e.g. contain spaces or hyphens)
            if not path_obj.name.isidentifier():
                continue

            # Ensure all parent directories relative to root are also valid identifiers
            # This prevents creating __init__.py in subdirectories of invalid packages (e.g., opensim-models/Geometry)
            try:
                rel_parts = path_obj.relative_to(root_path).parts
                if not all(part.isidentifier() for part in rel_parts):
                    continue
            except ValueError:
                # Should not happen as we are walking root_path
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
