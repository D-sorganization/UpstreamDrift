#!/usr/bin/env python3
"""
Apply quick fixes to the repository.
"""

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def add_missing_init_files():
    """Add __init__.py to directories containing python files that look like packages."""
    root = Path(".")
    fixed_count = 0

    # Heuristic: directories that look like they are part of a package structure
    # e.g. under 'src', 'shared', 'engines'
    target_roots = ["engines", "shared", "api", "launchers"]

    for target in target_roots:
        target_path = root / target
        if not target_path.exists():
            continue

        for path in target_path.rglob("*"):
            if path.is_dir():
                # Check if it contains .py files
                has_py = any(path.glob("*.py"))
                # Check if it has __init__.py
                has_init = (path / "__init__.py").exists()

                if has_py and not has_init:
                    # Exclude some directories like 'scripts' or 'tests' if ambiguous,
                    # but usually tests need init too.
                    # Let's verify it's not a purely data directory (already checked for .py)

                    logger.info(f"Adding __init__.py to {path}")
                    with open(path / "__init__.py", "w") as f:
                        f.write(f'"""Package initialization for {path.name}."""\n')
                    fixed_count += 1

    logger.info(f"Added {fixed_count} missing __init__.py files.")

if __name__ == "__main__":
    add_missing_init_files()
