#!/usr/bin/env python3
"""Migrate imports from backward-compatibility shim paths to canonical paths.

Reads each shim file, determines the real module path, finds all importers,
rewrites their import statements, then removes the shim file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED_DIR = REPO_ROOT / "src" / "shared" / "python"


def get_shim_mapping(shim_path: Path) -> str | None:
    """Extract the real module path from a shim file.

    Returns the dotted subpackage path (e.g., 'physics.aerodynamics')
    or None if this is not a shim file.
    """
    content = shim_path.read_text(encoding="utf-8")
    if "sys.modules[__name__]" not in content:
        return None

    # Match: from .physics import aerodynamics as _real_module
    match = re.search(r"from \.(\S+) import \S+ as _real_module", content)
    if match:
        raw = match.group(1)
        # Convert 'physics' -> 'physics.aerodynamics' using shim name
        module_name = shim_path.stem
        return f"{raw}.{module_name}"
    return None


def find_importers(
    shim_name: str, search_dirs: list[Path], exclude_path: Path
) -> list[tuple[Path, int, str]]:
    """Find all files importing from the shim path.

    Returns list of (filepath, line_number, line_content) tuples.
    Excludes the shim file itself and __pycache__ dirs.
    """
    results: list[tuple[Path, int, str]] = []
    pattern = re.compile(
        rf"(?:from\s+src\.shared\.python\.{shim_name}\s+import\s)"
        rf"|(?:from\s+src\.shared\.python\s+import\s+{shim_name}\b)"
        rf"|(?:import\s+src\.shared\.python\.{shim_name}\b)"
    )
    # Also match relative imports in shared subpackages
    rel_pattern = re.compile(
        rf"(?:from\s+\.\.{shim_name}\s+import\s)"
    )

    for search_dir in search_dirs:
        for py_file in search_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            if py_file.resolve() == exclude_path.resolve():
                continue

            try:
                lines = py_file.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, PermissionError):
                continue

            for i, line in enumerate(lines, 1):
                stripped = line.lstrip()
                # Skip lines in docstrings/comments
                if stripped.startswith("#"):
                    continue
                if pattern.search(line) or rel_pattern.search(line):
                    # Extra check: make sure it's not inside a docstring
                    # (simple heuristic: skip if line is inside triple quotes)
                    results.append((py_file, i, line))

    return results


def rewrite_import(
    filepath: Path,
    shim_name: str,
    real_subpath: str,
) -> bool:
    """Rewrite imports in a file from shim path to real path.

    Returns True if changes were made.
    """
    content = filepath.read_text(encoding="utf-8")
    original = content

    # Replace: from src.shared.python.shim_name import X
    # With:    from src.shared.python.real_subpath import X
    old_pattern = f"from src.shared.python.{shim_name} import"
    new_pattern = f"from src.shared.python.{real_subpath} import"
    content = content.replace(old_pattern, new_pattern)

    # Replace relative imports: from ..shim_name import X
    old_rel = f"from ..{shim_name} import"
    new_rel = f"from ..{real_subpath} import"
    content = content.replace(old_rel, new_rel)

    if content != original:
        filepath.write_text(content, encoding="utf-8")
        return True
    return False


def process_shim(shim_path: Path, dry_run: bool = False) -> dict:
    """Process a single shim file.

    Returns a summary dict.
    """
    shim_name = shim_path.stem
    real_subpath = get_shim_mapping(shim_path)

    if real_subpath is None:
        return {"shim": shim_name, "status": "not_a_shim"}

    search_dirs = [
        REPO_ROOT / "src",
        REPO_ROOT / "tests",
        REPO_ROOT / "scripts",
    ]

    importers = find_importers(shim_name, search_dirs, shim_path)

    # Filter out docstring matches (lines containing the import but inside
    # triple-quoted strings)
    real_importers = []
    for fpath, lineno, line in importers:
        stripped = line.strip()
        # Skip if it looks like a docstring example
        if stripped.startswith(('"""', "'''", "Usage:", "Example:", ">>>")):
            continue
        # Skip if indented with 4+ spaces AND inside a multi-line string
        # (heuristic: if the line starts with spaces and contains 'from')
        content = fpath.read_text(encoding="utf-8")
        lines = content.splitlines()
        if lineno <= len(lines):
            # Check if we're inside a docstring by counting triple quotes
            preceding = "\n".join(lines[: lineno - 1])
            triple_count = preceding.count('"""') + preceding.count("'''")
            if triple_count % 2 == 1:
                # Inside a docstring
                continue
        real_importers.append((fpath, lineno, line))

    result = {
        "shim": shim_name,
        "real_path": real_subpath,
        "importers": len(real_importers),
        "files_changed": [],
        "status": "ok",
    }

    if dry_run:
        for fpath, lineno, line in real_importers:
            result["files_changed"].append(
                f"  {fpath.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}"
            )
        return result

    # Rewrite imports
    changed_files = set()
    for fpath, _lineno, _line in real_importers:
        if rewrite_import(fpath, shim_name, real_subpath):
            changed_files.add(fpath)

    result["files_changed"] = [
        str(f.relative_to(REPO_ROOT)) for f in sorted(changed_files)
    ]

    # Delete shim file
    shim_path.unlink()
    result["status"] = "removed"

    return result


def main() -> None:
    """Main entry point."""
    dry_run = "--dry-run" in sys.argv
    max_imports = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 999

    # Find all shim files
    shim_files = sorted(SHARED_DIR.glob("*.py"))

    total_removed = 0
    total_imports_updated = 0

    for shim_path in shim_files:
        if shim_path.name == "__init__.py":
            continue

        mapping = get_shim_mapping(shim_path)
        if mapping is None:
            continue

        # Count importers first
        search_dirs = [
            REPO_ROOT / "src",
            REPO_ROOT / "tests",
            REPO_ROOT / "scripts",
        ]
        importers = find_importers(shim_path.stem, search_dirs, shim_path)
        if len(importers) > max_imports:
            if dry_run:
                print(
                    f"SKIP {shim_path.stem} ({len(importers)} imports > {max_imports})"
                )
            continue

        result = process_shim(shim_path, dry_run=dry_run)

        if result["status"] == "not_a_shim":
            continue

        prefix = "DRY" if dry_run else "OK"
        print(
            f"[{prefix}] {result['shim']} -> {result['real_path']} "
            f"({result['importers']} imports)"
        )
        if result.get("files_changed"):
            for f in result["files_changed"]:
                print(f"     {f}")

        if not dry_run and result["status"] == "removed":
            total_removed += 1
            total_imports_updated += result["importers"]

    print(f"\n{'DRY RUN - ' if dry_run else ''}Summary:")
    print(f"  Shims removed: {total_removed}")
    print(f"  Imports updated: {total_imports_updated}")


if __name__ == "__main__":
    main()
