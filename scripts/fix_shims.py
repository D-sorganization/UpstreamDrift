"""Script to fix all backward compatibility shim files in src/shared/python/.

For each shim file that uses the sys.modules[__name__] pattern,
this script reads the target module, discovers all public names,
and adds explicit re-exports so mypy can resolve them.
"""

import ast
import re
import sys
from pathlib import Path


def get_public_names(module_path: Path) -> list[str]:
    """Extract all public names from a Python module file."""
    if not module_path.exists():
        return []

    try:
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except SyntaxError:
        return []

    names: list[str] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_"):
                names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                names.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("_"):
                names.append(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            # Re-export star imports by checking __all__ instead
            pass

    # Check for __all__ to get the full list
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                if elt.value not in names:
                                    names.append(elt.value)

    return sorted(set(names))


def fix_shim_file(shim_path: Path) -> bool:
    """Fix a single shim file by adding explicit re-exports."""
    content = shim_path.read_text(encoding="utf-8")

    # Already fixed?
    if content.count("import") > 3:
        return False

    # Extract the import pattern: from .X.Y import Z as _real_module
    # or from .X import Y as _real_module
    import_match = re.search(
        r"from\s+(\.[\w.]+)\s+import\s+(\w+)\s+as\s+_real_module",
        content,
    )
    if not import_match:
        return False

    parent_path = import_match.group(1)  # e.g., ".core" or ".logging_pkg"
    module_name = import_match.group(2)  # e.g., "constants" or "logging_config"

    # Convert import path to file path
    # parent_path starts with "." meaning relative to src/shared/python/
    rel_parts = parent_path.lstrip(".").split(".")
    target_dir = shim_path.parent
    for part in rel_parts:
        target_dir = target_dir / part
    target_file = target_dir / f"{module_name}.py"

    if not target_file.exists():
        # Try as a package
        target_file = target_dir / module_name / "__init__.py"
        if not target_file.exists():
            print(f"  SKIP: target not found: {target_file}")
            return False

    # Get public names from the target module
    names = get_public_names(target_file)
    if not names:
        print(f"  SKIP: no public names found in {target_file}")
        return False

    # Build the import path for the re-exports
    import_from = f"{parent_path}.{module_name}"

    # Build new content
    docstring_match = re.search(r'(""".*?""")', content, re.DOTALL)
    docstring = docstring_match.group(1) if docstring_match else f'"""Backward compatibility shim for {module_name}."""'

    # Format the explicit imports
    import_lines = ",\n".join(f"    {name}" for name in names)

    new_content = f'''{docstring}

import sys as _sys

from {import_from} import (  # noqa: F401
{import_lines},
)

from {parent_path} import {module_name} as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
'''

    shim_path.write_text(new_content, encoding="utf-8")
    print(f"  FIXED: {shim_path.name} ({len(names)} exports from {import_from})")
    return True


def main() -> None:
    """Fix all shim files."""
    shared_dir = Path("src/shared/python")
    shim_files = sorted(shared_dir.glob("*.py"))

    fixed_count = 0
    skip_count = 0

    for shim_file in shim_files:
        if shim_file.name == "__init__.py":
            continue

        content = shim_file.read_text(encoding="utf-8")
        if "_sys.modules[__name__]" not in content:
            continue

        print(f"Processing: {shim_file.name}")
        if fix_shim_file(shim_file):
            fixed_count += 1
        else:
            skip_count += 1

    print(f"\nDone: {fixed_count} fixed, {skip_count} skipped")


if __name__ == "__main__":
    main()
