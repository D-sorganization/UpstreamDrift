"""
Stage 2 hygiene test: after the upstream_drift_tools -> sidekick rename,
no src/ or tests/ file (except this one) should import from upstream_drift_tools.

Stage 3 (#6564): the deprecated alias package itself has been removed, so the
directory must no longer exist anywhere in the tree.

Issue: #5619, #6564
"""

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[3]
pytestmark = pytest.mark.unit

DEPRECATED_PACKAGE_DIR = (
    REPO_ROOT / "src" / "shared" / "python" / "upstream_drift_tools"
)
DEPRECATED_PACKAGE_NAME = "upstream_drift_tools"


def _is_deprecated_module(module_name: str | None) -> bool:
    return module_name == DEPRECATED_PACKAGE_NAME or bool(
        module_name and module_name.startswith(f"{DEPRECATED_PACKAGE_NAME}.")
    )


def _has_deprecated_import(content: str) -> bool:
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            _is_deprecated_module(alias.name) for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and _is_deprecated_module(node.module):
            return True
    return False


def _find_deprecated_imports(directory: str, ignore_self: bool = False) -> list[str]:
    found_files = []
    dir_path = REPO_ROOT / directory
    for py_file in dir_path.rglob("*.py"):
        if ignore_self and py_file.name == "test_no_deprecated_imports.py":
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            if _has_deprecated_import(content):
                found_files.append(
                    str(py_file.relative_to(REPO_ROOT)).replace("\\", "/")
                )
        except (SyntaxError, UnicodeDecodeError):
            pass
    return found_files


def test_deprecated_import_detection_ignores_literal_strings():
    content = 'MODULE = "upstream_drift_tools"\n'

    assert not _has_deprecated_import(content)


def test_deprecated_import_detection_flags_import_statements():
    assert _has_deprecated_import("import upstream_drift_tools\n")
    assert _has_deprecated_import("import upstream_drift_tools.theme\n")
    assert _has_deprecated_import("from upstream_drift_tools.theme import Colors\n")


def test_no_upstream_drift_tools_imports_in_src():
    """After Stage 2, no src/ file should import from upstream_drift_tools."""
    files = _find_deprecated_imports("src")
    assert files == [], f"Found upstream_drift_tools imports in src/: {files}"


def test_no_upstream_drift_tools_imports_in_tests():
    """After Stage 2, no tests/ file (except this hygiene test) should import from upstream_drift_tools."""
    files = _find_deprecated_imports("tests", ignore_self=True)
    assert files == [], f"Found upstream_drift_tools imports in tests/: {files}"


def test_deprecated_alias_package_removed():
    """Stage 3 (#6564): the deprecated alias package must no longer exist."""
    assert not DEPRECATED_PACKAGE_DIR.exists(), (
        "The deprecated upstream_drift_tools/ package still exists at "
        f"{DEPRECATED_PACKAGE_DIR}; migrate remaining imports to sidekick "
        "and delete the directory."
    )
