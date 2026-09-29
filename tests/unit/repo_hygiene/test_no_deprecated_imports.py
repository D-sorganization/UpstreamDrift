"""
Stage 2 hygiene test: after the upstream_drift_tools -> sidekick rename,
no src/ or tests/ file (except this one) should import from upstream_drift_tools.

Stage 3 (#6564): the deprecated alias package itself has been removed, so the
directory must no longer exist anywhere in the tree.

Issue: #5619, #6564
"""

import pathlib

REPO_ROOT = pathlib.Path(__file__).parents[3]

DEPRECATED_PACKAGE_DIR = (
    REPO_ROOT / "src" / "shared" / "python" / "upstream_drift_tools"
)


def _find_deprecated_imports(directory: str, ignore_self: bool = False) -> list[str]:
    found_files = []
    dir_path = REPO_ROOT / directory
    for py_file in dir_path.rglob("*.py"):
        if ignore_self and py_file.name == "test_no_deprecated_imports.py":
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            if "upstream_drift_tools" in content:
                found_files.append(
                    str(py_file.relative_to(REPO_ROOT)).replace("\\", "/")
                )
        except UnicodeDecodeError:
            pass
    return found_files


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
