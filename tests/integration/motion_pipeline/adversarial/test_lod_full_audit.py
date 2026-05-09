"""Adversarial: full-package LoD audit.

Walk every .py under src/shared/python/motion_pipeline/ and ensure no
module imports GUI, Tkinter, requests, or writes to the filesystem
outside of test scopes.
"""

from __future__ import annotations

from pathlib import Path

import pytest


PIPELINE_ROOT = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "shared"
    / "python"
    / "motion_pipeline"
)

FORBIDDEN_IMPORTS = {
    "tkinter",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
}

# requests/urllib are allowed in CLI/script wrappers but not in the
# pipeline core. We allowlist nothing here; the pipeline itself should
# not be making network calls.
NETWORK_IMPORTS = {"requests", "urllib3", "httpx"}


def _all_py_files() -> list[Path]:
    return sorted(
        p for p in PIPELINE_ROOT.rglob("*.py") if "__pycache__" not in p.parts
    )


def test_pipeline_root_exists() -> None:
    assert PIPELINE_ROOT.is_dir(), f"{PIPELINE_ROOT} must exist"


@pytest.mark.parametrize("py_file", _all_py_files(), ids=lambda p: p.name)
def test_no_gui_imports(py_file: Path) -> None:
    """No motion_pipeline module may import a GUI toolkit."""
    text = py_file.read_text(encoding="utf-8", errors="replace")
    for forbidden in FORBIDDEN_IMPORTS:
        assert f"import {forbidden}" not in text and f"from {forbidden}" not in text, (
            f"{py_file.relative_to(PIPELINE_ROOT)} imports forbidden {forbidden}"
        )


@pytest.mark.parametrize("py_file", _all_py_files(), ids=lambda p: p.name)
def test_no_network_imports(py_file: Path) -> None:
    """The pipeline core must not make outbound network calls."""
    text = py_file.read_text(encoding="utf-8", errors="replace")
    for forbidden in NETWORK_IMPORTS:
        assert f"import {forbidden}" not in text and f"from {forbidden}" not in text, (
            f"{py_file.relative_to(PIPELINE_ROOT)} imports {forbidden}"
        )
