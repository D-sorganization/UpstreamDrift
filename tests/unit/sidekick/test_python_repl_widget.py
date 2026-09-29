"""Regression checks for the canonical Tools Python REPL widget."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
_OBSOLETE_COPY = (
    REPO_ROOT
    / "src"
    / "shared"
    / "python"
    / "sidekick"
    / "ui"
    / "tools_sidebar"
    / "python_repl.py"
)
_OBSOLETE_DEFAULT_TABS_COPY = _OBSOLETE_COPY.with_name("default_tabs.py")


def _tools_python_root() -> Path:
    """Return the explicit Tools candidate or the pinned vendored checkout."""
    tools_root = Path(
        os.environ.get("TOOLS_REPO_PATH", REPO_ROOT / "vendor" / "ud-tools")
    )
    return (tools_root / "src" / "shared" / "python").resolve()


def _canonical_module(request: pytest.FixtureRequest) -> Any:
    if request.config.getoption("--tools-mode") != "vendored":
        pytest.skip("canonical Tools widget tests require --tools-mode vendored")
    module = importlib.import_module("sidekick.ui.tools_sidebar.python_repl_tab")
    assert Path(module.__file__).resolve().is_relative_to(_tools_python_root())
    return module


def test_obsolete_downstream_python_repl_copy_is_absent() -> None:
    """The Tools implementation is the only supported Python REPL surface."""
    assert not _OBSOLETE_COPY.exists()


def test_obsolete_downstream_default_tabs_copy_is_absent() -> None:
    """The canonical Tools composition must not be shadowed by stale imports."""
    assert not _OBSOLETE_DEFAULT_TABS_COPY.exists()


def test_python_repl_comes_from_pinned_tools_candidate(
    request: pytest.FixtureRequest,
) -> None:
    """The supported module must resolve from the pinned Tools checkout."""
    assert _canonical_module(request).PythonReplWidget.__name__ == "PythonReplWidget"


@pytest.fixture(scope="module")
def qapp() -> Any:
    """Provide a QApplication when the local PyQt6 runtime is usable."""
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 is unavailable: {exc}")
    app = QApplication.instance()
    return app if app is not None else QApplication([])


def test_canonical_python_repl_exports_workspace_assignment(
    qapp: Any,
    request: pytest.FixtureRequest,
) -> None:
    """The canonical REPL writes evaluated assignments through its callback."""
    repl_module = _canonical_module(request)
    registry_module = importlib.import_module("sidekick.ui.tools_sidebar.registry")
    registry = registry_module.WorkspaceRegistry()
    repl = repl_module.PythonReplWidget(
        registry=registry,
        set_variable=registry.set,
    )

    repl.execute("answer = 42")
    qapp.processEvents()

    assert registry.get("answer") == 42
    assert repl.history() == ("answer = 42",)
