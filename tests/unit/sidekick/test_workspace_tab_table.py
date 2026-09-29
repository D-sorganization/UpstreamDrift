"""Regression checks for the canonical Tools workspace table widget."""

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
    / "workspace_table.py"
)


def _tools_python_root() -> Path:
    """Return the explicit Tools candidate or the pinned vendored checkout."""
    tools_root = Path(
        os.environ.get("TOOLS_REPO_PATH", REPO_ROOT / "vendor" / "ud-tools")
    )
    return (tools_root / "src" / "shared" / "python").resolve()


def _canonical_module(request: pytest.FixtureRequest) -> Any:
    if request.config.getoption("--tools-mode") != "vendored":
        pytest.skip("canonical Tools widget tests require --tools-mode vendored")
    module = importlib.import_module("sidekick.ui.tools_sidebar.workspace_tab")
    assert Path(module.__file__).resolve().is_relative_to(_tools_python_root())
    return module


def test_obsolete_downstream_workspace_table_copy_is_absent() -> None:
    """The Tools workspace table is the sole supported implementation."""
    assert not _OBSOLETE_COPY.exists()


def test_workspace_table_comes_from_pinned_tools_candidate(
    request: pytest.FixtureRequest,
) -> None:
    """The supported workspace widget must resolve from pinned Tools."""
    assert _canonical_module(request).WorkspaceTableWidget.__name__ == (
        "WorkspaceTableWidget"
    )


@pytest.fixture(scope="module")
def qapp() -> Any:
    """Provide a QApplication when the local PyQt6 runtime is usable."""
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 is unavailable: {exc}")
    app = QApplication.instance()
    return app if app is not None else QApplication([])


def test_canonical_workspace_table_reflects_registry_changes(
    qapp: Any,
    request: pytest.FixtureRequest,
) -> None:
    """The canonical widget refreshes from its canonical registry."""
    workspace_module = _canonical_module(request)
    registry_module = importlib.import_module("sidekick.ui.tools_sidebar.registry")
    registry = registry_module.WorkspaceRegistry()
    widget = workspace_module.WorkspaceTableWidget(registry=registry)

    registry.set("answer", 42)
    qapp.processEvents()

    assert widget.column_headers() == ("Name", "Type", "Size", "Preview")
    assert ("answer", "int", "", "42") in widget.row_data()
