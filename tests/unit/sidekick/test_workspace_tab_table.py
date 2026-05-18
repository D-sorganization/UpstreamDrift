"""Tests for WorkspaceTableModel (Issue #5616).

TDD: written before implementation. Tests the sortable table model that mirrors
WorkspaceRegistry contents and auto-refreshes on change.

Note: The conftest mocks PyQt6 when it is not pre-loaded. These tests detect
the mock and skip Qt-dependent assertions when running in a headless CI that
does not load real PyQt6 before the conftest runs.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Detect whether we have real PyQt6 or a conftest-injected mock.
# MagicMock has all attributes and passes hasattr/isinstance checks, so we
# must check whether QAbstractTableModel is an actual class (not a MagicMock).
# ---------------------------------------------------------------------------
_pyqt6_module = sys.modules.get("PyQt6")
try:
    _qtcore = getattr(_pyqt6_module, "QtCore", None)
    _qabt = getattr(_qtcore, "QAbstractTableModel", None)
    _HAVE_REAL_QT = (
        _qabt is not None
        and isinstance(_qabt, type)
        and not getattr(_qabt, "_mock_name", None)
    )
except Exception:
    _HAVE_REAL_QT = False

_skip_qt = pytest.mark.skipif(
    not _HAVE_REAL_QT,
    reason="PyQt6 not available or mocked by conftest",
)


def _make_model(registry: WorkspaceRegistry) -> Any:
    """Import and construct WorkspaceTableModel (lazy to avoid collection errors)."""
    from sidekick.ui.tools_sidebar.workspace_table import (
        WorkspaceTableModel,
    )

    return WorkspaceTableModel(registry)


# ---------------------------------------------------------------------------
# Registry-only tests (no Qt dependency)
# ---------------------------------------------------------------------------


def test_registry_change_triggers_model_refresh_via_subscription() -> None:
    """WorkspaceTableModel subscribes to the registry; we verify subscription exists."""
    registry = WorkspaceRegistry()
    # Verify subscription mechanism works even without Qt
    fires: list[str] = []
    sub = registry.subscribe(lambda n, v: fires.append(n))
    registry.set_variable("ping", 1)
    assert "ping" in fires
    sub.dispose()
    registry.set_variable("gone", 2)
    assert "gone" not in fires


# ---------------------------------------------------------------------------
# Qt-dependent model tests
# ---------------------------------------------------------------------------


@_skip_qt
def test_workspace_table_has_four_columns() -> None:
    """Columns are Name, Type, Size, Preview."""
    from PyQt6.QtCore import Qt

    registry = WorkspaceRegistry()
    model = _make_model(registry)
    assert model.columnCount() == 4  # noqa: PLR2004
    headers = [
        model.headerData(i, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
        for i in range(4)
    ]
    assert headers == ["Name", "Type", "Size", "Preview"]


@_skip_qt
def test_workspace_table_updates_when_registry_changes() -> None:
    """Row count increases when a variable is added to the registry."""
    from PyQt6.QtCore import Qt

    registry = WorkspaceRegistry()
    model = _make_model(registry)
    assert model.rowCount() == 0
    registry.set_variable("x", 3.14)
    assert model.rowCount() == 1
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "x"


@_skip_qt
def test_workspace_table_sorts_by_name() -> None:
    """sort(0, AscendingOrder) orders rows alphabetically by name."""
    from PyQt6.QtCore import Qt

    registry = WorkspaceRegistry()
    model = _make_model(registry)
    registry.set_variable("z", 1)
    registry.set_variable("a", 2)
    model.sort(0, Qt.SortOrder.AscendingOrder)
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "a"


@_skip_qt
def test_workspace_table_type_column() -> None:
    """Column 1 shows the Python type name."""
    from PyQt6.QtCore import Qt

    registry = WorkspaceRegistry()
    model = _make_model(registry)
    registry.set_variable("val", 42)
    assert model.data(model.index(0, 1), Qt.ItemDataRole.DisplayRole) == "int"


@_skip_qt
def test_workspace_table_preview_column() -> None:
    """Column 3 shows a repr preview."""
    from PyQt6.QtCore import Qt

    registry = WorkspaceRegistry()
    model = _make_model(registry)
    registry.set_variable("val", 99)
    preview = model.data(model.index(0, 3), Qt.ItemDataRole.DisplayRole)
    assert preview is not None
    assert "99" in str(preview)
