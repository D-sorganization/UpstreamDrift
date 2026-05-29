"""Tests for PythonReplWidget (Issue #5616).

TDD: written before implementation. Tests the shared REPL widget that
evaluates expressions, records history, and syncs assignments to a
WorkspaceRegistry.

Note: The conftest mocks PyQt6 when it is not pre-loaded. These tests detect
the mock and skip Qt-dependent assertions accordingly.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Detect whether we have real PyQt6 or a conftest-injected mock.
# MagicMock has all attributes, so check isinstance(..., type).
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
except Exception:  # noqa: BLE001 - any failure means real Qt is unavailable
    _HAVE_REAL_QT = False

_skip_qt = pytest.mark.skipif(
    not _HAVE_REAL_QT,
    reason="PyQt6 not available or mocked by conftest",
)


def _make_repl(namespace: dict, registry: WorkspaceRegistry | None = None) -> Any:
    """Construct PythonReplWidget lazily."""
    from sidekick.ui.tools_sidebar.python_repl import PythonReplWidget

    return PythonReplWidget(namespace=namespace, registry=registry)


@pytest.fixture(scope="module")
def qapp() -> Any:
    """Provide a module-scoped QApplication for the REPL tests."""
    if not _HAVE_REAL_QT:
        pytest.skip("Real Qt not available")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if not app:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# Qt-dependent tests
# ---------------------------------------------------------------------------


@_skip_qt
def test_repl_evaluates_expression(qapp: Any) -> None:
    """evaluate('2 + 2') returns a string containing '4'."""
    repl = _make_repl(namespace={})
    result = repl.evaluate("2 + 2")
    assert "4" in result


@_skip_qt
def test_repl_assignment_updates_registry(qapp: Any) -> None:
    """Assignment in REPL propagates value to the WorkspaceRegistry."""
    registry = WorkspaceRegistry()
    repl = _make_repl(namespace={}, registry=registry)
    repl.evaluate("x = 3")
    assert registry.get_variable("x") == 3  # noqa: PLR2004


@_skip_qt
def test_repl_history_is_recorded(qapp: Any) -> None:
    """Each evaluated expression is appended to history."""
    repl = _make_repl(namespace={})
    repl.evaluate("1 + 1")
    assert "1 + 1" in repl.history


@_skip_qt
def test_repl_exception_does_not_crash(qapp: Any) -> None:
    """Exceptions in user code are caught and returned as a string."""
    repl = _make_repl(namespace={})
    result = repl.evaluate("1 / 0")
    assert "ZeroDivisionError" in result


@_skip_qt
def test_repl_multiple_assignments(qapp: Any) -> None:
    """Multiple assignments update the registry correctly."""
    registry = WorkspaceRegistry()
    repl = _make_repl(namespace={}, registry=registry)
    repl.evaluate("a = 10\nb = 20")
    assert registry.get_variable("a") == 10  # noqa: PLR2004
    assert registry.get_variable("b") == 20  # noqa: PLR2004


@_skip_qt
def test_repl_no_registry_still_evaluates(qapp: Any) -> None:
    """Widget works without a registry."""
    repl = _make_repl(namespace={})
    result = repl.evaluate("3 * 7")
    assert "21" in result


# ---------------------------------------------------------------------------
# Non-Qt behaviour tests (evaluate logic can be tested on the logic layer)
# ---------------------------------------------------------------------------


def test_repl_evaluate_logic_expression() -> None:
    """Standalone test: evaluate helper produces correct output."""
    from sidekick.ui.tools_sidebar.python_repl import _evaluate_in_namespace

    ns: dict = {}
    result = _evaluate_in_namespace("2 + 2", ns)
    assert "4" in result


def test_repl_evaluate_logic_assignment() -> None:
    """Assignment via eval helper populates the namespace."""
    from sidekick.ui.tools_sidebar.python_repl import _evaluate_in_namespace

    ns: dict = {}
    _evaluate_in_namespace("x = 42", ns)
    assert ns.get("x") == 42  # noqa: PLR2004


def test_repl_evaluate_logic_exception() -> None:
    """Exceptions are returned as formatted strings, not raised."""
    from sidekick.ui.tools_sidebar.python_repl import _evaluate_in_namespace

    ns: dict = {}
    result = _evaluate_in_namespace("1 / 0", ns)
    assert "ZeroDivisionError" in result
