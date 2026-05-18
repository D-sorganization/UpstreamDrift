"""PythonReplWidget — shared REPL widget for Sidekick sidebar.

Provides a PyQt6 widget that evaluates Python expressions in a shared
namespace, records command history, and optionally syncs assignments to a
WorkspaceRegistry.

Design-by-Contract:
- evaluate(code): postcondition returns a non-None string; never raises.
- Assignments in code are written to the registry via set_variable().
- LOD: the widget receives the registry directly; does not reach into sidebar.
"""

from __future__ import annotations

import contextlib
import io
import logging
import types
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["PythonReplWidget", "_evaluate_in_namespace"]

_RESERVED_NAMES: frozenset[str] = frozenset(
    {
        "__builtins__",
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "np",
        "numpy",
        "pd",
        "pandas",
    }
)


def _evaluate_in_namespace(code: str, namespace: dict[str, Any]) -> str:
    """Execute *code* in *namespace* and return captured output.

    Postcondition: always returns a string; never raises.  Exceptions are
    formatted and included in the return value.

    For expressions (single-line, no assignment), the repr of the result is
    also included when non-None.

    Args:
        code: Python source code (expression or statement block).
        namespace: Mutable namespace dict used for execution.

    Returns:
        String containing stdout, stderr, and any exception message.
    """
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        compiled = compile(code, "<sidekick-repl>", "exec")
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compiled, namespace, namespace)  # noqa: S102  # nosec B102
    except Exception as exc:  # noqa: BLE001
        logger.debug("REPL execution error: %s", exc)
        parts = [s for s in (stdout.getvalue(), stderr.getvalue()) if s]
        parts.append(f"{type(exc).__name__}: {exc}")
        return "".join(parts).strip()

    # Try to also capture the repr of the last expression if the code is a
    # single line that looks like an expression (not an assignment).
    result_repr: str | None = None
    stripped = code.strip()
    if "\n" not in stripped and "=" not in stripped and stripped:
        with contextlib.suppress(Exception):
            val = eval(stripped, namespace)  # noqa: S307  # nosec B307
            if val is not None:
                result_repr = repr(val)

    parts = [s for s in (stdout.getvalue(), stderr.getvalue()) if s]
    if result_repr:
        parts.append(result_repr)
    return "".join(parts).strip() if parts else "Executed."


def _exportable_items(namespace: dict[str, Any]) -> dict[str, Any]:
    """Return namespace items that can be exported to the registry."""
    return {
        name: value
        for name, value in namespace.items()
        if _is_exportable_name(name) and _is_exportable_value(value)
    }


def _is_exportable_name(name: str) -> bool:
    return bool(name) and not name.startswith("_") and name not in _RESERVED_NAMES


def _is_exportable_value(value: Any) -> bool:
    return not isinstance(value, types.ModuleType) and not callable(value)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


class PythonReplWidget:
    """Shared Python REPL widget (PyQt6 or fallback shim).

    Evaluates Python code, records history, and optionally syncs variable
    assignments to a WorkspaceRegistry.

    Usage::

        registry = WorkspaceRegistry()
        repl = PythonReplWidget(namespace={}, registry=registry)
        output = repl.evaluate("x = 3.14")

    Args:
        namespace: Initial globals dict for the REPL.
        registry: Optional WorkspaceRegistry to receive variable assignments.
        parent: Optional Qt parent widget.
    """

    def __init__(
        self,
        namespace: dict[str, Any],
        registry: Any = None,
        parent: Any = None,
    ) -> None:
        self._namespace: dict[str, Any] = dict(namespace)
        self._registry = registry
        self._history: list[str] = []
        self._qt_widget: Any = None
        self._try_build_qt_widget(parent)

    def _try_build_qt_widget(self, parent: Any) -> None:
        """Attempt to build the PyQt6 widget; silently skip if unavailable."""
        try:
            from PyQt6.QtWidgets import (
                QPlainTextEdit,
                QPushButton,
                QVBoxLayout,
                QWidget,
            )
            from PyQt6.QtCore import QAbstractTableModel

            if not hasattr(QAbstractTableModel, "rowCount"):
                # We got a MagicMock; skip widget construction
                return

            widget = QWidget(parent)
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(4)

            self._input_edit = QPlainTextEdit(widget)
            self._input_edit.setObjectName("PythonReplInput")
            self._input_edit.setPlaceholderText("Type Python code here…")
            layout.addWidget(self._input_edit, stretch=2)

            run_button = QPushButton("Run", widget)
            run_button.setObjectName("PythonReplRun")
            run_button.clicked.connect(self._on_run_clicked)
            layout.addWidget(run_button)

            self._output_edit = QPlainTextEdit(widget)
            self._output_edit.setObjectName("PythonReplOutput")
            self._output_edit.setReadOnly(True)
            layout.addWidget(self._output_edit, stretch=3)

            self._qt_widget = widget
        except Exception:  # noqa: BLE001
            pass

    def _on_run_clicked(self) -> None:
        """Handler for the Run button in the Qt widget."""
        code = self._input_edit.toPlainText()
        result = self.evaluate(code)
        self._output_edit.setPlainText(result)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self) -> Any:
        """Return the underlying Qt widget (may be None in headless mode)."""
        return self._qt_widget

    @property
    def history(self) -> list[str]:
        """Return the ordered list of evaluated code strings."""
        return list(self._history)

    def evaluate(self, code: str) -> str:
        """Execute *code* in the shared namespace.

        Postcondition: returns a non-None string; never raises.
        Assignments detected in the namespace after execution are propagated
        to the WorkspaceRegistry if one was provided.

        Args:
            code: Python source code.

        Returns:
            Captured stdout/stderr and/or exception text.
        """
        if not code.strip():
            return "No code to run."
        self._history.append(code)
        before_keys = set(self._namespace.keys())
        result = _evaluate_in_namespace(code, self._namespace)
        self._sync_new_variables(before_keys)
        return result

    def _sync_new_variables(self, before_keys: set[str]) -> None:
        """Push newly assigned variables to the registry."""
        if self._registry is None:
            return
        new_exports = _exportable_items(self._namespace)
        for name, value in new_exports.items():
            if name not in before_keys or self._namespace.get(name) != value:
                with contextlib.suppress(Exception):
                    self._registry.set_variable(name, value)
