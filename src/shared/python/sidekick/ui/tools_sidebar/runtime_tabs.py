"""Embedded Sidekick runtime tab builders.

The concrete runtime widgets live in focused sibling modules.  This module
keeps the historical import surface stable for hosts and tests.
"""

from __future__ import annotations

import importlib as importlib
import logging
from functools import partial
from typing import Any

from . import design_tokens as theme
from .appearance import (
    DEFAULT_DARK_PANEL_APPEARANCE,
    PanelAppearance,
    coerce_appearance,
)
from .calculator_assist import (
    calculator_predictive_text_enabled,
    calculator_startup_config,
    set_calculator_predictive_text_enabled,
)
from .calculator_runtime import SidekickCalculatorWidget
from .calculator_startup import CalculatorStartupConfig, default_repl_startup_config
from .chat_tab import (
    SIDEKICK_CHAT_RUNTIME_OBJECT_NAME,
    SIDEKICK_CHAT_STATUS_OBJECT_NAME,
)
from .chat_tab import (
    _build_chat_status_tab as _build_chat_status_tab,
)
from .chat_tab import (
    _build_pyqt_chat_dock as _build_pyqt_chat_dock,
)
from .chat_tab import (
    _format_chat_import_error as _format_chat_import_error,
)
from .chat_tab import (
    _replace_sidebar_tab_widget as _replace_sidebar_tab_widget,
)
from .chat_tab import (
    _resolve_accent_color as _resolve_accent_color,
)
from .chat_tab import (
    _retry_chat_dock as _retry_chat_dock,
)
from .help_content import DEFAULT_SIDEBAR_TAB_HELP
from .notes_tab import SIDEKICK_NOTES_OBJECT_NAME, SidekickNotesWidget
from .python_repl_tab import (
    SIDEKICK_PYTHON_REPL_OBJECT_NAME,
    SIDEKICK_TERMINAL_OBJECT_NAME,
    PythonReplWidget,
    SetVariable,
    SidekickPythonReplWidget,
)
from .qt_compat import QT_API, QtWidgets

__all__ = [
    "PythonReplWidget",
    "SIDEKICK_CHAT_RUNTIME_OBJECT_NAME",
    "SIDEKICK_CHAT_STATUS_OBJECT_NAME",
    "SIDEKICK_NOTES_OBJECT_NAME",
    "SIDEKICK_PYTHON_REPL_OBJECT_NAME",
    "SIDEKICK_TERMINAL_OBJECT_NAME",
    "SetVariable",
    "SidekickNotesWidget",
    "SidekickPythonReplWidget",
    "build_calculator_tab",
    "build_chat_tab",
    "build_notes_tab",
    "build_python_repl_tab",
    "build_terminal_tab",
]

_logger = logging.getLogger(__name__)


def build_chat_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the embedded chat tab for a Sidekick sidebar."""
    if QT_API == "PyQt6":
        widget = _build_pyqt_chat_dock(sidebar)
        if widget is not None:
            widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["chat"]["summary"])
            return widget
    return _build_chat_status_tab(sidebar)


def build_terminal_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the OS-level terminal tab (UpstreamDrift #5617)."""
    from .os_terminal import SidekickOsTerminalWidget

    widget = SidekickOsTerminalWidget(
        project_root=sidebar.project_root,
        parent=sidebar,
    )
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["terminal"]["summary"])
    return widget


def build_python_repl_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the Python REPL tab (UpstreamDrift #5617)."""
    startup_config, appearance = _repl_settings_from_sidebar(sidebar)
    widget = SidekickPythonReplWidget(
        registry=sidebar.registry,
        set_variable=sidebar.set_context_variable,
        terminal_theme=theme.SidekickTerminalTheme.inherited(
            getattr(sidebar, "_design_tokens", None),
        ),
        startup_config=startup_config,
        appearance=appearance,
        parent=sidebar,
    )
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["python_repl"]["summary"])
    return widget


def _repl_settings_from_sidebar(
    sidebar: Any,
) -> tuple[CalculatorStartupConfig, PanelAppearance]:
    """Read persisted REPL startup imports + appearance from ``sidebar``."""
    values: dict[str, Any] = {}
    getter = getattr(sidebar, "tab_settings", None)
    if callable(getter):
        try:
            payload = getter("python_repl")
            if isinstance(payload, dict) and isinstance(payload.get("values"), dict):
                values = payload["values"]
        except Exception:  # noqa: BLE001 - degrade to defaults on store error
            _logger.debug("Reading python_repl tab settings failed", exc_info=True)
    raw_imports = values.get("startup_imports")
    if raw_imports in (None, ""):
        startup_config = default_repl_startup_config()
    else:
        try:
            startup_config = CalculatorStartupConfig.from_list(raw_imports)
        except (TypeError, ValueError):
            startup_config = default_repl_startup_config()
    return startup_config, coerce_appearance(values, DEFAULT_DARK_PANEL_APPEARANCE)


def build_calculator_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build an embedded symbolic calculator tab bound to workspace state."""
    widget = SidekickCalculatorWidget(
        registry=sidebar.registry,
        set_variable=sidebar.set_context_variable,
        predictive_text_enabled=calculator_predictive_text_enabled(sidebar),
        startup_import_config=calculator_startup_config(sidebar),
        set_predictive_text_enabled=partial(
            set_calculator_predictive_text_enabled,
            sidebar,
        ),
        refresh_workspace=sidebar.refresh_workspace,
        parent=sidebar,
    )
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["calculator"]["summary"])
    return widget


def build_notes_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build a project-persistent notes tab."""
    widget = SidekickNotesWidget(project_root=sidebar.project_root, parent=sidebar)
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["notes"]["summary"])
    return widget
