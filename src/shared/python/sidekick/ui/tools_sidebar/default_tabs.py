"""Default Sidekick tab builders kept outside the sidebar controller."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from . import design_tokens as theme
from .calculator_plotting import CALCULATOR_PLOT_TAB_ID
from .chat_settings import CHAT_TAB_SETTINGS
from .data_explorer_tab import (
    DATA_EXPLORER_TAB_ID,
    DATA_EXPLORER_TAB_SETTINGS,
    build_data_explorer_tab,
)
from .data_processor_tab import DATA_PROCESSOR_TAB_ID, build_data_processor_tab
from .help_content import DEFAULT_SIDEBAR_TAB_HELP
from .jupyter_tab import JUPYTER_TAB_ID
from .project_file_explorer import ProjectFileExplorer
from .qt_compat import QT_API, QtWidgets
from .reporting_tab import build_reporting_tab
from .runtime_tab_settings import (
    PYTHON_REPL_TAB_SETTINGS,
    TERMINAL_TAB_SETTINGS,
    WORKSPACE_TAB_SETTINGS,
)
from .runtime_tabs import (
    build_calculator_tab,
    build_chat_tab,
    build_notes_tab,
    build_python_repl_tab,
    build_terminal_tab,
)
from .workspace_tab import (
    WORKSPACE_TABLE_COLUMNS,
    QtGui_QStandardItem,
    QtGui_QStandardItemModel,
    WorkspaceTableWidget,
)

__all__ = [
    "FUNCTION_GENERATOR_TAB_ID",
    "QtGui_QStandardItem",
    "QtGui_QStandardItemModel",
    "ROTATION_CONVERTER_TAB_ID",
    "TabDefinitionFactory",
    "WORKSPACE_TABLE_COLUMNS",
    "WorkspaceTableWidget",
    "build_calculator_plot_tab",
    "build_default_tab_definitions",
    "build_file_explorer_tab",
    "build_function_generator_tab",
    "build_jupyter_tab",
    "build_rotation_converter_tab",
    "build_unit_converter_tab",
    "build_workspace_tab",
    "placeholder",
    "refresh_workspace_list",
    "set_project_explorer_root",
]


T = TypeVar("T")

_logger = logging.getLogger(__name__)

TabDefinitionFactory = Callable[..., T]
ROTATION_CONVERTER_TAB_ID = "rotation_converter"
FUNCTION_GENERATOR_TAB_ID = "function_generator"


def build_default_tab_definitions(
    sidebar: Any,
    tab_definition: TabDefinitionFactory,
) -> list[T]:
    """Return the standard Sidekick tabs for a host sidebar."""
    return [
        tab_definition(
            "chat",
            "Chat",
            build_chat_tab,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["chat"]),
            settings=CHAT_TAB_SETTINGS,
        ),
        tab_definition(
            "files",
            "Files",
            build_file_explorer_tab,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["files"]),
        ),
        tab_definition(
            "workspace",
            "Workspace",
            build_workspace_tab,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["workspace"]),
            settings=WORKSPACE_TAB_SETTINGS,
        ),
        tab_definition(
            "terminal",
            "Terminal",
            build_terminal_tab,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["terminal"]),
            settings=TERMINAL_TAB_SETTINGS,
        ),
        tab_definition(
            "python_repl",
            "Python REPL",
            build_python_repl_tab,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["python_repl"]),
            settings=PYTHON_REPL_TAB_SETTINGS,
        ),
        tab_definition(
            "calculator",
            "Calculator",
            build_calculator_tab,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["calculator"]),
        ),
        tab_definition(
            CALCULATOR_PLOT_TAB_ID,
            "Calculator Plot",
            build_calculator_plot_tab,
            visible=False,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["calculator_plot"]),
        ),
        tab_definition(
            DATA_EXPLORER_TAB_ID,
            "Data Explorer",
            build_data_explorer_tab,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP[DATA_EXPLORER_TAB_ID]),
            settings=DATA_EXPLORER_TAB_SETTINGS,
        ),
        tab_definition(
            DATA_PROCESSOR_TAB_ID,
            "Data Processor",
            build_data_processor_tab,
            visible=False,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP[DATA_PROCESSOR_TAB_ID]),
        ),
        tab_definition(
            "units",
            "Units",
            build_unit_converter_tab,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["units"]),
        ),
        tab_definition(
            ROTATION_CONVERTER_TAB_ID,
            "Rotation Converter",
            build_rotation_converter_tab,
            visible=False,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["rotation_converter"]),
        ),
        tab_definition(
            FUNCTION_GENERATOR_TAB_ID,
            "Function Generator",
            build_function_generator_tab,
            visible=False,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP[FUNCTION_GENERATOR_TAB_ID]),
        ),
        tab_definition(
            "notes",
            "Notes",
            build_notes_tab,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["notes"]),
        ),
        tab_definition(
            "reporting",
            "Reporting",
            build_reporting_tab,
            duplicate_enabled=False,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["reporting"]),
        ),
        tab_definition(
            JUPYTER_TAB_ID,
            "Jupyter",
            build_jupyter_tab,
            visible=False,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP[JUPYTER_TAB_ID]),
        ),
    ]


def build_file_explorer_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the project file explorer tab and forward open-file signals."""
    explorer = ProjectFileExplorer(sidebar.project_root, sidebar)
    explorer.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["files"]["summary"])
    explorer.file_open_requested.connect(sidebar.file_open_requested.emit)
    return explorer


def set_project_explorer_root(
    widget: QtWidgets.QWidget | None,
    project_root: Path,
) -> None:
    """Update a file explorer widget when the host changes project roots."""
    if isinstance(widget, ProjectFileExplorer):
        widget.set_project_root(project_root)


def build_workspace_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the workspace variable inspector tab (MATLAB-style table)."""
    from .workspace_tab import build_workspace_tab as _build_workspace_tab

    return _build_workspace_tab(sidebar)


def refresh_workspace_list(sidebar: Any) -> None:
    """Refresh the workspace table widget from the sidebar registry."""
    from .workspace_tab import refresh_workspace_list as _refresh_workspace_list

    _refresh_workspace_list(sidebar)


def build_unit_converter_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the unit converter tab when the PyQt widget is available."""
    if QT_API != "PyQt6":
        return placeholder(sidebar, "Unit converter")
    try:
        from ..widgets.unit_converter_widget import (
            UnitConverterWidget,
        )
    except Exception as exc:  # noqa: BLE001 - optional GUI widget
        _logger.debug("Unit converter unavailable for Sidekick: %s", exc)
        return placeholder(sidebar, "Unit converter")
    widget = UnitConverterWidget(sidebar)
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["units"]["summary"])
    return widget


def build_calculator_plot_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the Calculator Plot tab with graceful optional dependency handling."""
    if QT_API != "PyQt6":
        return placeholder(
            sidebar,
            "Calculator Plot",
            "Calculator plotting requires the PyQt6 UI backend.",
        )
    try:
        plot_widget_module = importlib.import_module("plot_engine.pyqt6_widget")
        plot_specs_module = importlib.import_module("plot_engine.specs")
    except Exception as exc:  # noqa: BLE001 - optional plot UI dependencies
        _logger.debug("Calculator plot tab unavailable for Sidekick: %s", exc)
        return placeholder(
            sidebar,
            "Calculator Plot",
            "Calculator plotting is unavailable because optional plot UI "
            "dependencies could not be loaded.",
        )

    widget = plot_widget_module.PlotWidget(parent=sidebar)
    widget.setObjectName("SidekickCalculatorPlotTab")
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["calculator_plot"]["summary"])
    widget.set_spec(
        plot_specs_module.PlotSpec(
            title="Calculator Plot",
            series=[],
        )
    )
    return widget


def build_rotation_converter_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the Rotation Converter tab when its PyQt6 surface is available."""
    if QT_API != "PyQt6":
        return placeholder(
            sidebar,
            "Rotation Converter",
            "Rotation Converter requires the PyQt6 UI backend.",
        )
    try:
        module = importlib.import_module("rotation_converter.ui.pyqt6.main_window")
        window_type = module.RotationConverterMainWindow
        widget = window_type(sidebar)
    except Exception as exc:  # noqa: BLE001 - optional GUI surface
        _logger.debug("Rotation converter unavailable for Sidekick: %s", exc)
        return placeholder(
            sidebar,
            "Rotation Converter",
            "Rotation Converter is unavailable because optional UI dependencies "
            "could not be loaded.",
        )
    widget.setObjectName(theme.SIDEKICK_ROTATION_CONVERTER_OBJECT_NAME)
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["rotation_converter"]["summary"])
    return widget


def build_function_generator_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the Function Generator tab when its PyQt6 surface is available."""
    if QT_API != "PyQt6":
        return placeholder(
            sidebar,
            "Function Generator",
            "Function Generator requires the PyQt6 UI backend.",
        )
    try:
        registration = importlib.import_module("function_generator.gui_registration")
        gui_info = registration.get_gui_info()
        pyqt_info = gui_info["pyqt6"]
        module = importlib.import_module(pyqt_info["module"])
        widget_type = getattr(module, pyqt_info["class"])
        widget = widget_type(sidebar, use_builtin_theme=False)
    except Exception as exc:  # noqa: BLE001 - optional GUI surface
        _logger.debug("Function Generator unavailable for Sidekick: %s", exc)
        return placeholder(
            sidebar,
            "Function Generator",
            "Function Generator is unavailable because optional UI dependencies "
            "could not be loaded.",
        )
    widget.setObjectName(theme.SIDEKICK_FUNCTION_GENERATOR_OBJECT_NAME)
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP[FUNCTION_GENERATOR_TAB_ID]["summary"])
    return widget


def build_jupyter_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the Sidekick Jupyter notebook tab.

    The factory is unconditionally registered so the tab is always
    discoverable. When the optional ``nbformat`` dependency is missing,
    the factory returns :class:`JupyterUnavailableWidget` which shows
    an actionable install hint. When the dependency is present the
    tab opens an empty :class:`JupyterNotebookWidget`; loading a
    specific notebook into the tab is wired in Phase 3 (#2877).
    """
    from .jupyter_tab import (
        JupyterNotebookWidget,
        JupyterTabAvailability,
        JupyterUnavailableWidget,
        NotebookDocument,
    )

    available, install_hint = JupyterTabAvailability.check()
    if not available:
        widget = JupyterUnavailableWidget(install_hint=install_hint, parent=sidebar)
    else:
        widget = JupyterNotebookWidget(document=NotebookDocument(), parent=sidebar)
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP[JUPYTER_TAB_ID]["summary"])
    return widget


def placeholder(
    sidebar: Any,
    title: str,
    message: str | None = None,
) -> QtWidgets.QWidget:
    """Build a compact placeholder for optional tabs."""
    widget = QtWidgets.QWidget(sidebar)
    widget.setObjectName(theme.SIDEKICK_PLACEHOLDER_OBJECT_NAME)
    layout = QtWidgets.QVBoxLayout(widget)
    label = QtWidgets.QLabel(title, widget)
    label.setObjectName(theme.SIDEKICK_PLACEHOLDER_LABEL_OBJECT_NAME)
    label.setWordWrap(True)
    layout.addWidget(label)
    if message:
        detail = QtWidgets.QLabel(message, widget)
        detail.setWordWrap(True)
        layout.addWidget(detail)
    layout.addStretch(1)
    return widget
