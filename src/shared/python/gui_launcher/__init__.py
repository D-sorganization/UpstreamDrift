"""Shared GUI Launcher Infrastructure.

This module provides a unified launcher infrastructure that supports:
- PyQt6 desktop applications
- React/Vite web applications
- Cross-repository component reuse

The launcher handles dependency checking, environment setup, and provides
consistent launch mechanisms across different GUI frameworks.

Example:
    from gui_launcher import GUILauncher, GUIType

    # Launch PyQt6 app
    launcher = GUILauncher(tool_name="data_processor", gui_type=GUIType.PYQT6)
    launcher.launch()

    # Launch React web app
    launcher = GUILauncher(tool_name="data_processor", gui_type=GUIType.REACT)
    launcher.launch()
"""

from .launcher import (
    GUILauncher,
    GUIType,
    LaunchConfig,
    launch_from_gui_info,
    launch_pyqt6_app,
    launch_tool_by_name,
    launch_web_app,
    launch_web_from_gui_info,
    make_pyqt6_launcher,
)
from .registry import GUIRegistry, auto_discover_guis, get_registry, register_gui

__all__ = [
    "GUILauncher",
    "GUIType",
    "LaunchConfig",
    "GUIRegistry",
    "auto_discover_guis",
    "launch_from_gui_info",
    "launch_pyqt6_app",
    "launch_tool_by_name",
    "launch_web_app",
    "launch_web_from_gui_info",
    "make_pyqt6_launcher",
    "register_gui",
    "get_registry",
]
