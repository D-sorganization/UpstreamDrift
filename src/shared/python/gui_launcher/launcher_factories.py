"""Factory and registry helpers for shared GUI launchers."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_launcher(
    tool_name: str,
    gui_type: Any,
    **kwargs: Any,
) -> Any:
    """Create a configured GUI launcher."""
    if tool_name is None:
        raise ValueError("tool_name must be provided")

    from .launcher import GUILauncher, LaunchConfig

    config = LaunchConfig(tool_name=tool_name, gui_type=gui_type, **kwargs)
    return GUILauncher(config=config)


def launch_tool_by_name(tool_name: str) -> int:
    """Launch a tool by its registered name."""
    from .launcher import GUIType, launch_pyqt6_app
    from .registry import get_registry

    registry = get_registry()
    registration = registry.get(tool_name)
    if registration is None:
        logger.info("Tool '%s' not found in registry.", tool_name)
        available = registry.list_tools()
        if available:
            logger.info("\nAvailable tools:")
            for reg in available:
                logger.info("  - %s (%s)", reg.tool_name, reg.display_name)
        return 1

    config = registration.gui_configs.get(GUIType.PYQT6)
    if config is None:
        logger.info("Tool '%s' has no PyQt6 configuration.", tool_name)
        return 1

    return launch_pyqt6_app(config)


def make_launcher(gui_info_module: str) -> int:
    """Convenience alias for ``make_pyqt6_launcher``."""
    return make_pyqt6_launcher(gui_info_module)


def generate_launch_script(
    gui_info_module: str,
    tool_display_name: str,
) -> str:
    """Generate the source text for a standard ``launch_pyqt6.py`` script."""
    if gui_info_module is None:
        raise ValueError("gui_info_module must be provided")
    return (
        "#!/usr/bin/env python3\n"
        f'"""Standalone PyQt6 launcher for {tool_display_name}."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "import sys\n"
        "\n"
        "from _bootstrap import bootstrap  # noqa: E402\n"
        "\n"
        "bootstrap(__file__)\n"
        "\n"
        "from gui_launcher import make_launcher  # noqa: E402\n"
        "\n"
        'if __name__ == "__main__":\n'
        f'    sys.exit(make_launcher("{gui_info_module}"))\n'
    )


def make_pyqt6_launcher(gui_info_module: str) -> int:
    """Launch a PyQt6 app from a module containing a ``GUI_INFO`` dict."""
    import importlib

    from .launcher import launch_from_gui_info

    try:
        mod = importlib.import_module(gui_info_module)
    except ImportError as exc:
        logger.error(
            "Failed to import GUI registration module %r: %s",
            gui_info_module,
            exc,
        )
        return 1

    gui_info = getattr(mod, "GUI_INFO", None)
    if gui_info is None:
        logger.error("Module %r does not define a GUI_INFO dict", gui_info_module)
        return 1

    return launch_from_gui_info(gui_info)
