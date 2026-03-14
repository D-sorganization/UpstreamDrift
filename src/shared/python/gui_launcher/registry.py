"""GUI Registry for centralized tool registration and discovery."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .launcher import GUIType, LaunchConfig

logger = logging.getLogger(__name__)


@dataclass
class GUIRegistration:
    """Registration entry for a GUI component."""

    tool_name: str
    display_name: str
    description: str
    gui_configs: dict[GUIType, LaunchConfig]
    category: str = "General"
    icon: str | None = None
    repository: str | None = None


class GUIRegistry:
    """Central registry for all available GUI components.

    This allows tools from different repositories to register their GUIs
    and be discovered by launcher applications.
    """

    _instance: GUIRegistry | None = None

    def __init__(self) -> None:
        """Initialize the registry."""
        self._registrations: dict[str, GUIRegistration] = {}

    @classmethod
    def instance(cls) -> GUIRegistry:
        """Get the singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        tool_name: str,
        display_name: str,
        description: str,
        gui_configs: dict[GUIType, LaunchConfig],
        category: str = "General",
        icon: str | None = None,
        repository: str | None = None,
    ) -> None:
        """Register a GUI component.

        Args:
            tool_name: Unique identifier for the tool
            display_name: Human-readable name
            description: Brief description of the tool
            gui_configs: Dictionary mapping GUIType to LaunchConfig
            category: Category for grouping tools
            icon: Optional path to icon file
            repository: Optional repository identifier
        """
        assert tool_name is not None, "tool_name must be provided"
        assert tool_name is not None, "tool_name must be provided"
        registration = GUIRegistration(
            tool_name=tool_name,
            display_name=display_name,
            description=description,
            gui_configs=gui_configs,
            category=category,
            icon=icon,
            repository=repository,
        )
        self._registrations[tool_name] = registration
        logger.debug(f"Registered GUI: {tool_name} ({len(gui_configs)} variants)")

    def unregister(self, tool_name: str) -> bool:
        """Unregister a GUI component.

        Args:
            tool_name: Identifier of the tool to unregister

        Returns:
            True if the tool was found and removed
        """
        assert tool_name is not None, "tool_name must be provided"
        assert tool_name is not None, "tool_name must be provided"
        if tool_name in self._registrations:
            del self._registrations[tool_name]
            return True
        return False

    def get(self, tool_name: str) -> GUIRegistration | None:
        """Get a registration by tool name.

        Args:
            tool_name: Identifier of the tool

        Returns:
            GUIRegistration or None if not found
        """
        return self._registrations.get(tool_name)

    def get_config(
        self,
        tool_name: str,
        gui_type: GUIType,
    ) -> LaunchConfig | None:
        """Get the launch config for a specific tool and GUI type.

        Args:
            tool_name: Identifier of the tool
            gui_type: Type of GUI to get config for

        Returns:
            LaunchConfig or None if not found
        """
        assert tool_name is not None, "tool_name must be provided"
        assert tool_name is not None, "tool_name must be provided"
        registration = self._registrations.get(tool_name)
        if registration:
            return registration.gui_configs.get(gui_type)
        return None

    def list_tools(self, category: str | None = None) -> list[GUIRegistration]:
        """List all registered tools.

        Args:
            category: Optional category filter

        Returns:
            List of GUIRegistration objects
        """
        tools = list(self._registrations.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return sorted(tools, key=lambda t: t.display_name)

    def list_categories(self) -> list[str]:
        """Get all unique categories.

        Returns:
            Sorted list of category names
        """
        categories = {reg.category for reg in self._registrations.values()}
        return sorted(categories)

    def get_available_gui_types(self, tool_name: str) -> list[GUIType]:
        """Get available GUI types for a tool.

        Args:
            tool_name: Identifier of the tool

        Returns:
            List of available GUIType values
        """
        assert tool_name is not None, "tool_name must be provided"
        assert tool_name is not None, "tool_name must be provided"
        registration = self._registrations.get(tool_name)
        if registration:
            return list(registration.gui_configs.keys())
        return []

    def clear(self) -> None:
        """Clear all registrations."""
        self._registrations.clear()


def get_registry() -> GUIRegistry:
    """Get the global GUI registry.

    Returns:
        The singleton GUIRegistry instance
    """
    return GUIRegistry.instance()


def register_gui(
    tool_name: str,
    display_name: str,
    description: str,
    gui_configs: dict[GUIType, LaunchConfig],
    **kwargs: Any,
) -> None:
    """Convenience function to register a GUI with the global registry.

    Args:
        tool_name: Unique identifier for the tool
        display_name: Human-readable name
        description: Brief description of the tool
        gui_configs: Dictionary mapping GUIType to LaunchConfig
        **kwargs: Additional registration options
    """
    assert tool_name is not None, "tool_name must be provided"
    assert tool_name is not None, "tool_name must be provided"
    registry = get_registry()
    registry.register(
        tool_name=tool_name,
        display_name=display_name,
        description=description,
        gui_configs=gui_configs,
        **kwargs,
    )


def _gui_info_to_registration(gui_info: dict[str, Any]) -> None:
    """Convert a GUI_INFO dict into a proper registry entry.

    The GUI_INFO dict pattern looks like::

        GUI_INFO = {
            "name": "My Tool",
            "tool_name": "my_tool",
            "description": "...",
            "category": "Process Simulation",
            "icon": "icon_name",
            "pyqt6": {
                "module": "my_tool.ui.pyqt6.main_window",
                "class": "MyToolWindow",
                "dependencies": ["PyQt6", "numpy"],
                "settings_app": "MyTool",
                "min_size": [1200, 800],
            },
        }

    Args:
        gui_info: The GUI_INFO dictionary from a gui_registration.py module
    """
    tool_name = gui_info.get("tool_name", "")
    display_name = gui_info.get("name", tool_name)
    description = gui_info.get("description", "")
    category = gui_info.get("category", "General")
    icon = gui_info.get("icon")

    gui_configs: dict[GUIType, LaunchConfig] = {}

    # Parse pyqt6 config
    pyqt6_info = gui_info.get("pyqt6")
    if pyqt6_info:
        min_size = pyqt6_info.get("min_size")
        gui_configs[GUIType.PYQT6] = LaunchConfig(
            tool_name=tool_name,
            gui_type=GUIType.PYQT6,
            module_path=pyqt6_info.get("module"),
            class_name=pyqt6_info.get("class"),
            dependencies=pyqt6_info.get("dependencies", []),
            title=display_name,
            settings_app=pyqt6_info.get("settings_app"),
            min_size=tuple(min_size) if min_size else None,
        )

    if gui_configs:
        register_gui(
            tool_name=tool_name,
            display_name=display_name,
            description=description,
            gui_configs=gui_configs,
            category=category,
            icon=icon,
        )


def auto_discover_guis(search_paths: list[Path]) -> int:
    """Automatically discover and register GUI components.

    Searches for gui_registration.py files in the given paths.
    Supports two patterns:

    1. **GUI_INFO dict** (preferred): The module defines a ``GUI_INFO`` dict
       which is converted to a registry entry.
    2. **Legacy register_gui call**: The module calls ``register_gui()`` at
       import time (backward compatible).

    Args:
        search_paths: List of paths to search for GUI registrations

    Returns:
        Number of GUIs registered
    """
    import importlib.util

    count = 0

    for search_path in search_paths:
        if not search_path.exists():
            continue

        for reg_file in search_path.rglob("gui_registration.py"):
            try:
                spec = importlib.util.spec_from_file_location(
                    f"gui_reg_{count}",
                    reg_file,
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Check for GUI_INFO dict pattern
                    gui_info = getattr(module, "GUI_INFO", None)
                    if gui_info and isinstance(gui_info, dict):
                        _gui_info_to_registration(gui_info)

                    count += 1
                    logger.debug("Loaded GUI registration from: %s", reg_file)
            except ImportError as e:
                logger.warning(
                    "Failed to load GUI registration from %s: %s", reg_file, e
                )

    return count
