"""Adapter to expose the Terrain Engine as an EmbeddableTool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QWidget

from src.shared.python.launcher_embed import EmbedCapabilities

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow


class TerrainEngineAdapter:
    """Implements the EmbeddableTool protocol for the launcher."""

    tool_id = "terrain_engine"
    display_name = "Terrain Engine"
    capabilities = EmbedCapabilities.NONE

    def create_widget(self, parent: QWidget | None = None) -> QMainWindow:
        """Create and return the tool's main window."""
        from .gui import get_dockable_ui

        return get_dockable_ui(parent) if parent else get_dockable_ui()

    def teardown(self) -> None:
        """Clean up resources when the tool is unloaded."""
