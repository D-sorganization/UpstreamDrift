"""Adapter to expose the Golf Simulation Suite as an EmbeddableTool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


from src.shared.python.launcher_embed import EmbedCapabilities

if TYPE_CHECKING:
    pass


class GolfSimulationSuiteAdapter:
    """Implements the EmbeddableTool protocol for the launcher."""

    tool_id = "golf_simulation_suite"
    display_name = "Golf Simulation Suite"

    def embed_capabilities(self) -> EmbedCapabilities:
        """Return how this tool wants to be embedded."""
        return EmbedCapabilities.NONE

    def create_main_widget(self, parent: Any = None) -> Any:
        """Create and return the tool's main window."""
        from .gui import get_dockable_ui

        return get_dockable_ui(parent) if parent else get_dockable_ui()

    def cleanup(self) -> None:
        """Clean up resources when the tool is unloaded."""

    def is_dirty(self) -> bool:
        """Return True if the tool has unsaved state."""
        return False
