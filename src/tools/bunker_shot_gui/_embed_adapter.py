"""Adapter to expose the Bunker Shot Gui as an EmbeddableTool."""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities


class BunkerShotGuiAdapter:
    """Implements the EmbeddableTool protocol for the launcher."""

    tool_id = "bunker_shot_gui"
    display_name = "Bunker Shot Gui"

    def embed_capabilities(self) -> EmbedCapabilities:
        """Return how this tool wants to be embedded."""
        return EmbedCapabilities()

    def create_main_widget(self, parent: Any) -> Any:
        """Create and return the tool's main window."""
        from .gui import get_dockable_ui

        return get_dockable_ui(parent) if parent else get_dockable_ui()

    def cleanup(self) -> None:
        """Clean up resources when the tool is unloaded."""

    def is_dirty(self) -> bool:
        """Return True if the tool has unsaved changes."""
        return False
