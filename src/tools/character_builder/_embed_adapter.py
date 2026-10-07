"""Embeddable-tool adapter for the Character Builder (AGENTS.md section D)."""

# background: yes (defaults); cleanup idempotent (drop widget reference first).
# CPU-only widget with no scarce resources, so structural defaults apply (#6013).

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities

__all__ = ["CharacterBuilderAdapter"]


class CharacterBuilderAdapter:
    """Implements the ``EmbeddableTool`` protocol; Qt is imported lazily."""

    tool_id = "character_builder"
    display_name = "Character Builder"

    def __init__(self) -> None:
        self._widget: Any | None = None

    def embed_capabilities(self) -> EmbedCapabilities:
        """Embeddable as a tab; needs room for the form and the summary."""
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(720, 520),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any = None) -> Any:
        """Create the builder widget (imports PyQt6 on first use)."""
        from .gui import CharacterBuilderWidget

        self._widget = CharacterBuilderWidget(parent)
        return self._widget

    def cleanup(self) -> None:
        """Release the widget. Idempotent."""
        widget, self._widget = self._widget, None
        if widget is not None:
            widget.cleanup()

    def is_dirty(self) -> bool:
        """The builder holds no unsaved documents; exports are explicit."""
        return False
