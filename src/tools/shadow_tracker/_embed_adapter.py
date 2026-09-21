"""Launcher embed adapter for Shadow Tracker (ST-11, #10134).

Exposes Shadow Tracker as an EmbeddableTool and BackgroundableTool in the launcher.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import (
    EmbedCapabilities,
    register_embeddable_tool,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["ShadowTrackerAdapter"]


class ShadowTrackerAdapter:
    """Adapter exposing Shadow Tracker to the launcher as an EmbeddableTool."""

    tool_id: str = "shadow_tracker"
    display_name: str = "Shadow Tracker"

    def __init__(self) -> None:
        self._widget: Any | None = None

    def embed_capabilities(self) -> EmbedCapabilities:
        """Return how this tool wants to be embedded."""
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(1024, 720),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any = None) -> Any:
        """Construct and return the top-level review workbench widget."""
        if self._widget is None:
            from .gui import ShadowTrackerWidget

            self._widget = ShadowTrackerWidget(parent=parent)
        return self._widget

    def cleanup(self) -> None:
        """Release resources held by the embedded widget (idempotent)."""
        widget = self._widget
        self._widget = None
        if widget is not None:
            try:
                if hasattr(widget, "cleanup") and callable(widget.cleanup):
                    widget.cleanup()
                if hasattr(widget, "deleteLater") and callable(widget.deleteLater):
                    widget.deleteLater()
            except Exception:  # pragma: no cover - defensive cleanup
                logger.exception("Error during Shadow Tracker widget cleanup")

    def is_dirty(self) -> bool:
        """Return True if the review session has unsaved mask changes."""
        if self._widget is not None and hasattr(self._widget, "is_dirty"):
            return bool(self._widget.is_dirty())
        return False

    # BackgroundableTool protocol implementation
    def pause(self) -> None:
        """Suspend background processing while hidden."""

    def pause_widget(self, widget: Any) -> None:
        """Suspend background processing for a specific widget."""

    def resume(self) -> None:
        """Resume background processing when unhidden."""

    def resume_widget(self, widget: Any) -> None:
        """Resume background processing for a specific widget."""

    def can_background(self) -> bool:
        """Allow the tool to stay loaded in memory when hidden."""
        return True

    def detach_to_window(self) -> bool:
        """Allow the tool to pop out into a dedicated standalone window."""
        return True


# Self-register at module import
_ADAPTER = ShadowTrackerAdapter()
register_embeddable_tool(_ADAPTER)
