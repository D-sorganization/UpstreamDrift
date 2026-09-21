"""Embeddable-tool adapter for the Tour Matching Viewer (Visuals Handoff Step 3).

Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
protocol so the launcher can host the Tour Matching Viewer as a tab or dock widget.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["_TourMatchingViewerEmbedAdapter"]


class _TourMatchingViewerEmbedAdapter:
    """Adapter exposing :class:`TourMatchingViewerWidget` through the embed contract."""

    tool_id: str = "tour_matching_viewer"

    def __init__(self) -> None:
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(900, 650),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        from .gui import TourMatchingViewerWidget

        widget = TourMatchingViewerWidget(parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            try:
                widget.cleanup()
            except Exception:
                logger.exception("tour_matching_viewer widget cleanup raised")

    def is_dirty(self) -> bool:
        return False
