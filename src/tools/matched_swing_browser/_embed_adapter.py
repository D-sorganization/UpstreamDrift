"""Embeddable-tool adapter for the Matched Swing Browser (MS-80, #10353).

Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
protocol so the launcher can host the Matched Swing Browser as a tab or dock widget.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["_MatchedSwingBrowserEmbedAdapter"]


class _MatchedSwingBrowserEmbedAdapter:
    """Adapter exposing :class:`MatchedSwingBrowserWidget` through the embed contract."""

    tool_id: str = "matched_swing_browser"

    def __init__(self) -> None:
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(1000, 700),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        from .gui import MatchedSwingBrowserWidget

        widget = MatchedSwingBrowserWidget(parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            try:
                widget.cleanup()
            except Exception:
                logger.exception("matched_swing_browser widget cleanup raised")

    def is_dirty(self) -> bool:
        return False
