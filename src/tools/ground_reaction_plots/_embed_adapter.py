"""Embeddable-tool adapter for the Ground Reaction Plots tool (GCV-5, #11711).

PyQt6 is imported lazily inside :meth:`create_main_widget` so headless CI can
introspect the adapter.  ``cleanup`` is idempotent.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import (
    EmbedCapabilities,
    register_embeddable_tool,
    release_widget,
)

__all__ = ["GroundReactionPlotsEmbedAdapter"]


class GroundReactionPlotsEmbedAdapter:
    """Expose :class:`GroundReactionPlotWidget` through the embed contract."""

    tool_id: str = "ground_reaction_plots"

    def __init__(self) -> None:
        self._widget: Any | None = None

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=True,
            min_size=(640, 560),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        from .gui import GroundReactionPlotWidget

        if self._widget is None:
            self._widget = GroundReactionPlotWidget(parent)
        return self._widget

    def cleanup(self) -> None:
        widget, self._widget = self._widget, None
        release_widget(widget)

    def is_dirty(self) -> bool:
        return False


register_embeddable_tool(GroundReactionPlotsEmbedAdapter())
