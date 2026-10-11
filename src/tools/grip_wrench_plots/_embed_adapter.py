"""Embeddable-tool adapter for the Grip Wrench Plots tool (GCV-10, #11716).

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

__all__ = ["GripWrenchPlotsEmbedAdapter"]


class GripWrenchPlotsEmbedAdapter:
    """Expose :class:`GripWrenchPlotWidget` through the embed contract."""

    tool_id: str = "grip_wrench_plots"

    def __init__(self) -> None:
        self._widget: Any | None = None

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=True,
            min_size=(420, 560),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        from .gui import GripWrenchPlotWidget

        if self._widget is None:
            self._widget = GripWrenchPlotWidget(parent)
        return self._widget

    def cleanup(self) -> None:
        widget, self._widget = self._widget, None
        release_widget(widget)

    def is_dirty(self) -> bool:
        return False


register_embeddable_tool(GripWrenchPlotsEmbedAdapter())
