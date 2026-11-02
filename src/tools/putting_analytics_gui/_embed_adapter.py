"""PyQt-free launcher registration for the putting analytics tool."""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities, register_embeddable_tool


class PuttingAnalyticsEmbedAdapter:
    """Create and release the putting analytics widget on demand."""

    tool_id = "putting_analytics"

    def __init__(self) -> None:
        self._widget: Any | None = None

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(900, 640),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        if self._widget is None:
            from src.tools.putting_analytics_gui.gui import MainWidget

            self._widget = MainWidget(parent=parent)
        return self._widget

    def cleanup(self) -> None:
        widget = self._widget
        self._widget = None
        if widget is None:
            return
        if widget.cleanup():
            widget.deleteLater()
            return
        # A study outlived the bounded wait: detach and delete once it ends.
        widget.setParent(None)
        widget.delete_when_idle()

    def is_dirty(self) -> bool:
        return bool(self._widget is not None and self._widget.is_dirty())


register_embeddable_tool(PuttingAnalyticsEmbedAdapter())
