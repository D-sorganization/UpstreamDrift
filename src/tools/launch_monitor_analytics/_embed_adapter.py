"""PyQt6-free launcher embed adapter for Launch Monitor Analytics."""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities, register_embeddable_tool


class LaunchMonitorAnalyticsEmbedAdapter:
    """Host the analytics workbench as a reusable launcher tab."""

    tool_id = "launch_monitor_analytics"

    def __init__(self) -> None:
        self._widget: Any | None = None

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(1100, 700),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        if self._widget is None:
            from src.tools.launch_monitor_analytics.gui import MainWidget

            self._widget = MainWidget(parent=parent)
        return self._widget

    def cleanup(self) -> None:
        widget = self._widget
        self._widget = None
        if widget is not None:
            widget.deleteLater()

    def is_dirty(self) -> bool:
        return bool(self._widget is not None and self._widget.is_dirty())


register_embeddable_tool(LaunchMonitorAnalyticsEmbedAdapter())
