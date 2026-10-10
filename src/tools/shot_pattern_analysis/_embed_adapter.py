"""PyQt-free launcher registration for shot-pattern analysis."""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities, register_embeddable_tool


class ShotPatternAnalysisEmbedAdapter:
    """Create and release the optional analysis widget on demand."""

    tool_id = "shot_pattern_analysis"

    def __init__(self) -> None:
        self._widget: Any | None = None

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(1000, 760),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        if self._widget is None:
            from src.tools.shot_pattern_analysis.gui import MainWidget

            self._widget = MainWidget(parent=parent)
        return self._widget

    def cleanup(self) -> None:
        widget = self._widget
        self._widget = None
        if widget is None:
            return
        stopped = widget.cleanup()
        if stopped:
            widget.deleteLater()
            return
        # Keep the QObject alive if bounded shutdown expires. The active
        # action's terminal signal schedules deletion after its thread exits.
        widget.setParent(None)
        widget.delete_when_idle()

    def is_dirty(self) -> bool:
        return bool(self._widget is not None and self._widget.is_dirty())


register_embeddable_tool(ShotPatternAnalysisEmbedAdapter())
