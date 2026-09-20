"""PyQt6-free launcher embed adapter for Golf Simulator Integration."""

from __future__ import annotations

import logging
from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities, register_embeddable_tool

logger = logging.getLogger(__name__)


class GolfSimulatorEmbedAdapter:
    """Host the Golf Simulator control console as a reusable launcher tab."""

    tool_id = "golf_simulator"
    display_name = "Golf Simulator"

    def __init__(self) -> None:
        self._widget: Any | None = None

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(1000, 650),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any = None) -> Any:
        if self._widget is None:
            try:
                from src.tools.golf_simulator.gui import MainWidget

                self._widget = MainWidget(parent=parent)
            except (ImportError, RuntimeError, AttributeError) as exc:
                logger.warning(
                    "Could not initialize PyQt6 MainWidget for golf_simulator: %s", exc
                )
                from src.tools.golf_simulator.gui import FallbackWidget

                self._widget = FallbackWidget(error_message=str(exc))
        return self._widget

    def cleanup(self) -> None:
        widget = self._widget
        self._widget = None
        if widget is not None:
            if hasattr(widget, "cleanup"):
                widget.cleanup()
            if hasattr(widget, "deleteLater"):
                widget.deleteLater()

    def is_dirty(self) -> bool:
        return bool(
            self._widget is not None
            and getattr(self._widget, "is_dirty", lambda: False)()
        )
