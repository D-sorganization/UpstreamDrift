"""Embeddable-tool adapter for the Training Controller."""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["_TrainingControllerEmbedAdapter"]


class _TrainingControllerEmbedAdapter:
    """Expose the PyQt Training Controller through the launcher embed API."""

    tool_id = "training_controller"

    def __init__(self) -> None:
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(1024, 720),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        from .gui import MainWidget, build_default_controller

        widget = MainWidget(build_default_controller(), parent=parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            cleanup = getattr(widget, "cleanup", None)
            if not callable(cleanup):
                continue
            try:
                cleanup()
            except (RuntimeError, ValueError, TypeError, OSError):
                logger.exception("training_controller widget cleanup raised")

    def is_dirty(self) -> bool:
        return False
