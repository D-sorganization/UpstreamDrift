"""Embeddable-tool adapter for the Data Explorer.

Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
protocol so the launcher can host the Data Explorer as a tab or dock
widget instead of always opening it as a standalone window.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities
from src.shared.python.logging_pkg.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = ["_DataExplorerEmbedAdapter"]


class _DataExplorerEmbedAdapter:
    """Adapter exposing :class:`MainWidget` through the embed contract."""

    tool_id: str = "data_explorer"

    def __init__(self) -> None:
        # Track every widget we hand out so :meth:`cleanup` can dispose
        # of any resources even if the host forgets to delete the
        # widget. Adapters live for the process lifetime; the registry
        # holds a single instance.
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        # Tabs work better than docks for table-heavy UIs — the Data
        # Explorer wants horizontal space.
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(640, 480),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        # Lazy import: ``gui`` pulls in PyQt6, and we want this module
        # to import cleanly in headless contexts where PyQt6 may be
        # absent (e.g., docs builds). The launcher only calls
        # ``create_main_widget`` once it has confirmed embedding is
        # supported.
        from .gui import MainWidget

        widget = MainWidget(parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        # Idempotent: hosts may call cleanup more than once during
        # shutdown. We forward to every widget we handed out, but never
        # raise — the host's shutdown path must not depend on us.
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            try:
                widget.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("data_explorer widget cleanup raised")

    def is_dirty(self) -> bool:
        # Read-only inspector of dataset metadata; nothing to save.
        return False
