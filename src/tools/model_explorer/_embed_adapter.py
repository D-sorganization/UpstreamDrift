"""Embeddable-tool adapter for the Model Explorer.

Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
protocol so the launcher can host the Model Explorer as a tab or dock
widget. Constructed at import time by
:mod:`src.tools.model_explorer.__init__` and registered with the
process-wide registry.

Part of Subtask 5 / #4998 of EPIC #4993.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import (
    EmbedCapabilities,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["_ModelExplorerEmbedAdapter"]


class _ModelExplorerEmbedAdapter:
    """Adapter exposing :class:`MainWidget` through the embed contract."""

    tool_id: str = "model_explorer"

    def __init__(self) -> None:
        # Track every widget we hand out so :meth:`cleanup` can dispose
        # of resources even if the host forgets to delete the widget.
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(700, 500),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        # Lazy import: the GUI module pulls in PyQt6, and we want the
        # registry / adapter to import cleanly in headless contexts
        # (CI, docs builds) where PyQt6 may be unavailable until a test
        # fixture installs ``QT_QPA_PLATFORM=offscreen``.
        from .gui import MainWidget

        widget = MainWidget(parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        # Idempotent: hosts may call cleanup more than once during
        # shutdown. Forward to every widget we handed out, but never
        # raise — the host's shutdown path must not depend on us.
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            try:
                widget.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("model_explorer widget cleanup raised")

    def is_dirty(self) -> bool:
        # The Model Explorer tracks a single in-memory URDFBuilder. If
        # any of the live widgets has unsaved changes, treat the tool
        # as dirty so the launcher can prompt before tearing down the
        # tab. Defensive: if probing a widget raises (e.g. it has been
        # deleted by Qt) treat it as clean.
        for widget in list(self._widgets):
            try:
                if widget.has_unsaved_changes():
                    return True
            except Exception:  # pragma: no cover - defensive  # noqa: BLE001
                logger.debug(
                    "model_explorer is_dirty: widget probe raised", exc_info=True
                )
        return False
