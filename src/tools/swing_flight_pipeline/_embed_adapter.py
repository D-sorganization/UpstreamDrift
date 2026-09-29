"""Embeddable-tool adapter for the Swing -> Flight Pipeline.

Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
protocol (ADR-0013) so the launcher resolves this tile through the
process-wide registry instead of the deprecated import-and-probe
fallback on the module-level ``get_dockable_ui`` in ``gui.py``
(issue #8857).
"""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import (
    EmbedCapabilities,
    get_embeddable_tool,
    register_embeddable_tool,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["_SwingFlightEmbedAdapter"]


class _SwingFlightEmbedAdapter:
    """Adapter exposing :class:`SwingFlightWidget` through the embed contract."""

    tool_id: str = "swing_flight_pipeline"

    def __init__(self) -> None:
        # Track every widget we hand out so :meth:`cleanup` can dispose
        # of resources even if the host forgets to delete the widget.
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(1100, 700),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        # Lazy import: the pipeline GUI pulls in PyQt6 and the physics
        # stack. Keeping this inside ``create_main_widget`` lets the
        # adapter (and its registry side-effect) import cleanly in
        # headless contexts.
        from src.tools.swing_flight_pipeline.gui import SwingFlightWidget

        widget = SwingFlightWidget(parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        # Idempotent: hosts may call cleanup more than once during
        # shutdown. Forward to every widget we handed out, but never
        # raise - the host's shutdown path must not depend on us.
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            try:
                widget.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("swing_flight_pipeline widget cleanup raised")

    def is_dirty(self) -> bool:
        # The pipeline renders results into a read-only text pane; there
        # is no implicit save state to prompt about on close.
        return False


# Register on first import. Guard against re-registration so importing
# the adapter from both bootstrap and tests does not raise.
_ADAPTER = _SwingFlightEmbedAdapter()
if get_embeddable_tool(_ADAPTER.tool_id) is None:
    register_embeddable_tool(_ADAPTER)
