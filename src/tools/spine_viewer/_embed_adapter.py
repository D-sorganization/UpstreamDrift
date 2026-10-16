"""Embeddable-tool adapter for the Spine Viewer (SPN-15b, #12772).

PyQt6 is imported lazily inside :meth:`create_main_widget` so headless CI can
introspect the adapter. ``cleanup`` is idempotent (swap-then-clear).
"""

# background: yes (defaults); cleanup idempotent. The viewer holds only a
# read-only hash-verified study and one figure, so structural defaults apply.

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import (
    EmbedCapabilities,
    register_embeddable_tool,
    release_widget,
)

__all__ = ["SpineViewerEmbedAdapter"]


class SpineViewerEmbedAdapter:
    """Expose :class:`SpineViewerWidget` through the embed contract."""

    tool_id: str = "spine_viewer"

    def __init__(self) -> None:
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(720, 520),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        from .gui import SpineViewerWidget

        widget = SpineViewerWidget(parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            widget.cleanup()
            release_widget(widget)

    def is_dirty(self) -> bool:
        return False


register_embeddable_tool(SpineViewerEmbedAdapter())
