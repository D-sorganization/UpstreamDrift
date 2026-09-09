"""Embed adapter for the Capture Rig launcher tile (ADR-0013)."""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities


class CaptureRigAdapter:
    """Implements the ``EmbeddableTool`` protocol for the launcher."""

    tool_id = "capture_rig"

    def __init__(self) -> None:
        self._widget: Any = None

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(1100, 700),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        """The tile widget, themed like the standalone window (#9816)."""
        from . import styling
        from .gui import CaptureRigWidget

        self._widget = CaptureRigWidget(parent=parent)
        styling.apply_theme(self._widget)
        return self._widget

    def cleanup(self) -> None:
        if self._widget is not None:
            self._widget.shutdown()
        self._widget = None

    def is_dirty(self) -> bool:
        """A running rig command must not be closed away silently."""
        return self._widget is not None and self._widget.busy
