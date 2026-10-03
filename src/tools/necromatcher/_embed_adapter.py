"""Lazy native registration; headless discovery does not import Qt."""

from __future__ import annotations
from typing import Any
from src.shared.python.launcher_embed import EmbedCapabilities, register_embeddable_tool


class NecromatcherAdapter:
    tool_id = "necromatcher"

    def __init__(self) -> None:
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(900, 650),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        from .gui import NecromatcherWidget

        widget = NecromatcherWidget(parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        for widget in self._widgets:
            widget.cleanup()
        self._widgets.clear()

    def is_dirty(self) -> bool:
        return False


register_embeddable_tool(NecromatcherAdapter())
