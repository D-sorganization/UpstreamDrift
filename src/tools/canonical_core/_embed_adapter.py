"""Embeddable adapters for canonical-core tool shell entries."""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import (
    EmbedCapabilities,
    register_embeddable_tool,
)
from src.tools.canonical_core.registry import (
    CanonicalCoreTool,
    canonical_core_tools,
)

_MIN_SIZE = (760, 520)


class CanonicalCoreToolEmbedAdapter:
    """Expose a canonical-core descriptor through ``EmbeddableTool``."""

    def __init__(self, descriptor: CanonicalCoreTool) -> None:
        self._descriptor = descriptor
        self.tool_id = descriptor.tool_id
        self._widget: Any | None = None

    @property
    def descriptor(self) -> CanonicalCoreTool:
        """Return immutable shell metadata for this adapter."""
        return self._descriptor

    def embed_capabilities(self) -> EmbedCapabilities:
        """Return how the canonical-core shell should be embedded."""
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=_MIN_SIZE,
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        """Construct the PyQt6 shell lazily.

        The actual CC analysis implementations remain behind service/API
        boundaries. This widget is only the shell entry point, so importing the
        adapter never imports engines or PyQt6.
        """
        from src.tools.canonical_core.pyqt_shell import CanonicalCoreShellWidget

        if self._widget is None:
            self._widget = CanonicalCoreShellWidget(self._descriptor, parent=parent)
        return self._widget

    def cleanup(self) -> None:
        """Release the current widget reference; safe to call repeatedly."""
        widget, self._widget = self._widget, None
        if widget is None:
            return
        delete_later = getattr(widget, "deleteLater", None)
        if callable(delete_later):
            delete_later()

    def is_dirty(self) -> bool:
        """Canonical-core launch shells do not hold unsaved local state."""
        return False


for _tool in canonical_core_tools():
    register_embeddable_tool(CanonicalCoreToolEmbedAdapter(_tool))
