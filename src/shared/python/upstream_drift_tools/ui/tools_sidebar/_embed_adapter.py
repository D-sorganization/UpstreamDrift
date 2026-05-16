"""EmbeddableTool adapter for the UnifiedToolsSidebar (Sidekick).

Wraps the sidebar so the launcher can mount it as a dock.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import (
    EmbedCapabilities,
    register_embeddable_tool,
)
from .api import (
    create_tools_sidebar,
)


class SidekickEmbedAdapter:
    """Adapter for embedding Sidekick (UnifiedToolsSidebar)."""

    tool_id = "sidekick"

    def embed_capabilities(self) -> EmbedCapabilities:
        """Declare that Sidekick prefers a dock and supports embedding."""
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=True,
            min_size=(300, 600),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        """Create the main unified tools sidebar."""
        return create_tools_sidebar(parent=parent)

    def cleanup(self) -> None:
        """Clean up resources if needed."""

    def is_dirty(self) -> bool:
        """Return True if there are unsaved changes."""
        return False


# Register on import
register_embeddable_tool(SidekickEmbedAdapter())
