"""Embeddable-tool adapter for the Sidekick AI assistant.

Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
protocol so the launcher can host the existing
:class:`~src.shared.python.ai.gui.assistant_panel.AIAssistantPanel` as
a tab or dock widget — the same way as ``model_explorer`` or
``starting_pose_matcher``. See issue #5460.

The chat panel is a side-panel-shaped GUI, so the default capabilities
favour a dock placement with a modest minimum size. Cleanup is a no-op
beyond dropping references: the panel manages its own session
persistence and there is no save state to flush.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["_SidekickEmbedAdapter"]


class _SidekickEmbedAdapter:
    """Adapter exposing :class:`AIAssistantPanel` through the embed contract."""

    tool_id: str = "sidekick"

    def __init__(self) -> None:
        # Track every widget we hand out so :meth:`cleanup` can release
        # references even if the host forgets to delete the widget.
        # Adapters live for the process lifetime; the registry holds a
        # single instance.
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        # The chat panel is shaped like a side dock by design (vertical
        # scroll of messages plus a compose row at the bottom). A dock
        # placement matches the hard-wired splitter slot the panel
        # currently lives in. The minimum size leaves enough room for
        # readable conversation history without crushing the input.
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=True,
            min_size=(360, 480),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        # Lazy import: ``assistant_panel`` pulls in PyQt6 and the AI
        # session manager, and we want the registry / adapter to import
        # cleanly in headless contexts (CI, docs builds) where PyQt6 may
        # be unavailable until a fixture installs
        # ``QT_QPA_PLATFORM=offscreen``. Surface ``ImportError`` to the
        # bootstrap so it can warn-and-skip per its existing contract.
        from src.shared.python.ai.gui.assistant_panel import AIAssistantPanel

        widget = AIAssistantPanel(parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        # Idempotent: hosts may call cleanup more than once during
        # shutdown. The chat panel persists sessions on its own; we
        # only need to drop our references. Never raise — the host's
        # shutdown path must not depend on us.
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            try:
                # ``AIAssistantPanel`` does not currently expose a
                # ``cleanup`` hook; deleteLater (if available) is the
                # canonical Qt teardown. Probe defensively so this
                # adapter remains usable against test doubles too.
                deleter = getattr(widget, "deleteLater", None)
                if callable(deleter):
                    deleter()
            except Exception:  # pragma: no cover - defensive
                logger.exception("sidekick widget cleanup raised")

    def is_dirty(self) -> bool:
        # Chat history is auto-persisted via ``ChatSessionManager``;
        # there is no in-memory dirty buffer the host needs to prompt
        # the user about before close.
        return False
