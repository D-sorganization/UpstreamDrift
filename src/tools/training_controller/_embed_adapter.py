"""Adapter to expose the Training Controller as an EmbeddableTool."""

# background: yes (defaults); cleanup idempotent (swap-then-clear). The widget
# observes a live training dashboard; keeping it running while hidden preserves
# that subscription. No scarce GPU context held at the adapter level, so
# structural defaults apply (#6013).

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.shared.python.launcher_embed import EmbedCapabilities
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class _TrainingControllerEmbedAdapter:
    """Implements the EmbeddableTool protocol for the launcher."""

    tool_id = "training_controller"
    display_name = "Training Controller"

    def __init__(self) -> None:
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        """Return how this tool wants to be embedded."""
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(1024, 720),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any = None) -> Any:
        """Create and return the tool's main window."""
        from .gui import MainWidget, build_default_controller

        controller = build_default_controller()
        widget = MainWidget(controller, parent=parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        """Clean up resources when the tool is unloaded.

        Idempotent: drop the widget references *first* (swap-then-clear)
        so a second call — or a raise mid-teardown — never re-cleans a
        widget. Never raises; the host's shutdown path must not depend
        on us.
        """
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            cleanup = getattr(widget, "cleanup", None)
            if not callable(cleanup):
                continue
            try:
                cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("training_controller widget cleanup raised")

    def is_dirty(self) -> bool:
        """Return True if the tool has unsaved state."""
        return False


# Alias for backward-compatibility
TrainingControllerAdapter = _TrainingControllerEmbedAdapter
