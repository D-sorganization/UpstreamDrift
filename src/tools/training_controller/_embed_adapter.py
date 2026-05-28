"""Adapter to expose the Training Controller as an EmbeddableTool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


from src.shared.python.launcher_embed import EmbedCapabilities

if TYPE_CHECKING:
    pass


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
        """Clean up resources when the tool is unloaded."""
        for widget in self._widgets:
            if hasattr(widget, "cleanup") and callable(widget.cleanup):
                widget.cleanup()
        self._widgets = []

    def is_dirty(self) -> bool:
        """Return True if the tool has unsaved state."""
        return False


# Alias for backward-compatibility
TrainingControllerAdapter = _TrainingControllerEmbedAdapter
