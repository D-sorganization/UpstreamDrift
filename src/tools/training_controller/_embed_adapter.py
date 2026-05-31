"""Adapter to expose the Training Controller as an EmbeddableTool.

In addition to the base :class:`EmbeddableTool` surface, this adapter
implements the optional :class:`BackgroundableTool` hooks introduced in
Sub-PR A (#6013): ``can_background``, ``detach_to_window``, ``pause``,
and ``resume``. The host resolves these structurally via ``getattr`` so
backgrounding a training tab keeps the scheduler running while the
widget is hidden, and re-surfacing it re-binds the widget to live
status + realtime progress.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities


class _TrainingControllerEmbedAdapter:
    """Implements the EmbeddableTool + BackgroundableTool protocols."""

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
        """Create and return the tool's main widget."""
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

    # --------------------------------------------------- BackgroundableTool

    def can_background(self) -> bool:
        """The scheduler runs independently of the GUI, so backgrounding is safe."""
        return True

    def detach_to_window(self) -> bool:
        """The dashboard may be popped out into its own window."""
        return True

    def pause(self) -> None:
        """No-op: training keeps running while the tab is hidden.

        The backend :class:`Scheduler` and any in-flight runs are
        unaffected by the GUI being backgrounded, so there is nothing to
        suspend. The widget detaches its live realtime subscriptions
        opportunistically (see :meth:`MainWidget.pause`) to stay cheap
        while hidden, but the authoritative job state lives in the
        scheduler registry, not the widget.
        """
        for widget in self._widgets:
            pause = getattr(widget, "pause", None)
            if callable(pause):
                pause()

    def resume(self) -> None:
        """Re-bind every live widget to scheduler status + realtime progress."""
        for widget in self._widgets:
            resume = getattr(widget, "resume", None)
            if callable(resume):
                resume()


# Alias for backward-compatibility
TrainingControllerAdapter = _TrainingControllerEmbedAdapter
