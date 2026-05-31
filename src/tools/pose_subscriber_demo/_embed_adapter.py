"""Embeddable-tool adapter for the Pose Subscriber demo.

Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
protocol so the launcher can host the demo as a tab or dock widget.
"""

# background: yes (pause/resume); cleanup idempotent (swap-then-clear). This
# tool holds a live ``pose/canonical`` realtime subscription — a scarce-ish
# resource. ``pause`` releases the subscription so a hidden subscriber stops
# consuming traffic; ``resume`` re-acquires it on re-surface (#6013).

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["_PoseSubscriberDemoEmbedAdapter"]


class _PoseSubscriberDemoEmbedAdapter:
    """Adapter exposing the demo's MainWidget through the embed contract."""

    tool_id: str = "pose_subscriber_demo"

    def __init__(self) -> None:
        # Track every widget we hand out so :meth:`cleanup` can dispose
        # of subscriptions even if the host forgets to delete the
        # widget. Adapters live for the process lifetime; the registry
        # holds a single instance.
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=True,
            min_size=(480, 480),
        )

    def create_main_widget(self, parent: Any) -> Any:
        # Lazy import: the demo widget pulls in PyQt6 + matplotlib, and
        # we want the registry / adapter to import cleanly in headless
        # contexts (CI, docs builds).
        from .gui import MainWidget

        widget = MainWidget(parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        # Idempotent: hosts may call cleanup more than once during
        # shutdown. We forward to every widget we handed out, but never
        # raise — the host's shutdown path must not depend on us.
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            try:
                widget.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("pose_subscriber_demo widget cleanup raised")

    def is_dirty(self) -> bool:
        # Read-only mirror of Pose Studio's pose; nothing to save.
        return False

    def pause_widget(self, widget: Any) -> None:
        """Pause only the widget instance being backgrounded."""

        self._call_widget_lifecycle(widget, "pause")

    def resume_widget(self, widget: Any) -> None:
        """Resume only the widget instance being re-surfaced."""

        self._call_widget_lifecycle(widget, "resume")

    def pause(self) -> None:
        # Backgrounding hook (#6013): release each live widget's
        # ``pose/canonical`` subscription so a hidden subscriber stops
        # consuming realtime traffic. Idempotent; never raises so the
        # host's background path cannot fail on us.
        for widget in list(self._widgets):
            self.pause_widget(widget)

    def resume(self) -> None:
        # Inverse of :meth:`pause`: re-acquire each widget's
        # subscription when the tab is re-surfaced. Idempotent.
        for widget in list(self._widgets):
            self.resume_widget(widget)

    @staticmethod
    def _call_widget_lifecycle(widget: Any, method_name: str) -> None:
        method = getattr(widget, method_name, None)
        if not callable(method):
            return
        try:
            method()
        except Exception:  # pragma: no cover - defensive
            logger.exception("pose_subscriber_demo widget %s raised", method_name)
