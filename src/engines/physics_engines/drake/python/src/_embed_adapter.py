"""Embeddable-tool adapter for the Drake Golf Swing dashboard.

Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
protocol so the launcher can host the Drake dashboard as a tab or dock
widget. Constructed and registered at import time by
:mod:`src.engines.physics_engines.drake.python.src.__init__`, guarded by
:func:`contextlib.suppress` so headless environments (no PyQt6, no
``pydrake`` wheel) can still import the package.

Caveat — Drake's Meshcat visualization opens in an external browser
window on dashboard construction. The Qt controls panel embeds cleanly
inside the launcher tab, but the 3D view itself remains a separate
browser tab. Wrapping Meshcat in a :class:`QWebEngineView` is tracked as
follow-up work.

Part of Subtask 5 / #4998 of EPIC #4993.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import (
    EmbedCapabilities,
    get_embeddable_tool,
    register_embeddable_tool,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["_DrakeDashboardEmbedAdapter"]


class _DrakeDashboardEmbedAdapter:
    """Adapter exposing :class:`MainWidget` through the embed contract."""

    tool_id: str = "drake_golf"

    def __init__(self) -> None:
        # Track every widget we hand out so :meth:`cleanup` can dispose
        # of resources even if the host forgets to delete the widget.
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(1000, 700),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        # Lazy import: the dashboard pulls in PyQt6 + pydrake. Keeping
        # this import inside ``create_main_widget`` lets the adapter
        # itself import cleanly in headless contexts where the Drake
        # wheel is unavailable.
        from .drake_gui_app import MainWidget

        widget = MainWidget(parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        # Idempotent: hosts may call cleanup more than once during
        # shutdown. Forward to every widget we handed out, but never
        # raise — the host's shutdown path must not depend on us.
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            try:
                widget.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("drake_golf widget cleanup raised")

    def is_dirty(self) -> bool:
        # The Drake dashboard records simulation data in-memory but the
        # user-facing flow is "click Export to save" — there is no
        # implicit save state to prompt about on close.
        return False


# Register on first import. Guard against re-registration so that
# importing the adapter module from both the package ``__init__`` and
# from a test does not raise ``ValueError: already registered``.
_ADAPTER = _DrakeDashboardEmbedAdapter()
if get_embeddable_tool(_ADAPTER.tool_id) is None:
    register_embeddable_tool(_ADAPTER)
