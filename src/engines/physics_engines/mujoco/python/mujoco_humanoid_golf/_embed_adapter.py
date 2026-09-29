"""Embeddable-tool adapter for the MuJoCo Analysis Dashboard.

Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
protocol so the launcher can host the MuJoCo dashboard as a tab or
dock widget instead of always opening a standalone top-level window.
Constructed at import time by
:mod:`src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.__init__`
and registered with the process-wide registry.

Embeddability notes
-------------------
The dashboard's viewport (:class:`MuJoCoSimWidget`) renders via the
offscreen ``mujoco.Renderer`` and blits frames into a ``QLabel``. It is
a plain ``QWidget`` and does **not** rely on a GLFW window or its own
``QApplication`` — embedding works without surgery. The adapter wraps
the existing :class:`AdvancedGolfAnalysisWindow` ``QMainWindow`` inside
:class:`MainWidget` with ``Qt.WindowType.Widget`` flags so its tabs,
mixin, and signal wiring continue to function unchanged.

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

__all__ = ["_MujocoDashboardEmbedAdapter"]


class _MujocoDashboardEmbedAdapter:
    """Adapter exposing :class:`MainWidget` through the embed contract."""

    tool_id: str = "mujoco_unified"

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
        # Lazy import: the GUI module pulls in PyQt6, mujoco, and
        # several heavy analysis modules. Keeping this import inside
        # ``create_main_widget`` lets the adapter (and the import-time
        # registry side-effect) load cleanly in headless contexts.
        from .gui.core.main_widget import MainWidget

        widget = MainWidget(parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        """Best-effort: stop simulation timers held by every widget.

        Idempotent: hosts may call cleanup more than once during
        shutdown. Forwards to every widget we handed out, but never
        raises — the host's shutdown path must not depend on us.
        """
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            try:
                widget.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("mujoco_unified widget cleanup raised")

    def is_dirty(self) -> bool:
        # The dashboard's recordings live in the launcher's recorder
        # and are explicitly exported on user action. There is no
        # in-memory state we'd want to prompt the user about on close.
        return False


# Register at import time. Guarded by a lookup so re-importing this
# module in tests does not raise on duplicate-id registration.
_ADAPTER = _MujocoDashboardEmbedAdapter()
if get_embeddable_tool(_ADAPTER.tool_id) is None:
    register_embeddable_tool(_ADAPTER)
