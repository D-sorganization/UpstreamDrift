"""Embeddable-tool adapter for the Pinocchio Dashboard.

Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
protocol so the launcher can host the Pinocchio dashboard as a tab or
dock widget instead of always opening a standalone top-level window.
Constructed at import time by
:mod:`src.engines.physics_engines.pinocchio.python.pinocchio_golf.__init__`
and registered with the process-wide registry.

Embeddability notes
-------------------
The Pinocchio dashboard's :class:`PinocchioGUI` is a heavy
``QMainWindow`` built from several mixins (``UISetupMixin``,
``SimulationMixin``, ``PinocchioAnalysisMixin``,
``PinocchioVisualizationMixin``) and ``SimulationGUIBase``. Following
the pattern from PR #5066 (MuJoCo), the existing ``QMainWindow`` is
wrapped as a child widget with ``Qt.WindowType.Widget`` set rather
than refactoring the mixin hierarchy. This keeps every existing tab,
slider, signal, and recorder wiring intact.

Visualization is delegated to a separate Meshcat browser process
(launched on construction); the embedded widget itself is a plain
``QWidget`` and does **not** require its own ``QApplication``.

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

__all__ = ["_PinocchioDashboardEmbedAdapter"]


class _PinocchioDashboardEmbedAdapter:
    """Adapter exposing :class:`MainWidget` through the embed contract."""

    tool_id: str = "pinocchio_golf"

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
        # Lazy import: the GUI module pulls in PyQt6, pinocchio, and
        # several heavy analysis modules. Keeping this import inside
        # ``create_main_widget`` lets the adapter (and the import-time
        # registry side-effect) load cleanly in headless contexts.
        from .gui import MainWidget

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
                logger.exception("pinocchio_golf widget cleanup raised")

    def is_dirty(self) -> bool:
        # The dashboard's recordings live in the engine's recorder and
        # are explicitly exported on user action. There is no in-memory
        # state we'd want to prompt the user about on close.
        return False


# Register at import time. Guarded by a lookup so re-importing this
# module in tests does not raise on duplicate-id registration.
_ADAPTER = _PinocchioDashboardEmbedAdapter()
if get_embeddable_tool(_ADAPTER.tool_id) is None:
    register_embeddable_tool(_ADAPTER)
