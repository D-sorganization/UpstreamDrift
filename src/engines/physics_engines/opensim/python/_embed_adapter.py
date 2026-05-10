"""Embeddable-tool adapter for the OpenSim Golf dashboard.

Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
protocol so the launcher can host the OpenSim dashboard as a tab or
dock widget instead of always opening a standalone top-level window.
Constructed at import time by
:mod:`src.engines.physics_engines.opensim.python.__init__`, guarded by
:func:`contextlib.suppress` so headless environments (no PyQt6, no
``opensim`` wheel) can still import the package.

Embeddability notes
-------------------
The dashboard's main surface (:class:`MainWidget`) is a plain
:class:`QWidget` that hosts a :class:`QTabWidget` with a Matplotlib
canvas (``QtAgg`` backend). It does not own a GLFW window, browser
process, or its own ``QApplication``; embedding works without surgery.
The optional Screw Kinematics tab and the OpenSim simulation engine
itself are imported lazily, so the adapter loads cleanly even when
``opensim`` is not installed.

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

__all__ = ["_OpenSimDashboardEmbedAdapter"]


class _OpenSimDashboardEmbedAdapter:
    """Adapter exposing :class:`MainWidget` through the embed contract."""

    tool_id: str = "opensim_golf"

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
        # Lazy import: the GUI module pulls in PyQt6 and Matplotlib's Qt
        # backend. Keeping this import inside ``create_main_widget``
        # lets the adapter (and the import-time registry side-effect)
        # load cleanly in headless contexts where those wheels are
        # absent.
        from .opensim_gui import MainWidget

        widget = MainWidget(parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        """Best-effort: forward cleanup to every widget we handed out.

        Idempotent: hosts may call cleanup more than once during
        shutdown. Forward to every widget but never raise — the host's
        shutdown path must not depend on us.
        """
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            try:
                widget.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("opensim_golf widget cleanup raised")

    def is_dirty(self) -> bool:
        # The OpenSim dashboard does not retain edits between runs;
        # simulation results are recomputed on every "Run Simulation"
        # click and there is no in-memory state we'd want to prompt
        # the user about on close.
        return False


# Register on first import. Guard against re-registration so that
# importing the adapter module from both the package ``__init__`` and
# from a test does not raise ``ValueError: already registered``.
_ADAPTER = _OpenSimDashboardEmbedAdapter()
if get_embeddable_tool(_ADAPTER.tool_id) is None:
    register_embeddable_tool(_ADAPTER)
