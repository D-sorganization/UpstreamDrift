"""Embeddable widget shell for the MuJoCo dashboard.

Subtask 5 / #4998 of EPIC #4993 refactors the MuJoCo Analysis Dashboard
so it can launch as either a standalone :class:`QMainWindow` *or* an
embedded ``QWidget`` inside the launcher host (tab / dock).

The historical entry point — :class:`AdvancedGolfAnalysisWindow` in
:mod:`gui.core.main_window` — still exists for back-compat, but its
contents are also exposed through :class:`MainWidget` defined here.
Embeddable hosts construct :class:`MainWidget` directly and never see
the top-level ``QMainWindow`` shell.

The dashboard is a heavy ``QMainWindow`` (custom status bar, central
``QTabWidget``, dockable AI chat panel). To reuse it inside a plain
``QWidget`` host without rewriting all of that machinery, this module
embeds an :class:`AdvancedGolfAnalysisWindow` instance as a child with
``Qt.WindowType.Widget`` flags. This is the idiomatic way to reuse a
``QMainWindow`` in a non-top-level context (see Qt docs for ``QMainWindow``
"Creating Main Window Components") and it keeps every existing tab,
mixin, and signal wiring intact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QWidget

from src.shared.python.logging_pkg.logging_config import get_logger

from .main_window import AdvancedGolfAnalysisWindow

if TYPE_CHECKING:  # pragma: no cover - hint only
    from ...sim_widget import MuJoCoSimWidget

logger = get_logger(__name__)


__all__ = ["MainWidget"]


class MainWidget(QWidget):
    """Embeddable MuJoCo Dashboard widget.

    Wraps an :class:`AdvancedGolfAnalysisWindow` instance with
    ``Qt.WindowType.Widget`` flags so the launcher can host the
    dashboard as a tab / dock without spawning a top-level window.

    The wrapped window is exposed as :attr:`inner_main_window` for
    callers that need access to the dashboard's tabs, sim widget, or
    status bar (e.g. tests).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Build the dashboard as a child window. ``Qt.WindowType.Widget``
        # tells Qt to treat this ``QMainWindow`` as a regular child
        # widget rather than a top-level window — that way it lays out
        # cleanly inside our ``QHBoxLayout`` instead of popping into
        # its own OS window.
        self._inner: AdvancedGolfAnalysisWindow = AdvancedGolfAnalysisWindow()
        self._inner.setWindowFlags(Qt.WindowType.Widget)
        self._inner.setParent(self)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._inner)

        logger.info("MuJoCo MainWidget initialized")

    # ---- accessors -----------------------------------------------------

    @property
    def inner_main_window(self) -> AdvancedGolfAnalysisWindow:
        """Return the wrapped :class:`AdvancedGolfAnalysisWindow`."""
        return self._inner

    @property
    def sim_widget(self) -> MuJoCoSimWidget:
        """Convenience accessor — the simulation viewport widget."""
        return self._inner.sim_widget

    # ---- lifecycle -----------------------------------------------------

    def cleanup(self) -> None:
        """Best-effort teardown of simulation timers and threads.

        Stops the periodic status timer and the simulation step timer
        (owned by :class:`MuJoCoSimWidget`) so the host process does
        not keep firing Qt timers after the embedded tab is closed.
        Idempotent and defensive: never raises.
        """
        try:
            status_timer = getattr(self._inner, "status_timer", None)
            if status_timer is not None:
                try:
                    status_timer.stop()
                except Exception:  # pragma: no cover - defensive
                    logger.debug("status_timer.stop raised", exc_info=True)

            sim_widget = getattr(self._inner, "sim_widget", None)
            if sim_widget is not None:
                # ``MuJoCoSimWidget.set_running(False)`` pauses stepping
                # without tearing down the model — safe to call multiple
                # times and on widgets that are already stopped.
                try:
                    sim_widget.set_running(False)
                except Exception:  # pragma: no cover - defensive
                    logger.debug("sim_widget.set_running raised", exc_info=True)
                # Stop the underlying QTimer if present.
                inner_timer = getattr(sim_widget, "timer", None)
                if inner_timer is not None:
                    try:
                        inner_timer.stop()
                    except Exception:  # pragma: no cover - defensive
                        logger.debug("sim_widget.timer.stop raised", exc_info=True)
        except Exception:  # pragma: no cover - defensive
            logger.exception("MuJoCo MainWidget cleanup raised")
