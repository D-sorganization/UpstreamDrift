"""Starting-pose matcher: align Simscape golfer skeleton to mocap targets.

A focused, professional-grade alignment tool that lets you place the
Simscape model in the right starting pose BEFORE running any optimiser.

Why: starting fmincon at zero-theta dropped it into a bad local minimum
because the model started in the wrong pose. This tool produces the
seed (rigid transform + scale) that fit_swing_full_pipeline uses as
input_overrides for the model workspace.

Workflow:
    1. Loads the Wiffle ProV1 motion-capture xlsx.
       NOTE: Wiffle xlsx positions are in CENTIMETRES — see
       MATLAB_GOLF_MODEL_GUIDE.md.  We bypass the legacy
       mocap_data_loader.py (which uses the wrong inches→m factor).
    2. Reads the row-1 event header (A=address, T=top, I=impact, F=finish).
    3. Loads up to two pose skeletons (TopofBackswing + Impact) from
       simscape_skeleton_<pose>.json (produced by export_default_skeleton.m).
       Falls back to a hardcoded approximate pose if absent.
    4. A 7-DOF transform (Tx/Ty/Tz/Rx/Ry/Rz/Scale) applies to all visible
       skeletons.  Rx/Ry are LOCKED by default (both data and model use
       Z-up; the only physical DoF that matters is global heading via Rz
       plus a translation).  Unlock with the checkbox if needed.
    5. Two-point shaft snap: solves Rz + Tx/Ty/Tz so the SHAFT (mid-hands
       to clubhead vector) of the model pose aligns with the mocap shaft
       at the chosen event frame — not just the mid-hands point.
    6. Save offsets to JSON; later it seeds model-workspace overrides in
       fit_swing_full_pipeline.

Run::

    python -m src.tools.starting_pose_matcher

Or, from the UpstreamDriftLauncher tile (registered in ``src/config/models.yaml``).

Subtask 5 / #4998 of EPIC #4993 split the original ~3.1k-line module
into the embeddable :class:`MainWidget` (in :mod:`gui_main_widget`)
plus three mixin modules (:mod:`gui_render_mixin`,
:mod:`gui_builders_mixin`, :mod:`gui_session_mixin`). This module
keeps the QMainWindow shell so ``python -m
src.tools.starting_pose_matcher`` still produces a top-level window
with the same title and geometry.
"""

from __future__ import annotations

import logging
import sys

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStyleFactory,
    QWidget,
)

from ._gui_common import LabelledControl, _QSS
from .gui_main_widget import MainWidget

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


__all__ = [
    "LabelledControl",
    "MainWidget",
    "StartingPoseMatcher",
    "main",
]


class StartingPoseMatcher(QMainWindow):
    """Standalone shell that hosts :class:`MainWidget` in a top-level window.

    Used when ``python -m src.tools.starting_pose_matcher`` runs the tool
    directly. The embed path goes through
    :class:`_MotionMatchPreviewEmbedAdapter` instead and never constructs
    this shell.

    Attribute access falls through to :attr:`_main_widget` so legacy
    callers and headless coverage tests can still reach the per-section
    widgets (``btn_load``, ``cb_clubhead_trace``, …) on the window
    object directly.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Starting-Pose Matcher")
        self.resize(1700, 1000)
        self._main_widget = MainWidget(self)
        self.setCentralWidget(self._main_widget)

    def __getattr__(self, name: str):  # noqa: ANN204 - matches QObject signature
        # ``__getattr__`` only fires when normal lookup misses, so this
        # transparently forwards anything not on QMainWindow itself
        # (e.g. ``btn_load``, ``cb_show_ball``, ``_redraw``) to the
        # embedded :class:`MainWidget`. Raises :class:`AttributeError`
        # when the widget is missing too, matching Python defaults.
        try:
            inner = self.__dict__["_main_widget"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(inner, name)


# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #


def get_dockable_ui() -> QtWidgets.QMainWindow:
    """Return the main window instance for docking in the unified launcher."""
    return StartingPoseMatcher()

def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    app.setStyleSheet(_QSS)
    base_font = QFont("Segoe UI", 10)
    app.setFont(base_font)
    win = StartingPoseMatcher()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
