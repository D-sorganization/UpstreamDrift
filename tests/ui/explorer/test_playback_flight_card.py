"""Offscreen: the Shot card beside the live Results Playback (#15150 G5b)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6", reason="the Shot card needs PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox  # noqa: E402

from src.shared.python.motion_matching.same_input.bundle import (  # noqa: E402
    InputBundle,
)
from src.tools.model_explorer import playback_flight_panel as shot  # noqa: E402
from src.tools.model_explorer.default_match import DEMO_BUNDLE  # noqa: E402
from src.tools.native_viewer_export import dialog_core  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.ui]
SUMMARY = shot.ShotSummary(
    "at impact", True, 114.7, ball_speed_mph=169.6, carry_m=222.5
)
RIGHT = Qt.DockWidgetArea.RightDockWidgetArea


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    inst = QApplication.instance()
    return inst if isinstance(inst, QApplication) else QApplication(["shot"])


def test_card_switches_units_with_the_export_dialog_options(qapp: QApplication) -> None:
    window = QMainWindow()
    dock = shot.attach_shot_dock(window, None, SUMMARY)
    card = dock.shot_card
    assert window.dockWidgetArea(dock) == RIGHT
    assert card.unit_choices() == (
        dialog_core.distance_unit_options(),
        dialog_core.speed_unit_options(),
    )
    assert dict(card.shown_rows())["Carry"] == "243 yd"
    card.set_units("m", "m/s")
    assert dict(card.shown_rows())["Ball Speed"] == "76 m/s"
    missed = shot.ShotSummary("at impact", contact=False)
    assert shot.attach_shot_dock(window, dock, missed) is dock
    assert dict(card.shown_rows())["Ball Flight"] == "no contact: no flight"
    window.deleteLater()


def test_loading_a_bundle_docks_the_card_beside_the_playback(
    qapp: QApplication, tmp_path: Path
) -> None:
    mujoco = pytest.importorskip("mujoco", reason="golfer playback needs mujoco")
    try:
        mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 16, 16).close()
    except Exception as exc:  # noqa: BLE001 - any GL backend failure means skip
        pytest.skip(f"no offscreen OpenGL for mujoco.Renderer: {exc}")
    from src.tools.model_explorer.gui import MainWidget

    src = InputBundle.load(DEMO_BUNDLE)  # every 12th sample: fast, no private data
    q = np.asarray(src.reference_q[::12], dtype=float)
    v, n = np.zeros_like(q), len(q) - 1
    cut = (src.spec_bytes, src.coordinate_order, float(src.dt_s) * 12, q[0], v[0])
    InputBundle(*cut, np.zeros((n, q.shape[1])), q, v, src.reference_engine).save(
        tmp_path / "demo.npz"
    )
    explorer = MainWidget()
    with patch.object(QMessageBox, "warning", lambda *a, **k: 0):
        assert explorer.load_results(tmp_path / "demo.npz") is True
    dock, inner = explorer.shot_dock, explorer._inner
    assert dock is not None and not dock.isHidden()
    assert dock not in inner.tabifiedDockWidgets(explorer.playback_dock)
    assert inner.dockWidgetArea(dock) == RIGHT
    rows = dict(dock.shot_card.shown_rows())  # the demo swing never meets the ball
    assert rows["Status"] == "no impact detected" and rows["Club Speed"] == "--"
    explorer._discard_results()
    assert dock.isHidden()
    explorer.cleanup()
    explorer.deleteLater()
