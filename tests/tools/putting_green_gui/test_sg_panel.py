"""Strokes-gained panel in the Putting Green Simulator (#14467, epic #14452).

The desktop simulator solves the player's own E(d) on the dialled green with
the shared service and scores each putt against it. Golden parity: the curve
and strokes gained are exactly what :func:`service_sg.sg_study` returns, and
a later putt on the same green scores on the cached curve without a re-solve.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.putting_analytics import service_sg  # noqa: E402
from src.shared.python.ui.units import UnitSystem  # noqa: E402
from src.tools.putting_green_gui._sg_panel import SGPanel  # noqa: E402

pytestmark = pytest.mark.unit

_FT_M = 0.3048
_GREEN = {
    "distance_ft": 6.0,
    "grade_pct": 2.0,
    "aspect_deg": -90.0,
    "stimp": 11.0,
    "face_sd_deg": 1.0,
    "speed_cv": 0.06,
    "n": 40,  # the batch size; the panel uses its own
    "seed": 0,
}
_SMALL = {"n": 12, "paces_ft": (1.0, 2.0), "distances_ft": (1.0, 3.0, 6.0, 12.0)}


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv[:1])


def _panel(units: list[UnitSystem], green: dict[str, Any] = _GREEN) -> SGPanel:
    return SGPanel(lambda: green, lambda: units[0], **_SMALL)  # type: ignore[arg-type]


def test_study_args_take_the_green_and_the_putt(qapp: QApplication) -> None:
    panel = _panel([UnitSystem.IMPERIAL])
    panel.set_putt(holed=False, leave_ft=2.0)
    args = panel.study_args()
    assert (args["distance_ft"], args["grade_pct"], args["stimp"]) == (6.0, 2.0, 11.0)
    assert (args["face_sd_deg"], args["speed_cv"]) == (1.0, 0.06)
    assert (args["n"], args["paces_ft"]) == (12, [1.0, 2.0])
    assert args["distances_ft"] == [1.0, 3.0, 6.0, 12.0]
    assert (args["holed"], args["leave_ft"]) == (False, pytest.approx(2.0))


def test_score_shows_the_service_answer(qapp: QApplication) -> None:
    panel = _panel([UnitSystem.IMPERIAL])
    panel.set_putt(holed=False, leave_ft=2.0)
    report = panel.run_score()
    assert report == service_sg.sg_study(panel.study_args())
    assert f"{report['sg_player']:+.2f}" in panel.summary_text()
    assert panel.curve_points() == list(
        zip(
            report["curve"]["distances_ft"],
            report["curve"]["expected_putts"],
            strict=True,
        )
    )


def test_a_later_putt_scores_on_the_cached_curve(qapp: QApplication) -> None:
    panel = _panel([UnitSystem.IMPERIAL])
    panel.set_putt(holed=False, leave_ft=2.0)
    report = panel.run_score()
    panel.set_putt(holed=True, leave_ft=0.0)
    assert panel.has_cached_curve()
    panel.start_score()  # cached: scores at once, no worker
    assert not panel.is_busy()
    expected = service_sg.score_putt(report, True, 0.0)
    assert f"{expected:+.2f}" in panel.summary_text()


def test_leave_and_curve_follow_the_units(qapp: QApplication) -> None:
    units = [UnitSystem.METRIC]
    panel = _panel(units)
    panel.set_leave_display(1.0)  # metres
    assert panel.study_args()["leave_ft"] == pytest.approx(1.0 / _FT_M)
    panel.run_score()
    assert panel.curve_axis_label().endswith("(m)")
    units[0] = UnitSystem.IMPERIAL
    panel.redraw()
    assert panel.curve_axis_label().endswith("(ft)")
    assert panel.study_args()["leave_ft"] == pytest.approx(1.0 / _FT_M, abs=0.01)


def test_off_grid_distance_is_reported_not_raised(qapp: QApplication) -> None:
    panel = _panel([UnitSystem.IMPERIAL], {**_GREEN, "distance_ft": 30.0})
    panel.start_score()
    deadline = time.monotonic() + 30.0
    while panel.is_busy() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    assert "rejected" in panel.summary_text()
    assert "grid" in panel.summary_text()


def test_button_solves_off_the_gui_thread(qapp: QApplication) -> None:
    panel = _panel([UnitSystem.IMPERIAL])
    panel.set_putt(holed=False, leave_ft=3.0)
    panel.start_score()
    assert panel.is_busy()
    deadline = time.monotonic() + 60.0
    while panel.is_busy() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    assert panel.has_cached_curve()
    report = service_sg.sg_study(panel.study_args())
    assert f"{report['sg_player']:+.2f}" in panel.summary_text()
