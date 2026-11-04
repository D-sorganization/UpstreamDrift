"""Batch dispersion panel in the Putting Green Simulator (#14459, epic #14452).

The desktop simulator runs the shared dispersion service on the green the
user has dialled in: a batch of putts with stroke noise (rest-position
scatter and make rate) and the 3-45 ft make-% spread. Golden parity: the
panel shows exactly what :mod:`service_dispersion` returns for the same
arguments, so desktop, web and studio agree.
"""

from __future__ import annotations

import math
import os
import sys
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.putting_analytics import service_dispersion  # noqa: E402
from src.shared.python.ui.units import UnitSystem  # noqa: E402
from src.tools.putting_green_gui._batch_panel import (  # noqa: E402
    BATCH_SIZES,
    BatchPanel,
    batch_args,
)
from src.tools.putting_green_gui._scene_builder import PuttConfig  # noqa: E402
from src.tools.putting_green_gui.gui import PuttingGreenWidget  # noqa: E402

pytestmark = pytest.mark.unit

_FT_M = 0.3048
_CONFIG = PuttConfig(
    putter_speed_ms=2.5,
    aim_deg=0.0,
    cup_distance_m=12.0 * _FT_M,
    stimp=11.0,
    slope_deg=2.0,
)
_NOISE = {"face_sd_deg": 1.5, "speed_cv": 0.08, "n": 40, "seed": 0}


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def panel(qapp: QApplication) -> BatchPanel:
    p = BatchPanel(
        lambda: _CONFIG,
        lambda: UnitSystem.IMPERIAL,
        spread_n=4,
        spread_grid_ft=(3.0, 20.0),
    )
    p.size_combo.setCurrentIndex(0)  # 40 putts keeps the suite quick
    return p


def test_batch_args_map_the_dialled_green() -> None:
    args = batch_args(_CONFIG, face_sd_deg=1.2, speed_cv=0.07, n=40, seed=3)
    assert args["distance_ft"] == pytest.approx(12.0)
    assert args["grade_pct"] == pytest.approx(100.0 * math.tan(math.radians(2.0)))
    # The scene's height rises to the left of the line, so the green falls right.
    assert args["aspect_deg"] == -90.0
    assert args["stimp"] == 11.0
    assert (args["face_sd_deg"], args["speed_cv"]) == (1.2, 0.07)
    assert (args["n"], args["seed"]) == (40, 3)
    flat = batch_args(PuttConfig(2.5, 0.0, 3.0, 10.0, 0.0), **_NOISE)
    assert (flat["grade_pct"], flat["aspect_deg"]) == (0.0, 0.0)  # no fall line


def test_batch_shows_the_service_answer(panel: BatchPanel) -> None:
    combo = panel.size_combo  # the studio's batch sizes
    assert [combo.itemData(i) for i in range(combo.count())] == [40, 100, 1000]
    panel.arrange_plots(300)  # narrow canvases stack the charts
    assert panel.plots_stacked()
    panel.arrange_plots(900)
    assert not panel.plots_stacked()
    report = panel.run_batch()
    expected = service_dispersion.dispersion_study(panel.current_args())
    assert report == expected
    assert f"{100 * expected['make_rate']:.0f}%" in panel.summary_text()


def test_scatter_is_drawn_in_the_chosen_units(qapp: QApplication) -> None:
    for system, label in ((UnitSystem.IMPERIAL, "ft"), (UnitSystem.METRIC, "m")):
        p = BatchPanel(lambda: _CONFIG, lambda s=system: s)
        p.size_combo.setCurrentIndex(0)
        report = p.run_batch()
        assert p.scatter_axis_label().endswith(f"({label})")
        xs = p.scatter_x()
        scale = 1.0 if system == UnitSystem.IMPERIAL else _FT_M
        assert xs[0] == pytest.approx(report["rest_x_ft"][0] * scale)


def test_a_unit_change_replots_the_last_batch(qapp: QApplication) -> None:
    units = [UnitSystem.IMPERIAL]
    p = BatchPanel(lambda: _CONFIG, lambda: units[0])
    p.size_combo.setCurrentIndex(0)
    p.run_batch()
    units[0] = UnitSystem.METRIC
    p.redraw()
    assert p.scatter_axis_label().endswith("(m)")


def test_spread_plots_the_service_make_rates(panel: BatchPanel) -> None:
    report = panel.run_spread()
    args = {**panel.current_args(), "n": 4, "distances_ft": [3.0, 20.0]}
    args.pop("distance_ft")
    assert report == service_dispersion.make_spread_study(args)
    assert report["samples_per_distance"] == 4
    assert panel.spread_points() == list(
        zip(report["distances_ft"], report["make_rate"], strict=True)
    )


def test_button_runs_the_batch_off_the_gui_thread(
    qapp: QApplication, panel: BatchPanel
) -> None:
    panel.start_batch()
    assert panel.is_busy()  # returned at once; the worker is rolling
    deadline = time.monotonic() + 60.0
    while panel.is_busy() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    assert not panel.is_busy()
    expected = service_dispersion.dispersion_study(panel.current_args())
    assert f"{100 * expected['make_rate']:.0f}%" in panel.summary_text()


def test_panel_spread_defaults_to_the_3_to_45_ft_grid(qapp: QApplication) -> None:
    units = [UnitSystem.METRIC]
    p = BatchPanel(lambda: _CONFIG, lambda: units[0])
    args = p.spread_args()
    assert args["distances_ft"] == list(service_dispersion.DEFAULT_SPREAD_GRID_FT)
    assert "distance_ft" not in args
    assert p.spread_label() == "0-14 m Spread"  # the button follows the units
    units[0] = UnitSystem.IMPERIAL
    p.redraw()  # the simulator redraws the panel on a units change
    assert p.spread_label() == "0-45 ft Spread"


def test_widget_feeds_its_green_into_the_panel(qapp: QApplication) -> None:
    w = PuttingGreenWidget(unit_system=UnitSystem.IMPERIAL)
    try:
        w._stimp_spin.setValue(12.5)
        w._slope_spin.setValue(0.0)
        args = w.batch_panel.current_args()
        assert (args["stimp"], args["grade_pct"]) == (12.5, 0.0)
        assert args["distance_ft"] == pytest.approx(10.0, abs=0.01)
    finally:
        w.cleanup()
