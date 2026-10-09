"""Offscreen PyQt6 tests for the grip wrench plot widget (GCV-10, #11716)."""

from __future__ import annotations

import json

import pytest

from src.shared.python.biomechanics.grip_plot_model import (
    build_grip_plot_series,
    plot_series_to_json,
)
from src.shared.python.biomechanics.grip_wrench import HandWrench, analyze_grip

pytestmark = pytest.mark.unit

_APP = None


def _qt():
    global _APP
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _series():
    analyses = [
        analyze_grip(
            HandWrench("L", (0, 0, 1.0), (10.0 * k, 0, 0), (0, 0, 0)),
            HandWrench("R", (0, 0, 0.8), (-10.0 * k, 0, 0), (0, 0, 0)),
            split_method="efc_force",
        )
        for k in range(4)
    ]
    return build_grip_plot_series(
        [0.0, 0.01, 0.02, 0.03], analyses, events={"impact": 0.03}
    )


def test_widget_shows_split_method_and_draws_four_panels():
    _qt()
    from src.tools.grip_wrench_plots.gui import GripWrenchPlotWidget

    w = GripWrenchPlotWidget()
    assert "unavailable" in w.status_label.text()
    w.set_series(_series())
    assert "efc_force" in w.status_label.text()
    assert len(w.figure.axes) == 4
    w.frame_combo.setCurrentText("club")
    assert w.figure.axes[2].get_title().endswith("(Club)")
    w.cleanup()
    w.cleanup()  # idempotent


def test_widget_loads_the_api_payload_json(tmp_path):
    _qt()
    from src.tools.grip_wrench_plots.gui import GripWrenchPlotWidget

    path = tmp_path / "grip.json"
    path.write_text(plot_series_to_json(_series()), encoding="utf-8")
    w = GripWrenchPlotWidget()
    w.load_json(path)
    assert "efc_force" in w.status_label.text()
    assert len(w.figure.axes) == 4

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"traces": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        w.load_json(bad)


def test_series_from_payload_round_trips_none_to_gaps():
    from src.shared.python.biomechanics.grip_plot_model import series_from_payload

    payload = _series().to_dict()
    payload["traces"]["left_force_n"]["magnitude"][1] = None
    again = series_from_payload(payload)
    assert again.traces["left_force_n"]["magnitude"][1] is None
    assert again.split_method == "efc_force" and again.events == {"impact": 0.03}
    with pytest.raises(ValueError):
        series_from_payload({"time_s": [0.0]})
