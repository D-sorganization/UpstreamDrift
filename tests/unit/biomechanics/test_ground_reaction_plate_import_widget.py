"""Offscreen PyQt6 test of the plate-trial import on the ground-reaction tile (OF-07)."""

from __future__ import annotations

import os

import pytest

from src.shared.python.biomechanics.ground_reaction_plot_model import (
    GroundReactionPlotSeries,
)
from src.shared.python.force_plates import (
    PlateTimeAlignment,
    PlateUncertainty,
    placement_for_target,
)
from src.shared.python.force_plates.open_fp_format import PlateDeclarations
from src.shared.python.impact_parameters.target_frame import TargetFrame
from tests.unit.force_plates import _open_fp_writer as w

pytestmark = pytest.mark.unit

_APP = None


def _qt():
    """Return a real QApplication, or skip when PyQt6 is absent or a stand-in."""
    global _APP
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication, QWidget
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    if getattr(QWidget, "__module__", "") != "PyQt6.QtWidgets":
        pytest.skip("PyQt6 is a test stand-in in this environment")
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _model_series() -> GroundReactionPlotSeries:
    times = (0.0, 0.005, 0.010, 0.015, 0.020, 0.025)
    flat = {a: [0.0] * len(times) for a in ("x", "y", "z", "magnitude")}
    return GroundReactionPlotSeries(
        time_s=times,
        feet=("left", "right"),
        traces={"left_force_n": flat, "net_force_n": flat},
        load_share={},
        available=True,
    )


def _declared() -> PlateDeclarations:
    return PlateDeclarations(
        uncertainty=PlateUncertainty(
            force_floor_n=2.0,
            force_fraction=0.01,
            cop_m=0.010,
            sensing_depth_sigma_m=0.002,
            source="declared hypothesis for tests",
        ),
        sensing_depth_m=0.015,
        capacity_n=3923.0,
    )


def test_unqualified_trial_shows_the_banner(tmp_path) -> None:
    app = _qt()
    from src.tools.ground_reaction_plots.gui import GroundReactionPlotWidget

    widget = GroundReactionPlotWidget()
    widget.set_series(_model_series())
    assert widget.banner_label.isHidden()
    alignment = PlateTimeAlignment(
        offset_s=0.0, sigma_offset_s=0.001, kind="manual", note="test clock"
    )
    placement = placement_for_target(
        "RH", TargetFrame(), origin_m=(0.0, 0.0, 0.0)
    )
    widget.load_plate_trial(w.write_swing(tmp_path), _declared(), placement, alignment)
    assert not widget.banner_label.isHidden()
    assert widget.banner_label.text() == "Unqualified measurement"
    assert "Measured Left Force" in widget._series.labels.values()
    widget.set_series(_model_series())
    assert widget.banner_label.isHidden()
    app.processEvents()
    widget.cleanup()


def test_import_without_a_model_series_is_refused(tmp_path) -> None:
    _qt()
    from src.tools.ground_reaction_plots.gui import GroundReactionPlotWidget

    widget = GroundReactionPlotWidget()
    alignment = PlateTimeAlignment(
        offset_s=0.0, sigma_offset_s=0.001, kind="manual", note="test clock"
    )
    placement = placement_for_target(
        "RH", TargetFrame(), origin_m=(0.0, 0.0, 0.0)
    )
    with pytest.raises(ValueError, match="model ground reaction"):
        widget.load_plate_trial(
            w.write_swing(tmp_path), _declared(), placement, alignment
        )
    widget.cleanup()
