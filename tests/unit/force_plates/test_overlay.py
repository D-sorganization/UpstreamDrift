"""Measured plate overlay on the ground-reaction plot series (OF-07, #13087)."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from src.shared.python.biomechanics.ground_reaction_plot_model import (
    GroundReactionPlotSeries,
    series_from_payload,
)
from src.shared.python.force_plates import (
    PlacedTrial,
    PlatePlacement,
    PlateTimeAlignment,
    ReadingOptions,
    attach_measured_overlay,
    measured_overlay,
    overlay_on_series,
)
from tests.unit.force_plates import _trial_factory as tf

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _align(offset_s: float = 0.0) -> PlateTimeAlignment:
    return PlateTimeAlignment(
        offset_s=offset_s, sigma_offset_s=0.001, kind="manual", note="test clock"
    )


def _placed(**overrides: object) -> PlacedTrial:
    return PlacedTrial(tf.make_trial(**overrides), PlatePlacement.identity())


def _opts() -> ReadingOptions:
    return ReadingOptions(coverage=2.0)


def _model_series(times: tuple[float, ...]) -> GroundReactionPlotSeries:
    flat = {a: [1.0] * len(times) for a in ("x", "y", "z", "magnitude")}
    return GroundReactionPlotSeries(
        time_s=times,
        feet=("left", "right"),
        traces={"left_force_n": flat, "right_force_n": flat},
        load_share={},
        available=True,
        events={"impact": 0.01},
    )


def test_shear_is_unobserved_and_load_is_the_reading() -> None:
    overlay = measured_overlay(_placed(), _align(), (0.0, 0.005), _opts())
    left = overlay.traces["measured_left_force_n"]
    assert left["x"] == [None, None]
    assert left["y"] == [None, None]
    assert left["z"] == pytest.approx([400.0, 300.0])
    right = overlay.traces["measured_right_force_n"]
    assert right["z"] == pytest.approx([400.0, 500.0])


def test_cop_is_on_the_sensing_plane_and_none_when_unloaded() -> None:
    overlay = measured_overlay(_placed(), _align(), (0.0, 0.010), _opts())
    cop = overlay.traces["measured_left_cop_m"]
    assert cop["x"][0] == pytest.approx(-0.185)
    assert cop["y"][0] == pytest.approx(0.02)
    assert cop["z"][0] == pytest.approx(-0.015)  # sensing depth below the top
    assert cop["x"][1] is None and cop["y"][1] is None


def test_frames_outside_the_trial_are_none_never_extrapolated() -> None:
    overlay = measured_overlay(_placed(), _align(), (-0.001, 0.005, 0.011), _opts())
    assert overlay.covered == (False, True, False)
    for name, trace in overlay.traces.items():
        for axis in ("x", "y", "z"):
            assert trace[axis][0] is None and trace[axis][2] is None, name
    assert overlay.traces["measured_left_force_n"]["z"][1] == pytest.approx(300.0)


def test_alignment_offset_moves_the_covered_window() -> None:
    overlay = measured_overlay(_placed(), _align(1.0), (0.0, 1.0, 1.005), _opts())
    assert overlay.covered == (False, True, True)


def test_qualified_follows_provenance_and_observability_is_vertical_only() -> None:
    unqualified = measured_overlay(_placed(), _align(), (0.0,), _opts())
    assert unqualified.qualified is False
    assert unqualified.observability == {
        "fx": False,
        "fy": False,
        "fz": True,
        "cop": True,
        "free_moment": False,
    }
    prov = replace(tf.provenance(), qualification="DV-13061")
    qualified = measured_overlay(_placed(provenance=prov), _align(), (0.0,), _opts())
    assert qualified.qualified is True


def test_non_finite_times_are_refused() -> None:
    with pytest.raises(ValueError, match="finite"):
        measured_overlay(_placed(), _align(), (float("nan"),), _opts())


def test_attach_adds_traces_events_and_provenance() -> None:
    series = _model_series((0.0, 0.005, 0.010))
    out = overlay_on_series(series, _placed(), _align(), _opts())
    assert "measured_left_force_n" in out.traces
    assert "left_force_n" in out.traces
    assert out.events["impact"] == 0.01
    assert out.events["force.max_trail"] == pytest.approx(0.005)
    assert out.measured_qualified is False
    assert out.labels["measured_left_force_n"] == "Measured Left Force"
    assert series.measured_qualified is None  # original untouched


def test_json_round_trip_keeps_overlay_fields() -> None:
    series = _model_series((0.0, 0.005, 0.010))
    out = overlay_on_series(series, _placed(), _align(), _opts())
    payload = json.loads(json.dumps(out.to_dict(), allow_nan=False))
    assert payload["measured_qualified"] is False
    assert payload["observability"]["fx"] is False
    rebuilt = series_from_payload(payload)
    assert rebuilt.measured_qualified is False
    assert rebuilt.traces["measured_left_force_n"]["x"] == [None] * 3
    assert rebuilt.to_dict() == out.to_dict()


def test_plain_payload_is_unchanged() -> None:
    payload = _model_series((0.0,)).to_dict()
    assert "measured_qualified" not in payload
    assert "observability" not in payload


def test_attach_refuses_wrong_length_and_double_attach() -> None:
    overlay = measured_overlay(_placed(), _align(), (0.0, 0.005), _opts())
    with pytest.raises(ValueError, match="length"):
        attach_measured_overlay(_model_series((0.0,)), overlay)
    once = attach_measured_overlay(_model_series((0.0, 0.005)), overlay)
    with pytest.raises(ValueError, match="already"):
        attach_measured_overlay(once, overlay)


def test_attach_rejects_wrong_types() -> None:
    overlay = measured_overlay(_placed(), _align(), (0.0,), _opts())
    with pytest.raises(TypeError):
        attach_measured_overlay("x", overlay)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        attach_measured_overlay(_model_series((0.0,)), "x")  # type: ignore[arg-type]


def test_force_namespace_cannot_clobber_model_events() -> None:
    series = replace(_model_series((0.0,)), events={"force.impact": 0.0})
    overlay = measured_overlay(_placed(), _align(), (0.0,), _opts())
    with pytest.raises(ValueError, match="force namespace"):
        attach_measured_overlay(series, overlay)


def test_load_sigma_is_the_random_one_sigma_not_the_band_equivalent() -> None:
    from src.shared.python.force_plates import reading_at

    placed, times = _placed(), (0.0, 0.0025, 0.005)
    # Between samples the reading carries an interpolation bias; declared
    # curvature floors make it computable.
    opts = ReadingOptions(
        coverage=2.0, load_curvature_floor=1.0e5, cop_curvature_floor=1.0
    )
    overlay = measured_overlay(placed, _align(), times, opts)
    sigma = overlay.traces["measured_left_force_sigma_n"]
    feet = [reading_at(placed, t, _align(), opts).feet["left"] for t in times]
    assert sigma["z"] == pytest.approx([f.load_sigma_random_n for f in feet])
    # sigma_load_n adds bias / coverage and is not a 1-sigma: never plot it as one.
    biased = [f for f in feet if f.load_bias_n]
    assert biased, "fixture must exercise a nonzero interpolation bias"
    assert all(f.load_sigma_random_n < f.sigma_load_n for f in biased)
    assert sigma["x"] == sigma["y"] == sigma["magnitude"] == [None] * 3
    assert all(s > 0.0 for s in sigma["z"])


def test_load_sigma_is_none_outside_the_trial() -> None:
    overlay = measured_overlay(_placed(), _align(), (-1.0, 0.0), _opts())
    assert overlay.traces["measured_left_force_sigma_n"]["z"][0] is None


def test_uncertainty_source_travels_with_the_overlay() -> None:
    placed = _placed()
    series = overlay_on_series(_model_series((0.0, 0.005)), placed, _align(), _opts())
    source = placed.trial.uncertainty.source
    assert series.measured_uncertainty_source == source
    payload = json.loads(json.dumps(series.to_dict(), allow_nan=False))
    assert payload["measured_uncertainty_source"] == source
    assert series_from_payload(payload).measured_uncertainty_source == source
    assert series.labels["measured_left_force_sigma_n"] == (
        "Measured Left Force Uncertainty (1 Sigma)"
    )
    assert series.units["measured_left_force_sigma_n"] == "N"


def test_plain_payload_has_no_uncertainty_source() -> None:
    assert "measured_uncertainty_source" not in _model_series((0.0,)).to_dict()


def test_cop_sigma_is_the_random_one_sigma_per_axis() -> None:
    from src.shared.python.force_plates import reading_at

    placed, times = _placed(), (0.0, 0.005)
    overlay = measured_overlay(placed, _align(), times, _opts())
    sigma = overlay.traces["measured_left_cop_sigma_m"]
    feet = [reading_at(placed, t, _align(), _opts()).feet["left"] for t in times]
    expected = [None if f.cop_m is None else f.cop_sigma_random_m for f in feet]
    assert expected[0] is not None, "fixture must load the left foot at t=0"
    assert sigma["z"] == expected
    # x and y carry the per-world-axis random sigma; z keeps the worse axis.
    per_axis = [None if f.cop_m is None else f.cop_sigma_random_xy_m for f in feet]
    assert [None if a is None else (a[0], a[1]) for a in per_axis] == [
        None if x is None else (x, y)
        for x, y in zip(sigma["x"], sigma["y"], strict=True)
    ]
    assert sigma["magnitude"] == [None, None]


def test_cop_sigma_is_none_where_the_cop_is() -> None:
    overlay = measured_overlay(_placed(), _align(), (-1.0, 0.0), _opts())
    assert overlay.traces["measured_left_cop_sigma_m"]["z"][0] is None
    for foot in ("left", "right"):
        cop = overlay.traces[f"measured_{foot}_cop_m"]["x"]
        sigma = overlay.traces[f"measured_{foot}_cop_sigma_m"]["z"]
        assert [c is None for c in cop] == [s is None for s in sigma]


def test_cop_sigma_trace_is_labelled() -> None:
    series = overlay_on_series(
        _model_series((0.0, 0.005)), _placed(), _align(), _opts()
    )
    assert series.labels["measured_left_cop_sigma_m"] == (
        "Measured Left Centre of Pressure Uncertainty (1 Sigma)"
    )
    assert series.units["measured_left_cop_sigma_m"] == "m"


def test_cop_covariance_trace_carries_the_reading_covariance() -> None:
    from src.shared.python.force_plates import reading_at

    placed, times = _placed(), (-1.0, 0.0, 0.005, 0.010)
    overlay = measured_overlay(placed, _align(), times, _opts())
    for foot in ("left", "right"):
        cov = overlay.traces[f"measured_{foot}_cop_cov_m2"]
        cop = overlay.traces[f"measured_{foot}_cop_m"]["x"]
        expected = [
            None
            if c is None
            else reading_at(placed, t, _align(), _opts())
            .feet[foot]
            .cop_cov_random_xy_m2
            for t, c in zip(times, cop, strict=True)
        ]
        got = [
            None if xx is None else (xx, yy, xy)
            for xx, yy, xy in zip(cov["x"], cov["y"], cov["z"], strict=True)
        ]
        assert got == expected
        assert cov["magnitude"] == [None] * len(times)
    assert overlay.traces["measured_left_cop_cov_m2"]["x"][1] is not None


def test_cop_covariance_trace_is_labelled() -> None:
    series = overlay_on_series(
        _model_series((0.0, 0.005)), _placed(), _align(), _opts()
    )
    assert series.labels["measured_left_cop_cov_m2"] == (
        "Measured Left Centre of Pressure Covariance (1 Sigma)"
    )
    assert series.units["measured_left_cop_cov_m2"] == "m^2"
