"""Tests for the ground-reaction plot series (GCV-5, #11711)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.shared.python.biomechanics.ground_reaction import (
    COP_MIN_FZ_N,
    ContactSet,
    GroundReactionSeries,
    analyze_ground_reaction,
)
from src.shared.python.biomechanics.ground_reaction_plot_model import (
    GroundReactionPlotSeries,
    build_ground_reaction_plot_series,
    plot_series_to_json,
    unavailable_ground_reaction_plot,
)
from src.shared.python.biomechanics.plot_traces import none_if_nan, vector_trace

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

G = 9.80665
BODY_WEIGHT_N = 80.0 * G


def _stance(fz_left: float, fz_right: float):
    contacts = {}
    for name, fz, y in (("left", fz_left, 0.15), ("right", fz_right, -0.15)):
        contacts[name] = (
            ContactSet(np.array([[0.0, 0.0, fz]]), np.array([[0.0, y, 0.0]]))
            if fz > 0.0
            else ContactSet.empty()
        )
    return analyze_ground_reaction(contacts, (0.0, 0.0, 0.95))


def _series() -> GroundReactionSeries:
    w = BODY_WEIGHT_N
    return GroundReactionSeries.from_breakdowns(
        [0.0, 0.1, 0.2, 0.3],
        [_stance(w / 2, w / 2), _stance(0.75 * w, 0.25 * w), _stance(w, 0.0),
         _stance(0.0, 0.0)],
    )  # fmt: skip


def test_none_if_nan_and_vector_trace_replace_non_finite_with_none() -> None:
    assert none_if_nan(np.array([1.0, np.nan, np.inf])) == [1.0, None, None]
    trace = vector_trace(np.array([[3.0, 4.0, 0.0], [np.nan, np.nan, np.nan]]))
    assert trace == {
        "x": [3.0, None],
        "y": [4.0, None],
        "z": [0.0, None],
        "magnitude": [5.0, None],
    }


def test_vector_trace_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(T, 3\)"):
        vector_trace(np.zeros((4, 2)))


def test_traces_cover_every_foot_and_net_quantity() -> None:
    plot = build_ground_reaction_plot_series(_series())
    for key in ("left", "right", "net"):
        for quantity in ("force_n", "cop_m", "free_moment_nm", "moment_com_nm"):
            trace = plot.traces[f"{key}_{quantity}"]
            assert set(trace) == {"x", "y", "z", "magnitude"}
            assert all(len(v) == 4 for v in trace.values())
    assert plot.feet == ("left", "right")
    assert plot.available
    assert plot.time_s == (0.0, 0.1, 0.2, 0.3)


def test_force_in_body_weights_only_when_body_weight_given() -> None:
    assert "net_force_bw" not in build_ground_reaction_plot_series(_series()).traces
    plot = build_ground_reaction_plot_series(_series(), body_weight_n=BODY_WEIGHT_N)
    assert plot.traces["net_force_bw"]["z"][:3] == pytest.approx([1.0, 1.0, 1.0])
    assert plot.traces["left_force_bw"]["z"][1] == pytest.approx(0.75)
    assert plot.units["net_force_bw"] == "BW"


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_body_weight_must_be_positive_and_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="body_weight_n"):
        build_ground_reaction_plot_series(_series(), body_weight_n=bad)


def test_vertical_load_share_is_none_without_support() -> None:
    plot = build_ground_reaction_plot_series(_series())
    assert plot.load_share["left"][:3] == pytest.approx([0.5, 0.75, 1.0])
    assert plot.load_share["right"][:3] == pytest.approx([0.5, 0.25, 0.0])
    assert plot.load_share["left"][3] is None  # flight phase: unavailable, not 0
    assert plot.load_share["right"][3] is None


def test_load_share_needs_the_cop_threshold_of_vertical_support() -> None:
    light = COP_MIN_FZ_N / 4.0  # both feet together stay below the threshold
    series = GroundReactionSeries.from_breakdowns([0.0], [_stance(light, light)])
    plot = build_ground_reaction_plot_series(series)
    assert plot.load_share == {"left": [None], "right": [None]}


def test_unavailable_cop_is_none_never_zero() -> None:
    plot = build_ground_reaction_plot_series(_series())
    assert plot.traces["right_cop_m"]["x"][2] is None
    assert plot.traces["net_cop_m"]["y"][3] is None
    assert plot.traces["right_free_moment_nm"]["z"][2] is None
    assert plot.traces["left_cop_m"]["y"][0] == pytest.approx(0.15)


def test_events_are_passed_through_and_validated() -> None:
    plot = build_ground_reaction_plot_series(
        _series(), events={"address": 0.0, "impact": 0.2}
    )
    assert plot.events == {"address": 0.0, "impact": 0.2}
    with pytest.raises(ValueError, match="event"):
        build_ground_reaction_plot_series(_series(), events={"top": math.nan})


def test_rejects_wrong_input_type() -> None:
    with pytest.raises(TypeError, match="GroundReactionSeries"):
        build_ground_reaction_plot_series("not a series")  # type: ignore[arg-type]


def test_no_contact_anywhere_is_unavailable_with_reason() -> None:
    series = GroundReactionSeries.from_breakdowns(
        [0.0, 0.1], [_stance(0.0, 0.0), _stance(0.0, 0.0)]
    )
    plot = build_ground_reaction_plot_series(series)
    assert not plot.available
    assert "contact" in plot.reason


def test_unavailable_engine_payload_has_reason_and_no_traces() -> None:
    plot = unavailable_ground_reaction_plot("Simscape does not report GRF (GCV-3)")
    assert not plot.available and plot.traces == {} and plot.time_s == ()
    payload = plot.to_dict()
    assert payload["available"] is False
    assert payload["reason"] == "Simscape does not report GRF (GCV-3)"
    with pytest.raises(ValueError, match="reason"):
        unavailable_ground_reaction_plot("  ")


def test_payload_is_strict_json_with_labels_in_title_case() -> None:
    plot = build_ground_reaction_plot_series(
        _series(), body_weight_n=BODY_WEIGHT_N, events={"impact": 0.2}
    )
    payload = json.loads(plot_series_to_json(plot))  # allow_nan=False
    assert payload["feet"] == ["left", "right"]
    assert payload["load_share"]["left"][3] is None
    assert set(payload["labels"]) == set(payload["traces"]) == set(payload["units"])
    assert payload["labels"]["net_moment_com_nm"] == "Net Moment About CoM"
    assert payload["labels"]["left_force_n"] == "Left Force"
    assert payload["units"]["left_cop_m"] == "m"
    assert payload["units"]["net_free_moment_nm"] == "N*m"
    assert isinstance(plot, GroundReactionPlotSeries)
