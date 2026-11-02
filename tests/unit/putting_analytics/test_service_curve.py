"""JSON service for the simulated strokes-gained putting curve (#14177)."""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.putting_analytics import service_curve
from src.shared.python.putting_analytics.sg_curve import simulate_putting_curve
from src.shared.python.putting_analytics.service_variance import profile_from_args
from src.shared.python.sg_optimizer.shot_model.putting_provider import (
    REFERENCE_STIMP,
    SimulatedPuttingCurve,
    putting_provider,
)

pytestmark = pytest.mark.unit


def _args(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "distances_ft": [3.0, 8.0, 20.0],
        "face_sd_deg": 1.5,
        "speed_cv": 0.08,
        "n": 200,
        "seed": 5,
    }
    return {**args, **overrides}


def test_curve_matches_the_simulation_for_the_same_player() -> None:
    report = service_curve.curve_study(_args())
    expected = simulate_putting_curve(
        profile_from_args(_args()), [3.0, 8.0, 20.0], n=200, seed=5
    )
    assert report["distances_ft"] == list(expected.distances_ft)
    assert report["make_rate"] == list(expected.make_rate)
    assert report["leave_scale_ft"] == list(expected.leave_scale_ft)
    assert report["leave_sigma_log"] == list(expected.leave_sigma_log)
    assert report["samples_per_distance"] == 200
    assert report["reference_stimp"] == REFERENCE_STIMP


def test_report_round_trips_into_the_optimizer_provider() -> None:
    report = service_curve.curve_study(_args())
    curve = service_curve.curve_from_report(report)
    assert isinstance(curve, SimulatedPuttingCurve)
    provider = putting_provider("simulation", curve)
    assert provider is not None
    assert curve.make_rate_at(3.0) == pytest.approx(report["make_rate"][0])


def test_default_grid_is_the_optimizer_grid() -> None:
    report = service_curve.curve_study({"n": 40, "seed": 1})
    assert report["distances_ft"] == list(service_curve.DEFAULT_CURVE_GRID_FT)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"distances_ft": [3.0]}, "at least 2"),
        ({"distances_ft": [8.0, 3.0]}, "increase"),
        ({"distances_ft": [0.0, 3.0]}, "distances_ft"),
        ({"distances_ft": [3.0, 500.0]}, "distances_ft"),
        ({"distances_ft": "far"}, "list"),
        ({"n": 5}, "'n'"),
        ({"n": 20.5}, "whole number"),
        ({"seed": -1}, "seed"),
        ({"face_sd_deg": 99.0}, "face_sd_deg"),
    ],
)
def test_rejects_bad_arguments(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        service_curve.curve_study(_args(**overrides))


def test_work_is_capped() -> None:
    grid = [float(d) for d in range(2, 32)]
    with pytest.raises(ValueError, match="cap is"):
        service_curve.curve_study(_args(distances_ft=grid, n=service_curve.MAX_CURVE_N))
