"""Reference-shot regression of the production ball-flight model (#15443).

Each row of ``reference_shots_15443.json`` combines published aggregate
launch and flight metrics; the published observations are not individual
matched shots or normalized to standard air. The prescribed launch goes into
:class:`BallFlightSimulator` (the Rust kernel the swing pipeline uses), so a
failure here is the flight model's, never the impact model's.

Rows the current model misses are marked ``xfail(strict=True)`` with the
measured error; the bands are never widened to make a row pass.  When the
aerodynamics change, a strict xfail that starts passing fails the suite and
the row must be promoted.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from src.shared.python.physics.ball_launch_conditions import LaunchConditions

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).with_name("reference_shots_15443.json")
_DATA: dict[str, Any] = json.loads(_FIXTURE.read_text(encoding="utf-8"))
_SETTINGS: dict[str, float] = _DATA["settings"]
_METRICS = ("carry", "apex", "descent")


def _flight(shot: dict[str, Any]) -> dict[str, float]:
    """Carry and apex [yd] and descent [deg] of one reference launch."""
    from src.shared.python.physics.rust_kernel import is_rust_available

    if not is_rust_available():
        raise RuntimeError("a current native flight kernel is required")
    from src.shared.python.physics.ball_simulator import BallFlightSimulator

    launch = LaunchConditions(
        velocity=shot["ball_speed_mph"] * _SETTINGS["mph_to_mps"],
        launch_angle=math.radians(shot["launch_deg"]),
        azimuth_angle=0.0,
        spin_rate=shot["spin_rpm"],
        spin_axis=np.array([0.0, -1.0, 0.0]),
    )
    points = BallFlightSimulator().simulate_trajectory(
        launch, max_time=_SETTINGS["max_time_s"], dt=_SETTINGS["dt_s"]
    )
    path = np.array([p.position for p in points], dtype=float)
    if (
        path.ndim != 2
        or path.shape[1:] != (3,)
        or len(path) < 3
        or not np.isfinite(path).all()
        or np.any(path[1:-1, 2] <= 0.0)
        or abs(path[-1, 2]) > 1e-8
        or not np.isfinite(points[-1].velocity).all()
        or points[-1].velocity[2] >= 0.0
    ):
        raise ValueError("flight did not terminate at descending ground contact")
    vx, vy, vz = points[-1].velocity
    yard = _SETTINGS["yard_m"]
    return {
        "carry": float(math.hypot(path[-1, 0], path[-1, 1])) / yard,
        "apex": float(path[:, 2].max()) / yard,
        "descent": math.degrees(math.atan2(-vz, math.hypot(vx, vy))),
    }


def _band(shot: dict[str, Any], metric: str) -> tuple[float, float]:
    """Accepted ``(low, high)`` for ``metric`` of ``shot`` from the settings."""
    if metric == "descent":
        ref, half = shot["descent_deg"], _SETTINGS["descent_band_deg"]
    else:
        ref = shot[f"{metric}_yd"]
        half = ref * _SETTINGS[f"{metric}_band_fraction"]
    return ref - half, ref + half


def _cases() -> list[Any]:
    cases = []
    for shot in _DATA["shots"]:
        for metric in _METRICS:
            reason = shot.get("xfail", {}).get(metric)
            marks = (
                [pytest.mark.xfail(strict=True, reason=reason, raises=AssertionError)]
                if reason
                else []
            )
            cases.append(
                pytest.param(shot, metric, id=f"{shot['id']}-{metric}", marks=marks)
            )
    return cases


def test_fixture_rows_carry_source_and_confidence() -> None:
    """Every reference row states where it comes from and how sure we are."""
    for shot in _DATA["shots"]:
        assert shot["source"].strip() and shot["confidence"].strip(), shot["id"]
        assert set(shot.get("xfail", {})) <= set(_METRICS), shot["id"]


def test_fixture_cites_a_fetched_published_table() -> None:
    """The fixture names its publisher, URLs, year and access date (#15443)."""
    source = _DATA["tour_table_source"]
    for key in ("title", "publisher", "year", "accessed", "notes"):
        assert source[key].strip(), key
    assert source["urls"] and all(u.startswith("https://") for u in source["urls"])
    for shot in _DATA["shots"]:
        assert "from memory" not in (shot["source"] + shot["confidence"]), shot["id"]


def test_expected_mismatches_only_accept_assertion_failures() -> None:
    marks = [mark for case in _cases() for mark in case.marks if mark.name == "xfail"]
    assert marks
    assert all(mark.kwargs.get("raises") is AssertionError for mark in marks)


def test_unavailable_kernel_is_an_error_not_a_skipped_flight(monkeypatch) -> None:
    from src.shared.python.physics import rust_kernel

    monkeypatch.setattr(rust_kernel, "is_rust_available", lambda: False)
    monkeypatch.setitem(sys.modules, "upstream_physics", None)
    with pytest.raises(RuntimeError, match="kernel"):
        _flight(_DATA["shots"][0])


def test_unlanded_trajectory_cannot_be_reported_as_carry(monkeypatch) -> None:
    from src.shared.python.physics import rust_kernel
    from src.shared.python.physics.ball_simulator import BallFlightSimulator

    monkeypatch.setattr(rust_kernel, "is_rust_available", lambda: True)
    points = [
        SimpleNamespace(position=np.array([0.0, 0.0, 0.0]), velocity=[20, 0, 10]),
        SimpleNamespace(position=np.array([10.0, 0.0, 2.0]), velocity=[10, 0, -2]),
    ]
    monkeypatch.setattr(
        BallFlightSimulator, "simulate_trajectory", lambda *a, **k: points
    )
    with pytest.raises(ValueError, match="ground"):
        _flight(_DATA["shots"][0])


@pytest.mark.parametrize(("shot", "metric"), _cases())
def test_reference_shot_within_band(shot: dict[str, Any], metric: str) -> None:
    """The simulated metric lies inside the published reference band."""
    value = _flight(shot)[metric]
    low, high = _band(shot, metric)
    assert low <= value <= high, (
        f"{shot['id']} {metric} = {value:.1f}, reference band [{low:.1f}, {high:.1f}]"
        f" ({shot['source']})"
    )
