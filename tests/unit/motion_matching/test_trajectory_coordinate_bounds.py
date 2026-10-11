"""Trajectory IK must preserve explicit source bounds without clipping seeds."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.full_body_ik import (
    solve_full_body_ik_trajectory,
)
from src.shared.python.motion_matching.marker_calibration import Array, Pose
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

pytestmark = pytest.mark.unit


def _capture() -> TourCapture:
    return TourCapture(
        np.array([0.0]),
        ("marker",),
        np.array([[[2.0, 0.0, 0.0]]]),
        np.ones((1, 1), dtype=bool),
        "a" * 64,
    )


def test_bounded_trajectory_keeps_unreachable_residual_and_native_limits() -> None:
    evaluated = []

    def poses(q: Array) -> dict[str, Pose]:
        assert -0.5 <= q[0] <= 0.5
        evaluated.append(q[0])
        return {"body": (np.eye(3), np.array([q[0], 0.0, 0.0]))}

    result = solve_full_body_ik_trajectory(
        poses,
        {"marker": ("body", (0.0, 0.0, 0.0))},
        _capture(),
        np.array([0.0]),
        reg_weight=0,
        max_nfev=30,
        coordinate_bounds=(np.array([-0.5]), np.array([0.5])),
    )
    assert evaluated
    assert result[0, 0] == pytest.approx(0.5, abs=1e-7)
    assert 2.0 - result[0, 0] >= 1.5


@pytest.mark.parametrize(
    "low, high, seed",
    [
        ([-0.5], [0.5], [0.6]),
        ([0.5], [0.5], [0.5]),
        ([np.nan], [0.5], [0.0]),
        ([-np.inf], [0.5], [0.0]),
        ([-0.5, -0.5], [0.5], [0.0]),
    ],
)
def test_invalid_bounds_reject_before_geometry(
    low: list[float],
    high: list[float],
    seed: list[float],
) -> None:
    def unexpected(q: Array) -> dict[str, Pose]:
        pytest.fail("Invalid source bounds reached geometry")

    with pytest.raises(ValueError, match="bounds"):
        solve_full_body_ik_trajectory(
            unexpected,
            {"marker": ("body", (0.0, 0.0, 0.0))},
            _capture(),
            np.array(seed),
            coordinate_bounds=(np.array(low), np.array(high)),
        )


def test_default_trajectory_retains_unbounded_lm_behavior() -> None:
    def poses(q: Array) -> dict[str, Pose]:
        return {"body": (np.eye(3), np.array([q[0], 0.0, 0.0]))}

    result = solve_full_body_ik_trajectory(
        poses,
        {"marker": ("body", (0.0, 0.0, 0.0))},
        _capture(),
        np.array([0.0]),
        reg_weight=0,
    )
    assert result[0, 0] == pytest.approx(2.0)


def test_bounded_trajectory_moves_from_an_active_seed_bound() -> None:
    def poses(q: Array) -> dict[str, Pose]:
        return {"body": (np.eye(3), np.array([q[0], 0.0, 0.0]))}

    result = solve_full_body_ik_trajectory(
        poses,
        {"marker": ("body", (0.0, 0.0, 0.0))},
        _capture(),
        np.array([0.0]),
        reg_weight=0,
        max_nfev=50,
        coordinate_bounds=(np.array([0.0]), np.array([3.0])),
    )
    assert result[0, 0] == pytest.approx(2.0, abs=1e-6)
