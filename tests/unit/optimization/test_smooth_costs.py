"""Tests for the smooth cost surrogates (epic #8390, B1/#8396).

Acceptance criteria from the issue: smooth functions must have nonzero
finite-difference gradients across a sampled trajectory neighborhood, and
must agree in ranking with the legacy hard scores on separated cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.optimization._swing_models import SwingTrajectory
from src.shared.python.optimization._swing_objectives import compute_injury_risk
from src.shared.python.optimization.smooth_costs import (
    smooth_abs_max,
    smooth_injury_risk,
    smooth_kinematic_sequence_penalty,
    smooth_peak_time,
    softplus_excess,
)

pytestmark = pytest.mark.unit


def _trajectory(vel_scale: float, torque_scale: float, trunk_peak: float):
    t = np.linspace(0.0, 1.0, 50)
    vel = vel_scale * np.sin(np.pi * t)
    torque = torque_scale * np.sin(np.pi * t)
    trunk = trunk_peak * np.sin(np.pi * t)
    return SwingTrajectory(
        time=t,
        joint_angles={"trunk_rotation": trunk},
        joint_velocities={"hip_rotation": vel},
        joint_torques={"hip_rotation": torque},
        clubhead_position=np.zeros((50, 3)),
        clubhead_velocity=np.zeros((50, 3)),
    )


def _smooth_risk(traj: SwingTrajectory, limits: dict[str, float]) -> float:
    return smooth_injury_risk(
        traj.joint_velocities,
        traj.joint_torques,
        traj.joint_angles,
        limits,
    )


def test_softplus_excess_is_sigmoidal_around_threshold() -> None:
    assert softplus_excess(0.0, 10.0) < 0.01
    assert softplus_excess(10.0, 10.0) == pytest.approx(0.5)
    assert softplus_excess(20.0, 10.0) > 0.99


def test_smooth_abs_max_upper_bounds_true_max() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=100)
    smooth = smooth_abs_max(values)
    true_max = float(np.max(np.abs(values)))
    assert smooth >= true_max
    assert smooth == pytest.approx(true_max, abs=np.log(100) / 60.0)


def test_smooth_peak_time_matches_argmax_for_peaked_signal() -> None:
    t = np.linspace(0.0, 1.0, 200)
    v = np.exp(-((t - 0.62) ** 2) / 0.002)
    assert smooth_peak_time(t, v) == pytest.approx(0.62, abs=0.01)


def test_sequence_penalty_zero_for_correct_order_positive_for_violation() -> None:
    t = np.linspace(0.0, 1.0, 200)

    def peak_at(when: float) -> np.ndarray:
        return np.exp(-((t - when) ** 2) / 0.002)

    ordered = [peak_at(0.2), peak_at(0.4), peak_at(0.6)]
    violated = [peak_at(0.6), peak_at(0.4), peak_at(0.2)]
    good = smooth_kinematic_sequence_penalty(t, ordered)
    bad = smooth_kinematic_sequence_penalty(t, violated)
    # Softplus leaves a small smooth tail (~log(1+e^-k*sep)/k) on the good
    # ordering; what matters is separation from the violated case.
    assert good < 0.06
    assert bad > 5 * good


def test_smooth_risk_ranks_like_legacy_on_separated_cases() -> None:
    limits = {"hip_rotation": 100.0}
    calm = _trajectory(vel_scale=5.0, torque_scale=30.0, trunk_peak=0.5)
    risky = _trajectory(vel_scale=30.0, torque_scale=95.0, trunk_peak=1.6)

    legacy_calm = compute_injury_risk(calm, limits)
    legacy_risky = compute_injury_risk(risky, limits)
    smooth_calm = _smooth_risk(calm, limits)
    smooth_risky = _smooth_risk(risky, limits)

    assert legacy_risky > legacy_calm
    assert smooth_risky > smooth_calm
    # Deep-in-region cases agree with the hard scores to a few points.
    assert smooth_calm == pytest.approx(legacy_calm, abs=5.0)
    assert smooth_risky == pytest.approx(legacy_risky, abs=5.0)


def test_smooth_risk_has_nonzero_fd_gradient_near_threshold() -> None:
    """The whole point: the legacy score has zero gradient almost
    everywhere; the surrogate must not."""
    limits = {"hip_rotation": 100.0}
    eps = 1e-4
    grads = []
    for scale in (18.0, 20.0, 22.0):  # below / at / above the 20 rad/s step
        lo = _trajectory(scale - eps, 30.0, 0.5)
        hi = _trajectory(scale + eps, 30.0, 0.5)
        grad = (_smooth_risk(hi, limits) - _smooth_risk(lo, limits)) / (2 * eps)
        grads.append(grad)
    assert all(g > 0.0 for g in grads)

    # Legacy comparison at the same probe points: flat except astride the step.
    legacy_flat = compute_injury_risk(
        _trajectory(18.0 + eps, 30.0, 0.5), limits
    ) - compute_injury_risk(_trajectory(18.0 - eps, 30.0, 0.5), limits)
    assert legacy_flat == 0.0


def test_sequence_penalty_has_nonzero_fd_gradient() -> None:
    t = np.linspace(0.0, 1.0, 200)

    def traces(shift: float) -> list[np.ndarray]:
        return [
            np.exp(-((t - 0.5 - shift) ** 2) / 0.002),
            np.exp(-((t - 0.5) ** 2) / 0.002),
        ]

    eps = 1e-4
    for shift in (-0.05, 0.0, 0.05):
        lo = smooth_kinematic_sequence_penalty(t, traces(shift - eps))
        hi = smooth_kinematic_sequence_penalty(t, traces(shift + eps))
        assert (hi - lo) / (2 * eps) > 0.0


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="sharpness"):
        softplus_excess(1.0, 0.0, sharpness=0.0)
    with pytest.raises(ValueError, match="non-empty"):
        smooth_abs_max(np.array([]))
    with pytest.raises(ValueError, match="same length"):
        smooth_peak_time(np.array([0.0, 1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="two segments"):
        smooth_kinematic_sequence_penalty(np.array([0.0]), [np.array([1.0])])
