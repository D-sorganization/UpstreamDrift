"""Unit tests for the pure Crocoddyl problem assembly (no engine required).

MS-31 (#10338): the assembly module turns a capture window and a full-body
document into node targets, bounds and warm-start helpers that the native
Crocoddyl fit consumes. Everything here is plain numpy.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.crocoddyl_problem import (
    FitHorizon,
    FitWeights,
    actuated_mask,
    build_marker_targets,
    coordinate_bounds,
    finite_difference_rates,
    least_squares_controls,
    per_coordinate_effort_bounds,
    range_barrier,
)

pytestmark = pytest.mark.unit


def test_horizon_nodes_are_capture_frames() -> None:
    horizon = FitHorizon(t_start_s=0.0, t_end_s=0.85, dt_s=1.0 / 360.0)
    times = horizon.node_times()
    assert times[0] == 0.0
    assert times.shape == (307,)
    assert np.allclose(np.diff(times), 1.0 / 360.0)


def test_horizon_rejects_empty_window() -> None:
    with pytest.raises(ValueError):
        FitHorizon(t_start_s=0.5, t_end_s=0.5, dt_s=1.0 / 360.0)


def test_weights_are_frozen_and_positive() -> None:
    weights = FitWeights()
    assert weights.marker > 0 and weights.effort > 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        weights.marker = 2.0  # type: ignore[misc]


def test_marker_targets_sample_exact_frames() -> None:
    rate = 360.0
    frames = 12
    time_s = np.arange(frames) / rate
    points = np.zeros((frames, 2, 3))
    points[:, 0, 0] = np.arange(frames)  # marker 0 moves along x
    points[:, 1, :] = np.nan
    valid = np.ones((frames, 2), dtype=bool)
    valid[:, 1] = False
    valid[5, 0] = False
    horizon = FitHorizon(t_start_s=0.0, t_end_s=10.0 / rate, dt_s=1.0 / rate)
    targets = build_marker_targets(
        time_s, points, valid, ("A", "B"), horizon.node_times()
    )
    assert targets.targets.shape == (11, 2, 3)
    assert targets.valid.shape == (11, 2)
    assert targets.targets[3, 0, 0] == 3.0
    assert not targets.valid[5, 0]
    assert not targets.valid[:, 1].any()
    # Invalid samples are zeroed, never NaN, so a masked cost stays finite.
    assert np.isfinite(targets.targets).all()


def test_marker_targets_reject_off_grid_nodes() -> None:
    time_s = np.arange(4) / 360.0
    points = np.zeros((4, 1, 3))
    valid = np.ones((4, 1), dtype=bool)
    with pytest.raises(ValueError):
        build_marker_targets(time_s, points, valid, ("A",), np.array([0.0, 0.0011]))


def test_coordinate_bounds_from_document_ranges() -> None:
    spec = {"coordinate_ranges_deg": {"knee_angle_r": [-5.0, 140.0]}}
    lower, upper = coordinate_bounds(spec, ("TranslationInputX", "knee_angle_r"))
    assert np.isneginf(lower[0]) and np.isposinf(upper[0])
    assert lower[1] == pytest.approx(np.deg2rad(-5.0))
    assert upper[1] == pytest.approx(np.deg2rad(140.0))


def test_actuated_mask_excludes_root() -> None:
    order = (
        "TranslationInputX",
        "TranslationInputY",
        "TranslationInputZ",
        "HipInputX",
        "HipInputY",
        "HipInputZ",
        "TorsoInput",
        "knee_angle_r",
    )
    mask = actuated_mask(order)
    assert mask.tolist() == [False] * 6 + [True, True]


def test_least_squares_controls_recover_known_efforts() -> None:
    rng = np.random.default_rng(0)
    n = 6
    deffort = rng.normal(size=(n, n))
    deffort[:, :2] = 0.0  # unactuated columns
    mask = np.array([False, False, True, True, True, True])
    u_true = rng.normal(size=4)
    a_zero = rng.normal(size=n)
    a_ref = a_zero + deffort[:, mask] @ u_true
    u = least_squares_controls(a_ref, a_zero, deffort, mask)
    assert np.allclose(u, u_true, atol=1e-9)


def test_finite_difference_rates_match_linear_motion() -> None:
    dt = 0.01
    q = np.outer(np.arange(10) * dt, np.array([1.0, -2.0]))
    v = finite_difference_rates(q, dt)
    assert v.shape == q.shape
    assert np.allclose(v, np.array([1.0, -2.0]))


def test_range_barrier_is_zero_inside_and_quadratic_outside() -> None:
    lower = np.array([-1.0, -np.inf])
    upper = np.array([1.0, np.inf])
    cost, grad, hess = range_barrier(np.array([0.5, 3.0]), lower, upper, 10.0)
    assert cost == 0.0 and np.all(grad == 0.0) and np.all(hess == 0.0)
    cost, grad, hess = range_barrier(np.array([1.5, 3.0]), lower, upper, 10.0)
    assert cost == pytest.approx(0.5 * 10.0 * 0.25)
    assert grad[0] == pytest.approx(10.0 * 0.5) and grad[1] == 0.0
    assert hess[0] == pytest.approx(10.0) and hess[1] == 0.0


def test_per_coordinate_effort_bounds_follow_patterns() -> None:
    order = ("TranslationInputX", "mtp_angle_l", "knee_angle_r", "LSInputZ", "custom")
    actuated = np.array([False, True, True, True, True])
    bounds = per_coordinate_effort_bounds(order, actuated, 600.0)
    assert bounds.tolist() == [10.0, 250.0, 120.0, 600.0]


def test_ridge_damps_near_singular_sensitivity() -> None:
    deffort = np.diag([1.0, 1e-6])
    actuated = np.array([True, True])
    a_ref = np.array([1.0, 1.0])
    plain = least_squares_controls(a_ref, np.zeros(2), deffort, actuated)
    damped = least_squares_controls(a_ref, np.zeros(2), deffort, actuated, ridge=1e-3)
    assert plain[1] > 1e5
    assert abs(damped[1]) < 1.0
    assert abs(damped[0] - a_ref[0]) < 1e-2
