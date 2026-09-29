"""Unit tests for piecewise polynomial torque and multi-phase stitching (#9921)."""

import numpy as np
import pytest

from src.shared.python.motion_matching.piecewise_polynomial import (
    PiecewisePolynomialTorque,
    PolynomialSegment,
    stitch_two_phase_trajectories,
)

pytestmark = pytest.mark.unit


def test_polynomial_segment_bernstein_eval_and_endpoints() -> None:
    # 2 channels, degree 6
    coeffs = np.array(
        [
            [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
            [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        ],
        dtype=float,
    )
    seg = PolynomialSegment(
        start_s=0.0, end_s=0.85, coefficients=coeffs, is_bernstein=True
    )
    assert seg.duration_s == pytest.approx(0.85)
    assert seg.degree == 6
    assert seg.n_channels == 2

    # In Bernstein basis on [0, T], tau(0) = c_0, tau(T) = c_d
    np.testing.assert_allclose(seg.evaluate(0.0), coeffs[:, 0])
    np.testing.assert_allclose(seg.evaluate(0.85), coeffs[:, -1])

    # Grid evaluation
    grid = np.linspace(0.0, 0.85, 11)
    values = seg.evaluate_grid(grid)
    assert values.shape == (11, 2)
    np.testing.assert_allclose(values[0], coeffs[:, 0])
    np.testing.assert_allclose(values[-1], coeffs[:, -1])


def test_polynomial_segment_power_basis() -> None:
    # Constant + linear: p0 + p1 * s, where s = (t - start) / dur
    coeffs = np.array([[3.0, 10.0], [5.0, -4.0]], dtype=float)
    seg = PolynomialSegment(
        start_s=1.0, end_s=3.0, coefficients=coeffs, is_bernstein=False
    )
    assert seg.duration_s == 2.0
    # At t=1.0, s=0 => [3.0, 5.0]
    np.testing.assert_allclose(seg.evaluate(1.0), [3.0, 5.0])
    # At t=3.0, s=1 => [13.0, 1.0]
    np.testing.assert_allclose(seg.evaluate(3.0), [13.0, 1.0])
    # At t=2.0, s=0.5 => [8.0, 3.0]
    np.testing.assert_allclose(seg.evaluate(2.0), [8.0, 3.0])


def test_polynomial_segment_derivative() -> None:
    # Linear Bernstein: c0=2, c1=8 on [0, 2].
    # Slope is (8 - 2) / 2 = 3.0
    coeffs = np.array([[2.0, 8.0]], dtype=float)
    seg = PolynomialSegment(
        start_s=0.0, end_s=2.0, coefficients=coeffs, is_bernstein=True
    )
    dtau = seg.evaluate_derivative(1.0)
    np.testing.assert_allclose(dtau, [3.0])


def test_stitch_two_phase_trajectories_c0_continuity() -> None:
    t_top = 0.85
    t_final = 1.814
    # Backswing: 2 channels, degree 6
    back_coeffs = np.random.RandomState(42).uniform(-50, 50, (2, 7))
    backswing = PolynomialSegment(0.0, t_top, back_coeffs, is_bernstein=True)

    # Downswing initial random controls
    down_coeffs = np.random.RandomState(43).uniform(-50, 50, (2, 7))
    downswing = PolynomialSegment(t_top, t_final, down_coeffs, is_bernstein=True)

    # Stitch with C0 enforcement
    stitched = stitch_two_phase_trajectories(
        backswing, downswing, enforce_c0=True, enforce_c1=False
    )
    assert stitched.duration_s == pytest.approx(t_final)
    assert stitched.n_segments == 2
    assert stitched.breakpoints == (0.0, t_top, t_final)

    # Check continuity
    cont = stitched.check_continuity(tol=1e-8)
    assert cont["c0_continuous"]
    assert cont["max_c0_jump"] < 1e-8


def test_stitch_two_phase_trajectories_c1_continuity() -> None:
    t_top = 0.85
    t_final = 1.814
    back_coeffs = np.random.RandomState(44).uniform(-50, 50, (3, 7))
    backswing = PolynomialSegment(0.0, t_top, back_coeffs, is_bernstein=True)

    down_coeffs = np.random.RandomState(45).uniform(-50, 50, (3, 7))
    downswing = PolynomialSegment(t_top, t_final, down_coeffs, is_bernstein=True)

    stitched = stitch_two_phase_trajectories(
        backswing, downswing, enforce_c0=True, enforce_c1=True
    )
    cont = stitched.check_continuity(tol=1e-8)
    assert cont["c0_continuous"]
    assert cont["c1_continuous"]
    assert cont["max_c0_jump"] < 1e-8
    assert cont["max_c1_jump"] < 1e-8


def test_project_to_global_polynomial_exact_endpoints() -> None:
    # Create smooth two-phase composite
    t_top = 0.85
    t_final = 1.814
    back_coeffs = np.tile(np.linspace(10, 50, 7), (2, 1))
    backswing = PolynomialSegment(0.0, t_top, back_coeffs, is_bernstein=True)
    down_coeffs = np.tile(np.linspace(50, 10, 7), (2, 1))
    downswing = PolynomialSegment(t_top, t_final, down_coeffs, is_bernstein=True)

    stitched = stitch_two_phase_trajectories(backswing, downswing, enforce_c0=True)

    # Project to single degree 6 polynomial with pinned endpoints
    global_coeffs = stitched.project_to_global_polynomial(degree=6, pin_endpoints=True)
    assert global_coeffs.shape == (2, 7)

    # Check exact match at s=0 and s=1
    tau_start = stitched.evaluate(0.0)
    tau_end = stitched.evaluate(t_final)

    # At s=0, sum c_k * 0^k = c_0
    np.testing.assert_allclose(global_coeffs[:, 0], tau_start, atol=1e-8)
    # At s=1, sum c_k = tau_end
    np.testing.assert_allclose(np.sum(global_coeffs, axis=1), tau_end, atol=1e-8)
