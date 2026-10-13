"""Unit tests for floating-base manifold spline on SE(3) (#11536).

Verifies:
- Exponential and logarithm maps on SE(3) with Taylor stability.
- Retraction and local-coordinates Jacobians against numerical differentiation
  at nonzero velocities and away from identity.
- Spline evaluation of pose, velocity, and acceleration agreeing with
  finite differences on SE(3) for known screw motions.
- Degenerate planar motion matching PlanarChain oracle.
- Outer solve on the manifold recovering a floating-base trajectory from
  noisy synthetic marker observations, reporting convergence and iterations.
- Basis-resolution (whiteness at gamma = 0) diagnostics for qualification gate 1.
- Precondition enforcement via core.contracts.require.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import pytest

from src.shared.python.core.contracts import PreconditionError
from src.shared.python.estimation.mosaic.kinematic_basis import BSplineBasis
from src.shared.python.estimation.mosaic.manifold_spline import (
    ManifoldFitResult,
    SE3ManifoldSpline,
    WhitenessDiagnostics,
    compute_whiteness_diagnostics,
    fit_floating_base_manifold,
    matrix_to_pose7,
    pose7_to_matrix,
    se3_exp,
    se3_left_jacobian,
    se3_left_jacobian_inverse,
    se3_local_coordinates,
    se3_local_coordinates_jacobian,
    se3_log,
    se3_retract,
    se3_retract_jacobian,
)
from src.shared.python.estimation.mosaic.planar_chain import (
    PlanarChain,
    PlanarMarkerSet,
)

H_FD = 1e-6
TOL_FD = 1e-7
SEED = 11536


def _central_difference(
    func: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, h: float = H_FD
) -> np.ndarray:
    """Central finite-difference Jacobian of ``func`` at ``x0``."""
    f0 = np.asarray(func(x0))
    jac = np.zeros((f0.size, x0.size), dtype=np.float64)
    for k in range(x0.size):
        step = np.zeros_like(x0)
        step[k] = h
        jac[:, k] = (np.asarray(func(x0 + step)) - np.asarray(func(x0 - step))) / (
            2.0 * h
        )
    return jac


def _random_unit_quaternion(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


def _random_se3_matrix(rng: np.random.Generator) -> np.ndarray:
    p = rng.normal(scale=1.0, size=3)
    q = _random_unit_quaternion(rng)
    pose7 = np.concatenate([p, q])
    return pose7_to_matrix(pose7)


# ---------------------------------------------------------------------------
# Lie Group Maps & Jacobians
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_se3_exponential_and_logarithm_roundtrip() -> None:
    rng = np.random.default_rng(SEED)
    angles = [1e-8, 1e-5, 1e-3, 0.4, 1.2, 2.5, np.pi - 1e-3]
    for angle in angles:
        for _ in range(3):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            phi = angle * axis
            rho = rng.normal(size=3)
            xi = np.concatenate([rho, phi])

            T = se3_exp(xi)
            xi_recovered = se3_log(T)
            np.testing.assert_allclose(xi_recovered, xi, atol=1e-10)

            T_recovered = se3_exp(xi_recovered)
            np.testing.assert_allclose(T_recovered, T, atol=1e-10)


@pytest.mark.unit
def test_se3_retract_jacobian_matches_finite_differences() -> None:
    rng = np.random.default_rng(SEED + 1)
    angles = [1e-5, 1e-3, 0.4, 1.5, 2.7]
    for angle in angles:
        for _ in range(3):
            T = _random_se3_matrix(rng)
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            delta_xi = np.concatenate([rng.normal(size=3), angle * axis])

            def f_retract_p7(d: np.ndarray, base_T: np.ndarray = T) -> np.ndarray:
                T_new = se3_retract(base_T, d)
                return matrix_to_pose7(T_new)

            expected = _central_difference(f_retract_p7, delta_xi, h=H_FD)
            actual = se3_retract_jacobian(T, delta_xi)
            assert actual.shape == (7, 6)
            np.testing.assert_allclose(actual, expected, atol=TOL_FD, rtol=0.0)


@pytest.mark.unit
def test_se3_local_coordinates_jacobian_matches_finite_differences() -> None:
    rng = np.random.default_rng(SEED + 2)
    zero6 = np.zeros(6)
    angles = [1e-4, 1e-2, 0.3, 1.4, 2.4]
    for angle in angles:
        for _ in range(3):
            T0 = _random_se3_matrix(rng)
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            rel_twist = np.concatenate([rng.normal(size=3), angle * axis])
            T1 = se3_retract(T0, rel_twist)

            def f_local(
                d: np.ndarray, base0: np.ndarray = T0, base1: np.ndarray = T1
            ) -> np.ndarray:
                T1_pert = se3_retract(base1, d)
                return se3_local_coordinates(base0, T1_pert)

            expected = _central_difference(f_local, zero6, h=H_FD)
            actual = se3_local_coordinates_jacobian(T0, T1)
            assert actual.shape == (6, 6)
            np.testing.assert_allclose(actual, expected, atol=TOL_FD, rtol=0.0)


@pytest.mark.unit
def test_se3_local_coordinates_respects_quaternion_sign_equivalence() -> None:
    rng = np.random.default_rng(SEED + 3)
    T0 = _random_se3_matrix(rng)
    twist = np.array([0.2, -0.1, 0.4, 0.3, -0.5, 0.2])
    T1 = se3_retract(T0, twist)

    p7_1 = matrix_to_pose7(T1)
    p7_1_neg = p7_1.copy()
    p7_1_neg[3:] = -p7_1[3:]
    T1_neg = pose7_to_matrix(p7_1_neg)

    local_1 = se3_local_coordinates(T0, T1)
    local_neg = se3_local_coordinates(T0, T1_neg)
    np.testing.assert_allclose(local_neg, local_1, atol=1e-12)

    jac_1 = se3_local_coordinates_jacobian(T0, T1)
    jac_neg = se3_local_coordinates_jacobian(T0, T1_neg)
    np.testing.assert_allclose(jac_neg, jac_1, atol=1e-12)


# ---------------------------------------------------------------------------
# Manifold Spline Kinematics & Screw Motion
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_manifold_spline_screw_motion_matches_finite_differences() -> None:
    times = np.linspace(0.0, 1.0, 51)
    twist = np.array([0.5, -0.2, 0.3, 0.4, 0.1, -0.6])
    T0 = np.eye(4)
    T0[:3, 3] = np.array([0.1, 0.2, -0.1])

    spline = SE3ManifoldSpline.from_screw_motion(
        times=times, T0=T0, twist=twist, n_coefficients=15, degree=5
    )

    poses, vels, accs = spline.evaluate()

    # Analytical truth for screw motion:
    for i, t in enumerate(times):
        T_expected = se3_exp(t * twist) @ T0
        np.testing.assert_allclose(poses[i], T_expected, atol=1e-10)

    # Constant velocity:
    np.testing.assert_allclose(vels, np.tile(twist, (times.size, 1)), atol=1e-7)
    # Zero acceleration:
    np.testing.assert_allclose(accs, np.zeros_like(accs), atol=1e-7)

    # Compare with high-accuracy central finite differences of pose along the spline
    h = 1e-5
    for idx in range(1, times.size - 1):
        t = times[idx]
        T_plus = spline.evaluate_at(t + h)
        T_minus = spline.evaluate_at(t - h)
        v_fd = se3_log(T_plus @ np.linalg.inv(T_minus)) / (2.0 * h)
        np.testing.assert_allclose(vels[idx], v_fd, atol=1e-6)


@pytest.mark.unit
def test_manifold_spline_generic_trajectory_derivatives() -> None:
    times = np.linspace(0.0, 1.0, 41)
    rng = np.random.default_rng(SEED + 4)
    n_ctrl = 12
    coeffs = rng.normal(scale=0.1, size=(n_ctrl, 6))

    basis = BSplineBasis.uniform(times, n_coefficients=n_ctrl, degree=5)
    T_ref = np.repeat(np.eye(4)[None, ...], times.size, axis=0)
    spline = SE3ManifoldSpline(basis=basis, reference_poses=T_ref, coefficients=coeffs)

    poses, vels, accs = spline.evaluate()

    # Verify velocities against numerical derivative of pose at intermediate nodes
    h = 1e-5
    for idx in range(2, times.size - 2):
        t = times[idx]
        T_plus = spline.evaluate_at(t + h)
        T_minus = spline.evaluate_at(t - h)
        v_fd = se3_log(T_plus @ np.linalg.inv(T_minus)) / (2.0 * h)
        np.testing.assert_allclose(vels[idx], v_fd, atol=1e-6)

        # Acceleration against numerical derivative of velocity
        v_plus = spline.evaluate_velocity_at(t + h)
        v_minus = spline.evaluate_velocity_at(t - h)
        a_fd = (v_plus - v_minus) / (2.0 * h)
        np.testing.assert_allclose(accs[idx], a_fd, atol=1e-6)


@pytest.mark.unit
def test_manifold_spline_retract_updates_poses() -> None:
    times = np.linspace(0.0, 1.0, 31)
    basis = BSplineBasis.uniform(times, n_coefficients=10, degree=5)
    T_ref = np.repeat(np.eye(4)[None, ...], times.size, axis=0)
    spline = SE3ManifoldSpline(basis=basis, reference_poses=T_ref)

    delta_C = np.ones((10, 6)) * 0.05
    spline_retracted = spline.retract(delta_C)

    np.testing.assert_allclose(spline_retracted.coefficients, np.zeros((10, 6)))
    poses_retracted = spline_retracted.evaluate_poses()

    xi_expected = basis.position @ delta_C
    for i in range(times.size):
        T_exp = se3_exp(xi_expected[i])
        np.testing.assert_allclose(poses_retracted[i], T_exp, atol=1e-12)


# ---------------------------------------------------------------------------
# Degenerate Planar Case matches PlanarChain
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_degenerate_planar_case_matches_planar_chain() -> None:
    # Build 1-link planar chain (revolute joint around z at origin)
    chain = PlanarChain(
        link_lengths=np.array([1.0]),
        actuated=np.array([True]),
        gravity=np.array([0.0, -9.81]),
    )
    marker_offsets = np.array([[0.5, 0.0], [1.0, 0.0]])
    markers = PlanarMarkerSet(
        link_index=np.array([0, 0], dtype=np.int64),
        offsets=marker_offsets,
    )

    times = np.linspace(0.0, 1.0, 21)
    # Planar motion: theta(t) rotates around z, base pinned at origin
    theta = 0.5 * np.sin(2.0 * np.pi * times)
    q_planar = theta[:, None]

    markers_planar = chain.marker_positions(q_planar, markers)

    # 3D representation via SE3ManifoldSpline
    # Rotation about z: phi = [0, 0, theta]
    poses_3d = np.zeros((times.size, 4, 4))
    for i in range(times.size):
        th = theta[i]
        poses_3d[i] = np.array(
            [
                [np.cos(th), -np.sin(th), 0.0, 0.0],
                [np.sin(th), np.cos(th), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    spline = SE3ManifoldSpline.from_poses(
        times=times, poses=poses_3d, n_coefficients=10, degree=5
    )
    poses_spline = spline.evaluate_poses()

    offsets_3d = np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    for i in range(times.size):
        R = poses_spline[i, :3, :3]
        p = poses_spline[i, :3, 3]
        for m in range(2):
            pos_3d = R @ offsets_3d[m] + p
            np.testing.assert_allclose(pos_3d[:2], markers_planar[i, m], atol=1e-12)
            assert abs(pos_3d[2]) < 1e-12


# ---------------------------------------------------------------------------
# Whiteness Diagnostics for Gate 1
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_whiteness_diagnostics_for_gate_1() -> None:
    rng = np.random.default_rng(SEED + 5)
    sigma = 0.01

    # Case 1: Pure white noise matching sigma
    white_noise = rng.normal(0.0, sigma, size=(200, 4, 3))
    diag_white = compute_whiteness_diagnostics(white_noise, sigma)
    assert 0.9 <= diag_white.rms <= 1.1
    assert abs(diag_white.lag1_autocorrelation) < 0.15
    assert diag_white.passed

    # Case 2: Colored / correlated noise (under-resolved basis signature)
    colored = np.zeros((200, 4, 3))
    for t in range(1, 200):
        colored[t] = 0.9 * colored[t - 1] + rng.normal(0.0, sigma, size=(4, 3))
    diag_colored = compute_whiteness_diagnostics(colored, sigma)
    assert diag_colored.lag1_autocorrelation > 0.5
    assert not diag_colored.passed


# ---------------------------------------------------------------------------
# Outer Solve on the Manifold
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_outer_solve_manifold_recovers_synthetic_floating_base_trajectory() -> None:
    rng = np.random.default_rng(SEED + 6)
    times = np.linspace(0.0, 1.0, 51)
    twist = np.array([0.3, 0.1, -0.2, 0.5, -0.2, 0.4])
    T0 = np.eye(4)
    T0[:3, 3] = np.array([0.2, -0.3, 0.1])

    # Ground truth trajectory
    true_poses = np.zeros((times.size, 4, 4))
    for i, t in enumerate(times):
        true_poses[i] = se3_exp(t * twist) @ T0

    # Markers in body frame
    marker_offsets = np.array(
        [
            [0.2, 0.0, 0.0],
            [-0.2, 0.1, 0.0],
            [0.0, -0.2, 0.1],
            [0.1, 0.1, -0.1],
        ]
    )
    marker_sigma = 5e-4  # 0.5 mm

    # True marker trajectories + noise
    clean_markers = np.zeros((times.size, 4, 3))
    for i in range(times.size):
        clean_markers[i] = (
            marker_offsets @ true_poses[i, :3, :3].T + true_poses[i, :3, 3]
        )
    noisy_markers = clean_markers + rng.normal(
        0.0, marker_sigma, size=clean_markers.shape
    )

    # Initial guess: perturbed from truth
    perturbed_poses = true_poses.copy()
    for i in range(times.size):
        pert = rng.normal(scale=0.03, size=6)
        perturbed_poses[i] = se3_retract(perturbed_poses[i], pert)

    init_spline = SE3ManifoldSpline.from_poses(
        times=times, poses=perturbed_poses, n_coefficients=15, degree=5
    )

    # Run outer solve on the manifold
    result: ManifoldFitResult = fit_floating_base_manifold(
        spline=init_spline,
        marker_positions=noisy_markers,
        marker_offsets=marker_offsets,
        marker_sigma=marker_sigma,
        max_iterations=25,
        relative_cost_tolerance=1e-8,
    )

    assert result.converged
    assert result.iterations > 0
    assert result.iterations <= 25
    assert result.final_cost < result.cost_history[0]

    # Verify trajectory recovery
    fitted_poses = result.spline.evaluate_poses()
    for i in range(times.size):
        # Position error < 1 mm
        pos_err = np.linalg.norm(fitted_poses[i, :3, 3] - true_poses[i, :3, 3])
        assert pos_err < 2e-3
        # Rotation error < 0.01 rad
        rot_err = np.linalg.norm(
            se3_local_coordinates(true_poses[i], fitted_poses[i])[3:]
        )
        assert rot_err < 1e-2

    # Verify whiteness diagnostic
    assert result.whiteness.passed


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_preconditions_reject_invalid_inputs() -> None:
    with pytest.raises(PreconditionError):
        se3_exp(np.zeros(5))

    with pytest.raises(PreconditionError):
        se3_log(np.zeros((3, 3)))

    with pytest.raises(PreconditionError):
        se3_retract(np.eye(4), np.zeros(7))

    with pytest.raises(PreconditionError):
        se3_local_coordinates(np.eye(4), np.zeros(7))

    with pytest.raises(PreconditionError):
        se3_retract_jacobian(np.eye(4), np.zeros(5))

    with pytest.raises(PreconditionError):
        se3_local_coordinates_jacobian(np.eye(4), np.zeros((4, 3)))
