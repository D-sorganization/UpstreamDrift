"""Tests for rigid swing plane calibration (TB-03 #10588)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.tour_baselines.plane import (
    DegeneratePlaneError,
    PlaneFitDiagnostics,
    RigidSwingPlane,
    fit_rigid_swing_plane,
)

pytestmark = pytest.mark.unit


def test_rigid_plane_known_tilted_synthetic_roundtrip() -> None:
    """A known tilted synthetic plane recovers normal, gravity, and round-trip points exactly."""
    # Define a 45-degree tilted plane around X axis (Y-up world)
    # Normal n = [0, 1/sqrt(2), 1/sqrt(2)]
    # In-plane u = [1, 0, 0]
    # In-plane v = [0, -1/sqrt(2), 1/sqrt(2)]
    # Origin P0 = [0, 1.0, 0]
    origin = np.array([0.0, 1.0, 0.0])
    u_axis = np.array([1.0, 0.0, 0.0])
    v_axis = np.array([0.0, 1.0 / math.sqrt(2), -1.0 / math.sqrt(2)])
    n_axis = np.cross(u_axis, v_axis)
    n_axis = n_axis / np.linalg.norm(n_axis)

    plane = RigidSwingPlane(
        origin=origin,
        normal=n_axis,
        u_axis=u_axis,
        v_axis=v_axis,
    )

    # Test round-trip for arbitrary in-plane and out-of-plane 3D points
    test_points_3d = np.array(
        [
            [0.5, 1.2, -0.2],
            [-0.3, 0.8, 0.4],
            [1.0, 1.5, -0.5],
        ]
    )

    coords_2d, residuals = plane.project_points(test_points_3d)
    assert coords_2d.shape == (3, 2)
    assert residuals.shape == (3,)

    # Unproject back to 3D with residuals
    recovered_3d = plane.unproject_points(coords_2d, residuals)
    np.testing.assert_allclose(recovered_3d, test_points_3d, atol=1e-12)

    # In-plane points have zero residual
    pure_in_plane_3d = origin + 0.3 * u_axis - 0.4 * v_axis
    p2d, res = plane.project_point(pure_in_plane_3d)
    assert abs(res) < 1e-12
    np.testing.assert_allclose(p2d, [0.3, -0.4], atol=1e-12)
    np.testing.assert_allclose(plane.unproject_point(p2d), pure_in_plane_3d, atol=1e-12)

    # Gravity projection in Y-up world
    g_world = np.array([0.0, -9.81, 0.0])
    g_plane = plane.transform_gravity(g_world)
    assert np.isfinite(g_plane.g_u)
    assert np.isfinite(g_plane.g_v)
    assert np.isfinite(g_plane.g_normal)
    # Total magnitude must be preserved
    total_g_mag = math.sqrt(g_plane.g_u**2 + g_plane.g_v**2 + g_plane.g_normal**2)
    assert abs(total_g_mag - 9.81) < 1e-9


def test_fit_rigid_plane_synthetic_points_and_diagnostics() -> None:
    """Fit a rigid plane from synthetic points, recovering normal and diagnostics."""
    rng = np.random.default_rng(42)
    # True plane: inclined at 30 deg to vertical
    tilt_rad = math.radians(30.0)
    normal_true = np.array([0.0, math.sin(tilt_rad), math.cos(tilt_rad)])
    normal_true /= np.linalg.norm(normal_true)

    u_true = np.array([1.0, 0.0, 0.0])
    v_true = np.cross(normal_true, u_true)
    v_true /= np.linalg.norm(v_true)
    origin_true = np.array([0.1, 1.2, -0.3])

    # Generate 50 points on the plane with small out-of-plane perturbation
    n_points = 50
    u_coords = rng.uniform(-0.8, 0.8, n_points)
    v_coords = rng.uniform(-0.8, 0.8, n_points)
    out_of_plane_true = rng.normal(0.0, 0.005, n_points)  # 5 mm noise

    points = np.zeros((n_points, 3))
    for i in range(n_points):
        points[i] = (
            origin_true
            + u_coords[i] * u_true
            + v_coords[i] * v_true
            + out_of_plane_true[i] * normal_true
        )

    plane, diag = fit_rigid_swing_plane(points)

    assert isinstance(plane, RigidSwingPlane)
    assert isinstance(diag, PlaneFitDiagnostics)
    assert abs(abs(np.dot(plane.normal, normal_true)) - 1.0) < 1e-3
    assert diag.rmse_m < 0.01  # Noise was ~5mm
    assert diag.max_residual_m < 0.02
    assert diag.sample_count == n_points
    assert np.isclose(np.linalg.norm(plane.normal), 1.0)
    assert np.isclose(np.linalg.norm(plane.u_axis), 1.0)
    assert np.isclose(np.linalg.norm(plane.v_axis), 1.0)
    assert abs(np.dot(plane.u_axis, plane.v_axis)) < 1e-9
    # Right-handedness
    np.testing.assert_allclose(
        np.cross(plane.u_axis, plane.v_axis), plane.normal, atol=1e-9
    )


def test_degenerate_point_clouds_fail() -> None:
    """Degenerate inputs (collinear points, fewer than 3 points, zero variance) fail closed."""
    # Fewer than 3 points
    with pytest.raises(DegeneratePlaneError, match="At least 3 non-collinear points"):
        fit_rigid_swing_plane(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))

    # Perfectly collinear points
    collinear = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ]
    )
    with pytest.raises(DegeneratePlaneError, match="collinear or rank < 2"):
        fit_rigid_swing_plane(collinear)

    # Identical points (zero variance)
    identical = np.tile([1.0, 2.0, 3.0], (5, 1))
    with pytest.raises(DegeneratePlaneError, match="collinear or rank < 2"):
        fit_rigid_swing_plane(identical)


def test_missing_markers_and_weights() -> None:
    """NaN frames and zero weights are ignored; missing markers do not move the plane."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [np.nan, np.nan, np.nan],
        ]
    )
    plane1, diag1 = fit_rigid_swing_plane(points)
    assert diag1.sample_count == 4

    # Fit without the NaN row
    plane2, diag2 = fit_rigid_swing_plane(points[:4])
    np.testing.assert_allclose(plane1.normal, plane2.normal, atol=1e-12)
    np.testing.assert_allclose(plane1.origin, plane2.origin, atol=1e-12)

    # Weighted points: an outlier with weight 0 does not move the plane
    points_with_outlier = np.vstack([points[:4], [100.0, 100.0, 100.0]])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    plane3, diag3 = fit_rigid_swing_plane(points_with_outlier, weights=weights)
    np.testing.assert_allclose(plane3.normal, plane2.normal, atol=1e-12)
    np.testing.assert_allclose(plane3.origin, plane2.origin, atol=1e-12)


def test_manufactured_out_of_plane_trajectory_retains_nonzero_residual() -> None:
    """A trajectory with manufactured out-of-plane curvature retains measurable 3D residual."""
    t = np.linspace(0.0, 1.0, 100)
    # Circle in XY plus sinusoidal out-of-plane Z excursion
    x = np.cos(np.pi * t)
    y = np.sin(np.pi * t)
    z = 0.15 * np.sin(2.0 * np.pi * t)  # 15 cm out-of-plane excursion
    traj = np.column_stack([x, y, z])

    plane, diag = fit_rigid_swing_plane(traj)
    # Plane should roughly be XY plane (normal along Z)
    assert abs(abs(plane.normal[2]) - 1.0) < 0.1
    # Residuals must remain visible and non-zero
    assert diag.rmse_m > 0.05
    assert diag.max_residual_m > 0.10
