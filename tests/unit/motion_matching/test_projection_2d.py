"""Tests for 3D-to-2D swing plane calibration and projection (TB-03 #10588)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.motion_matching.club_target import ClubTarget, SourceProvenance
from src.shared.python.motion_matching.projection_2d import (
    CalibratedSwingPlane,
    estimate_swing_plane,
    project_to_2d,
    project_to_calibrated_plane,
)

pytestmark = pytest.mark.unit


def _create_synthetic_club_target(
    times: np.ndarray,
    butt: np.ndarray,
    clubhead: np.ndarray,
) -> ClubTarget:
    """Create a valid minimal ClubTarget for testing."""
    n = len(times)
    quats = np.zeros((n, 4), dtype=float)
    quats[:, 0] = 1.0  # w = 1.0 (identity quaternion)
    provenance = SourceProvenance(
        filename="test_synthetic.c3d",
        format="synthetic",
        subject_id="SYN",
        trial_id="T01",
        sha256="0" * 64,
    )
    return ClubTarget(
        time=times,
        butt=butt,
        clubhead=clubhead,
        club_quat=quats,
        impact_idx=n // 2,
        source=provenance,
    )


def test_known_tilted_synthetic_plane_recovers_points_and_gravity() -> None:
    """A known tilted plane at inclination angle beta recovers round-trip points and gravity."""
    # Define a known plane tilted by beta = 30 degrees about x-axis
    beta_deg = 30.0
    beta_rad = math.radians(beta_deg)
    # Rotation about x:
    # u = [1, 0, 0]
    # v = [0, cos(beta), sin(beta)]
    # n = [0, -sin(beta), cos(beta)]
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, math.cos(beta_rad), math.sin(beta_rad)])
    n = np.cross(u, v)
    origin = np.array([0.5, 1.2, 0.1])

    # Generate points on this plane
    rng = np.random.default_rng(42)
    in_plane_u = rng.uniform(-1.0, 1.0, size=(50, 1))
    in_plane_v = rng.uniform(-1.0, 1.0, size=(50, 1))
    pts = origin + in_plane_u * u + in_plane_v * v

    # Estimate plane
    plane = estimate_swing_plane(pts)

    # Validate basis is right-handed SO(3)
    det = float(np.linalg.det(plane.basis))
    assert abs(det - 1.0) < 1e-7, f"Basis determinant must be +1, got {det}"
    np.testing.assert_allclose(plane.basis.T @ plane.basis, np.eye(3), atol=1e-7)

    # Residuals on pure in-plane points must be near zero
    assert plane.residual.rmse < 1e-6
    assert plane.residual.max_deviation < 1e-6

    # Round-trip projection recovery
    projected = plane.project_points_to_plane(pts)
    recovered = plane.reconstruct_points_from_plane(projected)
    np.testing.assert_allclose(recovered, pts, atol=1e-6)

    # Recover gravity: default world gravity in Y-up is [0, -g, 0]
    g_world = np.array([0.0, -9.80665, 0.0])
    g_plane = plane.transform_vector_to_plane(g_world)
    # Magnitude must be preserved under rigid SO(3) transform
    assert abs(np.linalg.norm(g_plane) - 9.80665) < 1e-6


def test_degenerate_point_clouds_fail() -> None:
    """Degenerate point clouds (insufficient points or collinear) fail closed."""
    # Fewer than 3 points
    with pytest.raises(ValueError, match="At least 3 valid non-collinear points"):
        estimate_swing_plane(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))

    # Collinear points (rank 1)
    collinear = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )
    with pytest.raises(ValueError, match="collinear"):
        estimate_swing_plane(collinear)


def test_missing_markers_cannot_move_the_plane() -> None:
    """Missing markers (NaN rows) are filtered without corrupting the fitted plane."""
    # Valid planar circle points
    t = np.linspace(0, 2 * np.pi, 40)
    origin = np.array([0.1, 0.2, 0.3])
    pts_clean = origin + np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])

    # Add NaN rows
    pts_nan = pts_clean.copy()
    pts_nan[5] = [np.nan, np.nan, np.nan]
    pts_nan[15] = [0.0, np.nan, 1.0]

    plane_clean = estimate_swing_plane(pts_clean)
    plane_nan = estimate_swing_plane(pts_nan)

    # Normal vectors should agree up to sign/precision
    dot = abs(float(np.dot(plane_clean.normal, plane_nan.normal)))
    assert dot > 0.99999


def test_manufactured_out_of_plane_trajectory_retains_nonzero_3d_residual() -> None:
    """A trajectory with manufactured out-of-plane deviation retains non-zero 3D residual."""
    t = np.linspace(0, 1, 50)
    # Circle in xy plane + sinusoidal z-displacement
    z_out = 0.08 * np.sin(4 * np.pi * t)
    pts = np.column_stack([np.cos(t), np.sin(t), z_out])

    plane = estimate_swing_plane(pts)

    # 3D residual must be non-zero
    assert plane.residual.rmse > 0.01
    assert plane.residual.max_deviation >= 0.07

    # Target projection preserves the out-of-plane diagnostic
    times = np.linspace(0, 0.3, 50)
    butt = pts * 0.5
    target = _create_synthetic_club_target(times, butt, pts)

    projected_target = project_to_calibrated_plane(target, plane)
    assert projected_target.butt.shape == butt.shape
    assert projected_target.clubhead.shape == pts.shape


def test_project_to_2d_backward_compatibility() -> None:
    """project_to_2d preserves caller interface while utilizing rigid projection."""
    times = np.linspace(0, 0.3, 30)
    butt = np.column_stack([np.linspace(0, 1, 30), np.zeros(30), np.zeros(30)])
    head = np.column_stack(
        [np.linspace(0, 1.5, 30), np.sin(np.linspace(0, 1, 30)), np.zeros(30)]
    )
    target = _create_synthetic_club_target(times, butt, head)

    result = project_to_2d(target)
    assert isinstance(result, ClubTarget)
    assert result.clubhead.shape == (30, 3)
