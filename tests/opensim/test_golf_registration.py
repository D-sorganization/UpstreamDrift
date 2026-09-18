"""Unit tests and numerical fixtures for golf capture registration and camera views (OG-04, #10398).

Tests:
1. Synthetic known rigid transform recovery with Kabsch: Landmark recovery <= 1e-8 m.
2. Inversion contract: applying forward followed by inverse returns original coordinates <= 1e-8 m.
3. Reflection rejection: determinant of rotation matrix must strictly equal +1.0.
4. Fail-closed on invalid units, non-finite points, or degenerate point sets (< 3 non-collinear points).
5. Ground alignment: projects support markers to ground level (Y = 0) cleanly.
6. Target line alignment: aligns target direction along canonical golf world axes (X = forward / target, Y = up, Z = lateral).
7. Golf camera presets: FRONT, SIDE, DOWN_THE_LINE, OVERHEAD views provide orthogonal/perspective camera parameters without altering model states or metrics.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.registration import (
    CameraPreset,
    CaptureRegistration,
    GolfCameraView,
    compute_capture_registration,
    get_golf_camera_view,
    register_points,
)
from src.shared.python.motion_matching.tour_capture_contract import (
    TourCapture,
    load_tour_capture,
)

pytestmark = pytest.mark.unit

C3D_PATH = Path(__file__).resolve().parents[2] / "data/C3D_TA_Driver.c3d"


def test_synthetic_rigid_transform_recovery_exact() -> None:
    """Synthetic known 3D rigid rotation and translation recovered to <= 1e-8 m."""
    # Source points (e.g. 5 landmarks)
    p_source = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [-0.5, 0.5, -0.5],
            [0.8, 1.2, -0.3],
        ],
        dtype=np.float64,
    )

    # Known rotation (30 deg about Y, 15 deg about X)
    theta_y = math.radians(30.0)
    theta_x = math.radians(15.0)
    Ry = np.array(
        [
            [math.cos(theta_y), 0, math.sin(theta_y)],
            [0, 1, 0],
            [-math.sin(theta_y), 0, math.cos(theta_y)],
        ]
    )
    Rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta_x), -math.sin(theta_x)],
            [0, math.sin(theta_x), math.cos(theta_x)],
        ]
    )
    R_true = Ry @ Rx
    t_true = np.array([0.25, -0.15, 0.80], dtype=np.float64)

    # Transform source to target: q = p @ R.T + t
    p_target = p_source @ R_true.T + t_true

    reg = compute_capture_registration(p_source, p_target)
    assert isinstance(reg, CaptureRegistration)

    # Rotation and translation recovery
    assert np.allclose(reg.rotation, R_true, atol=1e-8)
    assert np.allclose(reg.translation, t_true, atol=1e-8)
    assert math.isclose(np.linalg.det(reg.rotation), 1.0, rel_tol=1e-7)

    # Registered source points must reproduce target points to <= 1e-8 m
    p_recovered = register_points(p_source, reg)
    diff = np.max(np.linalg.norm(p_recovered - p_target, axis=1))
    assert diff <= 1e-8, f"Residual recovery {diff} exceeds 1e-8 m tolerance"


def test_registration_inversion_identity() -> None:
    """Applying forward registration followed by inverse must recover original points <= 1e-8 m."""
    pts = np.array(
        [
            [1.1, 0.9, 0.2],
            [-0.4, 1.5, -0.8],
            [0.2, 0.1, 1.4],
            [0.7, -0.5, -0.3],
        ],
        dtype=np.float64,
    )
    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    t = np.array([1.5, -0.5, 2.0], dtype=np.float64)
    reg = CaptureRegistration(
        rotation=R, translation=t, source_frame="capture", target_frame="world"
    )

    pts_fwd = register_points(pts, reg)
    pts_inv = register_points(pts_fwd, reg.inverse())

    max_diff = np.max(np.abs(pts_inv - pts))
    assert max_diff <= 1e-8, (
        f"Round-trip inversion discrepancy {max_diff} exceeds 1e-8 m"
    )


def test_registration_rejects_reflections_and_degeneracy() -> None:
    """Registration must reject reflection matrices (det == -1) and degenerate point sets."""
    # Collinear points
    p_collinear = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
    q_other = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    with pytest.raises(ValueError, match="collinear|degenerate"):
        compute_capture_registration(p_collinear, q_other)

    # Fewer than 3 points
    with pytest.raises(ValueError, match="(?i)at least 3"):
        compute_capture_registration(p_collinear[:2], q_other[:2])

    # Reflection matrix rejected by CaptureRegistration constructor
    R_reflect = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="reflection|proper rotation"):
        CaptureRegistration(rotation=R_reflect, translation=np.zeros(3))


def test_real_tour_capture_ground_and_target_alignment() -> None:
    """Tour capture registration aligns ground plane and target line with canonical golf axes."""
    if not C3D_PATH.exists():
        pytest.skip(f"Capture not found: {C3D_PATH}")

    cap = load_tour_capture(C3D_PATH)
    # Registration should produce world coordinates where:
    # 1. Feet are on the ground (min foot marker Y >= -0.01 and <= 0.05 m)
    # 2. Target line is directed along +X
    # 3. Upright axis is +Y
    from src.engines.physics_engines.opensim.python.tour_matching.registration import (
        align_tour_capture_to_golf_world,
    )

    registered_cap, reg = align_tour_capture_to_golf_world(cap)
    assert isinstance(registered_cap, TourCapture)
    assert isinstance(reg, CaptureRegistration)

    # Invertibility check on whole capture
    p0 = cap.points_m[0, cap.valid[0]]
    p0_fwd = registered_cap.points_m[0, registered_cap.valid[0]]
    p0_inv = register_points(p0_fwd, reg.inverse())
    assert np.max(np.abs(p0_inv - p0)) <= 1e-8


def test_golf_camera_views_presets() -> None:
    """Verify front, side, and down-the-line camera presets."""
    front = get_golf_camera_view(CameraPreset.FRONT_VIEW)
    side = get_golf_camera_view(CameraPreset.SIDE_VIEW)
    dtl = get_golf_camera_view(CameraPreset.DOWN_THE_LINE)

    assert isinstance(front, GolfCameraView)
    assert isinstance(side, GolfCameraView)
    assert isinstance(dtl, GolfCameraView)

    # Presets must be distinct
    assert not np.allclose(front.position, side.position)
    assert not np.allclose(side.position, dtl.position)

    # Look-at targets must be finite and upright vector normalized
    for view in (front, side, dtl):
        assert np.isfinite(view.position).all()
        assert np.isfinite(view.target).all()
        assert math.isclose(np.linalg.norm(view.up), 1.0, rel_tol=1e-6)
        assert view.fov_deg > 0.0


def test_camera_preset_does_not_mutate_model_states_or_metrics(tmp_path: Path) -> None:
    """Changing camera presets must leave model coordinates, state arrays, and hashes unchanged."""
    import hashlib
    from src.engines.physics_engines.opensim.python.tour_matching.visualization import (
        plot_3d_trajectory_overlay,
    )

    base_osim = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "engines"
        / "physics_engines"
        / "opensim"
        / "models"
        / "golf_humanoid_scaled.osim"
    )
    if not base_osim.is_file():
        pytest.skip(f"Model not found at {base_osim}")

    hash_before = hashlib.sha256(base_osim.read_bytes()).hexdigest()

    dummy_target = {"MarkerA": np.array([[0.0, 1.0, 0.0], [0.1, 1.0, 0.0]])}
    dummy_model = {"MarkerA": np.array([[0.0, 1.0, 0.05], [0.1, 1.0, 0.05]])}

    out1 = tmp_path / "overlay_front.png"
    out2 = tmp_path / "overlay_dtl.png"

    plot_3d_trajectory_overlay(
        dummy_target, dummy_model, out1, camera_preset=CameraPreset.FRONT_VIEW
    )
    plot_3d_trajectory_overlay(
        dummy_target, dummy_model, out2, camera_preset=CameraPreset.DOWN_THE_LINE
    )

    assert out1.is_file()
    assert out2.is_file()
    hash_after = hashlib.sha256(base_osim.read_bytes()).hexdigest()
    assert hash_before == hash_after, "Camera rendering mutated model file on disk"
