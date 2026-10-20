"""Tests for the address-hold registration helpers (#13434)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.tour_matching.static_address_registration import (
    TRUNK_MARKERS,
    quiet_segment_frames,
    solve_pelvis_rotations,
    waist_cluster_frame,
)
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
REAL_MODEL = (
    ROOT
    / "docs/development/opensim_tour_matching/evidence/os7_moco_g1/inputs/golf_humanoid_scaled_tour_markers.osim"
)


# ---------------------------------------------------------------------------
# Pure numpy tests
# ---------------------------------------------------------------------------


def test_quiet_segment_detection_synthetic() -> None:
    """Quiet segment detects initial low-speed frames and respects min_duration and threshold."""
    rate_hz = 360.0
    dt = 1.0 / rate_hz
    n_frames = 100
    time_s = np.arange(n_frames) * dt
    labels = tuple(TRUNK_MARKERS)
    n_markers = len(labels)

    # 50 frames stationary (speed = 0), then moving at 0.20 m/s
    points_m = np.zeros((n_frames, n_markers, 3))
    # Add tiny stationary noise < 0.01 m/s
    for f in range(50):
        points_m[f] = np.full((n_markers, 3), 0.00002 * (f % 2))
    # Frames 50..99 moving steadily at 0.20 m/s along x
    for f in range(50, n_frames):
        points_m[f] = points_m[49] + (f - 49) * (0.20 * dt)

    valid = np.ones((n_frames, n_markers), dtype=bool)
    cap = TourCapture(time_s=time_s, labels=labels, points_m=points_m, valid=valid)

    frames = quiet_segment_frames(cap, threshold_mps=0.05, min_duration_s=0.05)
    # Frames 0..50 have dt segments with speed < 0.05 m/s
    assert len(frames) >= 50
    assert frames[0] == 0
    duration = time_s[frames[-1]] - time_s[0]
    assert duration >= 0.05

    # Threshold too tight rejects or shortens
    with pytest.raises(ValueError, match="duration"):
        quiet_segment_frames(cap, threshold_mps=0.0001, min_duration_s=0.05)

    # Motion starts immediately (frame 5): duration 4 * dt = 0.011 s < 0.05 s
    points_fast = np.zeros((n_frames, n_markers, 3))
    for f in range(5, n_frames):
        points_fast[f] = (f - 4) * (0.20 * dt)
    cap_fast = TourCapture(
        time_s=time_s, labels=labels, points_m=points_fast, valid=valid
    )
    with pytest.raises(ValueError, match="duration"):
        quiet_segment_frames(cap_fast, threshold_mps=0.05, min_duration_s=0.05)


def test_waist_cluster_frame_properties() -> None:
    """Waist cluster frame is orthonormal, det +1, x anterior, z right."""
    # Canonical waist cluster in OpenSim pelvis frame:
    # x anterior, y superior, z right
    waist_points = {
        "WaistLeft": np.array([0.0, 0.0, -0.15]),
        "WaistRight": np.array([0.0, 0.0, 0.15]),
        "WaistLBack": np.array([-0.10, 0.0, -0.12]),
        "WaistRBack": np.array([-0.10, 0.0, 0.12]),
    }

    R_ident = waist_cluster_frame(waist_points)
    # Check orthonormality
    np.testing.assert_allclose(R_ident.T @ R_ident, np.eye(3), atol=1e-12)
    # Check det +1 (right-handed)
    assert np.isclose(np.linalg.det(R_ident), 1.0, atol=1e-12)
    # Check axis alignments: x anterior [1, 0, 0], y superior [0, 1, 0], z right [0, 0, 1]
    np.testing.assert_allclose(R_ident[:, 0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(R_ident[:, 1], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(R_ident[:, 2], [0.0, 0.0, 1.0], atol=1e-12)

    # Test under an arbitrary 3D rotation
    R_rot = Rotation.from_euler("zyx", [35.0, -20.0, 15.0], degrees=True).as_matrix()
    rotated_points = {
        k: R_rot @ v + np.array([1.2, 0.8, -0.5]) for k, v in waist_points.items()
    }
    R_solved = waist_cluster_frame(rotated_points)

    np.testing.assert_allclose(R_solved.T @ R_solved, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R_solved), 1.0, atol=1e-12)
    np.testing.assert_allclose(R_solved, R_rot, atol=1e-12)


# ---------------------------------------------------------------------------
# OpenSim tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_opensim
def test_solve_pelvis_rotations_recovers_heading() -> None:
    """A golfer facing -x (the capture heading) is recovered to 1e-6 rad."""
    opensim = pytest.importorskip("opensim")
    model = opensim.Model(str(REAL_MODEL))
    target = Rotation.from_euler("y", np.pi - 0.02).as_matrix() @ (
        Rotation.from_euler("z", 0.3).as_matrix()
    )
    coords, residual = solve_pelvis_rotations(model, target)
    assert residual < 1e-6
    assert set(coords) == {"pelvis_tilt", "pelvis_list", "pelvis_rotation"}
    state = model.initSystem()
    for name, value in coords.items():
        model.getCoordinateSet().get(name).setValue(state, value, False)
    model.realizePosition(state)
    rot = model.getBodySet().get("pelvis").getTransformInGround(state).R().asMat33()
    got = np.array([[rot.get(i, j) for j in range(3)] for i in range(3)])
    assert np.allclose(got, target, atol=1e-6)
