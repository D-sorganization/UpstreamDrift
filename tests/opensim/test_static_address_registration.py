"""Tests for static address-hold trunk marker re-registration (#13434)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.tour_matching.static_address_registration import (
    TRUNK_MARKERS,
    StaticRegistrationResult,
    quiet_segment_frames,
    register_static_trunk_markers,
    solve_pelvis_rotations,
    waist_cluster_frame,
    write_registered_model,
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
def test_known_offset_recovery_opensim(tmp_path: Path) -> None:
    """Recover known trunk offsets after deliberately perturbing them by 60 deg."""
    pytest.importorskip("opensim")
    from src.engines.physics_engines.opensim.python.tour_matching.marker_set import (
        MarkerPlacement,
        attach_marker_set,
        parse_model,
        write_model,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.trc import write_trc
    from docs.development.opensim_tour_matching.os8_continuous_ik import OsimKinematics

    if not REAL_MODEL.is_file():
        pytest.skip(f"Model file not found: {REAL_MODEL}")

    kin_orig = OsimKinematics(REAL_MODEL)

    # Define known true trunk placements
    known_placements: dict[str, MarkerPlacement] = {
        "WaistLeft": MarkerPlacement("pelvis", (-0.530, -0.115, -0.170)),
        "WaistRight": MarkerPlacement("pelvis", (-0.530, -0.115, 0.170)),
        "WaistLBack": MarkerPlacement("pelvis", (-0.680, -0.115, -0.170)),
        "WaistRBack": MarkerPlacement("pelvis", (-0.680, -0.115, 0.170)),
        "BackTop": MarkerPlacement("torso", (-0.400, 0.313, 0.179)),
        "BackLeft": MarkerPlacement("torso", (-0.494, 0.166, 0.063)),
        "BackRight": MarkerPlacement("torso", (-0.480, 0.139, 0.243)),
        "HeadTop": MarkerPlacement("torso", (-0.203, 0.557, 0.192)),
        "HeadFront": MarkerPlacement("torso", (-0.086, 0.454, 0.185)),
        "HeadSide": MarkerPlacement("torso", (-0.224, 0.490, 0.078)),
        "LShoulderTop": MarkerPlacement("torso", (-0.278, 0.286, -0.032)),
        "LShoulderBack": MarkerPlacement("torso", (-0.439, 0.293, 0.019)),
        "RShoulderTop": MarkerPlacement("torso", (-0.150, 0.250, 0.300)),
        "RShoulderBack": MarkerPlacement("torso", (-0.385, 0.247, 0.315)),
    }

    # Prepare ground-truth model with known offsets for non-trunk + trunk
    all_known_placements = {
        name: MarkerPlacement(body, offset)
        for name, (body, offset) in kin_orig.offsets.items()
    }
    all_known_placements.update(known_placements)

    gt_tree = parse_model(REAL_MODEL)
    attach_marker_set(gt_tree, all_known_placements)
    gt_model_path = tmp_path / "ground_truth.osim"
    write_model(gt_tree, gt_model_path)

    # In declared static pose:
    # pelvis tilt/list/rot from waist cluster of known placements
    # lumbar_extension = lumbar_bending = lumbar_rotation = 0
    gt_kin = OsimKinematics(gt_model_path)
    q_static = np.zeros(len(gt_kin.coords))
    for i, _name in enumerate(gt_kin.names):
        coord = gt_kin.coords[i]
        q_static[i] = float(coord.getDefaultValue())

    # Set pelvis rotations to match waist cluster in declared pose
    waist_pts_local = {
        k: np.array(known_placements[k].offset_m)
        for k in ("WaistLeft", "WaistRight", "WaistLBack", "WaistRBack")
    }
    # Pelvis orientation in ground
    R_target = waist_cluster_frame(waist_pts_local)
    pelvis_rot_coords, _ = solve_pelvis_rotations(gt_kin.model, R_target)
    for c_name, val in pelvis_rot_coords.items():
        q_static[gt_kin.names.index(c_name)] = val

    # Set lumbar to 0
    for l_name in ("lumbar_extension", "lumbar_bending", "lumbar_rotation"):
        q_static[gt_kin.names.index(l_name)] = 0.0

    poses = gt_kin.poses(q_static)

    # Generate synthetic TRC with 20 quiet frames in this pose
    n_frames = 20
    rate_hz = 360.0
    time_s = np.arange(n_frames) / rate_hz
    labels = list(gt_kin.offsets.keys())
    points_m = np.zeros((n_frames, len(labels), 3))
    valid = np.ones((n_frames, len(labels)), dtype=bool)

    for i, label in enumerate(labels):
        body = gt_kin.offsets[label][0]
        offset = np.array(gt_kin.offsets[label][1])
        R_b, p_b = poses[body]
        p_world = R_b @ offset + p_b
        points_m[:, i, :] = p_world

    # Mark RShoulderTop as missing (as in real capture)
    if "RShoulderTop" in labels:
        valid[:, labels.index("RShoulderTop")] = False
        points_m[:, labels.index("RShoulderTop"), :] = np.nan

    cap = TourCapture(
        time_s=time_s, labels=tuple(labels), points_m=points_m, valid=valid
    )
    trc_path = tmp_path / "synthetic_quiet.trc"
    write_trc(cap, trc_path, rate_hz=rate_hz)

    # Now create perturbed model: rotate trunk markers by 60 deg around Y
    R_pert = Rotation.from_euler("y", 60.0, degrees=True).as_matrix()
    perturbed_placements = dict(all_known_placements)
    for name in TRUNK_MARKERS:
        if name in perturbed_placements:
            orig_off = np.array(perturbed_placements[name].offset_m)
            pert_off = tuple(float(v) for v in (R_pert @ orig_off))
            perturbed_placements[name] = MarkerPlacement(
                perturbed_placements[name].body, pert_off
            )

    pert_tree = parse_model(REAL_MODEL)
    attach_marker_set(pert_tree, perturbed_placements)
    pert_model_path = tmp_path / "perturbed.osim"
    write_model(pert_tree, pert_model_path)

    # Run static registration on perturbed model
    result = register_static_trunk_markers(
        model_path=pert_model_path,
        trc_path=trc_path,
        speed_threshold_mps=0.05,
        min_duration_s=0.05,
    )

    assert isinstance(result, StaticRegistrationResult)
    assert result.pelvis_fit_residual_rad < 1e-6
    assert "RShoulderTop" in result.unregistered_markers

    # Assert known offsets are recovered to < 1 mm (< 0.001 m)
    for name in result.re_registered_markers:
        recovered = np.array(result.new_offsets[name][1])
        expected = np.array(known_placements[name].offset_m)
        err_mm = np.linalg.norm(recovered - expected) * 1e3
        assert err_mm < 1.0, (
            f"Marker {name} recovered with error {err_mm:.3f} mm >= 1 mm"
        )

    # Write registered model and check
    out_model_path = tmp_path / "registered.osim"
    written_path = write_registered_model(
        pert_model_path, result.new_offsets, out_model_path
    )
    assert written_path.is_file()
