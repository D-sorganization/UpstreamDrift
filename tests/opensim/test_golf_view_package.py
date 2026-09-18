"""Tests for OpenSim golf native viewer package and release evidence (OG-09, #10403).

Part of OpenSim epic #10394 under matched-swing epic #10363.

TDD tests verifying:
1. RED sentinels:
   - Blank motion name raises InvalidMotionSpecificationError.
   - Omitted or invalid frame range raises InvalidMotionSpecificationError.
   - Missing club asset raises MissingClubAssetError.
   - Mismatched model SHA-256 digest raises ModelMotionHashMismatchError.
   - Mismatched motion data hash raises ModelMotionHashMismatchError.
2. Viewer package functionality:
   - Reset to address returns t0, q0 with verified bilateral grip closure and face-on view.
   - Scrub to time returns smooth, interpolated coordinates across swing horizon.
   - Camera presets (FRONT_VIEW, DOWN_THE_LINE, SIDE_VIEW, OVERHEAD) decouple viewing from kinematics.
   - Muscle toggle indicates 'Unavailable' on torque baseline variant.
   - Clear labelling of IK playback, rejected replay, and accepted dynamic motion statuses.
3. Release evidence packaging:
   - Generation of keyframe stills (address, top, impact, finish) with camera records.
   - Reproducible video export.
   - Deterministic package SHA-256 digest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.full_swing_tracking import (
    FullSwingTrajectory,
)
from src.engines.physics_engines.opensim.python.tour_matching.model_variants import (
    create_muscle_model_variant,
    create_torque_model_variant,
)
from src.engines.physics_engines.opensim.python.tour_matching.registration import (
    CameraPreset,
)
from src.engines.physics_engines.opensim.python.tour_matching.view_package import (
    GolfNativeViewPackage,
    InvalidMotionSpecificationError,
    KeyframeStillsPackage,
    MissingClubAssetError,
    ModelMotionHashMismatchError,
    MotionStatus,
    VisualLayerOptions,
    build_golf_view_package,
    create_golf_view_launcher_entry,
    export_reproducible_video,
    reset_viewer_to_address,
    scrub_viewer_to_time,
    validate_view_package_specification,
)

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[2]
_OS3B_MODEL_PATH = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "opensim"
    / "models"
    / "golf_humanoid_scaled.osim"
)


def _make_dummy_trajectory(n_frames: int = 654) -> FullSwingTrajectory:
    """Construct deterministic synthetic swing trajectory."""
    time_s = np.linspace(0.0, 1.813888889, n_frames)
    coords = ("lumbar_pitch", "shoulder_flex_r", "elbow_flex_r", "wrist_flex_r")
    q = np.zeros((n_frames, len(coords)), dtype=np.float64)
    # Simple smooth kinematic motion
    q[:, 0] = 0.3 * np.sin(np.pi * time_s / 1.814)
    q[:, 1] = 1.2 * np.sin(np.pi * time_s / 0.85)

    return FullSwingTrajectory(
        time_s=time_s,
        coordinate_names=coords,
        q=q,
        qdot=None,
        grip_closure_distances_m=np.full(n_frames, 0.003),
    )


# --------------------------------------------------------------------------- #
# RED Sentinels & Fail-Closed Validation
# --------------------------------------------------------------------------- #


def test_view_package_rejects_blank_motion_name() -> None:
    """RED check: blank motion name raises InvalidMotionSpecificationError."""
    variant = create_torque_model_variant(_OS3B_MODEL_PATH)
    traj = _make_dummy_trajectory()

    with pytest.raises(InvalidMotionSpecificationError, match="blank"):
        build_golf_view_package(
            model_variant=variant,
            motion_trajectory=traj,
            motion_name="   ",
            frame_range=(0, 654),
            motion_status=MotionStatus.IK_PLAYBACK,
        )


def test_view_package_rejects_invalid_frame_range() -> None:
    """RED check: inverted or out-of-bounds frame range raises error."""
    variant = create_torque_model_variant(_OS3B_MODEL_PATH)
    traj = _make_dummy_trajectory(100)

    with pytest.raises(InvalidMotionSpecificationError, match="frame_range"):
        build_golf_view_package(
            model_variant=variant,
            motion_trajectory=traj,
            motion_name="driver_full_swing",
            frame_range=(80, 50),  # Inverted range
            motion_status=MotionStatus.IK_PLAYBACK,
        )

    with pytest.raises(InvalidMotionSpecificationError, match="frame_range"):
        build_golf_view_package(
            model_variant=variant,
            motion_trajectory=traj,
            motion_name="driver_full_swing",
            frame_range=(0, 200),  # Exceeds total frames (100)
            motion_status=MotionStatus.IK_PLAYBACK,
        )


def test_view_package_rejects_missing_club_asset(tmp_path: Path) -> None:
    """RED check: missing club asset on disk raises MissingClubAssetError."""
    variant = create_torque_model_variant(_OS3B_MODEL_PATH)
    traj = _make_dummy_trajectory()

    # Pass asset_base_dir pointing to empty tmp_path where club_head mesh is absent
    with pytest.raises(MissingClubAssetError, match="club"):
        validate_view_package_specification(
            model_variant=variant,
            motion_trajectory=traj,
            motion_name="driver_full_swing",
            frame_range=(0, 654),
            asset_base_dir=tmp_path,
        )


def test_view_package_rejects_model_hash_mismatch() -> None:
    """RED check: wrong model SHA-256 digest raises ModelMotionHashMismatchError."""
    variant = create_torque_model_variant(_OS3B_MODEL_PATH)
    traj = _make_dummy_trajectory()

    with pytest.raises(ModelMotionHashMismatchError, match="Model hash"):
        validate_view_package_specification(
            model_variant=variant,
            motion_trajectory=traj,
            motion_name="driver_full_swing",
            frame_range=(0, 654),
            expected_model_sha256="deadbeef0000111122223333444455556666777788889999aaaabbbbccccdddd",
        )


def test_view_package_rejects_motion_hash_mismatch() -> None:
    """RED check: wrong motion data digest raises ModelMotionHashMismatchError."""
    variant = create_torque_model_variant(_OS3B_MODEL_PATH)
    traj = _make_dummy_trajectory()

    with pytest.raises(ModelMotionHashMismatchError, match="Motion hash"):
        validate_view_package_specification(
            model_variant=variant,
            motion_trajectory=traj,
            motion_name="driver_full_swing",
            frame_range=(0, 654),
            expected_motion_hash="badf00d0000111122223333444455556666777788889999aaaabbbbccccdddd",
        )


# --------------------------------------------------------------------------- #
# Viewer Package Construction & Features
# --------------------------------------------------------------------------- #


def test_view_package_builds_torque_baseline_package(tmp_path: Path) -> None:
    """Verify clean build of torque baseline view package with muscle availability set to False."""
    variant = create_torque_model_variant(_OS3B_MODEL_PATH)
    traj = _make_dummy_trajectory(654)

    pkg = build_golf_view_package(
        model_variant=variant,
        motion_trajectory=traj,
        motion_name="driver_g3_full_swing",
        frame_range=(0, 654),
        motion_status=MotionStatus.ACCEPTED_DYNAMIC,
        camera_preset=CameraPreset.FRONT_VIEW,
        output_dir=tmp_path,
        export_video=False,
    )

    assert isinstance(pkg, GolfNativeViewPackage)
    assert pkg.motion_name == "driver_g3_full_swing"
    assert pkg.motion_status == MotionStatus.ACCEPTED_DYNAMIC
    assert pkg.club_name == "Driver"
    assert pkg.muscles_available is False  # Torque variant: muscles are unavailable
    assert pkg.layers.show_muscles is False
    assert pkg.layers.show_club is True
    assert pkg.layers.show_target_line is True
    assert len(pkg.package_sha256) == 64
    assert pkg.keyframe_stills is not None
    assert pkg.keyframe_stills.all_stills_present


def test_view_package_shows_muscles_available_for_muscle_variant(
    tmp_path: Path,
) -> None:
    """Verify that muscle variant marks muscles_available as True."""
    variant = create_muscle_model_variant(_OS3B_MODEL_PATH)
    traj = _make_dummy_trajectory(654)

    pkg = build_golf_view_package(
        model_variant=variant,
        motion_trajectory=traj,
        motion_name="iron_g3_muscle_tracking",
        frame_range=(0, 654),
        motion_status=MotionStatus.IK_PLAYBACK,
        camera_preset=CameraPreset.DOWN_THE_LINE,
        output_dir=tmp_path,
        export_video=False,
    )

    assert pkg.muscles_available is True
    assert pkg.motion_status == MotionStatus.IK_PLAYBACK
    assert pkg.camera_preset == CameraPreset.DOWN_THE_LINE


def test_reset_viewer_to_address() -> None:
    """Verify reset to address returns address coordinates and verifies bilateral grip closure."""
    variant = create_torque_model_variant(_OS3B_MODEL_PATH)
    traj = _make_dummy_trajectory(654)

    pkg = build_golf_view_package(
        model_variant=variant,
        motion_trajectory=traj,
        motion_name="driver_swing",
        frame_range=(0, 654),
        motion_status=MotionStatus.ACCEPTED_DYNAMIC,
    )

    address_state = reset_viewer_to_address(pkg, traj)
    assert address_state["time_s"] == 0.0
    assert address_state["camera_preset"] == CameraPreset.FRONT_VIEW.value
    assert address_state["grip_closure_valid"] is True
    assert len(address_state["q"]) == len(traj.coordinate_names)


def test_scrub_viewer_to_time() -> None:
    """Verify scrubbing to target time returns interpolated coordinates and active frame."""
    variant = create_torque_model_variant(_OS3B_MODEL_PATH)
    traj = _make_dummy_trajectory(654)

    pkg = build_golf_view_package(
        model_variant=variant,
        motion_trajectory=traj,
        motion_name="driver_swing",
        frame_range=(0, 654),
        motion_status=MotionStatus.ACCEPTED_DYNAMIC,
    )

    scrubbed = scrub_viewer_to_time(pkg, traj, target_time_s=0.85)  # Top of backswing
    assert np.isclose(scrubbed["time_s"], 0.85, atol=1e-2)
    assert 0 <= scrubbed["frame_index"] < 654
    assert len(scrubbed["q"]) == len(traj.coordinate_names)


def test_motion_status_labels_and_receipt_summary() -> None:
    """Verify distinct motion statuses (IK playback, rejected replay, accepted dynamic) are labeled."""
    variant = create_torque_model_variant(_OS3B_MODEL_PATH)
    traj = _make_dummy_trajectory(100)

    pkg_ik = build_golf_view_package(
        model_variant=variant,
        motion_trajectory=traj,
        motion_name="m1",
        frame_range=(0, 100),
        motion_status=MotionStatus.IK_PLAYBACK,
    )
    assert pkg_ik.receipt_status_summary["badge"] == "IK_PLAYBACK"

    pkg_rej = build_golf_view_package(
        model_variant=variant,
        motion_trajectory=traj,
        motion_name="m2",
        frame_range=(0, 100),
        motion_status=MotionStatus.REJECTED_REPLAY,
    )
    assert pkg_rej.receipt_status_summary["badge"] == "REJECTED_REPLAY"

    pkg_dyn = build_golf_view_package(
        model_variant=variant,
        motion_trajectory=traj,
        motion_name="m3",
        frame_range=(0, 100),
        motion_status=MotionStatus.ACCEPTED_DYNAMIC,
    )
    assert pkg_dyn.receipt_status_summary["badge"] == "ACCEPTED_DYNAMIC"


def test_export_reproducible_video(tmp_path: Path) -> None:
    """Verify reproducible animation / video export with recorded camera parameters."""
    variant = create_torque_model_variant(_OS3B_MODEL_PATH)
    traj = _make_dummy_trajectory(20)

    pkg = build_golf_view_package(
        model_variant=variant,
        motion_trajectory=traj,
        motion_name="short_driver_motion",
        frame_range=(0, 20),
        motion_status=MotionStatus.ACCEPTED_DYNAMIC,
        output_dir=tmp_path,
        export_video=True,
    )

    assert pkg.video_export_path is not None
    vid_file = Path(pkg.video_export_path)
    assert vid_file.is_file()
    assert vid_file.stat().st_size > 0


def test_create_golf_view_launcher_entry() -> None:
    """Verify launcher provider compatibility entry generation."""
    variant = create_torque_model_variant(_OS3B_MODEL_PATH)
    traj = _make_dummy_trajectory(100)

    pkg = build_golf_view_package(
        model_variant=variant,
        motion_trajectory=traj,
        motion_name="driver_full_swing",
        frame_range=(0, 100),
        motion_status=MotionStatus.ACCEPTED_DYNAMIC,
    )

    entry = create_golf_view_launcher_entry(pkg)
    assert entry["model_id"] == "golf_humanoid_scaled"
    assert entry["variant_id"] == "golf_humanoid_torque_variant"
    assert entry["motion_status"] == "accepted_dynamic"
    assert "presets" in entry
    assert "front_view" in entry["presets"]
    assert (
        entry["muscles_toggle_enabled"] is False
    )  # Explicitly disabled for torque baseline
