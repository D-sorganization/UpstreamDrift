"""Smoke tests for the synthetic CIR fixture builders.

Each builder is exercised with default arguments and the result is run
through a Pydantic ``model_dump_json`` -> ``model_validate_json``
round-trip. This guards the fixture contract so downstream waves don't
silently start producing invalid CIR objects when ``contracts.py`` evolves.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    CameraExtrinsics,
    CameraIntrinsics,
    JointStateFrame,
    JointTrajectory,
    KeypointFrame,
    KeypointSequence,
    MarkerFrame,
    MarkerTrajectory,
    MotionMatchingRequest,
    MotionMatchingResult,
    MotionTrajectory,
    SkeletonRig,
)

from tests.unit.motion_pipeline._fixtures import (
    make_calibration,
    make_camera_extrinsics,
    make_camera_intrinsics,
    make_cost_weights,
    make_joint_state_frame,
    make_joint_trajectory,
    make_keypoint_frame,
    make_keypoint_sequence,
    make_marker_frame,
    make_marker_trajectory,
    make_motion_matching_request,
    make_motion_matching_result,
    make_motion_trajectory,
    make_provenance,
    make_residual_report,
    make_skeleton_rig,
    make_torque_trajectory,
)


def _roundtrip(model: BaseModel) -> BaseModel:
    """Serialize -> deserialize a Pydantic model; raises on contract drift."""
    js = model.model_dump_json()
    return type(model).model_validate_json(js)


# =============================================================================
# Per-builder smoke tests
# =============================================================================


def test_make_camera_intrinsics_roundtrip() -> None:
    m = make_camera_intrinsics()
    assert isinstance(m, CameraIntrinsics)
    _roundtrip(m)


def test_make_camera_extrinsics_roundtrip() -> None:
    m = make_camera_extrinsics()
    assert isinstance(m, CameraExtrinsics)
    _roundtrip(m)


def test_make_calibration_roundtrip() -> None:
    m = make_calibration(n_cameras=2)
    assert isinstance(m, Calibration)
    assert len(m.cameras) == 2
    _roundtrip(m)


def test_make_keypoint_frame_2d() -> None:
    f = make_keypoint_frame(n_points=5, dim=2)
    assert isinstance(f, KeypointFrame)
    assert all(kp.z is None for kp in f.keypoints)
    _roundtrip(f)


def test_make_keypoint_frame_3d() -> None:
    f = make_keypoint_frame(n_points=33, dim=3)
    assert all(kp.z is not None for kp in f.keypoints)
    _roundtrip(f)


def test_make_keypoint_sequence_monotonic_timestamps() -> None:
    seq = make_keypoint_sequence(n_frames=5, n_points=33, fps=30.0)
    assert isinstance(seq, KeypointSequence)
    timestamps = [f.timestamp for f in seq.frames]
    assert timestamps == sorted(timestamps)
    _roundtrip(seq)


def test_make_marker_frame() -> None:
    f = make_marker_frame()
    assert isinstance(f, MarkerFrame)
    assert f.num_markers == 6
    _roundtrip(f)


def test_make_marker_trajectory() -> None:
    t = make_marker_trajectory(n_frames=4)
    assert isinstance(t, MarkerTrajectory)
    assert t.num_frames == 4
    _roundtrip(t)


def test_make_skeleton_rig_chain() -> None:
    rig = make_skeleton_rig(n_joints=4)
    assert isinstance(rig, SkeletonRig)
    assert rig.num_joints == 4
    assert rig.num_dofs == 4  # one axis per joint
    # Root has no parent; tip has no children.
    assert rig.joints[rig.root_joint].parent is None
    _roundtrip(rig)


def test_make_joint_state_frame_dimensions_match_rig() -> None:
    rig = make_skeleton_rig(n_joints=5)
    frame = make_joint_state_frame(rig, t=0.5, with_qdot=True, with_qddot=True)
    assert isinstance(frame, JointStateFrame)
    assert len(frame.q) == rig.num_dofs
    assert frame.qdot is not None and len(frame.qdot) == rig.num_dofs
    assert frame.qddot is not None and len(frame.qddot) == rig.num_dofs
    _roundtrip(frame)


def test_make_joint_trajectory() -> None:
    rig = make_skeleton_rig(n_joints=3)
    traj = make_joint_trajectory(rig, n_frames=8, with_qdot=True)
    assert isinstance(traj, JointTrajectory)
    assert traj.num_frames == 8
    _roundtrip(traj)


def test_make_provenance_keys() -> None:
    p = make_provenance()
    for key in ("subject_id", "sport", "source", "capture_system"):
        assert key in p


@pytest.mark.parametrize("with_markers", [False, True])
def test_make_motion_trajectory(with_markers: bool) -> None:
    motion = make_motion_trajectory(n_frames=6, n_joints=4, with_markers=with_markers)
    assert isinstance(motion, MotionTrajectory)
    assert motion.num_frames == 6
    assert (motion.marker_reference is not None) == with_markers
    _roundtrip(motion)


def test_make_cost_weights_default_and_override() -> None:
    default = make_cost_weights()
    assert "joint_position" in default
    custom = make_cost_weights({"foo": 2.0})
    assert custom == {"foo": 2.0}


@pytest.mark.parametrize("engine", ["mujoco", "drake", "pinocchio", "opensim"])
def test_make_motion_matching_request(engine: str) -> None:
    req = make_motion_matching_request(engine=engine)  # type: ignore[arg-type]
    assert isinstance(req, MotionMatchingRequest)
    assert req.solver_config["engine"] == engine
    _roundtrip(req)


def test_make_torque_trajectory_shape() -> None:
    rig = make_skeleton_rig(n_joints=4)
    tt = make_torque_trajectory(rig, n_frames=5)
    assert len(tt["timestamps"]) == 5
    assert len(tt["tau"]) == 5
    assert all(len(row) == rig.num_dofs for row in tt["tau"])


def test_make_residual_report_keys() -> None:
    rig = make_skeleton_rig(n_joints=3)
    r = make_residual_report(rig)
    assert set(r["per_joint_rmse"].keys()) == set(rig.joints.keys())


@pytest.mark.parametrize("kind", ["torque", "kinematic"])
def test_make_motion_matching_result(kind: str) -> None:
    rig = make_skeleton_rig(n_joints=4)
    res = make_motion_matching_result(rig, n_frames=5, kind=kind)  # type: ignore[arg-type]
    assert isinstance(res, MotionMatchingResult)
    assert ("torque_trajectory" in res.metadata) == (kind == "torque")
    _roundtrip(res)
