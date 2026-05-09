"""Reusable synthetic CIR fixture builders for the motion pipeline.

These builders produce **valid** Canonical Intermediate Representation (CIR)
objects that satisfy every Pydantic v2 invariant in
``src.shared.python.motion_pipeline.contracts``. Every downstream wave of
the motion pipeline (preprocessing, scaling, IK backends, motion matching,
orchestrator, sources) is expected to reuse these builders rather than
re-deriving synthetic data inline. The builders are deterministic
(``numpy.linspace`` for time, ``numpy.sin`` for joint angles, zeros for
unset velocity/acceleration channels) so tests stay reproducible.

The contracts in ``contracts.py`` model :data:`SchemaName`, :data:`UpAxis`,
unit systems, and engine choices as ``typing.Literal`` strings rather than
``enum.Enum`` types. The builders accept those raw string literals (with
sensible defaults) for compatibility with the spec, and the public types
:class:`KeypointSchemaT`, :class:`UnitSystemT`, :class:`WorldUpT`, and
:class:`EngineTypeT` are re-exported below as the canonical names so
downstream callers do not need to know whether a particular field is a
literal or an enum.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    CameraExtrinsics,
    CameraIntrinsics,
    JointDef,
    JointStateFrame,
    JointTrajectory,
    Keypoint,
    KeypointFrame,
    KeypointSequence,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
    MotionMatchingRequest,
    MotionMatchingResult,
    MotionTrajectory,
    SkeletonRig,
)

# Public type aliases - canonical names downstream code can import without
# caring whether the underlying contract uses enum or Literal. The contract
# module currently models these as ``typing.Literal`` strings; we mirror
# that here so callers can use the same identifiers regardless.
KeypointSchemaT = Literal["BODY_25", "MediaPipe_33", "COCO_17", "OpenPose_25", "custom"]
UnitSystemT = Literal["meters", "millimeters", "pixels"]
WorldUpT = Literal["+Y", "+Z", "+X", "-Y", "-Z", "-X"]
EngineTypeT = Literal["mujoco", "drake", "pinocchio", "opensim", "myosuite"]

# Standard golf marker set used as a default for marker fixtures.
DEFAULT_MARKER_NAMES: tuple[str, ...] = (
    "LSHO",
    "RSHO",
    "LASI",
    "RASI",
    "LKNE",
    "LANK",
)


# =============================================================================
# Calibration / Camera
# =============================================================================


def make_camera_intrinsics(
    fx: float = 800.0, fy: float = 800.0, cx: float = 640.0, cy: float = 480.0
) -> CameraIntrinsics:
    """Build a default pinhole-camera intrinsics with no lens distortion."""
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


def make_camera_extrinsics() -> CameraExtrinsics:
    """Build identity-rotation, zero-translation extrinsics."""
    return CameraExtrinsics(
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation=[0.0, 0.0, 0.0],
    )


def make_calibration(
    fps: float = 60.0,
    unit_system: UnitSystemT = "meters",
    world_up: WorldUpT = "+Y",
    *,
    n_cameras: int = 1,
    cal_id: str = "cal-synthetic",
) -> Calibration:
    """Build a multi-camera calibration with identity extrinsics.

    Each synthesized camera carries a default intrinsics + extrinsics dict
    so :class:`Calibration`'s ``check_cameras_have_intrinsics`` invariant is
    satisfied.
    """
    cameras: dict[str, dict[str, Any]] = {}
    for i in range(n_cameras):
        cameras[f"cam{i}"] = {
            "intrinsics": make_camera_intrinsics().model_dump(),
            "extrinsics": make_camera_extrinsics().model_dump(),
        }
    return Calibration(
        id=cal_id,
        cameras=cameras,
        unit_system=unit_system,
        source_fps=fps,
        world_up_axis=world_up,
    )


# =============================================================================
# Keypoints
# =============================================================================


def make_keypoint_frame(
    n_points: int = 33,
    schema: KeypointSchemaT = "MediaPipe_33",
    t: float = 0.0,
    dim: int = 3,
    *,
    frame_index: int | None = None,
) -> KeypointFrame:
    """Build a keypoint frame with ``n_points`` deterministic keypoints.

    ``dim=2`` produces 2D keypoints (``z=None``); ``dim=3`` produces 3D
    keypoints with monotonic-z layout. All confidences are 1.0.
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    keypoints: list[Keypoint] = []
    for i in range(n_points):
        keypoints.append(
            Keypoint(
                x=float(i) * 0.01,
                y=float(i) * 0.02,
                z=(float(i) * 0.03 if dim == 3 else None),
                confidence=1.0,
                name=f"kp_{i}",
            )
        )
    return KeypointFrame(
        timestamp=t,
        keypoints=keypoints,
        schema_name=schema,
        frame_index=frame_index,
    )


def make_keypoint_sequence(
    n_frames: int = 10,
    n_points: int = 33,
    schema: KeypointSchemaT = "MediaPipe_33",
    fps: float = 60.0,
    *,
    seq_id: str = "kp-seq-synthetic",
    with_calibration: bool = False,
) -> KeypointSequence:
    """Build a keypoint sequence with ``linspace`` timestamps."""
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    times = np.linspace(0.0, max(n_frames - 1, 0) / fps, n_frames)
    frames = [
        make_keypoint_frame(n_points=n_points, schema=schema, t=float(t), frame_index=i)
        for i, t in enumerate(times)
    ]
    return KeypointSequence(
        id=seq_id,
        frames=frames,
        calibration=make_calibration(fps=fps) if with_calibration else None,
    )


# =============================================================================
# Markers
# =============================================================================


def make_marker_frame(
    marker_names: tuple[str, ...] = DEFAULT_MARKER_NAMES,
    t: float = 0.0,
    *,
    frame_index: int | None = None,
) -> MarkerFrame:
    """Build a marker frame with deterministic xyz per named marker."""
    markers: dict[str, Marker] = {}
    for i, name in enumerate(marker_names):
        markers[name] = Marker(
            name=name,
            x=float(i) * 0.1,
            y=float(i) * 0.2,
            z=float(i) * 0.3,
            occluded=False,
        )
    return MarkerFrame(timestamp=t, markers=markers, frame_index=frame_index)


def make_marker_trajectory(
    n_frames: int = 10,
    marker_names: tuple[str, ...] = DEFAULT_MARKER_NAMES,
    fps: float = 60.0,
    *,
    traj_id: str = "marker-traj-synthetic",
    subject_id: str | None = "S001",
) -> MarkerTrajectory:
    """Build a marker trajectory with stationary markers and linspace times."""
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    times = np.linspace(0.0, max(n_frames - 1, 0) / fps, n_frames)
    frames = [
        make_marker_frame(marker_names=marker_names, t=float(t), frame_index=i)
        for i, t in enumerate(times)
    ]
    return MarkerTrajectory(id=traj_id, frames=frames, subject_id=subject_id)


# =============================================================================
# Skeleton Rig
# =============================================================================


def make_skeleton_rig(
    n_joints: int = 6, name_prefix: str = "joint", *, rig_id: str = "rig-synthetic"
) -> SkeletonRig:
    """Build a simple parent-chain rig: root -> joint_1 -> joint_2 -> ...

    Each joint has a single rotation axis ('Y') so DOF count == joint count.
    Root joint name is ``f"{name_prefix}_0"``.
    """
    if n_joints < 1:
        raise ValueError("n_joints must be >= 1")
    joints: dict[str, JointDef] = {}
    for i in range(n_joints):
        name = f"{name_prefix}_{i}"
        parent = f"{name_prefix}_{i - 1}" if i > 0 else None
        children = [f"{name_prefix}_{i + 1}"] if i < n_joints - 1 else []
        joints[name] = JointDef(
            name=name,
            parent=parent,
            children=children,
            tpose_offset=[0.0, 0.1, 0.0],
            axes=["Y"],
            limits=[],
            semantic_label=None,
        )
    return SkeletonRig(
        id=rig_id,
        joints=joints,
        root_joint=f"{name_prefix}_0",
        up_axis="+Y",
        scale=1.0,
    )


# =============================================================================
# Joint States
# =============================================================================


def _sine_q(num_dofs: int, t: float) -> list[float]:
    """Deterministic sine-wave joint angles."""
    return [float(np.sin(t + 0.1 * i)) for i in range(num_dofs)]


def make_joint_state_frame(
    rig: SkeletonRig,
    t: float = 0.0,
    *,
    with_qdot: bool = False,
    with_qddot: bool = False,
    frame_index: int | None = None,
) -> JointStateFrame:
    """Build a joint-state frame whose dimensions match ``rig.num_dofs``."""
    n = rig.num_dofs
    q = _sine_q(n, t)
    qdot = [0.0] * n if with_qdot else None
    qddot = [0.0] * n if with_qddot else None
    return JointStateFrame(
        timestamp=t, q=q, qdot=qdot, qddot=qddot, frame_index=frame_index
    )


def make_joint_trajectory(
    rig: SkeletonRig,
    n_frames: int = 10,
    fps: float = 60.0,
    *,
    with_qdot: bool = False,
    with_qddot: bool = False,
    traj_id: str = "joint-traj-synthetic",
) -> JointTrajectory:
    """Build a joint trajectory with sine-wave q values and linspace times."""
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    times = np.linspace(0.0, max(n_frames - 1, 0) / fps, n_frames)
    frames = [
        make_joint_state_frame(
            rig,
            t=float(t),
            with_qdot=with_qdot,
            with_qddot=with_qddot,
            frame_index=i,
        )
        for i, t in enumerate(times)
    ]
    return JointTrajectory(id=traj_id, skeleton=rig, frames=frames)


# =============================================================================
# Provenance / Motion / Matching
# =============================================================================


def make_provenance(
    subject_id: str = "S001",
    sport: str = "golf",
    *,
    source: str = "synthetic",
) -> dict[str, Any]:
    """Build a source-provenance dict.

    The CIR encodes provenance as an open ``dict[str, Any]`` rather than a
    dedicated model; this builder centralizes the keys so downstream waves
    converge on the same schema.
    """
    return {
        "subject_id": subject_id,
        "sport": sport,
        "source": source,
        "capture_system": "synthetic-fixture",
    }


def make_motion_trajectory(
    n_frames: int = 10,
    n_joints: int = 6,
    fps: float = 60.0,
    *,
    with_markers: bool = False,
    motion_id: str = "motion-synthetic",
    sport: str = "golf",
    rig: SkeletonRig | None = None,
) -> MotionTrajectory:
    """Build a top-level :class:`MotionTrajectory` (rig + traj + provenance).

    If ``rig`` is provided, it is used instead of creating a new synthetic rig.
    This ensures consistency when the caller has a specific rig (e.g., with
    custom joint IDs or rig ID) that should be preserved throughout the fixture.
    """
    if rig is None:
        rig = make_skeleton_rig(n_joints=n_joints)
    traj = make_joint_trajectory(rig, n_frames=n_frames, fps=fps)
    markers = (
        make_marker_trajectory(n_frames=n_frames, fps=fps) if with_markers else None
    )
    return MotionTrajectory(
        id=motion_id,
        skeleton=rig,
        trajectory=traj,
        marker_reference=markers,
        subject={"height_m": 1.80, "mass_kg": 80.0},
        sport=sport,
        club="driver",
        source_provenance=make_provenance(sport=sport),
    )


def make_cost_weights(weights: dict[str, float] | None = None) -> dict[str, float]:
    """Build a cost-weights dict for motion matching constraints.

    Cost weights are encoded inside :attr:`MotionMatchingRequest.constraints`
    in the current contracts. This builder produces a sensible default the
    matching/orchestrator waves can extend.
    """
    if weights is not None:
        return dict(weights)
    return {
        "joint_position": 1.0,
        "joint_velocity": 0.1,
        "marker_position": 1.0,
        "regularization": 0.01,
    }


def make_motion_matching_request(
    engine: EngineTypeT = "mujoco",
    *,
    request_id: str = "mm-req-synthetic",
    n_frames: int = 10,
    n_joints: int = 6,
) -> MotionMatchingRequest:
    """Build a :class:`MotionMatchingRequest` with a target trajectory."""
    target = make_motion_trajectory(n_frames=n_frames, n_joints=n_joints)
    return MotionMatchingRequest(
        id=request_id,
        target_trajectory=target,
        skeleton=target.skeleton,
        constraints={"weights": make_cost_weights()},
        solver_config={"engine": engine, "max_iters": 100},
    )


# =============================================================================
# Solver-output stand-ins
# =============================================================================
# The CIR does not yet ship dedicated TorqueTrajectory / ResidualReport
# models; downstream waves pass these as enriched dict payloads inside
# :attr:`MotionMatchingResult.metadata`. The builders below produce the
# canonical shape so wave-2+ tests can adopt them without divergence.


def make_torque_trajectory(
    rig: SkeletonRig, n_frames: int = 10, fps: float = 60.0
) -> dict[str, Any]:
    """Build a torque-trajectory payload (per-frame tau aligned to the rig).

    Returns a dict with ``timestamps`` (length ``n_frames``) and ``tau`` (a
    list of length ``n_frames``, each entry a list of ``rig.num_dofs``
    zeros). Stored on
    :attr:`MotionMatchingResult.metadata['torque_trajectory']`.
    """
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    times = np.linspace(0.0, max(n_frames - 1, 0) / fps, n_frames).tolist()
    tau = [[0.0] * rig.num_dofs for _ in range(n_frames)]
    return {"timestamps": times, "tau": tau, "skeleton_id": rig.id}


def make_residual_report(rig: SkeletonRig) -> dict[str, Any]:
    """Build a residual report payload keyed by joint name."""
    return {
        "skeleton_id": rig.id,
        "per_joint_rmse": dict.fromkeys(rig.joints, 0.0),
        "global_rmse": 0.0,
        "max_residual": 0.0,
    }


def make_motion_matching_result(
    rig: SkeletonRig,
    n_frames: int = 10,
    *,
    kind: Literal["torque", "kinematic"] = "torque",
    request_id: str = "mm-req-synthetic",
    success: bool = True,
) -> MotionMatchingResult:
    """Build a :class:`MotionMatchingResult` with embedded torque + residual.

    ``kind="torque"`` attaches a :func:`make_torque_trajectory` payload to
    metadata; ``kind="kinematic"`` omits it.

    The passed ``rig`` is preserved throughout the fixture by passing it to
    ``make_motion_trajectory``, ensuring that ``metadata['residual_report']``,
    ``metadata['torque_trajectory']``, and ``matched_trajectory.skeleton`` all
    reference the same rig.
    """
    motion = make_motion_trajectory(
        n_frames=n_frames, n_joints=rig.num_joints, rig=rig
    )
    metadata: dict[str, Any] = {
        "residual_report": make_residual_report(rig),
        "kind": kind,
    }
    if kind == "torque":
        metadata["torque_trajectory"] = make_torque_trajectory(rig, n_frames=n_frames)
    return MotionMatchingResult(
        request_id=request_id,
        success=success,
        matched_trajectory=motion,
        error_metrics={"rmse": 0.0, "max_error": 0.0},
        iterations=10,
        solve_time=0.5,
        message="synthetic ok",
        metadata=metadata,
    )


__all__ = [
    "DEFAULT_MARKER_NAMES",
    "EngineTypeT",
    "KeypointSchemaT",
    "UnitSystemT",
    "WorldUpT",
    "make_calibration",
    "make_camera_extrinsics",
    "make_camera_intrinsics",
    "make_cost_weights",
    "make_joint_state_frame",
    "make_joint_trajectory",
    "make_keypoint_frame",
    "make_keypoint_sequence",
    "make_marker_frame",
    "make_marker_trajectory",
    "make_motion_matching_request",
    "make_motion_matching_result",
    "make_motion_trajectory",
    "make_provenance",
    "make_residual_report",
    "make_skeleton_rig",
    "make_torque_trajectory",
]
