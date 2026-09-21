"""Bridge between CIR SkeletonRig / JointTrajectory and CanonicalPose (MS-12 / #8867).

Provides bidirectional conversion with cross-representation parity between
the pipeline's Canonical Intermediate Representation (CIR) models
(:class:`SkeletonRig`, :class:`JointTrajectory`) and the canonical pose interchange
representation (:class:`CanonicalPose`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)
from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointStateFrame,
    JointTrajectory,
    SkeletonRig,
)
from src.shared.python.pose_interchange.canonical import CanonicalPose

_REFERENCE_FIELD_SET = frozenset(REFERENCE_GOLFER_FIELDS)


def _is_pelvis_translation_joint(joint_name: str, joint: JointDef) -> bool:
    name = joint_name.lower()
    label = (joint.semantic_label or "").lower()
    return "trans" in name or "pos" in name or "trans" in label or "pos" in label


def _is_pelvis_rotation_joint(joint_name: str, joint: JointDef, root_name: str) -> bool:
    if _is_pelvis_translation_joint(joint_name, joint):
        return False
    name = joint_name.lower()
    label = (joint.semantic_label or "").lower()
    return (
        joint_name == root_name
        or "rot" in name
        or "pelvis" in name
        or "rot" in label
        or "pelvis" in label
    )


def canonical_pose_to_joint_trajectory(
    pose: CanonicalPose,
    rig: SkeletonRig,
    *,
    timestamp: float = 0.0,
    trajectory_id: str = "canonical_pose_trajectory",
) -> JointTrajectory:
    """Convert a :class:`CanonicalPose` into a CIR :class:`JointTrajectory`.

    Preconditions:
    - ``pose`` is a valid :class:`CanonicalPose`.
    - ``rig`` is a valid :class:`SkeletonRig`.

    Postconditions:
    - The returned :class:`JointTrajectory` has one frame matching ``rig.num_dofs``.
    - Root joint carries the pelvis translation (m) and rotation (converted to rad).
    - Joint angles match pose entries in radians.
    """
    t_m = pose.pelvis_translation_m
    r_deg = pose.pelvis_rotation_xyz_deg
    r_rad = [math.radians(float(a)) for a in r_deg]

    has_trans_joint = any(
        _is_pelvis_translation_joint(name, j) for name, j in rig.joints.items()
    )

    q: list[float] = []
    for joint_name, joint in rig.joints.items():
        num_dofs = len(joint.axes)
        if num_dofs == 0:
            continue

        if _is_pelvis_translation_joint(joint_name, joint):
            for i in range(num_dofs):
                q.append(float(t_m[i]) if i < 3 else 0.0)
        elif _is_pelvis_rotation_joint(joint_name, joint, rig.root_joint) and (
            has_trans_joint or joint_name == rig.root_joint
        ):
            for i in range(num_dofs):
                q.append(float(r_rad[i]) if i < 3 else 0.0)
        elif num_dofs == 1:
            val_deg = (
                pose.joint_angles_deg.get(joint_name)
                or (
                    pose.joint_angles_deg.get(joint.semantic_label)
                    if joint.semantic_label
                    else None
                )
                or pose.joint_angles_deg.get(f"{joint_name}X")
                or pose.joint_angles_deg.get(f"{joint_name}_x")
                or 0.0
            )
            q.append(math.radians(float(val_deg)))
        else:
            for axis in joint.axes:
                clean_axis = axis.lstrip("+-").upper()
                candidate_keys = [
                    f"{joint_name}{clean_axis}",
                    f"{joint_name}_{clean_axis}",
                    f"{joint_name}_{clean_axis.lower()}",
                ]
                if joint.semantic_label:
                    candidate_keys.extend(
                        [
                            f"{joint.semantic_label}{clean_axis}",
                            f"{joint.semantic_label}_{clean_axis}",
                            f"{joint.semantic_label}_{clean_axis.lower()}",
                        ]
                    )
                val_deg = 0.0
                for k in candidate_keys:
                    if k in pose.joint_angles_deg:
                        val_deg = pose.joint_angles_deg[k]
                        break
                q.append(math.radians(float(val_deg)))

    frame = JointStateFrame(timestamp=float(timestamp), q=q, frame_index=0)
    return JointTrajectory(
        id=trajectory_id,
        skeleton=rig,
        frames=[frame],
        metadata={"source": "canonical_pose", "convention_tag": pose.convention_tag},
    )


def joint_trajectory_to_canonical_pose(
    trajectory: JointTrajectory,
    *,
    frame_index: int = 0,
) -> CanonicalPose:
    """Convert a CIR :class:`JointTrajectory` frame into a :class:`CanonicalPose`.

    Preconditions:
    - ``trajectory`` has at least one frame and ``frame_index`` is in range.

    Postconditions:
    - Returns a valid, frozen :class:`CanonicalPose`.
    - Joint angles within REFERENCE_GOLFER_FIELDS are mapped in degrees.
    """
    if not trajectory.frames:
        raise ValueError("JointTrajectory contains no frames")
    if frame_index < 0 or frame_index >= len(trajectory.frames):
        raise IndexError(
            f"frame_index {frame_index} out of range [0, {len(trajectory.frames)})"
        )

    frame = trajectory.frames[frame_index]
    q = frame.q
    rig = trajectory.skeleton

    t_m = np.zeros(3, dtype=np.float64)
    r_deg = np.zeros(3, dtype=np.float64)
    joint_angles_deg: dict[str, float] = {}

    has_trans_joint = any(
        _is_pelvis_translation_joint(name, j) for name, j in rig.joints.items()
    )

    dof_idx = 0
    for joint_name, joint in rig.joints.items():
        num_dofs = len(joint.axes)
        if num_dofs == 0:
            continue

        if _is_pelvis_translation_joint(joint_name, joint):
            for i in range(min(num_dofs, 3)):
                if dof_idx + i < len(q):
                    t_m[i] = q[dof_idx + i]
            dof_idx += num_dofs
        elif (
            _is_pelvis_rotation_joint(joint_name, joint, rig.root_joint)
            and (has_trans_joint or joint_name == rig.root_joint)
            and not np.any(r_deg)
        ):
            for i in range(min(num_dofs, 3)):
                if dof_idx + i < len(q):
                    r_deg[i] = math.degrees(q[dof_idx + i])
            dof_idx += num_dofs
        elif num_dofs == 1:
            if dof_idx < len(q):
                val_deg = math.degrees(q[dof_idx])
                if joint_name in _REFERENCE_FIELD_SET:
                    joint_angles_deg[joint_name] = val_deg
                elif (
                    joint.semantic_label
                    and joint.semantic_label in _REFERENCE_FIELD_SET
                ):
                    joint_angles_deg[joint.semantic_label] = val_deg
                else:
                    for suffix in ("X", "Y", "Z", ""):
                        candidate = f"{joint_name}{suffix}"
                        if candidate in _REFERENCE_FIELD_SET:
                            joint_angles_deg[candidate] = val_deg
                            break
            dof_idx += 1
        else:
            for axis in joint.axes:
                clean_axis = axis.lstrip("+-").upper()
                if dof_idx < len(q):
                    val_deg = math.degrees(q[dof_idx])
                    candidates = [
                        f"{joint_name}{clean_axis}",
                        f"{joint_name}_{clean_axis}",
                    ]
                    if joint.semantic_label:
                        candidates.extend(
                            [
                                f"{joint.semantic_label}{clean_axis}",
                                f"{joint.semantic_label}_{clean_axis}",
                            ]
                        )
                    for c in candidates:
                        if c in _REFERENCE_FIELD_SET:
                            joint_angles_deg[c] = val_deg
                            break
                dof_idx += 1

    return CanonicalPose(
        pelvis_translation_m=t_m,
        pelvis_rotation_xyz_deg=r_deg,
        joint_angles_deg=joint_angles_deg,
    )
