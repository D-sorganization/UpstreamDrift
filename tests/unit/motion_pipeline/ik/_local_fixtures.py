"""Self-contained fixtures for ik unit tests."""

from __future__ import annotations

import numpy as np

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointLimit,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
    SkeletonRig,
)


def make_3dof_phantom_rig() -> SkeletonRig:
    """3-DOF chain: root (3 axes) -> link1 (3 axes) -> end_effector."""
    joints = {
        "root": JointDef(
            name="root",
            parent=None,
            children=["link1"],
            tpose_offset=[0.0, 0.0, 0.0],
            axes=["X", "Y", "Z"],
            limits=[
                JointLimit(lower=-3.14, upper=3.14),
                JointLimit(lower=-3.14, upper=3.14),
                JointLimit(lower=-3.14, upper=3.14),
            ],
        ),
        "link1": JointDef(
            name="link1",
            parent="root",
            children=[],
            tpose_offset=[0.5, 0.0, 0.0],
            axes=["X", "Y", "Z"],
            limits=[
                JointLimit(lower=-1.5, upper=1.5),
                JointLimit(lower=-1.5, upper=1.5),
                JointLimit(lower=-1.5, upper=1.5),
            ],
        ),
    }
    return SkeletonRig(id="phantom_3dof", joints=joints, root_joint="root")


def make_phantom_marker_trajectory(
    num_frames: int = 10, fps: float = 100.0
) -> MarkerTrajectory:
    frames: list[MarkerFrame] = []
    t = np.arange(num_frames) / fps
    for i, ts in enumerate(t):
        markers = {
            "M1": Marker(
                name="M1",
                x=0.5 * np.sin(2 * np.pi * ts),
                y=0.5 * np.cos(2 * np.pi * ts),
                z=0.0,
            ),
        }
        frames.append(MarkerFrame(timestamp=float(ts), markers=markers, frame_index=i))
    return MarkerTrajectory(id="phantom_traj", frames=frames)
