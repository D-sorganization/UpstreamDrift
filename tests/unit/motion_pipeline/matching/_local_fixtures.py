"""Self-contained fixtures for matching unit tests."""

from __future__ import annotations

import numpy as np

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointStateFrame,
    JointTrajectory,
    SkeletonRig,
)


def make_simple_rig(num_joints: int = 2) -> SkeletonRig:
    joints = {}
    for i in range(num_joints):
        parent = None if i == 0 else f"j{i - 1}"
        joints[f"j{i}"] = JointDef(
            name=f"j{i}",
            parent=parent,
            children=[f"j{i + 1}"] if i < num_joints - 1 else [],
            tpose_offset=[0.1, 0.0, 0.0],
            axes=["X"],
        )
    return SkeletonRig(id="rig_simple", joints=joints, root_joint="j0")


def make_pendulum_reference_trajectory(
    num_frames: int = 50, fps: float = 100.0, freq_hz: float = 1.0
) -> JointTrajectory:
    """Sinusoidal single-DOF reference trajectory (a pendulum-like signal)."""
    rig = make_simple_rig(num_joints=1)
    t = np.arange(num_frames) / fps
    q_vals = 0.5 * np.sin(2 * np.pi * freq_hz * t)
    frames = [
        JointStateFrame(timestamp=float(ts), q=[float(q_vals[i])], frame_index=i)
        for i, ts in enumerate(t)
    ]
    return JointTrajectory(id="pendulum_ref", skeleton=rig, frames=frames)
