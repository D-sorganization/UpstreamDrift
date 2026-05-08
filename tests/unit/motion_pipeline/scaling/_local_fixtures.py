"""Self-contained fixtures for scaling unit tests."""

from __future__ import annotations

import numpy as np

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
    SkeletonRig,
)


def make_simple_skeleton(scale: float = 1.0) -> SkeletonRig:
    """Build a minimal 3-joint skeleton (root, mid, distal).

    T-pose offsets sized so the chain length equals 1.0 at scale=1.0.
    """
    joints = {
        "root": JointDef(
            name="root", parent=None, children=["mid"], tpose_offset=[0.0, 0.0, 0.0]
        ),
        "mid": JointDef(
            name="mid", parent="root", children=["distal"], tpose_offset=[0.5, 0.0, 0.0]
        ),
        "distal": JointDef(
            name="distal", parent="mid", children=[], tpose_offset=[0.5, 0.0, 0.0]
        ),
    }
    return SkeletonRig(
        id="simple",
        joints=joints,
        root_joint="root",
        scale=scale,
    )


def make_marker_frame_for_scale(target_scale: float = 1.0) -> MarkerFrame:
    """Synthetic markers consistent with a known scale factor.

    Uses pelvis-width pair (RASI, LASI) ~0.15 * 1.75m * target_scale.
    """
    pelvis_width = 0.15 * 1.75 * target_scale
    markers = {
        "RASI": Marker(name="RASI", x=0.0, y=0.0, z=0.0),
        "LASI": Marker(name="LASI", x=pelvis_width, y=0.0, z=0.0),
        "RKNE": Marker(name="RKNE", x=0.0, y=-0.245 * 1.75 * target_scale, z=0.0),
        "RTHI": Marker(name="RTHI", x=0.0, y=0.0, z=0.0),
    }
    return MarkerFrame(timestamp=0.0, markers=markers, frame_index=0)


def make_marker_trajectory_for_scale(
    target_scale: float = 1.0, num_frames: int = 5
) -> MarkerTrajectory:
    frames: list[MarkerFrame] = []
    for i in range(num_frames):
        f = make_marker_frame_for_scale(target_scale)
        # bump the timestamp
        frames.append(MarkerFrame(timestamp=i * 0.01, markers=f.markers, frame_index=i))
    return MarkerTrajectory(id=f"calib_{target_scale}", frames=frames)
