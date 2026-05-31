"""Reusable fixtures for synthetic estimation success-metric tests."""

from __future__ import annotations

import numpy as np

from src.shared.python.estimation.identifiability import ParameterSpec
from src.shared.python.motion_pipeline import (
    CameraExtrinsics,
    CameraIntrinsics,
    JointDef,
    JointStateFrame,
    JointTrajectory,
    SkeletonRig,
)


def make_planar_two_link_skeleton() -> SkeletonRig:
    """Return a minimal two-segment rig with known metre-scale lengths."""
    joints = {
        "root": JointDef(
            name="root",
            parent=None,
            children=["elbow"],
            tpose_offset=[0.0, 0.0, 3.0],
            axes=["Z"],
        ),
        "elbow": JointDef(
            name="elbow",
            parent="root",
            children=["wrist"],
            tpose_offset=[0.4, 0.0, 0.0],
            axes=["Z"],
        ),
        "wrist": JointDef(
            name="wrist",
            parent="elbow",
            children=[],
            tpose_offset=[0.3, 0.0, 0.0],
            axes=["Z"],
        ),
    }
    return SkeletonRig(id="synthetic-two-link", joints=joints, root_joint="root")


def make_two_link_trajectory(n_frames: int = 8, fps: float = 60.0) -> JointTrajectory:
    """Return a deterministic trajectory for the two-link fixture."""
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    skeleton = make_planar_two_link_skeleton()
    times = np.linspace(0.0, (n_frames - 1) / fps, n_frames)
    frames = []
    for i, timestamp in enumerate(times):
        phase = i / max(n_frames - 1, 1)
        q = [0.15 * np.sin(np.pi * phase), 0.4 * phase, -0.2 * phase]
        frames.append(
            JointStateFrame(
                timestamp=float(timestamp),
                q=[float(value) for value in q],
                qdot=[0.0, 0.0, 0.0],
                qddot=[0.0, 0.0, 0.0],
                frame_index=i,
            )
        )
    return JointTrajectory(
        id="synthetic-two-link-motion", skeleton=skeleton, frames=frames
    )


def make_fixture_cameras() -> tuple[
    tuple[str, CameraIntrinsics, CameraExtrinsics], ...
]:
    """Return two calibrated pinhole cameras looking along positive depth."""
    intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=480.0)
    return (
        ("cam0", intrinsics, CameraExtrinsics()),
        (
            "cam1",
            intrinsics,
            CameraExtrinsics(
                rotation=[
                    [0.9659, 0.0, -0.2588],
                    [0.0, 1.0, 0.0],
                    [0.2588, 0.0, 0.9659],
                ],
                translation=[0.2, 0.0, 0.0],
            ),
        ),
    )


def length_mass_parameter_spec() -> ParameterSpec:
    """Parameter layout used by CC-19/CC-20 synthetic recovery tests."""
    return ParameterSpec(("upper_length_m", "lower_length_m", "mass_scale"))


def two_link_observation_model(parameters: np.ndarray) -> np.ndarray:
    """Observation model where mass scale is intentionally unobservable."""
    upper_length, lower_length, _mass_scale = parameters
    angles = np.linspace(0.0, 0.8, 6)
    rows = []
    for angle in angles:
        elbow = np.array([upper_length * np.cos(angle), upper_length * np.sin(angle)])
        wrist = elbow + np.array(
            [
                lower_length * np.cos(2.0 * angle),
                lower_length * np.sin(2.0 * angle),
            ]
        )
        rows.extend([elbow[0], elbow[1], wrist[0], wrist[1]])
    return np.asarray(rows, dtype=np.float64)
