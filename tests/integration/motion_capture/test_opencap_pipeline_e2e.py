"""End-to-end integration test for OpenCap pipeline (#11405).

Verifies the complete flow from triangulated keypoints to 43 augmented
anatomical markers to OpenSim model scaling and inverse kinematics.

Acceptance criteria:
1. Synthetic capture -> augmented markers -> scaled model + IK, end to end.
2. Output is labelled 'model-conditioned' per ADR-0041.
3. TensorFlow is never a core dependency (ADR-0053).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    CanonicalObservations,
    JointTrajectory,
    MarkerFrame,
    MarkerTrajectory,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.sources.opencap_markers import (
    OPENCAP_AUGMENTED_MARKERS,
    OPENCAP_MARKER_SET_NAME,
)
from src.motion_capture.opencap_ingest import (
    OpenCapAugmenterConfig,
    OpenCapMarkerAugmenter,
    OpenCapPipelineResult,
    create_opencap_rig,
    run_opencap_pipeline_from_keypoints,
)

pytestmark = [pytest.mark.integration]


def _generate_synthetic_triangulated_capture(
    n_frames: int = 10,
    fps: float = 30.0,
) -> tuple[np.ndarray, list[str]]:
    """Synthesize 15 triangulated 3-D joint keypoints from a multi-camera capture."""
    names = [
        "mid_hip",
        "neck",
        "nose",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
    ]
    arr = np.zeros((n_frames, len(names), 3), dtype=float)
    dt = 1.0 / fps
    for f in range(n_frames):
        t = f * dt
        # Natural standing pose with a gentle arm flexion/extension swing
        arr[f, 0] = [0.0, 0.95, 0.0]  # mid_hip
        arr[f, 1] = [0.0, 1.45, 0.0]  # neck
        arr[f, 2] = [0.0, 1.65, 0.0]  # nose

        arr[f, 3] = [0.0, 1.42, 0.20]  # r_shoulder
        arr[f, 4] = [0.05 * np.cos(2 * np.pi * t), 1.15, 0.25]  # r_elbow
        arr[f, 5] = [0.10 * np.sin(2 * np.pi * t), 0.90, 0.25]  # r_wrist

        arr[f, 6] = [0.0, 1.42, -0.20]  # l_shoulder
        arr[f, 7] = [-0.05 * np.cos(2 * np.pi * t), 1.15, -0.25]  # l_elbow
        arr[f, 8] = [-0.10 * np.sin(2 * np.pi * t), 0.90, -0.25]  # l_wrist

        arr[f, 9] = [0.0, 0.90, 0.12]  # r_hip
        arr[f, 10] = [0.0, 0.50, 0.12]  # r_knee
        arr[f, 11] = [0.0, 0.10, 0.12]  # r_ankle

        arr[f, 12] = [0.0, 0.90, -0.12]  # l_hip
        arr[f, 13] = [0.0, 0.50, -0.12]  # l_knee
        arr[f, 14] = [0.0, 0.10, -0.12]  # l_ankle

    return arr, names


def test_tensorflow_is_never_a_core_dependency() -> None:
    """ADR-0053: TensorFlow must NEVER be imported by core motion capture / pipeline modules."""
    # Ensure importing augmenter and running pipeline does not import tensorflow
    assert "tensorflow" not in sys.modules, "TensorFlow was imported into sys.modules"


def test_synthetic_capture_to_augmented_markers_to_scaled_model_and_ik() -> None:
    """Acceptance criterion 1: End-to-end pipeline from keypoints to scaled model + IK."""
    keypoints_arr, names = _generate_synthetic_triangulated_capture(
        n_frames=6, fps=30.0
    )

    result = run_opencap_pipeline_from_keypoints(
        (keypoints_arr, names),
        height_m=1.85,
        mass_kg=82.0,
        allow_fallback=True,
    )

    # 1. Output type and shape assertions
    assert isinstance(result, OpenCapPipelineResult)
    assert result.num_frames == 6

    # 2. Augmented observations contain all 43 LaiUhlrich2022 markers
    obs = result.augmented_observations
    assert isinstance(obs, CanonicalObservations)
    assert len(obs.frames) == 6
    assert obs.marker_set_name == OPENCAP_MARKER_SET_NAME

    for frame in obs.frames:
        for marker_name in OPENCAP_AUGMENTED_MARKERS:
            assert marker_name in frame.markers
            m = frame.markers[marker_name]
            assert np.isfinite(m.x)
            assert np.isfinite(m.y)
            assert np.isfinite(m.z)

    # 3. Scaled rig postconditions
    scaled_rig = result.scaled_rig
    assert isinstance(scaled_rig, SkeletonRig)
    for jname, jdef in scaled_rig.joints.items():
        if jdef.parent is not None:
            length = float(np.linalg.norm(jdef.tpose_offset))
            assert length > 0.0, f"Segment for {jname} must have positive length"

    # 4. Solved inverse kinematics trajectory
    joint_traj = result.joint_trajectory
    assert isinstance(joint_traj, JointTrajectory)
    assert len(joint_traj.frames) == 6
    for frame in joint_traj.frames:
        assert frame.q is not None
        assert len(frame.q) > 0
        assert all(np.isfinite(q_val) for q_val in frame.q)

    # 5. Acceptance criterion 2: Evidence level labelled model-conditioned
    assert result.evidence_level == "model-conditioned"
    assert obs.metadata.get("evidence_level") == "model-conditioned"
    assert obs.source_provenance.get("generator") == "opencap-marker-augmenter"
    assert obs.metadata.get("source") == "opencap_marker_augmenter"

    # 6. Acceptance criterion 3: TensorFlow still not imported
    assert "tensorflow" not in sys.modules


def test_pipeline_with_marker_trajectory_input() -> None:
    """Verify input adaptability when feeding MarkerTrajectory directly."""
    kpts_arr, names = _generate_synthetic_triangulated_capture(n_frames=3, fps=30.0)
    frames = []
    for f in range(3):
        m_dict = {
            names[j]: (kpts_arr[f, j, 0], kpts_arr[f, j, 1], kpts_arr[f, j, 2])
            for j in range(len(names))
        }
        # Create MarkerFrame
        from src.shared.python.motion_pipeline.contracts import Marker

        mf = MarkerFrame(
            timestamp=f * 0.033,
            markers={
                k: Marker(name=k, x=v[0], y=v[1], z=v[2]) for k, v in m_dict.items()
            },
            frame_index=f,
        )
        frames.append(mf)

    traj = MarkerTrajectory(id="test-triangulated-traj", frames=frames)
    result = run_opencap_pipeline_from_keypoints(
        traj,
        height_m=1.75,
        mass_kg=70.0,
        allow_fallback=True,
    )
    assert result.num_frames == 3
    assert len(result.joint_trajectory.frames) == 3
    assert result.evidence_level == "model-conditioned"
