"""Unit tests for OpenCap marker augmenter and end-to-end pipeline (#11405)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts import (
    ContractViolationError,
    PreconditionError,
)
from src.shared.python.motion_pipeline.contracts import (
    CanonicalObservationFrame,
    CanonicalObservations,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.ik.base import IKConfig
from src.shared.python.motion_pipeline.ik.opensim_backend import (
    OpenSimIKBackend,
    OpenSimIKSolver,
)
from src.shared.python.motion_pipeline.scaling.marker_maps import (
    _OPENCAP_MARKER_TO_SEGMENT,
)
from src.shared.python.motion_pipeline.scaling.opensim_scale import (
    OpenSimScaleBackend,
)
from src.shared.python.motion_pipeline.sources.opencap_markers import (
    OPENCAP_AUGMENTED_MARKERS,
    OPENCAP_MARKER_SET_NAME,
)
from src.motion_capture.opencap_ingest.augmenter import (
    OpenCapAugmenterConfig,
    OpenCapMarkerAugmenter,
    OpenCapPipelineResult,
    create_opencap_rig,
    run_opencap_pipeline_from_keypoints,
)
from src.motion_capture.opencap_ingest.launcher import (
    OpenCapSidecarNotFoundError,
)

pytestmark = pytest.mark.unit


def _synthetic_15_keypoints(
    n_frames: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """Generate 15 canonical detector keypoints for n_frames."""
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
    for f in range(n_frames):
        t = f * 0.05
        # Base spine
        arr[f, 0] = [0.0, 0.95, 0.0]  # mid_hip
        arr[f, 1] = [0.0, 1.45, 0.0]  # neck
        arr[f, 2] = [0.0, 1.65, 0.0]  # nose
        # Shoulders
        arr[f, 3] = [0.0, 1.42, 0.20]  # r_shoulder
        arr[f, 4] = [0.0, 1.15, 0.25]  # r_elbow
        arr[f, 5] = [np.sin(t) * 0.1, 0.90, 0.25]  # r_wrist
        arr[f, 6] = [0.0, 1.42, -0.20]  # l_shoulder
        arr[f, 7] = [0.0, 1.15, -0.25]  # l_elbow
        arr[f, 8] = [-np.sin(t) * 0.1, 0.90, -0.25]  # l_wrist
        # Hips and legs
        arr[f, 9] = [0.0, 0.90, 0.12]  # r_hip
        arr[f, 10] = [0.0, 0.50, 0.12]  # r_knee
        arr[f, 11] = [0.0, 0.10, 0.12]  # r_ankle
        arr[f, 12] = [0.0, 0.90, -0.12]  # l_hip
        arr[f, 13] = [0.0, 0.50, -0.12]  # l_knee
        arr[f, 14] = [0.0, 0.10, -0.12]  # l_ankle
    return arr, names


def test_augmenter_config_validation() -> None:
    cfg = OpenCapAugmenterConfig(height_m=1.75, mass_kg=70.0)
    assert cfg.height_m == 1.75
    assert cfg.mass_kg == 70.0

    with pytest.raises(ContractViolationError):
        OpenCapAugmenterConfig(height_m=-1.0)
    with pytest.raises(ContractViolationError):
        OpenCapAugmenterConfig(height_m=0.0)
    with pytest.raises(ContractViolationError):
        OpenCapAugmenterConfig(mass_kg=0.0)


def test_create_opencap_rig() -> None:
    rig = create_opencap_rig("test-rig")
    assert isinstance(rig, SkeletonRig)
    assert rig.root_joint == "pelvis"
    assert "torso" in rig.joints
    assert "right_thigh" in rig.joints
    assert "left_thigh" in rig.joints
    assert "right_knee" in rig.joints
    assert "left_knee" in rig.joints
    assert "right_ankle" in rig.joints
    assert "left_ankle" in rig.joints
    assert "right_shoulder" in rig.joints
    assert "left_shoulder" in rig.joints


def test_augmenter_fails_closed_when_sidecar_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.motion_capture.opencap_ingest.launcher import OpenCapLauncher

    monkeypatch.setattr(OpenCapLauncher, "find_opencap_python", lambda env: None)
    monkeypatch.setattr(OpenCapLauncher, "is_docker_available", lambda: False)

    kpt_arr, names = _synthetic_15_keypoints(3)
    augmenter = OpenCapMarkerAugmenter(
        OpenCapAugmenterConfig(synthetic_fallback=False, dry_run=False)
    )

    with pytest.raises(
        OpenCapSidecarNotFoundError, match="opencap-core sidecar is not installed"
    ):
        augmenter.augment((kpt_arr, names))


def test_augmenter_synthetic_fallback() -> None:
    kpt_arr, names = _synthetic_15_keypoints(4)
    augmenter = OpenCapMarkerAugmenter(OpenCapAugmenterConfig(synthetic_fallback=True))
    obs = augmenter.augment((kpt_arr, names), fps=30.0)

    assert isinstance(obs, CanonicalObservations)
    assert len(obs.frames) == 4
    assert obs.marker_set_name == OPENCAP_MARKER_SET_NAME
    assert obs.metadata.get("evidence_level") == "model-conditioned"
    assert obs.source_provenance.get("evidence_level") == "model-conditioned"


def test_augmented_markers_contain_all_43_laiuhlrich2022_markers() -> None:
    kpt_arr, names = _synthetic_15_keypoints(3)
    augmenter = OpenCapMarkerAugmenter(OpenCapAugmenterConfig(synthetic_fallback=True))
    obs = augmenter.augment((kpt_arr, names))

    for frame in obs.frames:
        for marker_name in OPENCAP_AUGMENTED_MARKERS:
            assert marker_name in frame.markers, f"Missing marker {marker_name}"
            m = frame.markers[marker_name]
            assert np.isfinite(m.x)
            assert np.isfinite(m.y)
            assert np.isfinite(m.z)


def test_augment_from_canonical_observations() -> None:
    kpt_arr, names = _synthetic_15_keypoints(3)
    frames = []
    for f in range(3):
        m_dict = {
            names[j]: Marker(
                name=names[j],
                x=kpt_arr[f, j, 0],
                y=kpt_arr[f, j, 1],
                z=kpt_arr[f, j, 2],
            )
            for j in range(len(names))
        }
        frames.append(CanonicalObservationFrame(timestamp=f * 0.033, markers=m_dict))
    input_obs = CanonicalObservations(id="input-kpts", frames=frames)

    augmenter = OpenCapMarkerAugmenter(OpenCapAugmenterConfig(synthetic_fallback=True))
    augmented = augmenter.augment(input_obs)

    assert len(augmented.frames) == 3
    assert set(augmented.frames[0].markers.keys()) == set(OPENCAP_AUGMENTED_MARKERS)


def test_scaling_augmented_markers_with_opensim_scale_backend() -> None:
    kpt_arr, names = _synthetic_15_keypoints(2)
    augmenter = OpenCapMarkerAugmenter(
        OpenCapAugmenterConfig(synthetic_fallback=True, height_m=1.80, mass_kg=75.0)
    )
    obs = augmenter.augment((kpt_arr, names))

    rig = create_opencap_rig()
    static_frame = MarkerFrame(
        timestamp=obs.frames[0].timestamp, markers=obs.frames[0].markers
    )

    scale_backend = OpenSimScaleBackend(
        mass_kg=75.0, height_m=1.80, allow_fallback=True
    )
    scaled_rig = scale_backend.scale(rig, static_frame, _OPENCAP_MARKER_TO_SEGMENT)

    assert isinstance(scaled_rig, SkeletonRig)
    assert scaled_rig.id.endswith("-scaled")
    for jname, jdef in scaled_rig.joints.items():
        if jdef.parent is not None:
            length = float(np.linalg.norm(jdef.tpose_offset))
            assert length > 0.0, f"Joint {jname} must have positive length"


def test_opensim_ik_backend_fallback_solve() -> None:
    kpt_arr, names = _synthetic_15_keypoints(3)
    augmenter = OpenCapMarkerAugmenter(OpenCapAugmenterConfig(synthetic_fallback=True))
    obs = augmenter.augment((kpt_arr, names))

    rig = create_opencap_rig()
    marker_frames = [
        MarkerFrame(timestamp=f.timestamp, markers=f.markers, frame_index=i)
        for i, f in enumerate(obs.frames)
    ]
    marker_traj = MarkerTrajectory(id="aug-traj", frames=marker_frames)

    ik_backend = OpenSimIKBackend(allow_fallback=True)
    joint_traj = ik_backend.solve(marker_traj, rig)

    assert len(joint_traj.frames) == 3
    assert joint_traj.skeleton == rig
    assert joint_traj.metadata.get("backend") == "opensim"


def test_run_opencap_pipeline_from_keypoints() -> None:
    kpt_arr, names = _synthetic_15_keypoints(5)
    result = run_opencap_pipeline_from_keypoints(
        (kpt_arr, names),
        height_m=1.82,
        mass_kg=79.0,
        allow_fallback=True,
    )

    assert isinstance(result, OpenCapPipelineResult)
    assert result.evidence_level == "model-conditioned"
    assert result.marker_set_name == OPENCAP_MARKER_SET_NAME
    assert result.num_frames == 5
    assert len(result.augmented_observations.frames) == 5
    assert len(result.joint_trajectory.frames) == 5
    assert result.scaled_rig is not None
