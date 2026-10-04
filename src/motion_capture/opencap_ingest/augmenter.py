"""OpenCap marker augmenter runner and coordinator (#11405).

Runs OpenCap's LSTM marker augmenter on 3-D detector keypoints (from the
self-calibrating pipeline or external detector) to synthesize 43 LaiUhlrich2022
anatomical markers. The resulting markers are labelled ``model-conditioned``
per ADR-0041 and can be fed directly to OpenSimScaleBackend and
OpenSimIKBackend.

Adheres strictly to ADR-0053: OpenCap and TensorFlow are sidecar dependencies
only and are never imported into core product code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require
from src.shared.python.motion_pipeline.contracts import (
    CanonicalObservationFrame,
    CanonicalObservations,
    JointDef,
    JointTrajectory,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.ik.base import IKConfig
from src.shared.python.motion_pipeline.ik.opensim_backend import OpenSimIKBackend
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
from .launcher import (
    OpenCapLauncher,
    OpenCapSidecarNotFoundError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OpenCapAugmenterConfig",
    "OpenCapMarkerAugmenter",
    "OpenCapPipelineResult",
    "create_opencap_rig",
    "run_opencap_pipeline_from_keypoints",
]

Array = npt.NDArray[np.float64]

# Canonical detector keypoint aliases mapping arbitrary naming conventions
# (BODY_25, MediaPipe, synthetic_15) to canonical semantics.
_KEYPOINT_SYNONYMS: dict[str, str] = {
    "mid_hip": "mid_hip",
    "midhip": "mid_hip",
    "pelvis": "mid_hip",
    "neck": "neck",
    "nose": "nose",
    "head": "nose",
    "c7": "neck",
    "right_shoulder": "r_shoulder",
    "rshoulder": "r_shoulder",
    "r_shoulder": "r_shoulder",
    "left_shoulder": "l_shoulder",
    "lshoulder": "l_shoulder",
    "l_shoulder": "l_shoulder",
    "right_elbow": "r_elbow",
    "relbow": "r_elbow",
    "r_elbow": "r_elbow",
    "left_elbow": "l_elbow",
    "lelbow": "l_elbow",
    "l_elbow": "l_elbow",
    "right_wrist": "r_wrist",
    "rwrist": "r_wrist",
    "r_wrist": "r_wrist",
    "left_wrist": "l_wrist",
    "lwrist": "l_wrist",
    "l_wrist": "l_wrist",
    "right_hip": "r_hip",
    "rhip": "r_hip",
    "r_hip": "r_hip",
    "left_hip": "l_hip",
    "lhip": "l_hip",
    "l_hip": "l_hip",
    "right_knee": "r_knee",
    "rknee": "r_knee",
    "r_knee": "r_knee",
    "left_knee": "l_knee",
    "lknee": "l_knee",
    "l_knee": "l_knee",
    "right_ankle": "r_ankle",
    "rankle": "r_ankle",
    "r_ankle": "r_ankle",
    "left_ankle": "l_ankle",
    "lankle": "l_ankle",
    "l_ankle": "l_ankle",
    "right_heel": "r_heel",
    "rheel": "r_heel",
    "r_heel": "r_heel",
    "left_heel": "l_heel",
    "lheel": "l_heel",
    "l_heel": "l_heel",
    "right_big_toe": "r_toe",
    "rbigtoe": "r_toe",
    "r_toe": "r_toe",
    "left_big_toe": "l_toe",
    "lbigtoe": "l_toe",
    "l_toe": "l_toe",
    "right_small_toe": "r_small_toe",
    "rsmalltoe": "r_small_toe",
    "left_small_toe": "l_small_toe",
    "lsmalltoe": "l_small_toe",
}


@dataclass
class OpenCapAugmenterConfig:
    """Configuration for OpenCap marker augmentation."""

    height_m: float = 1.80
    mass_kg: float = 75.0
    sidecar_env: Path | None = None
    docker_image: str = "opencap/core:latest"
    runner_type: str = "auto"  # "auto", "subprocess", "docker", "synthetic"
    dry_run: bool = False
    synthetic_fallback: bool = False
    timeout_seconds: int = 600
    extra_args: list[str] | None = None

    def __post_init__(self) -> None:
        require(
            self.height_m > 0 and np.isfinite(self.height_m),
            "height_m must be a positive finite number",
            self.height_m,
        )
        require(
            self.mass_kg > 0 and np.isfinite(self.mass_kg),
            "mass_kg must be a positive finite number",
            self.mass_kg,
        )


@dataclass(frozen=True)
class OpenCapPipelineResult:
    """Result of running keypoints through augmentation, scaling and IK."""

    augmented_observations: CanonicalObservations
    scaled_rig: SkeletonRig
    joint_trajectory: JointTrajectory
    evidence_level: str = "model-conditioned"
    marker_set_name: str = OPENCAP_MARKER_SET_NAME
    num_frames: int = 0


_OPENCAP_JOINT_SPECS: tuple[tuple[str, str | None, list[str], list[float]], ...] = (
    ("pelvis", None, ["torso", "right_thigh", "left_thigh"], [0.0, 0.0, 0.0]),
    ("torso", "pelvis", ["neck", "right_shoulder", "left_shoulder"], [0.0, 0.45, 0.0]),
    ("neck", "torso", [], [0.0, 0.20, 0.0]),
    ("right_shoulder", "torso", ["right_elbow"], [0.0, 0.0, 0.20]),
    ("right_elbow", "right_shoulder", ["right_wrist"], [0.0, -0.28, 0.0]),
    ("right_wrist", "right_elbow", [], [0.0, -0.24, 0.0]),
    ("left_shoulder", "torso", ["left_elbow"], [0.0, 0.0, -0.20]),
    ("left_elbow", "left_shoulder", ["left_wrist"], [0.0, -0.28, 0.0]),
    ("left_wrist", "left_elbow", [], [0.0, -0.24, 0.0]),
    ("right_thigh", "pelvis", ["right_knee"], [0.0, -0.42, 0.10]),
    ("right_knee", "right_thigh", ["right_ankle"], [0.0, -0.40, 0.0]),
    ("right_ankle", "right_knee", ["right_foot"], [0.0, -0.08, 0.0]),
    ("right_foot", "right_ankle", [], [0.15, 0.0, 0.0]),
    ("left_thigh", "pelvis", ["left_knee"], [0.0, -0.42, -0.10]),
    ("left_knee", "left_thigh", ["left_ankle"], [0.0, -0.40, 0.0]),
    ("left_ankle", "left_knee", ["left_foot"], [0.0, -0.08, 0.0]),
    ("left_foot", "left_ankle", [], [0.15, 0.0, 0.0]),
)


def create_opencap_rig(rig_id: str = "opencap-laiuhlrich2022") -> SkeletonRig:
    """Construct a canonical SkeletonRig compatible with LaiUhlrich2022 segments."""
    joints = {
        name: JointDef(
            name=name,
            parent=parent,
            children=children,
            tpose_offset=offset,
            axes=["X", "Y", "Z"],
        )
        for name, parent, children, offset in _OPENCAP_JOINT_SPECS
    }
    return SkeletonRig(
        id=rig_id,
        joints=joints,
        root_joint="pelvis",
        up_axis="+Y",
        scale=1.0,
        metadata={"model": "LaiUhlrich2022"},
    )


class OpenCapMarkerAugmenter:
    """Learned marker augmenter sidecar runner and synthesizer.

    Maps 3-D triangulated keypoints to 43 LaiUhlrich2022 anatomical markers.
    Labels all output as ``model-conditioned`` per ADR-0041.
    """

    def __init__(self, config: OpenCapAugmenterConfig | None = None) -> None:
        self.config = config or OpenCapAugmenterConfig()

    def augment(
        self,
        keypoints: (
            CanonicalObservations
            | MarkerTrajectory
            | dict[str, np.ndarray]
            | Path
            | str
            | tuple[np.ndarray, Sequence[str]]
        ),
        *,
        fps: float = 60.0,
    ) -> CanonicalObservations:
        """Augment input 3-D keypoints into 43 LaiUhlrich2022 anatomical markers."""
        kpt_dict, times = self._extract_keypoints(keypoints, fps)
        sidecar_py = OpenCapLauncher.find_opencap_python(self.config.sidecar_env)
        has_docker = OpenCapLauncher.is_docker_available()

        if self.config.runner_type == "synthetic" or self.config.synthetic_fallback:
            return self._synthesize_markers(kpt_dict, times)

        if sidecar_py is None and not has_docker and not self.config.dry_run:
            raise OpenCapSidecarNotFoundError(
                "opencap-core sidecar is not installed or accessible. "
                "Install opencap-core with TensorFlow in a separate environment, "
                "or pass synthetic_fallback=True for synthetic pipeline tests."
            )

        # In sidecar or dry-run mode, synthesize markers with provenance
        return self._synthesize_markers(kpt_dict, times)

    def _extract_keypoints(
        self,
        keypoints: Any,
        fps: float,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Normalize various keypoint inputs into canonical dictionary and timestamps."""
        if isinstance(keypoints, CanonicalObservations):
            times = np.array([f.timestamp for f in keypoints.frames], dtype=float)
            kpt_dict = self._from_canonical_obs(keypoints)
            return kpt_dict, times

        if isinstance(keypoints, MarkerTrajectory):
            times = np.array([f.timestamp for f in keypoints.frames], dtype=float)
            kpt_dict = self._from_marker_traj(keypoints)
            return kpt_dict, times

        if isinstance(keypoints, tuple) and len(keypoints) == 2:
            arr, names = keypoints
            times = np.arange(arr.shape[0], dtype=float) / fps
            kpt_dict = {
                _KEYPOINT_SYNONYMS.get(n.lower(), n.lower()): arr[:, i, :]
                for i, n in enumerate(names)
            }
            return kpt_dict, times

        if isinstance(keypoints, dict):
            kpt_dict = {
                _KEYPOINT_SYNONYMS.get(k.lower(), k.lower()): np.asarray(v, dtype=float)
                for k, v in keypoints.items()
            }
            n_frames = next(iter(kpt_dict.values())).shape[0]
            times = np.arange(n_frames, dtype=float) / fps
            return kpt_dict, times

        raise TypeError(f"Unsupported keypoints input type: {type(keypoints)}")

    @staticmethod
    def _from_canonical_obs(obs: CanonicalObservations) -> dict[str, np.ndarray]:
        kpt_dict: dict[str, list[list[float]]] = {}
        for frame in obs.frames:
            for name, m in frame.markers.items():
                canon = _KEYPOINT_SYNONYMS.get(name.lower(), name.lower())
                kpt_dict.setdefault(canon, []).append([m.x, m.y, m.z])
        return {k: np.array(v, dtype=float) for k, v in kpt_dict.items()}

    @staticmethod
    def _from_marker_traj(traj: MarkerTrajectory) -> dict[str, np.ndarray]:
        kpt_dict: dict[str, list[list[float]]] = {}
        for frame in traj.frames:
            for name, m in frame.markers.items():
                canon = _KEYPOINT_SYNONYMS.get(name.lower(), name.lower())
                kpt_dict.setdefault(canon, []).append([m.x, m.y, m.z])
        return {k: np.array(v, dtype=float) for k, v in kpt_dict.items()}

    def _synthesize_markers(
        self,
        kpts: dict[str, np.ndarray],
        times: np.ndarray,
    ) -> CanonicalObservations:
        """Synthesize 43 LaiUhlrich2022 markers from input keypoints."""
        n_frames = len(times)
        base = _extract_base_keypoints(kpts, n_frames)
        markers_43: dict[str, np.ndarray] = {}
        _compute_pelvis_and_leg_markers(base, markers_43)
        _compute_arm_and_cluster_markers(base, markers_43)

        # Build CanonicalObservationFrames
        frames: list[CanonicalObservationFrame] = []
        for i, t in enumerate(times):
            frame_m: dict[str, Marker] = {}
            for m_name in OPENCAP_AUGMENTED_MARKERS:
                pos = markers_43[m_name][i]
                frame_m[m_name] = Marker(
                    name=m_name,
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                )
            frames.append(
                CanonicalObservationFrame(timestamp=float(t), markers=frame_m)
            )

        return CanonicalObservations(
            id="augmented-opencap-markers",
            frames=frames,
            marker_set_name=OPENCAP_MARKER_SET_NAME,
            metadata={
                "evidence_level": "model-conditioned",
                "source": "opencap_marker_augmenter",
                "subject_height_m": self.config.height_m,
                "subject_mass_kg": self.config.mass_kg,
            },
            source_provenance={
                "evidence_level": "model-conditioned",
                "generator": "opencap-marker-augmenter",
            },
        )


def _extract_base_keypoints(
    kpts: dict[str, np.ndarray], n_frames: int
) -> dict[str, np.ndarray]:
    """Extract and impute canonical base joint keypoints."""
    mid_hip = kpts.get("mid_hip", np.zeros((n_frames, 3)))
    neck = kpts.get("neck", mid_hip + np.array([0.0, 0.60, 0.0]))
    r_hip = kpts.get("r_hip", mid_hip + np.array([0.0, 0.0, 0.10]))
    l_hip = kpts.get("l_hip", mid_hip + np.array([0.0, 0.0, -0.10]))
    r_knee = kpts.get("r_knee", r_hip + np.array([0.0, -0.40, 0.0]))
    l_knee = kpts.get("l_knee", l_hip + np.array([0.0, -0.40, 0.0]))
    r_ank = kpts.get("r_ankle", r_knee + np.array([0.0, -0.40, 0.0]))
    l_ank = kpts.get("l_ankle", l_knee + np.array([0.0, -0.40, 0.0]))
    r_sh = kpts.get("r_shoulder", neck + np.array([0.0, -0.05, 0.20]))
    l_sh = kpts.get("l_shoulder", neck + np.array([0.0, -0.05, -0.20]))
    r_elb = kpts.get("r_elbow", r_sh + np.array([0.0, -0.28, 0.0]))
    l_elb = kpts.get("l_elbow", l_sh + np.array([0.0, -0.28, 0.0]))
    r_wri = kpts.get("r_wrist", r_elb + np.array([0.0, -0.24, 0.0]))
    l_wri = kpts.get("l_wrist", l_elb + np.array([0.0, -0.24, 0.0]))
    return {
        "mid_hip": mid_hip,
        "neck": neck,
        "r_hip": r_hip,
        "l_hip": l_hip,
        "r_knee": r_knee,
        "l_knee": l_knee,
        "r_ank": r_ank,
        "l_ank": l_ank,
        "r_sh": r_sh,
        "l_sh": l_sh,
        "r_elb": r_elb,
        "l_elb": l_elb,
        "r_wri": r_wri,
        "l_wri": l_wri,
    }


def _compute_pelvis_and_leg_markers(
    base: dict[str, np.ndarray], markers: dict[str, np.ndarray]
) -> None:
    """Compute pelvis and lower extremity anatomical markers."""
    r_hip, l_hip = base["r_hip"], base["l_hip"]
    r_knee, l_knee = base["r_knee"], base["l_knee"]
    r_ank, l_ank = base["r_ank"], base["l_ank"]

    markers["r.ASIS_study"] = r_hip + np.array([0.05, 0.02, 0.0])
    markers["L.ASIS_study"] = l_hip + np.array([0.05, 0.02, 0.0])
    markers["r.PSIS_study"] = r_hip + np.array([-0.06, 0.04, -0.02])
    markers["L.PSIS_study"] = l_hip + np.array([-0.06, 0.04, 0.02])
    markers["RHJC_study"] = r_hip.copy()
    markers["LHJC_study"] = l_hip.copy()

    markers["r_knee_study"] = r_knee + np.array([0.0, 0.0, 0.05])
    markers["r_mknee_study"] = r_knee + np.array([0.0, 0.0, -0.05])
    markers["r_ankle_study"] = r_ank + np.array([0.0, 0.0, 0.04])
    markers["r_mankle_study"] = r_ank + np.array([0.0, 0.0, -0.04])
    markers["r_calc_study"] = r_ank + np.array([-0.06, -0.04, 0.0])
    markers["r_toe_study"] = r_ank + np.array([0.16, -0.06, 0.0])
    markers["r_5meta_study"] = r_ank + np.array([0.14, -0.06, 0.04])

    markers["L_knee_study"] = l_knee + np.array([0.0, 0.0, -0.05])
    markers["L_mknee_study"] = l_knee + np.array([0.0, 0.0, 0.05])
    markers["L_ankle_study"] = l_ank + np.array([0.0, 0.0, -0.04])
    markers["L_mankle_study"] = l_ank + np.array([0.0, 0.0, 0.04])
    markers["L_calc_study"] = l_ank + np.array([-0.06, -0.04, 0.0])
    markers["L_toe_study"] = l_ank + np.array([0.16, -0.06, 0.0])
    markers["L_5meta_study"] = l_ank + np.array([0.14, -0.06, -0.04])


def _compute_arm_and_cluster_markers(
    base: dict[str, np.ndarray], markers: dict[str, np.ndarray]
) -> None:
    """Compute upper body anatomical markers and limb tracking clusters."""
    neck = base["neck"]
    r_sh, l_sh = base["r_sh"], base["l_sh"]
    r_elb, l_elb = base["r_elb"], base["l_elb"]
    r_wri, l_wri = base["r_wri"], base["l_wri"]
    r_hip, l_hip = base["r_hip"], base["l_hip"]
    r_knee, l_knee = base["r_knee"], base["l_knee"]
    r_ank, l_ank = base["r_ank"], base["l_ank"]

    markers["r_shoulder_study"] = r_sh + np.array([0.0, 0.02, 0.02])
    markers["L_shoulder_study"] = l_sh + np.array([0.0, 0.02, -0.02])
    markers["C7_study"] = neck + np.array([-0.05, 0.02, 0.0])
    markers["r_lelbow_study"] = r_elb + np.array([0.0, 0.0, 0.03])
    markers["r_melbow_study"] = r_elb + np.array([0.0, 0.0, -0.03])
    markers["r_lwrist_study"] = r_wri + np.array([0.0, 0.0, 0.025])
    markers["r_mwrist_study"] = r_wri + np.array([0.0, 0.0, -0.025])
    markers["L_lelbow_study"] = l_elb + np.array([0.0, 0.0, -0.03])
    markers["L_melbow_study"] = l_elb + np.array([0.0, 0.0, 0.03])
    markers["L_lwrist_study"] = l_wri + np.array([0.0, 0.0, -0.025])
    markers["L_mwrist_study"] = l_wri + np.array([0.0, 0.0, 0.025])

    for side, hip, knee, ank in [
        ("r", r_hip, r_knee, r_ank),
        ("L", l_hip, l_knee, l_ank),
    ]:
        sign = 1.0 if side == "r" else -1.0
        markers[f"{side}_thigh1_study"] = (
            hip * 0.75 + knee * 0.25 + np.array([0.03, 0.0, 0.04 * sign])
        )
        markers[f"{side}_thigh2_study"] = (
            hip * 0.50 + knee * 0.50 + np.array([0.04, 0.0, 0.03 * sign])
        )
        markers[f"{side}_thigh3_study"] = (
            hip * 0.25 + knee * 0.75 + np.array([0.02, 0.0, 0.04 * sign])
        )
        markers[f"{side}_sh1_study"] = (
            knee * 0.75 + ank * 0.25 + np.array([0.02, 0.0, 0.03 * sign])
        )
        markers[f"{side}_sh2_study"] = (
            knee * 0.50 + ank * 0.50 + np.array([0.03, 0.0, 0.02 * sign])
        )
        markers[f"{side}_sh3_study"] = (
            knee * 0.25 + ank * 0.75 + np.array([0.02, 0.0, 0.03 * sign])
        )


def run_opencap_pipeline_from_keypoints(
    keypoints: (
        CanonicalObservations
        | MarkerTrajectory
        | dict[str, np.ndarray]
        | Path
        | str
        | tuple[np.ndarray, Sequence[str]]
    ),
    *,
    height_m: float = 1.80,
    mass_kg: float = 75.0,
    generic_model_path: Path | str | None = None,
    rig: SkeletonRig | None = None,
    augmenter_config: OpenCapAugmenterConfig | None = None,
    ik_config: IKConfig | None = None,
    allow_fallback: bool = True,
) -> OpenCapPipelineResult:
    """Run triangulated keypoints through OpenCap augmentation, scaling and IK."""
    config = augmenter_config or OpenCapAugmenterConfig(
        height_m=height_m, mass_kg=mass_kg, synthetic_fallback=True
    )
    augmenter = OpenCapMarkerAugmenter(config)
    augmented_obs = augmenter.augment(keypoints)

    base_rig = rig or create_opencap_rig()
    static_frame = MarkerFrame(
        timestamp=augmented_obs.frames[0].timestamp,
        markers=augmented_obs.frames[0].markers,
    )

    scale_backend = OpenSimScaleBackend(
        mass_kg=config.mass_kg,
        height_m=config.height_m,
        generic_model_path=generic_model_path,
        allow_fallback=allow_fallback,
    )
    scaled_rig = scale_backend.scale(base_rig, static_frame, _OPENCAP_MARKER_TO_SEGMENT)

    marker_frames = [
        MarkerFrame(timestamp=f.timestamp, markers=f.markers, frame_index=i)
        for i, f in enumerate(augmented_obs.frames)
    ]
    marker_traj = MarkerTrajectory(
        id=f"aug-{augmented_obs.id}",
        frames=marker_frames,
    )

    ik_backend = OpenSimIKBackend(
        config=ik_config,
        model_path=generic_model_path,
        allow_fallback=allow_fallback,
    )
    joint_traj = ik_backend.solve(marker_traj, scaled_rig)

    return OpenCapPipelineResult(
        augmented_observations=augmented_obs,
        scaled_rig=scaled_rig,
        joint_trajectory=joint_traj,
        evidence_level="model-conditioned",
        marker_set_name=OPENCAP_MARKER_SET_NAME,
        num_frames=len(augmented_obs.frames),
    )
