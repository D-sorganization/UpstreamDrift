"""OpenCap marker augmenter runner and coordinator (#11405, #12971).

Runs OpenCap's LSTM marker augmenter on 3-D detector keypoints (from the
self-calibrating pipeline or external detector) to synthesize 43 LaiUhlrich2022
anatomical markers via the opencap-core sidecar. Output from real sidecar
execution is labelled ``model-conditioned`` per ADR-0041 and can be fed directly
to OpenSimScaleBackend and OpenSimIKBackend.

When running in synthetic or fallback mode, markers are synthesized using
deterministic geometric offsets and stamped ``synthetic`` (generator:
``geometric-offsets``).

Adheres strictly to ADR-0053: OpenCap and TensorFlow are sidecar dependencies
only and are never imported into core product code.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
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
    "head": "head",
    "c7": "c7",
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
    "right_foot_index": "r_toe",
    "r_foot_index": "r_toe",
    "rfootindex": "r_toe",
    "left_foot_index": "l_toe",
    "l_foot_index": "l_toe",
    "lfootindex": "l_toe",
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
    session_dir: Path | None = None

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
        require(
            self.runner_type.lower() in ("auto", "subprocess", "docker", "synthetic"),
            "runner_type must be one of 'auto', 'subprocess', 'docker', 'synthetic'",
            self.runner_type,
        )


@dataclass(frozen=True)
class OpenCapPipelineResult:
    """Result of running keypoints through augmentation, scaling and IK."""

    augmented_observations: CanonicalObservations
    scaled_rig: SkeletonRig
    joint_trajectory: JointTrajectory
    evidence_level: str = "synthetic"
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


def _write_keypoints_trc(
    path: Path,
    kpts: dict[str, np.ndarray],
    times: np.ndarray,
    fps: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = len(times)
    kpt_names = list(kpts.keys())
    n_markers = len(kpt_names)
    if not kpt_names:
        rows = [
            f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
            (
                "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
            ),
            f"{fps:.2f}\t{fps:.2f}\t{n_frames}\t0\tm\t{fps:.2f}\t1\t{n_frames}",
            "Frame#\tTime",
            "",
        ]
        for f in range(n_frames):
            rows.append(f"{f + 1}\t{times[f]:.6f}")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return

    name_row = "Frame#\tTime\t" + "\t\t\t".join(kpt_names) + "\t\t\t"
    axis_row = "\t\t" + "\t".join(f"X{i}\tY{i}\tZ{i}" for i in range(1, n_markers + 1))
    rows = [
        f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
        (
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
            "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
        ),
        f"{fps:.2f}\t{fps:.2f}\t{n_frames}\t{n_markers}\tm\t{fps:.2f}\t1\t{n_frames}",
        name_row,
        axis_row,
    ]
    for f in range(n_frames):
        t = times[f]
        coords = []
        for name in kpt_names:
            pt = kpts[name][f]
            coords.extend(
                [
                    f"{pt[0]:.4f}" if np.isfinite(pt[0]) else "nan",
                    f"{pt[1]:.4f}" if np.isfinite(pt[1]) else "nan",
                    f"{pt[2]:.4f}" if np.isfinite(pt[2]) else "nan",
                ]
            )
        rows.append(f"{f + 1}\t{t:.6f}\t" + "\t".join(coords))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_session_metadata(
    path: Path,
    height_m: float,
    mass_kg: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "calibrationSettings:\n"
        "  overwriteDeployedIntrinsics: false\n"
        "  saveSessionIntrinsics: false\n"
        "gender_mf: m\n"
        f"height_m: {height_m}\n"
        "iphoneModel:\n"
        "  Cam0: iphone13,3\n"
        "  Cam1: iphone13,3\n"
        "markerAugmentationSettings:\n"
        "  markerAugmenterModel: LSTM\n"
        f"mass_kg: {mass_kg}\n"
        "openSimModel: LaiUhlrich2022\n"
        "subjectID: subject-01\n"
    )
    path.write_text(content, encoding="utf-8")


class OpenCapMarkerAugmenter:
    """Learned marker augmenter sidecar runner and geometric synthesizer.

    Maps 3-D triangulated keypoints to 43 LaiUhlrich2022 anatomical markers.
    Real sidecar execution is labelled ``model-conditioned`` per ADR-0041.
    Synthetic fallback or dry-run execution is labelled ``synthetic`` with
    generator ``geometric-offsets``.
    """

    def __init__(
        self,
        config: OpenCapAugmenterConfig | None = None,
        launcher: OpenCapLauncher | None = None,
    ) -> None:
        self.config = config or OpenCapAugmenterConfig()
        self._custom_launcher = launcher is not None
        self.launcher = launcher or OpenCapLauncher()

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

        config = self.config
        runner = config.runner_type.lower()
        if runner == "synthetic" or config.synthetic_fallback:
            return self._synthesize_markers(kpt_dict, times)

        sidecar_py = OpenCapLauncher.find_opencap_python(self.config.sidecar_env)
        has_docker = OpenCapLauncher.is_docker_available()

        if self._custom_launcher:
            sidecar_available = True
        elif runner == "subprocess":
            sidecar_available = sidecar_py is not None
        elif runner == "docker":
            sidecar_available = has_docker
        else:  # "auto"
            sidecar_available = sidecar_py is not None or has_docker

        if not sidecar_available and not self.config.dry_run:
            raise OpenCapSidecarNotFoundError(
                "opencap-core sidecar is not installed or accessible. "
                "Install opencap-core with TensorFlow in a separate environment, "
                "or pass synthetic_fallback=True for synthetic pipeline tests."
            )

        return self._run_sidecar(keypoints, kpt_dict, times, fps)

    def _run_sidecar(
        self,
        keypoints: Any,
        kpt_dict: dict[str, np.ndarray],
        times: np.ndarray,
        fps: float,
    ) -> CanonicalObservations:
        from .launcher import OpenCapLaunchConfig

        session_dir = self.config.session_dir
        cleanup_temp = False
        if session_dir is None:
            if isinstance(keypoints, Path) and keypoints.is_dir():
                session_dir = keypoints
            elif isinstance(keypoints, str) and Path(keypoints).is_dir():
                session_dir = Path(keypoints)
            else:
                session_dir = Path(tempfile.mkdtemp(prefix="opencap_augment_"))
                cleanup_temp = True

        try:
            trc_path = session_dir / "MarkerData" / "trial1.trc"
            meta_path = session_dir / "sessionMetadata.yaml"
            if not trc_path.exists():
                _write_keypoints_trc(trc_path, kpt_dict, times, fps)
            if not meta_path.exists():
                _write_session_metadata(
                    meta_path, self.config.height_m, self.config.mass_kg
                )

            launch_cfg = OpenCapLaunchConfig(
                session_dir=session_dir,
                opencap_env=self.config.sidecar_env,
                docker_image=self.config.docker_image,
                runner_type=self.config.runner_type,
                timeout_seconds=self.config.timeout_seconds,
                trial_name="trial1",
                extra_args=self.config.extra_args,
                dry_run=self.config.dry_run,
            )

            result = self.launcher.launch(launch_cfg, raise_on_error=True)

            if result.session is not None and result.session.observations is not None:
                obs = result.session.observations
                if result.used_real_opencap:
                    obs.metadata["evidence_level"] = "model-conditioned"
                    obs.metadata["generator"] = "opencap-marker-augmenter"
                    obs.source_provenance["evidence_level"] = "model-conditioned"
                    obs.source_provenance["generator"] = "opencap-marker-augmenter"
                    return obs
                if all(m in obs.frames[0].markers for m in OPENCAP_AUGMENTED_MARKERS):
                    obs.metadata["evidence_level"] = "synthetic"
                    obs.metadata["generator"] = "geometric-offsets"
                    obs.metadata["dry_run"] = True
                    obs.source_provenance["evidence_level"] = "synthetic"
                    obs.source_provenance["generator"] = "geometric-offsets"
                    obs.source_provenance["dry_run"] = True
                    return obs

            if self.config.dry_run:
                obs = self._synthesize_markers(kpt_dict, times)
                obs.metadata["dry_run"] = True
                obs.source_provenance["dry_run"] = True
                return obs

            if result.error_message:
                raise RuntimeError(result.error_message)
            raise RuntimeError(
                "OpenCap sidecar completed without returning observations"
            )
        finally:
            if cleanup_temp and session_dir.exists():
                shutil.rmtree(session_dir, ignore_errors=True)

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
            n_frames = next(iter(kpt_dict.values())).shape[0] if kpt_dict else 0
            times = np.arange(n_frames, dtype=float) / fps
            require(
                all(arr.shape[0] == n_frames for arr in kpt_dict.values()),
                "All keypoint arrays must share length n_frames",
            )
            return kpt_dict, times

        raise TypeError(f"Unsupported keypoints input type: {type(keypoints)}")

    @staticmethod
    def _from_canonical_obs(obs: CanonicalObservations) -> dict[str, np.ndarray]:
        n_frames = len(obs.frames)
        all_canons: set[str] = set()
        for frame in obs.frames:
            for name in frame.markers:
                all_canons.add(_KEYPOINT_SYNONYMS.get(name.lower(), name.lower()))
        kpt_dict = {c: np.full((n_frames, 3), np.nan, dtype=float) for c in all_canons}
        for f_idx, frame in enumerate(obs.frames):
            for name, m in frame.markers.items():
                canon = _KEYPOINT_SYNONYMS.get(name.lower(), name.lower())
                if not m.occluded and np.all(np.isfinite([m.x, m.y, m.z])):
                    kpt_dict[canon][f_idx] = [m.x, m.y, m.z]
        return kpt_dict

    @staticmethod
    def _from_marker_traj(traj: MarkerTrajectory) -> dict[str, np.ndarray]:
        n_frames = len(traj.frames)
        all_canons: set[str] = set()
        for frame in traj.frames:
            for name in frame.markers:
                all_canons.add(_KEYPOINT_SYNONYMS.get(name.lower(), name.lower()))
        kpt_dict = {c: np.full((n_frames, 3), np.nan, dtype=float) for c in all_canons}
        for f_idx, frame in enumerate(traj.frames):
            for name, m in frame.markers.items():
                canon = _KEYPOINT_SYNONYMS.get(name.lower(), name.lower())
                if not m.occluded and np.all(np.isfinite([m.x, m.y, m.z])):
                    kpt_dict[canon][f_idx] = [m.x, m.y, m.z]
        return kpt_dict

    def _synthesize_markers(
        self,
        kpts: dict[str, np.ndarray],
        times: np.ndarray,
    ) -> CanonicalObservations:
        """Synthesize 43 LaiUhlrich2022 markers from input keypoints."""
        n_frames = len(times)
        base, imputed_keypoints = _extract_base_keypoints(kpts, n_frames)
        markers_43: dict[str, np.ndarray] = {}
        _compute_pelvis_and_leg_markers(base, markers_43, imputed_keypoints)
        _compute_arm_and_cluster_markers(base, markers_43, imputed_keypoints)

        # Build CanonicalObservationFrames with NaN-safe propagation and gap counting
        gap_count = 0
        frames: list[CanonicalObservationFrame] = []
        for i, t in enumerate(times):
            frame_m: dict[str, Marker] = {}
            frame_gaps = 0
            for m_name in OPENCAP_AUGMENTED_MARKERS:
                pos = markers_43[m_name][i]
                if np.all(np.isfinite(pos)):
                    frame_m[m_name] = Marker(
                        name=m_name,
                        x=float(pos[0]),
                        y=float(pos[1]),
                        z=float(pos[2]),
                        occluded=False,
                    )
                else:
                    gap_count += 1
                    frame_gaps += 1
                    frame_m[m_name] = Marker.model_construct(
                        name=m_name,
                        x=float(pos[0]),
                        y=float(pos[1]),
                        z=float(pos[2]),
                        residual=None,
                        occluded=True,
                    )
            frame_meta = {"gap_count": frame_gaps} if frame_gaps else {}
            frames.append(
                CanonicalObservationFrame(
                    timestamp=float(t),
                    markers=frame_m,
                    metadata=frame_meta,
                )
            )

        imputed_list = sorted(imputed_keypoints)
        meta = {
            "evidence_level": "synthetic",
            "source": "opencap_marker_augmenter",
            "generator": "geometric-offsets",
            "subject_height_m": self.config.height_m,
            "subject_mass_kg": self.config.mass_kg,
            "imputed_keypoints": imputed_list,
            "gap_count": gap_count,
        }
        prov = {
            "evidence_level": "synthetic",
            "generator": "geometric-offsets",
            "imputed_keypoints": imputed_list,
            "gap_count": gap_count,
        }
        if self.config.dry_run:
            meta["dry_run"] = True
            prov["dry_run"] = True

        return CanonicalObservations(
            id="augmented-opencap-markers",
            frames=frames,
            marker_set_name=OPENCAP_MARKER_SET_NAME,
            metadata=meta,
            source_provenance=prov,
        )


def _derive_mid_hip(
    kpts: dict[str, np.ndarray], n_frames: int, imputed: set[str]
) -> np.ndarray:
    """Derive mid_hip from available keypoints before falling back to offsets."""
    if "mid_hip" in kpts:
        return kpts["mid_hip"]
    imputed.add("mid_hip")
    if "r_hip" in kpts and "l_hip" in kpts:
        return (kpts["r_hip"] + kpts["l_hip"]) / 2.0
    if "r_hip" in kpts:
        return kpts["r_hip"] + np.array([0.0, 0.0, -0.10])
    if "l_hip" in kpts:
        return kpts["l_hip"] + np.array([0.0, 0.0, 0.10])
    if "neck" in kpts:
        return kpts["neck"] - np.array([0.0, 0.60, 0.0])
    if "r_shoulder" in kpts and "l_shoulder" in kpts:
        sh_mid = (kpts["r_shoulder"] + kpts["l_shoulder"]) / 2.0
        return sh_mid - np.array([0.0, 0.55, 0.0])
    if "r_shoulder" in kpts:
        return kpts["r_shoulder"] - np.array([0.0, 0.55, 0.20])
    if "l_shoulder" in kpts:
        return kpts["l_shoulder"] - np.array([0.0, 0.55, -0.20])
    if "r_knee" in kpts and "l_knee" in kpts:
        return (kpts["r_knee"] + kpts["l_knee"]) / 2.0 + np.array([0.0, 0.40, 0.0])
    if len(kpts) > 0:
        return next(iter(kpts.values())).copy()
    return np.zeros((n_frames, 3))


def _derive_neck(
    kpts: dict[str, np.ndarray], mid_hip: np.ndarray, imputed: set[str]
) -> np.ndarray:
    """Derive neck from available keypoints before falling back to mid_hip offset."""
    if "neck" in kpts:
        return kpts["neck"]
    imputed.add("neck")
    if "c7" in kpts:
        return kpts["c7"] + np.array([0.05, -0.02, 0.0])
    if "r_shoulder" in kpts and "l_shoulder" in kpts:
        sh_mid = (kpts["r_shoulder"] + kpts["l_shoulder"]) / 2.0
        return sh_mid + np.array([0.0, 0.05, 0.0])
    if "r_shoulder" in kpts:
        return kpts["r_shoulder"] + np.array([0.0, 0.05, -0.20])
    if "l_shoulder" in kpts:
        return kpts["l_shoulder"] + np.array([0.0, 0.05, 0.20])
    if "nose" in kpts:
        return kpts["nose"] - np.array([0.0, 0.20, 0.0])
    return mid_hip + np.array([0.0, 0.60, 0.0])


def _extract_base_keypoints(
    kpts: dict[str, np.ndarray], n_frames: int
) -> tuple[dict[str, np.ndarray], set[str]]:
    """Extract and impute canonical base joint keypoints."""
    imputed: set[str] = set()
    mid_hip = _derive_mid_hip(kpts, n_frames, imputed)
    neck = _derive_neck(kpts, mid_hip, imputed)

    def _resolve(key: str, fallback_calc: Any) -> np.ndarray:
        if key in kpts:
            return kpts[key]
        imputed.add(key)
        return fallback_calc()

    r_hip = _resolve("r_hip", lambda: mid_hip + np.array([0.0, 0.0, 0.10]))
    l_hip = _resolve("l_hip", lambda: mid_hip + np.array([0.0, 0.0, -0.10]))
    r_knee = _resolve("r_knee", lambda: r_hip + np.array([0.0, -0.40, 0.0]))
    l_knee = _resolve("l_knee", lambda: l_hip + np.array([0.0, -0.40, 0.0]))
    r_ank = _resolve("r_ankle", lambda: r_knee + np.array([0.0, -0.40, 0.0]))
    l_ank = _resolve("l_ankle", lambda: l_knee + np.array([0.0, -0.40, 0.0]))
    r_sh = _resolve("r_shoulder", lambda: neck + np.array([0.0, -0.05, 0.20]))
    l_sh = _resolve("l_shoulder", lambda: neck + np.array([0.0, -0.05, -0.20]))
    r_elb = _resolve("r_elbow", lambda: r_sh + np.array([0.0, -0.28, 0.0]))
    l_elb = _resolve("l_elbow", lambda: l_sh + np.array([0.0, -0.28, 0.0]))
    r_wri = _resolve("r_wrist", lambda: r_elb + np.array([0.0, -0.24, 0.0]))
    l_wri = _resolve("l_wrist", lambda: l_elb + np.array([0.0, -0.24, 0.0]))

    base = {
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
    for opt in (
        "c7",
        "nose",
        "head",
        "r_toe",
        "l_toe",
        "r_heel",
        "l_heel",
        "r_small_toe",
        "l_small_toe",
    ):
        if opt in kpts:
            base[opt] = kpts[opt]

    return base, imputed


def _compute_pelvis_and_leg_markers(
    base: dict[str, np.ndarray],
    markers: dict[str, np.ndarray],
    imputed: set[str] | None = None,
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

    if "r_calc" in base:
        markers["r_calc_study"] = base["r_calc"].copy()
    elif "r_heel" in base:
        markers["r_calc_study"] = base["r_heel"].copy()
    else:
        markers["r_calc_study"] = r_ank + np.array([-0.06, -0.04, 0.0])

    if "r_toe" in base:
        markers["r_toe_study"] = base["r_toe"].copy()
    else:
        markers["r_toe_study"] = r_ank + np.array([0.16, -0.06, 0.0])
        if imputed is not None:
            imputed.add("r_toe")

    if "r_small_toe" in base:
        markers["r_5meta_study"] = base["r_small_toe"].copy()
    else:
        markers["r_5meta_study"] = r_ank + np.array([0.14, -0.06, 0.04])

    markers["L_knee_study"] = l_knee + np.array([0.0, 0.0, -0.05])
    markers["L_mknee_study"] = l_knee + np.array([0.0, 0.0, 0.05])
    markers["L_ankle_study"] = l_ank + np.array([0.0, 0.0, -0.04])
    markers["L_mankle_study"] = l_ank + np.array([0.0, 0.0, 0.04])

    if "l_calc" in base:
        markers["L_calc_study"] = base["l_calc"].copy()
    elif "l_heel" in base:
        markers["L_calc_study"] = base["l_heel"].copy()
    else:
        markers["L_calc_study"] = l_ank + np.array([-0.06, -0.04, 0.0])

    if "l_toe" in base:
        markers["L_toe_study"] = base["l_toe"].copy()
    else:
        markers["L_toe_study"] = l_ank + np.array([0.16, -0.06, 0.0])
        if imputed is not None:
            imputed.add("l_toe")

    if "l_small_toe" in base:
        markers["L_5meta_study"] = base["l_small_toe"].copy()
    else:
        markers["L_5meta_study"] = l_ank + np.array([0.14, -0.06, -0.04])


def _compute_arm_and_cluster_markers(
    base: dict[str, np.ndarray],
    markers: dict[str, np.ndarray],
    imputed: set[str] | None = None,
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

    if "c7" in base:
        markers["C7_study"] = base["c7"].copy()
    else:
        markers["C7_study"] = neck + np.array([-0.05, 0.02, 0.0])
        if imputed is not None:
            imputed.add("c7")

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
        height_m=height_m, mass_kg=mass_kg, synthetic_fallback=allow_fallback
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

    evidence_level = augmented_obs.metadata.get(
        "evidence_level",
        augmented_obs.source_provenance.get("evidence_level", "synthetic"),
    )

    return OpenCapPipelineResult(
        augmented_observations=augmented_obs,
        scaled_rig=scaled_rig,
        joint_trajectory=joint_traj,
        evidence_level=evidence_level,
        marker_set_name=OPENCAP_MARKER_SET_NAME,
        num_frames=len(augmented_obs.frames),
    )
