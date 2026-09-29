"""Pose2Sim session ingestion into canonical motion-pipeline observations.

Pose2Sim's local pipeline commonly writes one 2-D detection stream per
camera plus a triangulated OpenSim TRC file. This adapter keeps both pieces:
camera-wise 2-D keypoints retain per-keypoint detector confidence, while the
optional 3-D sequence receives an aggregate confidence derived from the
supporting camera observations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    CanonicalObservationFrame,
    CanonicalObservations,
    Keypoint,
    KeypointFrame,
    KeypointSequence,
    Marker,
    MarkerTrajectory,
)
from src.shared.python.motion_pipeline.sources.mediapipe_json_adapter import (
    MediaPipeJSONAdapter,
)
from src.shared.python.motion_pipeline.sources.openpose_json_adapter import (
    OpenPoseJSONAdapter,
)
from src.shared.python.motion_pipeline.sources.trc_adapter import TRCAdapter


class Pose2SimDetector(str, Enum):
    """Supported detector layouts for Pose2Sim ingestion."""

    MEDIAPIPE = "mediapipe"
    OPENPOSE = "openpose"


@dataclass(frozen=True)
class KeypointQualityRecord:
    """Per-marker reconstruction quality assessment from multi-view observations.

    Tracks explicit unknown/invalid quality and reasons, preserving individual
    per-view detector confidences without presenting an arithmetic mean as a
    calibrated probability.
    """

    quality: str  # "valid", "invalid", "unknown"
    reason: (
        str | None
    )  # None, "missing_observation", "insufficient_views", "timestamp_drift"
    contributing_views: int
    view_confidences: dict[str, float]
    reconstruction_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality": self.quality,
            "reason": self.reason,
            "contributing_views": self.contributing_views,
            "view_confidences": dict(self.view_confidences),
            "reconstruction_confidence": self.reconstruction_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeypointQualityRecord:
        return cls(
            quality=str(data["quality"]),
            reason=data.get("reason"),
            contributing_views=int(data["contributing_views"]),
            view_confidences=dict(data.get("view_confidences", {})),
            reconstruction_confidence=float(data.get("reconstruction_confidence", 0.0)),
        )


@dataclass(frozen=True)
class Pose2SimObservations:
    """Canonical observation bundle produced from a Pose2Sim session.

    ``KeypointSequence`` is the current public observation CIR on ``origin/main``.
    CC-12's future ``CanonicalObservations`` schema can wrap the same fields
    without losing confidence or calibration data.
    """

    camera_observations: dict[str, KeypointSequence]
    calibration: Calibration
    detector: str
    triangulated: KeypointSequence | None = None
    metadata: dict[str, object] | None = None
    reconstruction_quality: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if len(self.camera_observations) < 2:
            raise ValueError(
                "Pose2Sim observations require at least two camera streams"
            )
        if self.detector not in {item.value for item in Pose2SimDetector}:
            raise ValueError(f"Unsupported Pose2Sim detector: {self.detector!r}")

    def get_keypoint_quality(
        self, frame_index: int, marker_name: str
    ) -> KeypointQualityRecord | None:
        """Retrieve the quality record for a specific frame and marker."""
        if not self.reconstruction_quality:
            return None
        for f_idx, frame in enumerate(self.reconstruction_quality):
            if frame.get("frame_index") == frame_index or f_idx == frame_index:
                marker_data = frame.get("markers", {}).get(marker_name)
                if marker_data:
                    return KeypointQualityRecord.from_dict(marker_data)
        return None

    def to_canonical_observations(self) -> CanonicalObservations:
        """Convert Pose2Sim observations into the canonical observation stream (#9422)."""
        frames: list[CanonicalObservationFrame] = []
        if self.triangulated is not None:
            for f_idx, k_frame in enumerate(self.triangulated.frames):
                markers = {
                    kp.name or f"marker_{i}": Marker(
                        name=kp.name or f"marker_{i}",
                        x=kp.x,
                        y=kp.y,
                        z=kp.z if kp.z is not None else 0.0,
                    )
                    for i, kp in enumerate(k_frame.keypoints)
                }
                frames.append(
                    CanonicalObservationFrame(
                        timestamp=k_frame.timestamp,
                        frame_index=(
                            k_frame.frame_index
                            if k_frame.frame_index is not None
                            else f_idx
                        ),
                        markers=markers,
                        keypoints=k_frame.keypoints,
                    )
                )
        return CanonicalObservations(
            id=f"pose2sim-{self.calibration.id}",
            frames=frames,
            calibration=self.calibration,
            marker_set_name="Pose2Sim-OpenSim",
            metadata=dict(self.metadata or {}),
        )


@dataclass(frozen=True)
class Pose2SimAdapter:
    """Load Pose2Sim multi-camera outputs into canonical CIR objects."""

    detector: Pose2SimDetector | str = Pose2SimDetector.MEDIAPIPE
    fps: float = 30.0

    def __post_init__(self) -> None:
        detector = Pose2SimDetector(self.detector)
        object.__setattr__(self, "detector", detector.value)
        if self.fps <= 0:
            raise ValueError("Pose2Sim fps must be positive")

    def load(
        self,
        session_dir: Path,
        calibration: Calibration | None = None,
    ) -> Pose2SimObservations:
        """Load a Pose2Sim session directory.

        Postconditions: at least two camera streams are returned; calibration is
        attached to each stream; any triangulated TRC is converted to a 3-D
        ``KeypointSequence`` with confidence aggregated from camera detections.
        """
        root = Path(session_dir)
        if not root.exists():
            raise FileNotFoundError(f"Pose2Sim session not found: {root}")
        if not root.is_dir():
            raise ValueError(f"Pose2Sim session must be a directory: {root}")

        loaded_calibration = calibration or load_pose2sim_calibration(root)
        camera_observations = self._load_camera_observations(root, loaded_calibration)
        if len(camera_observations) < 2:
            raise ValueError("Pose2Sim ingest requires at least two camera streams")

        triangulated, quality_records = self._load_triangulated_sequence(
            root,
            loaded_calibration,
            camera_observations,
        )
        metadata: dict[str, object] = {
            "session_dir": str(root),
            "source": "Pose2Sim",
            "fps": self.fps,
            "reconstruction_quality": quality_records,
        }
        return Pose2SimObservations(
            camera_observations=camera_observations,
            calibration=loaded_calibration,
            detector=str(self.detector),
            triangulated=triangulated,
            metadata=metadata,
            reconstruction_quality=quality_records,
        )

    def _load_camera_observations(
        self,
        root: Path,
        calibration: Calibration,
    ) -> dict[str, KeypointSequence]:
        adapter = self._camera_adapter()
        observations: dict[str, KeypointSequence] = {}
        for path in _find_detection_files(root, str(self.detector)):
            camera_id = _camera_id_from_detection_path(path)
            if camera_id in observations:
                raise ValueError(
                    f"Duplicate camera stream detected for camera {camera_id!r} at {path}"
                )
            sequence = adapter.load(path, calibration=calibration)
            metadata = {
                **sequence.metadata,
                "camera_id": camera_id,
                "pose2sim_detector": str(self.detector),
            }
            observations[camera_id] = sequence.model_copy(update={"metadata": metadata})
        return dict(sorted(observations.items()))

    def _camera_adapter(self) -> MediaPipeJSONAdapter | OpenPoseJSONAdapter:
        if self.detector == Pose2SimDetector.OPENPOSE.value:
            return OpenPoseJSONAdapter(fps=self.fps)
        return MediaPipeJSONAdapter()

    def _load_triangulated_sequence(
        self,
        root: Path,
        calibration: Calibration,
        camera_observations: dict[str, KeypointSequence],
    ) -> tuple[KeypointSequence | None, list[dict[str, Any]] | None]:
        trc_path = _find_triangulated_trc(root)
        if trc_path is None:
            return None, None
        markers = TRCAdapter().load(trc_path, calibration=calibration)
        return _marker_trajectory_to_keypoints(
            markers, camera_observations, fps=self.fps
        )


def load_pose2sim_observations(
    session_dir: Path,
    *,
    detector: Pose2SimDetector | str = Pose2SimDetector.MEDIAPIPE,
    fps: float = 30.0,
    calibration: Calibration | None = None,
) -> Pose2SimObservations:
    """Load a Pose2Sim session with a permissive MediaPipe detector default."""
    return Pose2SimAdapter(detector=detector, fps=fps).load(
        session_dir,
        calibration=calibration,
    )


def load_pose2sim_calibration(session_dir: Path) -> Calibration:
    """Load Pose2Sim calibration JSON from a session directory."""
    root = Path(session_dir)
    candidates = (
        root / "calibration.json",
        root / "camera_calibration.json",
        root / "pose2sim_calibration.json",
    )
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return Calibration.model_validate(data)
    raise FileNotFoundError(
        "Pose2Sim calibration not found; expected calibration.json, "
        "camera_calibration.json, or pose2sim_calibration.json"
    )


def _find_detection_files(root: Path, detector: str) -> list[Path]:
    directories = (root / "detections", root / "pose-2d", root / "pose2d")
    files: list[Path] = []
    for directory in directories:
        if directory.exists():
            files.extend(directory.rglob("*.json"))
    if not files:
        files.extend(
            path
            for path in root.rglob("*.json")
            if path.name not in {"calibration.json", "camera_calibration.json"}
        )
    if detector == Pose2SimDetector.OPENPOSE.value:
        return [path for path in files if OpenPoseJSONAdapter.supports(path)]
    return [path for path in files if MediaPipeJSONAdapter.supports(path)]


def _camera_id_from_detection_path(path: Path) -> str:
    stem = path.stem
    for suffix in ("_mediapipe", "_openpose", "_keypoints"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    parent = path.parent.name
    if parent.lower() not in {"detections", "pose-2d", "pose2d"}:
        return parent
    return stem


def _find_triangulated_trc(root: Path) -> Path | None:
    preferred_dirs = (root / "pose-3d", root / "pose3d", root / "triangulated")
    for directory in preferred_dirs:
        if directory.exists():
            matches = sorted(directory.glob("*.trc"))
            if matches:
                return matches[0]
    matches = sorted(root.rglob("*.trc"))
    return matches[0] if matches else None


MEDIAPIPE_33_LANDMARK_MAP: dict[int, str] = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

BODY_25_KEYPOINT_MAP: dict[int, str] = {
    0: "nose",
    1: "neck",
    2: "right_shoulder",
    3: "right_elbow",
    4: "right_wrist",
    5: "left_shoulder",
    6: "left_elbow",
    7: "left_wrist",
    8: "mid_hip",
    9: "right_hip",
    10: "right_knee",
    11: "right_ankle",
    12: "left_hip",
    13: "left_knee",
    14: "left_ankle",
    15: "right_eye",
    16: "left_eye",
    17: "right_ear",
    18: "left_ear",
    19: "left_big_toe",
    20: "left_small_toe",
    21: "left_heel",
    22: "right_big_toe",
    23: "right_small_toe",
    24: "right_heel",
}


def _normalize_name(name: str) -> str:
    cleaned = name.lower().strip().replace(" ", "_").replace("-", "_")
    prefix_map = {
        "r_": "right_",
        "l_": "left_",
    }
    for pre, repl in prefix_map.items():
        if cleaned.startswith(pre):
            cleaned = repl + cleaned[len(pre) :]
            break
    single_letter_map = {
        "rshoulder": "right_shoulder",
        "lshoulder": "left_shoulder",
        "relbow": "right_elbow",
        "lelbow": "left_elbow",
        "rwrist": "right_wrist",
        "lwrist": "left_wrist",
        "rhip": "right_hip",
        "lhip": "left_hip",
        "rknee": "right_knee",
        "lknee": "left_knee",
        "rankle": "right_ankle",
        "lankle": "left_ankle",
        "rheel": "right_heel",
        "lheel": "left_heel",
        "reye": "right_eye",
        "leye": "left_eye",
        "rear": "right_ear",
        "lear": "left_ear",
    }
    return single_letter_map.get(cleaned, cleaned)


def _find_frame_by_timestamp(
    sequence: KeypointSequence,
    target_time: float,
    tolerance_s: float,
) -> KeypointFrame | None:
    if not sequence.frames:
        return None
    best_frame: KeypointFrame | None = None
    best_diff = float("inf")
    for frame in sequence.frames:
        diff = abs(frame.timestamp - target_time)
        if diff <= tolerance_s and diff < best_diff:
            best_diff = diff
            best_frame = frame
    return best_frame


def _find_keypoint(
    frame: KeypointFrame,
    keypoint_name: str,
    keypoint_index: int,
) -> Keypoint | None:
    target_norm = _normalize_name(keypoint_name)

    # 1. Search existing named keypoints
    for kp in frame.keypoints:
        if kp.name is not None and _normalize_name(kp.name) == target_norm:
            return kp

    # 2. If keypoints lack names, map by schema ONLY if schema is explicitly supported
    schema_map: dict[int, str] | None = None
    if frame.schema_name == "MediaPipe_33":
        schema_map = MEDIAPIPE_33_LANDMARK_MAP
    elif frame.schema_name in {"BODY_25", "OpenPose_25"}:
        schema_map = BODY_25_KEYPOINT_MAP

    if schema_map is not None:
        for idx, kp in enumerate(frame.keypoints):
            if kp.name is None:
                mapped_name = schema_map.get(idx)
                if mapped_name and _normalize_name(mapped_name) == target_norm:
                    return kp

    # Never fall back to positional index across incompatible schemas or unknown names
    return None


def assess_reconstruction_quality(
    camera_observations: dict[str, KeypointSequence],
    frame_index: int,
    keypoint_name: str,
    keypoint_index: int,
    timestamp: float | None = None,
    tolerance_s: float = 0.02,
    min_views: int = 2,
) -> KeypointQualityRecord:
    view_confidences: dict[str, float] = {}

    for camera_id, sequence in camera_observations.items():
        if timestamp is not None:
            frame = _find_frame_by_timestamp(sequence, timestamp, tolerance_s)
        else:
            frame = (
                sequence.frames[frame_index]
                if frame_index < len(sequence.frames)
                else None
            )

        if frame is None:
            continue

        match = _find_keypoint(frame, keypoint_name, keypoint_index)
        if match is not None:
            view_confidences[camera_id] = float(match.confidence)

    contributing_views = len(view_confidences)

    if contributing_views == 0:
        return KeypointQualityRecord(
            quality="unknown",
            reason="missing_observation",
            contributing_views=0,
            view_confidences={},
            reconstruction_confidence=0.0,
        )

    if contributing_views < min_views:
        return KeypointQualityRecord(
            quality="invalid",
            reason="insufficient_views",
            contributing_views=contributing_views,
            view_confidences=view_confidences,
            reconstruction_confidence=0.0,
        )

    mean_conf = sum(view_confidences.values()) / contributing_views
    return KeypointQualityRecord(
        quality="valid",
        reason=None,
        contributing_views=contributing_views,
        view_confidences=view_confidences,
        reconstruction_confidence=mean_conf,
    )


def _aggregate_confidence(
    camera_observations: dict[str, KeypointSequence],
    frame_index: int,
    keypoint_name: str,
    keypoint_index: int,
    timestamp: float | None = None,
    tolerance_s: float = 0.02,
    min_views: int = 2,
) -> float:
    record = assess_reconstruction_quality(
        camera_observations=camera_observations,
        frame_index=frame_index,
        keypoint_name=keypoint_name,
        keypoint_index=keypoint_index,
        timestamp=timestamp,
        tolerance_s=tolerance_s,
        min_views=min_views,
    )
    return record.reconstruction_confidence


def _marker_trajectory_to_keypoints(
    markers: MarkerTrajectory,
    camera_observations: dict[str, KeypointSequence],
    fps: float = 30.0,
) -> tuple[KeypointSequence, list[dict[str, Any]]]:
    tolerance_s = 0.5 / fps if fps > 0 else 0.02
    frames: list[KeypointFrame] = []
    quality_records: list[dict[str, Any]] = []

    for frame_index, marker_frame in enumerate(markers.frames):
        keypoints: list[Keypoint] = []
        frame_quality: dict[str, Any] = {}
        for marker_index, marker in enumerate(marker_frame.markers.values()):
            q_rec = assess_reconstruction_quality(
                camera_observations=camera_observations,
                frame_index=frame_index,
                keypoint_name=marker.name,
                keypoint_index=marker_index,
                timestamp=marker_frame.timestamp,
                tolerance_s=tolerance_s,
                min_views=2,
            )
            frame_quality[marker.name] = q_rec.to_dict()
            keypoints.append(
                Keypoint(
                    x=marker.x,
                    y=marker.y,
                    z=marker.z,
                    confidence=q_rec.reconstruction_confidence,
                    name=marker.name,
                )
            )
        quality_records.append(
            {
                "frame_index": marker_frame.frame_index,
                "timestamp": marker_frame.timestamp,
                "markers": frame_quality,
            }
        )
        frames.append(
            KeypointFrame(
                timestamp=marker_frame.timestamp,
                keypoints=keypoints,
                schema_name="custom",
                frame_index=marker_frame.frame_index,
            )
        )
    sequence = KeypointSequence(
        id=f"pose2sim-{markers.id}",
        frames=frames,
        calibration=markers.calibration,
        metadata={
            **markers.metadata,
            "source": "Pose2Sim triangulated TRC",
            "reconstruction_quality": quality_records,
            "sync_tolerance_s": tolerance_s,
            "raw_marker_count": len(markers.frames[0].markers) if markers.frames else 0,
        },
    )
    return sequence, quality_records
