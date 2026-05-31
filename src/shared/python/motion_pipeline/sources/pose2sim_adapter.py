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

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    Keypoint,
    KeypointFrame,
    KeypointSequence,
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

    def __post_init__(self) -> None:
        if len(self.camera_observations) < 2:
            raise ValueError(
                "Pose2Sim observations require at least two camera streams"
            )
        if self.detector not in {item.value for item in Pose2SimDetector}:
            raise ValueError(f"Unsupported Pose2Sim detector: {self.detector!r}")


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

        triangulated = self._load_triangulated_sequence(
            root,
            loaded_calibration,
            camera_observations,
        )
        return Pose2SimObservations(
            camera_observations=camera_observations,
            calibration=loaded_calibration,
            detector=str(self.detector),
            triangulated=triangulated,
            metadata={"session_dir": str(root), "source": "Pose2Sim"},
        )

    def _load_camera_observations(
        self,
        root: Path,
        calibration: Calibration,
    ) -> dict[str, KeypointSequence]:
        adapter = self._camera_adapter()
        observations: dict[str, KeypointSequence] = {}
        for path in _find_detection_files(root, str(self.detector)):
            sequence = adapter.load(path, calibration=calibration)
            camera_id = _camera_id_from_detection_path(path)
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
    ) -> KeypointSequence | None:
        trc_path = _find_triangulated_trc(root)
        if trc_path is None:
            return None
        markers = TRCAdapter().load(trc_path, calibration=calibration)
        return _marker_trajectory_to_keypoints(markers, camera_observations)


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


def _marker_trajectory_to_keypoints(
    markers: MarkerTrajectory,
    camera_observations: dict[str, KeypointSequence],
) -> KeypointSequence:
    frames: list[KeypointFrame] = []
    for frame_index, marker_frame in enumerate(markers.frames):
        keypoints: list[Keypoint] = []
        for marker_index, marker in enumerate(marker_frame.markers.values()):
            keypoints.append(
                Keypoint(
                    x=marker.x,
                    y=marker.y,
                    z=marker.z,
                    confidence=_aggregate_confidence(
                        camera_observations,
                        frame_index,
                        marker.name,
                        marker_index,
                    ),
                    name=marker.name,
                )
            )
        frames.append(
            KeypointFrame(
                timestamp=marker_frame.timestamp,
                keypoints=keypoints,
                schema_name="custom",
                frame_index=marker_frame.frame_index,
            )
        )
    return KeypointSequence(
        id=f"pose2sim-{markers.id}",
        frames=frames,
        calibration=markers.calibration,
        metadata={**markers.metadata, "source": "Pose2Sim triangulated TRC"},
    )


def _aggregate_confidence(
    camera_observations: dict[str, KeypointSequence],
    frame_index: int,
    keypoint_name: str,
    keypoint_index: int,
) -> float:
    values: list[float] = []
    for sequence in camera_observations.values():
        if frame_index >= len(sequence.frames):
            continue
        frame = sequence.frames[frame_index]
        match = _find_keypoint(frame, keypoint_name, keypoint_index)
        if match is not None:
            values.append(float(match.confidence))
    if not values:
        return 1.0
    return sum(values) / len(values)


def _find_keypoint(
    frame: KeypointFrame,
    keypoint_name: str,
    keypoint_index: int,
) -> Keypoint | None:
    for keypoint in frame.keypoints:
        if keypoint.name == keypoint_name:
            return keypoint
    if keypoint_index < len(frame.keypoints):
        return frame.keypoints[keypoint_index]
    return None
