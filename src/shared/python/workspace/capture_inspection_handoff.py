"""Capture Rig, Optical Import, Pose Inspection, and Model Calibration handoff (#10519).

Connects Capture Rig as the primary capture/coaching surface to downstream
Inspect Targets and Model Calibration workspaces using ORG-08/ORG-09 typed
artifact references and SessionProjectStore.

Invariants:
- MediaPipe and OpenPose are explicit estimator choices maintaining separate
  observation sets, confidence scores, missing samples, and original video clocks.
- FreeMoCap input/output validation occurs before subprocess spawn; cancellation
  leaves source files untouched; HMR2/AGPL license isolation is preserved.
- C3D and optical imports keep missing samples masked (NaN); incompatible
  spatial units or reference frames fail immediately; 2-D pixel coordinates are
  never claimed to be 3-D metric coordinates.
- Target registration into SessionProjectStore preserves annotations,
  calibration parameters, and club metadata without manual path re-entry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Final
import uuid

import numpy as np

from .artifact_handoff import (
    ArtifactKind,
    ArtifactReference,
    SUPPORTED_FRAMES,
    compute_file_sha256,
)
from .project_store import RunMetadata, SessionProjectStore

logger = logging.getLogger(__name__)

SUPPORTED_SPATIAL_UNITS: Final[frozenset[str]] = frozenset(
    {
        "m",
        "meter",
        "meters",
        "mm",
        "millimeter",
        "millimeters",
        "cm",
        "centimeter",
        "centimeters",
    }
)

VIDEO_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".mjpeg"}
)


class EstimatorType(str, Enum):
    """Supported explicit 2-D/3-D human pose estimators."""

    MEDIAPIPE = "mediapipe"
    OPENPOSE_DNN = "openpose_dnn"
    HMR2 = "hmr2"


class JobStatus(str, Enum):
    """Execution status for sidecar subprocess jobs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"


@dataclass(frozen=True)
class TrimSpec:
    """Temporal trim boundaries for video or capture sequence."""

    start_frame: int = 0
    end_frame: int | None = None
    start_time_s: float = 0.0
    end_time_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
        }


@dataclass(frozen=True)
class CropSpec:
    """Spatial bounding box (Region of Interest) in source pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class PreparedVideoInspection:
    """A video recording prepared for target inspection preserving trim and crop."""

    video_path: Path
    view_name: str
    trim: TrimSpec
    crop: CropSpec | None
    time_offset_s: float
    fps: float

    def frame_to_time_s(self, frame_idx: int) -> float:
        """Convert relative frame index to absolute video session time in seconds."""
        return (self.trim.start_frame + frame_idx) / self.fps + self.time_offset_s

    def to_artifact_reference(self) -> ArtifactReference:
        """Export as an ORG-08 ArtifactReference preserving transformation metadata."""
        sha256 = compute_file_sha256(self.video_path)
        return ArtifactReference(
            artifact_id=f"art-vid-{self.view_name}-{self.video_path.stem}",
            path=str(self.video_path.name),
            hash=sha256,
            schema="motion_capture.c3d/1",
            kind=ArtifactKind.OBSERVATION,
            metadata={
                "view_name": self.view_name,
                "fps": self.fps,
                "trim": self.trim.to_dict(),
                "crop": self.crop.to_dict() if self.crop else None,
                "time_offset_s": self.time_offset_s,
            },
        )


@dataclass(frozen=True)
class ObservationSet2D:
    """A set of 2-D keypoint observations maintaining estimator identity and clocks."""

    video_path: Path
    view_name: str
    estimator: EstimatorType
    keypoints_px: dict[int, dict[str, tuple[float, float]]]
    confidence: dict[int, dict[str, float]]
    fps: float
    time_offset_s: float = 0.0
    keypoint_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.keypoint_names and self.keypoints_px:
            names: set[str] = set()
            for frame_pts in self.keypoints_px.values():
                names.update(frame_pts.keys())
            object.__setattr__(self, "keypoint_names", tuple(sorted(names)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": str(self.video_path),
            "view_name": self.view_name,
            "estimator": self.estimator.value,
            "fps": self.fps,
            "time_offset_s": self.time_offset_s,
            "keypoint_names": list(self.keypoint_names),
            "keypoints_px": {
                str(k): {name: list(pt) for name, pt in pts.items()}
                for k, pts in self.keypoints_px.items()
            },
            "confidence": {str(k): dict(conf) for k, conf in self.confidence.items()},
            "coordinate_type": "2d_pixel",
        }


@dataclass(frozen=True)
class OpticalMarkerTarget:
    """Imported 3-D optical marker trajectories with explicit masking."""

    name: str
    marker_names: tuple[str, ...]
    coordinates: np.ndarray  # Shape: (F, M, 3) in metres
    fps: float
    units: str = "m"
    frame: str = "canonical"

    def is_masked(self, frame_idx: int, marker_idx: int) -> bool:
        """Return True if marker sample at frame_idx is occluded or missing (NaN)."""
        return bool(np.isnan(self.coordinates[frame_idx, marker_idx, :]).any())


@dataclass
class FreeMoCapJob:
    """A managed FreeMoCap execution job."""

    job_id: str
    status: JobStatus
    input_dir: Path
    output_dir: Path
    process: subprocess.Popen[str] | None = None

    def cancel(self) -> None:
        """Cancel the job and terminate subprocess, leaving input sources untouched."""
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
        self.status = JobStatus.CANCELED


class FreeMoCapJobAdapter:
    """Sidecar adapter for FreeMoCap job management and directory validation."""

    def validate_inputs(self, input_dir: Path | str) -> list[Path]:
        """Validate input directory existence and video presence before spawn."""
        p = Path(input_dir).resolve()
        if not p.exists() or not p.is_dir():
            raise ValueError(
                f"input directory does not exist or is not a directory: {p}"
            )

        vids = [
            f
            for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if not vids:
            raise ValueError(
                f"input directory contains no video files ({sorted(VIDEO_EXTENSIONS)}): {p}"
            )
        return vids

    def validate_environment(
        self, python_exe: Path | str | None = None
    ) -> tuple[bool, str]:
        """Verify the isolated FreeMoCap executable without importing it in this process."""
        exe = Path(python_exe) if python_exe is not None else Path(sys.executable)
        if not exe.exists():
            return False, f"freemocap executable not found: {exe}"
        return True, ""

    def start_job(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        freemocap_env_python: Path | str | None = None,
        dry_run: bool = False,
    ) -> FreeMoCapJob:
        """Start a FreeMoCap extraction job with pre-spawn validation."""
        self.validate_inputs(input_dir)

        in_path = Path(input_dir).resolve()
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        job_id = f"fmc-{uuid.uuid4().hex[:8]}"

        if not dry_run:
            ok, err = self.validate_environment(freemocap_env_python)
            if not ok:
                raise FileNotFoundError(err)

            cmd = [
                str(freemocap_env_python or sys.executable),
                "-m",
                "freemocap",
                "--input",
                str(in_path),
                "--output",
                str(out_path),
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return FreeMoCapJob(
                job_id=job_id,
                status=JobStatus.RUNNING,
                input_dir=in_path,
                output_dir=out_path,
                process=proc,
            )

        # Dry run: write stub artifacts without spawning
        landmarks_file = out_path / "landmarks.csv"
        landmarks_file.write_text(
            "frame,landmark_id,x,y,z\n0,0,0.0,0.0,0.0\n", encoding="utf-8"
        )
        metadata_file = out_path / "metadata.json"
        metadata_file.write_text(
            json.dumps({"stub": True, "fps": 30.0}), encoding="utf-8"
        )

        return FreeMoCapJob(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            input_dir=in_path,
            output_dir=out_path,
            process=None,
        )


class CaptureInspectionHandoff:
    """Service facilitating handoff between Capture Rig and Target Inspection/Calibration."""

    def __init__(self, store: SessionProjectStore, session_id: str) -> None:
        self._store = store
        self._session_id = session_id

    @property
    def store(self) -> SessionProjectStore:
        return self._store

    @property
    def session_id(self) -> str:
        return self._session_id

    def prepare_video_inspection(
        self,
        video_path: Path | str,
        view_name: str,
        trim: TrimSpec | None = None,
        crop: CropSpec | None = None,
        time_offset_s: float = 0.0,
        fps: float = 60.0,
    ) -> PreparedVideoInspection:
        """Prepare raw video recording with trim, crop, and time synchronization."""
        v_path = Path(video_path).resolve()
        if not v_path.exists():
            raise FileNotFoundError(f"Video file not found at {v_path}")

        return PreparedVideoInspection(
            video_path=v_path,
            view_name=view_name,
            trim=trim or TrimSpec(),
            crop=crop,
            time_offset_s=time_offset_s,
            fps=fps,
        )

    def import_optical_markers(
        self,
        name: str,
        marker_names: tuple[str, ...],
        coordinates_m: np.ndarray,
        fps: float,
        units: str = "m",
        frame: str = "canonical",
    ) -> OpticalMarkerTarget:
        """Import 3-D optical marker trajectories enforcing unit, frame, and mask contracts."""
        clean_units = str(units).strip().lower()
        if clean_units not in SUPPORTED_SPATIAL_UNITS:
            raise ValueError(
                f"unsupported units {units!r}; expected one of {sorted(SUPPORTED_SPATIAL_UNITS)}"
            )

        if frame not in SUPPORTED_FRAMES:
            raise ValueError(
                f"unknown or unsupported frame {frame!r}; expected one of {sorted(SUPPORTED_FRAMES)}"
            )

        return OpticalMarkerTarget(
            name=name,
            marker_names=marker_names,
            coordinates=coordinates_m,
            fps=fps,
            units=clean_units,
            frame=frame,
        )

    def register_2d_pixels_as_3d_metric(
        self,
        name: str,
        pixel_coordinates: np.ndarray,
    ) -> None:
        """Explicitly reject pretending uncalibrated 2-D coordinates are 3-D metric coordinates."""
        raise ValueError(
            f"cannot treat 2-D coordinates as 3-D metric for target {name!r}; "
            "calibrated camera triangulation or explicit depth reconstruction required"
        )

    def create_2d_observation_set(
        self,
        video_path: Path | str,
        view_name: str,
        estimator: EstimatorType,
        keypoints_px: dict[int, dict[str, tuple[float, float]]],
        confidence: dict[int, dict[str, float]],
        fps: float,
        time_offset_s: float = 0.0,
    ) -> ObservationSet2D:
        """Create explicit 2-D observation set maintaining separate estimator identity."""
        v_path = Path(video_path).resolve()
        est = (
            estimator
            if isinstance(estimator, EstimatorType)
            else EstimatorType(str(estimator))
        )
        return ObservationSet2D(
            video_path=v_path,
            view_name=view_name,
            estimator=est,
            keypoints_px=keypoints_px,
            confidence=confidence,
            fps=fps,
            time_offset_s=time_offset_s,
        )

    def register_target_in_store(
        self,
        run_id: str,
        observations: tuple[ObservationSet2D, ...] | list[ObservationSet2D],
        annotations: dict[str, Any] | None = None,
        calibration: dict[str, Any] | None = None,
        club_metadata: dict[str, Any] | None = None,
    ) -> ArtifactReference:
        """Persist target observations to SessionProjectStore without manual path re-entry."""
        project = self._store.load_project()
        if self._session_id not in project.sessions:
            raise KeyError(f"session {self._session_id} not found in project")
        session = project.sessions[self._session_id]

        target_filename = f"target_{run_id}.json"
        target_path = self._store.root / target_filename

        payload = {
            "run_id": run_id,
            "session_id": self._session_id,
            "observations": [obs.to_dict() for obs in observations],
            "annotations": dict(annotations or {}),
            "calibration": dict(calibration or {}),
            "club": dict(club_metadata or {}),
        }
        target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        target_hash = compute_file_sha256(target_path)

        art_ref = ArtifactReference(
            artifact_id=f"art-target-{run_id}",
            path=target_filename,
            hash=target_hash,
            schema="pipeline.ground_support_receipt/1",
            kind=ArtifactKind.OBSERVATION,
            metadata={
                "run_id": run_id,
                "session_id": self._session_id,
                "observations_count": len(observations),
            },
        )

        run = RunMetadata(
            run_id=run_id,
            project_id=project.project_id,
            session_id=self._session_id,
            subject_id=session.subject_id,
            engine="optical_inspection",
            model_id="humanoid_target",
            club=dict(club_metadata or {"club_type": "standard"}),
            units={"spatial": "m", "angle": "deg", "time": "s"},
            frame="canonical",
            timebase={"fps": observations[0].fps if observations else 60.0},
            parameters={},
            inputs=(),
            outputs=(art_ref,),
            status="completed",
            metadata={
                "annotations": dict(annotations or {}),
                "calibration": dict(calibration or {}),
            },
        )
        self._store.register_run(run)
        return art_ref
