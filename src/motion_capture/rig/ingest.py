"""Recordings to per-view 2-D observations.

Ingest reads a session bundle, runs a registered pose estimator over each
recording, and writes one observation file per view using UpstreamDrift's
existing records (:class:`~pose_estimation.observations.KeypointObservation`,
:class:`~pose_estimation.observations.DetectorLayout`). Every frame's time is
the frame index over the recording's rate — a per-view clock; the session's
``timing`` block, copied into the index, is the evidence that relates those
clocks. Nothing here calls single-camera depth observed, and
``CanonicalObservations`` — which requires camera calibrations — is left to the
calibration stage rather than assembled from invented intrinsics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.shared.python.core.contracts import StateError, require
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.pose_estimation.observations import (
    DetectorLayout,
    KeypointObservation,
)

from .bundle import RecordingEntry, load_bundle

logger = get_logger(__name__)

VIEW_OBSERVATIONS_SCHEMA_VERSION = "view-observations/1.0.0"
INGEST_INDEX_FILE = "observations.json"


@dataclass(frozen=True)
class FramePose:
    """One frame's detections in normalized image coordinates."""

    keypoints_norm: np.ndarray  # (K, 2) in [0, 1]
    confidence: np.ndarray  # (K,) in [0, 1]


class FrameEstimator(Protocol):
    """What ingest needs from a pose estimator."""

    @property
    def layout(self) -> DetectorLayout: ...

    @property
    def provenance(self) -> dict[str, Any]: ...

    def estimate(self, image: np.ndarray, timestamp_ms: int) -> FramePose | None: ...

    def close(self) -> None: ...


class MediaPipeFrameEstimator:
    """Adapter over the registered ``mediapipe`` estimator (Tasks API)."""

    def __init__(self, **options: Any) -> None:
        from src.shared.python.pose_estimation.registry import create_estimator

        self._estimator = create_estimator("mediapipe", **options)
        self._estimator.load_model()
        names = getattr(type(self._estimator), "LANDMARK_MAP", None)
        if not isinstance(names, dict) or not names:
            raise ValueError("estimator exposes no LANDMARK_MAP; cannot name keypoints")
        ordered = [str(names[i]) for i in sorted(names)]
        self._layout = DetectorLayout(name="mediapipe_pose_33", keypoint_names=ordered)

    @property
    def layout(self) -> DetectorLayout:
        return self._layout

    @property
    def provenance(self) -> dict[str, Any]:
        import mediapipe

        model_path = getattr(self._estimator, "model_path", None)
        return {
            "estimator": "mediapipe",
            "mediapipe_version": getattr(mediapipe, "__version__", None),
            "model_path": str(model_path) if model_path else None,
            "model_variant": getattr(self._estimator, "model_variant", None),
        }

    def estimate(self, image: np.ndarray, timestamp_ms: int) -> FramePose | None:
        result = self._estimator.estimate_from_image(image, timestamp_ms)  # type: ignore[call-arg]
        points = result.raw_keypoints
        if not points:
            return None
        confidences = result.raw_confidences or {}
        names = self._layout.keypoint_names
        keypoints = np.array([[points[n][0], points[n][1]] for n in names], dtype=float)
        confidence = np.array(
            [confidences.get(n, result.confidence) for n in names], dtype=float
        )
        return FramePose(
            keypoints_norm=keypoints, confidence=np.clip(confidence, 0.0, 1.0)
        )

    def close(self) -> None:
        close = getattr(self._estimator, "close", None)
        if callable(close):
            close()


EstimatorFactory = Callable[[], FrameEstimator]


def registry_estimator_factory(name: str, **options: Any) -> EstimatorFactory:
    """Factory for the named registered estimator (only ``mediapipe`` is adapted)."""
    require(
        name == "mediapipe", "only the mediapipe estimator is adapted for ingest", name
    )
    return lambda: MediaPipeFrameEstimator(**options)


class ViewObservations(BaseModel):
    """``observations/<view>.json``: one view's 2-D detections and their provenance."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = VIEW_OBSERVATIONS_SCHEMA_VERSION
    view: str
    identity: str
    camera_id: str
    fps: float
    width: int | None
    height: int | None
    frames_total: int
    frames_with_pose: int
    detector_layout: dict[str, Any]
    frames: tuple[dict[str, Any], ...]
    provenance: dict[str, Any] = Field(default_factory=dict)


class ViewIngestStatus(BaseModel):
    """One row of ``observations.json``."""

    model_config = ConfigDict(frozen=True)

    view: str
    identity: str
    status: str  # available | unavailable
    file: str | None = None
    frames_total: int = 0
    frames_with_pose: int = 0
    reason: str | None = None


class IngestIndex(BaseModel):
    """``observations.json``."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = VIEW_OBSERVATIONS_SCHEMA_VERSION
    plan_name: str
    views: tuple[ViewIngestStatus, ...]
    timing: dict[str, Any] = Field(default_factory=dict)
    tools_schema: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


def iter_video_frames(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, BGR image)`` for every frame ffmpeg wrote."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise StateError(f"could not open recording {path}")
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                return
            yield index, frame
            index += 1
    finally:
        cap.release()


def _rate_for(entry: RecordingEntry) -> float:
    return entry.achieved_fps or float(entry.requested_mode.fps)


def ingest_view(
    entry: RecordingEntry,
    video_path: Path,
    estimator: FrameEstimator,
    *,
    max_frames: int | None = None,
) -> ViewObservations:
    """Estimate every frame of one recording into :class:`KeypointObservation` rows.

    ``time_s`` is ``frame_index / rate`` with the rate taken from the probe when
    present, else the requested mode — a per-view clock, not the session clock.
    Postcondition: ``frames_with_pose <= frames_total``.
    """
    require(video_path.is_file(), "recording must exist", str(video_path))
    require(
        max_frames is None or max_frames > 0, "max_frames must be positive", max_frames
    )
    rate = _rate_for(entry)
    layout = estimator.layout
    rows: list[dict[str, Any]] = []
    total = 0
    for index, image in iter_video_frames(video_path):
        if max_frames is not None and index >= max_frames:
            break
        total = index + 1
        pose = estimator.estimate(image, int(round(index * 1000.0 / rate)))
        if pose is None:
            continue
        height, width = image.shape[:2]
        obs = KeypointObservation(
            camera_id=entry.identity,
            time_s=index / rate,
            keypoints_px=pose.keypoints_norm * np.array([width, height], dtype=float),
            confidence=pose.confidence,
        )
        rows.append(obs.to_dict())
    return ViewObservations(
        view=entry.view,
        identity=entry.identity,
        camera_id=entry.identity,
        fps=rate,
        width=entry.width,
        height=entry.height,
        frames_total=total,
        frames_with_pose=len(rows),
        detector_layout=layout.to_dict(),
        frames=tuple(rows),
        provenance={
            **estimator.provenance,
            "requested_mode": entry.requested_mode.model_dump(),
        },
    )


def ingest_bundle(
    bundle_dir: Path,
    out_dir: Path,
    estimator_factory: EstimatorFactory,
    *,
    max_frames: int | None = None,
) -> IngestIndex:
    """Ingest every recording of a bundle; write per-view files and the index.

    Views whose recording failed are recorded as ``unavailable`` with the
    bundle's reason rather than dropped. Postcondition: one status per plan view.
    """
    plan, index, manifest = load_bundle(bundle_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    estimator = estimator_factory()
    statuses: list[ViewIngestStatus] = []
    try:
        for entry in index.recordings:
            statuses.append(
                _ingest_entry(entry, bundle_dir, out_dir, estimator, max_frames)
            )
    finally:
        estimator.close()
    result = IngestIndex(
        plan_name=plan.name,
        views=tuple(statuses),
        timing=dict(manifest.timing),
        tools_schema=dict(manifest.tools_schema),
        provenance=estimator.provenance,
    )
    (out_dir / INGEST_INDEX_FILE).write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("ingest %s -> %s", bundle_dir, out_dir)
    return result


def _ingest_entry(
    entry: RecordingEntry,
    bundle_dir: Path,
    out_dir: Path,
    estimator: FrameEstimator,
    max_frames: int | None,
) -> ViewIngestStatus:
    if not entry.ok:
        reason = (
            f"recording not usable (returncode={entry.returncode}, bytes={entry.bytes})"
        )
        return ViewIngestStatus(
            view=entry.view,
            identity=entry.identity,
            status="unavailable",
            reason=reason,
        )
    observations = ingest_view(
        entry, bundle_dir / entry.file, estimator, max_frames=max_frames
    )
    target = out_dir / f"{entry.view}.json"
    target.write_text(observations.model_dump_json(indent=2), encoding="utf-8")
    return ViewIngestStatus(
        view=entry.view,
        identity=entry.identity,
        status="available",
        file=target.name,
        frames_total=observations.frames_total,
        frames_with_pose=observations.frames_with_pose,
    )
