"""Recordings to per-view 2-D observations.

Ingest reads a session bundle, runs a registered pose estimator over each
recording, and writes one observation file per view using UpstreamDrift's
existing records (:class:`~pose_estimation.observations.KeypointObservation`,
:class:`~pose_estimation.observations.DetectorLayout`). Every frame's time is
the frame index over the recording's rate — a per-view clock; the session's
``timing`` block, copied into the index, is the evidence that relates those
clocks, and :mod:`.alignment` adds the reference-clock expression beside each
row when that evidence exists. Nothing here calls single-camera depth observed,
and ``CanonicalObservations`` — which requires camera calibrations — is left to
the calibration stage rather than assembled from invented intrinsics.
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

from ..provenance import write_stamped
from .alignment import (
    TIMING_REPORT_FILE,
    annotate_rows,
    build_timing_report,
    view_timing,
)
from .bundle import RecordingEntry, RecordingsIndex, load_bundle

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


class RegisteredFrameEstimator:
    """Adapter over any registered estimator that names its landmarks.

    The estimator instance or its class must expose ``LANDMARK_MAP``
    (index -> name; the instance wins, so variants can pick their landmark
    set at construction, #9648); an optional ``LAYOUT_NAME`` names the
    detector layout, else ``<estimator>_<count>`` is used.
    """

    def __init__(self, name: str = "mediapipe", **options: Any) -> None:
        from src.shared.python.pose_estimation.registry import create_estimator

        self.name = name
        self._options = dict(options)
        self._estimator = create_estimator(name, **options)
        self._estimator.load_model()
        # Instance attributes win: variant estimators (e.g. rtmpose_onnx
        # with keypoint_set="halpe26", #9648) pick their landmark set at
        # construction, while class attributes remain the fallback.
        names = getattr(self._estimator, "LANDMARK_MAP", None)
        if not isinstance(names, dict) or not names:
            raise ValueError(
                f"estimator {name!r} exposes no LANDMARK_MAP; cannot name keypoints"
            )
        ordered = [str(names[i]) for i in sorted(names)]
        layout_name = getattr(self._estimator, "LAYOUT_NAME", f"{name}_{len(ordered)}")
        self._layout = DetectorLayout(name=layout_name, keypoint_names=ordered)

    @property
    def layout(self) -> DetectorLayout:
        return self._layout

    @property
    def provenance(self) -> dict[str, Any]:
        from src.shared.python.pose_estimation.registry import get_estimator_info

        probe = get_estimator_info(self.name).probe_module
        try:
            module = __import__(probe)
        except ImportError:  # pragma: no cover - registry already probed it
            module = None
        model_path = getattr(self._estimator, "model_path", None)
        return {
            "estimator": self.name,
            f"{probe}_version": getattr(module, "__version__", None),
            "model_path": str(model_path) if model_path else None,
            "model_variant": getattr(self._estimator, "model_variant", None),
            "options": dict(self._options),
        }

    def estimate(self, image: np.ndarray, timestamp_ms: int) -> FramePose | None:
        result = self._estimator.estimate_from_image(image, timestamp_ms)  # type: ignore[call-arg]
        points = result.raw_keypoints
        if not points:
            return None
        confidences = result.raw_confidences or {}
        names = self._layout.keypoint_names
        # A detector may omit joints it did not find (OpenPose below its peak
        # threshold). KeypointObservation requires finite coordinates, so the
        # row keeps its shape with a (0, 0) placeholder and confidence 0.0:
        # confidence 0 means "unobserved" and consumers must gate on it.
        keypoints = np.array(
            [
                [points[n][0], points[n][1]] if n in points else [0.0, 0.0]
                for n in names
            ],
            dtype=float,
        )
        confidence = np.array(
            [
                confidences.get(n, result.confidence) if n in points else 0.0
                for n in names
            ],
            dtype=float,
        )
        return FramePose(
            keypoints_norm=keypoints, confidence=np.clip(confidence, 0.0, 1.0)
        )

    def close(self) -> None:
        close = getattr(self._estimator, "close", None)
        if callable(close):
            close()


EstimatorFactory = Callable[[], FrameEstimator]


MediaPipeFrameEstimator = RegisteredFrameEstimator  # backwards-compatible name


def registry_estimator_factory(name: str, **options: Any) -> EstimatorFactory:
    """Factory for a registered estimator; unknown names fail before any load."""
    from src.shared.python.pose_estimation.registry import list_estimators

    known = [info.name for info in list_estimators()]
    require(name in known, f"unknown estimator; registered: {known}", name)
    return lambda: RegisteredFrameEstimator(name, **options)


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
    timing_report: str | None = None
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


def _with_reference_clock(
    observations: ViewObservations, timing: dict[str, Any]
) -> ViewObservations:
    """Add reference-clock fields beside ``time_s`` when this view has an offset."""
    entry = view_timing(timing, observations.view)
    if not entry or entry.get("status") != "available":
        return observations
    rows = annotate_rows(
        list(observations.frames),
        int(entry["offset_ns"]),
        int(entry["uncertainty_ns"]),
        str(timing.get("method", "flash_event")),
    )
    provenance = {**observations.provenance, "timing_applied": timing.get("method")}
    return observations.model_copy(update={"frames": rows, "provenance": provenance})


def _write_timing_report(
    timing: dict[str, Any], index: RecordingsIndex, out_dir: Path
) -> str | None:
    """Write ``timing_report.json`` when the session carried strobe evidence."""
    if not timing:
        return None
    durations = {
        e.view: e.duration_s for e in index.recordings if e.duration_s is not None
    }
    report = build_timing_report(timing, durations)
    (out_dir / TIMING_REPORT_FILE).write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    return TIMING_REPORT_FILE


def _ingest_entry(
    entry: RecordingEntry,
    bundle_dir: Path,
    out_dir: Path,
    estimator: FrameEstimator,
    max_frames: int | None,
    timing: dict[str, Any],
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
    observations = _with_reference_clock(
        ingest_view(entry, bundle_dir / entry.file, estimator, max_frames=max_frames),
        timing,
    )
    target = out_dir / f"{entry.view}.json"
    write_stamped(
        target,
        observations.model_dump(mode="json"),
        schema_version=VIEW_OBSERVATIONS_SCHEMA_VERSION,
        module=__name__,
        inputs=[bundle_dir / entry.file],
        parameters={"view": entry.view},
        base=bundle_dir,
    )
    return ViewIngestStatus(
        view=entry.view,
        identity=entry.identity,
        status="available",
        file=target.name,
        frames_total=observations.frames_total,
        frames_with_pose=observations.frames_with_pose,
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
    bundle's reason rather than dropped. When the manifest carries a ``timing``
    block, rows gain reference-clock fields and ``timing_report.json`` is
    written. Postcondition: one status per plan view.
    """
    plan, index, manifest = load_bundle(bundle_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    estimator = estimator_factory()
    statuses: list[ViewIngestStatus] = []
    timing = dict(manifest.timing)
    try:
        for entry in index.recordings:
            statuses.append(
                _ingest_entry(entry, bundle_dir, out_dir, estimator, max_frames, timing)
            )
    finally:
        estimator.close()
    result = IngestIndex(
        plan_name=plan.name,
        views=tuple(statuses),
        timing=timing,
        timing_report=_write_timing_report(timing, index, out_dir),
        tools_schema=dict(manifest.tools_schema),
        provenance=estimator.provenance,
    )
    written = [
        out_dir / s.file for s in statuses if s.file and (out_dir / s.file).is_file()
    ]
    write_stamped(
        out_dir / INGEST_INDEX_FILE,
        result.model_dump(mode="json"),
        schema_version=VIEW_OBSERVATIONS_SCHEMA_VERSION,
        module=__name__,
        inputs=written,
        derived_from=written,
        base=bundle_dir,
    )
    logger.info("ingest %s -> %s", bundle_dir, out_dir)
    return result
