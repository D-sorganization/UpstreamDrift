"""Player-independent image observations, without physical-motion claims."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.shared.python.pose_estimation.interface import PoseEstimationResult


class ImageEstimator(Protocol):
    """Narrow detector seam used by the streaming capture runner."""

    def estimate_from_image(
        self, image: np.ndarray, timestamp_ms: int
    ) -> PoseEstimationResult: ...


@dataclass(frozen=True, kw_only=True)
class CaptureWindow:
    """Half-open interval in presentation seconds, never physical swing time."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start_s) and math.isfinite(self.end_s)):
            raise ValueError("Capture interval must be finite")
        if not 0 <= self.start_s < self.end_s:
            raise ValueError("Capture interval requires 0 <= start < end")

    def contains(self, presentation_time: Fraction) -> bool:
        """Return membership with an excluded end boundary."""
        return (
            Fraction(str(self.start_s)) <= presentation_time < Fraction(str(self.end_s))
        )


def observation_record(result: PoseEstimationResult) -> dict[str, Any]:
    """Export finite image XY observations, preserving missing visibility.

    Postconditions: no detector Z, inferred joint angles or wall-clock timestamp
    becomes anatomical evidence or physical time. Coordinates may be offscreen.
    """
    if not math.isfinite(result.confidence) or not 0 <= result.confidence <= 1:
        raise ValueError("Detector confidence must be finite and in [0, 1]")
    landmarks = {}
    confidences = result.raw_confidences or {}
    for name, point in (result.raw_keypoints or {}).items():
        values = np.asarray(point, dtype=float)
        if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
            raise ValueError(f"Invalid detector coordinates for {name}")
        visibility = confidences.get(name)
        if visibility is not None:
            if not math.isfinite(visibility) or not 0 <= visibility <= 1:
                raise ValueError(f"Invalid visibility for {name}")
            visibility = float(visibility)
        landmarks[name] = {
            "x": float(values[0]),
            "y": float(values[1]),
            "visibility": visibility,
        }
    return {
        "status": "detected" if landmarks else "missing",
        "coordinate_system": "normalized_image_xy",
        "confidence": float(result.confidence),
        "landmarks": landmarks,
        "physical_time_s": None,
        "physical_time_reason": "Historical playback scale is unverified",
    }


def export_capture(
    source: Path,
    destination: Path,
    window: CaptureWindow,
    *,
    subject_id: str,
    estimator: ImageEstimator,
    detector_identity: dict[str, Any],
) -> dict[str, Any]:
    """Stream a bounded window into lossless frames and source-bound records.

    Preconditions: output directory must not exist; decoder supplies increasing
    container PTS. Postconditions: completed receipt binds source, observations
    and detector identities. Partial failures have no completion receipt.
    Memory is bounded to one decoded image and detector state.
    """
    import av
    import cv2

    from .ingestion import compute_frame_hash, ingest_source_asset
    from .source_records import FrameIdentity

    if not isinstance(window, CaptureWindow):
        raise TypeError("window must be CaptureWindow")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not subject_id or subject_id.strip() != subject_id:
        raise ValueError("subject_id must be non-empty and trimmed")
    destination.mkdir(parents=True, exist_ok=False)
    count = detected = 0
    previous_pts: Fraction | None = None
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        asset = ingest_source_asset(
            source,
            asset_id=f"{subject_id}-source",
            width_px=stream.width,
            height_px=stream.height,
        )
        source_id = f"source-{asset.content_sha256}"
        asset = replace(asset, asset_id=source_id, source_uri=f"urn:asset:{source_id}")
        # Seek to the preceding keyframe, then discard pre-window frames.
        container.seek(int(window.start_s / stream.time_base), stream=stream)
        with (destination / "observations.jsonl").open("w", encoding="utf-8") as output:
            for decoded in container.decode(stream):
                if decoded.pts is None or decoded.time_base is None:
                    raise ValueError("Decoder did not supply container PTS")
                presentation = decoded.pts * decoded.time_base
                if presentation >= window.end_s:
                    break
                if not window.contains(presentation):
                    continue
                if previous_pts is not None and presentation <= previous_pts:
                    raise ValueError("Non-increasing container PTS")
                previous_pts = presentation
                image = decoded.to_ndarray(format="bgr24")
                frame_id = f"frame-{decoded.pts}"
                identity = FrameIdentity(
                    schema_version="shadow-tracker/frame/1.1.0",
                    asset_id=asset.asset_id,
                    shot_id=f"{subject_id}-window",
                    swing_id=f"{subject_id}-unreviewed",
                    camera_id="source-camera",
                    frame_id=frame_id,
                    pts_ticks=decoded.pts,
                    timebase_numerator=decoded.time_base.numerator,
                    timebase_denominator=decoded.time_base.denominator,
                    physical_time_s=None,
                    physical_time_reason="Historical playback scale is unverified",
                    frame_sha256=compute_frame_hash(
                        image.tobytes(), decoder_name="pyav"
                    ),
                    timing_mode="container_pts",
                    is_timing_exact=True,
                    clock_evidence="Decoded original container PTS; physical clock unknown",
                    decoder_name="pyav",
                    decoder_version=av.__version__,
                    pixel_format="bgr24",
                )
                result = estimator.estimate_from_image(image, int(presentation * 1000))
                observation = observation_record(result)
                name = f"{frame_id}.png"
                if not cv2.imwrite(
                    str(destination / name), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]
                ):
                    raise OSError(f"Could not write {name}")
                row = {
                    "frame": identity.to_dict(),
                    "image": name,
                    "observation": observation,
                }
                output.write(json.dumps(row, allow_nan=False) + "\n")
                count += 1
                detected += observation["status"] == "detected"
    if not count:
        raise ValueError("Capture window contains no decoded frames")
    observations_path = destination / "observations.jsonl"
    with observations_path.open("rb") as handle:
        observations_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    receipt = {
        "schema_version": "historical-capture/1.0.0",
        "subject_id": subject_id,
        "source": asset.to_dict(),
        "window_presentation_s": [window.start_s, window.end_s],
        "frame_count": count,
        "detected_count": detected,
        "observations_sha256": observations_sha256,
        "detector": detector_identity,
        "qualification": "image_observations_only",
        "recording_year": None,
        "physical_time_verified": False,
        "shot_continuity_reviewed": False,
        "publication_permitted": False,
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    (destination / "receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return receipt
