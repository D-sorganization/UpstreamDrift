"""Exact capture clock and decoded/encoded image identities share no hash domain."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np

from src.shared.python.shadow_tracker.ingestion import compute_frame_hash
from src.shared.python.shadow_tracker.source_records import FrameIdentity, SourceAsset
from src.shared.python.shadow_tracker import check_sha256
from .project_store import validate_workspace_id
from .necromatcher import NecromatcherLibrary
from .necromatcher_review import CaptureReview


def capture_clock_sha256(review: CaptureReview) -> str:
    """Preserve the canonical complete frame-ID/PTS clock digest byte convention."""
    clock = []
    for index in range(review.frame_count):
        frame = FrameIdentity.from_dict(review.frame(index)["frame"])
        clock.append(
            {
                "frame_index": index,
                "frame_id": frame.frame_id,
                "pts_ticks": frame.pts_ticks,
                "timebase_numerator": frame.timebase_numerator,
                "timebase_denominator": frame.timebase_denominator,
            }
        )
    raw = json.dumps(
        clock, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class CaptureIdentity:
    """A captured immutable source receipt, full frame identities and PNG hashes."""

    capture_id: str
    capture_hash: str
    source: SourceAsset
    source_clock_sha256: str
    frames: tuple[FrameIdentity, ...]
    png_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        """Copy containers; scalar source/frame DTOs remain frozen and detached."""
        validate_workspace_id(self.capture_id, "capture_id")
        for name, value in (
            ("capture_hash", self.capture_hash),
            ("source_clock_sha256", self.source_clock_sha256),
            *(("png_sha256", digest) for digest in self.png_sha256),
        ):
            if not isinstance(value, str) or not value.startswith("sha256:"):
                raise ValueError(f"{name} requires a prefixed SHA256")
            check_sha256(value.removeprefix("sha256:"), name)
        frames, pngs = tuple(self.frames), tuple(self.png_sha256)
        if (
            not isinstance(self.source, SourceAsset)
            or not frames
            or len(frames) != len(pngs)
            or any(not isinstance(frame, FrameIdentity) for frame in frames)
            or any(not isinstance(digest, str) for digest in pngs)
        ):
            raise ValueError("Capture identity requires aligned typed frame/PNG tuples")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "png_sha256", pngs)

    @property
    def source_sha256(self) -> str:
        return "sha256:" + self.source.content_sha256

    def to_record(self) -> dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "capture_hash": self.capture_hash,
            "source": self.source.to_dict(),
            "source_clock_sha256": self.source_clock_sha256,
            "frames": [frame.to_dict() for frame in self.frames],
            "png_sha256": list(self.png_sha256),
        }


def capture_identity(library: NecromatcherLibrary, capture_id: str) -> CaptureIdentity:
    """Authenticate every original PNG/decoded frame and exact nonphysical PTS."""
    from .necromatcher_authenticated_read import _read_capture_identity

    return _read_capture_identity(
        library, capture_id, lambda: _authenticate_capture_identity(library, capture_id)
    )


def _authenticate_capture_identity(
    library: NecromatcherLibrary, capture_id: str
) -> CaptureIdentity:
    """The unchanged full authentication algorithm, including fresh asset checks."""
    import cv2

    asset = library.load_asset(capture_id)
    if asset.kind != "image_capture":
        raise ValueError("Hypothesis requires an image capture")
    with ZipFile(Path(asset.path)) as archive:
        source = SourceAsset.from_dict(
            json.loads(archive.read("receipt.json"))["source"]
        )
    if asset.metadata.get("source_sha256") != source.content_sha256:
        raise ValueError("Capture source identity differs from stored metadata")
    frames: list[FrameIdentity] = []
    pngs: list[str] = []
    with CaptureReview(library, capture_id) as review:
        previous = None
        for index in range(review.frame_count):
            frame = FrameIdentity.from_dict(review.frame(index)["frame"])
            if frames and any(
                getattr(frame, key) != getattr(frames[0], key)
                for key in ("asset_id", "shot_id", "swing_id", "camera_id")
            ):
                raise ValueError(
                    "Hypothesis capture crosses source/shot/swing/camera identities"
                )
            if (
                frame.asset_id != source.asset_id
                or frame.timing_mode != "container_pts"
                or not frame.is_timing_exact
                or frame.physical_time_s is not None
                or frame.pixel_format != "bgr24"
            ):
                raise ValueError(
                    "Hypothesis source requires exact unqualified BGR/container PTS identities"
                )
            if previous is not None and frame.presentation_time <= previous:
                raise ValueError("Hypothesis source PTS must strictly increase")
            previous = frame.presentation_time
            encoded = review.image(index)
            image = cv2.imdecode(
                np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None or image.shape != (source.height_px, source.width_px, 3):
                raise ValueError("Original source PNG dimensions differ")
            if (
                compute_frame_hash(image.tobytes(), decoder_name=frame.decoder_name)
                != frame.frame_sha256
            ):
                raise ValueError(
                    "Decoded source frame hash differs from original identity"
                )
            frames.append(frame)
            pngs.append("sha256:" + hashlib.sha256(encoded).hexdigest())
        clock = capture_clock_sha256(review)
    library.load_asset(capture_id)
    return CaptureIdentity(
        capture_id, asset.metadata["hash"], source, clock, tuple(frames), tuple(pngs)
    )
