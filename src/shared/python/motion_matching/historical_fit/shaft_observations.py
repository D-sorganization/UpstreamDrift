"""Immutable reviewed image bearings; visible fragments are not club endpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Literal

from src.shared.python.shadow_tracker.source_records import FrameIdentity

ShaftStatus = Literal["observed", "occluded", "offscreen", "ambiguous", "unreviewed"]


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} requires nonempty trimmed text")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError(f"{name} requires a lowercase prefixed SHA256")
    return value


def _number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} requires a finite number")
    return float(value)


@dataclass(frozen=True)
class ShaftAxisSegment:
    """Two visible image-axis points, in original pixels; abstention is explicit.

    ``sigma_px`` is declared perpendicular localization uncertainty, not a
    physical accuracy claim. Confidence and visibility retain separate meanings.
    """

    status: ShaftStatus
    points_px: tuple[tuple[float, float], tuple[float, float]] | None
    reviewer: str
    reason: str
    confidence: float | None
    visibility: float | None
    sigma_px: float | None
    weighting_status: Literal["authored_uncalibrated"] = "authored_uncalibrated"

    def __post_init__(self) -> None:
        _text(self.reviewer, "reviewer")
        _text(self.reason, "reason")
        if self.weighting_status != "authored_uncalibrated":
            raise ValueError(
                "Shaft confidence and uncertainty must remain authored uncalibrated"
            )
        if self.status not in (
            "observed",
            "occluded",
            "offscreen",
            "ambiguous",
            "unreviewed",
        ):
            raise ValueError("Unknown shaft observation status")
        if self.status != "observed":
            if any(
                value is not None
                for value in (
                    self.points_px,
                    self.confidence,
                    self.visibility,
                    self.sigma_px,
                )
            ):
                raise ValueError(
                    "Abstention must not contain fabricated point evidence"
                )
            return
        if (
            self.points_px is None
            or len(self.points_px) != 2
            or any(len(point) != 2 for point in self.points_px)
        ):
            raise ValueError("Observed shaft requires two image XY points")
        points = tuple(
            tuple(_number(value, "pixel") for value in point)
            for point in self.points_px
        )
        if points[0] == points[1]:
            raise ValueError("Observed shaft segment must be nondegenerate")
        object.__setattr__(self, "points_px", points)
        for name in ("confidence", "visibility"):
            value = getattr(self, name)
            if name == "visibility" and value is None:
                continue
            numeric = _number(value, name)
            if not 0 <= numeric <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, numeric)
        sigma = _number(self.sigma_px, "sigma_px")
        if sigma <= 0:
            raise ValueError("sigma_px must be positive")
        object.__setattr__(self, "sigma_px", sigma)


@dataclass(frozen=True)
class SourceBoundShaftFrame:
    """Distinct decoded-pixel and PNG-byte hashes with an exact source clock."""

    frame_index: int
    frame: FrameIdentity
    png_sha256: str
    segment: ShaftAxisSegment

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a nonnegative integer")
        if not isinstance(self.frame, FrameIdentity) or not isinstance(
            self.segment, ShaftAxisSegment
        ):
            raise ValueError("Typed source frame and shaft segment are required")
        if (
            self.frame.physical_time_s is not None
            or not self.frame.is_timing_exact
            or self.frame.timing_mode != "container_pts"
        ):
            raise ValueError(
                "Historical shaft evidence requires exact PTS and unknown physical time"
            )
        _digest(self.png_sha256, "png_sha256")


@dataclass(frozen=True)
class ShaftAxisEvidence:
    """Sparse reviewed observations; all source and image bindings are explicit."""

    capture_id: str
    capture_sha256: str
    source_sha256: str
    image_size: tuple[int, int]
    frames: tuple[SourceBoundShaftFrame, ...]

    def __post_init__(self) -> None:
        _text(self.capture_id, "capture_id")
        _digest(self.capture_sha256, "capture_sha256")
        _digest(self.source_sha256, "source_sha256")
        if len(self.image_size) != 2 or any(
            isinstance(x, bool) or not isinstance(x, int) or x <= 0
            for x in self.image_size
        ):
            raise ValueError("image_size requires positive integer width and height")
        object.__setattr__(self, "image_size", tuple(self.image_size))
        frames = tuple(self.frames)
        if not frames or any(
            not isinstance(frame, SourceBoundShaftFrame) for frame in frames
        ):
            raise ValueError("At least one typed reviewed shaft frame is required")
        if len({frame.frame.frame_id for frame in frames}) != len(frames):
            raise ValueError("Reviewed source frame identity must be unique")
        indices = [frame.frame_index for frame in frames]
        times = [frame.frame.presentation_time for frame in frames]
        if any(b <= a for a, b in zip(indices, indices[1:], strict=False)) or any(
            b <= a for a, b in zip(times, times[1:], strict=False)
        ):
            raise ValueError("Reviewed indices and source PTS must increase strictly")
        for frame in frames:
            if frame.frame.asset_id != "source-" + self.source_sha256.removeprefix(
                "sha256:"
            ):
                raise ValueError("Frame differs from declared source hash")
            if frame.segment.points_px is not None and any(
                not 0 <= x < self.image_size[0] or not 0 <= y < self.image_size[1]
                for x, y in frame.segment.points_px
            ):
                raise ValueError("Observed shaft points must lie in the original image")
        anchor = frames[0].frame
        if any(
            (item.frame.shot_id, item.frame.swing_id, item.frame.camera_id)
            != (anchor.shot_id, anchor.swing_id, anchor.camera_id)
            for item in frames
        ):
            raise ValueError("Shaft evidence crosses source shot, swing or camera")
        object.__setattr__(self, "frames", frames)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["schema"] = "necromatcher/shaft-axis-evidence/1"
        return json.loads(json.dumps(record, allow_nan=False))

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ShaftAxisEvidence:
        if (
            set(record)
            != {
                "schema",
                "capture_id",
                "capture_sha256",
                "source_sha256",
                "image_size",
                "frames",
            }
            or record["schema"] != "necromatcher/shaft-axis-evidence/1"
        ):
            raise ValueError("Malformed shaft evidence record")
        try:
            frames = tuple(
                SourceBoundShaftFrame(
                    item["frame_index"],
                    FrameIdentity.from_dict(item["frame"]),
                    item["png_sha256"],
                    ShaftAxisSegment(**item["segment"]),
                )
                for item in record["frames"]
            )
            if any(
                set(item) != {"frame_index", "frame", "png_sha256", "segment"}
                for item in record["frames"]
            ):
                raise ValueError("Unknown shaft frame fields")
            return cls(
                record["capture_id"],
                record["capture_sha256"],
                record["source_sha256"],
                tuple(record["image_size"]),
                frames,
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed shaft evidence record") from exc

    @property
    def sha256(self) -> str:
        raw = json.dumps(
            self.to_record(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(raw).hexdigest()
