"""Source asset and frame identity records for Shadow Tracker (Packet A)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Literal

from ._validation import (
    FRAME_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
    check_bool,
    check_id,
    check_int,
    check_payload_keys,
    check_pos_int,
    check_schema_version,
    check_sha256,
    check_str,
    check_uri,
)

RightsStatus = Literal["unknown", "restricted", "permitted"]
_VALID_RIGHTS_STATUSES = frozenset(("unknown", "restricted", "permitted"))
_VALID_TIMING_MODES = frozenset(("authoritative", "container_pts", "estimated_cfr"))
_FRAME_SCHEMA_VERSIONS = frozenset(
    (
        FRAME_SCHEMA_VERSION,
        "shadow-tracker/frame/1.1.0",
    )
)
_SOURCE_ASSET_KEYS = frozenset(
    (
        "schema_version",
        "asset_id",
        "source_uri",
        "content_sha256",
        "width_px",
        "height_px",
        "rights_status",
        "rights_note",
    )
)
_FRAME_IDENTITY_KEYS = frozenset(
    (
        "schema_version",
        "asset_id",
        "shot_id",
        "swing_id",
        "camera_id",
        "frame_id",
        "pts_ticks",
        "timebase_numerator",
        "timebase_denominator",
        "physical_time_s",
        "physical_time_reason",
        "frame_sha256",
        "timing_mode",
        "is_timing_exact",
        "clock_evidence",
        "decoder_name",
        "decoder_version",
        "pixel_format",
    )
)
_LEGACY_FRAME_IDENTITY_KEYS = frozenset(
    (
        "schema_version",
        "asset_id",
        "shot_id",
        "swing_id",
        "camera_id",
        "frame_id",
        "pts_ticks",
        "timebase_numerator",
        "timebase_denominator",
        "physical_time_s",
        "physical_time_reason",
        "frame_sha256",
    )
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceAsset:
    """Source asset metadata record for an ingestion sequence authority."""

    schema_version: str
    asset_id: str
    source_uri: str
    content_sha256: str
    width_px: int
    height_px: int
    rights_status: RightsStatus
    rights_note: str

    def __post_init__(self) -> None:
        check_schema_version(self.schema_version, SOURCE_SCHEMA_VERSION)
        check_id(self.asset_id, "asset_id")
        check_uri(self.source_uri, "source_uri")
        check_sha256(self.content_sha256, "content_sha256")
        check_pos_int(self.width_px, "width_px")
        check_pos_int(self.height_px, "height_px")
        if not isinstance(self.rights_status, str):
            raise TypeError(
                f"rights_status must be a str, got {type(self.rights_status).__name__}"
            )
        if self.rights_status not in _VALID_RIGHTS_STATUSES:
            raise ValueError(
                f"rights_status must be one of {sorted(_VALID_RIGHTS_STATUSES)}, got {self.rights_status!r}"
            )
        if not isinstance(self.rights_note, str):
            raise TypeError(
                f"rights_note must be a str, got {type(self.rights_note).__name__}"
            )
        if self.rights_status != "unknown":
            if not self.rights_note or self.rights_note.strip() != self.rights_note:
                raise ValueError(
                    f"rights_note must be non-empty and trimmed when rights_status is {self.rights_status!r}, got {self.rights_note!r}"
                )
        else:
            if self.rights_note.strip() != self.rights_note:
                raise ValueError(
                    f"rights_note must be trimmed, got {self.rights_note!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to JSON-compatible dictionary."""
        return {
            "schema_version": self.schema_version,
            "asset_id": self.asset_id,
            "source_uri": self.source_uri,
            "content_sha256": self.content_sha256,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "rights_status": self.rights_status,
            "rights_note": self.rights_note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SourceAsset:
        """Deserialize from dictionary, rejecting unknown fields."""
        check_payload_keys(payload, _SOURCE_ASSET_KEYS)
        return cls(**payload)


@dataclass(frozen=True, slots=True, kw_only=True)
class FrameIdentity:
    """Frame identity record linking PTS ticks, timebase and physical time."""

    schema_version: str
    asset_id: str
    shot_id: str
    swing_id: str
    camera_id: str
    frame_id: str
    pts_ticks: int
    timebase_numerator: int
    timebase_denominator: int
    physical_time_s: float | None
    physical_time_reason: str
    frame_sha256: str
    timing_mode: str = "estimated_cfr"
    is_timing_exact: bool = False
    clock_evidence: str = "unverified_legacy_record"
    decoder_name: str = "opencv"
    decoder_version: str = "legacy"
    pixel_format: str = "bgr24"

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str):
            raise TypeError(
                f"schema_version must be a str, got {type(self.schema_version).__name__}"
            )
        if self.schema_version not in _FRAME_SCHEMA_VERSIONS:
            raise ValueError(
                f"schema_version must be one of {sorted(_FRAME_SCHEMA_VERSIONS)}, got {self.schema_version!r}"
            )
        check_id(self.asset_id, "asset_id")
        check_id(self.shot_id, "shot_id")
        check_id(self.swing_id, "swing_id")
        check_id(self.camera_id, "camera_id")
        check_id(self.frame_id, "frame_id")
        check_int(self.pts_ticks, "pts_ticks")
        num = check_pos_int(self.timebase_numerator, "timebase_numerator")
        den = check_pos_int(self.timebase_denominator, "timebase_denominator")
        if math.gcd(num, den) != 1:
            raise ValueError(
                f"timebase fraction {num}/{den} must be reduced (gcd={math.gcd(num, den)})"
            )
        if self.physical_time_s is not None:
            if isinstance(self.physical_time_s, bool) or not isinstance(
                self.physical_time_s, float
            ):
                raise TypeError(
                    f"physical_time_s must be float or None, got {type(self.physical_time_s).__name__}"
                )
            if not math.isfinite(self.physical_time_s):
                raise ValueError(
                    f"physical_time_s must be finite, got {self.physical_time_s}"
                )
        if not isinstance(self.physical_time_reason, str):
            raise TypeError(
                f"physical_time_reason must be a str, got {type(self.physical_time_reason).__name__}"
            )
        if self.physical_time_s is None:
            if (
                not self.physical_time_reason
                or self.physical_time_reason.strip() != self.physical_time_reason
            ):
                raise ValueError(
                    "physical_time_reason must be non-empty and trimmed when physical_time_s is unknown"
                )
        else:
            if self.physical_time_reason.strip() != self.physical_time_reason:
                raise ValueError(
                    f"physical_time_reason must be trimmed, got {self.physical_time_reason!r}"
                )
        check_sha256(self.frame_sha256, "frame_sha256")
        check_str(self.timing_mode, "timing_mode")
        if self.timing_mode not in _VALID_TIMING_MODES:
            raise ValueError(
                f"timing_mode must be one of {sorted(_VALID_TIMING_MODES)}, got {self.timing_mode!r}"
            )
        check_bool(self.is_timing_exact, "is_timing_exact")
        check_str(self.clock_evidence, "clock_evidence")
        check_str(self.decoder_name, "decoder_name")
        check_str(self.decoder_version, "decoder_version")
        check_str(self.pixel_format, "pixel_format")

    @property
    def presentation_time(self) -> Fraction:
        """Exact rational presentation time in seconds."""
        return Fraction(
            self.pts_ticks * self.timebase_numerator,
            self.timebase_denominator,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to JSON-compatible dictionary."""
        return {
            "schema_version": self.schema_version,
            "asset_id": self.asset_id,
            "shot_id": self.shot_id,
            "swing_id": self.swing_id,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "pts_ticks": self.pts_ticks,
            "timebase_numerator": self.timebase_numerator,
            "timebase_denominator": self.timebase_denominator,
            "physical_time_s": self.physical_time_s,
            "physical_time_reason": self.physical_time_reason,
            "frame_sha256": self.frame_sha256,
            "timing_mode": self.timing_mode,
            "is_timing_exact": self.is_timing_exact,
            "clock_evidence": self.clock_evidence,
            "decoder_name": self.decoder_name,
            "decoder_version": self.decoder_version,
            "pixel_format": self.pixel_format,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FrameIdentity:
        """Deserialize from dictionary, rejecting unknown fields."""
        if not isinstance(payload, dict):
            raise TypeError(f"Payload must be a dict, got {type(payload).__name__}")
        extra = set(payload.keys()) - _FRAME_IDENTITY_KEYS
        if extra:
            raise ValueError(f"Unknown fields rejected: {sorted(extra)}")
        missing = _LEGACY_FRAME_IDENTITY_KEYS - set(payload.keys())
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")
        return cls(
            schema_version=payload["schema_version"],
            asset_id=payload["asset_id"],
            shot_id=payload["shot_id"],
            swing_id=payload["swing_id"],
            camera_id=payload["camera_id"],
            frame_id=payload["frame_id"],
            pts_ticks=payload["pts_ticks"],
            timebase_numerator=payload["timebase_numerator"],
            timebase_denominator=payload["timebase_denominator"],
            physical_time_s=payload["physical_time_s"],
            physical_time_reason=payload["physical_time_reason"],
            frame_sha256=payload["frame_sha256"],
            timing_mode=payload.get("timing_mode", "estimated_cfr"),
            is_timing_exact=payload.get("is_timing_exact", False),
            clock_evidence=payload.get("clock_evidence", "unverified_legacy_record"),
            decoder_name=payload.get("decoder_name", "opencv"),
            decoder_version=payload.get("decoder_version", "legacy"),
            pixel_format=payload.get("pixel_format", "bgr24"),
        )


def validate_frame_sequence(frames: Sequence[FrameIdentity]) -> None:
    """Validate sequence constraints on a series of frame identities.

    Preconditions:
        - Sequence must be nonempty and contain only FrameIdentity instances.
        - All frames must share identical asset_id, shot_id, swing_id, and camera_id.
        - Frame IDs must be unique.
        - Presentation time must strictly increase (pts_ticks order).
        - Known physical times must strictly increase across the known subset.
    """
    if not frames:
        raise ValueError("Frame sequence cannot be empty")

    for idx, item in enumerate(frames):
        if not isinstance(item, FrameIdentity):
            raise TypeError(
                f"Element at index {idx} must be a FrameIdentity, got {type(item).__name__}"
            )

    first = frames[0]
    expected_asset = first.asset_id
    expected_shot = first.shot_id
    expected_swing = first.swing_id
    expected_camera = first.camera_id

    seen_ids: set[str] = set()
    last_pts: Fraction | None = None
    last_physical: float | None = None

    for idx, frame in enumerate(frames):
        if frame.asset_id != expected_asset:
            raise ValueError(
                f"asset_id mismatch at frame {idx}: expected {expected_asset!r}, got {frame.asset_id!r}"
            )
        if frame.shot_id != expected_shot:
            raise ValueError(
                f"shot_id mismatch at frame {idx}: expected {expected_shot!r}, got {frame.shot_id!r}"
            )
        if frame.swing_id != expected_swing:
            raise ValueError(
                f"swing_id mismatch at frame {idx}: expected {expected_swing!r}, got {frame.swing_id!r}"
            )
        if frame.camera_id != expected_camera:
            raise ValueError(
                f"camera_id mismatch at frame {idx}: expected {expected_camera!r}, got {frame.camera_id!r}"
            )

        if frame.frame_id in seen_ids:
            raise ValueError(f"Duplicate frame_id {frame.frame_id!r} at index {idx}")
        seen_ids.add(frame.frame_id)

        pts = frame.presentation_time
        if last_pts is not None and pts <= last_pts:
            raise ValueError(
                f"Presentation time must strictly increase in sequence (pts_ticks order violated at index {idx}): "
                f"previous {last_pts} >= current {pts}"
            )
        last_pts = pts

        if frame.physical_time_s is not None:
            if last_physical is not None and frame.physical_time_s <= last_physical:
                raise ValueError(
                    f"physical_time_s must strictly increase across known frames at index {idx}: "
                    f"previous {last_physical} >= current {frame.physical_time_s}"
                )
            last_physical = frame.physical_time_s
