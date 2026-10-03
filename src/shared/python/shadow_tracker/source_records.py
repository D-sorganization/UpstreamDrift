"""Source asset and frame identity records for Shadow Tracker (Packet A)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import Any, Literal, cast

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


class VariableFrameRateError(ValueError):
    """Raised when uniform-PTS consumers encounter variable frame rate (VFR) streams."""


NonUniformPTSError = VariableFrameRateError


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoTimingEvidence:
    """Validated timing facts extracted from container and video streams."""

    container_fps: float
    r_frame_rate: str
    avg_frame_rate: str
    is_vfr: bool
    physical_clock: Literal["known", "ratio_known", "unknown"]
    slow_motion_tags: dict[str, Any] = field(default_factory=dict)
    rotation_degrees: int = 0
    rotation_applied: bool = False
    creation_time: str | None = None
    capture_fps: float | None = None
    playback_fps: float | None = None
    capture_to_playback_ratio: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.container_fps) and self.container_fps > 0):
            raise ValueError(
                f"container_fps must be positive and finite, got {self.container_fps}"
            )
        if self.physical_clock not in ("known", "ratio_known", "unknown"):
            raise ValueError(
                f"physical_clock must be 'known', 'ratio_known', or 'unknown', got {self.physical_clock!r}"
            )
        if not self.slow_motion_tags and self.physical_clock in (
            "known",
            "ratio_known",
        ):
            raise ValueError(
                "physical_clock cannot be known or ratio_known without slow-motion tags"
            )
        if self.rotation_degrees not in (0, 90, 180, 270):
            raise ValueError(
                f"rotation_degrees must be in {{0, 90, 180, 270}}, got {self.rotation_degrees}"
            )
        if not isinstance(self.is_vfr, bool):
            raise TypeError(f"is_vfr must be bool, got {type(self.is_vfr).__name__}")
        if self.creation_time is not None and not isinstance(self.creation_time, str):
            raise TypeError(
                f"creation_time must be str or None, got {type(self.creation_time).__name__}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize timing evidence to dictionary."""
        return {
            "container_fps": self.container_fps,
            "r_frame_rate": self.r_frame_rate,
            "avg_frame_rate": self.avg_frame_rate,
            "is_vfr": self.is_vfr,
            "physical_clock": self.physical_clock,
            "slow_motion_tags": dict(self.slow_motion_tags),
            "rotation_degrees": self.rotation_degrees,
            "rotation_applied": self.rotation_applied,
            "creation_time": self.creation_time,
            "capture_fps": self.capture_fps,
            "playback_fps": self.playback_fps,
            "capture_to_playback_ratio": self.capture_to_playback_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoTimingEvidence:
        """Construct from dictionary."""
        return cls(
            container_fps=float(data["container_fps"]),
            r_frame_rate=str(data["r_frame_rate"]),
            avg_frame_rate=str(data["avg_frame_rate"]),
            is_vfr=bool(data["is_vfr"]),
            physical_clock=cast(
                Literal["known", "ratio_known", "unknown"],
                data["physical_clock"],
            ),
            slow_motion_tags=dict(data.get("slow_motion_tags", {})),
            rotation_degrees=int(data.get("rotation_degrees", 0)),
            rotation_applied=bool(data.get("rotation_applied", False)),
            creation_time=data.get("creation_time"),
            capture_fps=(
                float(data["capture_fps"])
                if data.get("capture_fps") is not None
                else None
            ),
            playback_fps=(
                float(data["playback_fps"])
                if data.get("playback_fps") is not None
                else None
            ),
            capture_to_playback_ratio=(
                float(data["capture_to_playback_ratio"])
                if data.get("capture_to_playback_ratio") is not None
                else None
            ),
        )


def _parse_fps_str(val: str) -> float:
    """Parse fraction or float string into finite float."""
    if "/" in val:
        num, den = val.split("/", 1)
        d = float(den)
        return float(num) / d if d != 0.0 else 0.0
    return float(val)


def extract_video_timing_evidence(
    probe_data: dict[str, Any],
) -> VideoTimingEvidence:
    """Extract and validate timing, clock, and orientation evidence from ffprobe output."""
    streams = probe_data.get("streams", [])
    video_stream: dict[str, Any] | None = None
    for s in streams:
        if isinstance(s, dict) and s.get("codec_type") == "video":
            video_stream = s
            break
    if video_stream is None:
        raise ValueError("No video stream found in probe data")

    r_fps = str(video_stream.get("r_frame_rate", "30/1"))
    avg_fps = str(video_stream.get("avg_frame_rate", r_fps))
    container_fps = (
        _parse_fps_str(avg_fps) if avg_fps != "0/0" else _parse_fps_str(r_fps)
    )
    if container_fps <= 0.0:
        container_fps = 30.0
    is_vfr = r_fps != avg_fps

    stream_tags = video_stream.get("tags", {})
    format_tags = probe_data.get("format", {}).get("tags", {})

    creation_time = stream_tags.get("creation_time") or format_tags.get("creation_time")

    rotation = 0
    if "rotate" in stream_tags:
        try:
            rotation = int(float(stream_tags["rotate"])) % 360
        except (ValueError, TypeError):
            rotation = 0
    elif "side_data_list" in video_stream:
        for sd in video_stream.get("side_data_list", []):
            if (
                isinstance(sd, dict)
                and sd.get("side_data_type", "").lower() == "displaymatrix"
            ):
                try:
                    rotation = int(float(sd.get("rotation", 0))) % 360
                except (ValueError, TypeError):
                    rotation = 0
                break

    sm_tags: dict[str, Any] = {}
    for source in (stream_tags, format_tags):
        for k, v in source.items():
            k_lower = k.lower()
            if (
                k_lower.startswith(("com.apple.quicktime.", "com.android."))
                or "slow_motion" in k_lower
                or "capture_fps" in k_lower
                or "capture.fps" in k_lower
            ):
                sm_tags[k] = v

    capture_fps: float | None = None
    capture_ratio: float | None = None
    for k, v in sm_tags.items():
        k_lower = k.lower()
        if "fps" in k_lower:
            try:
                capture_fps = float(v)
            except (ValueError, TypeError):
                pass
        elif "rate" in k_lower or "ratio" in k_lower:
            try:
                capture_ratio = float(v)
            except (ValueError, TypeError):
                pass

    physical_clock: Literal["known", "ratio_known", "unknown"]
    if sm_tags:
        if capture_fps is not None and container_fps > 0:
            physical_clock = "known"
            capture_ratio = capture_fps / container_fps
        elif capture_ratio is not None:
            physical_clock = "ratio_known"
        else:
            physical_clock = "ratio_known"
    else:
        physical_clock = "unknown"
        capture_fps = None
        capture_ratio = None

    return VideoTimingEvidence(
        container_fps=container_fps,
        r_frame_rate=r_fps,
        avg_frame_rate=avg_fps,
        is_vfr=is_vfr,
        physical_clock=physical_clock,
        slow_motion_tags=sm_tags,
        rotation_degrees=rotation,
        rotation_applied=False,
        creation_time=str(creation_time) if creation_time is not None else None,
        capture_fps=capture_fps,
        playback_fps=container_fps if sm_tags else None,
        capture_to_playback_ratio=capture_ratio,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class SwingWindow:
    """Half-open interval and view facts for a continuous swing candidate."""

    swing_id: str
    clip_id: str
    start_pts_s: float
    end_pts_s: float
    view: Literal["face_on", "down_the_line", "oblique", "other"]
    camera_azimuth_deg: float | None = None
    address_frame: int | None = None
    top_frame: int | None = None
    impact_frame: int | None = None
    finish_frame: int | None = None
    uncertainty_frames: dict[str, int] = field(default_factory=dict)
    is_practice: bool = False
    is_partial: bool = False
    has_cut: bool = False
    visibility: dict[str, bool] = field(default_factory=dict)
    blur_at_impact: str = "low"
    camera_motion: str = "static"

    def __post_init__(self) -> None:
        check_id(self.swing_id, "swing_id")
        check_id(self.clip_id, "clip_id")
        if not (math.isfinite(self.start_pts_s) and math.isfinite(self.end_pts_s)):
            raise ValueError("start_pts_s and end_pts_s must be finite")
        if not 0 <= self.start_pts_s < self.end_pts_s:
            raise ValueError(
                f"Swing window requires 0 <= start < end, got [{self.start_pts_s}, {self.end_pts_s})"
            )
        if self.view not in ("face_on", "down_the_line", "oblique", "other"):
            raise ValueError(
                f"view must be one of 'face_on', 'down_the_line', 'oblique', 'other', got {self.view!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize swing window to dictionary."""
        return {
            "swing_id": self.swing_id,
            "clip_id": self.clip_id,
            "start_pts_s": self.start_pts_s,
            "end_pts_s": self.end_pts_s,
            "view": self.view,
            "camera_azimuth_deg": self.camera_azimuth_deg,
            "address_frame": self.address_frame,
            "top_frame": self.top_frame,
            "impact_frame": self.impact_frame,
            "finish_frame": self.finish_frame,
            "uncertainty_frames": dict(self.uncertainty_frames),
            "is_practice": self.is_practice,
            "is_partial": self.is_partial,
            "has_cut": self.has_cut,
            "visibility": dict(self.visibility),
            "blur_at_impact": self.blur_at_impact,
            "camera_motion": self.camera_motion,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SwingWindow:
        """Construct swing window from dictionary."""
        return cls(
            swing_id=str(data["swing_id"]),
            clip_id=str(data["clip_id"]),
            start_pts_s=float(data["start_pts_s"]),
            end_pts_s=float(data["end_pts_s"]),
            view=cast(
                Literal["face_on", "down_the_line", "oblique", "other"],
                data["view"],
            ),
            camera_azimuth_deg=(
                float(data["camera_azimuth_deg"])
                if data.get("camera_azimuth_deg") is not None
                else None
            ),
            address_frame=data.get("address_frame"),
            top_frame=data.get("top_frame"),
            impact_frame=data.get("impact_frame"),
            finish_frame=data.get("finish_frame"),
            uncertainty_frames=dict(data.get("uncertainty_frames", {})),
            is_practice=bool(data.get("is_practice", False)),
            is_partial=bool(data.get("is_partial", False)),
            has_cut=bool(data.get("has_cut", False)),
            visibility=dict(data.get("visibility", {})),
            blur_at_impact=str(data.get("blur_at_impact", "low")),
            camera_motion=str(data.get("camera_motion", "static")),
        )


def validate_swing_windows(windows: Sequence[SwingWindow]) -> None:
    """Validate swing windows for sequence constraints and non-overlapping intervals."""
    by_clip: dict[str, list[SwingWindow]] = {}
    for idx, w in enumerate(windows):
        if not isinstance(w, SwingWindow):
            raise TypeError(
                f"Element at index {idx} must be a SwingWindow, got {type(w).__name__}"
            )
        by_clip.setdefault(w.clip_id, []).append(w)

    for clip_id, clip_windows in by_clip.items():
        sorted_windows = sorted(clip_windows, key=lambda sw: sw.start_pts_s)
        for i in range(len(sorted_windows) - 1):
            w1 = sorted_windows[i]
            w2 = sorted_windows[i + 1]
            if w1.end_pts_s > w2.start_pts_s:
                raise ValueError(
                    f"Overlapping swing windows in clip {clip_id}: "
                    f"{w1.swing_id} [{w1.start_pts_s}, {w1.end_pts_s}) and "
                    f"{w2.swing_id} [{w2.start_pts_s}, {w2.end_pts_s})"
                )


@dataclass(frozen=True, slots=True)
class SwingGradeResult:
    """Usability grade and mandatory reasons."""

    grade: Literal["A", "B", "C", "R"]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.grade not in ("A", "B", "C", "R"):
            raise ValueError(
                f"grade must be one of 'A', 'B', 'C', 'R', got {self.grade!r}"
            )
        if not self.reasons:
            raise ValueError("A grade must have at least one reason")
        for r in self.reasons:
            if not isinstance(r, str) or not r.strip():
                raise ValueError("Grade reasons must be non-empty strings")

    def __iter__(self):
        """Allow tuple unpacking (grade, reasons = result)."""
        yield self.grade
        yield self.reasons

    def to_dict(self) -> dict[str, Any]:
        """Serialize grade result to dictionary."""
        return {"grade": self.grade, "reasons": list(self.reasons)}


def grade_swing_window(
    window: SwingWindow,
    *,
    timing: VideoTimingEvidence | None = None,
    fps: float | None = None,
    reasons: Sequence[str] | None = None,
) -> SwingGradeResult:
    """Grade a swing window according to COV-2 usability rubric (pure function)."""
    if reasons is not None:
        return SwingGradeResult(
            grade="R" if window.is_partial or window.has_cut else "A",
            reasons=tuple(reasons),
        )

    if window.is_partial or window.has_cut or window.view == "other":
        return SwingGradeResult(
            grade="R",
            reasons=("Unusable clip: partial swing, cut, or unrecognized camera view",),
        )

    effective_fps = timing.container_fps if timing is not None else (fps or 30.0)

    if window.camera_motion != "static" or window.visibility.get("body") is False:
        return SwingGradeResult(
            grade="C",
            reasons=(
                "Camera moving or body partly cropped, but swing is identifiable",
            ),
        )

    if (
        effective_fps < 59.0
        or window.blur_at_impact != "low"
        or window.visibility.get("clubhead") is False
    ):
        return SwingGradeResult(
            grade="B",
            reasons=(
                f"Full body visible with {effective_fps:.1f} fps playback or impact club blur",
            ),
        )

    return SwingGradeResult(
        grade="A",
        reasons=(
            "Full body and club visible address through finish, >= 60 fps, static camera",
        ),
    )
