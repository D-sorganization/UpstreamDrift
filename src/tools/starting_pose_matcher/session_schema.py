"""Session JSON schema for the starting-pose matcher.

This module owns two blocks of the session JSON document written by
:mod:`src.tools.starting_pose_matcher.gui`:

* The ``playback`` block (issue #4482) — animated full-trajectory marker
  preview state: ``current_frame``, ``speed``, ``loop``, ``trail_frames``.
* The ``data_sources`` block (issue #4480) — captures which target sources
  (club, club+ball, body) the user toggled on, the file each was loaded
  from, and the shared ``AlignOptions`` used to resample.

The schema is at version 4. Older sessions (v3 or earlier) still load:
loaders treat a missing ``playback`` or ``data_sources`` block as an empty
default, and partial blocks fall back to the per-key defaults defined here.

Public API:
    SESSION_SCHEMA_VERSION
    PlaybackState           -- frozen dataclass for the playback block
    ALLOWED_SPEEDS, DEFAULT_TRAIL_FRAMES
    DataSourcesBlock        -- frozen dataclass for the data-sources block
    default_data_sources    -- empty default for legacy sessions
    serialize_data_sources  -- dataclass -> dict
    parse_data_sources      -- dict -> dataclass (forward-compatible)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# v4 adds the ``data_sources`` block. v3 sessions are still loadable.
SESSION_SCHEMA_VERSION: int = 4

# Allowed playback-speed multipliers exposed by the speed combo box.
ALLOWED_SPEEDS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)

# Default trail length, in frames, used by the show-trail layer.
DEFAULT_TRAIL_FRAMES: int = 30

# Default body marker-set names exposed in the source-toggle combo.
DEFAULT_BODY_MARKER_SETS: tuple[str, ...] = (
    "Anatomical 28",
    "Lower body only",
    "Upper body only",
    "All markers",
)
DEFAULT_BODY_MARKER_SET: str = "Anatomical 28"

# Time-alignment radio.  Mirrors ``AlignOptions.time_alignment`` literals.
TimeAlignmentLiteral = Literal["impact", "address"]
DEFAULT_TIME_ALIGNMENT: TimeAlignmentLiteral = "impact"


@dataclass
class PlaybackState:
    """Snapshot of the playback UI state.

    Attributes mirror the keys of the ``playback`` block in the session
    JSON document.
    """

    current_frame: int = 0
    speed: float = 1.0
    loop: bool = True
    trail_frames: int = DEFAULT_TRAIL_FRAMES

    def __post_init__(self) -> None:
        if self.current_frame < 0:
            raise ValueError(f"current_frame must be >= 0, got {self.current_frame}")
        if self.speed <= 0:
            raise ValueError(f"speed must be > 0, got {self.speed}")
        if self.trail_frames < 0:
            raise ValueError(f"trail_frames must be >= 0, got {self.trail_frames}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> PlaybackState:
        """Build a :class:`PlaybackState` from a possibly-partial mapping.

        Unknown keys are ignored. Missing keys fall back to the dataclass
        defaults so v3 sessions (which had no ``playback`` block, or only
        the legacy fields) load cleanly.
        """
        if not data:
            return cls()
        kwargs: dict[str, Any] = {}
        if "current_frame" in data:
            kwargs["current_frame"] = int(data["current_frame"])
        if "speed" in data:
            kwargs["speed"] = float(data["speed"])
        if "loop" in data:
            kwargs["loop"] = bool(data["loop"])
        if "trail_frames" in data:
            kwargs["trail_frames"] = int(data["trail_frames"])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return asdict(self)

    def snap_speed_to_allowed(self) -> float:
        """Return the closest value in :data:`ALLOWED_SPEEDS` to ``self.speed``.

        Useful when an out-of-band speed value is read from disk and the UI
        needs to display the nearest combo-box choice.
        """
        return min(ALLOWED_SPEEDS, key=lambda s: abs(s - self.speed))


@dataclass(frozen=True)
class ClubSourceBlock:
    """Persisted state of the Club row in the Data-sources panel."""

    enabled: bool = False
    file_path: str | None = None
    include_ball: bool = False  # False=Club only; True=Club+ball


@dataclass(frozen=True)
class BodySourceBlock:
    """Persisted state of the Body-markers row."""

    enabled: bool = False
    file_path: str | None = None
    marker_set: str = DEFAULT_BODY_MARKER_SET


@dataclass(frozen=True)
class AlignOptionsBlock:
    """Shared alignment / resampling controls."""

    sample_rate_hz: float = 1000.0
    simulation_time_s: float = 0.3
    time_alignment: str = DEFAULT_TIME_ALIGNMENT  # "impact" | "address"


@dataclass(frozen=True)
class DataSourcesBlock:
    """The ``data_sources`` section of session JSON (v4+)."""

    club: ClubSourceBlock = field(default_factory=ClubSourceBlock)
    body: BodySourceBlock = field(default_factory=BodySourceBlock)
    align: AlignOptionsBlock = field(default_factory=AlignOptionsBlock)


def default_data_sources() -> DataSourcesBlock:
    """Return the empty default used when loading a pre-v4 session."""
    return DataSourcesBlock()


def serialize_data_sources(block: DataSourcesBlock) -> dict[str, Any]:
    """Convert a ``DataSourcesBlock`` to a JSON-serialisable dict."""
    return asdict(block)


def _coerce_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_time_alignment(value: Any) -> str:
    s = str(value) if value is not None else DEFAULT_TIME_ALIGNMENT
    return s if s in ("impact", "address") else DEFAULT_TIME_ALIGNMENT


def parse_data_sources(d: dict[str, Any] | None) -> DataSourcesBlock:
    """Forward-compatible decode.

    Missing keys take their default value; unknown keys are ignored.
    A ``None`` or empty-dict input returns ``default_data_sources()``.
    """
    if not d:
        return default_data_sources()

    club_d: dict[str, Any] = d.get("club") or {}
    body_d: dict[str, Any] = d.get("body") or {}
    align_d: dict[str, Any] = d.get("align") or {}

    club = ClubSourceBlock(
        enabled=bool(club_d.get("enabled", False)),
        file_path=_coerce_str_or_none(club_d.get("file_path")),
        include_ball=bool(club_d.get("include_ball", False)),
    )
    body = BodySourceBlock(
        enabled=bool(body_d.get("enabled", False)),
        file_path=_coerce_str_or_none(body_d.get("file_path")),
        marker_set=str(body_d.get("marker_set", DEFAULT_BODY_MARKER_SET)),
    )
    align = AlignOptionsBlock(
        sample_rate_hz=float(align_d.get("sample_rate_hz", 1000.0)),
        simulation_time_s=float(align_d.get("simulation_time_s", 0.3)),
        time_alignment=_coerce_time_alignment(align_d.get("time_alignment")),
    )
    return DataSourcesBlock(club=club, body=body, align=align)


__all__ = [
    "ALLOWED_SPEEDS",
    "DEFAULT_BODY_MARKER_SET",
    "DEFAULT_BODY_MARKER_SETS",
    "DEFAULT_TIME_ALIGNMENT",
    "DEFAULT_TRAIL_FRAMES",
    "SESSION_SCHEMA_VERSION",
    "AlignOptionsBlock",
    "BodySourceBlock",
    "ClubSourceBlock",
    "DataSourcesBlock",
    "PlaybackState",
    "default_data_sources",
    "parse_data_sources",
    "serialize_data_sources",
]
