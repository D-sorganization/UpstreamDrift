"""Session JSON schema for the starting-pose matcher.

This module owns four blocks of the session JSON document written by
:mod:`src.tools.starting_pose_matcher.gui`:

* The ``playback`` block (issue #4482) — animated full-trajectory marker
  preview state: ``current_frame``, ``speed``, ``loop``, ``trail_frames``.
* The ``data_sources`` block (issue #4480) — captures which target sources
  (club, club+ball, body) the user toggled on, the file each was loaded
  from, and the shared ``AlignOptions`` used to resample.
* The ``body_skeleton`` block (issue #4767) — captures which renderer the
  body skeleton uses ("lines" or "library_shapes") so the choice survives
  a save/load round-trip.
* The ``plot_styles`` block (issue #4808) — captures the user-customised
  :class:`MarkerStyle` for body markers and (optionally) club markers
  used by the live-view controller.

The schema is at version 6. Older sessions still load: loaders treat a
missing ``playback``, ``data_sources``, ``body_skeleton``, or
``plot_styles`` block as an empty default, and partial blocks fall back
to the per-key defaults defined here.

Public API:
    SESSION_SCHEMA_VERSION
    PlaybackState           -- frozen dataclass for the playback block
    ALLOWED_SPEEDS, DEFAULT_TRAIL_FRAMES
    DataSourcesBlock        -- frozen dataclass for the data-sources block
    default_data_sources    -- empty default for legacy sessions
    serialize_data_sources  -- dataclass -> dict
    parse_data_sources      -- dict -> dataclass (forward-compatible)
    BodySkeletonBlock       -- frozen dataclass for the body-skeleton block
    BodySkeletonStyleLiteral
    BODY_SKELETON_STYLES
    DEFAULT_BODY_SKELETON_STYLE
    default_body_skeleton   -- empty default for pre-v5 sessions
    serialize_body_skeleton -- dataclass -> dict
    parse_body_skeleton     -- dict -> dataclass (forward-compatible)
    PlotStylesBlock         -- frozen dataclass for the plot-styles block
    default_plot_styles     -- empty default for pre-v6 sessions
    serialize_plot_styles   -- dataclass -> dict
    parse_plot_styles       -- dict -> dataclass (forward-compatible)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# v6 adds the ``plot_styles`` block. Pre-v6 sessions remain loadable
# because the loader treats a missing block as the matcher defaults.
SESSION_SCHEMA_VERSION: int = 6

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

# Body skeleton renderer style (issue #4767).
#   "lines"          — the legacy ``BodySkeletonLayer`` line-segments view.
#   "library_shapes" — body_part_viz ``ShapeLibrary`` meshes (head, torso,
#                      upper_arm, ...) bound by Plug-in-Gait marker pairs.
BodySkeletonStyleLiteral = Literal["lines", "library_shapes"]
BODY_SKELETON_STYLES: tuple[BodySkeletonStyleLiteral, ...] = (
    "lines",
    "library_shapes",
)
DEFAULT_BODY_SKELETON_STYLE: BodySkeletonStyleLiteral = "lines"


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


@dataclass(frozen=True)
class BodySkeletonBlock:
    """The ``body_skeleton`` section of session JSON (v5+).

    Captures which renderer style the body skeleton uses. New in v5;
    older sessions parse with the default style ("lines").
    """

    style: BodySkeletonStyleLiteral = DEFAULT_BODY_SKELETON_STYLE


def default_body_skeleton() -> BodySkeletonBlock:
    """Return the empty default used when loading a pre-v5 session."""
    return BodySkeletonBlock()


def serialize_body_skeleton(block: BodySkeletonBlock) -> dict[str, Any]:
    """Convert a :class:`BodySkeletonBlock` to a JSON-serialisable dict."""
    return asdict(block)


def _coerce_body_skeleton_style(value: Any) -> BodySkeletonStyleLiteral:
    s = str(value) if value is not None else DEFAULT_BODY_SKELETON_STYLE
    if s in BODY_SKELETON_STYLES:
        return s  # type: ignore[return-value]
    return DEFAULT_BODY_SKELETON_STYLE


def parse_body_skeleton(d: dict[str, Any] | None) -> BodySkeletonBlock:
    """Forward-compatible decode of the ``body_skeleton`` block.

    Missing keys take their default value; unknown keys are ignored. A
    ``None`` or empty-dict input returns :func:`default_body_skeleton`.
    """
    if not d:
        return default_body_skeleton()
    return BodySkeletonBlock(style=_coerce_body_skeleton_style(d.get("style")))


# ---------------------------------------------------------------------------
# Plot styles block (v6) — issue #4808
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlotStylesBlock:
    """The ``plot_styles`` section of session JSON (v6+).

    Holds two raw JSON-ready :class:`MarkerStyle` payloads — one for the
    live view's body markers and one for the club markers. The session
    schema deliberately stores raw dicts (not :class:`MarkerStyle`
    instances) so the schema module remains free of any matplotlib /
    plot_style imports. Conversion happens in the live-view controller
    via :func:`materialise_plot_styles`.

    ``None`` for either field means "use the controller default" (the
    controller falls back to a built-in preset entry); this keeps a
    pre-v6 round-trip intact.
    """

    body: dict[str, Any] | None = None
    club: dict[str, Any] | None = None


def default_plot_styles() -> PlotStylesBlock:
    """Return the empty default used when loading a pre-v6 session."""
    return PlotStylesBlock()


def serialize_plot_styles(block: PlotStylesBlock) -> dict[str, Any]:
    """Convert a :class:`PlotStylesBlock` to a JSON-serialisable dict.

    ``None`` entries are written as JSON ``null`` so that the load path
    can faithfully detect "use controller default".
    """
    return {"body": block.body, "club": block.club}


def parse_plot_styles(d: dict[str, Any] | None) -> PlotStylesBlock:
    """Forward-compatible decode of the ``plot_styles`` block.

    Missing keys take ``None`` (i.e. controller default). Unknown keys
    are ignored. A ``None`` or empty-dict input returns
    :func:`default_plot_styles`.
    """
    if not d:
        return default_plot_styles()
    body_raw = d.get("body")
    club_raw = d.get("club")
    body = dict(body_raw) if isinstance(body_raw, Mapping) else None
    club = dict(club_raw) if isinstance(club_raw, Mapping) else None
    return PlotStylesBlock(body=body, club=club)


__all__ = [
    "ALLOWED_SPEEDS",
    "BODY_SKELETON_STYLES",
    "DEFAULT_BODY_MARKER_SET",
    "DEFAULT_BODY_MARKER_SETS",
    "DEFAULT_BODY_SKELETON_STYLE",
    "DEFAULT_TIME_ALIGNMENT",
    "DEFAULT_TRAIL_FRAMES",
    "SESSION_SCHEMA_VERSION",
    "AlignOptionsBlock",
    "BodySkeletonBlock",
    "BodySkeletonStyleLiteral",
    "BodySourceBlock",
    "ClubSourceBlock",
    "DataSourcesBlock",
    "PlaybackState",
    "PlotStylesBlock",
    "default_body_skeleton",
    "default_data_sources",
    "default_plot_styles",
    "parse_body_skeleton",
    "parse_data_sources",
    "parse_plot_styles",
    "serialize_body_skeleton",
    "serialize_data_sources",
    "serialize_plot_styles",
]
