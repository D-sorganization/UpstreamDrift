"""Canonical launch-monitor metric and session contracts.

Numeric shot metrics use one canonical unit per metric. Raw source columns are
retained separately by the importer so conversion is reversible and auditable.
Angles are radians, speeds are metres per second, distances are metres, spin is
radians per second, and time is seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pandas as pd


@dataclass(frozen=True)
class MetricDefinition:
    """Definition of one canonical shot metric."""

    name: str
    label: str
    category: str
    canonical_unit: str
    display_unit: str
    derived_from: tuple[str, ...] = ()
    description: str = ""


def _metric(
    name: str,
    label: str,
    category: str,
    unit: str,
    display: str,
    *,
    derived_from: tuple[str, ...] = (),
) -> MetricDefinition:
    return MetricDefinition(name, label, category, unit, display, derived_from)


METRICS: Final[dict[str, MetricDefinition]] = {
    item.name: item
    for item in (
        _metric("club_speed", "Club Speed", "club", "m/s", "mph"),
        _metric("attack_angle", "Attack Angle", "club", "rad", "deg"),
        _metric("club_path", "Club Path", "club", "rad", "deg"),
        _metric("face_angle", "Face Angle", "club", "rad", "deg"),
        _metric(
            "face_to_path",
            "Face to Path",
            "club",
            "rad",
            "deg",
            derived_from=("face_angle", "club_path"),
        ),
        _metric("dynamic_loft", "Dynamic Loft", "club", "rad", "deg"),
        _metric("dynamic_lie", "Dynamic Lie", "club", "rad", "deg"),
        _metric("spin_loft", "Spin Loft", "club", "rad", "deg"),
        _metric("swing_direction", "Swing Direction", "club", "rad", "deg"),
        _metric("swing_plane", "Swing Plane", "club", "rad", "deg"),
        _metric("low_point_distance", "Low Point", "club", "m", "in"),
        _metric("impact_height", "Impact Height", "club", "m", "mm"),
        _metric("impact_offset", "Impact Offset", "club", "m", "mm"),
        _metric("ball_speed", "Ball Speed", "launch", "m/s", "mph"),
        _metric("launch_angle", "Launch Angle", "launch", "rad", "deg"),
        _metric("launch_direction", "Launch Direction", "launch", "rad", "deg"),
        _metric("spin_rate", "Spin Rate", "launch", "rad/s", "rpm"),
        _metric("back_spin", "Back Spin", "launch", "rad/s", "rpm"),
        _metric("side_spin", "Side Spin", "launch", "rad/s", "rpm"),
        _metric("spin_axis", "Spin Axis", "launch", "rad", "deg"),
        _metric(
            "smash_factor",
            "Smash Factor",
            "launch",
            "1",
            "1",
            derived_from=("ball_speed", "club_speed"),
        ),
        _metric("carry_distance", "Carry Distance", "flight", "m", "yd"),
        _metric("total_distance", "Total Distance", "flight", "m", "yd"),
        _metric("roll_distance", "Roll Distance", "flight", "m", "yd"),
        _metric("lateral_carry", "Lateral Carry", "flight", "m", "yd"),
        _metric("lateral_total", "Lateral Total", "flight", "m", "yd"),
        _metric("apex_height", "Apex Height", "flight", "m", "ft"),
        _metric("flight_time", "Flight Time", "flight", "s", "s"),
        _metric("descent_angle", "Descent Angle", "flight", "rad", "deg"),
        _metric("curve", "Curve", "flight", "m", "yd"),
        _metric("putt_distance", "Putt Distance", "putting", "m", "ft"),
        _metric("skid_distance", "Skid Distance", "putting", "m", "in"),
        _metric("roll_speed", "Roll Speed", "putting", "m/s", "mph"),
    )
}

IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "shot_id",
    "session_id",
    "source_row",
    "monitor_vendor",
    "monitor_model",
    "software_version",
    "captured_at",
    "player",
    "club",
    "ball",
    "tags",
)


@dataclass(frozen=True)
class ColumnMapping:
    """Map a source column onto a canonical metric or identity field."""

    source_column: str
    target_column: str
    source_unit: str | None = None
    multiplier: float = 1.0
    measurement_status: str = "reported"

    def __post_init__(self) -> None:
        if not self.source_column.strip():
            raise ValueError("source_column must be non-empty")
        valid = set(METRICS) | set(IDENTITY_COLUMNS) | {"date", "time"}
        if self.target_column not in valid:
            raise ValueError(f"Unknown target column: {self.target_column}")
        if not pd.notna(self.multiplier) or self.multiplier == 0:
            raise ValueError("multiplier must be finite and non-zero")
        allowed_statuses = {"reported", "measured", "estimated", "derived", "unknown"}
        if self.measurement_status not in allowed_statuses:
            raise ValueError(
                "measurement_status must be reported, measured, estimated, "
                "derived, or unknown"
            )


@dataclass(frozen=True)
class ImportOptions:
    """User-controlled import options."""

    profile_id: str | None = None
    mappings: tuple[ColumnMapping, ...] = ()
    session_name: str | None = None
    player: str | None = None
    monitor_model: str | None = None
    software_version: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImportManifest:
    """Immutable provenance and mapping record for one source file."""

    source_path: str
    file_sha256: str
    profile_id: str
    vendor: str
    imported_at: str
    row_count: int
    source_columns: tuple[str, ...]
    metric_sources: dict[str, str]
    source_units: dict[str, str]
    unit_evidence: dict[str, str]
    warnings: tuple[str, ...] = ()


@dataclass
class ImportedSession:
    """Canonical shots and provenance for one imported session."""

    session_id: str
    name: str
    shots: pd.DataFrame
    manifest: ImportManifest
    source_path: Path | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if self.shots.empty:
            raise ValueError("ImportedSession must contain at least one shot")
        required = {"shot_id", "session_id", "source_row", "monitor_vendor"}
        missing = required - set(self.shots.columns)
        if missing:
            raise ValueError(f"shots missing required columns: {sorted(missing)}")


def numeric_metric_columns(frame: pd.DataFrame) -> list[str]:
    """Return canonical numeric metric columns present in ``frame``."""
    return [
        name
        for name in METRICS
        if name in frame.columns and pd.api.types.is_numeric_dtype(frame[name])
    ]
