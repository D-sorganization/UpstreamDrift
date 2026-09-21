"""Data models and schemas for Tour Target Audit (TB-01 #10586)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any


class MeasurementCategory(str, Enum):
    """Categorization of motion capture spatial measurements."""

    OBSERVED_SURFACE = "observed_surface"
    INFERRED_JOINT_CENTER = "inferred_joint_center"
    CLUSTER_CENTROID = "cluster_centroid"
    CALIBRATED_POINT = "calibrated_point"


@dataclass(frozen=True)
class MeasurementEntry:
    """Individual measurement mapping entry."""

    name: str
    category: MeasurementCategory
    source_markers: tuple[str, ...]
    is_inferred: bool = False
    is_available: bool = True
    notes: str = ""


@dataclass(frozen=True)
class MeasurementMapping:
    """Versioned measurement map partitioning observations."""

    version: str = "1.0"
    entries: dict[str, MeasurementEntry] = field(default_factory=dict)

    def get_by_category(
        self, category: MeasurementCategory
    ) -> dict[str, MeasurementEntry]:
        return {k: v for k, v in self.entries.items() if v.category == category}


@dataclass(frozen=True)
class MarkerMissingSpan:
    """Detailed missing sample counts and contiguous missing intervals."""

    label: str
    valid_samples: int
    missing_samples: int
    coverage_ratio: float
    spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class EventPoint:
    """Discrete swing event point with inference metadata."""

    frame: int
    time_s: float
    is_inferred: bool = True
    detection_method: str = "inferred_clubhead_speed_peak"
    confidence: str = "high"


@dataclass(frozen=True)
class EventInterval:
    """Continuous swing interval."""

    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float


SwingEventInterval = EventInterval


@dataclass(frozen=True)
class SwingEvents:
    """Swing phase and event structure on native capture clock."""

    rate_hz: float
    address: EventInterval
    backswing: EventInterval
    top: EventPoint
    downswing: EventInterval
    impact: EventPoint
    follow_through: EventInterval


@dataclass(frozen=True)
class UnresolvedProvenanceItem:
    """Explicit declaration of unverified or missing provenance metadata."""

    field_name: str
    description: str
    status: str = "unresolved"


@dataclass(frozen=True)
class ProvenanceRecord:
    """Provenance audit separating subject anatomy, capture geometry, and gaps."""

    source_file: str
    capture_type: str
    averaging_normalization: str
    asserted_subject_anatomy: dict[str, Any]
    capture_specific_geometry: dict[str, Any]
    unresolved_provenance: tuple[UnresolvedProvenanceItem, ...]


@dataclass(frozen=True)
class TargetAudit:
    """Complete frozen target audit record."""

    capture_kind: str
    sha256: str
    rate_hz: float
    frames: int
    duration_s: float
    units: str
    vertical_axis: str
    handedness: str
    labels: tuple[str, ...]
    measurement_map: MeasurementMapping
    missing_spans: dict[str, MarkerMissingSpan]
    events: SwingEvents
    provenance: ProvenanceRecord
    duplicate_copies: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Convert audit record to serializable dictionary."""
        return {
            "capture_kind": self.capture_kind,
            "sha256": self.sha256,
            "rate_hz": self.rate_hz,
            "frames": self.frames,
            "duration_s": self.duration_s,
            "units": self.units,
            "vertical_axis": self.vertical_axis,
            "handedness": self.handedness,
            "labels": list(self.labels),
            "measurement_map": {
                "version": self.measurement_map.version,
                "entries": {
                    k: asdict(v) for k, v in self.measurement_map.entries.items()
                },
            },
            "missing_spans": {k: asdict(v) for k, v in self.missing_spans.items()},
            "events": {
                "rate_hz": self.events.rate_hz,
                "address": asdict(self.events.address),
                "backswing": asdict(self.events.backswing),
                "top": asdict(self.events.top),
                "downswing": asdict(self.events.downswing),
                "impact": asdict(self.events.impact),
                "follow_through": asdict(self.events.follow_through),
            },
            "provenance": {
                "source_file": self.provenance.source_file,
                "capture_type": self.provenance.capture_type,
                "averaging_normalization": self.provenance.averaging_normalization,
                "asserted_subject_anatomy": self.provenance.asserted_subject_anatomy,
                "capture_specific_geometry": self.provenance.capture_specific_geometry,
                "unresolved_provenance": [
                    asdict(u) for u in self.provenance.unresolved_provenance
                ],
            },
            "duplicate_copies": self.duplicate_copies,
        }

    def save_receipt(self, path: Path | str) -> None:
        """Write audit receipt JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_receipt(cls, path: Path | str) -> TargetAudit:
        """Load audit receipt from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        m_entries = {
            k: MeasurementEntry(
                name=v["name"],
                category=MeasurementCategory(v["category"]),
                source_markers=tuple(v["source_markers"]),
                is_inferred=v["is_inferred"],
                is_available=v["is_available"],
                notes=v["notes"],
            )
            for k, v in data["measurement_map"]["entries"].items()
        }
        missing = {
            k: MarkerMissingSpan(
                label=v["label"],
                valid_samples=v["valid_samples"],
                missing_samples=v["missing_samples"],
                coverage_ratio=v["coverage_ratio"],
                spans=tuple(tuple(s) for s in v["spans"]),
            )
            for k, v in data["missing_spans"].items()
        }
        ev = data["events"]
        events = SwingEvents(
            rate_hz=ev["rate_hz"],
            address=EventInterval(**ev["address"]),
            backswing=EventInterval(**ev["backswing"]),
            top=EventPoint(**ev["top"]),
            downswing=EventInterval(**ev["downswing"]),
            impact=EventPoint(**ev["impact"]),
            follow_through=EventInterval(**ev["follow_through"]),
        )
        pr = data["provenance"]
        prov = ProvenanceRecord(
            source_file=pr["source_file"],
            capture_type=pr["capture_type"],
            averaging_normalization=pr["averaging_normalization"],
            asserted_subject_anatomy=pr["asserted_subject_anatomy"],
            capture_specific_geometry=pr["capture_specific_geometry"],
            unresolved_provenance=tuple(
                UnresolvedProvenanceItem(**u) for u in pr["unresolved_provenance"]
            ),
        )
        return cls(
            capture_kind=data["capture_kind"],
            sha256=data["sha256"],
            rate_hz=data["rate_hz"],
            frames=data["frames"],
            duration_s=data["duration_s"],
            units=data["units"],
            vertical_axis=data["vertical_axis"],
            handedness=data["handedness"],
            labels=tuple(data["labels"]),
            measurement_map=MeasurementMapping(
                version=data["measurement_map"]["version"], entries=m_entries
            ),
            missing_spans=missing,
            events=events,
            provenance=prov,
            duplicate_copies=data["duplicate_copies"],
        )
