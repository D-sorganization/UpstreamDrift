"""Target audit contracts and receipt generation for tour baselines (TB-01 #10586).

Provides full audit receipts recording content SHA, labels, residual validity,
missing spans, coverage fractions, measurement semantics, and native timing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

from .events import SWING_EVENTS_DRIVER, SWING_EVENTS_IRON, detect_tour_events
from .measurement_map import get_measurement_map
from .provenance import get_tour_provenance

AUDIT_SCHEMA = "tour-target-audit/1.0.0"


@dataclass(frozen=True)
class MarkerAuditDetail:
    """Detailed audit metrics for a single capture marker channel."""

    label: str
    measurement_class: str
    segment: str
    valid_samples: int
    missing_samples: int
    coverage_fraction: float
    missing_spans: tuple[tuple[int, int], ...]
    first_position_m: tuple[float, float, float] | None
    range_m: tuple[float, float, float] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "measurement_class": self.measurement_class,
            "segment": self.segment,
            "valid_samples": self.valid_samples,
            "missing_samples": self.missing_samples,
            "coverage_fraction": self.coverage_fraction,
            "missing_spans": [list(span) for span in self.missing_spans],
            "first_position_m": (
                list(self.first_position_m)
                if self.first_position_m is not None
                else None
            ),
            "range_m": list(self.range_m) if self.range_m is not None else None,
        }


@dataclass(frozen=True)
class TargetAuditReceipt:
    """Comprehensive, reproducible target audit receipt."""

    schema: str
    kind: str
    source_file: str
    source_sha256: str
    rate_hz: float
    frames: int
    duration_s: float
    sample_interval_s: float
    units: str
    vertical_axis: str
    handedness: str
    total_channels: int
    total_valid_samples: int
    total_missing_samples: int
    overall_coverage_fraction: float
    markers: dict[str, MarkerAuditDetail]
    events: dict[str, Any]
    provenance: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "rate_hz": self.rate_hz,
            "frames": self.frames,
            "duration_s": self.duration_s,
            "sample_interval_s": self.sample_interval_s,
            "units": self.units,
            "vertical_axis": self.vertical_axis,
            "handedness": self.handedness,
            "total_channels": self.total_channels,
            "total_valid_samples": self.total_valid_samples,
            "total_missing_samples": self.total_missing_samples,
            "overall_coverage_fraction": self.overall_coverage_fraction,
            "markers": {k: v.as_dict() for k, v in self.markers.items()},
            "events": self.events,
            "provenance": self.provenance,
        }

    def write_json(self, path: Path | str) -> None:
        """Serialize audit receipt to formatted JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=2, allow_nan=False)
            f.write("\n")


def _audit_marker_channel(
    capture: TourCapture,
    label: str,
    measurement_class: str,
    segment: str,
) -> MarkerAuditDetail:
    col = capture.index(label)
    pts = capture.points_m[:, col, :]
    val = capture.valid[:, col]
    valid_count = int(np.count_nonzero(val))
    missing_count = int(capture.frames - valid_count)
    coverage = float(valid_count / capture.frames)
    spans = capture.missing_spans(label)

    if valid_count > 0:
        first_valid = int(np.where(val)[0][0])
        first_pos = (
            float(pts[first_valid, 0]),
            float(pts[first_valid, 1]),
            float(pts[first_valid, 2]),
        )
        valid_pts = pts[val]
        ptp_vals = np.ptp(valid_pts, axis=0)
        extent = (
            float(ptp_vals[0]),
            float(ptp_vals[1]),
            float(ptp_vals[2]),
        )
    else:
        first_pos = None
        extent = None

    return MarkerAuditDetail(
        label=label,
        measurement_class=measurement_class,
        segment=segment,
        valid_samples=valid_count,
        missing_samples=missing_count,
        coverage_fraction=coverage,
        missing_spans=spans,
        first_position_m=first_pos,
        range_m=extent,
    )


@precondition(
    lambda capture, kind: isinstance(capture, TourCapture),
    "capture must be TourCapture",
)
@postcondition(lambda r: r.schema == AUDIT_SCHEMA, "receipt must follow audit schema")
def audit_tour_target(capture: TourCapture, kind: str) -> TargetAuditReceipt:
    """Generate exhaustive target audit receipt for a validated TourCapture."""
    normalized_kind = kind.strip().lower()
    m_map = get_measurement_map(normalized_kind)
    prov = get_tour_provenance(normalized_kind)

    # Use native events
    if normalized_kind == "driver":
        events = SWING_EVENTS_DRIVER
    elif normalized_kind == "iron":
        events = SWING_EVENTS_IRON
    else:
        events = detect_tour_events(capture, normalized_kind)

    markers: dict[str, MarkerAuditDetail] = {}
    for label in capture.labels:
        sem = m_map[label]
        markers[label] = _audit_marker_channel(
            capture=capture,
            label=label,
            measurement_class=sem.measurement_class.value,
            segment=sem.segment,
        )

    tot_valid = capture.valid_count()
    tot_missing = capture.missing_count()
    overall_cov = capture.coverage_fraction()

    return TargetAuditReceipt(
        schema=AUDIT_SCHEMA,
        kind=normalized_kind,
        source_file=f"C3D_TA_{normalized_kind.capitalize()}.c3d",
        source_sha256=capture.source_sha256 or "",
        rate_hz=capture.rate_hz,
        frames=capture.frames,
        duration_s=capture.duration_s,
        sample_interval_s=1.0 / capture.rate_hz if capture.rate_hz > 0 else 0.0,
        units="m",
        vertical_axis="y",
        handedness=prov.handedness,
        total_channels=len(capture.labels),
        total_valid_samples=tot_valid,
        total_missing_samples=tot_missing,
        overall_coverage_fraction=overall_cov,
        markers=markers,
        events=events.as_dict(),
        provenance=prov.as_dict(),
    )
