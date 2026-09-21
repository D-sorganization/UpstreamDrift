"""Canonical targets consumed by kinematic reference and dynamics fitting paths (TB-01 #10586).

Unifies validated TourCapture, versioned measurement maps, native swing events,
and complete provenance under a single facade. Validates that missing club
clusters or phases are never scored as zero error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.tour_capture_contract import (
    TourCapture,
    load_tour_capture,
    verify_capture_content,
)

from .audit import TargetAuditReceipt, audit_tour_target
from .events import (
    SWING_EVENTS_DRIVER,
    SWING_EVENTS_IRON,
    TourSwingEvents,
    detect_tour_events,
)
from .measurement_map import (
    MarkerMeasurementSemantics,
    MeasurementClass,
    get_measurement_map,
)
from .provenance import TourProvenance, get_tour_provenance


@dataclass(frozen=True)
class CanonicalTourTarget:
    """Canonical tour swing target facade."""

    kind: str
    capture: TourCapture
    measurement_map: MappingProxyType[str, MarkerMeasurementSemantics]
    events: TourSwingEvents
    provenance: TourProvenance

    @property
    def frames(self) -> int:
        return self.capture.frames

    @property
    def rate_hz(self) -> float:
        return self.capture.rate_hz

    @property
    def duration_s(self) -> float:
        return self.capture.duration_s

    @property
    def labels(self) -> tuple[str, ...]:
        return self.capture.labels

    @property
    def time_s(self) -> np.ndarray:
        return self.capture.time_s

    @property
    def points_m(self) -> np.ndarray:
        return self.capture.points_m

    @property
    def valid(self) -> np.ndarray:
        return self.capture.valid

    def index(self, label: str) -> int:
        return self.capture.index(label)

    def valid_count(self) -> int:
        return self.capture.valid_count()

    def missing_count(self) -> int:
        return self.capture.missing_count()

    def coverage_fraction(self, label: str | None = None) -> float:
        return self.capture.coverage_fraction(label)

    def missing_spans(self, label: str) -> tuple[tuple[int, int], ...]:
        return self.capture.missing_spans(label)

    def tracked_labels(self) -> tuple[str, ...]:
        """Return non-sentinel labels with valid biomechanical or club tracking role."""
        return tuple(
            lbl
            for lbl in self.labels
            if self.measurement_map[lbl].measurement_class
            != MeasurementClass.UNASSIGNED_OR_SENTINEL
        )

    def as_audit_receipt(self) -> TargetAuditReceipt:
        return audit_tour_target(self.capture, self.kind)


@precondition(lambda path, **_: Path(path).is_file(), "path must name an existing file")
def load_canonical_tour_target(
    path: Path | str,
    kind: str | None = None,
) -> CanonicalTourTarget:
    """Read a canonical C3D, verify its content SHA, and return CanonicalTourTarget."""
    target_path = Path(path)
    detected_kind, _ = verify_capture_content(target_path, expected_kind=kind)
    capture = load_tour_capture(target_path)
    m_map = get_measurement_map(detected_kind)
    prov = get_tour_provenance(detected_kind)

    if detected_kind == "driver":
        events = SWING_EVENTS_DRIVER
    elif detected_kind == "iron":
        events = SWING_EVENTS_IRON
    else:
        events = detect_tour_events(capture, detected_kind)

    return CanonicalTourTarget(
        kind=detected_kind,
        capture=capture,
        measurement_map=m_map,
        events=events,
        provenance=prov,
    )


def _compute_group_rmse(
    dist_m: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Calculate RMSE in mm for a masked population, returning NaN if empty."""
    valid_dists = dist_m[mask]
    if valid_dists.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(valid_dists**2)) * 1000.0)


@precondition(
    lambda target, pred, **_: pred.shape == target.points_m.shape,
    "predicted points must match target shape (frames, markers, 3)",
)
def evaluate_target_tracking_error(
    target: CanonicalTourTarget,
    predicted_points_m: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute RMSE tracking errors in mm against measured valid data.

    Missing club clusters or entirely missing phases are NEVER scored as 0.0 error;
    when no valid samples exist in a category, the result is float('nan').
    """
    effective_valid = (
        target.valid if valid_mask is None else (target.valid & valid_mask)
    )

    diff = predicted_points_m - target.points_m
    dist_m = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    overall_rmse = _compute_group_rmse(dist_m, effective_valid)

    # Club segment (grip + head)
    club_cols = [
        target.index(lbl)
        for lbl in target.labels
        if target.measurement_map[lbl].measurement_class
        == MeasurementClass.OBSERVED_CLUSTER_CENTROID
    ]
    club_mask = np.zeros_like(effective_valid, dtype=bool)
    club_mask[:, club_cols] = effective_valid[:, club_cols]
    club_rmse = _compute_group_rmse(dist_m, club_mask)

    # Feet segment
    feet_cols = [
        target.index(lbl)
        for lbl in target.labels
        if any(term in lbl for term in ("Toe", "Ankle"))
    ]
    feet_mask = np.zeros_like(effective_valid, dtype=bool)
    feet_mask[:, feet_cols] = effective_valid[:, feet_cols]
    feet_rmse = _compute_group_rmse(dist_m, feet_mask)

    return {
        "overall_rmse_mm": overall_rmse,
        "club_rmse_mm": club_rmse,
        "feet_rmse_mm": feet_rmse,
    }
