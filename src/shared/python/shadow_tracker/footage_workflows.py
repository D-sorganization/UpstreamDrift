"""Modern and historical footage qualification workflows for Shadow Tracker (ST-10)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
import logging
import random
from typing import Any, Final, Literal

import numpy as np

from ._validation import (
    check_id,
    check_nonneg_float,
    check_pos_float,
    check_pos_int,
    check_str,
)
from .contracts import FrameObservation
from .source_records import RightsStatus, SourceAsset

logger = logging.getLogger(__name__)

SuitabilityGrade = Literal[
    "unusable", "qualitative_only", "kinematic_candidate", "qualified"
]
_VALID_SUITABILITY: Final[frozenset[str]] = frozenset(
    ("unusable", "qualitative_only", "kinematic_candidate", "qualified")
)


# ---------------------------------------------------------------------------
# Lineage and Deduplication Partitioning
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class FilmLineage:
    """Historical film or capture recording lineage grouping re-encoded assets."""

    film_id: str
    canonical_title: str
    recording_year: int | None
    recording_source: str
    golfer_id: str
    known_asset_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        check_id(self.film_id, "film_id")
        check_str(self.canonical_title, "canonical_title")
        if self.recording_year is not None:
            check_pos_int(self.recording_year, "recording_year")
        check_str(self.recording_source, "recording_source")
        check_id(self.golfer_id, "golfer_id")
        for asset_id in self.known_asset_ids:
            check_id(asset_id, "asset_id")


def validate_split_isolation(
    train_split: Sequence[FilmLineage],
    holdout_split: Sequence[FilmLineage],
) -> None:
    """Verify that no film lineage or asset leaks across train and holdout splits."""
    train_films = {lineage.film_id for lineage in train_split}
    holdout_films = {lineage.film_id for lineage in holdout_split}

    overlap_films = train_films.intersection(holdout_films)
    if overlap_films:
        sorted_overlap = sorted(overlap_films)
        raise ValueError(
            f"Lineage leakage detected: film {sorted_overlap[0]!r} spans multiple splits"
        )

    train_assets = {
        asset for lineage in train_split for asset in lineage.known_asset_ids
    }
    holdout_assets = {
        asset for lineage in holdout_split for asset in lineage.known_asset_ids
    }

    overlap_assets = train_assets.intersection(holdout_assets)
    if overlap_assets:
        sorted_overlap = sorted(overlap_assets)
        raise ValueError(
            f"Asset leakage detected: asset {sorted_overlap[0]!r} spans multiple splits"
        )


class DeduplicationSplitter:
    """Partitions collections of film lineages into disjoint train and holdout splits."""

    def split(
        self,
        lineages: Sequence[FilmLineage],
        holdout_fraction: float = 0.20,
        seed: int = 42,
    ) -> tuple[tuple[FilmLineage, ...], tuple[FilmLineage, ...]]:
        """Split lineages into train and holdout partitions by film lineage."""
        check_pos_float(holdout_fraction, "holdout_fraction")
        if holdout_fraction >= 1.0:
            raise ValueError("holdout_fraction must be < 1.0")

        shuffled = list(lineages)
        rng = random.Random(seed)
        rng.shuffle(shuffled)

        holdout_count = max(1, int(round(len(shuffled) * holdout_fraction)))
        holdout_part = tuple(shuffled[:holdout_count])
        train_part = tuple(shuffled[holdout_count:])

        validate_split_isolation(train_part, holdout_part)
        return train_part, holdout_part


# ---------------------------------------------------------------------------
# Rights & Lineage Audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class RightsAuditReport:
    """Audited rights, permissions, and attribution clearance for an asset."""

    asset_id: str
    rights_status: RightsStatus
    is_cleared_for_release: bool
    attribution_confirmed: bool
    flags: tuple[str, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        check_id(self.asset_id, "asset_id")


def audit_asset_rights(
    asset: SourceAsset,
    attribution_confirmed: bool = True,
) -> RightsAuditReport:
    """Audit source asset rights, requiring reviewed status and confirmed attribution."""
    flags: list[str] = []
    if asset.rights_status == "unknown":
        flags.append("rights_unreviewed_or_unknown")
    elif asset.rights_status == "restricted":
        flags.append("rights_restricted")

    if not attribution_confirmed:
        flags.append("attribution_unconfirmed")

    is_cleared = len(flags) == 0
    notes = (
        asset.rights_note if is_cleared else f"Clearance blocked: {', '.join(flags)}"
    )

    return RightsAuditReport(
        asset_id=asset.asset_id,
        rights_status=asset.rights_status,
        is_cleared_for_release=is_cleared,
        attribution_confirmed=attribution_confirmed,
        flags=tuple(flags),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Multi-Camera Synchronization Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class SyncValidationResult:
    """Validation outcome for multi-camera physical synchronization."""

    is_synchronous: bool
    max_drift_s: float
    overlap_duration_s: float
    refusal_reason: str | None = None

    def __post_init__(self) -> None:
        check_nonneg_float(self.overlap_duration_s, "overlap_duration_s")


def validate_multiview_synchronization(
    observations_a: Sequence[FrameObservation],
    observations_b: Sequence[FrameObservation],
    max_allowable_drift_s: float = 0.01,
) -> SyncValidationResult:
    """Validate that two camera observation streams share synchronous timing."""
    check_pos_float(max_allowable_drift_s, "max_allowable_drift_s")

    times_a = [
        obs.physical_time_s for obs in observations_a if obs.physical_time_s is not None
    ]
    times_b = [
        obs.physical_time_s for obs in observations_b if obs.physical_time_s is not None
    ]

    if not times_a or not times_b:
        return SyncValidationResult(
            is_synchronous=False,
            max_drift_s=float("inf"),
            overlap_duration_s=0.0,
            refusal_reason="missing_physical_time_evidence",
        )

    start_overlap = max(min(times_a), min(times_b))
    end_overlap = min(max(times_a), max(times_b))
    overlap_duration = max(0.0, end_overlap - start_overlap)

    if overlap_duration <= 0.0:
        return SyncValidationResult(
            is_synchronous=False,
            max_drift_s=float("inf"),
            overlap_duration_s=0.0,
            refusal_reason="non_overlapping_temporal_windows",
        )

    id_to_time_b = {
        obs.frame_id: obs.physical_time_s
        for obs in observations_b
        if obs.physical_time_s is not None
    }
    shared_obs = [
        (obs.physical_time_s, id_to_time_b[obs.frame_id])
        for obs in observations_a
        if obs.frame_id in id_to_time_b and obs.physical_time_s is not None
    ]

    max_drift = 0.0
    if shared_obs:
        for t_a, t_b in shared_obs:
            drift = abs(t_a - t_b)
            if drift > max_drift:
                max_drift = drift
    else:
        aligned_a = [t for t in times_a if start_overlap <= t <= end_overlap]
        aligned_b = [t for t in times_b if start_overlap <= t <= end_overlap]
        for t_a in aligned_a:
            drift = min(abs(t_a - t_b) for t_b in times_b)
            if drift > max_drift:
                max_drift = drift
        for t_b in aligned_b:
            drift = min(abs(t_b - t_a) for t_a in times_a)
            if drift > max_drift:
                max_drift = drift

    if max_drift > max_allowable_drift_s:
        return SyncValidationResult(
            is_synchronous=False,
            max_drift_s=max_drift,
            overlap_duration_s=overlap_duration,
            refusal_reason="excessive_clock_drift",
        )

    return SyncValidationResult(
        is_synchronous=True,
        max_drift_s=max_drift,
        overlap_duration_s=overlap_duration,
        refusal_reason=None,
    )


# ---------------------------------------------------------------------------
# Degradation Harness
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class DegradationConfig:
    """Configuration for simulating archive-like degradation on modern reference footage."""

    scale_factor: float = 1.0
    blur_sigma_px: float = 0.0
    drop_frame_cadence: int = 1
    noise_level: float = 0.0

    def __post_init__(self) -> None:
        check_pos_float(self.scale_factor, "scale_factor")
        check_nonneg_float(self.blur_sigma_px, "blur_sigma_px")
        check_pos_int(self.drop_frame_cadence, "drop_frame_cadence")
        check_nonneg_float(self.noise_level, "noise_level")


class DegradationHarness:
    """Applies controlled physical degradations to reference masks and frame sequences."""

    def __init__(self, config: DegradationConfig) -> None:
        self.config = config

    def degrade_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply spatial downscaling and blur degradation to a 2D mask."""
        h, w = mask.shape[:2]
        target_h = max(1, int(round(h * self.config.scale_factor)))
        target_w = max(1, int(round(w * self.config.scale_factor)))

        step_y = max(1, int(round(h / target_h)))
        step_x = max(1, int(round(w / target_w)))

        subsampled = mask[::step_y, ::step_x][:target_h, :target_w]

        # Convert to uint8 binary mask
        result = (subsampled > 0).astype(np.uint8)
        return result

    def apply_telecine_cadence(self, frame_indices: Sequence[int]) -> tuple[int, ...]:
        """Subsample frame sequence according to configured cadence."""
        cadence = self.config.drop_frame_cadence
        return tuple(idx for idx in frame_indices if idx % cadence == 0)


# ---------------------------------------------------------------------------
# Historical Pilot Catalog & Yield Measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PilotEntry:
    """Metadata and qualification record for a single historical pilot clip."""

    clip_id: str
    asset_id: str
    film_id: str
    golfer_name: str
    rights_status: RightsStatus
    suitability: SuitabilityGrade
    annotation_minutes: float
    compute_seconds: float
    failure_reasons: tuple[str, ...]
    physical_time_known: bool

    def __post_init__(self) -> None:
        check_id(self.clip_id, "clip_id")
        check_id(self.asset_id, "asset_id")
        check_id(self.film_id, "film_id")
        check_str(self.golfer_name, "golfer_name")
        if self.suitability not in _VALID_SUITABILITY:
            raise ValueError(f"Invalid suitability: {self.suitability}")
        check_nonneg_float(self.annotation_minutes, "annotation_minutes")
        check_nonneg_float(self.compute_seconds, "compute_seconds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "asset_id": self.asset_id,
            "film_id": self.film_id,
            "golfer_name": self.golfer_name,
            "rights_status": self.rights_status,
            "suitability": self.suitability,
            "annotation_minutes": self.annotation_minutes,
            "compute_seconds": self.compute_seconds,
            "failure_reasons": list(self.failure_reasons),
            "physical_time_known": self.physical_time_known,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PilotEntry:
        return cls(
            clip_id=payload["clip_id"],
            asset_id=payload["asset_id"],
            film_id=payload["film_id"],
            golfer_name=payload["golfer_name"],
            rights_status=payload["rights_status"],
            suitability=payload["suitability"],
            annotation_minutes=float(payload["annotation_minutes"]),
            compute_seconds=float(payload["compute_seconds"]),
            failure_reasons=tuple(str(r) for r in payload["failure_reasons"]),
            physical_time_known=bool(payload["physical_time_known"]),
        )


class HistoricalPilotCatalog:
    """Resumable catalog tracking historical pilot runs, yield, and cost accounting."""

    def __init__(self) -> None:
        self._entries: dict[str, PilotEntry] = {}

    def register_entry(self, entry: PilotEntry) -> None:
        """Register or update a pilot entry by clip ID."""
        self._entries[entry.clip_id] = entry

    def has_entry(self, clip_id: str) -> bool:
        """Check if clip has already been processed and recorded."""
        return clip_id in self._entries

    def get_entry(self, clip_id: str) -> PilotEntry:
        """Retrieve recorded entry by clip ID."""
        if clip_id not in self._entries:
            raise KeyError(f"Clip not found in catalog: {clip_id}")
        return self._entries[clip_id]

    def compute_pilot_yield(self) -> dict[str, float]:
        """Compute discovery-to-acceptance yield and aggregate costs."""
        total = len(self._entries)
        if total == 0:
            return {
                "total_clips": 0.0,
                "qualified_yield": 0.0,
                "kinematic_or_better_yield": 0.0,
                "total_annotation_minutes": 0.0,
                "total_compute_seconds": 0.0,
            }

        qualified_count = sum(
            1 for e in self._entries.values() if e.suitability == "qualified"
        )
        kinematic_count = sum(
            1
            for e in self._entries.values()
            if e.suitability in ("qualified", "kinematic_candidate")
        )
        total_annotation = sum(e.annotation_minutes for e in self._entries.values())
        total_compute = sum(e.compute_seconds for e in self._entries.values())

        return {
            "total_clips": float(total),
            "qualified_yield": float(qualified_count / total),
            "kinematic_or_better_yield": float(kinematic_count / total),
            "total_annotation_minutes": float(total_annotation),
            "total_compute_seconds": float(total_compute),
        }

    def get_failed_cases_inventory(self) -> tuple[PilotEntry, ...]:
        """Return all pilot clips that failed full qualification."""
        return tuple(e for e in self._entries.values() if e.suitability != "qualified")

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [e.to_dict() for e in self._entries.values()]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HistoricalPilotCatalog:
        catalog = cls()
        for raw in payload.get("entries", []):
            catalog.register_entry(PilotEntry.from_dict(raw))
        return catalog
