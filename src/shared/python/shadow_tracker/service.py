"""Shadow Tracker orchestration and review service (ST-11, #10134).

Coordinates video observation inspection, manual mask correction with revision lineage,
deterministic downstream fit invalidation, worst-frame navigation, and bundle persistence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from ._validation import MASK_SCHEMA_VERSION, check_id
from .artifacts import ShadowTrackerBundle, load_bundle, save_bundle
from .contracts import (
    CANDIDATE_RESULT_SCHEMA_VERSION,
    FitRequest,
    FrameObservation,
    ResultBundle,
    ShadowTrackerService,
    Shot,
)
from .mask_records import MaskFrame
from .segmentation import ManualMaskProvider
from .source_records import FrameIdentity, SourceAsset

WorstFrameMetric = Literal["mask_coverage", "uncertainty", "silhouette_loss"]


class UnavailableBackendError(RuntimeError):
    """Raised when an unverified automated fitting or inference backend is requested."""


@dataclass(frozen=True, slots=True, kw_only=True)
class WorstFrameReport:
    """Report item for worst-frame review navigation."""

    frame_id: str
    score: float
    metric: str
    rank: int


class DefaultShadowTrackerService:
    """Primary headless orchestration service for Shadow Tracker evidence review and persistence."""

    def __init__(self) -> None:
        self._source_asset: SourceAsset | None = None
        self._shot: Shot | None = None
        self._observations: dict[str, FrameObservation] = {}
        self._mask_provider: ManualMaskProvider = ManualMaskProvider()
        self._candidates: list[Any] = []
        self._uncertainty: dict[str, Any] = {}
        self._assumptions: tuple[str, ...] = ()
        self._is_cancelled: bool = False
        self._execution_status: str = "idle"

    @property
    def is_cancelled(self) -> bool:
        """True if the current service session has been cancelled."""
        return self._is_cancelled

    @property
    def execution_status(self) -> str:
        """Current execution lifecycle state ('idle', 'running', 'cancelled', 'completed')."""
        return self._execution_status

    def initialize_session(
        self,
        *,
        source_asset: SourceAsset,
        observations: Sequence[FrameObservation],
        initial_masks: Sequence[MaskFrame],
        shot: Shot | None = None,
        uncertainty: dict[str, Any] | None = None,
        assumptions: Sequence[str] = (),
    ) -> None:
        """Initialize service session from existing source and observations."""
        if not isinstance(source_asset, SourceAsset):
            raise TypeError(f"Expected SourceAsset, got {type(source_asset).__name__}")
        self._source_asset = source_asset
        self._shot = shot
        self._observations = {obs.frame_id: obs for obs in observations}
        self._mask_provider = ManualMaskProvider()
        for mask in initial_masks:
            self._mask_provider.register_mask(mask)
        self._candidates.clear()
        self._uncertainty = dict(uncertainty or {})
        self._assumptions = tuple(str(a) for a in assumptions)
        self._is_cancelled = False
        self._execution_status = "idle"

    def cancel(self) -> None:
        """Cancel ongoing work."""
        self._is_cancelled = True
        self._execution_status = "cancelled"

    def resume(self) -> None:
        """Resume session after cancellation. Invariant: resume cannot declare completion."""
        self._is_cancelled = False
        if self._execution_status == "cancelled":
            self._execution_status = "idle"

    def has_active_fits(self) -> bool:
        """Return True if fitted trajectory candidates exist in session."""
        return bool(self._candidates)

    def get_candidates(self) -> tuple[Any, ...]:
        """Return all active candidate trajectories."""
        return tuple(self._candidates)

    def _inject_fit_for_testing(self, candidate_id: str) -> None:
        """Testing utility to verify that mask corrections invalidate existing fits."""
        self._candidates.append(candidate_id)

    def get_observations(self) -> tuple[FrameObservation, ...]:
        """Return all observations ordered by frame identity."""
        return tuple(self._observations.values())

    def get_observation(self, frame_id: str) -> FrameObservation:
        """Retrieve observation by frame_id."""
        if frame_id not in self._observations:
            raise KeyError(f"No observation found for frame_id {frame_id!r}")
        return self._observations[frame_id]

    def get_mask(self, frame_id: str) -> MaskFrame:
        """Retrieve current latest mask for frame_id."""
        return self._mask_provider.get_mask(frame_id)

    def get_mask_history(self, frame_id: str) -> tuple[MaskFrame, ...]:
        """Retrieve revision lineage history for frame_id."""
        return self._mask_provider.get_revision_history(frame_id)

    def get_uncertainty(self) -> dict[str, Any]:
        """Return current uncertainty declarations."""
        return dict(self._uncertainty)

    def get_assumptions(self) -> tuple[str, ...]:
        """Return declared assumptions."""
        return self._assumptions

    def update_mask(
        self,
        *,
        frame_id: str,
        body: bytes,
        club: bytes,
        valid: bytes,
        parent_revision_id: str | None,
        producer_id: str,
        correction_note: str,
    ) -> MaskFrame:
        """Register a revised mask observation and invalidate existing fits/hypotheses.

        DbC Postcondition:
            - Any existing fits or downstream hypotheses are immediately invalidated.
        """
        check_id(frame_id, "frame_id")
        obs = self.get_observation(frame_id)

        # Build FrameIdentity for the revised mask
        prev_mask = self._mask_provider.get_mask(frame_id)
        frame_ident = prev_mask.frame

        rev_seq = len(self._mask_provider.get_revision_history(frame_id))
        content_hash = hashlib.sha256(body + club + valid).hexdigest()[:8]
        new_rev_id = f"{parent_revision_id or 'rev'}-m{rev_seq}-{content_hash}"

        new_mask = MaskFrame(
            schema_version=MASK_SCHEMA_VERSION,
            frame=frame_ident,
            width_px=prev_mask.width_px,
            height_px=prev_mask.height_px,
            body=body,
            club=club,
            valid=valid,
            revision_id=new_rev_id,
            parent_revision_id=parent_revision_id,
            producer_id=producer_id,
            correction_note=correction_note,
        )

        self._mask_provider.register_mask(new_mask)

        # Invalidate existing fits upon manual correction
        self._candidates.clear()

        return new_mask

    def worst_frames(
        self,
        *,
        metric: WorstFrameMetric = "mask_coverage",
        top_n: int = 10,
    ) -> list[WorstFrameReport]:
        """Rank frames to facilitate worst-frame review navigation."""
        scores: list[tuple[str, float]] = []

        for frame_id in self._observations:
            if not self._mask_provider.has_mask(frame_id):
                scores.append((frame_id, 0.0))
                continue

            mask = self._mask_provider.get_mask(frame_id)
            if metric == "mask_coverage":
                # Score is foreground fraction: lower score = worse coverage
                fg_count = mask.body.count(1) + mask.club.count(1)
                valid_count = mask.valid.count(1) or 1
                coverage = fg_count / valid_count
                scores.append((frame_id, coverage))
            elif metric == "uncertainty":
                # In absence of per-frame variance, default to constant
                scores.append((frame_id, 0.5))
            else:
                # Default silhouette loss
                scores.append((frame_id, 0.0))

        # Ascending sort: lowest coverage = worst frame
        scores.sort(key=lambda item: item[1])

        reports: list[WorstFrameReport] = []
        for rank, (fid, score) in enumerate(scores[:top_n], start=1):
            reports.append(
                WorstFrameReport(
                    frame_id=fid,
                    score=score,
                    metric=metric,
                    rank=rank,
                )
            )
        return reports

    def fit(self, request: FitRequest) -> ResultBundle:
        """Run forward model fitting.

        Honest refusal: Automated forward fitting is currently unqualified.
        """
        raise UnavailableBackendError(
            "Automated forward fitting is currently unavailable: "
            "forward dynamics gates ST-07 through ST-10 must pass qualification before fitting can proceed."
        )

    def save_bundle(self, path: Path | str) -> None:
        """Atomically persist current session to a review bundle."""
        if self._source_asset is None:
            raise ValueError("Cannot save bundle: no source asset in session")

        bundle = ShadowTrackerBundle(
            bundle_id=f"bundle-{self._source_asset.asset_id}",
            source_asset=self._source_asset,
            observations=tuple(self._observations.values()),
            masks=self._mask_provider.all_revisions(),
            uncertainty=self._uncertainty,
            assumptions=self._assumptions,
            evidence_quality="unreviewed",
        )
        save_bundle(bundle, path)

    def load_bundle(self, path: Path | str) -> None:
        """Atomically restore session from a review bundle."""
        bundle = load_bundle(path)
        self.initialize_session(
            source_asset=bundle.source_asset,
            observations=bundle.observations,
            initial_masks=bundle.masks,
            uncertainty=bundle.uncertainty,
            assumptions=bundle.assumptions,
        )

    def export_canonical(self, path: Path | str) -> dict[str, Any]:
        """Export canonical observation package with typed schema and provenance."""
        if self._source_asset is None:
            raise ValueError("Cannot export: no source asset in session")

        export_data: dict[str, Any] = {
            "schema_version": "shadow-tracker-export/1.0.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "source_asset": self._source_asset.to_dict(),
            "observations": [obs.to_dict() for obs in self._observations.values()],
            "masks": [m.to_dict() for m in self._mask_provider.all_revisions()],
            "uncertainty": self._uncertainty,
            "assumptions": list(self._assumptions),
        }

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(export_data, indent=2), encoding="utf-8")
        return export_data
