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

import numpy as np

from ._validation import (
    CANDIDATE_RESULT_SCHEMA_VERSION,
    MASK_SCHEMA_VERSION,
    RESULT_BUNDLE_SCHEMA_VERSION,
    check_id,
)
from .artifacts import ShadowTrackerBundle, load_bundle, save_bundle
from .contracts import (
    CANONICAL_ARTICULATED_CONVENTION,
    CandidateResult,
    FitRequest,
    ForwardModel,
    FrameObservation,
    ModelCapabilities,
    POINT_LANDMARKS_CONVENTION,
    ReplayAudit,
    ResultBundle,
    ShadowTrackerService,
    Shot,
    SilhouetteRenderer,
)
from .evaluation import create_evaluated_result_bundle
from .fitting import (
    ControlFitter,
    OptimizationCheckpoint,
    OptimizationConfig,
)
from .mask_records import MaskFrame
from .projection import PinholeCameraModel, camera_calibration_fingerprint
from .segmentation import ManualMaskProvider
from .source_records import FrameIdentity, SourceAsset

WorstFrameMetric = Literal["mask_coverage", "uncertainty", "silhouette_loss"]


class UnavailableBackendError(RuntimeError):
    """Raised when an unverified automated fitting or inference backend is requested."""


class StaleCheckpointError(RuntimeError):
    """Raised when attempting to resume from or replay a checkpoint whose masks or camera have changed."""


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
        self._forward_model: ForwardModel | None = None
        self._renderer: SilhouetteRenderer | None = None
        self._camera: PinholeCameraModel | None = None
        self._checkpoints: list[OptimizationCheckpoint] = []

    @property
    def is_cancelled(self) -> bool:
        """True if the current service session has been cancelled."""
        return self._is_cancelled

    @property
    def execution_status(self) -> str:
        """Current execution lifecycle state ('idle', 'running', 'cancelled', 'completed')."""
        return self._execution_status

    @property
    def checkpoints(self) -> tuple[OptimizationCheckpoint, ...]:
        """Return all recorded optimization checkpoints from the most recent run."""
        return tuple(self._checkpoints)

    def register_backend(
        self,
        *,
        forward_model: ForwardModel,
        renderer: SilhouetteRenderer,
        camera: PinholeCameraModel,
    ) -> None:
        """Register dynamic forward model, renderer, and camera models for fitting."""
        self._forward_model = forward_model
        self._renderer = renderer
        self._camera = camera

    def update_camera(self, camera: PinholeCameraModel) -> None:
        """Update session camera calibration and invalidate existing fits and checkpoints.

        In-package silhouette renderers expose ``update_camera`` and are refreshed in
        place so rendering reflects the recalibration; renderers without the hook must
        be re-registered by their owner via :meth:`register_backend`.
        """
        if not isinstance(camera, PinholeCameraModel):
            raise TypeError(f"Expected PinholeCameraModel, got {type(camera).__name__}")
        renderer = self._renderer
        if renderer is not None and hasattr(renderer, "update_camera"):
            renderer.update_camera(camera)
        self._camera = camera
        self._candidates.clear()
        self._checkpoints.clear()

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
        self._checkpoints.clear()
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

        # Invalidate existing fits and checkpoints upon manual correction
        self._candidates.clear()
        self._checkpoints.clear()

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

    def resume_from_checkpoint(
        self,
        checkpoint: OptimizationCheckpoint,
        request: FitRequest,
    ) -> ResultBundle:
        """Resume optimization from a previously recorded checkpoint.

        Preconditions:
            - Checkpoint's mask revision IDs must match the current session masks.
            - Checkpoint's camera_id must match the current session camera.
        """
        if not isinstance(checkpoint, OptimizationCheckpoint):
            raise TypeError(
                f"Expected OptimizationCheckpoint, got {type(checkpoint).__name__}"
            )

        current_mask_revs = tuple(
            self._mask_provider.get_mask(fid).revision_id
            for fid in sorted(self._observations.keys())
            if self._mask_provider.has_mask(fid)
        )
        if checkpoint.mask_revision_ids != current_mask_revs:
            raise StaleCheckpointError(
                f"Checkpoint is stale: mask revisions ({checkpoint.mask_revision_ids}) "
                f"do not match current session masks ({current_mask_revs})."
            )

        if self._camera is None:
            raise StaleCheckpointError(
                "Checkpoint is stale: session has no registered camera "
                f"(checkpoint camera {checkpoint.camera_id!r})."
            )
        if checkpoint.camera_id != self._camera.camera_id:
            raise StaleCheckpointError(
                f"Checkpoint is stale: camera ({checkpoint.camera_id!r}) "
                f"does not match current session camera ({getattr(self._camera, 'camera_id', None)!r})."
            )
        current_calibration_sha = camera_calibration_fingerprint(self._camera)
        if (
            not checkpoint.camera_calibration_sha256
            or checkpoint.camera_calibration_sha256 != current_calibration_sha
        ):
            raise StaleCheckpointError(
                "Checkpoint is stale: camera calibration changed since the checkpoint "
                f"(checkpoint fingerprint {checkpoint.camera_calibration_sha256[:12]!r} "
                f"vs current {current_calibration_sha[:12]!r})."
            )

        return self.fit(request, initial_controls=checkpoint.parameters)

    def _validate_fit_capabilities(
        self,
        request: FitRequest,
    ) -> tuple[
        ForwardModel,
        SilhouetteRenderer,
        PinholeCameraModel,
        ModelCapabilities,
        bool,
    ]:
        """Validate backend presence, capabilities, and qualification gates."""
        if not isinstance(request, FitRequest):
            raise TypeError(f"Expected FitRequest, got {type(request).__name__}")

        if not self._observations:
            raise ValueError("Cannot fit: session has no observations")

        if (
            self._forward_model is None
            or self._renderer is None
            or self._camera is None
        ):
            raise UnavailableBackendError(
                "Automated forward fitting is currently unavailable: "
                "forward dynamics gates ST-07 through ST-10 must pass qualification before fitting can proceed."
            )

        caps = self._forward_model.capabilities()
        if not caps.is_available:
            raise UnavailableBackendError(
                f"Backend runtime is unavailable: forward model {type(self._forward_model).__name__} is not available."
            )

        if request.engine_capability_requirement:
            supported = (
                set(caps.actuator_modes)
                | set(caps.contact_modes)
                | set(caps.supported_bodies)
                | {caps.state_convention}
            )
            for req_cap in request.engine_capability_requirement:
                if req_cap not in supported:
                    raise UnavailableBackendError(
                        f"Backend does not satisfy required engine capability: {req_cap!r}"
                    )

        is_synthetic = bool(
            getattr(caps, "is_synthetic", False)
            or getattr(self._forward_model, "is_synthetic", False)
        )
        is_qualified = bool(
            getattr(caps, "is_qualified", False)
            or getattr(self._forward_model, "is_qualified", False)
        )
        if not is_qualified and not is_synthetic:
            raise UnavailableBackendError(
                "Automated forward fitting is currently unavailable: backend is unqualified for scientific release."
            )

        return (
            self._forward_model,
            self._renderer,
            self._camera,
            caps,
            is_synthetic,
        )

    def _build_cancelled_bundle(self, request: FitRequest) -> ResultBundle:
        """Construct a fail-closed cancelled ResultBundle."""
        self._execution_status = "cancelled"
        return ResultBundle(
            schema_version=RESULT_BUNDLE_SCHEMA_VERSION,
            bundle_id=f"bundle-cancelled-{request.request_id}",
            request=request,
            candidates=(),
            replay_audits=(),
            execution_status="cancelled",
            evidence_quality="insufficient_evidence",
            metrics={"status": "cancelled"},
            hashes={"candidates_sha256": hashlib.sha256(b"[]").hexdigest()},
        )

    def _assemble_fit_result(
        self,
        request: FitRequest,
        outcome: Any,
        is_synthetic: bool,
        observations: Sequence[FrameObservation],
    ) -> ResultBundle:
        """Construct and qualify the evaluated ResultBundle from fitter outcome."""
        self._checkpoints = list(outcome.checkpoints)
        self._execution_status = outcome.status

        cand = outcome.candidate
        if is_synthetic:
            cand_diag = dict(cand.diagnostics)
            cand_diag["is_synthetic"] = True
            cand = CandidateResult(
                schema_version=cand.schema_version,
                candidate_id=cand.candidate_id,
                request_id=cand.request_id,
                initial_state=cand.initial_state,
                trajectory=cand.trajectory,
                diagnostics=cand_diag,
                uncertainty_method="synthetic_residual",
                replay_audit=cand.replay_audit,
                is_accepted=False,
            )

        self._candidates = [cand]
        bundle = create_evaluated_result_bundle(
            bundle_id=f"bundle-{request.request_id}",
            request=request,
            candidates=(cand,),
            observations=tuple(observations),
            execution_status=outcome.status,
        )

        if is_synthetic:
            bundle_metrics = dict(bundle.metrics)
            bundle_metrics["is_synthetic"] = True
            eq = (
                "dynamic_candidate"
                if bundle.evidence_quality == "validated_profile"
                else bundle.evidence_quality
            )
            bundle = ResultBundle(
                schema_version=bundle.schema_version,
                bundle_id=bundle.bundle_id,
                request=bundle.request,
                candidates=bundle.candidates,
                replay_audits=bundle.replay_audits,
                execution_status=outcome.status,
                evidence_quality=eq,
                metrics=bundle_metrics,
                hashes=dict(bundle.hashes),
            )

        return bundle

    def fit(
        self,
        request: FitRequest,
        *,
        initial_controls: np.ndarray | Sequence[Sequence[float]] | None = None,
    ) -> ResultBundle:
        """Run forward model fitting with fail-closed capability gates and cancellation wiring."""
        (
            forward_model,
            renderer,
            camera,
            caps,
            is_synthetic,
        ) = self._validate_fit_capabilities(request)

        if self._is_cancelled:
            return self._build_cancelled_bundle(request)

        window_start_pts = int(request.time_window_start_pts)
        window_end_pts = int(request.time_window_end_pts)
        if window_end_pts < window_start_pts:
            raise ValueError(
                "Invalid fit request: time_window_end_pts "
                f"({window_end_pts}) is before time_window_start_pts ({window_start_pts})."
            )

        sorted_frame_ids = sorted(self._observations.keys())
        windowed_obs = [
            obs
            for obs in (self._observations[fid] for fid in sorted_frame_ids)
            if window_start_pts <= int(obs.pts_ticks) <= window_end_pts
        ]
        if not windowed_obs:
            raise ValueError(
                "Cannot fit: the requested PTS window "
                f"[{window_start_pts}, {window_end_pts}] contains no observations "
                f"(session holds {len(sorted_frame_ids)} frames)."
            )

        missing_time = [
            obs.frame_id for obs in windowed_obs if obs.physical_time_s is None
        ]
        if missing_time:
            raise ValueError(
                "Cannot fit: authoritative physical_time_s is unknown for frame(s) "
                f"{', '.join(missing_time)}; refusing to fabricate timestamps. "
                "Derive timing from authoritative PTS before fitting."
            )

        missing_masks = [
            obs.frame_id
            for obs in windowed_obs
            if not self._mask_provider.has_mask(obs.frame_id)
        ]
        if missing_masks:
            raise ValueError(
                "Cannot fit: incomplete observation/mask coverage — the following "
                f"frame(s) lack a manual mask: {', '.join(missing_masks)}."
            )

        obs_list = list(windowed_obs)
        mask_list = [self._mask_provider.get_mask(obs.frame_id) for obs in obs_list]
        time_points = [
            float(time_s)
            for obs in obs_list
            if (time_s := obs.physical_time_s) is not None
        ]
        init_dim = (
            42 if caps.state_convention == CANONICAL_ARTICULATED_CONVENTION else 6
        )
        init_st = tuple(0.0 for _ in range(init_dim))

        fitter_config = OptimizationConfig(
            budget_seconds=max(0.1, float(request.budget_seconds)),
            max_iterations=50,
        )
        fitter = ControlFitter(
            forward_model=forward_model,
            renderer=renderer,
            camera=camera,
            observed_masks=mask_list,
            time_points_s=time_points,
            initial_state=init_st,
            config=fitter_config,
            cancel_callback=lambda: self._is_cancelled,
        )

        self._execution_status = "running"
        outcome = fitter.fit(initial_controls=initial_controls)
        return self._assemble_fit_result(request, outcome, is_synthetic, obs_list)

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
