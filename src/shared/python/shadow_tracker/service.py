"""Shadow Tracker orchestration and review service (ST-11, #10134).

Coordinates video observation inspection, manual mask correction with revision lineage,
deterministic downstream fit invalidation, worst-frame navigation, and bundle persistence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
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
from .source_records import FrameIdentity, RightsStatus, SourceAsset

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
    shot_id: str = ""


class DefaultShadowTrackerService:
    """Primary headless orchestration service for Shadow Tracker evidence review and persistence."""

    def __init__(self) -> None:
        self._source_asset: SourceAsset | None = None
        self._shot: Shot | None = None
        self._observations_order: list[FrameObservation] = []
        self._obs_by_scope: dict[tuple[str, str], FrameObservation] = {}
        self._obs_by_frame_id: dict[str, list[FrameObservation]] = {}
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
        self._observations_order = list(observations)
        self._obs_by_scope = {(obs.shot_id, obs.frame_id): obs for obs in observations}
        self._obs_by_frame_id = {}
        for obs in observations:
            self._obs_by_frame_id.setdefault(obs.frame_id, []).append(obs)
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

    def _validate_staged_video_import(
        self,
        *,
        asset: SourceAsset,
        new_observations: Sequence[FrameObservation],
        new_masks: Sequence[MaskFrame],
    ) -> None:
        """Fail closed on staged import conflicts before any session mutation.

        Rejects duplicated scope/revision entries inside the staged batch, an
        imported asset other than the session's (single) source asset, shot/frame
        scopes that already exist in the session, and revision ids that would
        collide with the live revision lineage.
        """
        staged_scopes = [(obs.shot_id, obs.frame_id) for obs in new_observations]
        if len(set(staged_scopes)) != len(staged_scopes):
            raise ValueError("Staged video import contains duplicate shot/frame scopes")
        staged_revision_ids = [mask.revision_id for mask in new_masks]
        if len(set(staged_revision_ids)) != len(staged_revision_ids):
            raise ValueError("Staged video import contains duplicate revision ids")

        if self._source_asset is None:
            # First import: initialize_session replaces the empty session wholesale.
            return

        if asset.asset_id != self._source_asset.asset_id:
            raise ValueError(
                f"Session already owns source asset {self._source_asset.asset_id!r}; "
                f"refusing to mix imported asset {asset.asset_id!r}. Start a fresh "
                "session for a different asset."
            )

        colliding_scopes = sorted(
            f"{shot_id}/{frame_id}"
            for shot_id, frame_id in staged_scopes
            if (shot_id, frame_id) in self._obs_by_scope
        )
        if colliding_scopes:
            raise ValueError(
                "Imported video collides with already-reviewed shot/frame scopes: "
                f"{colliding_scopes[:8]}"
            )

        existing_revision_ids = {
            mask.revision_id for mask in self._mask_provider.all_revisions()
        }
        colliding_revisions = sorted(set(staged_revision_ids) & existing_revision_ids)
        if colliding_revisions:
            raise ValueError(
                "Imported video would re-register revision ids already present in "
                f"the session: {colliding_revisions[:8]}"
            )

    def get_observations(self) -> tuple[FrameObservation, ...]:
        """Return all observations ordered by frame identity."""
        return tuple(self._observations_order)

    def get_observation(
        self, frame_id: str, *, shot_id: str | None = None
    ) -> FrameObservation:
        """Retrieve observation by frame_id, disambiguating by shot_id if provided."""
        check_id(frame_id, "frame_id")
        if shot_id is not None:
            check_id(shot_id, "shot_id")
            key = (shot_id, frame_id)
            if key not in self._obs_by_scope:
                raise KeyError(
                    f"No observation found for frame_id {frame_id!r} in shot {shot_id!r}"
                )
            return self._obs_by_scope[key]

        matches = self._obs_by_frame_id.get(frame_id, [])
        if not matches:
            raise KeyError(f"No observation found for frame_id {frame_id!r}")
        if len(matches) > 1:
            shots = [m.shot_id for m in matches]
            raise ValueError(
                f"Multiple observations ({len(matches)}) match frame_id {frame_id!r} "
                f"across shots {shots}. Specify shot_id to disambiguate."
            )
        return matches[0]

    def get_mask(self, frame_id: str, *, shot_id: str | None = None) -> MaskFrame:
        """Retrieve current latest mask for frame_id, optionally filtered by shot_id."""
        check_id(frame_id, "frame_id")
        if shot_id is None:
            matches = self._obs_by_frame_id.get(frame_id, [])
            if len(matches) == 1:
                shot_id = matches[0].shot_id
        return self._mask_provider.get_mask(frame_id, shot_id=shot_id)

    def get_mask_history(
        self, frame_id: str, *, shot_id: str | None = None
    ) -> tuple[MaskFrame, ...]:
        """Retrieve revision lineage history for frame_id."""
        check_id(frame_id, "frame_id")
        if shot_id is None:
            matches = self._obs_by_frame_id.get(frame_id, [])
            if len(matches) == 1:
                shot_id = matches[0].shot_id
        return self._mask_provider.get_revision_history(frame_id, shot_id=shot_id)

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
        shot_id: str | None = None,
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
        obs = self.get_observation(frame_id, shot_id=shot_id)

        # Build FrameIdentity for the revised mask
        prev_mask = self.get_mask(frame_id, shot_id=obs.shot_id)
        frame_ident = prev_mask.frame

        rev_seq = len(
            self._mask_provider.get_revision_history(frame_id, shot_id=obs.shot_id)
        )
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
        scores: list[tuple[str, str, float]] = []

        for obs in self._observations_order:
            frame_id = obs.frame_id
            shot_id = obs.shot_id
            if not self._mask_provider.has_mask(frame_id, shot_id=shot_id):
                scores.append((shot_id, frame_id, 0.0))
                continue

            mask = self._mask_provider.get_mask(frame_id, shot_id=shot_id)
            if metric == "mask_coverage":
                fg_count = mask.body.count(1) + mask.club.count(1)
                valid_count = mask.valid.count(1) or 1
                coverage = fg_count / valid_count
                scores.append((shot_id, frame_id, coverage))
            elif metric == "uncertainty":
                scores.append((shot_id, frame_id, 0.5))
            else:
                scores.append((shot_id, frame_id, 0.0))

        scores.sort(key=lambda item: item[2])

        reports: list[WorstFrameReport] = []
        for rank, (sid, fid, score) in enumerate(scores[:top_n], start=1):
            reports.append(
                WorstFrameReport(
                    frame_id=fid,
                    shot_id=sid,
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
            for fid in sorted({obs.frame_id for obs in self._observations_order})
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

        if not self._observations_order:
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

        sorted_frame_ids = sorted(self._obs_by_frame_id.keys())
        windowed_obs = [
            obs
            for obs in (
                frame_obs
                for fid in sorted_frame_ids
                for frame_obs in self._obs_by_frame_id[fid]
            )
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

    def import_video(
        self,
        video_path: Path | str,
        *,
        asset_id: str | None = None,
        shot_id: str = "shot-001",
        swing_id: str = "swing-001",
        camera_id: str = "camera-001",
        subject_id: str = "subject-001",
        timing_mapping: Any | None = None,
        cuts: Sequence[tuple[int, int]] = (),
        transforms: Sequence[str] = (),
        max_frames: int | None = None,
        cancel_token: Any | None = None,
        rights_status: RightsStatus = "unknown",
        rights_note: str = "",
    ) -> tuple[FrameObservation, ...]:
        """Import video file into review session, preserving timing and isolated shot namespaces."""
        from .ingestion import (
            AffineTimingMapping,
            DecodeLimits,
            OpenCvVideoDecoder,
            ShotDefinition,
            create_shot,
            decode_video_frames,
            filter_shot_frames,
            ingest_source_asset,
            map_frame_to_observation,
        )

        path = Path(video_path)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Invalid or empty media file: {path}")

        decoder = OpenCvVideoDecoder(path)
        if decoder.frame_count == 0:
            raise ValueError(f"Failed to decode frames from media: {path}")

        aid = asset_id if asset_id is not None else f"asset-{path.stem}"
        asset = ingest_source_asset(
            path,
            asset_id=aid,
            width_px=decoder.width,
            height_px=decoder.height,
            rights_status=rights_status,
            rights_note=rights_note,
        )

        phys_fn = None
        # Without an evidenced timing mapping, physical time stays unknown
        # (None). Container PTS authority remains in timing_mode/clock_evidence;
        # the ingestion adapter records its canonical unknown-time reason
        # instead of fabricating physical-time provenance.
        phys_reason = ""
        if timing_mapping is not None:

            def phys_fn(_idx: int, pres_time: Fraction) -> float:
                return float(timing_mapping.to_physical_time(pres_time))

            phys_reason = (
                "timing_mapping_affine"
                if isinstance(timing_mapping, AffineTimingMapping)
                else "timing_mapping_piecewise"
            )

        def _is_cancelling() -> bool:
            return self._is_cancelled or (
                cancel_token() if cancel_token is not None else False
            )

        limits = DecodeLimits(max_frames=max_frames, is_cancelled=_is_cancelling)
        raw_frames = list(
            decode_video_frames(
                decoder,
                asset=asset,
                shot_id=shot_id,
                swing_id=swing_id,
                camera_id=camera_id,
                physical_time_s_fn=phys_fn,
                physical_time_reason=phys_reason,
                limits=limits,
            )
        )

        was_cancelled = _is_cancelling()
        if was_cancelled:
            self._is_cancelled = True
            self._execution_status = "cancelled"

        if raw_frames:
            shot_def = ShotDefinition(
                shot_id=shot_id,
                start_pts=raw_frames[0].pts_ticks,
                end_pts=raw_frames[-1].pts_ticks,
                start_frame_id=raw_frames[0].frame_id,
                end_frame_id=raw_frames[-1].frame_id,
                subject_id=subject_id,
                swing_id=swing_id,
                camera_id=camera_id,
                cuts=tuple(cuts),
                transforms=tuple(transforms),
            )
            shot_rec = create_shot(asset, shot_def)
            filtered_frames = filter_shot_frames(shot_rec, raw_frames)
        else:
            shot_rec = None
            filtered_frames = []

        new_obs: list[FrameObservation] = []
        new_masks: list[MaskFrame] = []
        blank_valid = bytes([1] * (decoder.width * decoder.height))
        blank_fg = bytes([0] * (decoder.width * decoder.height))
        for f in filtered_frames:
            obs = map_frame_to_observation(
                f,
                body_mask_ref=f"mask-body-{f.frame_id}",
                club_mask_ref=f"mask-club-{f.frame_id}",
                valid_mask_ref=f"mask-valid-{f.frame_id}",
                confidence_provenance="manual_review",
            )
            new_obs.append(obs)
            m = MaskFrame(
                schema_version=MASK_SCHEMA_VERSION,
                frame=f,
                width_px=decoder.width,
                height_px=decoder.height,
                body=blank_fg,
                club=blank_fg,
                valid=blank_valid,
                revision_id=f"rev-init-{shot_id}-{f.frame_id}",
                parent_revision_id=None,
                producer_id="reviewer-import",
                correction_note="initial import mask",
            )
            new_masks.append(m)

        if self._source_asset is None:
            self._validate_staged_video_import(
                asset=asset,
                new_observations=new_obs,
                new_masks=new_masks,
            )
            self.initialize_session(
                source_asset=asset,
                observations=new_obs,
                initial_masks=new_masks,
                shot=shot_rec,
            )
        else:
            # Atomic repeat import: every scope/revision conflict is rejected
            # before the session state is touched.
            self._validate_staged_video_import(
                asset=asset,
                new_observations=new_obs,
                new_masks=new_masks,
            )
            for o in new_obs:
                self._observations_order.append(o)
                self._obs_by_scope[(o.shot_id, o.frame_id)] = o
                self._obs_by_frame_id.setdefault(o.frame_id, []).append(o)
            for m in new_masks:
                self._mask_provider.register_mask(m)

        if was_cancelled:
            self._is_cancelled = True
            self._execution_status = "cancelled"

        return tuple(new_obs)

    def save_bundle(self, path: Path | str) -> None:
        """Atomically persist current session to a review bundle."""
        if self._source_asset is None:
            raise ValueError("Cannot save bundle: no source asset in session")

        bundle = ShadowTrackerBundle(
            bundle_id=f"bundle-{self._source_asset.asset_id}",
            source_asset=self._source_asset,
            observations=tuple(self._observations_order),
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
            "observations": [obs.to_dict() for obs in self._observations_order],
            "masks": [m.to_dict() for m in self._mask_provider.all_revisions()],
            "uncertainty": self._uncertainty,
            "assumptions": list(self._assumptions),
        }

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(export_data, indent=2), encoding="utf-8")
        return export_data
