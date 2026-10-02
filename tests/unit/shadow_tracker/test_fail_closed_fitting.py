"""Unit and contract tests for fail-closed Shadow Fit capability and cancellation wiring (MMR-15-I, #11111).

Tests:
1. Unavailable runtime: fit fails-closed when backend runtime is unavailable.
2. Unqualified backend: fit refuses automated fitting when backend lacks qualification.
3. Engine capability check: fit rejects when requested capabilities are unsatisfied.
4. Synthetic backend orchestration: synthetic backends execute orchestration only,
   label results as synthetic, and cannot achieve validated_profile evidence quality.
5. Optimizer success without replay: low image residual cannot pass physical acceptance
   if fresh independent replay fails Gate G4.
6. Changed masks/camera after checkpoint: updating masks or camera invalidates active fits
   and prevents resuming from stale checkpoints.
7. Interrupted save: failure during bundle save leaves original target directory completely intact.
8. Recoverable cancellation: cancellation halts optimizer, records immutable checkpoints,
   and permits resuming when masks/camera are unchanged.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
import tempfile

import numpy as np
import pytest

from src.shared.python.shadow_tracker.artifacts import (
    ShadowTrackerBundle,
    load_bundle,
    save_bundle,
)
from src.shared.python.shadow_tracker._validation import (
    FIT_REQUEST_SCHEMA_VERSION,
    FRAME_OBSERVATION_SCHEMA_VERSION,
    FRAME_SCHEMA_VERSION,
    MASK_SCHEMA_VERSION,
    REPLAY_AUDIT_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
)
from src.shared.python.shadow_tracker.contracts import (
    CANONICAL_ARTICULATED_CONVENTION,
    POINT_LANDMARKS_CONVENTION,
    FitRequest,
    FrameObservation,
    ModelCapabilities,
    ReplayAudit,
    RenderRequest,
    RenderResult,
    RolloutRequest,
    RolloutResult,
)
from src.shared.python.shadow_tracker.fitting import (
    ControlFitter,
    OptimizationCheckpoint,
    OptimizationConfig,
)
from src.shared.python.shadow_tracker.mask_records import MaskFrame
from src.shared.python.shadow_tracker.projection import (
    AnalyticSilhouetteRenderer,
    PinholeCameraModel,
)
from src.shared.python.shadow_tracker.service import (
    DefaultShadowTrackerService,
    StaleCheckpointError,
    UnavailableBackendError,
)
from src.shared.python.shadow_tracker.source_records import (
    FrameIdentity,
    SourceAsset,
)

pytestmark = [pytest.mark.unit]

SAMPLE_SHA256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


# ---------------------------------------------------------------------------
# Test Doubles
# ---------------------------------------------------------------------------


class MockForwardModel:
    """Configurable mock forward model for capability and orchestration testing."""

    def __init__(
        self,
        *,
        is_available: bool = True,
        is_synthetic: bool = True,
        is_qualified: bool = False,
        fail_physics: bool = False,
        raise_on_rollout: bool = False,
        actuator_modes: tuple[str, ...] = ("torque_polynomial_deg6",),
        contact_modes: tuple[str, ...] = ("hunt_crossley_regularized_coulomb",),
    ) -> None:
        self._is_available = is_available
        self._is_synthetic = is_synthetic
        self._is_qualified = is_qualified
        self._fail_physics = fail_physics
        self._raise_on_rollout = raise_on_rollout
        self._actuator_modes = actuator_modes
        self._contact_modes = contact_modes
        self.rollout_count = 0
        self.last_time_points: tuple[float, ...] | None = None

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supported_bodies=tuple(f"b_{i}" for i in range(41)),
            state_convention=POINT_LANDMARKS_CONVENTION,
            actuator_modes=self._actuator_modes,
            contact_modes=self._contact_modes,
            is_available=self._is_available,
            is_synthetic=self._is_synthetic,
            is_qualified=self._is_qualified,
        )

    def rollout(self, request: RolloutRequest) -> RolloutResult:
        self.rollout_count += 1
        times = tuple(request.time_points_s)
        self.last_time_points = times
        if self._raise_on_rollout:
            raise RuntimeError("Simulated forward-model rollout failure")
        controls_arr = np.asarray(request.controls, dtype=np.float64)
        tau_mean = float(np.mean(controls_arr))

        traj: list[tuple[float, ...]] = []
        for t in times:
            pos_y = tau_mean * float(t)
            traj.append((0.0, pos_y, 1.0, 0.0, pos_y, 0.2))

        is_phys = not self._fail_physics
        audit = ReplayAudit(
            schema_version=REPLAY_AUDIT_SCHEMA_VERSION,
            candidate_id=f"cand_mock_{self.rollout_count}",
            reset_count=1,
            integrator_name="mock_integrator",
            integrator_version="1.0.0",
            coverage_start_s=float(times[0]) if times else 0.0,
            coverage_end_s=float(times[-1]) if times else 0.0,
            max_grip_translation_error_m=0.0005 if is_phys else 0.05,
            max_grip_rotation_error_rad=0.001 if is_phys else 0.5,
            is_physically_accepted=is_phys,
        )
        return RolloutResult(
            trajectory=tuple(traj),
            realized_controls=request.controls,
            time_points_s=request.time_points_s,
            audit=audit,
        )


def _make_source() -> SourceAsset:
    return SourceAsset(
        schema_version=SOURCE_SCHEMA_VERSION,
        asset_id="asset-11111",
        source_uri="https://example.com/test.mp4",
        content_sha256=SAMPLE_SHA256,
        width_px=64,
        height_px=64,
        rights_status="permitted",
        rights_note="open test footage",
    )


def _make_obs(frame_num: int) -> FrameObservation:
    return FrameObservation(
        schema_version=FRAME_OBSERVATION_SCHEMA_VERSION,
        shot_id="shot-11111",
        camera_id="cam-01",
        frame_id=f"frame-{frame_num:03d}",
        pts_ticks=frame_num * 1000,
        timebase_numerator=1,
        timebase_denominator=30000,
        physical_time_s=frame_num / 30.0,
        physical_time_reason="container_presentation_timestamp",
        body_mask_ref=f"mask-body-{frame_num}",
        club_mask_ref=f"mask-club-{frame_num}",
        valid_mask_ref=f"mask-valid-{frame_num}",
        confidence_provenance="manual_review",
        timing_mode="container_pts",
        is_timing_exact=True,
        clock_evidence="iso_bmff_pts",
        decoder_name="opencv",
        decoder_version="4.10.0",
        pixel_format="bgr24",
    )


def _make_mask(
    frame_num: int,
    *,
    revision_id: str | None = None,
    producer_id: str = "reviewer",
    correction_note: str = "test mask",
) -> MaskFrame:
    actual_rev_id = revision_id if revision_id is not None else f"rev-{frame_num}-0"
    ident = FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id="asset-11111",
        shot_id="shot-11111",
        swing_id="swing-01",
        camera_id="cam-01",
        frame_id=f"frame-{frame_num:03d}",
        pts_ticks=frame_num * 1000,
        timebase_numerator=1,
        timebase_denominator=30000,
        physical_time_s=frame_num / 30.0,
        physical_time_reason="container_presentation_timestamp",
        frame_sha256=SAMPLE_SHA256,
        timing_mode="container_pts",
        is_timing_exact=True,
        clock_evidence="iso_bmff_pts",
        decoder_name="opencv",
        decoder_version="4.10.0",
        pixel_format="bgr24",
    )
    # 64x64 mask: 4096 bytes
    valid = bytes([1] * 4096)
    body = bytes([1 if 1000 <= i < 2000 else 0 for i in range(4096)])
    club = bytes([1 if 2000 <= i < 2200 else 0 for i in range(4096)])
    return MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=ident,
        width_px=64,
        height_px=64,
        body=body,
        club=club,
        valid=valid,
        revision_id=actual_rev_id,
        parent_revision_id=None,
        producer_id=producer_id,
        correction_note=correction_note,
    )


def _make_camera(
    *, camera_id: str = "cam-01", fx: float = 100.0, fy: float = 100.0
) -> PinholeCameraModel:
    return PinholeCameraModel(
        camera_id=camera_id,
        width_px=64,
        height_px=64,
        fx=fx,
        fy=fy,
        cx=32.0,
        cy=32.0,
    )


def _make_request(*, req_capabilities: tuple[str, ...] = ()) -> FitRequest:
    return FitRequest(
        schema_version=FIT_REQUEST_SCHEMA_VERSION,
        request_id="req-11111",
        shot_id="shot-11111",
        model_hash=SAMPLE_SHA256,
        candidate_count=1,
        objective_profile="silhouette_iou",
        time_window_start_pts=0,
        time_window_end_pts=2000,
        budget_seconds=2.0,
        engine_capability_requirement=req_capabilities,
    )


def _make_renderer(cam: PinholeCameraModel) -> AnalyticSilhouetteRenderer:
    return AnalyticSilhouetteRenderer(cameras={cam.camera_id: cam})


# ---------------------------------------------------------------------------
# 1. Unavailable Runtime & Unqualified Backend Checks
# ---------------------------------------------------------------------------


def test_fit_rejects_unavailable_runtime() -> None:
    """When registered backend has is_available=False, fit must raise UnavailableBackendError."""
    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source(),
        observations=[_make_obs(0), _make_obs(1)],
        initial_masks=[_make_mask(0), _make_mask(1)],
    )
    backend = MockForwardModel(is_available=False)
    cam = _make_camera()
    service.register_backend(
        forward_model=backend,
        renderer=_make_renderer(cam),
        camera=cam,
    )

    req = _make_request()
    with pytest.raises(UnavailableBackendError) as exc_info:
        service.fit(req)
    assert "unavailable" in str(exc_info.value).lower()
    # Verify session inputs remained intact
    assert len(service.get_observations()) == 2
    assert service.execution_status == "idle"


def test_fit_rejects_unqualified_non_synthetic_backend() -> None:
    """When backend is neither synthetic nor qualified, fit must refuse automated fitting."""
    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source(),
        observations=[_make_obs(0), _make_obs(1)],
        initial_masks=[_make_mask(0), _make_mask(1)],
    )
    # Production backend that has not yet passed parent qualification
    backend = MockForwardModel(
        is_available=True, is_synthetic=False, is_qualified=False
    )
    cam = _make_camera()
    service.register_backend(
        forward_model=backend,
        renderer=_make_renderer(cam),
        camera=cam,
    )

    req = _make_request()
    with pytest.raises(UnavailableBackendError) as exc_info:
        service.fit(req)
    assert "unqualified" in str(exc_info.value).lower()


def test_fit_rejects_missing_engine_capability() -> None:
    """Fit must reject when requested engine capability is not supported by backend."""
    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source(),
        observations=[_make_obs(0), _make_obs(1)],
        initial_masks=[_make_mask(0), _make_mask(1)],
    )
    backend = MockForwardModel(
        is_available=True,
        is_synthetic=True,
        actuator_modes=("torque_polynomial_deg6",),
    )
    cam = _make_camera()
    service.register_backend(
        forward_model=backend,
        renderer=_make_renderer(cam),
        camera=cam,
    )

    req = _make_request(req_capabilities=("closed_chain_loop",))
    with pytest.raises(UnavailableBackendError) as exc_info:
        service.fit(req)
    assert "closed_chain_loop" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. Synthetic Backend Orchestration & Evidence Labeling
# ---------------------------------------------------------------------------


def test_synthetic_backend_labels_synthetic_and_blocks_validated_profile() -> None:
    """Synthetic test backend may test orchestration, but results must be labeled synthetic

    and cannot pass the real-data release profile (evidence_quality cannot be validated_profile).
    """
    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source(),
        observations=[_make_obs(0), _make_obs(1)],
        initial_masks=[_make_mask(0), _make_mask(1)],
    )
    backend = MockForwardModel(is_available=True, is_synthetic=True)
    cam = _make_camera()
    service.register_backend(
        forward_model=backend,
        renderer=_make_renderer(cam),
        camera=cam,
    )

    req = _make_request()
    bundle = service.fit(req)

    assert bundle.execution_status == "completed"
    assert bundle.evidence_quality != "validated_profile"
    assert bundle.metrics.get("is_synthetic") is True
    assert len(bundle.candidates) == 1
    # Candidate must not be accepted for real-data release profile
    assert bundle.candidates[0].is_accepted is False


def test_synthetic_masks_block_validated_profile_even_with_qualified_backend() -> None:
    """When session masks originate from synthetic fallback, fit cannot achieve validated_profile."""
    service = DefaultShadowTrackerService()
    # Masks explicitly tagged as synthetic
    synth_mask0 = _make_mask(
        0,
        producer_id="synthetic:sam-vit-b-golf:abcd1234",
        correction_note="Synthetic silhouette fallback fixture (unobserved model output)",
    )
    synth_mask1 = _make_mask(
        1,
        producer_id="synthetic:sam-vit-b-golf:abcd1234",
        correction_note="Synthetic silhouette fallback fixture (unobserved model output)",
    )
    service.initialize_session(
        source_asset=_make_source(),
        observations=[_make_obs(0), _make_obs(1)],
        initial_masks=[synth_mask0, synth_mask1],
    )
    # Qualified non-synthetic backend
    backend = MockForwardModel(is_available=True, is_synthetic=False, is_qualified=True)
    cam = _make_camera()
    service.register_backend(
        forward_model=backend,
        renderer=_make_renderer(cam),
        camera=cam,
    )

    req = _make_request()
    bundle = service.fit(req)

    assert bundle.execution_status == "completed"
    assert bundle.evidence_quality != "validated_profile"
    assert bundle.metrics.get("is_synthetic") is True
    assert bundle.candidates[0].diagnostics.get("is_synthetic") is True


# ---------------------------------------------------------------------------
# 3. Optimizer Success Without Replay (Gate G4)
# ---------------------------------------------------------------------------


def test_optimizer_success_without_replay_fails_physical_acceptance() -> None:
    """If independent fresh replay fails physical audit, candidate must not be accepted."""
    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source(),
        observations=[_make_obs(0), _make_obs(1)],
        initial_masks=[_make_mask(0), _make_mask(1)],
    )
    # Backend fails physics during replay
    backend = MockForwardModel(is_available=True, is_synthetic=True, fail_physics=True)
    cam = _make_camera()
    service.register_backend(
        forward_model=backend,
        renderer=_make_renderer(cam),
        camera=cam,
    )

    req = _make_request()
    bundle = service.fit(req)

    assert bundle.candidates[0].is_accepted is False
    assert bundle.evidence_quality in ("insufficient_evidence", "dynamic_candidate")


# ---------------------------------------------------------------------------
# 4. Stale-Mask and Stale-Camera Invalidation
# ---------------------------------------------------------------------------


def test_changed_masks_after_checkpoint_invalidates_and_rejects_resume() -> None:
    """Updating masks after taking checkpoints invalidates them and blocks resuming."""
    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source(),
        observations=[_make_obs(0), _make_obs(1)],
        initial_masks=[
            _make_mask(0, revision_id="rev-0-init"),
            _make_mask(1, revision_id="rev-1-init"),
        ],
    )
    backend = MockForwardModel(is_available=True, is_synthetic=True)
    cam = _make_camera()
    service.register_backend(
        forward_model=backend,
        renderer=_make_renderer(cam),
        camera=cam,
    )

    req = _make_request()
    service.fit(req)
    assert service.has_active_fits() is True
    assert len(service.checkpoints) > 0

    # Save a reference to a recorded checkpoint
    ckpt = service.checkpoints[-1]

    # Mask correction occurs
    service.update_mask(
        frame_id="frame-000",
        body=bytes([0] * 4096),
        club=bytes([0] * 4096),
        valid=bytes([1] * 4096),
        parent_revision_id="rev-0-init",
        producer_id="reviewer",
        correction_note="manual mask update",
    )

    # Fits and checkpoints must be invalidated
    assert service.has_active_fits() is False
    assert len(service.checkpoints) == 0

    # Attempting to resume from stale checkpoint must fail
    with pytest.raises(StaleCheckpointError) as exc_info:
        service.resume_from_checkpoint(ckpt, req)
    assert (
        "stale" in str(exc_info.value).lower() or "mask" in str(exc_info.value).lower()
    )


def test_changed_camera_after_checkpoint_invalidates_and_rejects_resume() -> None:
    """Updating camera after taking checkpoints invalidates them and blocks resuming."""
    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source(),
        observations=[_make_obs(0), _make_obs(1)],
        initial_masks=[_make_mask(0), _make_mask(1)],
    )
    backend = MockForwardModel(is_available=True, is_synthetic=True)
    cam = _make_camera()
    service.register_backend(
        forward_model=backend,
        renderer=_make_renderer(cam),
        camera=cam,
    )

    req = _make_request()
    service.fit(req)
    ckpt = service.checkpoints[-1]

    # Update camera with different calibration
    new_cam = _make_camera(camera_id="cam-01-modified", fx=120.0, fy=120.0)
    service.update_camera(new_cam)

    assert service.has_active_fits() is False
    with pytest.raises(StaleCheckpointError) as exc_info:
        service.resume_from_checkpoint(ckpt, req)
    assert "camera" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 5. Interrupted Save Resilience (Atomic Persistence)
# ---------------------------------------------------------------------------


def test_interrupted_bundle_save_preserves_original_target_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure during save_bundle must leave the original bundle untouched and uncorrupted."""
    bundle_path = tmp_path / "target_bundle"
    source = _make_source()
    obs = (_make_obs(0),)
    masks = (_make_mask(0),)

    bundle_v1 = ShadowTrackerBundle(
        bundle_id="bundle-11111",
        source_asset=source,
        observations=obs,
        masks=masks,
        uncertainty={"v": 1},
    )
    # Save initial bundle v1 (succeeds)
    save_bundle(bundle_v1, bundle_path)

    # Verify initial bundle loaded cleanly
    loaded_v1 = load_bundle(bundle_path)
    assert loaded_v1.uncertainty == {"v": 1}

    # Simulate an error/interrupt halfway during a second save
    bundle_v2 = ShadowTrackerBundle(
        bundle_id="bundle-11111",
        source_asset=source,
        observations=obs,
        masks=masks,
        uncertainty={"v": 2},
    )

    should_fail = False
    original_write = Path.write_bytes

    def faulty_write_bytes(self: Path, data: bytes) -> int:
        if should_fail and "masks.json" in str(self):
            raise OSError("Simulated disk full or process interruption")
        return original_write(self, data)

    monkeypatch.setattr(Path, "write_bytes", faulty_write_bytes)

    should_fail = True
    with pytest.raises(OSError, match="Simulated disk full"):
        save_bundle(bundle_v2, bundle_path)

    # Original bundle must still be valid and uncorrupted
    loaded_after_fail = load_bundle(bundle_path)
    assert loaded_after_fail.uncertainty == {"v": 1}


# ---------------------------------------------------------------------------
# 6. Recoverable Cancellation Wiring
# ---------------------------------------------------------------------------


def test_cancellation_during_fitting_stops_cleanly_and_allows_resume() -> None:
    """Cancelling during fit halts cleanly, records checkpoints, and permits resuming."""
    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source(),
        observations=[_make_obs(0), _make_obs(1)],
        initial_masks=[_make_mask(0), _make_mask(1)],
    )

    backend = MockForwardModel(is_available=True, is_synthetic=True)
    cam = _make_camera()
    service.register_backend(
        forward_model=backend,
        renderer=_make_renderer(cam),
        camera=cam,
    )

    # Trigger cancellation before/during fit
    service.cancel()
    assert service.is_cancelled is True
    assert service.execution_status == "cancelled"

    req = _make_request()
    cancelled_bundle = service.fit(req)

    assert cancelled_bundle.execution_status == "cancelled"
    assert service.execution_status == "cancelled"

    # Resuming clears cancelled flag without declaring completion
    service.resume()
    assert service.is_cancelled is False
    assert service.execution_status != "completed"

    # Now fit can proceed
    completed_bundle = service.fit(req)
    assert completed_bundle.execution_status == "completed"


# ---------------------------------------------------------------------------
# Review-Gate Regressions (#11121): PTS window, aligned coverage, unknown timing,
# status propagation, renderer calibration sync, calibration fingerprints, atomic publish
# ---------------------------------------------------------------------------


def _register_synthetic_backend(
    service: DefaultShadowTrackerService,
    *,
    camera: PinholeCameraModel | None = None,
) -> MockForwardModel:
    backend = MockForwardModel(is_available=True, is_synthetic=True)
    cam = camera or _make_camera()
    service.register_backend(
        forward_model=backend,
        renderer=_make_renderer(cam),
        camera=cam,
    )
    return backend


def test_fit_filters_observations_to_requested_pts_window() -> None:
    """Frames outside time_window_start_pts..time_window_end_pts must not reach the fitter."""
    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source(),
        observations=[_make_obs(0), _make_obs(1), _make_obs(2), _make_obs(3)],
        initial_masks=[_make_mask(0), _make_mask(1), _make_mask(2), _make_mask(3)],
    )
    backend = _register_synthetic_backend(service)

    req = _make_request()  # window 0..2000 pts covers frames 000..002 inclusive
    service.fit(req)

    assert backend.last_time_points is not None
    assert backend.last_time_points == pytest.approx(
        (0.0, 1 / 30.0, 2 / 30.0), rel=1e-9
    )


def test_fit_fails_closed_on_empty_pts_window() -> None:
    """A requested window containing no frames must fail closed, not silently fit everything."""
    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source(),
        observations=[_make_obs(0), _make_obs(1)],
        initial_masks=[_make_mask(0), _make_mask(1)],
    )
    _register_synthetic_backend(service)

    req = FitRequest(
        schema_version=FIT_REQUEST_SCHEMA_VERSION,
        request_id="req-11111-empty-window",
        shot_id="shot-11111",
        model_hash=SAMPLE_SHA256,
        candidate_count=1,
        objective_profile="silhouette_iou",
        time_window_start_pts=900_000,
        time_window_end_pts=1_000_000,
        budget_seconds=2.0,
        engine_capability_requirement=(),
    )
    with pytest.raises(ValueError, match="PTS window"):
        service.fit(req)


def test_fit_rejects_incomplete_mask_coverage() -> None:
    """Every frame inside the requested window must pair with a mask; gaps must fail closed."""
    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source(),
        observations=[_make_obs(0), _make_obs(1), _make_obs(2)],
        initial_masks=[_make_mask(0), _make_mask(2)],
    )
    _register_synthetic_backend(service)

    req = _make_request()
    with pytest.raises(ValueError, match="incomplete"):
        service.fit(req)


def test_fit_fails_closed_when_physical_time_unknown() -> None:
    """Unknown physical_time_s must fail closed instead of substituting 0.0."""
    service = DefaultShadowTrackerService()
    obs_unknown = FrameObservation.from_dict(
        {**_make_obs(1).to_dict(), "physical_time_s": None}
    )
    service.initialize_session(
        source_asset=_make_source(),
        observations=[_make_obs(0), obs_unknown],
        initial_masks=[_make_mask(0), _make_mask(1)],
    )
    _register_synthetic_backend(service)

    req = _make_request()
    with pytest.raises(ValueError, match="physical_time_s"):
        service.fit(req)


def test_fit_propagates_failed_execution_status_for_non_synthetic_backends() -> None:
    """A qualified non-synthetic backend that fails must report its status, not 'completed'."""
    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source(),
        observations=[_make_obs(0), _make_obs(1)],
        initial_masks=[_make_mask(0), _make_mask(1)],
    )
    backend = MockForwardModel(
        is_available=True,
        is_synthetic=False,
        is_qualified=True,
        raise_on_rollout=True,
    )
    cam = _make_camera()
    service.register_backend(
        forward_model=backend, renderer=_make_renderer(cam), camera=cam
    )

    req = _make_request()
    bundle = service.fit(req)
    assert bundle.execution_status == "failed"
    assert service.execution_status == "failed"


def test_update_camera_refreshes_registered_renderer_calibration() -> None:
    """update_camera() must reach the registered renderer's camera map, not only the fitter."""
    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source(),
        observations=[_make_obs(0), _make_obs(1)],
        initial_masks=[_make_mask(0), _make_mask(1)],
    )
    old_cam = _make_camera()
    renderer = AnalyticSilhouetteRenderer(
        cameras={old_cam.camera_id: old_cam}, body_radius_m=0.05
    )
    backend = MockForwardModel(is_available=True, is_synthetic=True)
    service.register_backend(forward_model=backend, renderer=renderer, camera=old_cam)

    render_request = RenderRequest(
        camera_id="cam-01", state=(0.0, 0.0, 1.0), image_size_px=(64, 64)
    )
    before = renderer.render(render_request).body_mask

    new_cam = _make_camera(camera_id="cam-01", fx=120.0, fy=120.0)
    service.update_camera(new_cam)

    refresh_fresh = AnalyticSilhouetteRenderer(
        cameras={"cam-01": new_cam}, body_radius_m=0.05
    )
    expected = refresh_fresh.render(render_request).body_mask

    assert renderer.render(render_request).body_mask == expected
    assert before != expected


def test_recalibrated_camera_same_id_rejects_checkpoint_resume() -> None:
    """Same camera_id with different calibration must stale-invalidate checkpoints."""
    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source(),
        observations=[_make_obs(0), _make_obs(1)],
        initial_masks=[_make_mask(0), _make_mask(1)],
    )
    _register_synthetic_backend(service)

    req = _make_request()
    service.fit(req)
    ckpt = service.checkpoints[-1]
    assert ckpt.camera_id == "cam-01"

    # Recalibrate the same camera id: intrinsics change, identity does not
    recalibrated = _make_camera(camera_id="cam-01", fx=130.0, fy=130.0)
    service.update_camera(recalibrated)

    with pytest.raises(StaleCheckpointError) as exc_info:
        service.resume_from_checkpoint(ckpt, req)
    assert (
        "calibration" in str(exc_info.value).lower()
        or "camera" in str(exc_info.value).lower()
    )


def test_save_bundle_replaces_target_with_no_staging_leftovers(
    tmp_path: Path,
) -> None:
    """Replacing a bundle must publish once and leave no staging directories behind."""
    bundle_path = tmp_path / "target_bundle"
    source = _make_source()
    obs = (_make_obs(0),)
    masks = (_make_mask(0),)
    save_bundle(
        ShadowTrackerBundle(
            bundle_id="bundle-11111",
            source_asset=source,
            observations=obs,
            masks=masks,
            uncertainty={"v": 1},
        ),
        bundle_path,
    )
    save_bundle(
        ShadowTrackerBundle(
            bundle_id="bundle-11111",
            source_asset=source,
            observations=obs,
            masks=masks,
            uncertainty={"v": 2},
        ),
        bundle_path,
    )

    loaded = load_bundle(bundle_path)
    assert loaded.uncertainty == {"v": 2}
    assert tuple(p.name for p in tmp_path.iterdir()) == ("target_bundle",)


def test_failed_atomic_publish_keeps_original_target_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted publish (after staging) must never lose or corrupt the live bundle."""
    from src.shared.python.shadow_tracker import artifacts as artifacts_module

    bundle_path = tmp_path / "target_bundle"
    source = _make_source()
    obs = (_make_obs(0),)
    masks = (_make_mask(0),)
    save_bundle(
        ShadowTrackerBundle(
            bundle_id="bundle-11111",
            source_asset=source,
            observations=obs,
            masks=masks,
            uncertainty={"v": 1},
        ),
        bundle_path,
    )

    # Force the publish step to fail exactly as an interrupted save would
    publish_count = 0

    def broken_publish(staged_dir: Path, target_dir: Path) -> bool:
        nonlocal publish_count
        publish_count += 1
        raise OSError("Simulated process kill during publish")

    monkeypatch.setattr(artifacts_module, "publish_directory", broken_publish)

    with pytest.raises(OSError, match="during publish"):
        save_bundle(
            ShadowTrackerBundle(
                bundle_id="bundle-11111",
                source_asset=source,
                observations=obs,
                masks=masks,
                uncertainty={"v": 2},
            ),
            bundle_path,
        )

    # The live bundle is the untouched original
    loaded = load_bundle(bundle_path)
    assert loaded.uncertainty == {"v": 1}
    assert publish_count >= 1
    # No staging leftovers pollute the parent
    assert tuple(p.name for p in tmp_path.iterdir()) == ("target_bundle",)
