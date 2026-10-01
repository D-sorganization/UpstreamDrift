"""Unit tests for Shadow Tracker Review Service & Bundle Persistence (ST-11, #10134).

Tests verify:
- Headless bundle round trip (save -> reopen -> exact observation/mask/timing identity).
- Save/reload preserves uncertainty, assumptions, clock evidence, and mask lineage.
- Manual mask correction deterministically invalidates existing fits and hypotheses.
- Cancel/resume cannot become complete.
- Missing/unsupported automated backend raises an actionable error.
- Worst-frame navigation ranks frames correctly for reviewer triage.
- Export canonical package includes typed schema, timing authority, and provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.shared.python.shadow_tracker._validation import (
    FRAME_OBSERVATION_SCHEMA_VERSION,
    FRAME_SCHEMA_VERSION,
    MASK_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
)
from src.shared.python.shadow_tracker.artifacts import (
    BUNDLE_SCHEMA_VERSION,
    ShadowTrackerBundle,
    load_bundle,
    save_bundle,
)
from src.shared.python.shadow_tracker.contracts import (
    FIT_REQUEST_SCHEMA_VERSION,
    FitRequest,
    FrameObservation,
)
from src.shared.python.shadow_tracker.mask_records import MaskFrame
from src.shared.python.shadow_tracker.service import (
    DefaultShadowTrackerService,
    UnavailableBackendError,
    WorstFrameReport,
)
from src.shared.python.shadow_tracker.source_records import (
    FrameIdentity,
    SourceAsset,
)

pytestmark = pytest.mark.unit

SAMPLE_SHA256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


def _make_source_asset() -> SourceAsset:
    return SourceAsset(
        schema_version=SOURCE_SCHEMA_VERSION,
        asset_id="asset-10134",
        source_uri="https://example.com/test_video.mp4",
        content_sha256=SAMPLE_SHA256,
        width_px=320,
        height_px=240,
        rights_status="permitted",
        rights_note="open research test clip",
    )


def _make_observation(frame_num: int) -> FrameObservation:
    return FrameObservation(
        schema_version=FRAME_OBSERVATION_SCHEMA_VERSION,
        shot_id="shot-10134",
        camera_id="cam-front",
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


def _make_mask_frame(
    frame_num: int,
    *,
    revision_id: str | None = None,
    parent_revision_id: str | None = None,
    correction_note: str = "initial reviewed mask",
) -> MaskFrame:
    actual_rev_id = revision_id if revision_id is not None else f"rev-{frame_num}-0"
    frame_ident = FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id="asset-10134",
        shot_id="shot-10134",
        swing_id="swing-1",
        camera_id="cam-front",
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
    # 4x4 image
    valid = bytes([1] * 16)
    body = bytes([1 if i < 4 else 0 for i in range(16)])
    club = bytes([1 if 4 <= i < 6 else 0 for i in range(16)])
    return MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=frame_ident,
        width_px=4,
        height_px=4,
        body=body,
        club=club,
        valid=valid,
        revision_id=actual_rev_id,
        parent_revision_id=parent_revision_id,
        producer_id="reviewer-local",
        correction_note=correction_note,
    )


# ---------------------------------------------------------------------------
# 1. Bundle Persistence and Headless Round-Trip
# ---------------------------------------------------------------------------


def test_bundle_save_and_load_round_trip(tmp_path: Path) -> None:
    """Save and reload must preserve observations, masks, uncertainty, and timing authority exactly."""
    bundle_dir = tmp_path / "review_bundle"
    source = _make_source_asset()
    obs_list = tuple(_make_observation(i) for i in range(3))
    masks = tuple(_make_mask_frame(i) for i in range(3))

    uncertainty_dict = {
        "camera_focal_uncertainty_pct": 0.5,
        "timing_uncertainty_s": 0.001,
    }
    assumptions = ("rigid_club_grip", "monocular_static_camera")

    bundle = ShadowTrackerBundle(
        bundle_id="bundle-test-1",
        schema_version=BUNDLE_SCHEMA_VERSION,
        source_asset=source,
        observations=obs_list,
        masks=masks,
        uncertainty=uncertainty_dict,
        assumptions=assumptions,
        evidence_quality="unreviewed",
    )

    save_bundle(bundle, bundle_dir)

    # Verify atomic files exist
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "observations.json").exists()
    assert (bundle_dir / "masks.json").exists()

    loaded = load_bundle(bundle_dir)

    assert loaded.bundle_id == "bundle-test-1"
    assert loaded.schema_version == BUNDLE_SCHEMA_VERSION
    assert loaded.source_asset.asset_id == source.asset_id
    assert loaded.observations[0].timing_mode == "container_pts"
    assert loaded.observations[0].is_timing_exact is True
    assert len(loaded.observations) == 3
    for orig, roundtrip in zip(obs_list, loaded.observations, strict=True):
        assert orig.frame_id == roundtrip.frame_id
        assert orig.pts_ticks == roundtrip.pts_ticks
        assert orig.timing_mode == roundtrip.timing_mode
        assert orig.clock_evidence == roundtrip.clock_evidence
        assert orig.physical_time_s == roundtrip.physical_time_s

    assert len(loaded.masks) == 3
    for orig_m, roundtrip_m in zip(masks, loaded.masks, strict=True):
        assert orig_m.revision_id == roundtrip_m.revision_id
        assert orig_m.body == roundtrip_m.body
        assert orig_m.club == roundtrip_m.club
        assert orig_m.observation_hash == roundtrip_m.observation_hash

    assert loaded.uncertainty == uncertainty_dict
    assert loaded.assumptions == assumptions


def test_bundle_tamper_detection(tmp_path: Path) -> None:
    """Tampering with bundle contents must fail checksum verification."""
    bundle_dir = tmp_path / "tampered_bundle"
    source = _make_source_asset()
    obs_list = (_make_observation(0),)
    masks = (_make_mask_frame(0),)

    bundle = ShadowTrackerBundle(
        bundle_id="bundle-tamper-1",
        schema_version=BUNDLE_SCHEMA_VERSION,
        source_asset=source,
        observations=obs_list,
        masks=masks,
        uncertainty={},
        assumptions=(),
        evidence_quality="unreviewed",
    )
    save_bundle(bundle, bundle_dir)

    # Tamper with observations.json
    obs_file = bundle_dir / "observations.json"
    data = json.loads(obs_file.read_text(encoding="utf-8"))
    data[0]["physical_time_s"] = 999.0
    obs_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Checksum mismatch|tampered|corrupt"):
        load_bundle(bundle_dir)


# ---------------------------------------------------------------------------
# 2. Service Mask Updates Invalidate Existing Fits
# ---------------------------------------------------------------------------


def test_mask_correction_invalidates_fits() -> None:
    """Correcting a mask revision must invalidate any cached or completed fits."""
    service = DefaultShadowTrackerService()
    source = _make_source_asset()
    obs_list = tuple(_make_observation(i) for i in range(2))
    masks = tuple(_make_mask_frame(i) for i in range(2))

    service.initialize_session(
        source_asset=source,
        observations=obs_list,
        initial_masks=masks,
    )

    # Inject a prior fit record into the service state
    service._inject_fit_for_testing("cand-001")
    assert service.has_active_fits() is True

    # Now make a manual mask correction on frame 0
    updated_mask = service.update_mask(
        frame_id="frame-000",
        body=bytes([1] * 8 + [0] * 8),
        club=bytes([0] * 8 + [1] * 8),
        valid=bytes([1] * 16),
        parent_revision_id="rev-0-0",
        producer_id="reviewer-local",
        correction_note="corrected shoulder and shaft boundary",
    )

    assert updated_mask.revision_id != "rev-0-0"
    assert updated_mask.parent_revision_id == "rev-0-0"

    # Crucial acceptance criterion: correction invalidates fits
    assert service.has_active_fits() is False
    assert len(service.get_candidates()) == 0


def test_save_reload_preserves_mask_lineage(tmp_path: Path) -> None:
    """Multiple mask revisions must maintain complete parentage and history across reload."""
    service = DefaultShadowTrackerService()
    source = _make_source_asset()
    obs = (_make_observation(0),)
    mask0 = _make_mask_frame(0, revision_id="rev-0")

    service.initialize_session(
        source_asset=source,
        observations=obs,
        initial_masks=(mask0,),
        uncertainty={"sigma": 0.05},
        assumptions=("planar_motion",),
    )

    # Revise mask
    service.update_mask(
        frame_id="frame-000",
        body=bytes([0] * 16),
        club=bytes([0] * 16),
        valid=bytes([1] * 16),
        parent_revision_id="rev-0",
        producer_id="reviewer-local",
        correction_note="cleared false foreground detection",
    )

    history = service.get_mask_history("frame-000")
    assert len(history) == 2
    rev1_id = history[1].revision_id

    # Save to disk
    bundle_path = tmp_path / "lineage_bundle"
    service.save_bundle(bundle_path)

    # Reload into fresh service
    new_service = DefaultShadowTrackerService()
    new_service.load_bundle(bundle_path)

    reloaded_history = new_service.get_mask_history("frame-000")
    assert len(reloaded_history) == 2
    assert reloaded_history[0].revision_id == "rev-0"
    assert reloaded_history[1].revision_id == rev1_id
    assert reloaded_history[1].parent_revision_id == "rev-0"
    assert reloaded_history[1].correction_note == "cleared false foreground detection"
    assert new_service.get_uncertainty() == {"sigma": 0.05}
    assert new_service.get_assumptions() == ("planar_motion",)


# ---------------------------------------------------------------------------
# 3. Cancellation and Resume Semantics
# ---------------------------------------------------------------------------


def test_cancel_resume_cannot_become_complete() -> None:
    """Cancelled operations cannot magically transition to complete status upon resume."""
    service = DefaultShadowTrackerService()
    source = _make_source_asset()
    service.initialize_session(
        source_asset=source,
        observations=(_make_observation(0),),
        initial_masks=(_make_mask_frame(0),),
    )

    service.cancel()
    assert service.is_cancelled is True
    assert service.execution_status == "cancelled"

    service.resume()
    # Resume on a cancelled task must not declare completion
    assert service.execution_status != "completed"
    assert service.is_cancelled is False


# ---------------------------------------------------------------------------
# 4. Actionable Error for Unavailable Automated Fitting Backend
# ---------------------------------------------------------------------------


def test_missing_fitting_backend_gives_actionable_error() -> None:
    """Automated fitting must report unavailable with explicit guidance until ST-07..10 pass."""
    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source_asset(),
        observations=(_make_observation(0),),
        initial_masks=(_make_mask_frame(0),),
    )

    request = FitRequest(
        schema_version=FIT_REQUEST_SCHEMA_VERSION,
        request_id="req-001",
        shot_id="shot-10134",
        model_hash=SAMPLE_SHA256,
        candidate_count=1,
        objective_profile="silhouette_iou",
        time_window_start_pts=0,
        time_window_end_pts=1000,
        budget_seconds=5.0,
        engine_capability_requirement=("rigid_body",),
    )

    with pytest.raises(UnavailableBackendError) as exc_info:
        service.fit(request)

    error_msg = str(exc_info.value)
    assert "unavailable" in error_msg.lower()
    assert "ST-07" in error_msg or "forward" in error_msg.lower()


# ---------------------------------------------------------------------------
# 5. Worst-Frame Navigation
# ---------------------------------------------------------------------------


def test_worst_frame_navigation_ranking() -> None:
    """Worst-frame navigation ranks frames by mask quality or coverage for rapid reviewer triage."""
    service = DefaultShadowTrackerService()
    obs_list = tuple(_make_observation(i) for i in range(4))

    # Frame 0: good coverage (4 body pixels)
    m0 = _make_mask_frame(0)
    # Frame 1: empty body (0 body pixels -> worst)
    m1 = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=_make_mask_frame(1).frame,
        width_px=4,
        height_px=4,
        body=bytes([0] * 16),
        club=bytes([0] * 16),
        valid=bytes([1] * 16),
        revision_id="rev-0-1",
        parent_revision_id=None,
        producer_id="reviewer-local",
        correction_note="empty frame",
    )
    # Frame 2: partial coverage (2 body pixels)
    m2 = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=_make_mask_frame(2).frame,
        width_px=4,
        height_px=4,
        body=bytes([1, 1] + [0] * 14),
        club=bytes([0] * 16),
        valid=bytes([1] * 16),
        revision_id="rev-0-2",
        parent_revision_id=None,
        producer_id="reviewer-local",
        correction_note="partial frame",
    )

    service.initialize_session(
        source_asset=_make_source_asset(),
        observations=obs_list[:3],
        initial_masks=(m0, m1, m2),
    )

    ranked = service.worst_frames(metric="mask_coverage")
    assert len(ranked) == 3
    # Frame 1 has 0 foreground pixels, so it should rank first (worst)
    assert ranked[0].frame_id == "frame-001"
    assert ranked[1].frame_id == "frame-002"
    assert ranked[2].frame_id == "frame-000"


# ---------------------------------------------------------------------------
# 6. Canonical Export
# ---------------------------------------------------------------------------


def test_export_canonical_package(tmp_path: Path) -> None:
    """Canonical export must write typed provenance, timing authority, and mask records."""
    service = DefaultShadowTrackerService()
    source = _make_source_asset()
    obs = (_make_observation(0),)
    mask = (_make_mask_frame(0),)

    service.initialize_session(
        source_asset=source,
        observations=obs,
        initial_masks=mask,
    )

    export_path = tmp_path / "canonical_export.json"
    exported = service.export_canonical(export_path)

    assert export_path.exists()
    assert exported["schema_version"] == "shadow-tracker-export/1.0.0"
    assert exported["source_asset"]["asset_id"] == "asset-10134"
    assert exported["observations"][0]["timing_mode"] == "container_pts"
    assert len(exported["observations"]) == 1
    assert len(exported["masks"]) == 1
    assert "exported_at" in exported


# ---------------------------------------------------------------------------
# 7. MMR-12 Review Workflow & Ingestion Journey Tests
# ---------------------------------------------------------------------------


def test_multi_shot_no_frame_id_collision_and_invalidation() -> None:
    """Same frame IDs across distinct shots must not collide, and corrections invalidate fits (MMR-12)."""
    service = DefaultShadowTrackerService()
    source = _make_source_asset()

    # Create observations for shot A and shot B sharing identical local frame_id
    obs_a0 = _make_observation(0)
    obs_a1 = _make_observation(1)

    obs_b0 = FrameObservation(
        schema_version=FRAME_OBSERVATION_SCHEMA_VERSION,
        shot_id="shot-B",
        camera_id="cam-front",
        frame_id="frame-000",
        pts_ticks=0,
        timebase_numerator=1,
        timebase_denominator=30000,
        physical_time_s=0.0,
        physical_time_reason="container_presentation_timestamp",
        body_mask_ref="mask-body-b0",
        club_mask_ref="mask-club-b0",
        valid_mask_ref="mask-valid-b0",
        confidence_provenance="manual_review",
        timing_mode="container_pts",
        is_timing_exact=True,
        clock_evidence="iso_bmff_pts",
        decoder_name="opencv",
        decoder_version="4.10.0",
        pixel_format="bgr24",
    )
    obs_b1 = FrameObservation(
        schema_version=FRAME_OBSERVATION_SCHEMA_VERSION,
        shot_id="shot-B",
        camera_id="cam-front",
        frame_id="frame-001",
        pts_ticks=1000,
        timebase_numerator=1,
        timebase_denominator=30000,
        physical_time_s=1.0 / 30.0,
        physical_time_reason="container_presentation_timestamp",
        body_mask_ref="mask-body-b1",
        club_mask_ref="mask-club-b1",
        valid_mask_ref="mask-valid-b1",
        confidence_provenance="manual_review",
        timing_mode="container_pts",
        is_timing_exact=True,
        clock_evidence="iso_bmff_pts",
        decoder_name="opencv",
        decoder_version="4.10.0",
        pixel_format="bgr24",
    )

    mask_a0 = _make_mask_frame(0)
    mask_a1 = _make_mask_frame(1)

    frame_ident_b0 = FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id="asset-10134",
        shot_id="shot-B",
        swing_id="swing-002",
        camera_id="cam-front",
        frame_id="frame-000",
        pts_ticks=0,
        timebase_numerator=1,
        timebase_denominator=30000,
        physical_time_s=0.0,
        physical_time_reason="container_presentation_timestamp",
        frame_sha256=SAMPLE_SHA256,
        timing_mode="container_pts",
        is_timing_exact=True,
        clock_evidence="iso_bmff_pts",
        decoder_name="opencv",
        decoder_version="4.10.0",
        pixel_format="bgr24",
    )
    mask_b0 = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=frame_ident_b0,
        width_px=4,
        height_px=4,
        body=bytes([0] * 16),
        club=bytes([0] * 16),
        valid=bytes([1] * 16),
        revision_id="rev-b0-0",
        parent_revision_id=None,
        producer_id="reviewer-local",
        correction_note="shot B initial mask",
    )

    frame_ident_b1 = FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id="asset-10134",
        shot_id="shot-B",
        swing_id="swing-002",
        camera_id="cam-front",
        frame_id="frame-001",
        pts_ticks=1000,
        timebase_numerator=1,
        timebase_denominator=30000,
        physical_time_s=1.0 / 30.0,
        physical_time_reason="container_presentation_timestamp",
        frame_sha256=SAMPLE_SHA256,
        timing_mode="container_pts",
        is_timing_exact=True,
        clock_evidence="iso_bmff_pts",
        decoder_name="opencv",
        decoder_version="4.10.0",
        pixel_format="bgr24",
    )
    mask_b1 = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=frame_ident_b1,
        width_px=4,
        height_px=4,
        body=bytes([0] * 16),
        club=bytes([0] * 16),
        valid=bytes([1] * 16),
        revision_id="rev-b1-0",
        parent_revision_id=None,
        producer_id="reviewer-local",
        correction_note="shot B initial mask",
    )

    service.initialize_session(
        source_asset=source,
        observations=(obs_a0, obs_a1, obs_b0, obs_b1),
        initial_masks=(mask_a0, mask_a1, mask_b0, mask_b1),
    )

    # 4 distinct observations must exist without colliding
    assert len(service.get_observations()) == 4

    # Scoped retrieval disambiguates same frame_id across shots
    retrieved_a = service.get_observation("frame-000", shot_id="shot-10134")
    retrieved_b = service.get_observation("frame-000", shot_id="shot-B")
    assert retrieved_a.shot_id == "shot-10134"
    assert retrieved_b.shot_id == "shot-B"

    # Ambiguous retrieval without shot_id must raise ValueError
    with pytest.raises(ValueError, match="Multiple observations .* match frame_id"):
        service.get_observation("frame-000")

    # Scoped mask retrieval
    m_a = service.get_mask("frame-000", shot_id="shot-10134")
    m_b = service.get_mask("frame-000", shot_id="shot-B")
    assert m_a.revision_id == "rev-0-0"
    assert m_b.revision_id == "rev-b0-0"

    # Inject fit and verify that updating mask for shot A invalidates fits
    service._inject_fit_for_testing("fit-candidate-001")
    assert service.has_active_fits() is True

    service.update_mask(
        frame_id="frame-000",
        shot_id="shot-10134",
        body=bytes([1] * 16),
        club=bytes([0] * 16),
        valid=bytes([1] * 16),
        parent_revision_id="rev-0-0",
        producer_id="reviewer-human",
        correction_note="corrected shot A frame 0",
    )

    # Downstream fits must be invalidated
    assert service.has_active_fits() is False

    # Mask for shot B must remain unchanged
    assert service.get_mask("frame-000", shot_id="shot-B").revision_id == "rev-b0-0"


def test_import_video_real_vfr_cuts_slow_motion_timing(tmp_path: Path) -> None:
    """Importing real VFR video preserves timing across cuts and slow motion (MMR-12)."""
    from fractions import Fraction
    from src.shared.python.shadow_tracker.ingestion import AffineTimingMapping
    from tests.unit.shadow_tracker.test_video_ingestion import make_vfr_mp4_fixture

    vfr_clip = tmp_path / "review_vfr.mp4"
    deltas = [100, 300, 200, 400]
    make_vfr_mp4_fixture(vfr_clip, deltas, timescale=1000)

    service = DefaultShadowTrackerService()

    # 2x slow-motion mapping (0.5x speed) with offset
    timing_mapping = AffineTimingMapping(
        scale=Fraction(1, 2),
        offset_seconds=0.1,
    )

    # Cut frame index 2 (pts = 400)
    # PTS sequence: frame 0 -> 0, frame 1 -> 100, frame 2 -> 400, frame 3 -> 600
    cuts = ((350, 450),)

    imported_obs = service.import_video(
        vfr_clip,
        asset_id="asset-vfr-review",
        shot_id="shot-review-01",
        swing_id="swing-review-01",
        camera_id="cam-main",
        timing_mapping=timing_mapping,
        cuts=cuts,
        transforms=("rotate_0", "playback_speed_0.5"),
    )

    # Frame 2 (pts 400) was in cuts, so 3 frames remain
    assert len(imported_obs) == 3
    assert len(service.get_observations()) == 3

    # Check frame 0: pts=0 -> physical_time = 0.5 * 0 + 0.1 = 0.1s
    f0 = imported_obs[0]
    assert f0.pts_ticks == 0
    assert f0.is_timing_exact is True
    assert f0.timing_mode == "container_pts"
    assert pytest.approx(f0.physical_time_s, rel=1e-5) == 0.1

    # Check frame 1: pts=100 -> presentation_time = 100/1000 = 0.1s -> physical_time = 0.5 * 0.1 + 0.1 = 0.15s
    f1 = imported_obs[1]
    assert f1.pts_ticks == 100
    assert pytest.approx(f1.physical_time_s, rel=1e-5) == 0.15

    # Check frame 3 (imported as 3rd): pts=600 -> presentation_time = 600/1000 = 0.6s -> physical_time = 0.5 * 0.6 + 0.1 = 0.4s
    f2 = imported_obs[2]
    assert f2.pts_ticks == 600
    assert pytest.approx(f2.physical_time_s, rel=1e-5) == 0.4


def test_import_video_cancellation_during_decode_bounded_memory(tmp_path: Path) -> None:
    """Cancellation during decode stops promptly and keeps memory bounded (MMR-12)."""
    from tests.unit.shadow_tracker.test_video_ingestion import make_vfr_mp4_fixture

    clip = tmp_path / "long_clip.mp4"
    deltas = [100] * 5
    make_vfr_mp4_fixture(clip, deltas, timescale=1000)

    service = DefaultShadowTrackerService()

    cancel_after_n = 2
    decoded_count = 0

    def cancel_token() -> bool:
        nonlocal decoded_count
        decoded_count += 1
        return decoded_count > cancel_after_n

    imported = service.import_video(
        clip,
        asset_id="asset-cancelled",
        shot_id="shot-cancelled",
        cancel_token=cancel_token,
    )

    # Must have stopped promptly
    assert len(imported) <= cancel_after_n + 1
    assert service.is_cancelled is True


def test_import_corrupt_media_leaves_session_recoverable(tmp_path: Path) -> None:
    """Corrupt media file import raises error and preserves pre-existing session (MMR-12)."""
    service = DefaultShadowTrackerService()
    source = _make_source_asset()
    obs = (_make_observation(0),)
    mask = (_make_mask_frame(0),)

    service.initialize_session(
        source_asset=source,
        observations=obs,
        initial_masks=mask,
    )

    corrupt_clip = tmp_path / "corrupt.mp4"
    corrupt_clip.write_bytes(b"NOT_A_VALID_VIDEO_PAYLOAD")

    with pytest.raises((ValueError, RuntimeError, OSError)):
        service.import_video(
            corrupt_clip,
            asset_id="asset-corrupt",
            shot_id="shot-corrupt",
        )

    # Pre-existing session must remain completely intact and recoverable
    assert len(service.get_observations()) == 1
    assert service.get_observation("frame-000").shot_id == "shot-10134"
    assert service.get_mask("frame-000").revision_id == "rev-0-0"


# ---------------------------------------------------------------------------
# 8. Review-Fix Regressions: Atomic Repeat Imports & Unknown-Time Provenance
# ---------------------------------------------------------------------------


def test_repeat_import_same_scope_rejected_before_mutation(tmp_path: Path) -> None:
    """Re-importing a video that reuses an already-reviewed shot must fail atomically.

    Codex P1: the previous implementation appended observations before mask
    registration, so a duplicate revision id left the session mixed with the
    old asset and masks after the GUI reported a failure.
    """
    from tests.unit.shadow_tracker.test_video_ingestion import make_vfr_mp4_fixture

    clip = tmp_path / "short_clip.mp4"
    deltas = [100] * 3
    make_vfr_mp4_fixture(clip, deltas, timescale=1000)

    service = DefaultShadowTrackerService()
    first_import = service.import_video(clip, shot_id="shot-reviewed")
    assert len(first_import) == 3
    baseline_observations = service.get_observations()
    baseline_masks = service._mask_provider.all_revisions()

    with pytest.raises(ValueError):
        service.import_video(
            clip,
            asset_id="asset-short_clip",  # same asset as the first import
            shot_id="shot-reviewed",  # re-import reuses the already-reviewed scope
        )

    # Session must remain exactly as before the failed import.
    assert service.get_observations() == baseline_observations
    assert service._mask_provider.all_revisions() == baseline_masks
    assert len(service.get_observations()) == 3
    assert (
        service.get_observation("frame-000000", shot_id="shot-reviewed").frame_id
        == "frame-000000"
    )


def test_repeat_import_new_shot_appends_without_collision(tmp_path: Path) -> None:
    """A second shot imported from the same asset appends scope-isolated observations."""
    from tests.unit.shadow_tracker.test_video_ingestion import make_vfr_mp4_fixture

    clip = tmp_path / "same_clip.mp4"
    deltas = [100] * 2
    make_vfr_mp4_fixture(clip, deltas, timescale=1000)

    service = DefaultShadowTrackerService()
    service.import_video(clip, asset_id="asset-same_clip", shot_id="shot-A")

    imported = service.import_video(
        clip,
        asset_id="asset-same_clip",
        shot_id="shot-B",
        swing_id="swing-002",
    )
    assert len(imported) == 2
    assert len(service.get_observations()) == 4
    # Same local frame ids may recur across shots without masking each other.
    assert service.get_observation("frame-000000", shot_id="shot-B").shot_id == "shot-B"
    scoped_mask = service.get_mask("frame-000000", shot_id="shot-B")
    assert scoped_mask.revision_id == "rev-init-shot-B-frame-000000"
    # Original shot untouched.
    assert (
        service.get_mask("frame-000000", shot_id="shot-A").revision_id
        == "rev-init-shot-A-frame-000000"
    )


def test_import_video_other_asset_rejected_before_mutation(tmp_path: Path) -> None:
    """Importing a different source asset into an existing session must be refused."""
    from tests.unit.shadow_tracker.test_video_ingestion import make_vfr_mp4_fixture

    clip = tmp_path / "other_clip.mp4"
    deltas = [100] * 2
    make_vfr_mp4_fixture(clip, deltas, timescale=1000)

    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source_asset(),
        observations=(_make_observation(0),),
        initial_masks=(_make_mask_frame(0),),
    )
    baseline = service.get_observations()

    with pytest.raises(ValueError, match="refusing to mix imported asset"):
        service.import_video(clip, shot_id="shot-C")

    assert service.get_observations() == baseline


def test_import_video_unknown_physical_time_provenance_preserved(
    tmp_path: Path,
) -> None:
    """Without a timing mapping, physical time stays unknown; PTS authority stays in
    timing_mode/clock_evidence and the reason is the canonical unknown-time reason.

    Codex P1: the previous import claimed physical_time_reason='container_pts'
    although physical_time_s was None, fabricating physical-time provenance.
    """
    from tests.unit.shadow_tracker.test_video_ingestion import make_vfr_mp4_fixture

    clip = tmp_path / "unknown_time_clip.mp4"
    deltas = [100, 300, 200, 400]
    make_vfr_mp4_fixture(clip, deltas, timescale=1000)

    service = DefaultShadowTrackerService()
    imported = service.import_video(clip, shot_id="shot-unknown-time")

    assert len(imported) == 4
    first = imported[0]
    assert first.physical_time_s is None
    assert first.physical_time_reason == (
        "unknown physical time without evidenced clock mapping"
    )
    # Container PTS authority is preserved in the clock evidence fields.
    assert first.timing_mode == "container_pts"
    assert first.is_timing_exact is True
    assert first.pts_ticks == 0

    # Persisted evidence must keep the unknown-time provenance, not a fake reason.
    exported = service.export_canonical(tmp_path / "export.json")
    payload = json.loads(json.dumps(exported["observations"][0]))
    assert payload["physical_time_s"] is None
    assert payload["physical_time_reason"].startswith("unknown physical time")


# ---------------------------------------------------------------------------
# 9. MMR-12 Acceptance Criteria: VFR Journey, Multi-Shot Isolation, Invalidation
# ---------------------------------------------------------------------------


def test_vfr_import_manual_correction_save_reopen_and_invalidation(
    tmp_path: Path,
) -> None:
    """Exercise real VFR decode -> manual body/club correction -> save -> reopen;
    timing and revision lineage preserved, and correction invalidates saved candidate (MMR-12).
    """
    from fractions import Fraction
    from tests.unit.shadow_tracker.test_video_ingestion import make_vfr_mp4_fixture
    from shared.python.shadow_tracker.ingestion import AffineTimingMapping

    vfr_clip = tmp_path / "vfr_journey.mp4"
    deltas = [100, 300, 200, 400]
    make_vfr_mp4_fixture(vfr_clip, deltas, timescale=1000)

    service = DefaultShadowTrackerService()
    timing_mapping = AffineTimingMapping(
        scale=Fraction(1, 2),
        offset_seconds=0.05,
    )
    # Cut frame index 2 (pts=400)
    cuts = ((350, 450),)
    transforms = ("rotate_0", "crop_0", "playback_speed_0.5")

    imported = service.import_video(
        vfr_clip,
        asset_id="asset-vfr-journey",
        shot_id="shot-01",
        swing_id="swing-01",
        camera_id="cam-01",
        timing_mapping=timing_mapping,
        cuts=cuts,
        transforms=transforms,
    )
    assert len(imported) == 3
    assert len(service.get_observations()) == 3

    # Check timing preservation
    f0 = imported[0]
    assert f0.pts_ticks == 0
    assert f0.is_timing_exact is True
    assert f0.timing_mode == "container_pts"
    assert pytest.approx(f0.physical_time_s, rel=1e-5) == 0.05

    # Inject fit candidate to test downstream invalidation
    service._inject_fit_for_testing("candidate-active-001")
    assert service.has_active_fits() is True
    assert len(service.get_candidates()) == 1

    # Perform manual body and club correction on frame 0
    w, h = 64, 64
    body_mask = bytes([1] * (w * h))
    club_mask = bytes([1] * (w * h))
    valid_mask = bytes([1] * (w * h))
    initial_mask = service.get_mask("frame-000000", shot_id="shot-01")

    revised_mask = service.update_mask(
        frame_id="frame-000000",
        shot_id="shot-01",
        body=body_mask,
        club=club_mask,
        valid=valid_mask,
        parent_revision_id=initial_mask.revision_id,
        producer_id="expert-reviewer",
        correction_note="manual body and club correction",
    )
    assert revised_mask.parent_revision_id == initial_mask.revision_id
    assert revised_mask.body == body_mask
    assert revised_mask.club == club_mask

    # DbC Postcondition: candidate MUST be invalidated immediately
    assert service.has_active_fits() is False
    assert service.get_candidates() == ()
    assert service.checkpoints == ()

    # Persist session bundle to disk
    bundle_dir = tmp_path / "vfr_saved_bundle"
    service.save_bundle(bundle_dir)

    # Reopen in a clean service instance
    reopened_service = DefaultShadowTrackerService()
    reopened_service.load_bundle(bundle_dir)

    # Validate reopened session contracts
    assert len(reopened_service.get_observations()) == 3
    reopened_f0 = reopened_service.get_observation("frame-000000", shot_id="shot-01")
    assert reopened_f0.pts_ticks == 0
    assert reopened_f0.timing_mode == "container_pts"
    assert pytest.approx(reopened_f0.physical_time_s, rel=1e-5) == 0.05

    # Mask and lineage preserved in reopened session
    reopened_mask = reopened_service.get_mask("frame-000000", shot_id="shot-01")
    assert reopened_mask.revision_id == revised_mask.revision_id
    assert reopened_mask.body == body_mask
    assert reopened_mask.club == club_mask

    history = reopened_service.get_mask_history("frame-000000", shot_id="shot-01")
    assert len(history) == 2
    assert history[0].revision_id == initial_mask.revision_id
    assert history[1].revision_id == revised_mask.revision_id

    # Downstream candidate remains empty (no stale results reopened)
    assert reopened_service.has_active_fits() is False


def test_multi_shot_duplicate_local_frame_ids_no_collision_and_invalidation(
    tmp_path: Path,
) -> None:
    """Duplicate local frame IDs across shots must not collide in observations, masks,
    revisions, or triage navigation, and a correction in one shot invalidates fits without
    touching other shots (MMR-12)."""
    from tests.unit.shadow_tracker.test_video_ingestion import make_vfr_mp4_fixture

    clip = tmp_path / "multi_shot.mp4"
    deltas = [100, 200]
    make_vfr_mp4_fixture(clip, deltas, timescale=1000)

    service = DefaultShadowTrackerService()
    # Shot 1
    service.import_video(
        clip,
        asset_id="asset-shared",
        shot_id="shot-alpha",
        swing_id="swing-01",
    )
    # Shot 2: produces identical local frame IDs (frame-000000, frame-000001)
    service.import_video(
        clip,
        asset_id="asset-shared",
        shot_id="shot-beta",
        swing_id="swing-02",
    )

    assert len(service.get_observations()) == 4

    # Inject fit candidate
    service._inject_fit_for_testing("cand-multi-shot")
    assert service.has_active_fits() is True

    # Initial masks for frame-000000 in both shots are separate
    mask_alpha_init = service.get_mask("frame-000000", shot_id="shot-alpha")
    mask_beta_init = service.get_mask("frame-000000", shot_id="shot-beta")
    assert mask_alpha_init.frame.shot_id == "shot-alpha"
    assert mask_beta_init.frame.shot_id == "shot-beta"
    assert mask_alpha_init.revision_id != mask_beta_init.revision_id

    # Correct mask on shot-beta frame-000000 with parent_revision_id=None
    w, h = 64, 64
    body_fix = bytes([1] * (w * h))
    club_fix = bytes([1] * (w * h))
    valid_fix = bytes([1] * (w * h))

    mask_beta_rev = service.update_mask(
        frame_id="frame-000000",
        shot_id="shot-beta",
        body=body_fix,
        club=club_fix,
        valid=valid_fix,
        parent_revision_id=None,
        producer_id="reviewer",
        correction_note="corrected shot beta",
    )

    # Candidate invalidated
    assert service.has_active_fits() is False

    # Also correct mask on shot-alpha frame-000000 with identical mask data and parent_revision_id=None
    # Revisions MUST NOT collide!
    mask_alpha_rev = service.update_mask(
        frame_id="frame-000000",
        shot_id="shot-alpha",
        body=body_fix,
        club=club_fix,
        valid=valid_fix,
        parent_revision_id=None,
        producer_id="reviewer",
        correction_note="corrected shot alpha",
    )

    assert mask_alpha_rev.revision_id != mask_beta_rev.revision_id
    assert mask_alpha_rev.frame.shot_id == "shot-alpha"
    assert mask_beta_rev.frame.shot_id == "shot-beta"

    # Worst-frame triage navigation includes both shots and respects shot_id
    worst = service.worst_frames(metric="mask_coverage", top_n=10)
    assert len(worst) == 4
    shot_ids_in_worst = {w.shot_id for w in worst}
    assert shot_ids_in_worst == {"shot-alpha", "shot-beta"}

    # Save and reload bundle with multi-shot duplicate frame IDs
    bundle_path = tmp_path / "multi_shot_bundle"
    service.save_bundle(bundle_path)

    reloaded = DefaultShadowTrackerService()
    reloaded.load_bundle(bundle_path)

    assert len(reloaded.get_observations()) == 4
    # Reopened masks resolve without collision
    assert (
        reloaded.get_mask("frame-000000", shot_id="shot-alpha").revision_id
        == mask_alpha_rev.revision_id
    )
    assert (
        reloaded.get_mask("frame-000000", shot_id="shot-beta").revision_id
        == mask_beta_rev.revision_id
    )


def test_corrupt_bundle_and_media_preserves_session_recoverability(
    tmp_path: Path,
) -> None:
    """Corrupt media imports or corrupt bundle loads raise clear errors and leave
    the active review session fully intact and recoverable (MMR-12)."""
    service = DefaultShadowTrackerService()
    service.initialize_session(
        source_asset=_make_source_asset(),
        observations=(_make_observation(0),),
        initial_masks=(_make_mask_frame(0),),
    )
    baseline_obs = service.get_observations()
    baseline_mask = service.get_mask("frame-000")

    # 1. Corrupt empty media file
    empty_file = tmp_path / "corrupt_empty.mp4"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError):
        service.import_video(empty_file, shot_id="shot-empty")
    assert service.get_observations() == baseline_obs
    assert service.get_mask("frame-000") == baseline_mask

    # 2. Corrupt bundle with tampered manifest checksum
    good_bundle_dir = tmp_path / "good_bundle"
    service.save_bundle(good_bundle_dir)

    # Tamper with masks.json
    (good_bundle_dir / "masks.json").write_bytes(b"[]")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        service.load_bundle(good_bundle_dir)

    # Session must remain fully intact
    assert service.get_observations() == baseline_obs
    assert service.get_mask("frame-000") == baseline_mask

    # Session can still proceed normally (e.g. export or save)
    export_out = service.export_canonical(tmp_path / "recovered_export.json")
    assert len(export_out["observations"]) == 1


def test_multi_shot_duplicate_frame_ids_fit_mask_coverage(tmp_path: Path) -> None:
    """Multi-shot sessions with identical local frame IDs must scope mask lookups by shot_id
    during fit requests rather than raising ambiguous match errors (MMR-12)."""
    from tests.unit.shadow_tracker.test_video_ingestion import make_vfr_mp4_fixture

    clip = tmp_path / "clip.mp4"
    deltas = [100, 200]
    make_vfr_mp4_fixture(clip, deltas, timescale=1000)

    service = DefaultShadowTrackerService()
    service.import_video(
        clip,
        asset_id="asset-scoped",
        shot_id="shot-1",
        swing_id="swing-1",
    )
    service.import_video(
        clip,
        asset_id="asset-scoped",
        shot_id="shot-2",
        swing_id="swing-2",
    )

    # When multiple shots exist with the same frame_id ('frame-000000'),
    # fit() must properly scope mask retrieval to (obs.shot_id, obs.frame_id).
    # Since backend is not qualified, calling fit() must reach _validate_fit_capabilities
    # or handle the request without crashing on mask ambiguity.
    req = FitRequest(
        schema_version="shadow-tracker/fit-request/1.0.0",
        request_id="req-multi-shot",
        shot_id="shot-1",
        model_hash="0" * 64,
        candidate_count=1,
        objective_profile="silhouette_iou",
        time_window_start_pts=0,
        time_window_end_pts=500,
        budget_seconds=1.0,
        engine_capability_requirement=(),
    )
    # Fit fails with UnavailableBackendError (honest refusal), NOT ValueError about multiple masks!
    with pytest.raises(UnavailableBackendError):
        service.fit(req)
