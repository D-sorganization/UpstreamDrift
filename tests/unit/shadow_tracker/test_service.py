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
