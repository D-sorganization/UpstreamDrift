"""Unit tests for modern and historical footage qualification workflows (ST-10, #10133)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.shadow_tracker.contracts import FrameObservation
from shared.python.shadow_tracker.footage_workflows import (
    DegradationConfig,
    DegradationHarness,
    DeduplicationSplitter,
    FilmLineage,
    HistoricalPilotCatalog,
    PilotEntry,
    RightsAuditReport,
    SuitabilityGrade,
    SyncValidationResult,
    audit_asset_rights,
    validate_multiview_synchronization,
    validate_split_isolation,
)
from shared.python.shadow_tracker.source_records import RightsStatus, SourceAsset

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


def _dummy_source_asset(
    *,
    asset_id: str = "asset-001",
    rights_status: RightsStatus = "unknown",
    rights_note: str = "",
) -> SourceAsset:
    return SourceAsset(
        schema_version="shadow-tracker/source/1.0.0",
        asset_id=asset_id,
        source_uri=f"urn:asset:{asset_id}",
        content_sha256="a" * 64,
        width_px=1920,
        height_px=1080,
        rights_status=rights_status,
        rights_note=rights_note,
    )


def _dummy_observation(
    *,
    camera_id: str = "cam-001",
    frame_id: str = "frame-001",
    pts_ticks: int = 100,
    physical_time_s: float | None = 0.01,
) -> FrameObservation:
    return FrameObservation(
        schema_version="shadow-tracker/frame-observation/1.0.0",
        shot_id="shot-001",
        camera_id=camera_id,
        frame_id=frame_id,
        pts_ticks=pts_ticks,
        timebase_numerator=1,
        timebase_denominator=10000,
        physical_time_s=physical_time_s,
        physical_time_reason="hardware_sync",
        body_mask_ref="mask-body-001",
        club_mask_ref="mask-club-001",
        valid_mask_ref="mask-valid-001",
        confidence_provenance="ground_truth",
        timing_mode="hardware_genlock",
        is_timing_exact=True,
        clock_evidence="hardware_sync",
        decoder_name="nvdec",
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_same_film_re_encoding_cannot_cross_split() -> None:
    """ST-10 First Test 1: Same-film re-encodings must not cross train/evaluation split."""
    lineages = (
        FilmLineage(
            film_id="film-1953-oakmont",
            canonical_title="1953 US Open Final Round",
            recording_year=1953,
            recording_source="broadcast_archive_kinescope",
            golfer_id="ben_hogan",
            known_asset_ids=("asset-hogan-1953-hd", "asset-hogan-1953-sd"),
        ),
        FilmLineage(
            film_id="film-1960-cherry-hills",
            canonical_title="1960 US Open Drive",
            recording_year=1960,
            recording_source="16mm_film",
            golfer_id="arnold_palmer",
            known_asset_ids=("asset-palmer-1960-telecine",),
        ),
        FilmLineage(
            film_id="film-1965-bellerive",
            canonical_title="1965 US Open Swing",
            recording_year=1965,
            recording_source="35mm_newsreel",
            golfer_id="gary_player",
            known_asset_ids=("asset-player-1965-raw", "asset-player-1965-restored"),
        ),
    )

    splitter = DeduplicationSplitter()
    train_split, holdout_split = splitter.split(
        lineages, holdout_fraction=0.34, seed=42
    )

    # Invariant: Each film's assets must be fully contained in one split
    train_assets = {
        asset for lineage in train_split for asset in lineage.known_asset_ids
    }
    holdout_assets = {
        asset for lineage in holdout_split for asset in lineage.known_asset_ids
    }

    # Strict disjointness between splits
    assert train_assets.isdisjoint(holdout_assets)

    # Oakmont re-encodings must be kept together
    assert (
        "asset-hogan-1953-hd" in train_assets and "asset-hogan-1953-sd" in train_assets
    ) or (
        "asset-hogan-1953-hd" in holdout_assets
        and "asset-hogan-1953-sd" in holdout_assets
    )

    # Adversarial test: manually leaking re-encodings across splits must raise ValueError
    leaked_train = (lineages[0],)
    leaked_holdout = (
        FilmLineage(
            film_id="film-1953-oakmont",  # Duplicate film ID across split boundary
            canonical_title="1953 US Open Duplicate",
            recording_year=1953,
            recording_source="broadcast_archive_kinescope",
            golfer_id="ben_hogan",
            known_asset_ids=("asset-hogan-1953-sd",),
        ),
    )
    with pytest.raises(
        ValueError,
        match="Lineage leakage detected: film 'film-1953-oakmont' spans multiple splits",
    ):
        validate_split_isolation(leaked_train, leaked_holdout)


def test_source_rights_and_identity_unknown_remains_flagged() -> None:
    """ST-10 First Test 2: Source rights and identity unknown remains explicitly flagged."""
    # Case 1: Rights unknown
    unknown_asset = _dummy_source_asset(
        asset_id="asset-archive-001", rights_status="unknown"
    )
    report_unknown = audit_asset_rights(unknown_asset, attribution_confirmed=True)
    assert report_unknown.is_cleared_for_release is False
    assert "rights_unreviewed_or_unknown" in report_unknown.flags

    # Case 2: Rights restricted
    restricted_asset = _dummy_source_asset(
        asset_id="asset-archive-002",
        rights_status="restricted",
        rights_note="Internal research only",
    )
    report_restricted = audit_asset_rights(restricted_asset, attribution_confirmed=True)
    assert report_restricted.is_cleared_for_release is False
    assert "rights_restricted" in report_restricted.flags

    # Case 3: Identity / attribution unconfirmed
    permitted_asset = _dummy_source_asset(
        asset_id="asset-archive-003",
        rights_status="permitted",
        rights_note="Public domain USGA archive",
    )
    report_unconfirmed = audit_asset_rights(
        permitted_asset, attribution_confirmed=False
    )
    assert report_unconfirmed.is_cleared_for_release is False
    assert "attribution_unconfirmed" in report_unconfirmed.flags

    # Case 4: Fully cleared
    report_cleared = audit_asset_rights(permitted_asset, attribution_confirmed=True)
    assert report_cleared.is_cleared_for_release is True
    assert len(report_cleared.flags) == 0


def test_impossible_synchronization_cannot_enable_multiview() -> None:
    """ST-10 First Test 3: Impossible synchronization cannot enable multi-view fusion."""
    # Case A: Non-overlapping time intervals
    obs_cam1 = (
        _dummy_observation(camera_id="cam-01", frame_id="f1", physical_time_s=0.0),
        _dummy_observation(camera_id="cam-01", frame_id="f2", physical_time_s=0.1),
    )
    obs_cam2_disjoint = (
        _dummy_observation(camera_id="cam-02", frame_id="f3", physical_time_s=10.0),
        _dummy_observation(camera_id="cam-02", frame_id="f4", physical_time_s=10.1),
    )

    result_disjoint = validate_multiview_synchronization(
        obs_cam1,
        obs_cam2_disjoint,
        max_allowable_drift_s=0.01,
    )
    assert result_disjoint.is_synchronous is False
    assert result_disjoint.refusal_reason == "non_overlapping_temporal_windows"

    # Case B: Significant clock drift (e.g. 25 ms drift > 10 ms threshold)
    obs_cam2_drifting = (
        _dummy_observation(camera_id="cam-02", frame_id="f1", physical_time_s=0.025),
        _dummy_observation(camera_id="cam-02", frame_id="f2", physical_time_s=0.125),
    )
    result_drift = validate_multiview_synchronization(
        obs_cam1,
        obs_cam2_drifting,
        max_allowable_drift_s=0.01,
    )
    assert result_drift.is_synchronous is False
    assert result_drift.refusal_reason == "excessive_clock_drift"
    assert result_drift.max_drift_s == pytest.approx(0.025)

    # Case C: Calibrated synchronous multi-view (< 2 ms drift, overlapping)
    obs_cam2_synced = (
        _dummy_observation(camera_id="cam-02", frame_id="f1", physical_time_s=0.001),
        _dummy_observation(camera_id="cam-02", frame_id="f2", physical_time_s=0.101),
    )
    result_synced = validate_multiview_synchronization(
        obs_cam1,
        obs_cam2_synced,
        max_allowable_drift_s=0.01,
    )
    assert result_synced.is_synchronous is True
    assert result_synced.refusal_reason is None
    assert result_synced.overlap_duration_s == pytest.approx(0.1, abs=0.01)


def test_archive_degradation_harness_transforms_masks() -> None:
    """ST-10 Acceptance: Calibrated degradation harness for downsampling and blur."""
    config = DegradationConfig(
        scale_factor=0.5,
        blur_sigma_px=1.0,
        drop_frame_cadence=2,
        noise_level=0.05,
    )
    harness = DegradationHarness(config)

    # High-resolution synthetic 100x100 mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 1

    degraded_mask = harness.degrade_mask(mask)
    assert degraded_mask.shape == (50, 50)  # Downscaled by 0.5
    assert degraded_mask.dtype == np.uint8

    # Frame dropping cadence
    frame_indices = tuple(range(10))
    retained = harness.apply_telecine_cadence(frame_indices)
    assert retained == (0, 2, 4, 6, 8)


def test_historical_pilot_catalog_yield_and_resumability() -> None:
    """ST-10 Acceptance: Resumable archive catalog, yield measurement, and failed cases."""
    catalog = HistoricalPilotCatalog()

    # Register 4 pilot entries: 1 qualified, 1 kinematic, 1 qualitative, 1 unusable
    e1 = PilotEntry(
        clip_id="clip-hogan-oakmont",
        asset_id="asset-001",
        film_id="film-1953",
        golfer_name="Ben Hogan",
        rights_status="permitted",
        suitability="qualified",
        annotation_minutes=45.0,
        compute_seconds=120.0,
        failure_reasons=(),
        physical_time_known=True,
    )
    e2 = PilotEntry(
        clip_id="clip-palmer-cherry",
        asset_id="asset-002",
        film_id="film-1960",
        golfer_name="Arnold Palmer",
        rights_status="permitted",
        suitability="kinematic_candidate",
        annotation_minutes=30.0,
        compute_seconds=95.0,
        failure_reasons=("uncalibrated_moving_camera",),
        physical_time_known=True,
    )
    e3 = PilotEntry(
        clip_id="clip-jones-st-andrews",
        asset_id="asset-003",
        film_id="film-1930",
        golfer_name="Bobby Jones",
        rights_status="permitted",
        suitability="qualitative_only",
        annotation_minutes=20.0,
        compute_seconds=40.0,
        failure_reasons=("unknown_film_speed_blocks_si_kinetics",),
        physical_time_known=False,
    )
    e4 = PilotEntry(
        clip_id="clip-corrupted-telecine",
        asset_id="asset-004",
        film_id="film-1925",
        golfer_name="Unknown",
        rights_status="unknown",
        suitability="unusable",
        annotation_minutes=15.0,
        compute_seconds=10.0,
        failure_reasons=(
            "severe_damage",
            "attribution_unconfirmed",
            "rights_unreviewed",
        ),
        physical_time_known=False,
    )

    catalog.register_entry(e1)
    catalog.register_entry(e2)
    catalog.register_entry(e3)
    catalog.register_entry(e4)

    # Acceptance yield: 1 qualified out of 4 total = 0.25 (25%)
    metrics = catalog.compute_pilot_yield()
    assert metrics["total_clips"] == 4
    assert metrics["qualified_yield"] == pytest.approx(0.25)
    assert metrics["kinematic_or_better_yield"] == pytest.approx(0.50)
    assert metrics["total_annotation_minutes"] == pytest.approx(110.0)
    assert metrics["total_compute_seconds"] == pytest.approx(265.0)

    # Failed-case inventory
    failures = catalog.get_failed_cases_inventory()
    assert len(failures) == 3
    assert "clip-corrupted-telecine" in [f.clip_id for f in failures]

    # Resumability: Idempotent lookup by clip_id
    assert catalog.has_entry("clip-hogan-oakmont") is True
    assert catalog.get_entry("clip-hogan-oakmont") == e1

    # Round trip serialization
    dumped = catalog.to_dict()
    reloaded = HistoricalPilotCatalog.from_dict(dumped)
    assert reloaded.compute_pilot_yield() == metrics
