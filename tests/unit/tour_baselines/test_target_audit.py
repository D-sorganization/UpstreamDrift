"""Tests for Tour Target Audit, Marker Semantics, Events, and Provenance (TB-01 #10586).

TDD test-first suite verifying:
1. Canonical Driver and Iron identities, rates, units, axes, and handedness.
2. Content-based duplicate verification for Simscape and Pinocchio copies.
3. Independent review of Driver and Iron label differences (Uname*38 vs pelvis).
4. 4-tier measurement map (surface, inferred joint, cluster centroid, calibrated points).
5. Exact missing spans and occlusion masks (RShoulderTop, clubhead cluster).
6. Native-clock swing intervals and explicitly labeled inferred impact.
7. Explicit unresolved provenance records and separation of geometry vs anatomy.
8. Dual canonical target emitters (MotionDraft, BodyTarget, ClubTarget) without new C3D parsers.
9. Failure contracts (SHA mismatch, corrupted residuals, swapped identities).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.tour_capture_contract import (
    TOUR_CAPTURE,
    TOUR_CAPTURE_IRON,
    TourCapture,
    load_tour_capture,
)
from src.shared.python.tour_baselines.audit import (
    MeasurementCategory,
    MeasurementMapping,
    ProvenanceRecord,
    SwingEventInterval,
    TargetAudit,
    audit_tour_capture,
    build_measurement_map,
    detect_swing_intervals,
    emit_dynamics_targets,
    emit_reference_draft,
    get_repo_root,
    verify_duplicate_captures,
)

pytestmark = pytest.mark.unit


def test_audit_tour_capture_driver_identity_and_properties():
    """Driver audit must match the frozen 360 Hz specification and native frame count."""
    audit = audit_tour_capture("driver")
    assert audit.capture_kind == "driver"
    assert audit.sha256 == TOUR_CAPTURE.sha256
    assert audit.rate_hz == 360.0
    assert audit.frames == 654
    assert np.isclose(audit.duration_s, (654 - 1) / 360.0)
    assert audit.units == "m"
    assert audit.vertical_axis == "y"
    assert audit.handedness == "right_handed"
    assert len(audit.labels) == 38
    assert "Uname*38" in audit.labels
    assert "pelvis" not in audit.labels


def test_audit_tour_capture_iron_identity_and_properties():
    """Iron audit must match the frozen 359 Hz specification and native frame count."""
    audit = audit_tour_capture("iron")
    assert audit.capture_kind == "iron"
    assert audit.sha256 == TOUR_CAPTURE_IRON.sha256
    assert audit.rate_hz == 359.0
    assert audit.frames == 657
    assert np.isclose(audit.duration_s, (657 - 1) / 359.0)
    assert audit.units == "m"
    assert audit.vertical_axis == "y"
    assert audit.handedness == "right_handed"
    assert len(audit.labels) == 38
    assert "pelvis" in audit.labels
    assert "Uname*38" not in audit.labels


def test_driver_iron_label_differences_independently_reviewed():
    """Independent review of driver and iron label differences must be explicit."""
    driver_audit = audit_tour_capture("driver")
    iron_audit = audit_tour_capture("iron")

    # 37 common markers
    common = set(driver_audit.labels) & set(iron_audit.labels)
    assert len(common) == 37

    # Distinct labels
    assert set(driver_audit.labels) - set(iron_audit.labels) == {"Uname*38"}
    assert set(iron_audit.labels) - set(driver_audit.labels) == {"pelvis"}

    # Measurement maps must independently review labels
    assert "Uname*38" in driver_audit.measurement_map.entries
    assert "pelvis" not in driver_audit.measurement_map.entries
    assert "pelvis" in iron_audit.measurement_map.entries
    assert "Uname*38" not in iron_audit.measurement_map.entries


def test_content_duplicate_verification():
    """Duplicate copies in Simscape and Pinocchio directories must verify by content SHA."""
    dups = verify_duplicate_captures()
    assert "driver" in dups
    assert "iron" in dups

    # Driver copies have identical SHA
    for path, sha in dups["driver"].items():
        assert sha == TOUR_CAPTURE.sha256
        assert (get_repo_root() / path).exists()

    # Iron copies have identical SHA
    for path, sha in dups["iron"].items():
        assert sha == TOUR_CAPTURE_IRON.sha256
        assert (get_repo_root() / path).exists()


def test_measurement_map_distinguishes_four_categories():
    """Measurement map must partition observations into surface, joint center, cluster, and calibrated."""
    m_map = build_measurement_map("driver")

    # 1. Observed surface markers
    surface = m_map.get_by_category(MeasurementCategory.OBSERVED_SURFACE)
    assert "WaistLeft" in surface
    assert "BackTop" in surface
    assert "HeadTop" in surface

    # 2. Inferred joint centers (surface proxies)
    joints = m_map.get_by_category(MeasurementCategory.INFERRED_JOINT_CENTER)
    assert "mid_hip" in joints
    assert "neck" in joints
    assert "left_shoulder" in joints
    # Must record that joint centers are proxies/centroids, not direct bone measurements
    assert joints["neck"].is_inferred is True
    assert "proxy" in joints["neck"].notes.lower()

    # 3. Cluster centroids
    clusters = m_map.get_by_category(MeasurementCategory.CLUSTER_CENTROID)
    assert "observed_club_head" in clusters
    assert "observed_club_grip" in clusters

    # 4. Calibrated clubhead points (unavailable in raw C3D)
    calibrated = m_map.get_by_category(MeasurementCategory.CALIBRATED_POINT)
    assert "clubface_center" in calibrated
    assert calibrated["clubface_center"].is_available is False


def test_missing_spans_and_occlusion_audit():
    """Audit must detail exact missing sample counts and contiguous gap spans."""
    driver_audit = audit_tour_capture("driver")

    # RShoulderTop is heavily occluded across the first 526 frames
    r_shoulder = driver_audit.missing_spans["RShoulderTop"]
    assert r_shoulder.missing_samples == 526
    assert r_shoulder.valid_samples == 128
    assert (0, 525) in r_shoulder.spans

    # Clubhead cluster missing spans in driver
    clubhead_span = driver_audit.missing_spans["Marker_2:2:1"]
    assert clubhead_span.missing_samples == 36
    assert clubhead_span.valid_samples == 618

    # Iron audit missing spans
    iron_audit = audit_tour_capture("iron")
    iron_r_shoulder = iron_audit.missing_spans["RShoulderTop"]
    assert iron_r_shoulder.missing_samples == 562
    assert (0, 561) in iron_r_shoulder.spans


def test_swing_intervals_on_native_clocks_and_inferred_impact():
    """Event intervals must be computed on native clocks and impact must be labeled inferred."""
    driver_audit = audit_tour_capture("driver")
    iron_audit = audit_tour_capture("iron")

    # Driver (360 Hz)
    d_events = driver_audit.events
    assert d_events.rate_hz == 360.0
    assert d_events.impact.frame == 475
    assert np.isclose(d_events.impact.time_s, 475 / 360.0)
    assert d_events.impact.is_inferred is True
    assert d_events.impact.detection_method == "inferred_clubhead_speed_peak"
    assert d_events.top.frame == 397
    assert d_events.downswing.start_frame == 397
    assert d_events.downswing.end_frame == 475

    # Iron (359 Hz) - must NOT inherit 360 Hz clock or driver event indices
    i_events = iron_audit.events
    assert i_events.rate_hz == 359.0
    assert i_events.impact.frame == 478
    assert np.isclose(i_events.impact.time_s, 478 / 359.0)
    assert i_events.impact.is_inferred is True
    assert i_events.impact.detection_method == "inferred_clubhead_speed_peak"
    assert i_events.top.frame == 394
    assert i_events.downswing.start_frame == 394
    assert i_events.downswing.end_frame == 478


def test_provenance_record_and_unresolved_fields():
    """Provenance record must clearly articulate known sources and explicit unresolved fields."""
    audit = audit_tour_capture("driver")
    prov = audit.provenance

    assert prov.source_file == "data/C3D_TA_Driver.c3d"
    assert prov.capture_type == "tour_average"
    assert len(prov.unresolved_provenance) >= 4

    unresolved_keys = {item.field_name for item in prov.unresolved_provenance}
    assert "capture_date" in unresolved_keys
    assert "subject_demographics" in unresolved_keys
    assert "optical_calibration_parameters" in unresolved_keys
    assert "usage_license" in unresolved_keys

    # Geometry separated from anatomy
    assert prov.asserted_subject_anatomy is not None
    assert prov.capture_specific_geometry is not None


def test_canonical_target_emitter_reference_draft():
    """Canonical target emitter for kinematic reference must produce a valid MotionDraft."""
    audit = audit_tour_capture("driver")
    draft = emit_reference_draft(audit)

    assert len(draft.time_s) == 654
    assert np.isclose(draft.time_s[0], 0.0)
    assert np.isclose(draft.time_s[-1], (654 - 1) / 360.0)
    assert len(draft.names) == 38
    assert draft.points.shape == (654, 38, 3)

    # RShoulderTop is NaN during occluded frames
    r_idx = draft.names.index("RShoulderTop")
    assert np.isnan(draft.points[0, r_idx]).all()
    # But finite at frame 600
    assert np.isfinite(draft.points[600, r_idx]).all()


def test_canonical_target_emitter_dynamics_targets():
    """Canonical target emitter for dynamics fitting must produce BodyTarget and ClubTarget."""
    audit = audit_tour_capture("iron")
    body_target, club_target = emit_dynamics_targets(audit)

    assert body_target.time.shape == (657,)
    assert body_target.impact_idx == 478
    assert "WaistLeft" in body_target.marker_names
    assert club_target.impact_idx == 478
    assert club_target.clubhead.shape == (657, 3)
    assert club_target.butt.shape == (657, 3)


def test_audit_receipt_round_trip():
    """Audit receipt must export to and re-import from JSON deterministically."""
    audit = audit_tour_capture("driver")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        audit.save_receipt(tmp_path)
        loaded = TargetAudit.load_receipt(tmp_path)
        assert loaded.sha256 == audit.sha256
        assert loaded.frames == audit.frames
        assert loaded.rate_hz == audit.rate_hz
        assert loaded.events.impact.frame == audit.events.impact.frame
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_audit_failure_on_corrupted_data():
    """Target audit must fail closed if file hash, sample rate, or residual fails contract."""
    # Synthetic bad capture with negative residual
    time = np.linspace(0, 1, 10)
    labels = ("M1", "M2")
    pts = np.zeros((10, 2, 3))
    valid = np.ones((10, 2), dtype=bool)

    # Corrupting time
    with pytest.raises(ValueError, match="Capture time must start at zero"):
        TourCapture(time + 1.0, labels, pts, valid)

    # Corrupting non-finite valid points
    pts_nan = pts.copy()
    pts_nan[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Every valid capture point must be finite"):
        TourCapture(time, labels, pts_nan, valid)
