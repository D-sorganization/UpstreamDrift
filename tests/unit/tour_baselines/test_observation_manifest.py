"""Tests for Frozen Observation Manifests and Independent Error Metrics (MMR-02-I #11105).

TDD test-first suite verifying:
1. Pooled Euclidean RMSE is strictly distinct from median frame RMS across unequal observation counts.
2. Zero valid observations and non-finite valid coordinates fail closed.
3. Unit enforcement requires SI metres and rejects unscaled/uncalibrated coordinates.
4. Calibration and holdout frame splits are strictly disjoint and cover valid swing intervals.
5. Calibration routines reject holdout frames (holdout protection invariant).
6. Authoritative frozen observation manifests for Driver and 7-Iron reproduce exact frames, rate, and SHA.
7. Serialization round-trips to dict and JSON deterministically.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from src.shared.python.tour_baselines.calibration import calibrate_fixed_geometry
from src.shared.python.tour_baselines.events import (
    SWING_EVENTS_DRIVER,
    SWING_EVENTS_IRON,
)
from src.shared.python.tour_baselines.observation_manifest import (
    MarkerObservationSpec,
    ObservationManifest,
    build_frozen_observation_manifest,
    calibrate_with_manifest_protection,
    compute_frame_wise_rms,
    compute_pooled_rmse,
)

pytestmark = pytest.mark.unit


def _small_manifest() -> ObservationManifest:
    """Synthetic manifest: 10 frames, calibration (0..4, 5), holdout (6, 7, 8, 9)."""
    markers = {
        "A": MarkerObservationSpec(
            label="A",
            segment="body",
            is_tracked=True,
            valid_count=8,
            missing_count=2,
            missing_spans=((6, 7),),
        ),
        "B": MarkerObservationSpec(
            label="B",
            segment="body",
            is_tracked=True,
            valid_count=10,
            missing_count=0,
        ),
        "C": MarkerObservationSpec(
            label="C",
            segment="body",
            is_tracked=True,
            valid_count=10,
            missing_count=0,
        ),
    }
    return ObservationManifest(
        capture_kind="driver",
        source_file="test.c3d",
        source_sha256="abc123",
        frame_count=10,
        rate_hz=360.0,
        units="m",
        handedness="right",
        calibration_frames=(0, 1, 2, 3, 4, 5),
        holdout_frames=(6, 7, 8, 9),
        markers=markers,
    )


def test_manifest_persists_frame_level_validity() -> None:
    """Frame-by-marker validity must be frozen in the manifest, not just aggregates."""
    manifest = build_frozen_observation_manifest("driver")
    mask = manifest.frame_validity()
    labels = tuple(manifest.markers)

    assert mask.shape == (manifest.frame_count, len(labels))
    assert mask.flags.writeable is False

    # Reconstructed per-frame validity must agree with each frozen aggregate.
    for col, lbl in enumerate(labels):
        spec = manifest.markers[lbl]
        assert int(mask[:, col].sum()) == spec.valid_count
        assert int((~mask[:, col]).sum()) == spec.missing_count
        spans = sum(end - start + 1 for start, end in spec.missing_spans)
        assert spans == spec.missing_count

    # Known partially-failed driver channel: frozen spans must reproduce it.
    r_shoulder = manifest.markers["RShoulderTop"]
    assert r_shoulder.missing_count > 0
    assert len(r_shoulder.missing_spans) > 0


def test_frame_validity_json_round_trip() -> None:
    """Frozen frame-level validity must survive JSON round-trip exactly."""
    manifest = build_frozen_observation_manifest("driver")
    restored = ObservationManifest.from_json(manifest.to_json())
    np.testing.assert_array_equal(restored.frame_validity(), manifest.frame_validity())
    assert restored.markers["RShoulderTop"].missing_spans == (
        manifest.markers["RShoulderTop"].missing_spans
    )


def test_manifest_markers_are_immutable() -> None:
    """The frozen marker map must reject mutation after construction."""
    manifest = _small_manifest()
    spec_b = manifest.markers["B"]

    with pytest.raises(TypeError):
        manifest.markers["C"] = spec_b  # type: ignore[index]
    with pytest.raises(TypeError):
        del manifest.markers["A"]  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        manifest.markers["A"] = manifest.markers["B"]  # type: ignore[index]


def test_metrics_verify_frozen_validity_mask() -> None:
    """Manifest-bound metrics must use or verify the frozen frame-level mask."""
    manifest = _small_manifest()
    pred = np.zeros((manifest.frame_count, len(manifest.markers), 3))
    obs = np.zeros((manifest.frame_count, len(manifest.markers), 3))
    frozen = manifest.frame_validity()

    # The exact frozen mask is accepted.
    rmse = compute_pooled_rmse(pred, obs, frozen, manifest=manifest)
    assert rmse == 0.0

    # A caller-altered mask (gap-filled samples marked valid) must fail closed.
    altered = frozen.copy()
    altered[6, :] = True
    with pytest.raises(ValueError, match="frozen"):
        compute_pooled_rmse(pred, obs, altered, manifest=manifest)

    # Frame-wise RMS enforces the same contract.
    with pytest.raises(ValueError, match="frozen"):
        compute_frame_wise_rms(pred, obs, altered, manifest=manifest)


def test_calibration_entry_points_consume_calibration_frames_only() -> None:
    """Real calibration entry points must slice calibration frames via the manifest.

    Holdout rows are poisoned with implausible values; a manifest-bound calibration
    must reproduce the calibration-frames-only result, proving holdout observations
    can never reach the estimator.
    """
    manifest = _small_manifest()
    n_markers = len(manifest.markers)
    rng = np.random.default_rng(7)
    full = rng.uniform(0.5, 1.0, size=(manifest.frame_count, n_markers, 3))
    # Poison every holdout frame (6..9) - any leakage changes the result.
    full[list(manifest.holdout_frames), :, :] = 100.0

    shoulder, grip, clubhead = full[:, 0], full[:, 1], full[:, 2]
    calib_rows = [list(manifest.calibration_frames)]
    expected = calibrate_fixed_geometry(
        shoulder[calib_rows], grip[calib_rows], clubhead[calib_rows]
    )

    # Manifest-aware wrapper must delegate the slicing internally.
    res = calibrate_with_manifest_protection(manifest, shoulder, grip, clubhead)
    assert res.l1_arm_m == pytest.approx(expected.l1_arm_m, abs=1e-12)
    assert res.l2_club_m == pytest.approx(expected.l2_club_m, abs=1e-12)

    # The real calibration entry point must enforce the binding itself.
    direct = calibrate_fixed_geometry(shoulder, grip, clubhead, manifest=manifest)
    assert direct.l1_arm_m == pytest.approx(expected.l1_arm_m, abs=1e-12)
    assert direct.l2_club_m == pytest.approx(expected.l2_club_m, abs=1e-12)

    # Trajectories not covering the manifest's frame count fail closed.
    with pytest.raises(ValueError, match="manifest.frame_count"):
        calibrate_fixed_geometry(
            shoulder[:-1], grip[:-1], clubhead[:-1], manifest=manifest
        )


def test_holdout_starts_at_recorded_top_of_backswing() -> None:
    """The holdout must begin at the frozen top-of-backswing event of each club.

    Computed event-based boundary: driver top of backswing at frame 397 (of 654),
    iron at frame 394 (of 657) - not the flat frame 251.
    """
    driver = build_frozen_observation_manifest("driver")
    iron = build_frozen_observation_manifest("iron")

    tob_driver = SWING_EVENTS_DRIVER.top_of_backswing.frame_index
    tob_iron = SWING_EVENTS_IRON.top_of_backswing.frame_index

    assert tob_driver == 397
    assert tob_iron == 394

    assert min(driver.holdout_frames) == tob_driver
    assert max(driver.calibration_frames) == tob_driver - 1
    assert len(driver.calibration_frames) == tob_driver
    assert len(driver.holdout_frames) == driver.frame_count - tob_driver

    assert min(iron.holdout_frames) == tob_iron
    assert max(iron.calibration_frames) == tob_iron - 1
    assert len(iron.calibration_frames) == tob_iron
    assert len(iron.holdout_frames) == iron.frame_count - tob_iron


def test_pooled_rmse_distinct_from_median_frame_rms() -> None:
    """Prove pooled RMSE != median frame RMS when frame valid counts and errors differ.

    Setup:
      Frame 0: 1 valid marker with error 0.01 m  => frame RMS = 0.01 m
      Frame 1: 10 valid markers each with error 0.10 m => frame RMS = 0.10 m
      Total valid observations: 11
      Sum squared errors: 1 * 0.01^2 + 10 * 0.10^2 = 0.0001 + 0.1000 = 0.1001 m^2
      Pooled RMSE = sqrt(0.1001 / 11) = 0.09539392... m
      Median frame RMS = (0.01 + 0.10) / 2 = 0.0550 m
    """
    n_frames = 2
    n_markers = 10
    pred = np.zeros((n_frames, n_markers, 3), dtype=np.float64)
    obs = np.zeros((n_frames, n_markers, 3), dtype=np.float64)
    valid = np.zeros((n_frames, n_markers), dtype=bool)

    # Frame 0: marker 0 is valid, distance = 0.01 m along x
    valid[0, 0] = True
    pred[0, 0, 0] = 0.01

    # Frame 1: all 10 markers valid, distance = 0.10 m along y
    valid[1, :] = True
    pred[1, :, 1] = 0.10

    pooled = compute_pooled_rmse(pred, obs, valid)
    frame_rms, median_rms = compute_frame_wise_rms(pred, obs, valid)

    expected_pooled = math.sqrt(0.1001 / 11.0)
    expected_median = 0.055

    assert math.isclose(pooled, expected_pooled, rel_tol=1e-7)
    assert math.isclose(median_rms, expected_median, rel_tol=1e-7)
    assert not math.isclose(pooled, median_rms, rel_tol=1e-3)
    assert len(frame_rms) == 2
    assert math.isclose(frame_rms[0], 0.01, rel_tol=1e-7)
    assert math.isclose(frame_rms[1], 0.10, rel_tol=1e-7)


def test_zero_valid_observations_fails_closed() -> None:
    """Zero valid observations must raise ValueError, never silently return 0.0."""
    pred = np.zeros((3, 4, 3), dtype=np.float64)
    obs = np.zeros((3, 4, 3), dtype=np.float64)
    valid = np.zeros((3, 4), dtype=bool)

    with pytest.raises(ValueError, match="zero valid observations"):
        compute_pooled_rmse(pred, obs, valid)

    with pytest.raises(ValueError, match="zero valid observations"):
        compute_frame_wise_rms(pred, obs, valid)


def test_non_finite_valid_coordinates_fail_closed() -> None:
    """NaN or Inf at valid indices must fail closed."""
    pred = np.zeros((2, 2, 3), dtype=np.float64)
    obs = np.zeros((2, 2, 3), dtype=np.float64)
    valid = np.ones((2, 2), dtype=bool)

    # Put NaN at valid position
    pred[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite coordinates"):
        compute_pooled_rmse(pred, obs, valid)

    # Put Inf at valid position
    pred[0, 0, 0] = 0.0
    obs[1, 1, 2] = np.inf
    with pytest.raises(ValueError, match="Non-finite coordinates"):
        compute_pooled_rmse(pred, obs, valid)


def test_masked_nans_are_safely_ignored() -> None:
    """NaNs at unobserved/masked positions must not cause evaluation errors."""
    pred = np.zeros((2, 2, 3), dtype=np.float64)
    obs = np.zeros((2, 2, 3), dtype=np.float64)
    valid = np.array([[True, False], [True, False]], dtype=bool)

    # Marker 1 is invalid (masked) and contains NaN
    pred[:, 1, :] = np.nan
    obs[:, 1, :] = np.nan

    # Marker 0 is valid and has error 0.03 m
    pred[0, 0, 0] = 0.03
    pred[1, 0, 0] = 0.03

    rmse = compute_pooled_rmse(pred, obs, valid)
    assert math.isclose(rmse, 0.03, rel_tol=1e-7)


def test_dimension_and_shape_mismatches_rejected() -> None:
    """Mismatched shapes between pred, obs, and valid must be rejected."""
    pred = np.zeros((2, 3, 3), dtype=np.float64)
    obs = np.zeros((2, 4, 3), dtype=np.float64)
    valid = np.ones((2, 3), dtype=bool)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_pooled_rmse(pred, obs, valid)

    obs_correct = np.zeros((2, 3, 3), dtype=np.float64)
    valid_wrong = np.ones((2, 4), dtype=bool)
    with pytest.raises(ValueError, match="Valid mask shape"):
        compute_pooled_rmse(pred, obs_correct, valid_wrong)


def test_manifest_requires_si_metres() -> None:
    """ObservationManifest must strictly enforce SI metre units."""
    with pytest.raises(ValueError, match="units must be 'm'"):
        ObservationManifest(
            capture_kind="driver",
            source_file="test.c3d",
            source_sha256="abc123",
            frame_count=100,
            rate_hz=360.0,
            units="mm",  # Invalid!
            handedness="right",
            calibration_frames=(0, 10),
            holdout_frames=(11, 20),
            markers={},
        )


def test_calibration_and_holdout_splits_must_be_disjoint() -> None:
    """Calibration frames and holdout frames must not overlap."""
    with pytest.raises(ValueError, match="must be disjoint"):
        ObservationManifest(
            capture_kind="driver",
            source_file="test.c3d",
            source_sha256="abc123",
            frame_count=100,
            rate_hz=360.0,
            units="m",
            handedness="right",
            calibration_frames=(0, 1, 2, 3, 4, 5),
            holdout_frames=(5, 6, 7, 8, 9),  # Frame 5 is in both!
            markers={},
        )


def test_calibration_routine_cannot_consume_holdout_frames() -> None:
    """Manifest-bound calibration consumes calibration frames only, by construction.

    The holdout rows are poisoned with implausible values; the manifest-aware entry
    point must reproduce the calibration-frames-only result exactly, proving no
    holdout observation can update calibrated parameters. Trajectories that do not
    cover the manifest's frame count fail closed.
    """
    manifest = _small_manifest()
    n_markers = len(manifest.markers)
    full = np.full((manifest.frame_count, n_markers, 3), 0.6)
    full[list(manifest.holdout_frames), :, :] = 5.0  # poisoned holdout rows

    shoulder, grip, clubhead = full[:, 0], full[:, 1], full[:, 2]
    calib_rows = list(manifest.calibration_frames)
    expected = calibrate_fixed_geometry(
        shoulder[calib_rows], grip[calib_rows], clubhead[calib_rows]
    )

    res = calibrate_with_manifest_protection(manifest, shoulder, grip, clubhead)
    assert res.l1_arm_m == pytest.approx(expected.l1_arm_m, abs=1e-12)
    assert res.l2_club_m == pytest.approx(expected.l2_club_m, abs=1e-12)

    # Trajectories that do not cover the manifest's frame count fail closed:
    # a caller cannot pre-select which rows the calibrator is allowed to see.
    with pytest.raises(ValueError, match="manifest.frame_count"):
        calibrate_with_manifest_protection(
            manifest, shoulder[:-1], grip[:-1], clubhead[:-1]
        )


def test_frozen_driver_manifest_conformance() -> None:
    """Authoritative frozen Driver manifest matches canonical 654 frames @ 360 Hz."""
    manifest = build_frozen_observation_manifest("driver")
    assert manifest.capture_kind == "driver"
    assert manifest.frame_count == 654
    assert manifest.rate_hz == 360.0
    assert (
        manifest.source_sha256
        == "cbcb4d84aca0a1f558073f0c2328c3f501c4726aaf699e382cf20c5861a0971d"
    )
    assert manifest.units == "m"
    assert manifest.handedness == "right"
    assert len(manifest.calibration_frames) > 0
    assert len(manifest.holdout_frames) > 0
    assert set(manifest.calibration_frames).isdisjoint(set(manifest.holdout_frames))
    assert len(manifest.markers) == 38

    # Unassigned sentinel marker check
    marker_0 = manifest.markers["Marker_0:0:0"]
    assert marker_0.is_tracked is False
    assert marker_0.exclusion_reason is not None

    # Tracked marker check
    waist = manifest.markers["WaistLeft"]
    assert waist.is_tracked is True
    assert waist.valid_count == 654
    assert waist.missing_count == 0
    assert waist.is_interpolated is False


def test_frozen_iron_manifest_conformance() -> None:
    """Authoritative frozen Iron manifest matches canonical 657 frames @ 359 Hz."""
    manifest = build_frozen_observation_manifest("iron")
    assert manifest.capture_kind == "iron"
    assert manifest.frame_count == 657
    assert math.isclose(manifest.rate_hz, 359.0, abs_tol=1.0)
    assert (
        manifest.source_sha256
        == "00a70c1ec0a887c28f9c0397766904eb9e79dee9f6d2bbd0cbb0062445adf561"
    )
    assert manifest.units == "m"
    assert manifest.handedness == "right"
    assert len(manifest.calibration_frames) > 0
    assert len(manifest.holdout_frames) > 0
    assert set(manifest.calibration_frames).isdisjoint(set(manifest.holdout_frames))
    assert len(manifest.markers) == 38


def test_manifest_round_trip_serialization() -> None:
    """ObservationManifest serializes and deserializes deterministically."""
    manifest = build_frozen_observation_manifest("driver")
    as_dict = manifest.to_dict()
    restored = ObservationManifest.from_dict(as_dict)
    assert restored == manifest

    as_json = manifest.to_json()
    from_json = ObservationManifest.from_json(as_json)
    assert from_json == manifest
