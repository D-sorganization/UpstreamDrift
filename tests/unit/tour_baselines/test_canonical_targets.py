"""Unit tests for CanonicalTourTarget facade and evaluation contracts (TB-01 #10586)."""

from pathlib import Path
import pytest
import numpy as np

from src.shared.python.tour_baselines.canonical_targets import (
    load_canonical_tour_target,
    evaluate_target_tracking_error,
)
from src.shared.python.tour_baselines.measurement_map import MeasurementClass

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER_C3D = REPO_ROOT / "data" / "C3D_TA_Driver.c3d"
IRON_C3D = REPO_ROOT / "data" / "C3D_TA_Iron.c3d"


def test_canonical_driver_target_properties() -> None:
    target = load_canonical_tour_target(DRIVER_C3D)
    assert target.kind == "driver"
    assert target.frames == 654
    assert target.rate_hz == 360.0
    assert target.duration_s == pytest.approx(653 / 360.0)
    assert len(target.labels) == 38
    assert target.events.impact.frame_index == 476
    assert target.provenance.player_id == "967eac5b-2e78-4207-a99f-d57437296d70"
    assert target.provenance.manufacturer_software == "GearsSports"


def test_canonical_iron_target_properties() -> None:
    target = load_canonical_tour_target(IRON_C3D)
    assert target.kind == "iron"
    assert target.frames == 657
    assert target.rate_hz == 359.0
    assert target.events.impact.frame_index == 480
    assert target.provenance.player_id == "967eac5b-2e78-4207-a99f-d57437296d70"


def test_missing_club_clusters_not_scored_as_zero_error() -> None:
    target = load_canonical_tour_target(DRIVER_C3D)

    # If predicted markers are all zeros or dummy, evaluation on valid points gives real error
    pred_zeros = np.zeros_like(target.capture.points_m)
    metrics = evaluate_target_tracking_error(target, pred_zeros)

    assert metrics["overall_rmse_mm"] > 500.0  # definitely not zero!

    # When evaluating a slice where club markers are occluded/invalid,
    # the function must return NaN or exclude invalid samples, never score as 0.0 error!
    # Let's create an invalid mask for all club markers
    club_indices = [
        target.capture.index(lbl)
        for lbl in (
            "Marker_2:2:1",
            "Marker_2:2:2",
            "Marker_2:2:3",
            "Marker_3:3:1",
            "Marker_3:3:2",
            "Marker_3:3:3",
        )
    ]
    custom_valid = target.capture.valid.copy()
    custom_valid[:, club_indices] = False

    metrics_no_club = evaluate_target_tracking_error(
        target, pred_zeros, valid_mask=custom_valid
    )
    assert np.isnan(metrics_no_club["club_rmse_mm"])
