"""Regression tests for the golf-ball drag-crisis calibration table."""

from __future__ import annotations

import pytest

from src.shared.python.physics.drag_crisis import (
    calibrated_golf_ball_drag_coefficient,
    load_golf_ball_drag_calibration,
)


def test_committed_drag_calibration_table_is_sorted_and_source_sized() -> None:
    """The drag-crisis curve is backed by committed calibration points."""
    points = load_golf_ball_drag_calibration()

    assert len(points) >= 8
    assert all(
        points[index][0] < points[index + 1][0] for index in range(len(points) - 1)
    )
    assert min(cd for _, cd in points) == pytest.approx(0.225)


def test_calibrated_curve_has_drag_crisis_and_post_crisis_recovery() -> None:
    """Cd drops near critical Re and then recovers at high Reynolds number."""
    cd_subcritical = calibrated_golf_ball_drag_coefficient(30_000.0)
    cd_minimum = calibrated_golf_ball_drag_coefficient(80_000.0)
    cd_high_speed = calibrated_golf_ball_drag_coefficient(300_000.0)

    assert cd_minimum < cd_subcritical
    assert cd_high_speed > cd_minimum


def test_reference_drag_coefficient_scales_calibrated_curve() -> None:
    """Existing tunable Cd settings still scale the empirical curve."""
    default = calibrated_golf_ball_drag_coefficient(160_000.0, 0.25)
    higher = calibrated_golf_ball_drag_coefficient(160_000.0, 0.40)

    assert default == pytest.approx(0.25)
    assert higher == pytest.approx(0.40)
