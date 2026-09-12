"""Rigid marker constellations impose a lower bound independent of torques."""

import numpy as np
import pytest

from src.shared.python.motion_matching.marker_rigidity import (
    rigid_marker_lower_bound,
    marker_pair_statistics,
)

pytestmark = pytest.mark.unit


def test_rigid_motion_is_free_but_length_change_is_not() -> None:
    offsets = np.array([[0.0, 0, 0], [1, 0, 0], [0, 0, 0]])
    target = np.array([[0.0, 0, 0], [0, 2, 0], [9, 8, 7]])
    result = rigid_marker_lower_bound(
        offsets, target, ["A", "A", "B"], np.ones(3, dtype=bool)
    )
    assert result["rms_lower_bound_m"] == pytest.approx(np.sqrt(0.5 / 3))
    target[1] = [0, 1, 0]
    assert (
        rigid_marker_lower_bound(
            offsets, target, ["A", "A", "B"], np.ones(3, dtype=bool)
        )["rms_lower_bound_m"]
        < 1e-12
    )


def test_missing_marker_is_excluded_and_invalid_observation_rejected() -> None:
    offsets = np.zeros((2, 3))
    target = np.array([[1.0, 2, 3], [np.nan] * 3])
    result = rigid_marker_lower_bound(
        offsets, target, ["A", "A"], np.array([True, False])
    )
    assert result["observed_count"] == 1
    with pytest.raises(ValueError):
        rigid_marker_lower_bound(offsets, target, ["A", "A"], np.ones(2, dtype=bool))


def test_pair_bound_allows_rigid_motion_but_detects_length_variation() -> None:
    points = np.array([[[0.0, 0, 0], [1, 0, 0]], [[4.0, 3, 2], [4, 5, 2]]])
    result = marker_pair_statistics(points, np.ones((2, 2), dtype=bool), ["A", "B"])
    assert result[0]["distance_std_m"] == pytest.approx(0.5)
    assert result[0]["best_fixed_length_pair_rms_m"] == pytest.approx(0.25)
    points[1, 1] = [4, 4, 2]
    assert (
        marker_pair_statistics(points, np.ones((2, 2), dtype=bool), ["A", "B"])[0][
            "distance_std_m"
        ]
        == 0
    )


def test_pair_statistics_respect_missing_observations() -> None:
    points = np.zeros((2, 2, 3))
    points[1, 1] = np.nan
    mask = np.array([[True, True], [True, False]])
    assert marker_pair_statistics(points, mask, ["A", "B"])[0]["observed_frames"] == 1
