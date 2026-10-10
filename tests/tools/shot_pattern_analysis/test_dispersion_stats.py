"""Descriptive endpoint dispersion metrics and their coordinate conventions."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.unit

pytestmark = pytest.mark.unit


def test_perfect_long_left_line_has_negative_error_correlation() -> None:
    from src.tools.shot_pattern_analysis.dispersion_stats import landing_dispersion

    # x is downrange; y is right. These endpoints move long as they move left.
    points = [(101.0, -1.0), (102.0, -2.0), (103.0, -3.0)]
    stats = landing_dispersion(points, target_x_m=100.0)

    assert stats["n"] == 3
    assert stats["covariance_downrange_lateral_m2"] < 0
    assert stats["corr_lateral_downrange_error"] == pytest.approx(-1.0)
    assert stats["quadrant_fractions"]["left_long"] == 1.0
    assert stats["quadrant_fractions"]["right_long"] == 0.0
    assert stats["mean_downrange_error_m"] == pytest.approx(2.0)
    assert stats["mean_lateral_error_m"] == pytest.approx(-2.0)


def test_mirroring_swaps_left_right_bins_and_flips_covariance() -> None:
    from src.tools.shot_pattern_analysis.dispersion_stats import landing_dispersion

    right_points = [(101.0, 1.0), (102.0, 2.0), (103.0, 3.0)]
    stats = landing_dispersion(right_points, target_x_m=100.0)

    assert stats["corr_lateral_downrange_error"] == pytest.approx(1.0)
    assert stats["quadrant_fractions"]["right_long"] == 1.0
    assert stats["quadrant_fractions"]["left_long"] == 0.0
    assert stats["conditional_mean_downrange_error_m"]["left"] is None
    assert stats["conditional_mean_downrange_error_m"]["right"] == pytest.approx(2.0)


def test_radial_errors_principal_axis_and_nominal_offsets_are_deterministic() -> None:
    from src.tools.shot_pattern_analysis.dispersion_stats import landing_dispersion

    points = [(9.0, -1.0), (10.0, 0.0), (11.0, 1.0), (12.0, 2.0)]
    stats = landing_dispersion(points, target_x_m=10.0, nominal_point=(10.0, 0.0))

    expected_radii = [math.hypot(x - 10.0, y) for x, y in points]
    assert stats["median_radial_target_error_m"] == pytest.approx(
        np.median(expected_radii)
    )
    assert stats["p95_radial_target_error_m"] == pytest.approx(
        np.quantile(expected_radii, 0.95)
    )
    assert (
        stats["principal_axis_eigenvalues_m2"][0]
        >= stats["principal_axis_eigenvalues_m2"][1]
    )
    assert stats["principal_axis_orientation_deg"] == pytest.approx(45.0)
    assert stats["nominal_mean_offset_m"] == pytest.approx((0.5, 0.5))


def test_zero_variance_returns_none_correlations_and_zero_eigenvalues() -> None:
    from src.tools.shot_pattern_analysis.dispersion_stats import landing_dispersion

    stats = landing_dispersion([(100.0, 0.0), (100.0, 0.0)], target_x_m=100.0)

    assert stats["corr_lateral_downrange_error"] is None
    assert stats["corr_lateral_radial_carry"] is None
    assert stats["principal_axis_eigenvalues_m2"] == pytest.approx((0.0, 0.0))
    assert stats["principal_axis_orientation_deg"] is None
    assert sum(stats["quadrant_fractions"].values()) == 0.0


@pytest.mark.parametrize(
    ("points", "target_x_m", "nominal_point"),
    [
        ([(1.0, 2.0)], 100.0, None),
        ([(1.0, 2.0), (math.nan, 2.0)], 100.0, None),
        ([(1.0, 2.0), (3.0, 4.0)], math.inf, None),
        ([(1.0, 2.0), (3.0, 4.0)], 100.0, (math.nan, 0.0)),
    ],
)
def test_invalid_points_rejected(points, target_x_m, nominal_point) -> None:
    from src.tools.shot_pattern_analysis.dispersion_stats import landing_dispersion

    with pytest.raises(ValueError):
        landing_dispersion(points, target_x_m, nominal_point=nominal_point)
