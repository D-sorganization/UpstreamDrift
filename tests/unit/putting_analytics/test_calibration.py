"""Fitting a skill profile to a published make-rate table (#14177)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from src.shared.python.putting_analytics import calibration as cal
from src.shared.python.putting_analytics.make_rate_reference import (
    MakeRatePoint,
    MakeRateTable,
    reference_table,
)
from src.shared.python.putting_analytics.skill import (
    PuttingSkillProfile,
    ReadDispersion,
    StrokeDispersion,
)

pytestmark = pytest.mark.unit

BASE = PuttingSkillProfile(
    "base", StrokeDispersion(1.0, 2.0, 0.07, 3.0), ReadDispersion(0.15, 10.0, 0.08)
)


def toy_rates(profile: PuttingSkillProfile, grid: Sequence[float]) -> list[float]:
    """Line spread shrinks short-putt makes; pace spread shrinks long ones."""
    line = profile.stroke.face_sd_deg / BASE.stroke.face_sd_deg
    pace = profile.stroke.speed_cv / BASE.stroke.speed_cv
    return [math.exp(-((line * d / 8.0) ** 2)) / (1.0 + pace * d / 20.0) for d in grid]


def _table(points: Sequence[MakeRatePoint]) -> MakeRateTable:
    return MakeRateTable(
        "toy", "g", "reference", "make_pct", "p", "https://x", None, "d", tuple(points)
    )


def test_bands_are_predicted_from_the_centres_of_their_thirds() -> None:
    assert cal.evaluation_distances(MakeRatePoint(4.0, 4.0, 50.0)) == (4.0,)
    assert cal.evaluation_distances(MakeRatePoint(0.0, 3.0, 50.0)) == pytest.approx(
        (0.5, 1.5, 2.5)
    )


def test_compare_reports_residuals_in_percentage_points() -> None:
    table = _table([MakeRatePoint(2.0, 2.0, 50.0), MakeRatePoint(4.0, 4.0, 10.0)])
    comparison = cal.compare(BASE, table, lambda _p, grid: [0.6, 0.1][: len(grid)])
    assert comparison.predicted_pct == pytest.approx((60.0, 10.0))
    assert comparison.residuals_pct == pytest.approx((10.0, 0.0))
    assert comparison.rms_error_pct == pytest.approx(math.sqrt(50.0))


def test_scaled_profile_scales_line_and_pace_channels_apart() -> None:
    scaled = cal.scaled_profile(BASE, 2.0, 0.5, name="s")
    assert scaled.name == "s" and scaled.stroke.face_sd_deg == 2.0
    assert scaled.read.aspect_sd_deg == 20.0 and scaled.stroke.speed_cv == 0.035
    assert scaled.read.stimp_rel_sd == 0.04 and scaled.face_weight == BASE.face_weight


def test_fit_recovers_known_line_and_pace_scales() -> None:
    truth = cal.scaled_profile(BASE, 1.6, 0.7)
    grid = (1.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 45.0, 60.0)
    pct = toy_rates(truth, grid)
    table = _table(
        [MakeRatePoint(d, d, 100.0 * p) for d, p in zip(grid, pct, strict=True)]
    )
    fit = cal.fit_profile(BASE, table, cal.FitSettings(max_evaluations=200), toy_rates)
    assert (fit.line_scale, fit.pace_scale) == pytest.approx((1.6, 0.7), rel=0.03)
    assert fit.comparison.rms_error_pct < 0.2 and fit.profile.name == "fit:toy"


def test_settings_validate_their_bounds() -> None:
    with pytest.raises(ValueError, match="scale_bounds"):
        cal.FitSettings(scale_bounds=(1.0, 0.5))
    with pytest.raises(ValueError, match="max_evaluations"):
        cal.FitSettings(max_evaluations=2)


def test_compare_needs_two_distances() -> None:
    with pytest.raises(ValueError, match="two evaluation"):
        cal.compare(BASE, _table([MakeRatePoint(3.0, 3.0, 50.0)]), toy_rates)


def test_seeded_comparison_uses_the_simulated_curve() -> None:
    table = reference_table("golfmonthly_2024", "hcp_0")
    comparison = cal.compare_seeded(BASE, table, cal.FitSettings(n=50, seed=3))
    assert len(comparison.predicted_pct) == 2
    assert comparison.predicted_pct[0] > comparison.predicted_pct[1]


def test_tour_preset_matches_the_tour_target_and_beats_scratch_amateurs() -> None:
    settings = cal.FitSettings(n=300, seed=1)
    tour = cal.compare_seeded(
        cal.TOUR_AVERAGE, reference_table("golfmagic_2022", "pga_tour"), settings
    )
    assert tour.rms_error_pct < 3.0
    scratch = cal.compare_seeded(
        cal.TOUR_AVERAGE, reference_table("shotscope_mygolfspy", "hcp_0"), settings
    )
    assert sum(scratch.residuals_pct) > 0.0 and scratch.rms_error_pct < 8.0
