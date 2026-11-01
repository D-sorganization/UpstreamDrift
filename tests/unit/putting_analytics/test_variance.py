"""Variance attribution: analytic Sobol recovery and elimination behaviour (#14175)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.putting_analytics.sampling import ExecutionSample, ReadSample
from src.shared.python.putting_analytics.skill import (
    CHANNELS,
    PuttingSkillProfile,
    ReadDispersion,
    StrokeDispersion,
)
from src.shared.python.putting_analytics.variance import (
    AttributionConfig,
    PuttOutcomes,
    attribute_variance,
    elimination_gains,
    sobol_attribution,
    strokes_gained_per_round,
)

CUP_RADIUS_M = 0.054
SPEED_WINDOW = 0.15  # holed only while |speed ratio - 1| is inside this
PROFILE = PuttingSkillProfile(
    "test",
    StrokeDispersion(
        face_sd_deg=1.0, path_sd_deg=2.0, speed_cv=0.5, strike_toe_sd_mm=1.5
    ),
    ReadDispersion(grade_rel_sd=0.4, aspect_sd_deg=3.0, stimp_rel_sd=0.2),
)
# Variance of each factor as it enters the additive model below.
ADDITIVE_COEFFS = {"face": 1.0, "path": 1.0, "speed": 3.0, "strike": 0.5}
ADDITIVE_VARIANCE = {
    "face": 1.0**2,
    "path": 2.0**2,
    "speed": 3.0**2 * 0.5**2,  # log-normal multiplier with mean 1 has variance cv^2
    "strike": 0.5**2 * 1.5**2,
}


def _outcomes(strokes: np.ndarray) -> PuttOutcomes:
    return PuttOutcomes(strokes, strokes == 1.0, np.zeros_like(strokes))


def additive_model(ex: ExecutionSample, rd: ReadSample) -> PuttOutcomes:
    c = ADDITIVE_COEFFS
    strokes = (
        2.0
        + c["face"] * ex.face_deg
        + c["path"] * ex.path_deg
        + c["speed"] * (ex.speed_ratio - 1.0)
        + c["strike"] * ex.strike_toe_mm
    )
    return _outcomes(strokes)


def interaction_model(ex: ExecutionSample, rd: ReadSample) -> PuttOutcomes:
    return _outcomes(ex.face_deg * ex.path_deg)


def make_toy_model(distance_m: float):
    def toy(ex: ExecutionSample, rd: ReadSample) -> PuttOutcomes:
        line = np.tan(np.radians(ex.start_line_error_deg(PROFILE.face_weight)))
        miss_m = distance_m * line
        pace_ok = np.abs(ex.speed_ratio - 1.0) < SPEED_WINDOW
        holed = (np.abs(miss_m) < CUP_RADIUS_M) & pace_ok
        leave = np.hypot(miss_m, 0.6 * distance_m * (ex.speed_ratio - 1.0))
        strokes = np.where(holed, 1.0, np.where(leave < 0.9, 2.0, 3.0))
        return PuttOutcomes(strokes, holed, leave)

    return toy


@pytest.fixture(scope="module")
def additive_report():
    config = AttributionConfig(
        factors=tuple(ADDITIVE_COEFFS), base_samples=2048, seed=3
    )
    return sobol_attribution(additive_model, PROFILE, config)


def test_sobol_recovers_known_indices_on_additive_function(additive_report) -> None:
    total = sum(ADDITIVE_VARIANCE.values())
    for name, var in ADDITIVE_VARIANCE.items():
        est = additive_report.estimate(name, "strokes")
        truth = var / total
        assert est.availability == "available"
        for value, (low, high) in ((est.s1, est.s1_ci), (est.st, est.st_ci)):
            assert low <= truth <= high, (name, truth, value, low, high)
        assert est.s1 == pytest.approx(truth, abs=0.05)


def test_total_index_exceeds_first_order_under_interaction() -> None:
    config = AttributionConfig(factors=("face", "path"), base_samples=16384, seed=1)
    report = sobol_attribution(interaction_model, PROFILE, config)
    for name in ("face", "path"):
        est = report.estimate(name, "strokes")
        assert est.s1_ci[0] < 0.0 < est.s1_ci[1]  # pure product: no main effect
        assert est.st_ci[0] > 0.9  # but the whole variance is interaction


def test_sobol_is_deterministic_and_reports_all_outputs() -> None:
    config = AttributionConfig(factors=("face", "speed"), base_samples=256, seed=5)
    model = make_toy_model(2.0)
    first = sobol_attribution(model, PROFILE, config)
    second = sobol_attribution(model, PROFILE, config)
    assert first.to_json_dict() == second.to_json_dict()
    assert {e.output_name for e in first.estimates} == {"strokes", "holed", "leave_m"}


def test_elimination_is_deterministic_and_monotone_on_toy_model() -> None:
    config = AttributionConfig(n_samples=40_000, seed=7)
    model = make_toy_model(2.5)
    base, rows = elimination_gains(model, PROFILE, config)
    again_base, again_rows = elimination_gains(model, PROFILE, config)
    assert (base, rows) == (again_base, again_rows)
    by_name = {row.factor: row for row in rows}
    for row in rows:
        assert row.gain >= -3.0 * row.gain_se, row  # never worse beyond MC error
        assert -3.0 * row.gain_se <= row.gain_reduced <= row.gain + 3.0 * row.gain_se
    assert (
        by_name["speed"].gain > 10.0 * by_name["speed"].gain_se
    )  # pace is a real driver
    assert by_name["path"].gain > 0.0
    for ignored in ("strike", "read_grade", "read_aspect", "read_stimp"):
        assert by_name[ignored].gain == 0.0 == by_name[ignored].gain_reduced


def test_attribute_variance_and_per_round_weighting() -> None:
    config = AttributionConfig(
        factors=("face", "speed"), base_samples=128, n_samples=5000, n_bootstrap=20
    )
    models = {"short": make_toy_model(1.0), "long": make_toy_model(4.0)}
    result = attribute_variance(models, PROFILE, config)
    assert set(result) == {"short", "long"}
    assert result["short"].baseline_strokes < result["long"].baseline_strokes
    per_round = strokes_gained_per_round(result, {"short": 10.0, "long": 2.0})
    expected = 10.0 * result["short"].elimination("speed").gain
    expected += 2.0 * result["long"].elimination("speed").gain
    assert per_round["speed"][0] == pytest.approx(expected)
    with pytest.raises(ValueError):
        strokes_gained_per_round(result, {"short": 1.0})


def test_invalid_configuration_is_rejected() -> None:
    for bad in (
        {"factors": ()},
        {"factors": ("grain",)},
        {"factors": ("face", "face")},
        {"reduction": 0.0},
        {"n_samples": 1},
    ):
        with pytest.raises(ValueError):
            AttributionConfig(**bad)
    assert set(AttributionConfig().factors) == set(CHANNELS)
