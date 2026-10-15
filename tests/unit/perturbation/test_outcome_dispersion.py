"""Tests for OutcomeDispersion (NSX-82, #12711).

Spec: docs/research/loop_nullspace/loop_nullspace.tex ``sec:var-face``.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from src.shared.python.perturbation.outcome_dispersion import (
    OutcomeDispersion,
    compute_outcome_dispersion,
    wrap_angle,
)
from src.shared.python.perturbation.statistics import compute_metric_statistics

pytestmark = pytest.mark.unit


class TestWrapAngle:
    def test_range_is_half_open_minus_pi_exclusive(self) -> None:
        got = wrap_angle(np.array([np.pi, -np.pi, 3 * np.pi, 0.0]))
        np.testing.assert_allclose(got, [np.pi, np.pi, np.pi, 0.0])

    def test_degrees(self) -> None:
        got = wrap_angle(np.array([190.0, -190.0, 360.0]), period=360.0)
        np.testing.assert_allclose(got, [-170.0, 170.0, 0.0], atol=1e-12)


class TestWrapAcrossPlusMinus180:
    def test_bias_across_the_wrap(self) -> None:
        # three samples straddling +-180 deg, true mean error is 180 deg
        angles = np.array([178.0, -178.0, 180.0])
        d = compute_outcome_dispersion(
            angles, target=0.0, tolerance=5.0, angle_unit="deg", seed=0, n_boot=50
        )
        assert abs(abs(d.bias) - 180.0) < 1e-9
        # naive linear mean would say 60 deg and a huge SD
        assert d.sd == pytest.approx(np.sqrt(8.0 / 3.0), rel=1e-9)

    def test_target_near_the_wrap_gives_small_errors(self) -> None:
        target = 179.0
        angles = np.array([-179.0, 178.0, 179.0, -178.0])  # errors 2, -1, 0, 3
        d = compute_outcome_dispersion(
            angles, target=target, tolerance=2.5, angle_unit="deg", seed=0, n_boot=50
        )
        assert d.bias == pytest.approx(1.0, abs=0.05)
        assert d.tolerance_exceedance == pytest.approx(0.25)  # only +3 exceeds

    def test_same_statistics_when_shifted_by_a_full_turn(self) -> None:
        base = np.array([-3.0, -1.0, 0.5, 2.0, 4.0])
        a = compute_outcome_dispersion(
            base, 0.0, 2.0, angle_unit="deg", seed=1, n_boot=100
        )
        b = compute_outcome_dispersion(
            base + 360.0, 0.0, 2.0, angle_unit="deg", seed=1, n_boot=100
        )
        assert a.bias == pytest.approx(b.bias, abs=1e-9)
        assert a.sd == pytest.approx(b.sd, abs=1e-9)
        assert a.iqr == pytest.approx(b.iqr, abs=1e-9)
        assert a.tolerance_exceedance == b.tolerance_exceedance

    def test_radians_and_degrees_agree(self) -> None:
        deg = np.array([-4.0, -1.0, 0.0, 2.0, 5.0])
        d_deg = compute_outcome_dispersion(
            deg, 0.0, 3.0, angle_unit="deg", seed=2, n_boot=100
        )
        d_rad = compute_outcome_dispersion(
            np.deg2rad(deg), 0.0, np.deg2rad(3.0), angle_unit="rad", seed=2, n_boot=100
        )
        assert np.rad2deg(d_rad.sd) == pytest.approx(d_deg.sd)
        assert d_rad.tolerance_exceedance == d_deg.tolerance_exceedance


class TestStatisticsValues:
    def test_exceedance_and_miss(self) -> None:
        angles = np.array([-6.0, -2.0, 0.0, 1.0, 3.0, 9.0])
        d = compute_outcome_dispersion(
            angles,
            0.0,
            tolerance=2.5,
            miss_half_width=5.0,
            angle_unit="deg",
            seed=0,
            n_boot=50,
        )
        assert d.tolerance_exceedance == pytest.approx(3 / 6)  # -6, 3, 9
        assert d.miss_probability == pytest.approx(2 / 6)  # -6, 9

    def test_miss_probability_nan_with_reason_when_undeclared(self) -> None:
        d = compute_outcome_dispersion(
            np.array([0.0, 1.0]), 0.0, 5.0, angle_unit="deg", seed=0, n_boot=20
        )
        assert np.isnan(d.miss_probability)
        assert d.miss_reason is not None
        assert "miss_half_width" in d.miss_reason

    def test_non_positive_miss_half_width_rejected(self) -> None:
        with pytest.raises(ValueError, match="miss_half_width"):
            compute_outcome_dispersion(
                np.array([0.0, 1.0]), 0.0, 5.0, miss_half_width=0.0, angle_unit="deg"
            )

    def test_iqr_is_of_deviations_about_the_bias(self) -> None:
        angles = np.linspace(-10.0, 10.0, 101) + 40.0  # bias 40, IQR 10
        d = compute_outcome_dispersion(
            angles, 0.0, 5.0, angle_unit="deg", seed=0, n_boot=50
        )
        assert d.bias == pytest.approx(40.0, abs=1e-6)
        assert d.iqr == pytest.approx(10.0, abs=1e-6)

    def test_zero_variance_sample(self) -> None:
        d = compute_outcome_dispersion(
            np.full(5, 3.0), 0.0, 1.0, angle_unit="deg", seed=0, n_boot=20
        )
        assert d.sd == pytest.approx(0.0, abs=1e-12)
        assert d.iqr == pytest.approx(0.0, abs=1e-12)
        assert d.tolerance_exceedance == 1.0

    def test_sd_is_population_convention(self) -> None:
        d = compute_outcome_dispersion(
            np.array([-1.0, 1.0]), 0.0, 5.0, angle_unit="deg", seed=0, n_boot=20
        )
        assert d.sd == pytest.approx(np.std([-1.0, 1.0]))


class TestWideSymmetricDistributionIsNotRobust:
    def test_cv_old_failure_mode_now_visible(self) -> None:
        angles = np.array([-90.0, 90.0] * 50)  # mean error 0, enormous spread
        assert np.isnan(compute_metric_statistics(angles).cv)
        d = compute_outcome_dispersion(
            angles, 0.0, 5.0, angle_unit="deg", seed=0, n_boot=50
        )
        assert d.sd == pytest.approx(90.0)
        assert d.tolerance_exceedance == 1.0

    def test_exceedance_orders_narrow_below_wide(self) -> None:
        rng = np.random.default_rng(0)
        narrow = rng.normal(0.0, 1.0, 2000)
        wide = rng.normal(0.0, 20.0, 2000)
        dn = compute_outcome_dispersion(narrow, 0.0, 3.0, angle_unit="deg", seed=1)
        dw = compute_outcome_dispersion(wide, 0.0, 3.0, angle_unit="deg", seed=1)
        assert dn.tolerance_exceedance < 0.01 < dw.tolerance_exceedance


class TestBootstrap:
    def test_seeded_cis_are_reproducible(self) -> None:
        angles = np.random.default_rng(3).normal(2.0, 5.0, 200)
        kw = {"angle_unit": "deg", "seed": 4, "miss_half_width": 10.0}
        a = compute_outcome_dispersion(angles, 0.0, 5.0, **kw)  # type: ignore[arg-type]
        b = compute_outcome_dispersion(angles, 0.0, 5.0, **kw)  # type: ignore[arg-type]
        assert a == b

    def test_point_estimates_do_not_depend_on_the_seed(self) -> None:
        angles = np.random.default_rng(3).normal(2.0, 5.0, 200)
        a = compute_outcome_dispersion(angles, 0.0, 5.0, angle_unit="deg", seed=4)
        b = compute_outcome_dispersion(angles, 0.0, 5.0, angle_unit="deg", seed=5)
        assert a.bias == b.bias and a.sd == b.sd and a.iqr == b.iqr

    def test_cis_bracket_point_estimates_and_order(self) -> None:
        angles = np.random.default_rng(3).normal(2.0, 5.0, 300)
        d = compute_outcome_dispersion(
            angles, 0.0, 5.0, miss_half_width=10.0, angle_unit="deg", seed=4
        )
        for est, ci in (
            (d.bias, d.bias_ci),
            (d.sd, d.sd_ci),
            (d.iqr, d.iqr_ci),
            (d.tolerance_exceedance, d.tolerance_exceedance_ci),
            (d.miss_probability, d.miss_probability_ci),
        ):
            assert ci[0] <= est <= ci[1]

    def test_bias_ci_half_width_matches_normal_theory(self) -> None:
        n, sigma = 400, 5.0
        angles = np.random.default_rng(10).normal(2.0, sigma, n)
        d = compute_outcome_dispersion(
            angles, 0.0, 5.0, angle_unit="deg", seed=11, n_boot=2000
        )
        # normal-theory half-width of a 95% CI for a mean: 1.96 sigma / sqrt(n);
        # the bootstrap half-width has relative error ~ 1/sqrt(2 n_boot) ~ 1.6%
        half = 0.5 * (d.bias_ci[1] - d.bias_ci[0])
        assert half == pytest.approx(1.96 * sigma / np.sqrt(n), rel=0.1)

    def test_bias_ci_coverage_over_replicate_datasets(self) -> None:
        reps, n, true_bias = 150, 100, 2.0
        hits = 0
        for r in range(reps):
            x = np.random.default_rng(1000 + r).normal(true_bias, 5.0, n)
            d = compute_outcome_dispersion(
                x, 0.0, 5.0, angle_unit="deg", seed=r, n_boot=200
            )
            hits += d.bias_ci[0] <= true_bias <= d.bias_ci[1]
        nominal = 0.95
        binom_se = np.sqrt(nominal * (1 - nominal) / reps)
        # percentile-bootstrap undercoverage is O(1/n) = 0.01 at n = 100
        assert hits / reps >= nominal - 4.5 * binom_se - 1.0 / n

    def test_ci_level_recorded(self) -> None:
        d = compute_outcome_dispersion(
            np.array([0.0, 1.0, 2.0]),
            0.0,
            5.0,
            angle_unit="deg",
            seed=0,
            n_boot=50,
            ci_level=0.9,
        )
        assert d.ci_level == 0.9 and d.n_boot == 50 and d.n_samples == 3


class TestContract:
    def test_frozen_and_json_serialisable(self) -> None:
        d = compute_outcome_dispersion(
            np.array([0.0, 1.0, 2.0]), 0.0, 5.0, angle_unit="deg", seed=0, n_boot=30
        )
        assert isinstance(d, OutcomeDispersion)
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.sd = 1.0  # type: ignore[misc]
        json.dumps(d.to_dict(), allow_nan=False)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"angles": np.array([])},
            {"angles": np.array([1.0, np.nan])},
            {"angles": np.zeros((2, 2))},
            {"angles": np.zeros(3), "tolerance": 0.0},
            {"angles": np.zeros(3), "tolerance": -1.0},
            {"angles": np.zeros(3), "angle_unit": "grad"},
            {"angles": np.zeros(3), "ci_level": 1.0},
            {"angles": np.zeros(3), "n_boot": 5},
            {"angles": np.zeros(3), "target": float("inf")},
        ],
    )
    def test_preconditions(self, kwargs: dict[str, object]) -> None:
        base: dict[str, object] = {
            "target": 0.0,
            "tolerance": 1.0,
            "angle_unit": "deg",
        }
        base.update(kwargs)
        with pytest.raises(ValueError):
            compute_outcome_dispersion(**base)  # type: ignore[arg-type]
