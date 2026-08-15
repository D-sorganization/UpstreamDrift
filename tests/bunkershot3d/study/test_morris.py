"""Morris elementary-effects screening (#8615).

The acceptance criterion is that screening puts a genuinely inert factor at
``mu* = 0`` -- exactly zero, not "small" -- because that is the decision the
method exists to support: dropping a factor before paying for a Sobol'
analysis.
"""

from __future__ import annotations

import numpy as np
import pytest
from bunkershot3d.study import (
    DesignSpace,
    ishigami,
    ishigami_space,
    morris_design,
    morris_screening,
    morris_statistics,
)
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.unit


def unit_space(n_factors: int) -> DesignSpace:
    """Build a ``[0, 1]^n`` design space.

    Args:
        n_factors: Number of factors.

    Returns:
        The design space.
    """
    return DesignSpace.from_bounds({f"x{i + 1}": (0.0, 1.0) for i in range(n_factors)})


def linear_model(coefficients: np.ndarray):
    """Build a linear model with the given coefficients.

    Args:
        coefficients: ``(d,)`` slopes.

    Returns:
        A vectorised callable mapping ``(n, d)`` points to ``(n,)`` values.
    """

    def model(points: np.ndarray) -> np.ndarray:
        return points @ coefficients

    return model


class TestInertFactors:
    """A factor the model ignores must score exactly zero."""

    def test_inert_factor_gets_zero_mu_star(self) -> None:
        space = unit_space(4)
        model = linear_model(np.array([3.0, -2.0, 0.0, 0.5]))
        result = morris_screening(space, model, n_trajectories=12, seed=42)

        assert result.mu_star[2] == 0.0
        assert result.mu[2] == 0.0
        assert result.sigma[2] == 0.0
        assert result.inert() == ("x3",)

    def test_inert_factor_is_last_in_the_ranking(self) -> None:
        space = unit_space(5)
        model = linear_model(np.array([1.0, 4.0, 0.0, 2.0, 0.25]))
        result = morris_screening(space, model, n_trajectories=10, seed=7)
        assert result.ranked() == ("x2", "x4", "x1", "x5", "x3")

    def test_inert_factor_stays_inert_under_a_nonlinear_model(self) -> None:
        space = unit_space(3)

        def model(points: np.ndarray) -> np.ndarray:
            return np.sin(4.0 * points[:, 0]) + points[:, 1] ** 3

        result = morris_screening(space, model, n_trajectories=15, seed=11)
        assert result.mu_star[2] == 0.0
        assert result.mu_star[0] > 0.0
        assert result.mu_star[1] > 0.0


class TestElementaryEffectValues:
    """Effects are exact for a linear model."""

    def test_mu_star_equals_the_slope_magnitudes(self) -> None:
        coefficients = np.array([2.0, -5.0, 0.75])
        result = morris_screening(
            unit_space(3), linear_model(coefficients), n_trajectories=8, seed=3
        )
        np.testing.assert_allclose(result.mu_star, np.abs(coefficients), atol=1e-12)
        np.testing.assert_allclose(result.mu, coefficients, atol=1e-12)
        np.testing.assert_allclose(result.sigma, 0.0, atol=1e-12)

    def test_effects_are_reported_per_unit_cube_step(self) -> None:
        # A factor spanning 100 physical units with slope 1 has the same
        # elementary effect as a unit-span factor with slope 100.
        space = DesignSpace.from_bounds({"wide": (0.0, 100.0), "narrow": (0.0, 1.0)})
        result = morris_screening(
            space, linear_model(np.array([1.0, 100.0])), n_trajectories=6, seed=4
        )
        np.testing.assert_allclose(result.mu_star, [100.0, 100.0], atol=1e-9)

    @settings(deadline=None, max_examples=25)
    @given(
        coefficients=st.lists(
            st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=4,
        ),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_property_linear_model_effects_are_the_coefficients(
        self, coefficients: list[float], seed: int
    ) -> None:
        slopes = np.asarray(coefficients)
        result = morris_screening(
            unit_space(slopes.size),
            linear_model(slopes),
            n_trajectories=4,
            seed=seed,
        )
        np.testing.assert_allclose(result.mu, slopes, atol=1e-9)


class TestNonLinearityDetection:
    """``sigma`` separates linear factors from interacting ones."""

    def test_sigma_is_large_for_the_interacting_ishigami_factors(self) -> None:
        result = morris_screening(
            ishigami_space(), ishigami, n_trajectories=40, seed=21, n_levels=8
        )
        # x2 enters through sin^2 alone; x1 and x3 interact with each other.
        assert result.sigma[0] > result.sigma[1]
        assert result.sigma[2] > 0.0

    def test_screening_does_not_drop_the_interaction_only_factor(self) -> None:
        # x3 has a first-order Sobol' index of exactly zero, so a screen that
        # only looked at S1 would discard it. Morris measures derivatives, so
        # it keeps x3 -- which is the behaviour that makes it a safe *first*
        # step. Note this also means the Morris ranking need not match the
        # first-order Sobol' ranking; on Ishigami it does not.
        result = morris_screening(
            ishigami_space(), ishigami, n_trajectories=40, seed=5, n_levels=8
        )
        assert result.inert() == ()
        assert np.all(result.mu_star > 1.0)
        assert result.ranked()[0] == "x1"


class TestDesignStructure:
    """One-factor-at-a-time trajectory geometry."""

    def test_cost_is_r_times_dimension_plus_one(self) -> None:
        design = morris_design(unit_space(6), n_trajectories=10, seed=1)
        assert design.n_evaluations == 10 * (6 + 1)
        assert design.design_matrix().shape == (70, 6)

    def test_each_step_moves_exactly_one_factor_by_delta(self) -> None:
        design = morris_design(unit_space(4), n_trajectories=6, seed=2, n_levels=4)
        for trajectory in design.trajectories:
            steps = np.diff(trajectory, axis=0)
            moved = np.abs(steps) > 1e-12
            np.testing.assert_array_equal(moved.sum(axis=1), np.ones(4, dtype=int))
            np.testing.assert_allclose(np.abs(steps[moved]), design.delta, atol=1e-12)

    def test_each_factor_moves_exactly_once_per_trajectory(self) -> None:
        design = morris_design(unit_space(5), n_trajectories=4, seed=9)
        for trajectory in design.trajectories:
            moved = np.abs(np.diff(trajectory, axis=0)) > 1e-12
            np.testing.assert_array_equal(moved.sum(axis=0), np.ones(5, dtype=int))

    def test_trajectories_stay_inside_the_unit_cube(self) -> None:
        design = morris_design(unit_space(3), n_trajectories=20, seed=13, n_levels=6)
        assert design.trajectories.min() >= 0.0
        assert design.trajectories.max() <= 1.0

    def test_delta_follows_the_morris_formula(self) -> None:
        for levels in (4, 6, 8):
            design = morris_design(
                unit_space(2), n_trajectories=2, seed=1, n_levels=levels
            )
            assert design.delta == pytest.approx(levels / (2.0 * (levels - 1)))


class TestTrajectorySelection:
    """Oversampling keeps the requested number of trajectories."""

    def test_oversampling_returns_the_requested_count(self) -> None:
        design = morris_design(unit_space(3), n_trajectories=5, seed=6, oversample=4)
        assert design.n_trajectories == 5
        assert design.manifest.extra["oversample"] == 4

    def test_oversampling_preserves_exact_effects(self) -> None:
        coefficients = np.array([1.5, 0.0, -3.0])
        result = morris_screening(
            unit_space(3),
            linear_model(coefficients),
            n_trajectories=5,
            seed=6,
            oversample=4,
        )
        np.testing.assert_allclose(result.mu, coefficients, atol=1e-12)

    def test_oversampling_spreads_trajectories_further_apart(self) -> None:
        plain = morris_design(unit_space(4), n_trajectories=4, seed=31, oversample=1)
        spread = morris_design(unit_space(4), n_trajectories=4, seed=31, oversample=6)

        def mean_separation(design) -> float:
            flat = design.trajectories.reshape(design.n_trajectories, -1)
            diff = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
            return float(diff.sum() / (diff.size - diff.shape[0]))

        assert mean_separation(spread) > mean_separation(plain)


class TestReproducibility:
    """Screening designs replay from their manifest."""

    def test_same_seed_gives_identical_trajectories(self) -> None:
        first = morris_design(unit_space(3), n_trajectories=6, seed=77)
        second = morris_design(unit_space(3), n_trajectories=6, seed=77)
        np.testing.assert_array_equal(first.trajectories, second.trajectories)

    def test_manifest_records_the_screening_settings(self) -> None:
        design = morris_design(unit_space(3), n_trajectories=6, seed=None, n_levels=6)
        extra = design.manifest.extra
        assert extra["n_trajectories"] == 6
        assert extra["n_levels"] == 6
        assert design.manifest.method == "morris"
        replay = morris_design(
            unit_space(3),
            n_trajectories=6,
            seed=design.manifest.seed.entropy,
            n_levels=6,
        )
        np.testing.assert_array_equal(design.trajectories, replay.trajectories)


class TestFailureModes:
    """Bad inputs raise instead of returning a plausible ranking."""

    @pytest.mark.parametrize("levels", [3, 5, 1, 0])
    def test_rejects_odd_or_degenerate_level_counts(self, levels: int) -> None:
        with pytest.raises(ValueError, match="n_levels"):
            morris_design(unit_space(2), n_trajectories=2, seed=1, n_levels=levels)

    def test_rejects_non_positive_trajectory_count(self) -> None:
        with pytest.raises(ValueError, match="n_trajectories"):
            morris_design(unit_space(2), n_trajectories=0, seed=1)

    def test_rejects_oversample_below_one(self) -> None:
        with pytest.raises(ValueError, match="oversample"):
            morris_design(unit_space(2), n_trajectories=2, seed=1, oversample=0)

    def test_rejects_nan_outputs(self) -> None:
        design = morris_design(unit_space(2), n_trajectories=3, seed=1)
        outputs = np.zeros(design.n_evaluations)
        outputs[2] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            morris_statistics(design, outputs)

    def test_rejects_wrong_output_length(self) -> None:
        design = morris_design(unit_space(2), n_trajectories=3, seed=1)
        with pytest.raises(ValueError, match="expected"):
            morris_statistics(design, np.zeros(5))

    def test_rejects_model_returning_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="model returned"):
            morris_screening(
                unit_space(2), lambda x: np.zeros(2), n_trajectories=3, seed=1
            )
