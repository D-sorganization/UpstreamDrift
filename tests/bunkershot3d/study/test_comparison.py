"""Ranking candidate designs with uncertainty (#8615)."""

from __future__ import annotations

import numpy as np
import pytest
from bunkershot3d.study import compare_designs, compare_predicted_designs
from bunkershot3d.study.comparison import _IntervalEstimate
from bunkershot3d.study.rng import new_seed_record

pytestmark = pytest.mark.unit

GRIND_NAMES = ("low_bounce", "standard", "wide_sole")


def replicates(
    means: list[float],
    scatter: float,
    n_replicates: int,
    seed: int,
) -> np.ndarray:
    """Synthesise replicate evaluations around known means.

    Args:
        means: True mean per design.
        scatter: Standard deviation of the replicate noise.
        n_replicates: Replicates per design.
        seed: Entropy for the noise.

    Returns:
        A ``(len(means), n_replicates)`` observation matrix.
    """
    generator = new_seed_record(seed).generator()
    centre = np.asarray(means)[:, None]
    return centre + scatter * generator.standard_normal((len(means), n_replicates))


class TestBootstrapComparison:
    """Ranking from replicated evaluations."""

    def test_ranks_a_clearly_separated_set(self) -> None:
        data = replicates([1.0, 2.0, 3.0], scatter=0.05, n_replicates=12, seed=1)
        result = compare_designs(GRIND_NAMES, data, seed=2)

        assert result.ordered() == GRIND_NAMES
        assert result.best == "low_bounce"
        np.testing.assert_array_equal(result.rank, [0, 1, 2])
        assert result.probability_best[0] > 0.99
        assert result.is_separated()

    def test_higher_is_better_flips_the_ranking(self) -> None:
        data = replicates([1.0, 2.0, 3.0], scatter=0.05, n_replicates=12, seed=1)
        result = compare_designs(GRIND_NAMES, data, lower_is_better=False, seed=2)
        assert result.best == "wide_sole"
        assert result.probability_best[2] > 0.99

    def test_reports_overlap_when_designs_are_indistinguishable(self) -> None:
        # Means 1.00 and 1.02 with scatter 0.5 over 6 replicates cannot be
        # separated; the comparison must say so rather than crown a winner.
        data = replicates([1.0, 1.02, 1.01], scatter=0.5, n_replicates=6, seed=3)
        result = compare_designs(GRIND_NAMES, data, seed=4)

        assert not result.is_separated()
        assert len(result.indistinguishable_from_best()) == 3
        assert np.max(result.probability_best) < 0.9

    def test_probability_best_sums_to_one(self) -> None:
        data = replicates([1.0, 1.1, 1.05], scatter=0.2, n_replicates=8, seed=5)
        result = compare_designs(GRIND_NAMES, data, seed=6)
        assert result.probability_best.sum() == pytest.approx(1.0)

    def test_pairwise_probabilities_are_complementary(self) -> None:
        data = replicates([1.0, 1.4, 2.0], scatter=0.3, n_replicates=10, seed=7)
        result = compare_designs(GRIND_NAMES, data, seed=8)
        upper = result.probability_better
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert upper[i, j] == 0.0
                else:
                    assert upper[i, j] + upper[j, i] == pytest.approx(1.0, abs=1e-9)

    def test_intervals_bracket_the_mean_and_shrink_with_replicates(self) -> None:
        few = compare_designs(
            GRIND_NAMES,
            replicates([1.0, 2.0, 3.0], 0.4, n_replicates=4, seed=9),
            seed=10,
        )
        many = compare_designs(
            GRIND_NAMES,
            replicates([1.0, 2.0, 3.0], 0.4, n_replicates=64, seed=9),
            seed=10,
        )
        assert np.all(few.ci_low <= few.mean)
        assert np.all(few.mean <= few.ci_high)
        assert np.mean(many.ci_high - many.ci_low) < np.mean(few.ci_high - few.ci_low)

    def test_standard_error_matches_the_textbook_formula(self) -> None:
        data = replicates([1.0, 2.0, 3.0], 0.3, n_replicates=16, seed=11)
        result = compare_designs(GRIND_NAMES, data, seed=12)
        expected = data.std(axis=1, ddof=1) / np.sqrt(data.shape[1])
        np.testing.assert_allclose(result.std_error, expected, atol=1e-12)

    def test_same_seed_reproduces_the_comparison(self) -> None:
        data = replicates([1.0, 1.1, 1.2], 0.3, n_replicates=8, seed=13)
        first = compare_designs(GRIND_NAMES, data, seed=14)
        second = compare_designs(GRIND_NAMES, data, seed=14)
        np.testing.assert_array_equal(first.probability_best, second.probability_best)
        np.testing.assert_allclose(first.ci_low, second.ci_low, atol=1e-15)


class TestSurrogateComparison:
    """Ranking from surrogate predictions."""

    def test_ranks_by_predicted_mean(self) -> None:
        result = compare_predicted_designs(
            GRIND_NAMES,
            mean=np.array([1.0, 1.5, 2.0]),
            std=np.array([0.01, 0.01, 0.01]),
            seed=1,
        )
        assert result.ordered() == GRIND_NAMES
        assert result.probability_best[0] > 0.99

    def test_a_wide_interval_can_still_win(self) -> None:
        # The second design has a worse mean but far more uncertainty, so it
        # retains a real chance of being best; a point-estimate ranking would
        # hide that entirely.
        result = compare_predicted_designs(
            GRIND_NAMES,
            mean=np.array([1.0, 1.2, 3.0]),
            std=np.array([0.05, 1.0, 0.05]),
            seed=2,
            n_draws=20000,
        )
        assert result.rank[0] == 0
        assert result.probability_best[1] > 0.2
        assert not result.is_separated()

    def test_zero_uncertainty_is_deterministic(self) -> None:
        result = compare_predicted_designs(
            GRIND_NAMES,
            mean=np.array([3.0, 1.0, 2.0]),
            std=np.zeros(3),
            seed=3,
        )
        assert result.probability_best[1] == 1.0
        np.testing.assert_allclose(result.ci_low, result.ci_high, atol=1e-15)

    def test_intervals_are_the_gaussian_two_sigma_band(self) -> None:
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.1, 0.2, 0.3])
        result = compare_predicted_designs(GRIND_NAMES, mean, std, seed=4)
        np.testing.assert_allclose(result.ci_low, mean - 1.959964 * std, atol=1e-5)
        np.testing.assert_allclose(result.ci_high, mean + 1.959964 * std, atol=1e-5)


class TestIntervalEstimateInvariants:
    """The estimate and its band are checked as one object, not four arrays."""

    def test_rejects_arrays_of_different_lengths(self) -> None:
        with pytest.raises(ValueError, match="agree in length"):
            _IntervalEstimate(
                mean=np.array([1.0, 2.0]),
                std_error=np.array([0.1, 0.1]),
                ci_low=np.array([0.9]),
                ci_high=np.array([1.1, 2.1]),
            )

    def test_rejects_an_inverted_interval(self) -> None:
        with pytest.raises(ValueError, match="ci_high"):
            _IntervalEstimate(
                mean=np.array([1.0]),
                std_error=np.array([0.1]),
                ci_low=np.array([1.5]),
                ci_high=np.array([0.5]),
            )


class TestFailureModes:
    """Bad inputs must raise."""

    def test_rejects_single_replicate(self) -> None:
        with pytest.raises(ValueError, match="at least two replicates"):
            compare_designs(GRIND_NAMES, np.ones((3, 1)), seed=1)

    def test_rejects_name_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="names for"):
            compare_designs(("a", "b"), np.ones((3, 4)), seed=1)

    def test_rejects_duplicate_names(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            compare_designs(("a", "a"), np.ones((2, 4)), seed=1)

    def test_rejects_nan_observations(self) -> None:
        data = np.ones((2, 4))
        data[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            compare_designs(("a", "b"), data, seed=1)

    def test_rejects_negative_standard_deviations(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            compare_predicted_designs(
                ("a", "b"), np.array([1.0, 2.0]), np.array([0.1, -0.1]), seed=1
            )

    def test_rejects_mismatched_prediction_lengths(self) -> None:
        with pytest.raises(ValueError, match="entries"):
            compare_predicted_designs(
                ("a", "b"), np.array([1.0, 2.0]), np.array([0.1]), seed=1
            )

    @pytest.mark.parametrize("level", [0.0, 1.0, 1.5])
    def test_rejects_impossible_confidence_level(self, level: float) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            compare_designs(("a", "b"), np.ones((2, 4)), confidence_level=level, seed=1)

    def test_rejects_non_positive_bootstrap_count(self) -> None:
        with pytest.raises(ValueError, match="n_bootstrap"):
            compare_designs(("a", "b"), np.ones((2, 4)), n_bootstrap=0, seed=1)


def test_manifest_records_the_comparison_settings() -> None:
    data = replicates([1.0, 2.0, 3.0], 0.2, n_replicates=8, seed=20)
    result = compare_designs(GRIND_NAMES, data, seed=None, n_bootstrap=500)
    manifest = result.manifest
    assert manifest is not None
    assert manifest.method == "design-comparison-bootstrap"
    assert manifest.extra["design_names"] == list(GRIND_NAMES)
    assert manifest.extra["n_bootstrap"] == 500

    replay = compare_designs(
        GRIND_NAMES, data, seed=manifest.seed.entropy, n_bootstrap=500
    )
    np.testing.assert_array_equal(result.probability_best, replay.probability_best)
