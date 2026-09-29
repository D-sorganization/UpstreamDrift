"""Expected improvement and Bayesian optimisation (#8615).

Two acceptance properties: EI is non-negative everywhere, and it is zero at a
training point of a noiseless GP. The second is what stops the loop from
re-sampling a point it has already paid for.
"""

from __future__ import annotations

import numpy as np
import pytest
from bunkershot3d.study import (
    AcquisitionSettings,
    DesignSpace,
    GaussianProcess,
    GPHyperparameters,
    bayesian_optimisation,
    expected_improvement,
    propose_location,
)
from bunkershot3d.study.rng import new_seed_record
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.unit

SEARCH_SPACE = DesignSpace.from_bounds({"x": (-2.0, 2.0), "y": (-2.0, 2.0)})
#: EI at a noiseless training point is zero up to the Cholesky jitter, which
#: leaks in as a posterior standard deviation of order 1e-5.
EI_ZERO_TOLERANCE = 1e-4


def bowl(points: np.ndarray) -> np.ndarray:
    """A smooth unimodal objective with its minimum inside the box.

    Args:
        points: ``(n, 2)`` design points.

    Returns:
        A ``(n,)`` array of objective values, minimised at ``(0.4, -0.7)``.
    """
    return (points[:, 0] - 0.4) ** 2 + (points[:, 1] + 0.7) ** 2


def noiseless_gp(x: np.ndarray, y: np.ndarray) -> GaussianProcess:
    """Fit an interpolating GP to the given observations.

    Args:
        x: ``(n, 2)`` design points.
        y: ``(n,)`` objective values.

    Returns:
        The fitted process.
    """
    return GaussianProcess(
        hyperparameters=GPHyperparameters(1.0, np.full(2, 0.3), 0.0),
        space=SEARCH_SPACE,
    ).fit(x, y, optimize=False)


class TestExpectedImprovement:
    """Algebraic properties of the acquisition."""

    def test_is_non_negative_everywhere(self) -> None:
        generator = new_seed_record(1).generator()
        x = SEARCH_SPACE.to_physical(generator.random((12, 2)))
        gp = noiseless_gp(x, bowl(x))

        probe = SEARCH_SPACE.to_physical(generator.random((500, 2)))
        assert np.all(expected_improvement(gp, probe) >= 0.0)

    def test_is_zero_at_noiseless_training_points(self) -> None:
        generator = new_seed_record(2).generator()
        x = SEARCH_SPACE.to_physical(generator.random((15, 2)))
        y = bowl(x)
        gp = noiseless_gp(x, y)

        acquisition = expected_improvement(gp, x)
        assert np.max(acquisition) < EI_ZERO_TOLERANCE * (y.max() - y.min())

    def test_is_zero_at_the_incumbent_even_with_a_margin(self) -> None:
        generator = new_seed_record(3).generator()
        x = SEARCH_SPACE.to_physical(generator.random((10, 2)))
        y = bowl(x)
        gp = noiseless_gp(x, y)
        best = x[int(np.argmin(y))].reshape(1, -1)
        assert expected_improvement(gp, best, xi=0.5)[0] == 0.0

    def test_prefers_a_promising_unexplored_point(self) -> None:
        # Two observations far apart; the midpoint is unexplored, so it must
        # carry more expected improvement than either training point.
        x = np.array([[-1.5, -1.5], [1.5, 1.5]])
        gp = noiseless_gp(x, bowl(x))
        midpoint = np.array([[0.0, 0.0]])
        assert expected_improvement(gp, midpoint)[0] > np.max(
            expected_improvement(gp, x)
        )

    def test_larger_xi_never_increases_the_acquisition(self) -> None:
        generator = new_seed_record(4).generator()
        x = SEARCH_SPACE.to_physical(generator.random((12, 2)))
        gp = noiseless_gp(x, bowl(x))
        probe = SEARCH_SPACE.to_physical(generator.random((200, 2)))
        assert np.all(
            expected_improvement(gp, probe, xi=0.5)
            <= expected_improvement(gp, probe, xi=0.0) + 1e-12
        )

    def test_maximisation_mode_flips_the_preference(self) -> None:
        generator = new_seed_record(5).generator()
        x = SEARCH_SPACE.to_physical(generator.random((12, 2)))
        y = bowl(x)
        gp = noiseless_gp(x, y)
        probe = SEARCH_SPACE.to_physical(generator.random((300, 2)))

        minimising = probe[int(np.argmax(expected_improvement(gp, probe)))]
        maximising = probe[
            int(np.argmax(expected_improvement(gp, probe, minimise=False)))
        ]
        assert not np.allclose(minimising, maximising)

    def test_rejects_negative_exploration_margin(self) -> None:
        x = np.array([[0.0, 0.0], [1.0, 1.0]])
        gp = noiseless_gp(x, bowl(x))
        with pytest.raises(ValueError, match="xi"):
            expected_improvement(gp, x, xi=-0.1)

    @settings(deadline=None, max_examples=15)
    @given(seed=st.integers(min_value=0, max_value=2**16))
    def test_property_acquisition_is_finite_and_non_negative(self, seed: int) -> None:
        generator = new_seed_record(seed).generator()
        x = SEARCH_SPACE.to_physical(generator.random((8, 2)))
        gp = noiseless_gp(x, bowl(x))
        probe = SEARCH_SPACE.to_physical(generator.random((64, 2)))
        acquisition = expected_improvement(gp, probe, xi=0.01)
        assert np.all(np.isfinite(acquisition))
        assert np.all(acquisition >= 0.0)


class TestProposeLocation:
    """The acquisition maximiser."""

    def test_returns_a_point_inside_the_space(self) -> None:
        generator = new_seed_record(6).generator()
        x = SEARCH_SPACE.to_physical(generator.random((10, 2)))
        gp = noiseless_gp(x, bowl(x))
        candidate = propose_location(gp, SEARCH_SPACE, seed=7)
        assert candidate.shape == (2,)
        assert bool(SEARCH_SPACE.contains(candidate))

    def test_polishing_never_lowers_the_acquisition(self) -> None:
        generator = new_seed_record(8).generator()
        x = SEARCH_SPACE.to_physical(generator.random((10, 2)))
        gp = noiseless_gp(x, bowl(x))
        plain = propose_location(gp, SEARCH_SPACE, seed=9, polish=False)
        polished = propose_location(gp, SEARCH_SPACE, seed=9, polish=True)
        assert (
            expected_improvement(gp, polished.reshape(1, -1))[0]
            >= expected_improvement(gp, plain.reshape(1, -1))[0] - 1e-12
        )

    def test_rejects_empty_candidate_set(self) -> None:
        x = np.array([[0.0, 0.0], [1.0, 1.0]])
        gp = noiseless_gp(x, bowl(x))
        with pytest.raises(ValueError, match="n_candidates"):
            propose_location(gp, SEARCH_SPACE, n_candidates=0)


class TestAcquisitionSettings:
    """The acquisition is one decision, validated once."""

    def test_defaults_match_the_documented_loop(self) -> None:
        settings = AcquisitionSettings()
        assert settings.xi == pytest.approx(0.01)
        assert settings.minimise is True
        assert settings.n_candidates > 0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [({"xi": -1e-9}, "xi"), ({"n_candidates": 0}, "n_candidates")],
    )
    def test_rejects_inadmissible_settings(
        self, kwargs: dict[str, float], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            AcquisitionSettings(**kwargs)  # type: ignore[arg-type]

    def test_is_immutable(self) -> None:
        settings = AcquisitionSettings()
        with pytest.raises((AttributeError, TypeError)):
            settings.xi = 0.5  # type: ignore[misc]


class TestBayesianOptimisation:
    """The loop as a whole."""

    @pytest.mark.scientific
    def test_finds_the_minimum_of_a_smooth_bowl(self) -> None:
        result = bayesian_optimisation(
            SEARCH_SPACE,
            bowl,
            n_initial=8,
            n_iterations=15,
            seed=2026,
            noise_variance=1e-8,
        )
        np.testing.assert_allclose(result.best_x, [0.4, -0.7], atol=0.15)
        assert result.best_y < 0.05

    def test_improves_on_the_initial_design(self) -> None:
        result = bayesian_optimisation(
            SEARCH_SPACE,
            bowl,
            n_initial=8,
            n_iterations=12,
            seed=17,
            noise_variance=1e-8,
        )
        assert result.best_y < np.min(result.y[: result.n_initial])
        assert result.best_index >= result.n_initial

    def test_records_every_evaluation(self) -> None:
        result = bayesian_optimisation(
            SEARCH_SPACE, bowl, n_initial=8, n_iterations=5, seed=1
        )
        assert result.x.shape == (13, 2)
        assert result.y.shape == (13,)
        np.testing.assert_allclose(result.y, bowl(result.x), atol=1e-12)
        assert np.all(SEARCH_SPACE.contains(result.x))

    def test_reports_the_best_point_by_name(self) -> None:
        result = bayesian_optimisation(
            SEARCH_SPACE, bowl, n_initial=4, n_iterations=3, seed=2
        )
        named = result.best_as_dict()
        assert set(named) == {"x", "y"}
        np.testing.assert_allclose(list(named.values()), result.best_x)

    def test_maximisation_finds_a_maximum(self) -> None:
        result = bayesian_optimisation(
            SEARCH_SPACE,
            lambda p: -bowl(p),
            n_initial=8,
            n_iterations=12,
            seed=3,
            acquisition=AcquisitionSettings(minimise=False),
            noise_variance=1e-8,
        )
        assert result.best_y == np.max(result.y)
        np.testing.assert_allclose(result.best_x, [0.4, -0.7], atol=0.3)

    def test_same_seed_reproduces_the_whole_run(self) -> None:
        first = bayesian_optimisation(
            SEARCH_SPACE, bowl, n_initial=4, n_iterations=4, seed=555
        )
        second = bayesian_optimisation(
            SEARCH_SPACE, bowl, n_initial=4, n_iterations=4, seed=555
        )
        np.testing.assert_array_equal(first.x, second.x)
        np.testing.assert_array_equal(first.y, second.y)

    def test_manifest_replays_the_run(self) -> None:
        original = bayesian_optimisation(
            SEARCH_SPACE, bowl, n_initial=4, n_iterations=3, seed=None
        )
        replay = bayesian_optimisation(
            SEARCH_SPACE,
            bowl,
            n_initial=4,
            n_iterations=3,
            seed=original.manifest.seed.entropy,
        )
        np.testing.assert_array_equal(original.x, replay.x)
        assert original.manifest.method == "bayesopt-ei"
        assert original.manifest.extra["n_iterations"] == 3

    def test_zero_iterations_returns_the_initial_design(self) -> None:
        result = bayesian_optimisation(
            SEARCH_SPACE, bowl, n_initial=8, n_iterations=0, seed=4
        )
        assert result.x.shape == (8, 2)
        assert result.best_y == np.min(result.y)

    @pytest.mark.parametrize(("n_initial", "n_iterations"), [(1, 5), (8, -1)])
    def test_rejects_invalid_budgets(self, n_initial: int, n_iterations: int) -> None:
        with pytest.raises(ValueError):
            bayesian_optimisation(
                SEARCH_SPACE,
                bowl,
                n_initial=n_initial,
                n_iterations=n_iterations,
                seed=1,
            )

    def test_rejects_model_returning_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="model returned"):
            bayesian_optimisation(
                SEARCH_SPACE,
                lambda p: np.zeros(3),
                n_initial=8,
                n_iterations=1,
                seed=1,
            )
