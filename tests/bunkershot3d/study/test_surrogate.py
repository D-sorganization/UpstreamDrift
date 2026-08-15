"""Gaussian-process surrogate: interpolation, variance and calibration (#8615).

Three acceptance properties are asserted here:

1. with zero noise the GP is an *exact interpolant* -- the posterior mean
   reproduces the training targets and the posterior variance collapses to
   zero at those points;
2. the posterior variance grows away from the data;
3. the predictive variance is *calibrated*: about 95 % of held-out points fall
   inside the two-sigma band.

Property (3) is tested on functions drawn from the GP's own prior, which is
the only setting where 95 % is the right number by construction. The
misspecified case (hyperparameters fitted from data) is tested separately with
a weaker bound, because fitting costs coverage and pretending otherwise would
be the dishonest version of this test.
"""

from __future__ import annotations

import numpy as np
import pytest
from bunkershot3d.study import DesignSpace, GaussianProcess, GPHyperparameters
from bunkershot3d.study.rng import new_seed_record
from bunkershot3d.study.surrogate import squared_exponential_kernel
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.unit

#: Interpolation residual permitted at a training point with zero noise. The
#: floor is the relative Cholesky jitter (1e-10 of the signal variance), which
#: is what stops a nearly singular kernel matrix from failing to factorise.
INTERPOLATION_TOLERANCE = 1e-7
#: Posterior variance permitted at a training point; same jitter floor.
ZERO_VARIANCE_TOLERANCE = 1e-9

UNIT_SQUARE = DesignSpace.from_bounds({"a": (0.0, 1.0), "b": (0.0, 1.0)})


def smooth_target(points: np.ndarray) -> np.ndarray:
    """A smooth two-dimensional test function.

    Args:
        points: ``(n, 2)`` inputs.

    Returns:
        A ``(n,)`` array of outputs.
    """
    return np.sin(3.0 * points[:, 0]) + 0.5 * np.cos(4.0 * points[:, 1])


def interpolating_gp() -> GaussianProcess:
    """Build a GP configured as a noiseless interpolator.

    Returns:
        An unfitted process with zero noise and moderate length scales.
    """
    return GaussianProcess(
        hyperparameters=GPHyperparameters(
            signal_variance=1.0,
            length_scales=np.full(2, 0.3),
            noise_variance=0.0,
        ),
        space=UNIT_SQUARE,
        normalize_y=False,
    )


def prior_draw(
    points: np.ndarray,
    length_scales: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Sample a function from the GP prior at ``points``.

    Args:
        points: ``(n, d)`` locations.
        length_scales: ``(d,)`` kernel length scales.
        generator: Seeded generator.

    Returns:
        A ``(n,)`` draw from the zero-mean prior.
    """
    covariance = squared_exponential_kernel(points, points, 1.0, length_scales)
    covariance[np.diag_indices(points.shape[0])] += 1e-9
    return np.linalg.cholesky(covariance) @ generator.standard_normal(points.shape[0])


class TestKernel:
    """Basic algebraic properties of the covariance function."""

    def test_is_symmetric_with_signal_variance_on_the_diagonal(self) -> None:
        rng = new_seed_record(1).generator()
        x = rng.random((12, 3))
        k = squared_exponential_kernel(x, x, 2.5, np.array([0.4, 0.9, 1.2]))
        np.testing.assert_allclose(k, k.T, atol=1e-14)
        np.testing.assert_allclose(np.diag(k), 2.5, atol=1e-12)

    def test_is_positive_definite(self) -> None:
        rng = new_seed_record(2).generator()
        x = rng.random((20, 2))
        k = squared_exponential_kernel(x, x, 1.0, np.array([0.3, 0.3]))
        eigenvalues = np.linalg.eigvalsh(k)
        assert eigenvalues.min() > -1e-12

    def test_decays_with_distance(self) -> None:
        left = np.zeros((1, 1))
        right = np.array([[0.0], [0.5], [1.0], [4.0]])
        k = squared_exponential_kernel(left, right, 1.0, np.array([1.0]))[0]
        assert np.all(np.diff(k) < 0.0)
        assert k[0] == pytest.approx(1.0)

    def test_rejects_mismatched_dimensions(self) -> None:
        with pytest.raises(ValueError, match="dimension mismatch"):
            squared_exponential_kernel(
                np.zeros((3, 2)), np.zeros((3, 3)), 1.0, np.ones(2)
            )


class TestExactInterpolation:
    """Zero noise must give zero residual at the training points."""

    def test_posterior_mean_reproduces_the_training_targets(self) -> None:
        rng = new_seed_record(5).generator()
        x = rng.random((25, 2))
        y = smooth_target(x)
        gp = interpolating_gp().fit(x, y, optimize=False)

        mean = gp.predict(x)
        assert np.max(np.abs(mean - y)) < INTERPOLATION_TOLERANCE

    def test_posterior_variance_is_zero_at_the_training_points(self) -> None:
        rng = new_seed_record(6).generator()
        x = rng.random((25, 2))
        gp = interpolating_gp().fit(x, smooth_target(x), optimize=False)

        _, std = gp.predict(x, return_std=True)
        assert np.max(std**2) < ZERO_VARIANCE_TOLERANCE

    def test_interpolation_survives_hyperparameter_fitting(self) -> None:
        rng = new_seed_record(7).generator()
        x = rng.random((20, 2))
        y = smooth_target(x)
        gp = GaussianProcess(
            hyperparameters=GPHyperparameters(1.0, np.full(2, 0.3), 0.0),
            space=UNIT_SQUARE,
        ).fit(x, y, optimize=True, fit_noise=False, n_restarts=2, seed=3)

        residual = np.max(np.abs(gp.predict(x) - y))
        # Looser than the fixed-hyperparameter case on purpose: the marginal
        # likelihood happily walks to long length scales, where the kernel
        # matrix is nearly singular and the jitter shows up in the residual.
        assert residual < 1e-4 * (y.max() - y.min())

    def test_noise_free_prediction_between_points_stays_bounded(self) -> None:
        rng = new_seed_record(8).generator()
        x = rng.random((30, 2))
        y = smooth_target(x)
        gp = interpolating_gp().fit(x, y, optimize=False)
        probe = rng.random((50, 2))
        mean = gp.predict(probe)
        assert np.all(np.isfinite(mean))
        assert mean.min() > y.min() - 1.0
        assert mean.max() < y.max() + 1.0


class TestPosteriorVariance:
    """Variance must grow away from the data."""

    def test_variance_increases_along_a_ray_from_a_single_point(self) -> None:
        gp = interpolating_gp().fit(
            np.array([[0.5, 0.5]]), np.array([1.0]), optimize=False
        )
        probe = np.stack([np.linspace(0.5, 1.0, 12), np.full(12, 0.5)], axis=1)
        _, std = gp.predict(probe, return_std=True)
        assert np.all(np.diff(std) > 0.0)

    def test_variance_is_lower_near_the_data_than_far_from_it(self) -> None:
        rng = new_seed_record(9).generator()
        x = 0.2 + 0.1 * rng.random((15, 2))
        gp = interpolating_gp().fit(x, smooth_target(x), optimize=False)

        _, near = gp.predict(np.array([[0.25, 0.25]]), return_std=True)
        _, far = gp.predict(np.array([[0.95, 0.95]]), return_std=True)
        assert far[0] > near[0]

    def test_variance_approaches_the_prior_far_from_the_data(self) -> None:
        gp = interpolating_gp().fit(
            np.array([[0.0, 0.0]]), np.array([0.0]), optimize=False
        )
        _, std = gp.predict(np.array([[1.0, 1.0]]), return_std=True)
        assert std[0] == pytest.approx(1.0, abs=1e-6)

    def test_including_noise_inflates_the_predictive_variance(self) -> None:
        rng = new_seed_record(10).generator()
        x = rng.random((15, 2))
        gp = GaussianProcess(
            hyperparameters=GPHyperparameters(1.0, np.full(2, 0.3), 0.05),
            space=UNIT_SQUARE,
            normalize_y=False,
        ).fit(x, smooth_target(x), optimize=False)

        probe = rng.random((10, 2))
        _, latent = gp.predict(probe, return_std=True)
        _, observed = gp.predict(probe, return_std=True, include_noise=True)
        assert np.all(observed > latent)
        np.testing.assert_allclose(observed**2 - latent**2, 0.05, atol=1e-9)


class TestCalibration:
    """Held-out coverage of the two-sigma band."""

    @pytest.mark.scientific
    def test_two_sigma_band_covers_about_95_percent_of_held_out_points(self) -> None:
        length_scales = np.array([0.25, 0.25])
        hyper = GPHyperparameters(1.0, length_scales, 1e-8)
        generator = new_seed_record(2026).generator()

        inside = 0
        total = 0
        for _ in range(12):
            points = generator.random((200, 2))
            values = prior_draw(points, length_scales, generator)
            gp = GaussianProcess(
                hyperparameters=hyper, space=UNIT_SQUARE, normalize_y=False
            ).fit(points[:60], values[:60], optimize=False)

            mean, std = gp.predict(points[60:], return_std=True)
            inside += int(np.sum(np.abs(mean - values[60:]) <= 2.0 * std))
            total += points.shape[0] - 60

        coverage = inside / total
        assert 0.90 <= coverage <= 1.0, f"two-sigma coverage was {coverage:.3f}"

    @pytest.mark.scientific
    def test_one_sigma_band_covers_about_68_percent(self) -> None:
        length_scales = np.array([0.25, 0.25])
        hyper = GPHyperparameters(1.0, length_scales, 1e-8)
        generator = new_seed_record(4242).generator()

        inside = 0
        total = 0
        for _ in range(12):
            points = generator.random((200, 2))
            values = prior_draw(points, length_scales, generator)
            gp = GaussianProcess(
                hyperparameters=hyper, space=UNIT_SQUARE, normalize_y=False
            ).fit(points[:60], values[:60], optimize=False)
            mean, std = gp.predict(points[60:], return_std=True)
            inside += int(np.sum(np.abs(mean - values[60:]) <= std))
            total += points.shape[0] - 60

        coverage = inside / total
        assert 0.58 <= coverage <= 0.80, f"one-sigma coverage was {coverage:.3f}"

    @pytest.mark.scientific
    def test_fitted_hyperparameters_lose_some_coverage(self) -> None:
        # Honest bound: with hyperparameters estimated from 60 points the
        # nominal 95 % band under-covers. Asserting 0.95 here would be a
        # test that only passes by luck.
        length_scales = np.array([0.25, 0.25])
        generator = new_seed_record(31337).generator()

        inside = 0
        total = 0
        for draw in range(4):
            points = generator.random((160, 2))
            values = prior_draw(points, length_scales, generator)
            gp = GaussianProcess(space=UNIT_SQUARE).fit(
                points[:60], values[:60], optimize=True, n_restarts=2, seed=draw
            )
            mean, std = gp.predict(points[60:], return_std=True)
            inside += int(np.sum(np.abs(mean - values[60:]) <= 2.0 * std))
            total += 100

        coverage = inside / total
        assert coverage >= 0.75, f"two-sigma coverage was {coverage:.3f}"


class TestHyperparameterFitting:
    """Log-marginal-likelihood optimisation."""

    def test_analytic_gradient_matches_finite_differences(self) -> None:
        rng = new_seed_record(11).generator()
        x = rng.random((20, 2))
        y = smooth_target(x)
        gp = GaussianProcess(space=UNIT_SQUARE, normalize_y=False)
        gp.fit(x, y, optimize=False)

        # Reaching into the private objective is deliberate: the gradient is
        # the part of the fit that silently degrades to a slow, wrong search
        # if it is inconsistent with the objective.
        objective = gp._negative_lml_and_grad
        theta = np.log(np.array([1.3, 0.4, 0.9, 1e-4]))
        _, analytic = objective(theta, y)
        numeric = np.zeros_like(analytic)
        step = 1e-6
        for i in range(theta.size):
            plus = theta.copy()
            plus[i] += step
            minus = theta.copy()
            minus[i] -= step
            numeric[i] = (objective(plus, y)[0] - objective(minus, y)[0]) / (2.0 * step)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-4)

    def test_fitting_improves_the_log_marginal_likelihood(self) -> None:
        rng = new_seed_record(12).generator()
        x = rng.random((25, 2))
        y = smooth_target(x)

        fixed = GaussianProcess(space=UNIT_SQUARE).fit(x, y, optimize=False)
        fitted = GaussianProcess(space=UNIT_SQUARE).fit(
            x, y, optimize=True, n_restarts=3, seed=4
        )
        assert fitted.log_marginal_likelihood > fixed.log_marginal_likelihood

    def test_fitting_recovers_the_anisotropy(self) -> None:
        rng = new_seed_record(13).generator()
        x = rng.random((60, 2))
        # Fast in the first coordinate, nearly flat in the second.
        y = np.sin(8.0 * x[:, 0]) + 0.05 * x[:, 1]
        gp = GaussianProcess(space=UNIT_SQUARE).fit(
            x, y, optimize=True, n_restarts=4, seed=5
        )
        scales = gp.hyperparameters.length_scales
        assert scales[1] > 3.0 * scales[0]

    def test_fitting_recovers_an_injected_noise_level(self) -> None:
        rng = new_seed_record(14).generator()
        x = rng.random((120, 2))
        clean = smooth_target(x)
        noisy = clean + 0.2 * rng.standard_normal(x.shape[0])
        gp = GaussianProcess(space=UNIT_SQUARE).fit(
            x, noisy, optimize=True, n_restarts=3, seed=6
        )
        # Targets are standardised, so compare against the scaled variance.
        recovered = gp.hyperparameters.noise_variance * float(np.var(noisy))
        assert 0.5 * 0.04 < recovered < 2.0 * 0.04

    def test_fixed_noise_is_not_moved_by_the_optimiser(self) -> None:
        rng = new_seed_record(15).generator()
        x = rng.random((20, 2))
        gp = GaussianProcess(
            hyperparameters=GPHyperparameters(1.0, np.full(2, 0.4), 0.0),
            space=UNIT_SQUARE,
        ).fit(x, smooth_target(x), optimize=True, fit_noise=False, seed=1)
        assert gp.hyperparameters.noise_variance == 0.0

    def test_same_seed_gives_the_same_fit(self) -> None:
        rng = new_seed_record(16).generator()
        x = rng.random((20, 2))
        y = smooth_target(x)
        first = GaussianProcess(space=UNIT_SQUARE).fit(
            x, y, optimize=True, n_restarts=3, seed=99
        )
        second = GaussianProcess(space=UNIT_SQUARE).fit(
            x, y, optimize=True, n_restarts=3, seed=99
        )
        np.testing.assert_allclose(
            first.hyperparameters.length_scales,
            second.hyperparameters.length_scales,
            atol=1e-12,
        )

    @settings(deadline=None, max_examples=10)
    @given(seed=st.integers(min_value=0, max_value=2**16))
    def test_property_fit_is_finite_for_any_seed(self, seed: int) -> None:
        rng = new_seed_record(seed).generator()
        x = rng.random((15, 2))
        y = smooth_target(x)
        gp = GaussianProcess(space=UNIT_SQUARE).fit(
            x, y, optimize=True, n_restarts=1, seed=seed
        )
        assert np.isfinite(gp.log_marginal_likelihood)
        mean, std = gp.predict(x, return_std=True)
        assert np.all(np.isfinite(mean))
        assert np.all(std >= 0.0)


class TestFailureModes:
    """Contract violations must raise."""

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit must be called"):
            GaussianProcess().predict(np.zeros((1, 2)))

    def test_hyperparameters_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit must be called"):
            _ = GaussianProcess().hyperparameters

    def test_mismatched_training_shapes_raise(self) -> None:
        with pytest.raises(ValueError, match="rows"):
            GaussianProcess().fit(np.zeros((5, 2)), np.zeros(4))

    def test_empty_training_set_raises(self) -> None:
        with pytest.raises(ValueError, match="zero observations"):
            GaussianProcess().fit(np.zeros((0, 2)), np.zeros(0))

    def test_nan_training_data_raises(self) -> None:
        x = np.zeros((4, 2))
        y = np.array([0.0, np.nan, 1.0, 2.0])
        with pytest.raises(ValueError, match="NaN"):
            GaussianProcess().fit(x, y)

    def test_query_with_wrong_width_raises(self) -> None:
        rng = new_seed_record(17).generator()
        x = rng.random((6, 2))
        gp = GaussianProcess(space=UNIT_SQUARE).fit(x, smooth_target(x), optimize=False)
        with pytest.raises(ValueError, match="columns"):
            gp.predict(np.zeros((3, 5)))

    @pytest.mark.parametrize(
        ("signal", "scales", "noise"),
        [
            (0.0, np.ones(2), 0.1),
            (1.0, np.array([0.0, 1.0]), 0.1),
            (1.0, np.ones(2), -1.0),
            (np.nan, np.ones(2), 0.1),
        ],
    )
    def test_invalid_hyperparameters_raise(
        self, signal: float, scales: np.ndarray, noise: float
    ) -> None:
        with pytest.raises(ValueError):
            GPHyperparameters(signal, scales, noise)

    def test_negative_jitter_raises(self) -> None:
        with pytest.raises(ValueError, match="jitter"):
            GaussianProcess(jitter=-1e-9)

    def test_wrong_hyperparameter_dimension_raises(self) -> None:
        gp = GaussianProcess(hyperparameters=GPHyperparameters(1.0, np.ones(3), 1e-6))
        with pytest.raises(ValueError, match="length scales"):
            gp.fit(np.zeros((5, 2)), np.zeros(5), optimize=False)
