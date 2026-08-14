"""Expected-improvement Bayesian optimisation over a design space.

The acquisition is the closed-form expected improvement of Jones, Schonlau and
Welch (1998). For a minimisation problem with incumbent ``y*``:

``EI(x) = (y* - mu(x) - xi) Phi(z) + sigma(x) phi(z)``,
``z = (y* - mu(x) - xi) / sigma(x)``

with ``EI = 0`` wherever ``sigma(x) = 0``. Two consequences are worth knowing
because they are asserted in the tests: EI is non-negative everywhere, and it
is exactly zero at a training point of a noiseless GP -- a point already
sampled carries no expected improvement, which is what stops the loop
resampling it.

The loop is derivative-free by construction (ADR-0032: no adjoint exists for
the granular tiers), so candidates come from a scrambled Sobol' set and the
best of them is polished with L-BFGS-B on the acquisition surface.

References:
    Jones, D. R., Schonlau, M. and Welch, W. J. (1998). Efficient global
    optimization of expensive black-box functions. *Journal of Global
    Optimization*, 13(4), 455-492.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from src.shared.python.core.contracts import require

from .design_space import DesignSpace, make_qmc_engine
from .manifest import StudyManifest
from .rng import SeedRecord, as_generator, new_seed_record
from .surrogate import GaussianProcess, GPHyperparameters

__all__ = [
    "BayesOptResult",
    "bayesian_optimisation",
    "expected_improvement",
    "propose_location",
]

ModelFunction = Callable[[np.ndarray], np.ndarray]

#: Sobol' candidate-set size for the acquisition maximiser.
_DEFAULT_CANDIDATES = 1024


@dataclass(frozen=True, eq=False)
class BayesOptResult:
    """Outcome of a Bayesian-optimisation run.

    Attributes:
        space: The design space searched.
        x: ``(n, d)`` evaluated design points, in evaluation order.
        y: ``(n,)`` objective values.
        best_index: Row index of the best point found.
        n_initial: How many of the rows came from the initial design.
        surrogate: The GP fitted to all observations.
        manifest: Provenance, including the entropy that produced the run.
    """

    space: DesignSpace
    x: np.ndarray
    y: np.ndarray
    best_index: int
    n_initial: int
    surrogate: GaussianProcess
    manifest: StudyManifest

    @property
    def best_x(self) -> np.ndarray:
        """The best design point found.

        Returns:
            A ``(d,)`` array in physical units.
        """
        return self.x[self.best_index]

    @property
    def best_y(self) -> float:
        """The best objective value found.

        Returns:
            The extremal objective value.
        """
        return float(self.y[self.best_index])

    def best_as_dict(self) -> dict[str, float]:
        """Name the best design point.

        Returns:
            A ``{parameter_name: value}`` mapping.
        """
        return dict(zip(self.space.names, self.best_x.tolist(), strict=True))


def expected_improvement(
    surrogate: GaussianProcess,
    x: np.ndarray,
    y_best: float | None = None,
    xi: float = 0.0,
    minimise: bool = True,
) -> np.ndarray:
    """Evaluate expected improvement at candidate points.

    Args:
        surrogate: A fitted Gaussian process.
        x: ``(m, d)`` candidate points in physical units.
        y_best: The incumbent objective value. ``None`` uses the best value
            among the surrogate's training targets.
        xi: Exploration margin; larger values demand a bigger improvement
            before a point looks attractive.
        minimise: Treat lower objective values as better.

    Returns:
        A ``(m,)`` array of non-negative expected improvements.

    Raises:
        ValueError: If ``xi`` is negative.
        RuntimeError: If the surrogate is not fitted.
    """
    if xi < 0.0:
        raise ValueError(f"xi must be non-negative, got {xi}")
    targets = surrogate.y_train
    incumbent = (
        float(np.min(targets) if minimise else np.max(targets))
        if y_best is None
        else float(y_best)
    )
    mean, std = surrogate.predict(x, return_std=True)
    improvement = (incumbent - mean - xi) if minimise else (mean - incumbent - xi)

    positive = std > 0.0
    z = np.zeros_like(std)
    np.divide(improvement, std, out=z, where=positive)
    acquisition = np.where(
        positive,
        improvement * norm.cdf(z) + std * norm.pdf(z),
        0.0,
    )
    # EI is provably non-negative; the clip only removes cancellation noise.
    return np.maximum(acquisition, 0.0)


def propose_location(
    surrogate: GaussianProcess,
    space: DesignSpace,
    seed: int | SeedRecord | np.random.Generator | None = None,
    xi: float = 0.01,
    minimise: bool = True,
    n_candidates: int = _DEFAULT_CANDIDATES,
    polish: bool = True,
) -> np.ndarray:
    """Maximise expected improvement over the design space.

    Args:
        surrogate: A fitted Gaussian process.
        space: The design space to search.
        seed: Entropy for the Sobol' scramble of the candidate set.
        xi: Exploration margin.
        minimise: Treat lower objective values as better.
        n_candidates: Size of the Sobol' candidate set; rounded up to a power
            of two by the engine's requirements.
        polish: Refine the best candidate with L-BFGS-B on the acquisition.

    Returns:
        A ``(d,)`` design point in physical units.

    Raises:
        ValueError: If ``n_candidates`` is not positive.
    """
    if n_candidates <= 0:
        raise ValueError(f"n_candidates must be positive, got {n_candidates}")
    generator = as_generator(seed)
    engine = make_qmc_engine("sobol", space.dimension, generator)
    size = 1 << max(int(n_candidates - 1).bit_length(), 1)
    candidates = space.to_physical(np.asarray(engine.random(size), dtype=float))
    acquisition = expected_improvement(surrogate, candidates, xi=xi, minimise=minimise)
    best = candidates[int(np.argmax(acquisition))]
    if not polish:
        return best

    bounds = list(zip(space.lower.tolist(), space.upper.tolist(), strict=True))
    result = minimize(
        lambda point: (
            -float(
                expected_improvement(
                    surrogate, point.reshape(1, -1), xi=xi, minimise=minimise
                )[0]
            )
        ),
        best,
        method="L-BFGS-B",
        bounds=bounds,
    )
    polished = np.clip(np.asarray(result.x, dtype=float), space.lower, space.upper)
    if -float(result.fun) >= float(np.max(acquisition)):
        return polished
    return best


def bayesian_optimisation(
    space: DesignSpace,
    model: ModelFunction,
    n_initial: int = 8,
    n_iterations: int = 20,
    seed: int | SeedRecord | None = None,
    xi: float = 0.01,
    minimise: bool = True,
    n_candidates: int = _DEFAULT_CANDIDATES,
    noise_variance: float | None = None,
) -> BayesOptResult:
    """Run expected-improvement Bayesian optimisation.

    Args:
        space: The design space to search.
        model: Vectorised objective, mapping ``(m, d)`` points to ``(m,)``
            values. It is called once with the initial design and once per
            iteration with a single row.
        n_initial: Size of the initial Sobol' design; rounded up to a power of
            two.
        n_iterations: Number of surrogate-guided evaluations.
        seed: Explicit entropy or seed record; ``None`` draws fresh entropy.
        xi: Exploration margin for the acquisition.
        minimise: Treat lower objective values as better.
        n_candidates: Size of the acquisition candidate set.
        noise_variance: Fix the GP noise variance instead of fitting it. Use
            a tiny value (or ``0.0``) for a deterministic simulator.

    Returns:
        The optimisation result, including every evaluated point.

    Raises:
        ValueError: If ``n_initial`` or ``n_iterations`` is out of range, or
            the model returns the wrong shape.
    """
    if n_initial < 2:
        raise ValueError(f"n_initial must be at least 2, got {n_initial}")
    if n_iterations < 0:
        raise ValueError(f"n_iterations must be non-negative, got {n_iterations}")

    record = new_seed_record(seed)
    design_rng, acquisition_rng, fit_rng = record.spawn(3)

    initial_size = 1 << max(int(n_initial - 1).bit_length(), 1)
    engine = make_qmc_engine("sobol", space.dimension, design_rng)
    points = space.to_physical(np.asarray(engine.random(initial_size), dtype=float))
    values = np.asarray(model(points), dtype=float).ravel()
    if values.size != points.shape[0]:
        raise ValueError(
            f"model returned {values.size} values for {points.shape[0]} points"
        )

    surrogate = _fit_surrogate(space, points, values, noise_variance, fit_rng)
    for _ in range(n_iterations):
        candidate = propose_location(
            surrogate,
            space,
            seed=acquisition_rng,
            xi=xi,
            minimise=minimise,
            n_candidates=n_candidates,
        )
        observed = np.asarray(model(candidate.reshape(1, -1)), dtype=float).ravel()
        if observed.size != 1:
            raise ValueError(
                f"model returned {observed.size} values for a single point"
            )
        points = np.vstack([points, candidate.reshape(1, -1)])
        values = np.concatenate([values, observed])
        surrogate = _fit_surrogate(space, points, values, noise_variance, fit_rng)

    best_index = int(np.argmin(values) if minimise else np.argmax(values))
    require(
        points.shape[0] == values.size,
        "design and objective histories must stay aligned",
        value=(points.shape, values.shape),
    )
    manifest = StudyManifest(
        seed=record,
        method="bayesopt-ei",
        parameter_names=space.names,
        n_samples=int(values.size),
        extra={
            "n_initial": initial_size,
            "n_iterations": n_iterations,
            "xi": xi,
            "minimise": minimise,
        },
    )
    return BayesOptResult(
        space=space,
        x=points,
        y=values,
        best_index=best_index,
        n_initial=initial_size,
        surrogate=surrogate,
        manifest=manifest,
    )


def _fit_surrogate(
    space: DesignSpace,
    points: np.ndarray,
    values: np.ndarray,
    noise_variance: float | None,
    generator: np.random.Generator,
) -> GaussianProcess:
    """Fit the GP used by the acquisition.

    Args:
        space: The design space, used to normalise inputs consistently across
            iterations.
        points: ``(n, d)`` observed design points.
        values: ``(n,)`` observed objective values.
        noise_variance: Fixed noise variance, or ``None`` to fit it.
        generator: Generator for the optimiser restarts.

    Returns:
        The fitted process.
    """
    initial = (
        None
        if noise_variance is None
        else GPHyperparameters(
            signal_variance=1.0,
            length_scales=np.full(space.dimension, 0.5),
            noise_variance=noise_variance,
        )
    )
    surrogate = GaussianProcess(hyperparameters=initial, space=space)
    return surrogate.fit(
        points,
        values,
        optimize=True,
        n_restarts=2,
        fit_noise=noise_variance is None,
        seed=generator,
    )
