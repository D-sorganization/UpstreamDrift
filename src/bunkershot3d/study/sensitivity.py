"""Variance-based (Sobol') sensitivity analysis on numpy.

The estimators are the pair that is standard practice today:

- **first order**, Saltelli et al. (2010) Eq. (b):
  ``S_i = mean(f_B * (f_AB_i - f_A)) / Var(f)``
- **total order**, Jansen (1999):
  ``ST_i = mean((f_A - f_AB_i)^2) / (2 Var(f))``

with ``Var(f)`` taken over the union of the ``A`` and ``B`` samples. Without
second-order indices the design costs ``N (D + 2)`` model evaluations.

The base samples come from a single ``2 D``-dimensional scrambled Sobol'
sequence split down the middle, which is what keeps ``A`` and ``B``
uncorrelated; ``N`` must therefore be a power of two.

Confidence intervals are bootstrap percentile intervals over the ``N``
independent base rows (resampling rows, not evaluations, keeps the ``A``/``B``
pairing intact).

References:
    Saltelli, A. et al. (2010). Variance based sensitivity analysis of model
    output. *Computer Physics Communications*, 181(2), 259-270.

    Jansen, M. J. W. (1999). Analysis of variance designs for model output.
    *Computer Physics Communications*, 117(1-2), 35-43.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.shared.python.core.contracts import require

from .design_space import DesignSpace, is_power_of_two, make_qmc_engine
from .manifest import StudyManifest
from .rng import SeedRecord, as_generator, new_seed_record

__all__ = [
    "SaltelliDesign",
    "SobolIndices",
    "saltelli_design",
    "sobol_analysis",
    "sobol_indices_from_outputs",
]

#: Objective functions map an ``(n, d)`` design matrix to ``(n,)`` outputs.
ModelFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, eq=False)
class SaltelliDesign:
    """The ``N (D + 2)`` evaluation plan for a Sobol' analysis.

    Attributes:
        space: The design space sampled.
        n_base: Base sample size ``N`` (a power of two).
        a_unit: ``(N, D)`` first base sample, unit cube.
        b_unit: ``(N, D)`` second base sample, unit cube.
        ab_unit: ``(D, N, D)`` cross samples; ``ab_unit[i]`` is ``a_unit``
            with column ``i`` replaced by ``b_unit``'s column ``i``.
        manifest: Provenance, including the entropy that produced it.
    """

    space: DesignSpace
    n_base: int
    a_unit: np.ndarray
    b_unit: np.ndarray
    ab_unit: np.ndarray
    manifest: StudyManifest

    @property
    def dimension(self) -> int:
        """Number of design parameters ``D``.

        Returns:
            The design-space dimension.
        """
        return self.space.dimension

    @property
    def n_evaluations(self) -> int:
        """Total model evaluations required, ``N (D + 2)``.

        Returns:
            The evaluation count.
        """
        return self.n_base * (self.dimension + 2)

    def design_matrix(self) -> np.ndarray:
        """Stack the whole plan into one physical design matrix.

        Row blocks are ordered ``[A, B, AB_0, ..., AB_{D-1}]``.

        Returns:
            An ``(N (D + 2), D)`` array in physical units.
        """
        blocks = [self.a_unit, self.b_unit, *list(self.ab_unit)]
        return self.space.to_physical(np.concatenate(blocks, axis=0))

    def split_outputs(
        self,
        outputs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split a flat output vector back into ``f_A``, ``f_B`` and ``f_AB``.

        Args:
            outputs: ``(N (D + 2),)`` array of model outputs, in the row order
                produced by :meth:`design_matrix`.

        Returns:
            A tuple ``(f_a, f_b, f_ab)`` with shapes ``(N,)``, ``(N,)`` and
            ``(D, N)``.

        Raises:
            ValueError: If ``outputs`` has the wrong length or is not 1-D.
        """
        flat = np.asarray(outputs, dtype=float)
        if flat.ndim != 1:
            raise ValueError(f"outputs must be 1-dimensional, got {flat.ndim}D")
        if flat.size != self.n_evaluations:
            raise ValueError(
                f"expected {self.n_evaluations} outputs for N={self.n_base}, "
                f"D={self.dimension}; got {flat.size}"
            )
        n = self.n_base
        f_a = flat[:n]
        f_b = flat[n : 2 * n]
        f_ab = flat[2 * n :].reshape(self.dimension, n)
        return f_a, f_b, f_ab


@dataclass(frozen=True, eq=False)
class SobolIndices:
    """Estimated Sobol' indices with optional bootstrap intervals.

    Attributes:
        names: Parameter names, in column order.
        first_order: ``(D,)`` first-order indices ``S1``.
        total_order: ``(D,)`` total-order indices ``ST``.
        first_order_ci: ``(D, 2)`` percentile interval for ``S1``, or ``None``
            when bootstrapping was disabled.
        total_order_ci: ``(D, 2)`` percentile interval for ``ST``, or ``None``.
        variance: Output variance over the pooled ``A`` and ``B`` samples.
        n_base: Base sample size ``N``.
        n_evaluations: Model evaluations consumed, ``N (D + 2)``.
        confidence_level: Nominal coverage of the intervals.
        manifest: Provenance for the sample that produced the estimate.
    """

    names: tuple[str, ...]
    first_order: np.ndarray
    total_order: np.ndarray
    first_order_ci: np.ndarray | None
    total_order_ci: np.ndarray | None
    variance: float
    n_base: int
    n_evaluations: int
    confidence_level: float
    manifest: StudyManifest | None = None

    def ranked(self, use_total: bool = True) -> tuple[str, ...]:
        """Order parameter names from most to least influential.

        Args:
            use_total: Rank by total-order index when ``True``, first-order
                otherwise.

        Returns:
            Parameter names, most influential first.
        """
        values = self.total_order if use_total else self.first_order
        order = np.argsort(-values, kind="stable")
        return tuple(self.names[i] for i in order)

    def interaction_strength(self) -> np.ndarray:
        """Estimate how much of each factor's effect comes from interactions.

        Returns:
            ``(D,)`` array of ``ST - S1``, clipped at zero. Values near zero
            indicate an additive contribution.
        """
        return np.maximum(self.total_order - self.first_order, 0.0)


def saltelli_design(
    space: DesignSpace,
    n_base: int,
    seed: int | SeedRecord | None = None,
) -> SaltelliDesign:
    """Build the Saltelli cross-sampling plan for a design space.

    Args:
        space: The design space to explore.
        n_base: Base sample size ``N``; must be a power of two so the
            underlying Sobol' sequence keeps its balance property.
        seed: Explicit entropy or seed record; ``None`` draws fresh entropy.

    Returns:
        The evaluation plan.

    Raises:
        ValueError: If ``n_base`` is not a positive power of two.
    """
    if not is_power_of_two(n_base):
        raise ValueError(
            f"n_base must be a positive power of two for a Sobol' design, got {n_base}"
        )
    record = new_seed_record(seed)
    dimension = space.dimension
    engine = make_qmc_engine("sobol", 2 * dimension, record.generator())
    joint = np.asarray(engine.random(n_base), dtype=float)
    a_unit = joint[:, :dimension]
    b_unit = joint[:, dimension:]

    ab_unit = np.repeat(a_unit[np.newaxis, :, :], dimension, axis=0)
    for i in range(dimension):
        ab_unit[i, :, i] = b_unit[:, i]

    manifest = StudyManifest(
        seed=record,
        method="saltelli",
        parameter_names=space.names,
        n_samples=n_base * (dimension + 2),
        extra={"n_base": n_base, "estimator": "saltelli2010+jansen1999"},
    )
    return SaltelliDesign(
        space=space,
        n_base=n_base,
        a_unit=a_unit,
        b_unit=b_unit,
        ab_unit=ab_unit,
        manifest=manifest,
    )


def _centre(
    f_a: np.ndarray,
    f_b: np.ndarray,
    f_ab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Subtract the pooled output mean and return the pooled variance.

    Centring matters: the first-order estimator's finite-sample value depends
    on the output's offset, so an uncentred estimate drifts with the model's
    mean. ``scipy.stats.sobol_indices`` centres for the same reason, and this
    package's parity test with it is exact only because of this step.

    Args:
        f_a: ``(N,)`` outputs on the ``A`` sample.
        f_b: ``(N,)`` outputs on the ``B`` sample.
        f_ab: ``(D, N)`` outputs on the cross samples.

    Returns:
        A tuple ``(a, b, ab, variance)`` of the centred arrays and the pooled
        variance of ``A`` and ``B``.
    """
    pooled = np.concatenate([f_a, f_b])
    mean = float(np.mean(pooled))
    return f_a - mean, f_b - mean, f_ab - mean, float(np.var(pooled))


def _first_order(f_a: np.ndarray, f_b: np.ndarray, f_ab: np.ndarray) -> np.ndarray:
    """Saltelli (2010) first-order estimator, unnormalised by variance.

    Args:
        f_a: ``(N,)`` centred outputs on the ``A`` sample.
        f_b: ``(N,)`` centred outputs on the ``B`` sample.
        f_ab: ``(D, N)`` centred outputs on the cross samples.

    Returns:
        ``(D,)`` array of partial variances ``V_i``.
    """
    return np.mean(f_b[np.newaxis, :] * (f_ab - f_a[np.newaxis, :]), axis=1)


def _total_order(f_a: np.ndarray, f_ab: np.ndarray) -> np.ndarray:
    """Jansen (1999) total-order estimator, unnormalised by variance.

    Args:
        f_a: ``(N,)`` centred outputs on the ``A`` sample.
        f_ab: ``(D, N)`` centred outputs on the cross samples.

    Returns:
        ``(D,)`` array of total partial variances ``VT_i``.
    """
    return 0.5 * np.mean((f_a[np.newaxis, :] - f_ab) ** 2, axis=1)


def sobol_indices_from_outputs(
    f_a: np.ndarray,
    f_b: np.ndarray,
    f_ab: np.ndarray,
    names: tuple[str, ...],
    n_bootstrap: int = 0,
    confidence_level: float = 0.95,
    seed: int | SeedRecord | np.random.Generator | None = None,
    manifest: StudyManifest | None = None,
) -> SobolIndices:
    """Estimate Sobol' indices from pre-computed model outputs.

    This is the pure-numerics core: it never calls the model, so it can be
    tested against analytic functions and reused for outputs that were
    computed on a cluster.

    Args:
        f_a: ``(N,)`` outputs on the ``A`` sample.
        f_b: ``(N,)`` outputs on the ``B`` sample.
        f_ab: ``(D, N)`` outputs on the cross samples.
        names: ``D`` parameter names, in column order.
        n_bootstrap: Bootstrap resamples for the confidence intervals; ``0``
            disables them.
        confidence_level: Nominal coverage of the percentile intervals.
        seed: Entropy for the bootstrap resampling.
        manifest: Provenance of the sample, carried through to the result.

    Returns:
        The estimated indices.

    Raises:
        ValueError: If shapes disagree, outputs are non-finite, the output is
            constant (zero variance), or ``confidence_level`` is out of range.
    """
    a = np.asarray(f_a, dtype=float).ravel()
    b = np.asarray(f_b, dtype=float).ravel()
    ab = np.atleast_2d(np.asarray(f_ab, dtype=float))
    if a.size != b.size:
        raise ValueError(f"f_a has {a.size} entries but f_b has {b.size}")
    if ab.shape != (len(names), a.size):
        raise ValueError(
            f"f_ab must have shape ({len(names)}, {a.size}), got {ab.shape}"
        )
    # Safety-critical: a NaN here silently produces plausible-looking indices,
    # so this is a raise, not an assert (python -O strips asserts).
    if not (
        np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(ab))
    ):
        raise ValueError("model outputs contain NaN or inf; indices are undefined")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must lie in (0, 1), got {confidence_level}")

    centred_a, centred_b, centred_ab, variance = _centre(a, b, ab)
    if variance <= 0.0:
        raise ValueError(
            "output variance is zero; Sobol' indices are undefined for a "
            "constant model response"
        )

    first = _first_order(centred_a, centred_b, centred_ab) / variance
    total = _total_order(centred_a, centred_ab) / variance

    first_ci: np.ndarray | None = None
    total_ci: np.ndarray | None = None
    if n_bootstrap > 0:
        first_ci, total_ci = _bootstrap_intervals(
            a, b, ab, n_bootstrap, confidence_level, seed
        )

    require(
        first.shape == (len(names),) and total.shape == (len(names),),
        "index arrays must have one entry per parameter",
        value=(first.shape, total.shape),
    )
    return SobolIndices(
        names=tuple(names),
        first_order=first,
        total_order=total,
        first_order_ci=first_ci,
        total_order_ci=total_ci,
        variance=variance,
        n_base=int(a.size),
        n_evaluations=int(a.size * (len(names) + 2)),
        confidence_level=confidence_level,
        manifest=manifest,
    )


def _bootstrap_intervals(
    f_a: np.ndarray,
    f_b: np.ndarray,
    f_ab: np.ndarray,
    n_bootstrap: int,
    confidence_level: float,
    seed: int | SeedRecord | np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute percentile bootstrap intervals for ``S1`` and ``ST``.

    Rows (base samples) are resampled with replacement, keeping the ``A`` /
    ``B`` / ``AB`` pairing of each row intact.

    Args:
        f_a: ``(N,)`` outputs on the ``A`` sample.
        f_b: ``(N,)`` outputs on the ``B`` sample.
        f_ab: ``(D, N)`` outputs on the cross samples.
        n_bootstrap: Number of resamples.
        confidence_level: Nominal coverage.
        seed: Entropy for the resampling.

    Returns:
        A tuple of ``(D, 2)`` arrays holding the ``S1`` and ``ST`` intervals.
    """
    generator = as_generator(seed)
    n = f_a.size
    dimension = f_ab.shape[0]
    first_samples = np.empty((n_bootstrap, dimension), dtype=float)
    total_samples = np.empty((n_bootstrap, dimension), dtype=float)
    for draw in range(n_bootstrap):
        idx = generator.integers(0, n, size=n)
        a_r, b_r, ab_r, variance = _centre(f_a[idx], f_b[idx], f_ab[:, idx])
        if variance <= 0.0:
            first_samples[draw] = np.nan
            total_samples[draw] = np.nan
            continue
        first_samples[draw] = _first_order(a_r, b_r, ab_r) / variance
        total_samples[draw] = _total_order(a_r, ab_r) / variance

    tail = 100.0 * (1.0 - confidence_level) / 2.0
    quantiles = (tail, 100.0 - tail)
    first_ci = np.nanpercentile(first_samples, quantiles, axis=0).T
    total_ci = np.nanpercentile(total_samples, quantiles, axis=0).T
    return np.ascontiguousarray(first_ci), np.ascontiguousarray(total_ci)


def sobol_analysis(
    space: DesignSpace,
    model: ModelFunction,
    n_base: int,
    seed: int | SeedRecord | None = None,
    n_bootstrap: int = 0,
    confidence_level: float = 0.95,
) -> SobolIndices:
    """Run a complete Sobol' sensitivity analysis of a model.

    Args:
        space: The design space to explore.
        model: Vectorised model, mapping ``(m, D)`` design points to ``(m,)``
            outputs.
        n_base: Base sample size ``N`` (a power of two). The model is called
            once with all ``N (D + 2)`` rows.
        seed: Explicit entropy or seed record; ``None`` draws fresh entropy.
        n_bootstrap: Bootstrap resamples for confidence intervals.
        confidence_level: Nominal coverage of those intervals.

    Returns:
        The estimated indices, carrying the sampling manifest.

    Raises:
        ValueError: If ``n_base`` is invalid or the model returns the wrong
            shape.
    """
    record = new_seed_record(seed)
    # Spawn rather than reuse: the bootstrap must not share a stream with the
    # QMC scramble (research digest section 7).
    bootstrap_rng = record.spawn(2)[1]
    design = saltelli_design(space, n_base, seed=record)
    outputs = np.asarray(model(design.design_matrix()), dtype=float).ravel()
    if outputs.size != design.n_evaluations:
        raise ValueError(
            f"model returned {outputs.size} outputs, expected {design.n_evaluations}"
        )
    f_a, f_b, f_ab = design.split_outputs(outputs)
    return sobol_indices_from_outputs(
        f_a,
        f_b,
        f_ab,
        names=space.names,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=bootstrap_rng,
        manifest=design.manifest,
    )
