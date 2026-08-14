"""Ranking candidate designs with their uncertainty attached.

The question a design tool has to answer is "which wedge is better, and how
sure are we?" -- and the honest answer is frequently "we cannot tell from this
many runs". Reporting a point estimate hides that, so every comparison here
carries an interval and a probability.

Two entry points, one shared ranking core:

- :func:`compare_designs` when each design has been *evaluated* several times
  (replicate seeds, replicate sand states, replicate swing conditions). The
  uncertainty is a non-parametric bootstrap of the mean.
- :func:`compare_predicted_designs` when each design has been *predicted* by a
  surrogate, which supplies a mean and a standard deviation directly.

``probability_best`` is the Monte-Carlo probability that a design is the best
of the set, given the sampled uncertainty -- not a p-value, and it sums to one
across the set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from src.shared.python.core.contracts import require

from .manifest import StudyManifest
from .rng import SeedRecord, new_seed_record

__all__ = [
    "DesignComparison",
    "compare_designs",
    "compare_predicted_designs",
]

_DEFAULT_DRAWS = 2000


@dataclass(frozen=True, eq=False)
class DesignComparison:
    """A ranked set of candidate designs with uncertainty.

    Attributes:
        names: Design names, in input order.
        mean: ``(n,)`` central estimate of each design's objective.
        std_error: ``(n,)`` standard error of that estimate.
        ci_low: ``(n,)`` lower confidence bound.
        ci_high: ``(n,)`` upper confidence bound.
        rank: ``(n,)`` integer rank, ``0`` for the best design.
        probability_best: ``(n,)`` probability each design is the best of the
            set; sums to one.
        probability_better: ``(n, n)`` matrix where entry ``(i, j)`` is the
            probability design ``i`` beats design ``j``.
        lower_is_better: Objective direction.
        confidence_level: Nominal coverage of the intervals.
        n_draws: Monte-Carlo draws behind the probabilities.
        manifest: Provenance, including the entropy used for the draws.
    """

    names: tuple[str, ...]
    mean: np.ndarray
    std_error: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    rank: np.ndarray
    probability_best: np.ndarray
    probability_better: np.ndarray
    lower_is_better: bool
    confidence_level: float
    n_draws: int
    manifest: StudyManifest | None = None

    @property
    def best(self) -> str:
        """Name of the top-ranked design.

        Returns:
            The design with rank ``0``.
        """
        return self.names[int(np.argmin(self.rank))]

    def ordered(self) -> tuple[str, ...]:
        """List design names from best to worst.

        Returns:
            Names in rank order.
        """
        order = np.argsort(self.rank, kind="stable")
        return tuple(self.names[i] for i in order)

    def indistinguishable_from_best(self) -> tuple[str, ...]:
        """Names whose confidence interval overlaps the best design's.

        A non-empty result (beyond the winner itself) means the study has not
        separated the candidates and reporting a single winner would overstate
        what was measured.

        Returns:
            Names of the designs that overlap the leader, including it.
        """
        leader = int(np.argmin(self.rank))
        overlap = (self.ci_low <= self.ci_high[leader]) & (
            self.ci_high >= self.ci_low[leader]
        )
        return tuple(
            name for name, flag in zip(self.names, overlap, strict=True) if flag
        )

    def is_separated(self) -> bool:
        """Report whether the leader is distinguishable from every rival.

        Returns:
            ``True`` when only the leader's own interval overlaps its own.
        """
        return len(self.indistinguishable_from_best()) == 1


def _rank_from_draws(
    draws: np.ndarray,
    lower_is_better: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Monte-Carlo draws into win probabilities.

    Args:
        draws: ``(n_draws, n_designs)`` sampled objective values.
        lower_is_better: Objective direction.

    Returns:
        A tuple ``(probability_best, probability_better)`` with shapes
        ``(n_designs,)`` and ``(n_designs, n_designs)``.
    """
    oriented = draws if lower_is_better else -draws
    winners = np.argmin(oriented, axis=1)
    n_designs = draws.shape[1]
    counts = np.bincount(winners, minlength=n_designs).astype(float)
    probability_best = counts / draws.shape[0]

    better = (oriented[:, :, None] < oriented[:, None, :]).mean(axis=0)
    np.fill_diagonal(better, 0.0)
    return probability_best, better


def _build_comparison(
    names: tuple[str, ...],
    mean: np.ndarray,
    std_error: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    draws: np.ndarray,
    lower_is_better: bool,
    confidence_level: float,
    manifest: StudyManifest | None,
) -> DesignComparison:
    """Assemble a :class:`DesignComparison` from its parts.

    Args:
        names: Design names.
        mean: ``(n,)`` central estimates.
        std_error: ``(n,)`` standard errors.
        ci_low: ``(n,)`` lower bounds.
        ci_high: ``(n,)`` upper bounds.
        draws: ``(n_draws, n)`` Monte-Carlo draws.
        lower_is_better: Objective direction.
        confidence_level: Nominal coverage.
        manifest: Provenance to attach.

    Returns:
        The assembled comparison.
    """
    probability_best, probability_better = _rank_from_draws(draws, lower_is_better)
    oriented_mean = mean if lower_is_better else -mean
    order = np.argsort(oriented_mean, kind="stable")
    rank = np.empty(mean.size, dtype=int)
    rank[order] = np.arange(mean.size)

    require(
        abs(float(probability_best.sum()) - 1.0) < 1e-9,
        "probability_best must sum to one",
        value=float(probability_best.sum()),
    )
    return DesignComparison(
        names=names,
        mean=mean,
        std_error=std_error,
        ci_low=ci_low,
        ci_high=ci_high,
        rank=rank,
        probability_best=probability_best,
        probability_better=probability_better,
        lower_is_better=lower_is_better,
        confidence_level=confidence_level,
        n_draws=int(draws.shape[0]),
        manifest=manifest,
    )


def _validate_names(names: tuple[str, ...], n_designs: int) -> tuple[str, ...]:
    """Check design names against the data.

    Args:
        names: Candidate design names.
        n_designs: Expected count.

    Returns:
        The names as a tuple.

    Raises:
        ValueError: If the count is wrong or the names are not unique.
    """
    tidy = tuple(names)
    if len(tidy) != n_designs:
        raise ValueError(f"got {len(tidy)} names for {n_designs} designs")
    if len(set(tidy)) != len(tidy):
        raise ValueError("design names must be unique")
    return tidy


def compare_designs(
    names: tuple[str, ...],
    observations: np.ndarray,
    lower_is_better: bool = True,
    n_bootstrap: int = _DEFAULT_DRAWS,
    confidence_level: float = 0.95,
    seed: int | SeedRecord | None = None,
) -> DesignComparison:
    """Rank designs from replicated evaluations.

    Args:
        names: One name per design.
        observations: ``(n_designs, n_replicates)`` objective values. Every
            design must have the same number of replicates so the comparison
            is balanced.
        lower_is_better: Objective direction.
        n_bootstrap: Bootstrap resamples of the replicate means.
        confidence_level: Nominal coverage of the percentile intervals.
        seed: Explicit entropy or seed record; ``None`` draws fresh entropy.

    Returns:
        The ranked comparison.

    Raises:
        ValueError: If the shapes disagree, the data is non-finite, or there
            are fewer than two replicates per design.
    """
    data = np.atleast_2d(np.asarray(observations, dtype=float))
    if data.ndim != 2:
        raise ValueError(f"observations must be 2-dimensional, got {data.ndim}D")
    if data.shape[1] < 2:
        raise ValueError(
            "each design needs at least two replicates to estimate uncertainty; "
            f"got {data.shape[1]}"
        )
    # Safety-critical: a NaN replicate would rank a design by accident.
    if not np.all(np.isfinite(data)):
        raise ValueError("observations contain NaN or inf")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must lie in (0, 1), got {confidence_level}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}")

    tidy_names = _validate_names(names, data.shape[0])
    record = new_seed_record(seed)
    generator = record.generator()

    n_designs, n_replicates = data.shape
    index = generator.integers(
        0, n_replicates, size=(n_bootstrap, n_designs, n_replicates)
    )
    resampled = np.take_along_axis(
        np.broadcast_to(data, (n_bootstrap, n_designs, n_replicates)),
        index,
        axis=2,
    )
    draws = resampled.mean(axis=2)

    tail = 100.0 * (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.percentile(draws, [tail, 100.0 - tail], axis=0)
    manifest = StudyManifest(
        seed=record,
        method="design-comparison-bootstrap",
        parameter_names=(),
        n_samples=int(data.size),
        extra={
            "design_names": list(tidy_names),
            "n_bootstrap": n_bootstrap,
            "n_replicates": n_replicates,
        },
    )
    return _build_comparison(
        names=tidy_names,
        mean=data.mean(axis=1),
        std_error=data.std(axis=1, ddof=1) / np.sqrt(n_replicates),
        ci_low=ci_low,
        ci_high=ci_high,
        draws=draws,
        lower_is_better=lower_is_better,
        confidence_level=confidence_level,
        manifest=manifest,
    )


def compare_predicted_designs(
    names: tuple[str, ...],
    mean: np.ndarray,
    std: np.ndarray,
    lower_is_better: bool = True,
    n_draws: int = _DEFAULT_DRAWS,
    confidence_level: float = 0.95,
    seed: int | SeedRecord | None = None,
) -> DesignComparison:
    """Rank designs from surrogate predictions.

    Args:
        names: One name per design.
        mean: ``(n,)`` predicted objective values.
        std: ``(n,)`` predictive standard deviations, typically from
            :meth:`~bunkershot3d.study.surrogate.GaussianProcess.predict`.
        lower_is_better: Objective direction.
        n_draws: Monte-Carlo draws behind the probabilities.
        confidence_level: Nominal coverage of the intervals.
        seed: Explicit entropy or seed record; ``None`` draws fresh entropy.

    Returns:
        The ranked comparison. Intervals are the Gaussian
        ``mean +/- z * std``; the draws are independent per design, which
        ignores surrogate correlation between nearby designs and is therefore
        mildly conservative about which design wins.

    Raises:
        ValueError: If the shapes disagree, values are non-finite, or a
            standard deviation is negative.
    """
    centre = np.asarray(mean, dtype=float).ravel()
    spread = np.asarray(std, dtype=float).ravel()
    if centre.size != spread.size:
        raise ValueError(f"mean has {centre.size} entries but std has {spread.size}")
    if not (np.all(np.isfinite(centre)) and np.all(np.isfinite(spread))):
        raise ValueError("predictions contain NaN or inf")
    if np.any(spread < 0.0):
        raise ValueError("standard deviations must be non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must lie in (0, 1), got {confidence_level}")
    if n_draws < 1:
        raise ValueError(f"n_draws must be positive, got {n_draws}")

    tidy_names = _validate_names(names, centre.size)
    record = new_seed_record(seed)
    generator = record.generator()
    draws = centre + spread * generator.standard_normal((n_draws, centre.size))

    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    manifest = StudyManifest(
        seed=record,
        method="design-comparison-surrogate",
        parameter_names=(),
        n_samples=int(centre.size),
        extra={"design_names": list(tidy_names), "n_draws": n_draws},
    )
    return _build_comparison(
        names=tidy_names,
        mean=centre,
        std_error=spread,
        ci_low=centre - z * spread,
        ci_high=centre + z * spread,
        draws=draws,
        lower_is_better=lower_is_better,
        confidence_level=confidence_level,
        manifest=manifest,
    )
