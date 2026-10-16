"""Global sensitivity screening: Saltelli/Jansen Sobol indices and PRCC (NSX-96, #12725).

Estimators (inputs are mapped from the unit cube to declared ranges):

* **First order** ``S_i`` - Saltelli (2010): ``mean(f(B) (f(AB_i) - f(A))) / V``.
* **Total** ``ST_i`` - Jansen: ``mean((f(A) - f(AB_i))^2) / (2 V)``.
* ``A, B`` are the two halves of one scrambled Sobol sequence
  (``scipy.stats.qmc``), ``AB_i`` is ``A`` with column block ``i`` taken from
  ``B`` and ``V`` is the sample variance of ``f(A)`` and ``f(B)`` together.
* Intervals are percentile bootstrap over the ``N`` base rows;
  :func:`sobol_convergence` reports index versus ``N``.

**Correlated inputs.**  Ordinary independent indices assume independent inputs.
When any :class:`InputGroup` declares a correlation, :func:`sobol_indices`
refuses ``level="inputs"``; the only routes are ``level="groups"`` (group-wise
indices, each correlated group moves as one block, via a Gaussian copula with
the declared correlation) or a declared transformation of the inputs.

**PRCC** is the rank transform followed by partial correlation, on a Latin
hypercube.  It is a monotonicity screen, not a variance decomposition.

Non-claim: every output is SCREENING evidence on a model under declared input
ranges.  It is never a validated ranking of parameters of real golfers.
Pure numpy and scipy; no engine imports.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from src.shared.python.loop_nullspace.evidence import Array

__all__ = [
    "InputGroup",
    "InputSpace",
    "PrccResult",
    "SensitivityInput",
    "SobolResult",
    "prcc",
    "sobol_convergence",
    "sobol_indices",
]

Model = Callable[[Array], ArrayLike]
BOOT_LEVEL = 0.95
MIN_BOOT = 10
_EPS = 1e-12


@dataclass(frozen=True)
class SensitivityInput:
    """An uncertain input, uniform on ``[low, high]`` before any correlation."""

    name: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SensitivityInput requires a non-empty name")
        if not (np.isfinite(self.low) and np.isfinite(self.high)):
            raise ValueError(f"{self.name}: bounds must be finite")
        if not self.low < self.high:
            raise ValueError(f"{self.name}: requires low < high")


@dataclass(frozen=True)
class InputGroup:
    """Inputs declared correlated; ``correlation`` is the copula correlation."""

    names: tuple[str, ...]
    correlation: Array

    def __post_init__(self) -> None:
        corr = np.array(self.correlation, dtype=np.float64, copy=True)
        k = len(self.names)
        if k < 2:
            raise ValueError("an InputGroup needs at least two inputs")
        if corr.shape != (k, k) or not np.all(np.isfinite(corr)):
            raise ValueError(f"correlation must be a finite ({k}, {k}) matrix")
        if not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1.0):
            raise ValueError("correlation must be symmetric with unit diagonal")
        try:
            np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as exc:
            raise ValueError("correlation must be positive definite") from exc
        corr.flags.writeable = False
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "correlation", corr)

    @property
    def is_correlated(self) -> bool:
        """True when any off-diagonal correlation is non-zero."""
        return bool(np.any(np.abs(self.correlation - np.eye(len(self.names))) > _EPS))


class InputSpace:
    """Declared inputs and correlated groups; maps unit samples to physical ones.

    Raises:
        ValueError: On duplicate names, unknown or repeated group members, or
            an invalid correlation.
    """

    def __init__(
        self,
        inputs: Sequence[SensitivityInput],
        groups: Sequence[InputGroup] = (),
    ) -> None:
        self.inputs = tuple(inputs)
        if not self.inputs:
            raise ValueError("at least one SensitivityInput is required")
        self.input_names = tuple(i.name for i in self.inputs)
        if len(set(self.input_names)) != len(self.input_names):
            raise ValueError("input names must be unique")
        self.groups = tuple(groups)
        position = {n: k for k, n in enumerate(self.input_names)}
        seen: set[str] = set()
        grouped: dict[str, InputGroup] = {}
        for grp in self.groups:
            for name in grp.names:
                if name not in position:
                    raise ValueError(f"group member {name!r} is an unknown input")
                if name in seen:
                    raise ValueError(f"input {name!r} appears twice in groups")
                seen.add(name)
                grouped[name] = grp
        self._blocks: list[tuple[str, Array, Array | None]] = []
        emitted: set[int] = set()
        for k, name in enumerate(self.input_names):
            member_of = grouped.get(name)
            if member_of is None:
                self._blocks.append((name, np.array([k]), None))
            elif id(member_of) not in emitted:
                emitted.add(id(member_of))
                idx = np.array([position[n] for n in member_of.names])
                chol = np.linalg.cholesky(member_of.correlation)
                self._blocks.append(("+".join(member_of.names), idx, chol))
        self._low = np.array([i.low for i in self.inputs])
        self._span = np.array([i.high - i.low for i in self.inputs])

    @property
    def dimension(self) -> int:
        """Number of inputs."""
        return len(self.inputs)

    @property
    def is_correlated(self) -> bool:
        """True when any declared group has a non-trivial correlation."""
        return any(g.is_correlated for g in self.groups)

    @property
    def group_names(self) -> tuple[str, ...]:
        """Block names (a singleton input is its own block)."""
        return tuple(b[0] for b in self._blocks)

    def blocks(self, level: str) -> list[tuple[str, Array]]:
        """``(name, column indices)`` units that are varied together."""
        if level == "inputs":
            return [(n, np.array([k])) for k, n in enumerate(self.input_names)]
        return [(n, idx) for n, idx, _ in self._blocks]

    def to_physical(self, unit: Array) -> Array:
        """Map an ``(n, D)`` unit-cube sample to physical values.

        Correlated groups pass through a Gaussian copula with the declared
        correlation; marginals stay uniform on the declared ranges.
        """
        u = np.array(unit, dtype=np.float64, copy=True)
        for _, idx, chol in self._blocks:
            if chol is not None:
                z = ndtri(np.clip(u[:, idx], _EPS, 1.0 - _EPS)) @ chol.T
                u[:, idx] = ndtr(z)
        return np.asarray(self._low + self._span * u)


def _evaluate(model: Model, space: InputSpace, unit: Array) -> Array:
    out = np.asarray(model(space.to_physical(unit)), dtype=np.float64)
    if out.shape != (unit.shape[0],):
        raise ValueError(f"model must return shape ({unit.shape[0]},), got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError("model returned non-finite values; screening refuses them")
    return out


def _interval(samples: Array) -> Array:
    lo = (1.0 - BOOT_LEVEL) / 2.0
    return np.asarray(np.quantile(samples, [lo, 1.0 - lo], axis=0)).T


@dataclass(frozen=True)
class SobolResult:
    """Sobol indices with bootstrap intervals (SCREENING evidence)."""

    names: tuple[str, ...]
    level: str
    n: int
    n_evaluations: int
    first_order: Array
    total: Array
    first_order_ci: Array
    total_ci: Array
    seed: int


def _sobol_estimates(fa: Array, fb: Array, fab: Array) -> tuple[Array, Array]:
    var = float(np.var(np.concatenate([fa, fb])))
    if var <= 0.0:
        raise ValueError("model output has zero variance; indices are undefined")
    first = np.mean(fb[None, :] * (fab - fa[None, :]), axis=1) / var
    total = 0.5 * np.mean((fa[None, :] - fab) ** 2, axis=1) / var
    return first, total


def sobol_indices(
    model: Model,
    space: InputSpace,
    *,
    n: int,
    seed: int,
    level: str = "inputs",
    n_boot: int = 200,
) -> SobolResult:
    """Saltelli first-order and Jansen total Sobol indices.

    Args:
        model: ``(n, D) physical samples -> (n,)`` scalar output.
        space: Declared inputs and correlated groups.
        n: Base sample size, a power of two (cost ``n (blocks + 2)`` calls).
        seed: Scrambling and bootstrap seed (fixed seed is reproducible).
        level: ``"inputs"`` (ordinary) or ``"groups"`` (group-wise).
        n_boot: Bootstrap resamples for the intervals (at least 10).

    Raises:
        ValueError: ``n`` not a power of two; bad ``level``; ordinary indices
            requested for declared-correlated inputs (use ``level="groups"``);
            non-finite or mis-shaped model output; zero output variance.

    Postconditions:
        Arrays have one entry per unit of ``level``; deterministic in ``seed``.
    """
    if level not in ("inputs", "groups"):
        raise ValueError(f"level must be 'inputs' or 'groups', got {level!r}")
    if level == "inputs" and space.is_correlated:
        raise ValueError(
            "inputs are declared correlated: ordinary independent Sobol indices "
            "are refused; use level='groups' (grouped indices) or a declared "
            "transformation to independent inputs"
        )
    if n < 2 or n & (n - 1):
        raise ValueError(f"n must be a power of two (>= 2), got {n}")
    if n_boot < MIN_BOOT:
        raise ValueError(f"n_boot must be at least {MIN_BOOT}, got {n_boot}")
    dim = space.dimension
    sampler = qmc.Sobol(d=2 * dim, scramble=True, seed=seed)
    draw = sampler.random_base2(int(np.log2(n)))
    a, b = draw[:, :dim], draw[:, dim:]
    fa, fb = _evaluate(model, space, a), _evaluate(model, space, b)
    blocks = space.blocks(level)
    fab = np.empty((len(blocks), n))
    for k, (_, idx) in enumerate(blocks):
        mixed = a.copy()
        mixed[:, idx] = b[:, idx]
        fab[k] = _evaluate(model, space, mixed)
    first, total = _sobol_estimates(fa, fb, fab)
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, n, n) for _ in range(n_boot)]
    est = [_sobol_estimates(fa[r], fb[r], fab[:, r]) for r in boots]
    return SobolResult(
        names=tuple(name for name, _ in blocks),
        level=level,
        n=n,
        n_evaluations=n * (len(blocks) + 2),
        first_order=first,
        total=total,
        first_order_ci=_interval(np.array([e[0] for e in est])),
        total_ci=_interval(np.array([e[1] for e in est])),
        seed=seed,
    )


def sobol_convergence(
    model: Model,
    space: InputSpace,
    *,
    ns: Sequence[int],
    seed: int,
    level: str = "inputs",
    n_boot: int = 200,
) -> list[SobolResult]:
    """Index versus ``N``: one :class:`SobolResult` per increasing ``N``.

    The same ``seed`` is used for each size, so the sizes are nested prefixes of
    one scrambled Sobol sequence.  Bootstrap intervals give the convergence
    evidence; a ranking is only stable once the intervals separate.

    Raises:
        ValueError: If ``ns`` is empty or not strictly increasing.
    """
    sizes = list(ns)
    if not sizes or any(b <= a for a, b in zip(sizes, sizes[1:], strict=False)):
        raise ValueError("ns must be a non-empty strictly increasing sequence")
    return [
        sobol_indices(model, space, n=m, seed=seed, level=level, n_boot=n_boot)
        for m in sizes
    ]


@dataclass(frozen=True)
class PrccResult:
    """Partial rank correlation coefficients with bootstrap intervals."""

    names: tuple[str, ...]
    n: int
    prcc: Array
    ci: Array
    seed: int


def _prcc_from_ranks(ranks: Array) -> Array:
    corr = np.corrcoef(ranks, rowvar=False)
    try:
        prec = np.linalg.inv(corr)
    except np.linalg.LinAlgError as exc:
        raise ValueError("rank correlation matrix is singular; PRCC undefined") from exc
    d = prec.shape[0] - 1
    return np.asarray(-prec[:d, d] / np.sqrt(prec[:d, :d].diagonal() * prec[d, d]))


def prcc(
    model: Model,
    space: InputSpace,
    *,
    n: int,
    seed: int,
    n_boot: int = 200,
) -> PrccResult:
    """Rank-transform plus partial correlation on a Latin hypercube.

    Correlated groups are sampled through their copula; PRCC conditions on the
    other inputs, so it stays defined, but it remains a monotonicity screen.

    Raises:
        ValueError: ``n`` too small (needs more rows than inputs + 2), bad
            ``n_boot``, bad model output, or a singular rank correlation.
    """
    dim = space.dimension
    if n < dim + 3:
        raise ValueError(f"n must exceed dimension + 2 ({dim + 2}), got {n}")
    if n_boot < MIN_BOOT:
        raise ValueError(f"n_boot must be at least {MIN_BOOT}, got {n_boot}")
    unit = qmc.LatinHypercube(d=dim, seed=seed).random(n)
    y = _evaluate(model, space, unit)
    x = space.to_physical(unit)
    ranks = np.column_stack(
        [stats.rankdata(x[:, k]) for k in range(dim)] + [stats.rankdata(y)]
    )
    est = _prcc_from_ranks(ranks)
    rng = np.random.default_rng(seed)
    boots = np.array(
        [_prcc_from_ranks(ranks[rng.integers(0, n, n)]) for _ in range(n_boot)]
    )
    return PrccResult(space.input_names, n, est, _interval(boots), seed)
