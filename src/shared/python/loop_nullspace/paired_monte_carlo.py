"""Paired nonlinear Monte Carlo campaign runner (NSX-91, #12720; rules R10, R13, V7).

``run_campaign`` evaluates every :class:`Strategy` on the same
:class:`~paired_mc_spec.TrialDraw` of each trial (common random numbers) and then
on an independent hold-out seed set.  Nothing is dropped: a rollout that raises a
declared numerical error, returns non-finite or mis-shaped outcomes, misses the
ball (:class:`TrialMiss`) or leaves the admissible set
(:class:`TrialNonAdmissible`) becomes a :class:`TrialRecord` with its status and
reason, and :func:`tally` counts every status.  Programming errors (``TypeError``,
``AssertionError`` ...) propagate.

Open-loop and feedback-assisted strategies keep their own :class:`ReplayMode`
(R10).  The integrator and step of each strategy must differ from the
generator's when a generator is declared (R13).  ``batch_size`` only chunks the
loop: draws are pure functions of ``(seed set, index, disturbance)``, so serial
and batched runs are bit-identical.

Non-claim: Monte Carlo of a model is a statement about that model under the
declared noise, not about golfers; a failure rate depends on the declared
admissibility and noise, and is not a physical probability.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.paired_mc_spec import (
    SEED_SETS,
    CampaignSpec,
    IntegratorSignature,
    NoiseComparison,
    NoiseRelation,
    TrialDraw,
    compare_noise,
    require_independent_integrator,
)
from src.shared.python.loop_nullspace.replay_spec import ReplayMode

__all__ = [
    "RECOVERABLE_FAILURES",
    "CampaignResult",
    "FailureTally",
    "SeedSetResult",
    "Strategy",
    "TrialDraw",
    "TrialMiss",
    "TrialNonAdmissible",
    "TrialRecord",
    "TrialStatus",
    "run_campaign",
    "run_seed_set",
    "tally",
    "wilson_interval",
]

#: Declared numerical failures of a rollout; ``ArithmeticError`` covers
#: ``FloatingPointError`` and ``ZeroDivisionError``.
RECOVERABLE_FAILURES = (ValueError, RuntimeError, ArithmeticError)
TrialRunner = Callable[[TrialDraw], ArrayLike]
NOISE_FLAG = "noise_mismatched_to_generator"


class TrialStatus(str, Enum):
    """Outcome class of one rollout (every class is counted, none dropped)."""

    OK = "ok"
    FAILED = "failed"
    MISSED = "missed"
    NON_ADMISSIBLE = "non_admissible"


class TrialMiss(Exception):
    """The rollout ran but produced no physical impact event (V2)."""


class TrialNonAdmissible(Exception):
    """The rollout left the admissible set (slip, rank loss, bound violation)."""


@dataclass(frozen=True)
class Strategy:
    """A named strategy: a runner of one :class:`TrialDraw`, its mode and integrator.

    ``runner(draw)`` returns the outcome vector (one value per
    ``CampaignSpec.outcomes``) or raises one of the declared failures.
    ``hypotheses`` are the allocation priors the strategy is conditioned on.
    """

    name: str
    runner: TrialRunner
    mode: ReplayMode
    integrator: IntegratorSignature
    hypotheses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a strategy needs a name")
        if not callable(self.runner):
            raise TypeError("runner must be callable")
        if not isinstance(self.mode, ReplayMode):
            raise TypeError("mode must be a ReplayMode")
        if not isinstance(self.integrator, IntegratorSignature):
            raise TypeError("integrator must be an IntegratorSignature")


@dataclass(frozen=True)
class TrialRecord:
    """One retained trial; ``outcomes`` is all-NaN unless ``status`` is ``OK``."""

    index: int
    status: TrialStatus
    outcomes: Array
    reason: str | None = None


@dataclass(frozen=True)
class SeedSetResult:
    """Records of every strategy on one seed set, in trial order."""

    seed_set: str
    records: dict[str, tuple[TrialRecord, ...]]
    modes: dict[str, ReplayMode] = field(default_factory=dict)
    hypotheses: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def outcome_matrix(self, strategy: str) -> Array:
        """Outcomes of the ``OK`` trials of ``strategy``, ``(n_ok, n_outcomes)``."""
        rows = [
            r.outcomes for r in self.records[strategy] if r.status is TrialStatus.OK
        ]
        if not rows:
            width = len(self.records[strategy][0].outcomes)
            return np.empty((0, width))
        return np.array(rows)


@dataclass(frozen=True)
class CampaignResult:
    """Paired and hold-out results plus the honesty flags of the campaign."""

    spec: CampaignSpec
    paired: SeedSetResult
    holdout: SeedSetResult
    flags: tuple[str, ...] = ()
    noise_mismatch: NoiseComparison | None = None


@dataclass(frozen=True)
class FailureTally:
    """Counts of every status and the failure rate with a Wilson interval."""

    n_trials: int
    counts: dict[TrialStatus, int]
    failure_rate: float
    failure_rate_ci: tuple[float, float]
    reasons: tuple[tuple[str, int], ...] = field(default=())


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for ``k`` events in ``n`` trials.

    Valid at ``k = 0`` and ``k = n`` (unlike the Wald interval).  Pre:
    ``0 <= k <= n``, ``n >= 1``, ``0 < level < 1``.
    """
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n >= 1, got k={k}, n={n}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    z = NormalDist().inv_cdf(0.5 * (1.0 + level))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def tally(records: Sequence[TrialRecord], level: float = 0.95) -> FailureTally:
    """Count every status; the failure rate is every non-``OK`` trial."""
    if len(records) == 0:
        raise ValueError("no trial records to tally")
    counts = dict.fromkeys(TrialStatus, 0)
    for rec in records:
        counts[rec.status] += 1
    bad = len(records) - counts[TrialStatus.OK]
    reasons = Counter(r.reason for r in records if r.reason)
    ordered = tuple(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])))
    return FailureTally(
        n_trials=len(records),
        counts=counts,
        failure_rate=bad / len(records),
        failure_rate_ci=wilson_interval(bad, len(records), level),
        reasons=ordered,
    )


def _record(
    index: int, status: TrialStatus, n_out: int, reason: str | None
) -> TrialRecord:
    return TrialRecord(index, status, np.full(n_out, np.nan), reason)


def _run_trial(strategy: Strategy, draw: TrialDraw, n_out: int) -> TrialRecord:
    try:
        raw = strategy.runner(draw)
    except TrialMiss as exc:
        return _record(draw.index, TrialStatus.MISSED, n_out, str(exc) or "missed")
    except TrialNonAdmissible as exc:
        reason = str(exc) or "non-admissible"
        return _record(draw.index, TrialStatus.NON_ADMISSIBLE, n_out, reason)
    except RECOVERABLE_FAILURES as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return _record(draw.index, TrialStatus.FAILED, n_out, reason)
    out = np.asarray(raw, dtype=np.float64).reshape(-1)
    if out.shape != (n_out,):
        reason = f"runner returned {out.size} outcomes, expected {n_out}"
        return _record(draw.index, TrialStatus.FAILED, n_out, reason)
    if not np.isfinite(out).all():
        return _record(draw.index, TrialStatus.FAILED, n_out, "non-finite outcome")
    return TrialRecord(draw.index, TrialStatus.OK, out, None)


def _check_strategies(spec: CampaignSpec, strategies: Sequence[Strategy]) -> None:
    if len(strategies) == 0:
        raise ValueError("at least one strategy is required")
    if not all(isinstance(s, Strategy) for s in strategies):
        raise TypeError("strategies must be Strategy instances")
    names = [s.name for s in strategies]
    if len(set(names)) != len(names):
        raise ValueError("strategy names must be unique")
    if spec.generator is not None:
        for strat in strategies:
            require_independent_integrator(strat.integrator, spec.generator.integrator)


def run_seed_set(
    spec: CampaignSpec,
    strategies: Sequence[Strategy],
    seed_set: str,
    *,
    batch_size: int = 1,
) -> SeedSetResult:
    """Run every strategy on every trial of one seed set.

    Postconditions: each strategy has exactly ``trials_in(seed_set)`` records in
    index order; draws are identical across strategies and across ``batch_size``.
    """
    if seed_set not in SEED_SETS:
        raise ValueError(f"seed_set must be one of {SEED_SETS}, got {seed_set!r}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    _check_strategies(spec, strategies)
    n_trials, n_out = spec.trials_in(seed_set), len(spec.outcomes)
    records: dict[str, list[TrialRecord]] = {s.name: [] for s in strategies}
    for start in range(0, n_trials, batch_size):
        stop = min(n_trials, start + batch_size)
        draws = [TrialDraw(spec, seed_set, i) for i in range(start, stop)]
        for strat in strategies:
            records[strat.name].extend(_run_trial(strat, d, n_out) for d in draws)
    return SeedSetResult(
        seed_set,
        {k: tuple(v) for k, v in records.items()},
        modes={s.name: s.mode for s in strategies},
        hypotheses={s.name: s.hypotheses for s in strategies},
    )


def run_campaign(
    spec: CampaignSpec, strategies: Sequence[Strategy], *, batch_size: int = 1
) -> CampaignResult:
    """Paired seed set plus independent hold-out seed set, with R13 flags.

    Raises ``ValueError`` for an inverse crime (a strategy sharing the
    generator's integrator or step).  A generator with *different* noise is
    accepted and flagged ``noise_mismatched_to_generator`` with the differences.
    """
    _check_strategies(spec, strategies)
    flags: list[str] = []
    mismatch = None
    if spec.generator is not None:
        mismatch = compare_noise(spec.disturbances, spec.generator.noise)
        if mismatch.relation is NoiseRelation.MISMATCHED:
            flags.append(NOISE_FLAG)
    return CampaignResult(
        spec=spec,
        paired=run_seed_set(spec, strategies, "paired", batch_size=batch_size),
        holdout=run_seed_set(spec, strategies, "holdout", batch_size=batch_size),
        flags=tuple(flags),
        noise_mismatch=mismatch,
    )
