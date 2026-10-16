"""Ranking stability: alternative priors and held-out seeds (NSX-95, #12724).

A ranking is a prediction conditional on the declared priors, so it is reported
together with the ways it can fail:

* :func:`rank_under_priors` repeats the whole comparison in each alternative
  context (a *prior* is a full :class:`~.strategy_arms.ComparisonContext`: other
  noise amplitudes, mechanisms declared or not, a different baseline swing) and
  lists every pair of arms whose order flips between two priors as a
  :class:`Reversal`.  Nothing is averaged away: a flip is a finding.
* :func:`heldout_stability` re-checks the linear ranking against the NSX-91
  paired Monte Carlo campaign run on independent hold-out seeds, using the
  campaign machinery (:func:`~.paired_monte_carlo.run_campaign`,
  :func:`~.paired_statistics.compare_strategies`) unchanged.  A pair is
  ``CONFIRMED`` only when the hold-out reproduces the order with a standard
  deviation difference whose interval excludes zero, ``REVERSED`` when the
  hold-out contradicts it the same way, and ``INCONCLUSIVE`` otherwise.

Non-claim: conditional on the declared priors; not coaching advice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.shared.python.loop_nullspace.paired_mc_spec import CampaignSpec
from src.shared.python.loop_nullspace.paired_monte_carlo import (
    CampaignResult,
    Strategy,
    run_campaign,
    tally,
)
from src.shared.python.loop_nullspace.paired_statistics import (
    compare_strategies,
    holdout_consistent,
)
from src.shared.python.loop_nullspace.strategy_arms import ArmSpec, ComparisonContext
from src.shared.python.loop_nullspace.strategy_campaign import (
    ComparisonMode,
    ComparisonReport,
    MatchSettings,
    run_comparison,
)
from src.shared.python.loop_nullspace.strategy_evaluators import evaluate_arms
from src.shared.python.loop_nullspace.strategy_spec import NON_CLAIM

__all__ = [
    "HeldOutCheck",
    "PairCheck",
    "PriorRanking",
    "Reversal",
    "StabilityStatus",
    "heldout_stability",
    "rank_under_priors",
]


@dataclass(frozen=True)
class Reversal:
    """Two arms whose order differs between two priors."""

    first_prior: str
    second_prior: str
    better_in_first: str
    better_in_second: str


@dataclass(frozen=True)
class PriorRanking:
    """One report per prior and the reversals between them."""

    reports: dict[str, ComparisonReport]
    reversals: tuple[Reversal, ...]
    non_claim: str = NON_CLAIM

    @property
    def stable(self) -> bool:
        """No pair of arms changes order between priors."""
        return not self.reversals


def _reversals(reports: Mapping[str, ComparisonReport]) -> tuple[Reversal, ...]:
    out: list[Reversal] = []
    names = list(reports)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            a, b = reports[first], reports[second]
            rank_a = {r.arm.label: r.rank for r in a.ranked}
            rank_b = {r.arm.label: r.rank for r in b.ranked}
            shared = [lab for lab in rank_a if lab in rank_b]
            for j, x in enumerate(shared):
                for y in shared[j + 1 :]:
                    d_a = rank_a[x] - rank_a[y]  # type: ignore[operator]
                    d_b = rank_b[x] - rank_b[y]  # type: ignore[operator]
                    if d_a * d_b < 0:
                        better_first, better_second = (x, y) if d_a < 0 else (y, x)
                        out.append(Reversal(first, second, better_first, better_second))
    return tuple(out)


def rank_under_priors(
    specs: Sequence[ArmSpec],
    priors: Mapping[str, ComparisonContext],
    mode: ComparisonMode,
    settings: MatchSettings | None = None,
) -> PriorRanking:
    """Evaluate and rank ``specs`` in every prior context, then list reversals.

    Preconditions (``ValueError``): at least two priors.  ``settings`` defaults to
    each prior's own clubhead speed (matched effort needs explicit settings).
    Postcondition: ``reversals`` contains every pair ranked in both of two priors
    whose strict order is opposite (ties are not reversals).
    """
    if len(priors) < 2:
        raise ValueError("rank_under_priors needs at least two priors")
    reports: dict[str, ComparisonReport] = {}
    for name, ctx in priors.items():
        arms = evaluate_arms(specs, ctx)
        cfg = settings or MatchSettings(reference_chs_mps=ctx.chs_mps)
        reports[name] = run_comparison(arms, mode, cfg)
    return PriorRanking(reports=reports, reversals=_reversals(reports))


# ----------------------------------------------------------------- held-out
class StabilityStatus(str, Enum):
    """Verdict on one ordered pair against the hold-out seeds."""

    CONFIRMED = "confirmed"
    INCONCLUSIVE = "inconclusive"
    REVERSED = "reversed"


@dataclass(frozen=True)
class PairCheck:
    """``better`` is ranked above ``worse`` by the linear model.

    ``*_sd_diff`` is ``SD(better) - SD(worse)`` of the Monte Carlo outcome (the
    linear order predicts it negative) on the paired and the hold-out seed sets.
    """

    better: str
    worse: str
    paired_sd_diff: float | None
    holdout_sd_diff: float | None
    holdout_agrees: bool
    status: StabilityStatus


@dataclass(frozen=True)
class HeldOutCheck:
    """Pair checks, per-arm hold-out failure rates and the campaign flags."""

    outcome: str
    pairs: tuple[PairCheck, ...]
    failure_rates: dict[str, float]
    flags: tuple[str, ...]
    non_claim: str = NON_CLAIM

    @property
    def reversals(self) -> tuple[PairCheck, ...]:
        return tuple(p for p in self.pairs if p.status is StabilityStatus.REVERSED)

    @property
    def stable(self) -> bool:
        """Every pair confirmed on the hold-out (no reversal, none inconclusive)."""
        return all(p.status is StabilityStatus.CONFIRMED for p in self.pairs)


def _check_pair(
    spec: CampaignSpec,
    camp: CampaignResult,
    better: str,
    worse: str,
    outcome: str,
    **kw: Any,
) -> PairCheck:
    paired = compare_strategies(spec, camp.paired, better, worse, **kw)
    hold = compare_strategies(spec, camp.holdout, better, worse, **kw)
    first, second = paired.outcomes.get(outcome), hold.outcomes.get(outcome)
    agrees = holdout_consistent(paired, hold).get(outcome, False)
    status = StabilityStatus.INCONCLUSIVE
    if second is not None:
        low, high = second.sd_diff_ci
        if second.sd_diff < 0.0 and high < 0.0 and agrees:
            status = StabilityStatus.CONFIRMED
        elif second.sd_diff > 0.0 and low > 0.0:
            status = StabilityStatus.REVERSED
    return PairCheck(
        better,
        worse,
        None if first is None else first.sd_diff,
        None if second is None else second.sd_diff,
        bool(agrees),
        status,
    )


def heldout_stability(
    report: ComparisonReport,
    strategies: Mapping[str, Strategy],
    spec: CampaignSpec,
    *,
    outcome: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> HeldOutCheck:
    """Check the linear ranking of ``report`` against paired and hold-out MC.

    ``strategies`` maps an arm label to the NSX-91 :class:`Strategy` that runs
    that arm in an independent forward model (the inverse-crime check of the
    campaign applies).  Arms without a strategy are not checked.

    Preconditions (``ValueError``): two or more ranked arms with a strategy, each
    ``Strategy.name`` equal to its label, ``outcome`` a channel of ``spec``.
    Postconditions: every pair of differently ranked checked arms is reported,
    reversals included; the per-arm failure rate is the hold-out tally.
    """
    names = [r.arm.label for r in report.ranked if r.arm.label in strategies]
    ranks = {r.arm.label: r.rank for r in report.ranked}
    if len(names) < 2:
        raise ValueError(
            "heldout_stability needs two ranked arms with a Monte Carlo strategy"
        )
    for name in names:
        if strategies[name].name != name:
            raise ValueError(
                f"strategy for {name!r} is named {strategies[name].name!r}"
            )
    if outcome not in [o.name for o in spec.outcomes]:
        raise ValueError(f"unknown outcome {outcome!r}")
    camp = run_campaign(spec, [strategies[n] for n in names])
    kw = {"n_boot": n_boot, "seed": seed}
    pairs = tuple(
        _check_pair(spec, camp, better, worse, outcome, **kw)
        for i, better in enumerate(names)
        for worse in names[i + 1 :]
        if ranks[better] != ranks[worse]
    )
    return HeldOutCheck(
        outcome=outcome,
        pairs=pairs,
        failure_rates={n: tally(camp.holdout.records[n]).failure_rate for n in names},
        flags=camp.flags,
    )
