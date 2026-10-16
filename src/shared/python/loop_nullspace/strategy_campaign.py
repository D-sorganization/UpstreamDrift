"""Controlled strategy comparisons, ranking and variance budget (NSX-95, #12724).

Takes evaluated :class:`~.strategy_arms.ArmResult` arms and compares them under a
declared control:

* ``MATCHED_CHS``: only arms whose clubhead speed is within ``chs_rel_tol`` of the
  reference are ranked.  A slower (or faster) swing is excluded with
  ``SPEED_MISMATCH`` however quiet it is: speed is an allocation-invariant input,
  and trading it for consistency is the S8 question, answered by the unconstrained
  fronts, not by the ranking.
* ``MATCHED_EFFORT``: the speed match plus a common effort cap *within one effort
  basis* (allocated torque, expected layer torque and muscle activation are
  different lines, rule R10, and are never compared).  An arm over the cap is
  excluded, never relaxed.
* ``UNCONSTRAINED``: every attempted arm is ranked, and the non-dominated sets of
  variance against clubhead speed and against effort (per basis) are reported
  using the NSX-25 dominance rule.

Ranking basis.  Arms are ordered by the face-angle variance *ratio to the
strategy-off baseline of the arm's own representation* (lower is better).  The
catalogue's toy and model representations are not the same plant, so absolute
variances are reported but not compared across representations.  An arm that does
not beat its baseline (ratio within ``baseline_tol`` of one) is ranked but cannot
be the winner.

Non-claim: conditional on the declared priors; not coaching advice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np

from src.shared.python.loop_nullspace.evidence import QuantityRecord
from src.shared.python.loop_nullspace.pareto import ParetoPoint, _mark_dominated
from src.shared.python.loop_nullspace.solvers import SolveStatus
from src.shared.python.loop_nullspace.strategy_arms import ArmResult
from src.shared.python.loop_nullspace.strategy_spec import NON_CLAIM

__all__ = [
    "ASSUMPTIONS",
    "ComparisonMode",
    "ComparisonReport",
    "Exclusion",
    "MatchSettings",
    "RankedArm",
    "VarianceBudgetTable",
    "run_comparison",
    "variance_budget_table",
]

PRIMARY_OUTCOME = "face_angle_variance"
RANKING_BASIS = "variance_ratio_to_strategy_off_baseline"
ASSUMPTIONS = (
    "linear tangent model at the nominal motion; saturation, slip and missed "
    "impacts are outside it",
    "disturbance amplitudes are declared hypotheses, not measurements",
    "ranking basis is the face-angle variance ratio to the strategy-off baseline "
    "of each arm's own representation; absolute variances are not compared across "
    "representations",
    "effort bases (allocated torque, layer torque, muscle activation) are separate "
    "lines and are only compared within one basis",
    "clubhead speed is an allocation-invariant input of each case",
)
_TIE_RTOL = 1e-9


class ComparisonMode(str, Enum):
    """Which controls the comparison applies (module docstring)."""

    MATCHED_CHS = "matched_chs"
    MATCHED_EFFORT = "matched_effort"
    UNCONSTRAINED = "unconstrained"


class Exclusion(str, Enum):
    """Why an arm is not ranked (never relaxed to make it fit)."""

    SPEED_MISMATCH = "speed_mismatch"
    EFFORT_EXCEEDED = "effort_exceeded"
    EFFORT_BASIS_MISMATCH = "effort_basis_mismatch"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True)
class MatchSettings:
    """Reference speed, tolerances and (matched effort) the cap and its basis.

    ``chs_rel_tol`` is the relative clubhead-speed window; ``effort_rtol`` the
    solver-level slack on the cap; ``baseline_tol`` the relative margin by which a
    ratio must be below one to count as beating the baseline.
    """

    reference_chs_mps: float
    chs_rel_tol: float = 0.02
    effort_cap: float | None = None
    effort_basis: str | None = None
    effort_rtol: float = 1e-6
    baseline_tol: float = 1e-6

    def __post_init__(self) -> None:
        if not (math.isfinite(self.reference_chs_mps) and self.reference_chs_mps > 0):
            raise ValueError("reference_chs_mps must be positive and finite")
        if not (math.isfinite(self.chs_rel_tol) and self.chs_rel_tol > 0.0):
            raise ValueError(f"chs_rel_tol must be positive, got {self.chs_rel_tol}")
        if self.effort_cap is not None and not (
            math.isfinite(self.effort_cap) and self.effort_cap > 0.0
        ):
            raise ValueError(f"effort_cap must be positive, got {self.effort_cap}")
        for name in ("effort_rtol", "baseline_tol"):
            if not getattr(self, name) >= 0.0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class RankedArm:
    """One arm with its place: ``rank`` is ``None`` when excluded (competition
    ranking: arms tied within ``1e-9`` share a rank)."""

    arm: ArmResult
    rank: int | None
    admitted: bool
    exclusion: Exclusion | None
    beats_baseline: bool
    ratio_record: QuantityRecord | None = None


@dataclass(frozen=True)
class ComparisonReport:
    """Ranked strategy report with assumptions, failure rate and non-claim."""

    mode: ComparisonMode
    settings: MatchSettings
    rows: tuple[RankedArm, ...]
    fronts: dict[str, tuple[str, ...]]
    failure_rate: float
    not_attempted: tuple[str, ...]
    primary_outcome: str = PRIMARY_OUTCOME
    ranking_basis: str = RANKING_BASIS
    assumptions: tuple[str, ...] = ASSUMPTIONS
    non_claim: str = NON_CLAIM

    @property
    def ranked(self) -> tuple[RankedArm, ...]:
        """Admitted arms, best first."""
        rows = [r for r in self.rows if r.rank is not None]
        return tuple(sorted(rows, key=lambda r: r.rank))  # type: ignore[arg-type,return-value]

    @property
    def winner(self) -> str | None:
        """Label of the best arm that beats its baseline, else ``None``."""
        top = self.ranked[0] if self.ranked else None
        return top.arm.label if top is not None and top.beats_baseline else None


# --------------------------------------------------------------- admission
def _exclusion(
    arm: ArmResult, mode: ComparisonMode, settings: MatchSettings
) -> Exclusion | None:
    if not arm.status.attempted:
        return Exclusion.NOT_ATTEMPTED
    if arm.status.failed:
        return Exclusion.FAILED
    if mode is ComparisonMode.UNCONSTRAINED:
        return None
    ref = settings.reference_chs_mps
    if arm.chs_mps is None or abs(arm.chs_mps - ref) > settings.chs_rel_tol * ref:
        return Exclusion.SPEED_MISMATCH
    if mode is ComparisonMode.MATCHED_EFFORT:
        if arm.effort_basis != settings.effort_basis:
            return Exclusion.EFFORT_BASIS_MISMATCH
        cap = float(settings.effort_cap or 0.0)
        if arm.effort is None or arm.effort > cap * (1.0 + settings.effort_rtol):
            return Exclusion.EFFORT_EXCEEDED
    return None


def _ratio_record(arm: ArmResult) -> QuantityRecord | None:
    if arm.record is None or not math.isfinite(arm.ratio):
        return None
    return replace(arm.record, values=np.array([arm.ratio]), units="1")


def _competition_ranks(ratios: list[float]) -> list[int]:
    order = sorted(range(len(ratios)), key=lambda i: ratios[i])
    ranks = [0] * len(ratios)
    for pos, i in enumerate(order):
        prev = order[pos - 1] if pos else None
        tied = prev is not None and math.isclose(
            ratios[i], ratios[prev], rel_tol=_TIE_RTOL, abs_tol=0.0
        )
        ranks[i] = ranks[prev] if tied else pos + 1  # type: ignore[index]
    return ranks


def _nondominated(labels: list[str], first: list[float], second: list[float]) -> tuple:
    points = [
        ParetoPoint(eps=0.0, status=SolveStatus.OPTIMAL, primary=a, constrained=b)
        for a, b in zip(first, second, strict=True)
    ]
    marked = _mark_dominated(points)
    return tuple(n for n, p in zip(labels, marked, strict=True) if not p.dominated)


def _fronts(admitted: Sequence[ArmResult]) -> dict[str, tuple[str, ...]]:
    fronts: dict[str, tuple[str, ...]] = {}
    with_chs = [a for a in admitted if a.chs_mps is not None]
    if with_chs:
        fronts["variance_vs_chs"] = _nondominated(
            [a.label for a in with_chs],
            [a.ratio for a in with_chs],
            [-float(a.chs_mps) for a in with_chs],  # type: ignore[arg-type]
        )
    for basis in dict.fromkeys(a.effort_basis for a in admitted if a.effort_basis):
        group = [
            a for a in admitted if a.effort_basis == basis and a.effort is not None
        ]
        fronts[f"variance_vs_effort:{basis}"] = _nondominated(
            [a.label for a in group],
            [a.ratio for a in group],
            [float(a.effort) for a in group],  # type: ignore[arg-type]
        )
    return fronts


def run_comparison(
    arms: Sequence[ArmResult], mode: ComparisonMode, settings: MatchSettings
) -> ComparisonReport:
    """Rank ``arms`` under the controls of ``mode`` (module docstring).

    Preconditions (``ValueError``/``TypeError``): ``MATCHED_EFFORT`` needs
    ``settings.effort_cap`` and ``settings.effort_basis``; every arm is an
    :class:`ArmResult` with a unique label.  Postconditions: excluded arms are
    listed with the reason and never ranked; ``failure_rate`` counts failed arms
    over attempted ones; ``winner`` is ``None`` unless the best arm beats its
    baseline; nothing is relaxed to admit an arm.
    """
    if not isinstance(mode, ComparisonMode) or not isinstance(settings, MatchSettings):
        raise TypeError("mode must be a ComparisonMode and settings a MatchSettings")
    if mode is ComparisonMode.MATCHED_EFFORT and (
        settings.effort_cap is None or settings.effort_basis is None
    ):
        raise ValueError("MATCHED_EFFORT needs effort_cap and effort_basis")
    arms = tuple(arms)
    if not all(isinstance(a, ArmResult) for a in arms):
        raise TypeError("every arm must be an ArmResult")
    if len({a.label for a in arms}) != len(arms):
        raise ValueError("arm labels must be unique")
    exclusions = [_exclusion(a, mode, settings) for a in arms]
    admitted = [a for a, e in zip(arms, exclusions, strict=True) if e is None]
    ranks = dict(
        zip(
            [a.label for a in admitted],
            _competition_ranks([a.ratio for a in admitted]),
            strict=True,
        )
    )
    rows = tuple(
        RankedArm(
            arm=a,
            rank=ranks.get(a.label),
            admitted=e is None,
            exclusion=e,
            beats_baseline=e is None and a.ratio < 1.0 - settings.baseline_tol,
            ratio_record=_ratio_record(a) if e is None else None,
        )
        for a, e in zip(arms, exclusions, strict=True)
    )
    attempted = [a for a in arms if a.status.attempted]
    failed = [a for a in attempted if a.status.failed]
    return ComparisonReport(
        mode=mode,
        settings=settings,
        rows=rows,
        fronts=_fronts(admitted),
        failure_rate=len(failed) / len(attempted) if attempted else 0.0,
        not_attempted=tuple(a.label for a in arms if not a.status.attempted),
    )


# ---------------------------------------------------------- variance budget
@dataclass(frozen=True)
class VarianceBudgetTable:
    """Share of each arm's face-angle variance by input class (rows sum to one)."""

    classes: tuple[str, ...]
    rows: dict[str, dict[str, float]] = field(default_factory=dict)
    non_claim: str = NON_CLAIM


def variance_budget_table(arms: Sequence[ArmResult]) -> VarianceBudgetTable:
    """Fractions of the variance by input class for every arm that has a budget.

    Arms without a budget (not evaluated, or S8 whose front reports the variance
    only) are omitted, not zero-filled.  Postcondition: each row sums to one.
    """
    rows: dict[str, dict[str, float]] = {}
    for arm in arms:
        total = sum(arm.budget.values())
        if not arm.budget or not total > 0.0:
            continue
        rows[arm.label] = {k: float(v / total) for k, v in arm.budget.items()}
    classes = tuple(dict.fromkeys(k for row in rows.values() for k in row))
    return VarianceBudgetTable(classes=classes, rows=rows)
