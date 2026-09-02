"""Ranking two designs, or refusing to (issue #9243).

The one behaviour this module exists for
----------------------------------------

Two designs whose bands overlap are **not** ordered. A design tool that always
produces a winner is worse than one that admits a tie, because on this design
space the tie is usually the truth: the accelerated-mass interval of issue
#8659 is a factor of 2.4 wide at the nominal shot, and the objective band it
opens is wider than the difference between most pairs of soles.

:func:`rank_with_bands` therefore returns one of exactly three verdicts --
:attr:`RankingVerdict.A_BETTER`, :attr:`RankingVerdict.B_BETTER` or
:attr:`RankingVerdict.INDISTINGUISHABLE` -- and :attr:`BandedRanking.winner`
is ``None`` for the third. There is no fourth "probably A" state and no
tie-break on the central values, because the central value of a consistency
band is a stated convention and breaking a tie on a convention is how a point
estimate gets back in.

How this differs from :mod:`bunkershot3d.study.comparison`
----------------------------------------------------------

:func:`~bunkershot3d.study.comparison.compare_designs` bootstraps *replicate
spread*: how much the objective moves as the delivery conditions change. That
is real and it is one of the three classes here, but it is the only one it
sees. Two designs can have tight bootstrap intervals and still be
indistinguishable once the model-form band they were both computed under is
carried through -- and that is the ordinary case, not the exception. This
module ranks on a :class:`~bunkershot3d.vandv.budget.UncertaintyBudget`, so
the bootstrap enters as one ``SAMPLING`` term beside the model-form band
rather than as the whole story.

Separated is necessary, not sufficient
--------------------------------------

:attr:`BandedRanking.defensible` is true only when the bands separate **and**
neither budget left a contribution unquantified. The workbench's always does:
:data:`bunkershot3d.ball.splash.BALL_MOMENTUM_TRANSFER_EFFICIENCY` is 0.5 with
no measurement and no published range behind it, so no band computed through
it is complete. Reporting a separated verdict without that sentence beside it
would be the same overclaim in a narrower place.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..vandv.band import CONSISTENCY_BAND_NAMING_REASON, ConsistencyBand
from ..vandv.budget import DominantTerm, UncertaintyBudget, UnquantifiedTerm

__all__ = [
    "BandedRanking",
    "RankingVerdict",
    "rank_with_bands",
]


class RankingVerdict(StrEnum):
    """What a comparison of two designs is allowed to conclude.

    Three values, and deliberately no fourth. "Probably A" would be a
    probability statement, and nothing in the band behind it is a probability.
    """

    A_BETTER = "a-better"
    B_BETTER = "b-better"
    INDISTINGUISHABLE = "indistinguishable"

    @property
    def is_decided(self) -> bool:
        """True when the comparison separated the two designs."""
        return self is not RankingVerdict.INDISTINGUISHABLE


@dataclass(frozen=True)
class BandedRanking:
    """Two designs compared through their whole uncertainty budgets.

    Attributes:
        names: The two design names, in the order they were given.
        budgets: Each design's budget, so a report can re-derive the verdict.
        lower_is_better: Objective direction.
        verdict: A better, B better, or indistinguishable at this uncertainty.
        separation: Signed gap between the two bands, in the objective's unit.
            Positive when they are disjoint, negative by the depth of the
            overlap when they are not.
        dominant: The widest single term across both budgets, or ``None`` when
            neither budget has a sized term.
        unquantified: Every contribution either budget knows about and has no
            number for.
    """

    names: tuple[str, str]
    budgets: tuple[UncertaintyBudget, UncertaintyBudget]
    lower_is_better: bool
    verdict: RankingVerdict
    separation: float
    dominant: DominantTerm | None
    unquantified: tuple[UnquantifiedTerm, ...]

    @property
    def is_decided(self) -> bool:
        """True when the comparison separated the two designs."""
        return self.verdict.is_decided

    @property
    def bands(self) -> tuple[ConsistencyBand, ConsistencyBand]:
        """Each design's band, in input order."""
        return (self.budgets[0].band(), self.budgets[1].band())

    @property
    def winner(self) -> str | None:
        """The better design's name, or ``None`` when the bands overlap.

        ``None`` is the point of this property: a caller that reads a name off
        every comparison cannot accidentally report a tie as a win.
        """
        if self.verdict is RankingVerdict.A_BETTER:
            return self.names[0]
        if self.verdict is RankingVerdict.B_BETTER:
            return self.names[1]
        return None

    @property
    def defensible(self) -> bool:
        """Whether the verdict rests on a complete budget.

        False when the bands overlap, and false when either budget left a
        known contribution unquantified -- a separated verdict computed
        through an uncalibrated constant nobody has bounded is still a claim
        beyond the evidence.
        """
        return self.verdict.is_decided and not self.unquantified

    def dominance_statement(self) -> str:
        """One line naming the term that decides how wide the comparison is.

        Returns:
            The statement, or a sentence saying no term was sized.
        """
        if self.dominant is None:
            return (
                "no sized uncertainty term: this comparison is a point "
                "estimate and its verdict carries no width at all"
            )
        verb = "swamps the budget at" if self.dominant.swamps else "leads at"
        return (
            f"{self.dominant.term.name} "
            f"({self.dominant.term.uncertainty_class.value}) {verb} "
            f"{self.dominant.share:.0%} of the band -- "
            f"{self.dominant.term.source}"
        )

    def statement(self) -> str:
        """The whole comparison as report lines.

        Returns:
            A multi-line statement: the verdict, both bands, what dominates,
            what was left unquantified, and the naming disclaimer.
        """
        left, right = self.bands
        direction = "lower is better" if self.lower_is_better else "higher is better"
        headline = (
            f"{self.names[0]} and {self.names[1]} are INDISTINGUISHABLE at this "
            f"uncertainty (bands overlap by {-self.separation:.4g})"
            if self.verdict is RankingVerdict.INDISTINGUISHABLE
            else (f"{self.winner} is better, by a band gap of {self.separation:.4g}")
        )
        lines = [
            f"{headline} [{direction}]",
            f"  {self.names[0]}: {left.statement()}",
            f"  {self.names[1]}: {right.statement()}",
            f"  {self.dominance_statement()}",
        ]
        for term in self.unquantified:
            lines.append(
                f"  UNQUANTIFIED ({term.uncertainty_class.value}): "
                f"{term.name} -- {term.reason}"
            )
        if self.unquantified:
            lines.append(
                "  the bands above are a LOWER BOUND on the spread, so this "
                "verdict is not defensible on its own"
            )
        lines.append(f"  {CONSISTENCY_BAND_NAMING_REASON}")
        return "\n".join(lines)


def _verdict(
    left: ConsistencyBand, right: ConsistencyBand, lower_is_better: bool
) -> RankingVerdict:
    """Decide the comparison from the two bands alone.

    Args:
        left: First design's band.
        right: Second design's band.
        lower_is_better: Objective direction.

    Returns:
        The verdict. Overlapping bands -- touching included -- are never
        ordered: a shared endpoint is an equality, and breaking it on the
        central values would rank two designs on a stated convention.
    """
    if left.overlaps(right):
        return RankingVerdict.INDISTINGUISHABLE
    left_is_lower = left.upper < right.lower
    if left_is_lower == lower_is_better:
        return RankingVerdict.A_BETTER
    return RankingVerdict.B_BETTER


def rank_with_bands(
    first_name: str,
    first: UncertaintyBudget,
    second_name: str,
    second: UncertaintyBudget,
    *,
    lower_is_better: bool = True,
) -> BandedRanking:
    """Rank two designs on their bands, or report that they cannot be ranked.

    Args:
        first_name: First design's name.
        first: First design's uncertainty budget for the objective.
        second_name: Second design's name.
        second: Second design's budget for the same objective.
        lower_is_better: Objective direction. The workbench ranks on absolute
            carry error against a target, so its objective is lower-is-better.

    Returns:
        The ranking, whose :attr:`BandedRanking.winner` is ``None`` whenever
        the two bands overlap.

    Raises:
        ValueError: If the two designs share a name, which would make the
            comparison unreadable.
    """
    if first_name == second_name:
        raise ValueError(
            "the two designs must have different names; a ranking reports "
            f"them by name and both are {first_name!r}"
        )
    left = first.band()
    right = second.band()
    candidates = [
        candidate
        for candidate in (first.dominant(), second.dominant())
        if candidate is not None
    ]
    dominant = (
        max(candidates, key=lambda candidate: candidate.term.width)
        if candidates
        else None
    )
    return BandedRanking(
        names=(first_name, second_name),
        budgets=(first, second),
        lower_is_better=lower_is_better,
        verdict=_verdict(left, right, lower_is_better),
        separation=left.gap_to(right),
        dominant=dominant,
        # De-duplicated: both designs run through the same uncalibrated
        # constants, and printing each gap twice buries it rather than
        # emphasising it.
        unquantified=tuple(dict.fromkeys((*first.unquantified, *second.unquantified))),
    )
