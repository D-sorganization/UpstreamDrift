"""A comparison that is allowed to say "I cannot tell" (issue #9243).

The single behaviour these tests exist for: two designs whose bands overlap
are **not** ordered. A design tool that always produces a winner is worse
than one that admits a tie, because the tie is the truth about most of this
design space.
"""

from __future__ import annotations

import pytest

from src.bunkershot3d.study.ranking import (
    BandedRanking,
    RankingVerdict,
    rank_with_bands,
)
from src.bunkershot3d.vandv.budget import (
    NumericalBasis,
    UncertaintyBudget,
    UncertaintyClass,
    UncertaintyTerm,
    UnquantifiedTerm,
)

pytestmark = pytest.mark.unit


def _budget(
    central: float,
    half_width: float,
    *,
    unquantified: bool = False,
    name: str = "accelerated sand mass",
) -> UncertaintyBudget:
    """A one-term model-form budget at a stated centre and width."""
    unsized = (
        UnquantifiedTerm(
            name="ball momentum transfer efficiency",
            uncertainty_class=UncertaintyClass.MODEL_FORM,
            reason="no published measurement exists (#8616)",
        ),
    )
    return UncertaintyBudget(
        quantity="mean absolute carry error",
        central=central,
        terms=(
            UncertaintyTerm.symmetric(
                name=name,
                uncertainty_class=UncertaintyClass.MODEL_FORM,
                half_width=half_width,
                source="issue #8659",
            ),
        ),
        unquantified=unsized if unquantified else (),
    )


class TestRefusalToRank:
    """The central deliverable: overlapping bands are not ordered."""

    def test_overlapping_bands_are_indistinguishable(self) -> None:
        """Different centres do not make a winner when the bands overlap."""
        ranking = rank_with_bands("A", _budget(10.0, 2.0), "B", _budget(11.0, 2.0))
        assert ranking.verdict is RankingVerdict.INDISTINGUISHABLE
        assert ranking.winner is None

    def test_touching_bands_are_indistinguishable(self) -> None:
        """A shared endpoint is an equality, not a separation."""
        ranking = rank_with_bands("A", _budget(10.0, 1.0), "B", _budget(12.0, 1.0))
        assert ranking.verdict is RankingVerdict.INDISTINGUISHABLE

    def test_a_clearly_better_design_still_wins(self) -> None:
        """Refusing everything would be as useless as ranking everything."""
        ranking = rank_with_bands("A", _budget(5.0, 1.0), "B", _budget(20.0, 1.0))
        assert ranking.verdict is RankingVerdict.A_BETTER
        assert ranking.winner == "A"

    def test_the_second_design_can_win(self) -> None:
        """The verdict is not an artefact of argument order."""
        ranking = rank_with_bands("A", _budget(20.0, 1.0), "B", _budget(5.0, 1.0))
        assert ranking.verdict is RankingVerdict.B_BETTER
        assert ranking.winner == "B"

    def test_higher_is_better_flips_the_verdict(self) -> None:
        """Objective direction is an input, not an assumption."""
        ranking = rank_with_bands(
            "A",
            _budget(5.0, 1.0),
            "B",
            _budget(20.0, 1.0),
            lower_is_better=False,
        )
        assert ranking.verdict is RankingVerdict.B_BETTER

    def test_identical_designs_are_indistinguishable(self) -> None:
        """Two copies of one design must not be ordered by rounding."""
        ranking = rank_with_bands("A", _budget(10.0, 1.0), "B", _budget(10.0, 1.0))
        assert ranking.verdict is RankingVerdict.INDISTINGUISHABLE

    def test_zero_width_bands_can_separate(self) -> None:
        """A point comparison still works, and says its bands were points."""
        ranking = rank_with_bands("A", _budget(10.0, 0.0), "B", _budget(11.0, 0.0))
        assert ranking.verdict is RankingVerdict.A_BETTER

    def test_widening_the_band_can_erase_a_verdict(self) -> None:
        """This is the whole propagation story in one assertion."""
        narrow = rank_with_bands("A", _budget(10.0, 0.1), "B", _budget(11.0, 0.1))
        wide = rank_with_bands("A", _budget(10.0, 2.0), "B", _budget(11.0, 2.0))
        assert narrow.verdict is RankingVerdict.A_BETTER
        assert wide.verdict is RankingVerdict.INDISTINGUISHABLE

    def test_names_must_differ(self) -> None:
        """A ranking reports by name and two of one name is unreadable."""
        with pytest.raises(ValueError, match="different names"):
            rank_with_bands("A", _budget(10.0, 1.0), "A", _budget(11.0, 1.0))


class TestSeparation:
    """The size of the gap is reported, not just its sign."""

    def test_separated_designs_report_a_positive_gap(self) -> None:
        """The gap is in the objective's own unit."""
        ranking = rank_with_bands("A", _budget(5.0, 1.0), "B", _budget(20.0, 1.0))
        assert ranking.separation == pytest.approx(13.0)

    def test_overlapping_designs_report_a_negative_gap(self) -> None:
        """How badly the bands overlap is useful; hiding it is not."""
        ranking = rank_with_bands("A", _budget(10.0, 2.0), "B", _budget(11.0, 2.0))
        assert ranking.separation == pytest.approx(-3.0)

    def test_centres_are_still_reported_when_indistinguishable(self) -> None:
        """Refusing to rank is not refusing to report."""
        ranking = rank_with_bands("A", _budget(10.0, 2.0), "B", _budget(11.0, 2.0))
        assert ranking.bands[0].central == pytest.approx(10.0)
        assert ranking.bands[1].central == pytest.approx(11.0)


class TestDominance:
    """What dominates is reported for the comparison, not just per design."""

    def test_the_widest_term_across_both_designs_is_named(self) -> None:
        """One number decides the verdict, so the reader is told which."""
        left = UncertaintyBudget(
            quantity="objective",
            central=10.0,
            terms=(
                UncertaintyTerm.symmetric(
                    name="accelerated sand mass",
                    uncertainty_class=UncertaintyClass.MODEL_FORM,
                    half_width=5.0,
                    source="#8659",
                ),
                UncertaintyTerm.symmetric(
                    name="flight time step",
                    uncertainty_class=UncertaintyClass.NUMERICAL,
                    half_width=0.01,
                    source="GCI",
                    basis=NumericalBasis.TEMPORAL,
                ),
            ),
        )
        ranking = rank_with_bands("A", left, "B", _budget(11.0, 1.0))
        assert ranking.dominant is not None
        assert ranking.dominant.term.name == "accelerated sand mass"
        assert ranking.dominant.term.uncertainty_class is UncertaintyClass.MODEL_FORM

    def test_dominance_statement_says_the_class(self) -> None:
        """Numerics and modelling call for different follow-up work."""
        ranking = rank_with_bands("A", _budget(10.0, 5.0), "B", _budget(11.0, 1.0))
        assert "model-form" in ranking.dominance_statement()

    def test_no_terms_means_no_dominant_term(self) -> None:
        """Two point budgets name no culprit."""
        empty = UncertaintyBudget(quantity="objective", central=10.0)
        other = UncertaintyBudget(quantity="objective", central=11.0)
        assert rank_with_bands("A", empty, "B", other).dominant is None


class TestUnquantifiedCaveat:
    """A verdict resting on unsized assumptions says so in the same breath."""

    def test_unquantified_terms_are_collected_from_both_designs(self) -> None:
        """A caveat on either side applies to the comparison."""
        ranking = rank_with_bands(
            "A",
            _budget(5.0, 1.0, unquantified=True),
            "B",
            _budget(20.0, 1.0),
        )
        assert len(ranking.unquantified) == 1

    def test_a_verdict_with_unsized_terms_is_not_defensible(self) -> None:
        """Separated bands are necessary for a defensible verdict, not enough."""
        ranking = rank_with_bands(
            "A",
            _budget(5.0, 1.0, unquantified=True),
            "B",
            _budget(20.0, 1.0),
        )
        assert ranking.verdict is RankingVerdict.A_BETTER
        assert not ranking.defensible

    def test_a_fully_quantified_separated_verdict_is_defensible(self) -> None:
        """The flag must be able to be true or it says nothing."""
        ranking = rank_with_bands("A", _budget(5.0, 1.0), "B", _budget(20.0, 1.0))
        assert ranking.defensible

    def test_statement_names_the_unquantified_term(self) -> None:
        """The reader is told what was left out, by name."""
        ranking = rank_with_bands(
            "A",
            _budget(5.0, 1.0, unquantified=True),
            "B",
            _budget(20.0, 1.0),
        )
        assert "transfer efficiency" in ranking.statement()

    def test_statement_of_a_tie_says_indistinguishable(self) -> None:
        """The word a reader needs is in the text, not only in an enum."""
        ranking = rank_with_bands("A", _budget(10.0, 2.0), "B", _budget(11.0, 2.0))
        assert "indistinguishable" in ranking.statement().lower()


class TestValueObject:
    """The ranking is immutable and self-describing."""

    def test_ranking_is_frozen(self) -> None:
        """A verdict that a caller can edit is not a verdict."""
        ranking = rank_with_bands("A", _budget(5.0, 1.0), "B", _budget(20.0, 1.0))
        with pytest.raises(AttributeError):
            ranking.verdict = RankingVerdict.B_BETTER  # type: ignore[misc]

    def test_ranking_exposes_both_budgets(self) -> None:
        """The comparison carries its inputs, so a report can re-derive it."""
        left = _budget(5.0, 1.0)
        ranking: BandedRanking = rank_with_bands("A", left, "B", _budget(20.0, 1.0))
        assert ranking.budgets[0] is left
