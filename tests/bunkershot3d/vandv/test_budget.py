"""Uncertainty classes stay separate, and dominance is reported (#9243).

The point of these tests is what the budget **refuses**: it will not hand a
space-time GCI to V&V 20 as a spatial ``u_h`` without being told to, it will
not let a model-form band be mistaken for a numerical one, and it will not
report a dominant term without saying what it left unquantified.
"""

from __future__ import annotations

import math

import pytest

from src.bunkershot3d.vandv.band import ConsistencyBand
from src.bunkershot3d.vandv.budget import (
    NumericalBasis,
    UncertaintyBudget,
    UncertaintyClass,
    UncertaintyTerm,
    UnquantifiedTerm,
)
from src.bunkershot3d.vandv.exceptions import VandVError

pytestmark = pytest.mark.unit


def _model_form(name: str, lower: float, upper: float) -> UncertaintyTerm:
    """An asymmetric model-form term, for brevity in the tests below."""
    return UncertaintyTerm(
        name=name,
        uncertainty_class=UncertaintyClass.MODEL_FORM,
        lower_offset=lower,
        upper_offset=upper,
        source="test",
    )


def _numerical(name: str, half_width: float, basis: NumericalBasis) -> UncertaintyTerm:
    """A symmetric numerical term with a stated basis."""
    return UncertaintyTerm.symmetric(
        name=name,
        uncertainty_class=UncertaintyClass.NUMERICAL,
        half_width=half_width,
        source="test",
        basis=basis,
    )


class TestTerms:
    """A term that cannot say what class it belongs to is refused."""

    def test_numerical_term_needs_a_basis(self) -> None:
        """ "Numerical" alone does not say what was refined."""
        with pytest.raises(ValueError, match="basis"):
            UncertaintyTerm(
                name="u_h",
                uncertainty_class=UncertaintyClass.NUMERICAL,
                lower_offset=1.0,
                upper_offset=1.0,
                source="test",
            )

    def test_model_form_term_must_not_carry_a_basis(self) -> None:
        """A discretisation basis on a model-form term is a category error."""
        with pytest.raises(ValueError, match="basis"):
            UncertaintyTerm(
                name="eta",
                uncertainty_class=UncertaintyClass.MODEL_FORM,
                lower_offset=1.0,
                upper_offset=1.0,
                source="test",
                basis=NumericalBasis.SPATIAL,
            )

    def test_offsets_must_be_non_negative(self) -> None:
        """An offset is a distance from the centre, never a direction."""
        with pytest.raises(ValueError, match="non-negative"):
            _model_form("bad", -1.0, 1.0)

    def test_source_is_required(self) -> None:
        """A number with no provenance is not admissible in a budget."""
        with pytest.raises(ValueError, match="source"):
            UncertaintyTerm(
                name="mystery",
                uncertainty_class=UncertaintyClass.MODEL_FORM,
                lower_offset=1.0,
                upper_offset=1.0,
                source="   ",
            )

    def test_from_band_keeps_the_asymmetry(self) -> None:
        """The accelerated-mass band is asymmetric and must stay that way."""
        term = UncertaintyTerm.from_band(
            name="accelerated mass",
            uncertainty_class=UncertaintyClass.MODEL_FORM,
            band=ConsistencyBand(0.76, 1.59, 3.17),
            source="issue #8659",
        )
        assert term.lower_offset == pytest.approx(0.83)
        assert term.upper_offset == pytest.approx(1.58)


class TestClassSeparation:
    """Numerical and model-form uncertainty are never one number."""

    def test_subtotals_are_reported_per_class(self) -> None:
        """The split is available before anything is combined."""
        budget = UncertaintyBudget(
            quantity="carry",
            central=10.0,
            terms=(
                _model_form("mass", 1.0, 2.0),
                _numerical("grid", 0.5, NumericalBasis.SPATIAL),
            ),
        )
        split = budget.by_class()
        assert split[UncertaintyClass.MODEL_FORM].width == pytest.approx(3.0)
        assert split[UncertaintyClass.NUMERICAL].width == pytest.approx(1.0)

    def test_model_form_terms_add_within_their_class(self) -> None:
        """Two epistemic bands are not independent and do not root-sum."""
        budget = UncertaintyBudget(
            quantity="carry",
            central=10.0,
            terms=(_model_form("a", 1.0, 1.0), _model_form("b", 1.0, 1.0)),
        )
        subtotal = budget.by_class()[UncertaintyClass.MODEL_FORM]
        assert subtotal.lower_offset == pytest.approx(2.0)

    def test_sampling_terms_combine_in_quadrature(self) -> None:
        """Replicate noise is the one class with an independence claim."""
        term = UncertaintyTerm.symmetric(
            name="sweep",
            uncertainty_class=UncertaintyClass.SAMPLING,
            half_width=3.0,
            source="bootstrap",
        )
        other = UncertaintyTerm.symmetric(
            name="seeds",
            uncertainty_class=UncertaintyClass.SAMPLING,
            half_width=4.0,
            source="bootstrap",
        )
        budget = UncertaintyBudget(quantity="carry", central=10.0, terms=(term, other))
        subtotal = budget.by_class()[UncertaintyClass.SAMPLING]
        assert subtotal.upper_offset == pytest.approx(5.0)

    def test_band_is_the_conservative_sum_across_classes(self) -> None:
        """Across classes the treatment is additive, and it is stated."""
        budget = UncertaintyBudget(
            quantity="carry",
            central=10.0,
            terms=(
                _model_form("mass", 1.0, 2.0),
                _numerical("grid", 0.5, NumericalBasis.SPATIAL),
            ),
        )
        band = budget.band()
        assert band.lower == pytest.approx(10.0 - 1.5)
        assert band.upper == pytest.approx(10.0 + 2.5)

    def test_a_budget_with_no_terms_is_a_point(self) -> None:
        """No terms means no width, and the statement says so."""
        budget = UncertaintyBudget(quantity="carry", central=10.0)
        assert budget.band().is_point


class TestSpaceTimeRefusal:
    """ADR-0033: the shipped column GCI is a space-time band, not spatial."""

    def _budget(self) -> UncertaintyBudget:
        """A budget whose only numerical term is a space-time GCI."""
        return UncertaintyBudget(
            quantity="carry",
            central=10.0,
            terms=(_numerical("F1 column", 0.4, NumericalBasis.SPACE_TIME),),
        )

    def test_v20_mapping_refuses_a_space_time_term(self) -> None:
        """u_h is a spatial quantity; folding this in silently is the trap."""
        with pytest.raises(VandVError, match="(?i)space-time"):
            self._budget().as_v20_numerical()

    def test_the_refusal_names_the_courant_reason(self) -> None:
        """The message says why, not just that."""
        with pytest.raises(VandVError, match="Courant"):
            self._budget().as_v20_numerical()

    def test_explicit_opt_in_folds_it_into_u_h(self) -> None:
        """Saying which is which is exactly what makes the fold admissible."""
        numerical = self._budget().as_v20_numerical(space_time=NumericalBasis.SPATIAL)
        assert numerical.u_h == pytest.approx(0.4)

    def test_spatial_and_iterative_map_without_an_opt_in(self) -> None:
        """The ordinary case needs no ceremony."""
        budget = UncertaintyBudget(
            quantity="carry",
            central=10.0,
            terms=(
                _numerical("grid", 0.3, NumericalBasis.SPATIAL),
                _numerical("time step", 0.2, NumericalBasis.TEMPORAL),
                _numerical("solve", 0.1, NumericalBasis.ITERATIVE),
                _numerical("round", 0.05, NumericalBasis.ROUND_OFF),
            ),
        )
        numerical = budget.as_v20_numerical()
        assert numerical.u_h == pytest.approx(0.5)
        assert numerical.u_it == pytest.approx(0.1)
        assert numerical.u_ro == pytest.approx(0.05)

    def test_model_form_never_reaches_u_num(self) -> None:
        """The V&V 20 numerical combination is numerical terms only."""
        budget = UncertaintyBudget(
            quantity="carry",
            central=10.0,
            terms=(
                _model_form("mass", 5.0, 5.0),
                _numerical("grid", 0.3, NumericalBasis.SPATIAL),
            ),
        )
        assert budget.as_v20_numerical().total == pytest.approx(0.3)


class TestDominance:
    """The useful output is often which term is doing all the work."""

    def test_dominant_term_is_the_widest(self) -> None:
        """Ranked on total width, not on one side of an asymmetric band."""
        budget = UncertaintyBudget(
            quantity="carry",
            central=10.0,
            terms=(
                _model_form("mass", 1.0, 5.0),
                _numerical("grid", 0.5, NumericalBasis.SPATIAL),
            ),
        )
        dominant = budget.dominant()
        assert dominant is not None
        assert dominant.term.name == "mass"
        assert dominant.share == pytest.approx(6.0 / 7.0)

    def test_dominance_is_flagged_when_one_term_swamps(self) -> None:
        """Above the threshold the ranking is really about one assumption."""
        budget = UncertaintyBudget(
            quantity="carry",
            central=10.0,
            terms=(
                _model_form("mass", 1.0, 5.0),
                _numerical("grid", 0.05, NumericalBasis.SPATIAL),
            ),
        )
        dominant = budget.dominant()
        assert dominant is not None
        assert dominant.swamps

    def test_no_terms_means_no_dominant_term(self) -> None:
        """An empty budget names no culprit rather than inventing one."""
        assert UncertaintyBudget(quantity="carry", central=10.0).dominant() is None

    def test_statement_names_the_class_of_the_dominant_term(self) -> None:
        """A reader must be able to tell numerics from modelling."""
        budget = UncertaintyBudget(
            quantity="carry",
            central=10.0,
            terms=(_model_form("mass", 1.0, 5.0),),
        )
        assert "model-form" in budget.statement()


class TestUnquantified:
    """A term that is known but unsized must not read as absent."""

    def test_unquantified_terms_do_not_widen_the_band(self) -> None:
        """They cannot: nobody has a number for them."""
        budget = UncertaintyBudget(
            quantity="carry",
            central=10.0,
            terms=(_model_form("mass", 1.0, 1.0),),
            unquantified=(
                UnquantifiedTerm(
                    name="transfer efficiency",
                    uncertainty_class=UncertaintyClass.MODEL_FORM,
                    reason="no published measurement exists (#8616)",
                ),
            ),
        )
        assert budget.band().width == pytest.approx(2.0)

    def test_the_band_is_declared_a_lower_bound(self) -> None:
        """With an unsized term outstanding the width is not the whole story."""
        budget = UncertaintyBudget(
            quantity="carry",
            central=10.0,
            terms=(_model_form("mass", 1.0, 1.0),),
            unquantified=(
                UnquantifiedTerm(
                    name="transfer efficiency",
                    uncertainty_class=UncertaintyClass.MODEL_FORM,
                    reason="no published measurement exists (#8616)",
                ),
            ),
        )
        assert budget.band_is_lower_bound
        assert "LOWER BOUND" in budget.statement()

    def test_a_fully_quantified_budget_is_not_a_lower_bound(self) -> None:
        """The flag means something, so it must be able to be false."""
        budget = UncertaintyBudget(
            quantity="carry",
            central=10.0,
            terms=(_model_form("mass", 1.0, 1.0),),
        )
        assert not budget.band_is_lower_bound

    def test_unquantified_numerical_term_needs_a_basis_too(self) -> None:
        """Unsized does not mean unclassified."""
        with pytest.raises(ValueError, match="basis"):
            UnquantifiedTerm(
                name="F1 column GCI",
                uncertainty_class=UncertaintyClass.NUMERICAL,
                reason="not run for this quantity",
            )

    def test_statement_lists_every_unquantified_term(self) -> None:
        """Silence about them would be the whole failure mode."""
        budget = UncertaintyBudget(
            quantity="carry",
            central=10.0,
            unquantified=(
                UnquantifiedTerm(
                    name="transfer efficiency",
                    uncertainty_class=UncertaintyClass.MODEL_FORM,
                    reason="no published measurement exists (#8616)",
                ),
            ),
        )
        assert "transfer efficiency" in budget.statement()


class TestNonFinite:
    """A budget is arithmetic that reaches a ranking; NaN cannot enter it."""

    def test_non_finite_offset_is_refused(self) -> None:
        """An infinite offset would make every comparison a tie."""
        with pytest.raises(ValueError, match="finite"):
            _model_form("bad", 1.0, math.inf)

    def test_non_finite_central_is_refused(self) -> None:
        """A budget about NaN cannot be ranked."""
        with pytest.raises(ValueError, match="finite"):
            UncertaintyBudget(quantity="carry", central=math.nan)
