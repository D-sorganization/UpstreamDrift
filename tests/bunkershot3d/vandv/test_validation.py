"""Validation: the ASME V&V 20 metric, and what it refuses (issue #8616).

The tests that matter most here are the ones asserting that a validation
**cannot be constructed** for a quantity nobody has measured, and the one
asserting that the single comparison this package *can* form comes out
noise-limited.

Both are deliberate. A V&V suite whose headline is "everything agrees"
is usually a suite that compared the model against quantities chosen for
their agreeability.
"""

from __future__ import annotations

import math

import pytest

from bunkershot3d.solvers import DRFTSolver, MaterialResponse
from bunkershot3d.vandv import (
    COVERAGE_FACTOR,
    GRANULAR_INTRUSION_BENCHMARK,
    WIVOU_2016,
    NoReferenceDataError,
    NumericalUncertainty,
    ValidationComparison,
    ValidationReport,
    VandVError,
    VerificationError,
    quasi_static_plate_case,
    validate,
    validation_report,
)
from bunkershot3d.vandv.studies import (
    QUIKRETE_MEASURED_ALPHA_Z_N_PER_CM3,
    carry_correlation_comparison,
    correlation_standard_uncertainty,
    friction_angle_leverage_per_degree,
    plate_response_comparisons,
    predicted_alpha_z_n_per_cm3,
    surface_refinement_study,
)

pytestmark = [pytest.mark.unit, pytest.mark.scientific]

SMOKE_ANCHOR_N = 1550.0
"""The addendum's analytic order-of-magnitude anchor for a 20 x 80 mm sole."""


def _comparison(
    *, simulation: float, experiment: float, u_exp: float | None, u_input: float = 0.0
) -> ValidationComparison:
    """A minimal comparison on a quantity that *is* measurable."""
    return ValidationComparison(
        quantity="divot_depth_m",
        unit="m",
        simulation_value=simulation,
        experiment_value=experiment,
        numerical=NumericalUncertainty(u_h=0.0),
        u_input=u_input,
        u_exp=u_exp,
        reference="synthetic fixture for the metric's own unit tests",
    )


class TestNumericalUncertaintyCombination:
    """``u_num = u_h + u_it + u_ro`` by simple addition, not RMS."""

    def test_the_components_are_added_not_root_sum_squared(self) -> None:
        """The asymmetry against ``u_val`` is the substance of V&V 20.

        The three are correlated faces of one discrete solve, so they are
        treated as epistemic and added. Root-sum-squaring them would
        under-state ``u_num`` -- here by 42%.
        """
        numerical = NumericalUncertainty(u_h=3.0, u_it=4.0, u_ro=0.0)
        assert numerical.total == pytest.approx(7.0)
        assert numerical.root_sum_square == pytest.approx(5.0)
        assert numerical.total > numerical.root_sum_square

    def test_a_negative_component_is_refused(self) -> None:
        with pytest.raises(VandVError, match="non-negative"):
            NumericalUncertainty(u_h=-1.0)

    def test_iteration_and_round_off_default_to_zero_for_the_f0_tier(self) -> None:
        """F0 solves no linear system and iterates nothing."""
        assert NumericalUncertainty(u_h=2.0).total == pytest.approx(2.0)


class TestTheMetric:
    """``E = S - D``, ``u_val`` in quadrature, ``U = k u_val``."""

    def test_the_comparison_error_is_simulation_minus_experiment(self) -> None:
        result = validate(_comparison(simulation=0.045, experiment=0.040, u_exp=0.001))
        assert result.comparison_error == pytest.approx(0.005)

    def test_u_val_combines_the_three_sources_in_quadrature(self) -> None:
        comparison = ValidationComparison(
            quantity="divot_depth_m",
            unit="m",
            simulation_value=1.0,
            experiment_value=1.0,
            numerical=NumericalUncertainty(u_h=1.0, u_it=1.0, u_ro=1.0),
            u_input=4.0,
            u_exp=12.0,
            reference="synthetic fixture",
        )
        result = validate(comparison)
        assert result.u_num == pytest.approx(3.0)
        assert result.u_val == pytest.approx(13.0)

    def test_the_expanded_uncertainty_uses_k_equal_two(self) -> None:
        result = validate(_comparison(simulation=1.0, experiment=1.0, u_exp=2.0))
        assert COVERAGE_FACTOR == 2.0
        assert result.expanded_uncertainty == pytest.approx(4.0)

    def test_a_large_error_bounds_the_model_form_error_away_from_zero(self) -> None:
        result = validate(_comparison(simulation=0.100, experiment=0.040, u_exp=0.001))
        assert not result.noise_limited
        low, high = result.model_error_interval or (0.0, 0.0)
        assert low > 0.0 and high > low
        assert "model-form error is bounded" in result.statement()


class TestNoiseLimitedIsStatedOutLoud:
    """``|E| <= u_val`` means nothing has been learned. Say so."""

    def test_a_small_error_against_a_large_u_val_is_noise_limited(self) -> None:
        result = validate(_comparison(simulation=0.041, experiment=0.040, u_exp=0.010))
        assert result.noise_limited

    def test_the_statement_says_nothing_has_been_learned(self) -> None:
        result = validate(_comparison(simulation=0.041, experiment=0.040, u_exp=0.010))
        statement = result.statement()
        assert "NOISE-LIMITED" in statement
        assert "Nothing has been learned about model error" in statement
        assert "two error bars" in statement

    def test_the_boundary_case_counts_as_noise_limited(self) -> None:
        """``|E| == u_val`` is inside the interval, not outside it.

        The values are chosen to be exactly representable, so the test is
        about the boundary convention rather than about round-off.
        """
        result = validate(_comparison(simulation=1.5, experiment=1.0, u_exp=0.5))
        assert result.comparison_error == 0.5
        assert result.u_val == 0.5
        assert result.noise_limited

    def test_a_noise_limited_interval_straddles_zero(self) -> None:
        result = validate(_comparison(simulation=0.041, experiment=0.040, u_exp=0.010))
        low, high = result.model_error_interval or (1.0, 1.0)
        assert low < 0.0 < high


class TestUnknownExperimentalUncertaintyIsIndeterminate:
    """An unreported ``u_exp`` is not a zero one."""

    def test_a_missing_u_exp_gives_no_verdict(self) -> None:
        result = validate(_comparison(simulation=2.0, experiment=1.0, u_exp=None))
        assert result.is_indeterminate
        assert result.u_val is None
        assert result.model_error_interval is None

    def test_it_is_not_reported_as_noise_limited(self) -> None:
        """Indeterminate and noise-limited are different findings."""
        result = validate(_comparison(simulation=2.0, experiment=1.0, u_exp=None))
        assert not result.noise_limited

    def test_the_statement_explains_why_no_verdict_is_possible(self) -> None:
        result = validate(_comparison(simulation=2.0, experiment=1.0, u_exp=None))
        statement = result.statement()
        assert "INDETERMINATE" in statement
        assert "would claim the measurement was exact" in statement


class TestTheSuiteRefusesWhatWasNeverMeasured:
    """The register of unmeasured quantities, enforced at construction."""

    @pytest.mark.parametrize(
        "quantity",
        [
            "ball_launch_angle_rad",
            "ball_speed_m_s",
            "ball_spin_rad_s",
            "clubhead_deceleration_m_s2",
            "energy_split_fraction",
            "ejecta_mass_kg",
            "coefficient_of_restitution_through_sand",
        ],
    )
    def test_a_comparison_cannot_even_be_constructed(self, quantity: str) -> None:
        with pytest.raises(NoReferenceDataError, match="no published measurement"):
            ValidationComparison(
                quantity=quantity,
                unit="SI",
                simulation_value=1.0,
                experiment_value=1.0,
                numerical=NumericalUncertainty(u_h=0.0),
                u_input=0.0,
                u_exp=0.1,
                reference="there is no reference, which is the point",
            )

    def test_an_unsourced_experimental_value_is_refused(self) -> None:
        with pytest.raises(VandVError, match="no reference"):
            ValidationComparison(
                quantity="divot_depth_m",
                unit="m",
                simulation_value=1.0,
                experiment_value=1.0,
                numerical=NumericalUncertainty(u_h=0.0),
                u_input=0.0,
                u_exp=0.1,
                reference="   ",
            )

    def test_an_empty_report_is_refused(self) -> None:
        """An empty validation report is not the same as a passing one."""
        with pytest.raises(VandVError, match="at least one comparison"):
            ValidationReport(())


class TestThePlateResponseComparison:
    """The only validation that can be formed -- and it is noise-limited."""

    def test_it_is_indeterminate_as_published(self) -> None:
        as_published, _ = plate_response_comparisons()
        result = validate(as_published)
        assert result.is_indeterminate

    def test_it_stays_noise_limited_even_granting_an_exact_measurement(self) -> None:
        """The stronger statement: the flattering assumption does not help."""
        _, granted_exact = plate_response_comparisons()
        result = validate(granted_exact)
        assert result.noise_limited
        assert abs(result.comparison_error) <= (result.u_val or 0.0)

    def test_the_report_finds_nothing_informative(self) -> None:
        report = validation_report(list(plate_response_comparisons()))
        assert report.informative_results == ()
        assert report.noise_limited_fraction == pytest.approx(0.5)
        assert report.indeterminate_fraction == pytest.approx(0.5)

    def test_the_agreement_that_looks_good_is_about_five_percent(self) -> None:
        """The number the solver's docstring calls a 4.6% cross-check."""
        _, granted_exact = plate_response_comparisons()
        result = validate(granted_exact)
        assert result.relative_error is not None
        assert 0.03 < abs(result.relative_error) < 0.08

    def test_the_friction_angle_leverage_is_what_makes_it_noise_limited(self) -> None:
        """The cubic moves 12-13% per degree; the agreement is 5%.

        A model whose answer moves more per degree of an input than the
        gap being celebrated cannot be confirmed by that gap.
        """
        leverage = friction_angle_leverage_per_degree()
        assert 0.12 < leverage < 0.13
        _, granted_exact = plate_response_comparisons()
        result = validate(granted_exact)
        assert leverage > abs(result.relative_error or 0.0)

    def test_the_prediction_is_reproducible_from_the_published_inputs(self) -> None:
        _, granted_exact = plate_response_comparisons()
        assert granted_exact.simulation_value == pytest.approx(
            predicted_alpha_z_n_per_cm3(
                bulk_density_kg_m3=0.6 * 2600.0, friction_angle_deg=34.0
            )
        )
        assert granted_exact.experiment_value == QUIKRETE_MEASURED_ALPHA_Z_N_PER_CM3

    def test_the_numerical_uncertainty_is_legitimately_zero(self) -> None:
        """A uniform traction over a flat plate has no quadrature error."""
        as_published, _ = plate_response_comparisons()
        assert as_published.numerical.total == 0.0


class TestCarryCorrelationMachineryIsReadyButUnused:
    """Wivou's correlations: the metric exists, the model input does not."""

    def test_the_experimental_uncertainty_comes_from_a_fisher_interval(self) -> None:
        uncertainty = correlation_standard_uncertainty(
            correlation_r=-0.98, n_samples=55, n_controlled_variables=2
        )
        expected = (1.0 - 0.98**2) / math.sqrt(50)
        assert uncertainty == pytest.approx(expected)

    def test_a_correlation_at_the_boundary_is_refused(self) -> None:
        with pytest.raises(VerificationError, match="strictly inside"):
            correlation_standard_uncertainty(correlation_r=-1.0, n_samples=55)

    def test_too_few_samples_leave_no_degrees_of_freedom(self) -> None:
        with pytest.raises(VerificationError, match="no degrees of freedom"):
            correlation_standard_uncertainty(correlation_r=-0.5, n_samples=3)

    def test_a_factor_with_no_published_correlation_is_refused(self) -> None:
        with pytest.raises(VerificationError, match="publishes no carry correlation"):
            carry_correlation_comparison(
                factor="attack_angle_rad", model_correlation_r=-0.5
            )

    def test_the_comparison_records_that_it_has_not_been_run(self) -> None:
        """The notes must say the model correlation does not exist yet."""
        comparison = carry_correlation_comparison(
            factor="divot_depth_m", model_correlation_r=-0.9
        )
        assert any("has not been run" in note for note in comparison.notes)
        assert comparison.experiment_value == WIVOU_2016.correlations["divot_depth_m"]

    def test_the_assumed_degrees_of_freedom_are_disclosed(self) -> None:
        comparison = carry_correlation_comparison(
            factor="divot_depth_m", model_correlation_r=-0.9
        )
        assert any(
            "controlled variable(s) assumed" in note for note in comparison.notes
        )


class TestTheSmokeAnchorIsNotValidation:
    """1550 N is an analytic estimate, not a measurement, and is labelled so."""

    def test_the_sole_force_is_the_right_order_of_magnitude(
        self, default_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        case = quasi_static_plate_case(
            material, area_m2=20e-3 * 80e-3, depth_m=0.040, speed_m_s=25.0
        )
        force = default_solver.solve(case.state()).force_magnitude_n
        assert 0.5 * SMOKE_ANCHOR_N < force < 2.0 * SMOKE_ANCHOR_N

    def test_the_anchored_quantity_is_registered_as_unmeasured(self) -> None:
        """So no comparison against it can be dressed up as validation."""
        with pytest.raises(NoReferenceDataError):
            ValidationComparison(
                quantity="clubhead_deceleration_m_s2",
                unit="m/s^2",
                simulation_value=SMOKE_ANCHOR_N / 0.3,
                experiment_value=SMOKE_ANCHOR_N / 0.3,
                numerical=NumericalUncertainty(u_h=0.0),
                u_input=0.0,
                u_exp=0.0,
                reference="the addendum's C_d ~ 2 estimate, which is not data",
            )


class TestSolutionVerificationFeedsTheMetric:
    """``u_num`` must come from the GCI, not be invented."""

    def test_a_gci_derived_u_h_flows_into_u_val(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        study = surface_refinement_study(exact_solver, material)
        comparison = ValidationComparison(
            quantity="divot_depth_m",
            unit="m",
            simulation_value=1.0,
            experiment_value=1.0,
            numerical=NumericalUncertainty(u_h=study.numerical_uncertainty),
            u_input=0.0,
            u_exp=0.0,
            reference="synthetic fixture wiring the GCI into the metric",
        )
        result = validate(comparison)
        assert result.u_num == pytest.approx(study.numerical_uncertainty)
        assert result.u_val == pytest.approx(study.numerical_uncertainty)

    def test_the_intrusion_benchmark_records_the_speed_it_validates_at(self) -> None:
        """1.44 m/s against a 25 m/s design point: a factor of about 17."""
        assert GRANULAR_INTRUSION_BENCHMARK.max_speed_m_s == pytest.approx(1.44)
        assert 25.0 / GRANULAR_INTRUSION_BENCHMARK.max_speed_m_s > 15.0
