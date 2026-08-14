"""Solution verification: the Grid Convergence Index (issue #8616).

Celik et al. (2008). **No experimental data appears in this file.**

Two kinds of test here. The synthetic ones pin the algebra against
series with a known exact answer, so a defect in the fixed-point
iteration or the extrapolation cannot hide. The cylinder ones run the
GCI on the real solver where the exact answer is *also* known, which
turns the usual GCI leap of faith into a checkable claim: the estimate
must actually bound the error it is estimating.
"""

from __future__ import annotations

import pytest

from bunkershot3d.solvers import DRFTSolver, MaterialResponse
from bunkershot3d.vandv import (
    COMFORTABLE_REFINEMENT_RATIO,
    FACTOR_OF_SAFETY_THREE_GRID,
    FACTOR_OF_SAFETY_TWO_GRID,
    GCI_COVERAGE_FACTOR,
    ConvergenceType,
    GCIStudy,
    GridSolution,
    SolutionVerificationError,
    apparent_order,
    cylinder_case,
    error_amplification,
    grid_convergence_index,
    richardson_extrapolate,
    two_grid_gci,
)
from bunkershot3d.vandv.studies import surface_refinement_study

pytestmark = [pytest.mark.unit, pytest.mark.scientific]

EXACT_VALUE = 10.0
"""The limit of the synthetic series ``phi(h) = 10 + h^2``."""


def _synthetic_series(*, order: float = 2.0, ratio: float = 2.0) -> list[GridSolution]:
    """``phi(h) = EXACT + h^p`` on three grids refined by ``ratio``."""
    return [
        GridSolution(ratio**level, EXACT_VALUE + (ratio**level) ** order, f"h{level}")
        for level in range(3)
    ]


class TestErrorAmplification:
    """``r^p - 1``, and why the GCI divides by it."""

    def test_it_is_exactly_three_at_r_two_p_two(self) -> None:
        """The digest's headline: the bare difference over-states by 3x.

        ``phi2 - phi1`` spans the error on both grids, and the coarse
        error is ``r^p = 4`` times the fine one, so the difference is
        ``4 - 1 = 3`` times the fine-grid error.
        """
        assert error_amplification(2.0, 2.0) == pytest.approx(3.0)

    def test_a_ratio_of_one_is_refused(self) -> None:
        with pytest.raises(SolutionVerificationError, match="not refined"):
            error_amplification(1.0, 2.0)

    def test_a_zero_order_is_refused(self) -> None:
        """``r^0 - 1 = 0``: no error estimate can be formed."""
        with pytest.raises(SolutionVerificationError, match="non-positive"):
            error_amplification(2.0, 0.0)


class TestApparentOrder:
    """Equation (3a), including the ``q(p)`` term and its fixed point."""

    def test_a_uniform_refinement_recovers_the_known_order(self) -> None:
        fine, medium, coarse = _synthetic_series(order=2.0)
        observed = apparent_order(
            epsilon_21=medium.value - fine.value,
            epsilon_32=coarse.value - medium.value,
            r21=2.0,
            r32=2.0,
        )
        assert observed.order == pytest.approx(2.0, abs=1e-9)
        assert observed.convergence is ConvergenceType.MONOTONIC

    def test_the_q_term_vanishes_when_the_two_ratios_match(self) -> None:
        """``ln[(r^p - s)/(r^p - s)] = 0``, so one pass is exact."""
        fine, medium, coarse = _synthetic_series(order=3.0)
        observed = apparent_order(
            epsilon_21=medium.value - fine.value,
            epsilon_32=coarse.value - medium.value,
            r21=2.0,
            r32=2.0,
        )
        assert observed.order == pytest.approx(3.0, abs=1e-9)
        assert observed.iterations <= 2

    def test_a_non_uniform_refinement_needs_the_q_term_and_converges(self) -> None:
        """``h = (1, 2, 6)`` with ``phi = 10 + h^2`` must still give ``p = 2``.

        Dropping ``q(p)`` here would give ``ln(32/3)/ln 2 = 3.41``, so
        this test is exactly the one that fails if the term is omitted.
        """
        observed = apparent_order(
            epsilon_21=(10.0 + 4.0) - (10.0 + 1.0),
            epsilon_32=(10.0 + 36.0) - (10.0 + 4.0),
            r21=2.0,
            r32=3.0,
        )
        assert observed.order == pytest.approx(2.0, abs=1e-6)
        assert observed.converged

    def test_oscillatory_convergence_is_classified_not_hidden(self) -> None:
        observed = apparent_order(epsilon_21=3.0, epsilon_32=-6.0, r21=2.0, r32=2.0)
        assert observed.convergence is ConvergenceType.OSCILLATORY
        assert observed.convergence.is_oscillatory
        assert not observed.convergence.supports_richardson

    def test_oscillatory_divergence_is_a_separate_class(self) -> None:
        observed = apparent_order(epsilon_21=6.0, epsilon_32=-3.0, r21=2.0, r32=2.0)
        assert observed.convergence is ConvergenceType.OSCILLATORY_DIVERGENCE

    def test_monotonic_divergence_is_a_separate_class(self) -> None:
        observed = apparent_order(epsilon_21=6.0, epsilon_32=3.0, r21=2.0, r32=2.0)
        assert observed.convergence is ConvergenceType.MONOTONIC_DIVERGENCE

    def test_identical_fine_solutions_are_refused(self) -> None:
        with pytest.raises(SolutionVerificationError, match="no order of accuracy"):
            apparent_order(epsilon_21=0.0, epsilon_32=1.0, r21=2.0, r32=2.0)

    def test_a_ratio_at_or_below_one_is_refused(self) -> None:
        with pytest.raises(SolutionVerificationError, match="must exceed 1"):
            apparent_order(epsilon_21=1.0, epsilon_32=2.0, r21=1.0, r32=2.0)


class TestRichardsonExtrapolation:
    """``phi_ext = (r^p phi1 - phi2)/(r^p - 1)``."""

    def test_it_recovers_the_exact_limit_of_a_pure_power_law(self) -> None:
        fine, medium, _ = _synthetic_series(order=2.0)
        extrapolated = richardson_extrapolate(
            fine_value=fine.value,
            coarse_value=medium.value,
            refinement_ratio=2.0,
            order=2.0,
        )
        assert extrapolated == pytest.approx(EXACT_VALUE, abs=1e-12)


class TestThreeGridGCI:
    """The full procedure, ``Fs = 1.25``."""

    def test_the_factor_of_safety_is_the_three_grid_value(self) -> None:
        result = grid_convergence_index(_synthetic_series())
        assert result.factor_of_safety == FACTOR_OF_SAFETY_THREE_GRID
        assert FACTOR_OF_SAFETY_THREE_GRID == 1.25

    def test_the_order_is_observed_not_assumed(self) -> None:
        result = grid_convergence_index(_synthetic_series(order=2.0))
        assert not result.order_assumed
        assert result.apparent_order == pytest.approx(2.0, abs=1e-9)

    def test_the_gci_exceeds_the_true_error_by_exactly_the_safety_factor(
        self,
    ) -> None:
        """On a pure power law, Richardson is exact, so the margin is ``Fs``."""
        result = grid_convergence_index(_synthetic_series(order=2.0))
        true_relative_error = abs(1.0 / (EXACT_VALUE + 1.0))
        assert result.gci_fine == pytest.approx(
            FACTOR_OF_SAFETY_THREE_GRID * true_relative_error, rel=1e-9
        )

    def test_two_grids_are_refused(self) -> None:
        with pytest.raises(SolutionVerificationError, match="at least three"):
            grid_convergence_index(_synthetic_series()[:2])

    def test_a_zero_fine_value_is_refused(self) -> None:
        """A relative GCI at zero is undefined, and must say so."""
        solutions = [
            GridSolution(1.0, 0.0),
            GridSolution(2.0, 1.0),
            GridSolution(4.0, 4.0),
        ]
        with pytest.raises(SolutionVerificationError, match="undefined"):
            grid_convergence_index(solutions)

    def test_the_standard_uncertainty_is_the_expanded_band_over_k(self) -> None:
        """``u_h = GCI |phi1| / 2``, so V&V 20's ``k`` is not applied twice."""
        result = grid_convergence_index(_synthetic_series())
        assert result.standard_numerical_uncertainty == pytest.approx(
            result.expanded_numerical_uncertainty / GCI_COVERAGE_FACTOR
        )
        assert GCI_COVERAGE_FACTOR == 2.0


class TestTwoGridGCI:
    """Two grids, ``Fs = 3.0``, because ``p`` had to be assumed."""

    def test_the_factor_of_safety_triples(self) -> None:
        fine, coarse, _ = _synthetic_series()
        result = two_grid_gci(fine, coarse, assumed_order=2.0)
        assert result.factor_of_safety == FACTOR_OF_SAFETY_TWO_GRID
        assert FACTOR_OF_SAFETY_TWO_GRID == 3.0

    def test_the_order_is_recorded_as_assumed(self) -> None:
        fine, coarse, _ = _synthetic_series()
        result = two_grid_gci(fine, coarse, assumed_order=2.0)
        assert result.order_assumed

    def test_it_is_two_point_four_times_the_three_grid_estimate(self) -> None:
        """``3.0 / 1.25``: the price of not observing the order."""
        solutions = _synthetic_series()
        three = grid_convergence_index(solutions)
        two = two_grid_gci(solutions[0], solutions[1], assumed_order=2.0)
        assert two.gci_fine == pytest.approx(2.4 * three.gci_fine, rel=1e-9)

    def test_grids_supplied_the_wrong_way_round_are_refused(self) -> None:
        fine, coarse, _ = _synthetic_series()
        with pytest.raises(SolutionVerificationError, match="larger cell size"):
            two_grid_gci(coarse, fine, assumed_order=2.0)

    def test_a_non_positive_assumed_order_is_refused(self) -> None:
        fine, coarse, _ = _synthetic_series()
        with pytest.raises(SolutionVerificationError, match="assumed order"):
            two_grid_gci(fine, coarse, assumed_order=0.0)


class TestOscillationIsReportedAsAPercentage:
    """A study must not quietly drop the quantities that misbehaved."""

    def test_a_mixed_study_reports_the_oscillatory_share(self) -> None:
        monotonic = grid_convergence_index(_synthetic_series())
        oscillatory = grid_convergence_index(
            [
                GridSolution(1.0, 11.0),
                GridSolution(2.0, 14.0),
                GridSolution(4.0, 8.0),
            ]
        )
        study = GCIStudy((monotonic, oscillatory))
        assert study.oscillatory_percentage == pytest.approx(50.0)
        assert study.oscillatory_fraction == pytest.approx(0.5)

    def test_an_oscillatory_result_carries_no_richardson_estimate(self) -> None:
        result = grid_convergence_index(
            [
                GridSolution(1.0, 11.0),
                GridSolution(2.0, 14.0),
                GridSolution(4.0, 8.0),
            ]
        )
        assert result.is_oscillatory
        assert result.extrapolated_value is None
        assert "oscillatory convergence" in result.summary()

    def test_an_empty_study_is_refused(self) -> None:
        with pytest.raises(SolutionVerificationError, match="at least one quantity"):
            GCIStudy(())


class TestGCIOnTheRealSolver:
    """Run the GCI where the exact answer is also known, and check it bounds."""

    def test_the_apparent_order_matches_the_quadrature_order(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        study = surface_refinement_study(exact_solver, material)
        assert study.results[0].apparent_order == pytest.approx(2.0, abs=0.05)

    def test_convergence_is_monotonic_and_none_of_it_oscillates(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        study = surface_refinement_study(exact_solver, material)
        assert study.results[0].convergence is ConvergenceType.MONOTONIC
        assert study.oscillatory_percentage == 0.0

    def test_the_gci_actually_bounds_the_error_it_estimates(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        """The claim a GCI normally has to be taken on trust.

        The cylinder case has a closed form, so the true fine-grid error
        is known and the estimate can be held to it: the GCI must exceed
        the true error, and must not exceed it by an absurd margin.
        """
        counts = (64, 128, 256, 512)
        study = surface_refinement_study(exact_solver, material, facet_counts=counts)
        result = study.results[0]
        finest = cylinder_case(material, n_facets=counts[-1])
        true_relative = abs(
            (result.fine_value - finest.exact_inertial_force_x_n)
            / finest.exact_inertial_force_x_n
        )
        assert result.gci_fine > true_relative
        assert result.gci_fine < 5.0 * true_relative

    def test_the_extrapolated_value_beats_the_finest_grid(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        counts = (64, 128, 256, 512)
        study = surface_refinement_study(exact_solver, material, facet_counts=counts)
        result = study.results[0]
        exact = cylinder_case(material, n_facets=counts[-1]).exact_inertial_force_x_n
        assert result.extrapolated_value is not None
        extrapolated_error = abs(result.extrapolated_value - exact)
        fine_error = abs(result.fine_value - exact)
        assert extrapolated_error < 0.01 * fine_error

    def test_the_refinement_ratio_clears_celiks_comfort_threshold(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        study = surface_refinement_study(exact_solver, material)
        assert study.results[0].comfortable_refinement
        assert study.results[0].refinement_ratio > COMFORTABLE_REFINEMENT_RATIO

    def test_the_study_yields_a_numerical_uncertainty_for_the_metric(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        """``u_num`` is what solution verification exists to produce."""
        study = surface_refinement_study(exact_solver, material)
        assert study.numerical_uncertainty > 0.0
        assert study.numerical_uncertainty < abs(study.results[0].fine_value)
