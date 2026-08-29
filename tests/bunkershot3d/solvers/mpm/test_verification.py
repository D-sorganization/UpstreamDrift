"""Code verification for F1: conservation, an analytic case, GCI, and F0.

This module is the reason the tier is allowed to exist.  The
NASA-STD-7009B self-assessment records **validation at level 0 of 4**, so
a new solver that cannot demonstrate its own correctness makes that score
worse rather than better.

Note what each test is entitled to assert.  The conservation identities
are properties of the scheme, so they are asserted to round-off.  The
analytic case is asserted against a closed-form answer.  The F0
cross-check asserts only what a comparison between **two uncalibrated
models** can support -- that both oppose the motion, and that the
divergence is finite and reported -- because asserting agreement would be
claiming a validation that neither tier has.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bunkershot3d.sand import playing_condition
from bunkershot3d.sand.presets import PlayingCondition
from bunkershot3d.solvers import (
    DRFTSolver,
    IntrusionState,
    MaterialResponse,
    RefusalPolicy,
    SurfaceElements,
)
from bunkershot3d.solvers.exceptions import SolverInputError
from bunkershot3d.solvers.mpm.constitutive import SAND_POISSON_RATIO, SandContinuum
from bunkershot3d.solvers.mpm.solver import PlaneStrainMPMSolver
from bunkershot3d.solvers.mpm.limit_states import (
    passive_earth_pressure_limit,
    rankine_limits,
)
from bunkershot3d.solvers.mpm.order_of_accuracy import (
    ManufacturedField,
    column_temporal_convergence,
    manufactured_solution_convergence,
    uniform_stress_patch_residual,
)
from bunkershot3d.solvers.mpm.verification import (
    cohesive_elastic_strain_limit,
    cohesive_oscillation_residuals,
    column_grid_convergence,
    cross_check_against_f0,
    elastic_column_equilibrium,
    energy_residuals,
    free_fall_residuals,
)
from bunkershot3d.vandv.conservation import ConservationClass
from bunkershot3d.vandv.convergence import observed_order_from_residuals
from bunkershot3d.vandv.exceptions import ConservationClassError
from bunkershot3d.vandv.gci import ConvergenceType

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def material() -> SandContinuum:
    return SandContinuum.from_sand_state(playing_condition(PlayingCondition.FIRM))


@pytest.fixture(scope="module")
def cohesionless() -> SandContinuum:
    """A sand whose cone tip sits at the origin.

    The Rankine limits and the manufactured solution both want the
    friction cone tested on its own, and the ``FLUFFY`` preset is
    genuinely cohesionless -- its water content puts the moisture model
    at zero apparent cohesion, so ``tip_volumetric_strain`` is exactly
    zero rather than merely small.  Nothing is stripped or overridden to
    get that.
    """
    return SandContinuum.from_sand_state(playing_condition(PlayingCondition.FLUFFY))


def wedge_state(speed_m_s: float = 12.0, attack_deg: float = 20.0) -> IntrusionState:
    """A 40 x 16 mm sole section entering at a stated attack angle."""
    corners = np.array(
        [
            [-0.020, 0.0, -0.008],
            [0.020, 0.0, -0.008],
            [0.020, 0.0, 0.008],
            [-0.020, 0.0, 0.008],
            [0.024, 0.0, -0.004],
        ]
    )
    normals = np.tile([0.0, 0.0, -1.0], (corners.shape[0], 1))
    areas = np.full(corners.shape[0], 4.0e-4)
    angle = np.radians(attack_deg)
    velocity = (
        speed_m_s * np.cos(angle),
        0.0,
        -speed_m_s * np.sin(angle),
    )
    return IntrusionState(
        SurfaceElements(corners, normals, areas),
        velocity,
        free_surface_height_m=0.0,
    )


class TestMassAndMomentumConservation:
    """Identities of the scheme, so round-off class and no step size."""

    @pytest.fixture(scope="class")
    def residuals(self, material: SandContinuum):
        return free_fall_residuals(material, n_steps=40)

    def test_mass_is_conserved_exactly(self, residuals) -> None:
        mass = residuals[0]
        assert mass.conservation_class is ConservationClass.ROUND_OFF
        assert mass.residual == 0.0
        assert mass.within_round_off

    def test_momentum_matches_the_gravity_impulse(self, residuals) -> None:
        momentum = residuals[1]
        assert momentum.conservation_class is ConservationClass.ROUND_OFF
        assert momentum.within_round_off, momentum.summary()

    def test_a_round_off_residual_refuses_an_order_test(self, residuals) -> None:
        """The V&V machinery, not this suite, is what forbids it."""
        with pytest.raises(ConservationClassError):
            observed_order_from_residuals(list(residuals))


class TestEnergyConservation:
    """Truncation class: the order of the decay is the test, not a tolerance."""

    @pytest.fixture(scope="class")
    def residuals(self, material: SandContinuum):
        return energy_residuals(material)

    def test_every_residual_is_truncation_class(self, residuals) -> None:
        for residual in residuals:
            assert residual.conservation_class is ConservationClass.TRUNCATION
            assert residual.step_size_s is not None

    def test_a_truncation_residual_refuses_a_fixed_tolerance(self, residuals) -> None:
        with pytest.raises(ConservationClassError):
            _ = residuals[0].within_round_off

    def test_the_residual_decays_first_order_in_the_step(self, residuals) -> None:
        """Symplectic Euler loses exactly ``M g^2 dt^2 / 2`` per step.

        Over a fixed window that is ``M g^2 dt T / 2``, so the observed
        order should sit on 1 rather than merely be positive -- and it is
        asserted as a band, because "greater than zero" would pass for a
        scheme that had no order at all.
        """
        order = observed_order_from_residuals(list(residuals))
        assert 0.85 < order.order < 1.15, order.summary()

    def test_the_energy_error_is_small_at_the_finest_step(self, residuals) -> None:
        finest = min(residuals, key=lambda item: item.step_size_s or float("inf"))
        assert finest.relative < 1e-2, finest.summary()


class TestElasticColumn:
    """The analytic case: 1-D consolidation, found rather than seeded."""

    @pytest.fixture(scope="class")
    def column(self, material: SandContinuum):
        return elastic_column_equilibrium(material, cell_size_m=0.003)

    def test_it_relaxed(self, column) -> None:
        assert column.relaxed_fraction < 1e-3, column.summary()

    def test_the_equilibrium_is_elastic(self, column) -> None:
        """So the analytic elastic answer is exact, not an approximation."""
        assert column.n_yielded == 0, column.summary()

    def test_it_finds_the_closed_form_stress(self, column) -> None:
        assert column.relative_error < 0.05, column.summary()

    def test_it_finds_the_closed_form_strain_energy(self, column) -> None:
        """The elastic pathway, checked statically.

        A conservative *elastic* case does not exist for cohesionless
        sand -- anything that rebounds goes into tension and yields at the
        cone tip -- so the stored energy is checked against
        ``W (rho g)^2 H^3 / (6 M)`` at equilibrium instead.
        """
        assert column.elastic_energy_relative_error < 0.10, (
            f"{column.summary()}; U = {column.elastic_energy_j_per_m:.6g} against "
            f"{column.analytic_elastic_energy_j_per_m:.6g} J/m analytic"
        )

    def test_the_stress_is_compressive(self, column) -> None:
        assert column.mean_vertical_stress_pa < 0.0

    def test_the_wall_bands_actually_land_on_the_column(self, column) -> None:
        """The failure this case had first, kept as a regression.

        A generously padded grid puts the "fixed base" two cells below the
        column and the confining side walls two cells outside it. The
        column then stands in free space, falls, and reports a mean
        stress of zero through an otherwise entirely plausible run.
        """
        assert abs(column.mean_vertical_stress_pa) > 1.0

    def test_a_deeper_column_carries_more_stress(self, material: SandContinuum) -> None:
        """A sanity direction the analytic formula alone would not catch."""
        shallow = elastic_column_equilibrium(
            material, cell_size_m=0.004, height_m=0.024
        )
        deep = elastic_column_equilibrium(material, cell_size_m=0.004, height_m=0.048)
        assert deep.mean_vertical_stress_pa < shallow.mean_vertical_stress_pa


class TestGridConvergence:
    """Reported through the existing Celik implementation, not a new one."""

    @pytest.fixture(scope="class")
    def study(self, material: SandContinuum):
        return column_grid_convergence(material)

    def test_three_levels_were_solved(self, study) -> None:
        levels, _, _ = study
        assert len(levels) == 3
        assert len({level.n_particles for level in levels}) == 3

    def test_the_error_falls_under_refinement(self, study) -> None:
        levels, _, _ = study
        errors = [level.absolute_error_pa for level in levels]
        assert errors[0] > errors[-1], [level.summary() for level in levels]

    def test_a_first_order_or_better_rate_is_observed(self, study) -> None:
        _, order, _ = study
        assert order.order > 0.8, order.summary()
        assert order.monotone, order.summary()

    def test_the_gci_is_reported_and_bounded(self, study) -> None:
        _, _, gci = study
        assert np.isfinite(gci.gci_fine)
        assert gci.gci_fine < 0.10, gci.summary()

    def test_the_convergence_is_monotonic(self, study) -> None:
        """Oscillatory convergence would make the Richardson band a fiction."""
        _, _, gci = study
        assert not gci.is_oscillatory, gci.summary()

    def test_the_study_reuses_the_repo_celik_machinery(self, study) -> None:
        _, _, gci = study
        assert gci.n_grids == 3
        assert gci.quantity.startswith("F1 elastic column")

    def test_fewer_than_three_grids_is_refused(self, material: SandContinuum) -> None:
        from bunkershot3d.solvers.exceptions import SolverInputError

        with pytest.raises(SolverInputError, match="at least three grids"):
            column_grid_convergence(material, cell_sizes_m=(0.006, 0.004))


class TestF0CrossCheck:
    """A consistency check between two uncalibrated models, and nothing more."""

    @pytest.fixture(scope="class")
    def comparison(self, material: SandContinuum):
        sand = playing_condition(PlayingCondition.FIRM)
        f0 = DRFTSolver(
            material=MaterialResponse.from_sand_state(sand),
            refusal_policy=RefusalPolicy.REPORT,
        )
        f1 = PlaneStrainMPMSolver(
            material=material,
            cell_size_m=0.004,
            effective_width_m=0.030,
            bed_depth_m=0.06,
            refusal_policy=RefusalPolicy.REPORT,
            max_steps=4000,
        )
        return cross_check_against_f0(wedge_state(), f0, f1)

    def test_both_tiers_oppose_the_motion(self, comparison) -> None:
        """The one thing a comparison of two uncalibrated models can assert."""
        velocity = np.array([np.cos(np.radians(20.0)), 0.0, -np.sin(np.radians(20.0))])
        assert float(comparison.f0_force_n @ velocity) < 0.0, comparison.summary()
        assert float(comparison.f1_force_n @ velocity) < 0.0, comparison.summary()

    def test_the_divergence_is_finite_and_reported(self, comparison) -> None:
        assert np.isfinite(comparison.magnitude_ratio)
        assert comparison.magnitude_ratio > 0.0
        assert -1.0 <= comparison.direction_agreement <= 1.0

    def test_f1_produces_a_divot_that_f0_cannot(self, comparison) -> None:
        """The reason ADR-0033 chose a continuum at all."""
        assert comparison.f1_divot_depth_m > 0.0, comparison.summary()

    def test_both_split_their_force_into_two_parts(self, comparison) -> None:
        assert 0.0 <= comparison.f0_inertial_fraction <= 1.0
        assert 0.0 <= comparison.f1_flux_fraction <= 1.0

    def test_the_summary_names_the_declared_width(self, comparison) -> None:
        """No magnitude may be reproduced without its assumption."""
        assert "width=" in comparison.summary()
        assert comparison.effective_width_m == pytest.approx(0.030)


class TestRankineLimits:
    """The closed form is the *model's own* plane-strain Coulomb limit.

    The F1 cone is written on the two in-plane principal Kirchhoff
    stresses, so the plane-strain limit stress ratio it enforces is
    ``(1 - sqrt(2) alpha)/(1 + sqrt(2) alpha)`` -- a Rankine coefficient
    at an equivalent friction angle ``phi* = asin(sqrt(2) alpha)``, which
    is **not** the friction angle handed to
    :func:`~bunkershot3d.solvers.mpm.constitutive.drucker_prager_alpha`.
    Verifying against ``(1 - sin phi)/(1 + sin phi)`` at the *input*
    angle would be verifying the wrong number.
    """

    @pytest.fixture(scope="class")
    def limits(self, cohesionless: SandContinuum):
        return rankine_limits(cohesionless)

    def test_the_equivalent_angle_is_the_cone_slope_not_the_input(
        self, limits, cohesionless: SandContinuum
    ) -> None:
        assert limits.sin_phi_star == pytest.approx(np.sqrt(2.0) * cohesionless.alpha)
        assert limits.phi_star_deg < cohesionless.friction_angle_deg
        assert limits.phi_star_deg > 0.5 * cohesionless.friction_angle_deg

    def test_the_two_coefficients_are_reciprocal(self, limits) -> None:
        """``K_a K_p = 1`` is an identity of the Rankine pair."""
        assert limits.active_coefficient * limits.passive_coefficient == pytest.approx(
            1.0
        )

    def test_the_at_rest_state_lies_between_them(self, limits) -> None:
        """Otherwise the geostatic seed would already be at yield."""
        at_rest = SAND_POISSON_RATIO / (1.0 - SAND_POISSON_RATIO)
        assert limits.active_coefficient < at_rest < limits.passive_coefficient

    def test_a_cohesionless_sand_has_no_intercept(self, limits) -> None:
        assert limits.cone_tip_stress_pa == 0.0
        assert limits.passive_cohesive_intercept_pa == 0.0
        assert limits.active_cohesive_intercept_pa == 0.0

    def test_a_cohesive_sand_does(self, material: SandContinuum) -> None:
        cohesive = rankine_limits(material)
        assert cohesive.cone_tip_stress_pa > 0.0
        assert cohesive.passive_cohesive_intercept_pa > 0.0

    def test_a_cone_too_steep_for_a_rankine_state_is_refused(
        self, cohesionless: SandContinuum
    ) -> None:
        """``sqrt(2) alpha >= 1`` has no plane-strain Coulomb equivalent."""
        steep = replace(cohesionless, alpha=1.0)
        with pytest.raises(SolverInputError, match="no plane-strain"):
            rankine_limits(steep)


class TestPassiveEarthPressureLimit:
    """The plastic-limit case: a closed-form limit *load*, not an identity.

    Everything else in this module is elastic, kinematic, or a
    restatement of the yield function. This one drives the whole bed to
    the Drucker-Prager limit and compares the wall reaction the solver
    computes for itself against ``P_p = K_p rho g H^2 / 2``.
    """

    @pytest.fixture(scope="class")
    def limit(self, cohesionless: SandContinuum):
        return passive_earth_pressure_limit(cohesionless, cell_size_m=0.003)

    def test_the_bed_actually_reached_its_limit(self, limit) -> None:
        """A load read off an unmobilised bed is an elastic answer."""
        assert limit.yielded_fraction > 0.8, limit.summary()

    def test_the_push_was_quasi_static(self, limit) -> None:
        """Rankine is a static limit; an inertial answer is a different one."""
        assert limit.quasi_static_ratio < 1e-3, limit.summary()

    def test_it_finds_the_closed_form_limit_load(self, limit) -> None:
        assert limit.relative_error < 0.12, limit.summary()

    def test_the_thrust_is_far_above_the_at_rest_value(self, limit) -> None:
        """The failure mode this case has: reporting ``K_0`` as ``K_p``."""
        at_rest = SAND_POISSON_RATIO / (1.0 - SAND_POISSON_RATIO)
        at_rest_thrust = (
            at_rest * limit.analytic_thrust_n_per_m / limit.passive_coefficient
        )
        assert limit.thrust_n_per_m > 3.0 * at_rest_thrust, limit.summary()

    def test_none_of_the_load_is_cohesive_here(self, limit) -> None:
        """So the number is a test of the friction cone, not of the tip."""
        assert limit.cohesive_share == 0.0, limit.summary()

    def test_the_error_falls_when_the_grid_is_refined(
        self, cohesionless: SandContinuum, limit
    ) -> None:
        coarse = passive_earth_pressure_limit(cohesionless, cell_size_m=0.004)
        assert coarse.relative_error > limit.relative_error, (
            f"{coarse.summary()} / {limit.summary()}"
        )

    def test_a_wall_that_is_not_quasi_static_is_refused(
        self, cohesionless: SandContinuum
    ) -> None:
        with pytest.raises(SolverInputError, match="quasi-static"):
            passive_earth_pressure_limit(
                cohesionless, cell_size_m=0.004, wall_speed_m_s=20.0
            )


class TestManufacturedSolution:
    """MMS on the stress divergence and the transfer, taken together."""

    @pytest.fixture(scope="class")
    def patch(self, cohesionless: SandContinuum):
        return uniform_stress_patch_residual(cohesionless, cell_size_m=0.004)

    @pytest.fixture(scope="class")
    def study(self, cohesionless: SandContinuum):
        return manufactured_solution_convergence(cohesionless)

    def test_a_uniform_stress_field_produces_no_net_force(self, patch) -> None:
        """The patch test: an identity of the scheme, so round-off class."""
        assert patch.conservation_class is ConservationClass.ROUND_OFF
        assert patch.within_round_off, patch.summary()

    def test_the_patch_residual_refuses_an_order_test(self, patch) -> None:
        with pytest.raises(ConservationClassError):
            observed_order_from_residuals([patch, patch])

    def test_the_manufactured_error_is_not_round_off(self, study) -> None:
        """So a fixed tolerance would be the wrong test for it."""
        assert study.levels[0].relative_error > 1e-3, study.summary()

    def test_the_solution_stays_on_the_elastic_branch(self, study) -> None:
        """Otherwise the residual would be plastic dissipation."""
        for level in study.levels:
            assert level.n_yielded == 0, level.summary()
            assert level.worst_yield_pa < 0.0, level.summary()

    def test_the_observed_order_matches_the_design_order(self, study) -> None:
        assert study.design_order == 2.0
        assert 1.7 < study.observed_order.order < 2.2, study.summary()

    def test_the_error_falls_monotonically(self, study) -> None:
        assert study.observed_order.monotone, study.summary()
        assert study.observed_order.spread < 0.3, study.summary()

    def test_the_study_names_what_it_does_not_cover(self, study) -> None:
        """An honest partial: the plastic branch is not in this one."""
        assert "elastic" in study.summary()

    def test_a_field_that_yields_is_refused(self, cohesionless: SandContinuum) -> None:
        """A manufactured field outside the cone measures the return map."""
        tensile = ManufacturedField(
            size_m=0.048, amplitude_x=1.0e-4, amplitude_z=0.7e-4, mean_compression=0.0
        )
        with pytest.raises(SolverInputError, match="yield"):
            manufactured_solution_convergence(
                cohesionless, cell_sizes_m=(0.004, 0.003, 0.002), field=tensile
            )


class TestTemporalConvergence:
    """The step is refined on a *fixed* grid, through the same Celik code."""

    @pytest.fixture(scope="class")
    def study(self, material: SandContinuum):
        return column_temporal_convergence(material, transits=1.0)

    def test_three_steps_were_solved_at_one_cell_size(self, study) -> None:
        assert len(study.levels) == 3
        assert len({level.time_step_s for level in study.levels}) == 3
        assert len({level.n_steps for level in study.levels}) == 3

    def test_the_refinement_is_temporal_only(self, study) -> None:
        """A study that refined dx too would not isolate the step."""
        assert study.cell_size_m > 0.0
        assert "dt" in study.summary()

    def test_the_convergence_is_monotonic(self, study) -> None:
        assert study.converging, study.summary()
        assert not study.gci.is_oscillatory, study.summary()

    def test_an_order_of_at_least_one_is_observed(self, study) -> None:
        """The scheme is formally first order in the step."""
        assert 0.8 < study.apparent_order < 2.2, study.summary()
        assert 0.8 < study.difference_order.order < 2.2, study.summary()

    def test_the_temporal_gci_is_reported_and_small(self, study) -> None:
        assert np.isfinite(study.gci.gci_fine)
        assert study.gci.gci_fine < 0.02, study.summary()
        assert study.gci.n_grids == 3

    def test_the_band_is_declared_temporal_only(self, study) -> None:
        """It is not the total numerical uncertainty and must not read as one."""
        assert "temporal" in study.gci.quantity

    def test_the_relaxation_stays_elastic(self, study) -> None:
        for level in study.levels:
            assert level.n_yielded == 0, study.summary()

    def test_a_long_window_stops_converging_and_says_so(
        self, material: SandContinuum
    ) -> None:
        """The finding, not a failure: the transfer's cost is per step.

        The particle-grid round trip loses a fixed amount each step, so
        over a *fixed* physical window its total grows as the step is
        refined. Past about one elastic transit it overtakes the
        integrator's own ``O(dt)`` error and the series diverges.
        """
        long_window = column_temporal_convergence(material, transits=4.0)
        assert not long_window.converging, long_window.summary()
        assert long_window.gci.convergence is ConvergenceType.MONOTONIC_DIVERGENCE


class TestCohesiveElasticOscillation:
    """The conservative elastic case #8733 says "was not attempted".

    It is attempted here and it does not work, for a *different* reason
    than the cohesionless case does. The plasticity is genuinely gone --
    zero particles yield, and the function refuses to return if one
    does -- and the energy still drifts by about a tenth of the block's
    total without decaying with the step. What that identifies is the
    particle-grid transfer rather than the integrator, which is the same
    mechanism ``column_temporal_convergence`` finds over a long window.

    These tests pin the negative. If the order ever climbs to one, the
    transfer has changed and the tier's energy story wants rewriting
    rather than re-running.
    """

    @pytest.fixture(scope="class")
    def residuals(self, material: SandContinuum):
        return cohesive_oscillation_residuals(material)

    def test_a_cohesive_tip_buys_a_real_strain_budget(
        self, material: SandContinuum, cohesionless: SandContinuum
    ) -> None:
        budget = cohesive_elastic_strain_limit(material)
        assert budget > 0.0
        assert budget > 2.0 * 2.0e-5, budget
        assert cohesive_elastic_strain_limit(cohesionless) == 0.0

    def test_a_cohesionless_sand_is_refused_outright(
        self, cohesionless: SandContinuum
    ) -> None:
        """Running it there would report the plastic obstacle, not this one."""
        with pytest.raises(SolverInputError, match="no cohesive cone tip"):
            cohesive_oscillation_residuals(cohesionless)

    def test_an_amplitude_past_the_tip_is_refused(
        self, material: SandContinuum
    ) -> None:
        budget = cohesive_elastic_strain_limit(material)
        with pytest.raises(SolverInputError, match="initial_compression"):
            cohesive_oscillation_residuals(material, initial_compression=2.0 * budget)

    def test_every_residual_is_truncation_class(self, residuals) -> None:
        for residual in residuals:
            assert residual.conservation_class is ConservationClass.TRUNCATION
            assert residual.step_size_s is not None

    def test_the_oscillation_is_genuinely_elastic(self, residuals) -> None:
        """Zero yields is enforced by a raise, so reaching here is the check."""
        assert len(residuals) == 3

    def test_the_drift_does_not_decay_with_the_step(self, residuals) -> None:
        """The finding. Compare with 1.00 for the transfer-exact free fall."""
        order = observed_order_from_residuals(list(residuals))
        assert order.order < 0.5, order.summary()

    def test_the_drift_is_a_large_fraction_of_the_energy(self, residuals) -> None:
        """Not a small residual that merely fails to shrink."""
        for residual in residuals:
            assert residual.relative > 0.01, residual.summary()
