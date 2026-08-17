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
from bunkershot3d.solvers.mpm.constitutive import SandContinuum
from bunkershot3d.solvers.mpm.solver import PlaneStrainMPMSolver
from bunkershot3d.solvers.mpm.verification import (
    column_grid_convergence,
    cross_check_against_f0,
    elastic_column_equilibrium,
    energy_residuals,
    free_fall_residuals,
)
from bunkershot3d.vandv.conservation import ConservationClass
from bunkershot3d.vandv.convergence import observed_order_from_residuals
from bunkershot3d.vandv.exceptions import ConservationClassError

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def material() -> SandContinuum:
    return SandContinuum.from_sand_state(playing_condition(PlayingCondition.FIRM))


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
