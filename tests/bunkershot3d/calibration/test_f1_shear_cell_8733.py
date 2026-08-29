"""Tests for the F1 drained shear cell (issue #8733 section 6).

Section 6 records that F1's constitutive model has never been calibrated:
its friction angle is still borrowed from the Quikrete analogue (#7999)
and its shear modulus is a Hardin & Richart (1963) estimate.  ADR-0033
chose MPM because F1 and the F2 reference share that constitutive model,
so one calibration would carry between the tiers -- a rationale that was
unrealised because nothing in ``bunkershot3d.calibration`` referenced
:class:`~bunkershot3d.solvers.mpm.constitutive.SandContinuum` at all.

**The targets these tests drive are declared numbers, not measurements of
real bunker sand.**  Every assertion below is about self-consistency
between the constitutive model and a stated target.  None of them is
evidence that the model describes a golf bunker.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from bunkershot3d.calibration.f1_shear_cell import (
    PLANE_STRAIN_CROSSOVER_ANGLE_DEG,
    F1_SHEAR_CELL_CONFINING_STRESSES_PA,
    F1DrainedShearCellExperiment,
    MohrCoulombEnvelope,
    drained_biaxial_path,
    plane_strain_friction_angle_deg,
)
from bunkershot3d.calibration.optimizer import (
    CalibrationOptimizer,
    InertParameterError,
)
from bunkershot3d.sand.presets import PlayingCondition, playing_condition
from bunkershot3d.solvers.exceptions import CalibrationError
from bunkershot3d.solvers.mpm.constitutive import SandContinuum

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def cohesionless_sand():
    """The fluffy preset: the only one whose cone tip sits at the origin."""
    return playing_condition(PlayingCondition.FLUFFY)


@pytest.fixture(scope="module")
def cohesionless_material(cohesionless_sand) -> SandContinuum:
    return SandContinuum.from_sand_state(cohesionless_sand)


class TestPlaneStrainFrictionAngle:
    """``phi* = asin(sqrt(2) alpha)`` is the angle F1 actually enforces."""

    def test_matches_the_limit_state_module(
        self, cohesionless_material: SandContinuum
    ) -> None:
        expected = math.degrees(math.asin(math.sqrt(2) * cohesionless_material.alpha))
        got = plane_strain_friction_angle_deg(34.0)
        if abs(got - expected) > 1e-10:
            raise AssertionError(f"{got} != {expected}")

    def test_is_softer_below_the_crossover(self) -> None:
        """The inner-cone match is a 3-D fit; in plane strain it is softer.

        Only below :data:`PLANE_STRAIN_CROSSOVER_ANGLE_DEG`, which is
        where every sand preset and every plausible fit sits.
        """
        for phi in (20.0, 34.0, 40.0):
            if not plane_strain_friction_angle_deg(phi) < phi:
                raise AssertionError(f"phi* not below phi at {phi} deg")

    def test_is_stronger_above_the_crossover(self) -> None:
        """Stated rather than assumed: the inequality reverses at 43.68 deg."""
        for phi in (50.0, 60.0):
            if not plane_strain_friction_angle_deg(phi) > phi:
                raise AssertionError(f"phi* not above phi at {phi} deg")

    def test_the_crossover_is_a_fixed_point(self) -> None:
        crossover = PLANE_STRAIN_CROSSOVER_ANGLE_DEG
        if abs(plane_strain_friction_angle_deg(crossover) - crossover) > 1.0e-9:
            raise AssertionError(
                f"phi*({crossover}) = {plane_strain_friction_angle_deg(crossover)}"
            )

    def test_is_strictly_increasing(self) -> None:
        angles = [plane_strain_friction_angle_deg(p) for p in (20.0, 30.0, 40.0, 50.0)]
        if not all(b > a for a, b in zip(angles, angles[1:], strict=False)):
            raise AssertionError(f"phi* is not monotone: {angles}")

    def test_rejects_an_unusable_angle(self) -> None:
        with pytest.raises(CalibrationError):
            plane_strain_friction_angle_deg(0.0)

    def test_refuses_an_angle_whose_cone_is_too_steep(self) -> None:
        """``sqrt(2) alpha > 1`` has no ``asin``; it must not return a nan."""
        with pytest.raises(CalibrationError):
            plane_strain_friction_angle_deg(89.9)


class TestDrainedBiaxialPath:
    """The element test itself, driven through F1's own return map."""

    def test_reaches_the_cone_and_stays_there(
        self, cohesionless_material: SandContinuum
    ) -> None:
        path = drained_biaxial_path(
            cohesionless_material, confining_stress_pa=1.0e4, axial_strain=-0.04
        )
        expected = plane_strain_friction_angle_deg(
            cohesionless_material.friction_angle_deg
        )
        if abs(path[-1].mobilised_friction_deg - expected) > 1.0e-6:
            raise AssertionError(
                f"end of path {path[-1].mobilised_friction_deg} != {expected}"
            )

    def test_is_monotone_up_to_the_plateau(
        self, cohesionless_material: SandContinuum
    ) -> None:
        path = drained_biaxial_path(
            cohesionless_material, confining_stress_pa=1.0e4, axial_strain=-0.04
        )
        ratios = [point.stress_ratio for point in path]
        if not all(b >= a - 1e-12 for a, b in zip(ratios, ratios[1:], strict=False)):
            raise AssertionError("the stress ratio is not monotone along the path")

    def test_holds_the_confining_stress(
        self, cohesionless_material: SandContinuum
    ) -> None:
        """It is a *drained* test: the lateral stress is the control."""
        confining = 1.0e4
        path = drained_biaxial_path(
            cohesionless_material, confining_stress_pa=confining, axial_strain=-0.04
        )
        worst = max(abs(p.lateral_stress_pa - confining) for p in path)
        if worst > confining * 1.0e-6:
            raise AssertionError(f"lateral stress drifted by {worst} Pa")

    def test_is_independent_of_the_confining_stress(
        self, cohesionless_material: SandContinuum
    ) -> None:
        """A cohesionless cone has a straight envelope through the origin."""
        angles = [
            drained_biaxial_path(
                cohesionless_material, confining_stress_pa=s, axial_strain=-0.04
            )[-1].mobilised_friction_deg
            for s in (1.0e3, 1.0e4, 1.0e5)
        ]
        if max(angles) - min(angles) > 1.0e-8:
            raise AssertionError(f"envelope is not straight: {angles}")

    def test_rejects_a_tensile_confining_stress(
        self, cohesionless_material: SandContinuum
    ) -> None:
        with pytest.raises(CalibrationError):
            drained_biaxial_path(
                cohesionless_material, confining_stress_pa=-1.0, axial_strain=-0.04
            )

    def test_rejects_an_extensional_axial_strain(
        self, cohesionless_material: SandContinuum
    ) -> None:
        with pytest.raises(CalibrationError):
            drained_biaxial_path(
                cohesionless_material, confining_stress_pa=1.0e4, axial_strain=0.04
            )


class TestMohrCoulombEnvelope:
    """The multi-confinement p-q fit a real shear cell reports."""

    def test_recovers_the_cone_of_a_cohesionless_sand(
        self, cohesionless_material: SandContinuum
    ) -> None:
        experiment = F1DrainedShearCellExperiment(
            sand=dataclasses.replace(
                playing_condition(PlayingCondition.FLUFFY), friction_angle_deg=34.0
            )
        )
        envelope = experiment.envelope({"friction_angle_deg": 34.0})
        if abs(envelope.friction_angle_deg - plane_strain_friction_angle_deg(34.0)) > (
            1.0e-6
        ):
            raise AssertionError(f"envelope angle {envelope.friction_angle_deg}")
        if abs(envelope.cohesion_pa) > 1.0e-6:
            raise AssertionError(f"cohesionless sand grew {envelope.cohesion_pa} Pa")

    def test_a_damp_sand_shows_a_cohesion_intercept(self) -> None:
        """The cone tip is real: the envelope must not be forced through zero."""
        experiment = F1DrainedShearCellExperiment(
            sand=playing_condition(PlayingCondition.FIRM)
        )
        envelope = experiment.envelope({"friction_angle_deg": 34.0})
        if envelope.cohesion_pa <= 0.0:
            raise AssertionError(
                f"a damp sand reported {envelope.cohesion_pa} Pa of cohesion"
            )

    def test_needs_two_distinct_confinements(self) -> None:
        with pytest.raises(CalibrationError):
            MohrCoulombEnvelope.from_points(np.array([1.0e4]), np.array([5.0e3]))


class TestExperimentAgainstTheHarness:
    """The optimiser must be able to drive it with no changes of its own."""

    def test_declares_only_the_friction_angle(self) -> None:
        experiment = F1DrainedShearCellExperiment()
        if experiment.calibrated_parameters != ("friction_angle_deg",):
            raise AssertionError(experiment.calibrated_parameters)

    def test_bounds_are_declared_for_every_searched_parameter(self) -> None:
        experiment = F1DrainedShearCellExperiment()
        for name in experiment.calibrated_parameters:
            if name not in experiment.parameter_bounds:
                raise AssertionError(f"{name} has no declared bounds")

    def test_the_friction_angle_moves_the_objective(self) -> None:
        optimizer = CalibrationOptimizer(F1DrainedShearCellExperiment())
        sensitivity = optimizer.check_sensitivity()
        if sensitivity["friction_angle_deg"] <= 0.0:
            raise AssertionError(sensitivity)

    def test_the_shear_modulus_is_inert_and_is_refused(self) -> None:
        """#7999's guard, applied to F1: the modulus cannot be identified.

        The drained limit ratio ``q/p`` is a *ratio* of stresses that are
        both linear in the elastic constants, so the modulus cancels
        exactly.  Declaring it would return optimiser noise.
        """
        experiment = F1DrainedShearCellExperiment(include_shear_modulus=True)
        optimizer = CalibrationOptimizer(experiment)
        with pytest.raises(InertParameterError):
            optimizer.check_sensitivity()

    def test_the_modulus_does_not_move_the_measured_angle(self) -> None:
        """The stronger statement, measured rather than thresholded."""
        experiment = F1DrainedShearCellExperiment()
        angles = [
            experiment.run_simulation(
                {"friction_angle_deg": 34.0, "shear_modulus_pa": modulus}
            )[0]
            for modulus in (1.0e6, 1.0e7, 1.0e8, 1.0e9)
        ]
        if max(angles) - min(angles) > 1.0e-8:
            raise AssertionError(f"the modulus moved the angle: {angles}")

    def test_peak_and_residual_coincide(self) -> None:
        """A perfectly plastic cone has no softening. This is a *finding*."""
        experiment = F1DrainedShearCellExperiment()
        peak, residual = experiment.run_simulation({"friction_angle_deg": 34.0})
        if abs(peak - residual) > 1.0e-9:
            raise AssertionError(
                f"peak {peak} and residual {residual} differ; the model was "
                "believed to have no softening mechanism"
            )

    def test_the_irreducible_residual_is_reported(self) -> None:
        """Half the objective can never be removed, and must be declared."""
        experiment = F1DrainedShearCellExperiment()
        gap = experiment.target_phi_peak - experiment.target_phi_res
        expected = 2.0 * (gap / 2.0) ** 2
        if abs(experiment.irreducible_residual_deg2 - expected) > 1.0e-12:
            raise AssertionError(experiment.irreducible_residual_deg2)

    def test_optimising_lands_on_the_midpoint_of_the_two_targets(self) -> None:
        """Because peak == residual, the best fit is their midpoint."""
        experiment = F1DrainedShearCellExperiment()
        fitted = experiment.fit_friction_angle_deg()
        midpoint = 0.5 * (experiment.target_phi_peak + experiment.target_phi_res)
        if abs(plane_strain_friction_angle_deg(fitted) - midpoint) > 1.0e-6:
            raise AssertionError(
                f"fitted {fitted} deg gives phi* "
                f"{plane_strain_friction_angle_deg(fitted)}, not {midpoint}"
            )

    def test_default_confinements_are_distinct_and_compressive(self) -> None:
        stresses = F1_SHEAR_CELL_CONFINING_STRESSES_PA
        if len(set(stresses)) != len(stresses):
            raise AssertionError(stresses)
        if any(value <= 0.0 for value in stresses):
            raise AssertionError(stresses)


class TestHonestyBoundary:
    """The fit is self-consistency, not validation. Say so in the code."""

    def test_the_target_is_labelled_as_declared_not_measured(self) -> None:
        experiment = F1DrainedShearCellExperiment()
        note = experiment.target_provenance_note.lower()
        for phrase in ("not a measurement", "bunker sand"):
            if phrase not in note:
                raise AssertionError(f"{phrase!r} missing from the target note")

    def test_the_experiment_refuses_to_claim_a_measurement(self) -> None:
        experiment = F1DrainedShearCellExperiment()
        if experiment.is_measured_on_bunker_sand:
            raise AssertionError("the shear cell claimed to be a measurement")
