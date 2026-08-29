"""Tests for the F1 angle-of-repose attempt (issue #8733 section 6).

The measurement this module's subject reports is a **negative result**:
F1's plane-strain MPM discretisation does not arrest at a repose angle, so
the experiment refuses to return one.  These tests pin that refusal in
place, because the tempting failure is to lengthen the run until a number
appears and then quote it.

Nothing here is a measurement of real bunker sand.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
from bunkershot3d.calibration.f1_repose import (
    F1_REPOSE_ARREST_NOTE,
    F1_REPOSE_MIN_FRICTION_ANGLE_DEG,
    F1AngleOfReposeExperiment,
    SlopeRelaxation,
    SlopeRelaxationSettings,
    SlopeSample,
    relax_slope,
    wedge_bed,
)
from bunkershot3d.sand.presets import PlayingCondition, playing_condition
from bunkershot3d.solvers.exceptions import CalibrationError
from bunkershot3d.solvers.mpm.constitutive import SandContinuum

pytestmark = pytest.mark.unit


#: Small enough to run in a unit lane, large enough that the slope fit has
#: real bins to work with. It is *not* a resolution any result should be
#: quoted at; the measured numbers in the module docstring are 4 mm runs.
_SMOKE = SlopeRelaxationSettings(
    cell_size_m=6.0e-3,
    height_m=0.036,
    plateau_length_m=0.024,
    initial_slope_deg=45.0,
    settle_time_s=0.02,
    n_strides=2,
    runout_margin_m=0.04,
)


@pytest.fixture(scope="module")
def cohesionless_material() -> SandContinuum:
    return SandContinuum.from_sand_state(playing_condition(PlayingCondition.FLUFFY))


class TestSettings:
    def test_toe_follows_from_the_slope(self) -> None:
        settings = SlopeRelaxationSettings(
            height_m=0.06, plateau_length_m=0.04, initial_slope_deg=45.0
        )
        if abs(settings.toe_x_m - 0.10) > 1.0e-12:
            raise AssertionError(settings.toe_x_m)

    def test_surface_is_a_plateau_then_a_slope(self) -> None:
        import numpy as np

        settings = SlopeRelaxationSettings(
            height_m=0.06, plateau_length_m=0.04, initial_slope_deg=45.0
        )
        heights = settings.surface_height_m(np.array([0.0, 0.04, 0.07, 0.20]))
        expected = [0.06, 0.06, 0.03, 0.0]
        for got, want in zip(heights, expected, strict=True):
            if abs(float(got) - want) > 1.0e-12:
                raise AssertionError(f"{heights} != {expected}")

    def test_refuses_a_single_stride(self) -> None:
        """Arrest is a comparison between two times; one sample cannot make it."""
        with pytest.raises(CalibrationError):
            SlopeRelaxationSettings(n_strides=1)

    @pytest.mark.parametrize("field_name", ["cell_size_m", "height_m", "settle_time_s"])
    def test_refuses_a_non_positive_dimension(self, field_name: str) -> None:
        with pytest.raises(CalibrationError):
            SlopeRelaxationSettings(**{field_name: 0.0})

    def test_refuses_a_vertical_face(self) -> None:
        with pytest.raises(CalibrationError):
            SlopeRelaxationSettings(initial_slope_deg=90.0)


class TestWedgeBed:
    def test_holds_particles_under_the_declared_surface(
        self, cohesionless_material: SandContinuum
    ) -> None:
        bed = wedge_bed(cohesionless_material, _SMOKE)
        if bed.n_particles < 50:
            raise AssertionError(f"only {bed.n_particles} particles")
        surface = _SMOKE.surface_height_m(bed.position_m[:, 0])
        if bool((bed.position_m[:, 1] >= surface).any()):
            raise AssertionError("a particle sits above the declared surface")

    def test_starts_at_rest(self, cohesionless_material: SandContinuum) -> None:
        bed = wedge_bed(cohesionless_material, _SMOKE)
        if float(abs(bed.velocity_m_s).max()) != 0.0:
            raise AssertionError("the wedge was seeded moving")

    def test_refuses_a_seed_outside_the_yield_surface(self) -> None:
        """Below ~26.35 deg the wedge cannot stand up at zero confinement."""
        sand = dataclasses.replace(
            playing_condition(PlayingCondition.FLUFFY), friction_angle_deg=20.0
        )
        material = SandContinuum.from_sand_state(sand)
        with pytest.raises(CalibrationError, match="yield surface"):
            wedge_bed(material, _SMOKE)

    def test_the_declared_lower_bound_is_admissible(self) -> None:
        """The bound must be a value the seed actually accepts."""
        sand = dataclasses.replace(
            playing_condition(PlayingCondition.FLUFFY),
            friction_angle_deg=F1_REPOSE_MIN_FRICTION_ANGLE_DEG,
        )
        wedge_bed(SandContinuum.from_sand_state(sand), _SMOKE)


class TestArrestTest:
    """``require_arrested`` on synthetic histories, so both branches are hit."""

    @staticmethod
    def _relaxation(first: float, second: float, span: float) -> SlopeRelaxation:
        return SlopeRelaxation(
            samples=(
                SlopeSample(
                    time_s=0.0,
                    slope_deg=first,
                    n_bins=8,
                    kinetic_energy_j=0.0,
                    toe_x_m=0.1,
                ),
                SlopeSample(
                    time_s=span,
                    slope_deg=second,
                    n_bins=8,
                    kinetic_energy_j=0.0,
                    toe_x_m=0.1,
                ),
            ),
            friction_angle_deg=34.0,
            plane_strain_limit_deg=31.94,
            n_particles=100,
            n_steps=100,
            time_step_s=1.0e-5,
            wall_clock_s=1.0,
        )

    def test_a_still_slope_returns_its_angle(self) -> None:
        relaxation = self._relaxation(32.0, 32.0, 0.5)
        if relaxation.require_arrested() != 32.0:
            raise AssertionError(relaxation.require_arrested())

    def test_a_moving_slope_refuses(self) -> None:
        relaxation = self._relaxation(35.0, 30.0, 0.5)
        with pytest.raises(CalibrationError, match="has not arrested"):
            relaxation.require_arrested()

    def test_the_refusal_carries_the_history(self) -> None:
        relaxation = self._relaxation(35.0, 30.0, 0.5)
        with pytest.raises(CalibrationError) as excinfo:
            relaxation.require_arrested()
        message = str(excinfo.value)
        for fragment in ("35.00 deg", "30.00 deg", "31.94"):
            if fragment not in message:
                raise AssertionError(f"{fragment!r} missing from {message!r}")

    def test_a_relaxation_needs_two_samples(self) -> None:
        with pytest.raises(CalibrationError):
            SlopeRelaxation(
                samples=(
                    SlopeSample(
                        time_s=0.0,
                        slope_deg=32.0,
                        n_bins=8,
                        kinetic_energy_j=0.0,
                        toe_x_m=0.1,
                    ),
                ),
                friction_angle_deg=34.0,
                plane_strain_limit_deg=31.94,
                n_particles=100,
                n_steps=100,
                time_step_s=1.0e-5,
                wall_clock_s=1.0,
            )


class TestTheSolverActuallyRuns:
    """A real F1 march, small enough for a unit lane."""

    def test_the_relaxation_produces_a_history_and_a_cost(
        self, cohesionless_material: SandContinuum
    ) -> None:
        relaxation = relax_slope(cohesionless_material, _SMOKE)
        if len(relaxation.samples) != _SMOKE.n_strides:
            raise AssertionError(relaxation.samples)
        if relaxation.ms_per_step <= 0.0:
            raise AssertionError(relaxation.ms_per_step)
        if not math.isfinite(relaxation.final_slope_deg):
            raise AssertionError(relaxation.final_slope_deg)

    def test_the_limit_angle_is_reported_beside_the_measurement(
        self, cohesionless_material: SandContinuum
    ) -> None:
        """A slope with no limit to compare against is not a measurement."""
        relaxation = relax_slope(cohesionless_material, _SMOKE)
        expected = math.degrees(math.asin(math.sqrt(2.0) * cohesionless_material.alpha))
        if abs(relaxation.plane_strain_limit_deg - expected) > 1.0e-9:
            raise AssertionError(relaxation.plane_strain_limit_deg)

    def test_the_slope_is_still_moving(
        self, cohesionless_material: SandContinuum
    ) -> None:
        """The negative result, pinned. If this ever fails, F1 arrested."""
        relaxation = relax_slope(cohesionless_material, _SMOKE)
        if relaxation.has_arrested:
            raise AssertionError(
                "the F1 slope relaxation arrested, which issue #8733 section 6 "
                f"measured it not to (drift {relaxation.drift_deg_per_s} deg/s). "
                "That would be good news: re-open the angle-of-repose target "
                "and delete this test."
            )


class TestExperiment:
    def test_declares_the_harness_contract(self) -> None:
        experiment = F1AngleOfReposeExperiment(settings=_SMOKE)
        for attribute in ("target_angle", "calibrated_parameters", "parameter_bounds"):
            if not hasattr(experiment, attribute):
                raise AssertionError(f"{attribute} missing")
        for name in experiment.calibrated_parameters:
            if name not in experiment.parameter_bounds:
                raise AssertionError(f"{name} has no declared bounds")

    def test_the_bounds_respect_the_geostatic_seed(self) -> None:
        experiment = F1AngleOfReposeExperiment(settings=_SMOKE)
        low, _ = experiment.parameter_bounds["friction_angle_deg"]
        if low < F1_REPOSE_MIN_FRICTION_ANGLE_DEG:
            raise AssertionError(
                f"lower bound {low} is below the seed limit "
                f"{F1_REPOSE_MIN_FRICTION_ANGLE_DEG}"
            )

    def test_run_simulation_refuses_while_the_slope_moves(self) -> None:
        experiment = F1AngleOfReposeExperiment(settings=_SMOKE)
        with pytest.raises(CalibrationError, match="has not arrested"):
            experiment.run_simulation({"friction_angle_deg": 34.0})

    def test_relax_returns_the_history_without_a_verdict(self) -> None:
        """The honest entry point: what happened, not an answer."""
        experiment = F1AngleOfReposeExperiment(settings=_SMOKE)
        relaxation = experiment.relax({"friction_angle_deg": 34.0})
        if len(relaxation.samples) < 2:
            raise AssertionError(relaxation.samples)

    def test_refuses_an_unusable_friction_angle(self) -> None:
        experiment = F1AngleOfReposeExperiment(settings=_SMOKE)
        with pytest.raises(CalibrationError):
            experiment.relax({"friction_angle_deg": 0.0})

    def test_does_not_claim_a_measurement(self) -> None:
        if F1AngleOfReposeExperiment(settings=_SMOKE).is_measured_on_bunker_sand:
            raise AssertionError("the repose experiment claimed to be a measurement")

    def test_the_arrest_note_names_the_issue_and_the_cause(self) -> None:
        for fragment in ("#8733", "settle time", "does not arrest"):
            if fragment not in F1_REPOSE_ARREST_NOTE:
                raise AssertionError(f"{fragment!r} missing from the arrest note")
