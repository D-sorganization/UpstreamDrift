"""Marching the head's real trajectory once, as a whole shot (#8733 §3).

The claim being tested is not "the march runs". It is that this is a
*different question* from the one ``solve()`` answers, and that both are
still askable:

* ``solve()``/``_approach`` reverses the body and drives it back to the
  queried pose **at constant velocity**. That is the declared assumption
  that makes an F1 answer comparable to F0's memoryless one, and it is
  unchanged -- so it is tested here too, on the same delivery.
* :func:`simulate_f1_shot` integrates the head instead. Nothing tells it
  where to be, the sand it meets is sand it has already disturbed, and
  the wrench history comes off one continuous solve.

The visible consequence, and the thing that would be missing if the march
were secretly still prescribed, is that the head **slows down**.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.sand import playing_condition
from bunkershot3d.sand.presets import PlayingCondition
from bunkershot3d.solvers import (
    IntrusionState,
    OutOfEnvelopeError,
    RefusalPolicy,
    SurfaceElements,
)
from bunkershot3d.solvers.exceptions import ShotTruncatedError, SolverInputError
from bunkershot3d.solvers.mpm.ball import BallSection
from bunkershot3d.solvers.mpm.constitutive import SandContinuum
from bunkershot3d.solvers.mpm.solver import PlaneStrainMPMSolver
from bunkershot3d.solvers.mpm.wholeshot import (
    DEFAULT_EJECTA_HEADROOM_CELLS,
    DEFAULT_TRAVEL_SPANS,
    F1ShotResult,
    F1ShotSettings,
    simulate_f1_shot,
)

pytestmark = pytest.mark.unit


def delivery(speed_m_s: float, attack_deg: float) -> IntrusionState:
    """A 24 x 10 mm sole section delivered at a stated attack angle.

    A negative attack angle is a *rising* delivery: the head is already in
    the sand and on its way out, which is the cheapest configuration that
    exercises the exit crossing.
    """
    corners = np.array(
        [
            [-0.012, 0.0, -0.005],
            [0.012, 0.0, -0.005],
            [0.012, 0.0, 0.005],
            [-0.012, 0.0, 0.005],
        ]
    )
    normals = np.tile([0.0, 0.0, -1.0], (corners.shape[0], 1))
    areas = np.full(corners.shape[0], 2.0e-4)
    angle = math.radians(attack_deg)
    return IntrusionState(
        SurfaceElements(corners, normals, areas),
        (speed_m_s * math.cos(angle), 0.0, -speed_m_s * math.sin(angle)),
        free_surface_height_m=0.0,
    )


@pytest.fixture(scope="module")
def material() -> SandContinuum:
    return SandContinuum.from_sand_state(playing_condition(PlayingCondition.FIRM))


@pytest.fixture(scope="module")
def solver(material: SandContinuum) -> PlaneStrainMPMSolver:
    """Deliberately coarse: these tests are about the march, not the sand."""
    return PlaneStrainMPMSolver(
        material=material,
        cell_size_m=0.006,
        effective_width_m=0.03,
        bed_depth_m=0.018,
        run_in_lengths=0.5,
        refusal_policy=RefusalPolicy.REPORT,
    )


def windowed_settings(**overrides: object) -> F1ShotSettings:
    """A short fixed window: cheap, and it never has to reach an exit."""
    base: dict[str, object] = {
        "head_mass_kg": 0.05,
        "max_time_s": 0.006,
        "free_flight_lead_m": 0.002,
        "travel_allowance_m": 0.03,
        "ejecta_headroom_m": 0.04,
        "require_exit": False,
    }
    base.update(overrides)
    return F1ShotSettings(**base)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def windowed(solver: PlaneStrainMPMSolver) -> F1ShotResult:
    """One descending shot, marched to the end of its window."""
    return simulate_f1_shot(solver, delivery(2.0, 25.0), settings=windowed_settings())


@pytest.fixture(scope="module")
def rising(solver: PlaneStrainMPMSolver) -> F1ShotResult:
    """One head already in the sand and on its way out."""
    return simulate_f1_shot(
        solver,
        delivery(3.0, -40.0),
        settings=windowed_settings(head_mass_kg=0.2, max_time_s=0.008),
    )


class TestSettings:
    """Every allowance is declared, and a bad one is refused up front."""

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("head_mass_kg", 0.0, "head_mass_kg"),
            ("head_mass_kg", -1.0, "head_mass_kg"),
            ("max_time_s", 0.0, "max_time_s"),
            ("min_speed_m_s", 0.0, "min_speed_m_s"),
            ("gravity_m_s2", -1.0, "gravity_m_s2"),
            ("free_flight_lead_m", -0.001, "free_flight_lead_m"),
            ("travel_allowance_m", 0.0, "travel_allowance_m"),
            ("ejecta_headroom_m", -0.1, "ejecta_headroom_m"),
            ("free_surface_height_m", math.nan, "free_surface_height_m"),
        ],
    )
    def test_a_bad_setting_is_refused(
        self, field: str, value: float, match: str
    ) -> None:
        with pytest.raises(SolverInputError, match=match):
            F1ShotSettings(**{"head_mass_kg": 0.3, field: value})  # type: ignore[arg-type]

    def test_the_allowances_default_to_stated_multiples(self) -> None:
        assert DEFAULT_TRAVEL_SPANS > 0.0
        assert DEFAULT_EJECTA_HEADROOM_CELLS > 0.0

    def test_a_valid_settings_object_survives(self) -> None:
        assert F1ShotSettings(head_mass_kg=0.3).travel_allowance_m is None


class TestRefusals:
    """The march is refused before it costs anything."""

    def test_a_non_settings_object_is_refused(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        with pytest.raises(SolverInputError, match="F1ShotSettings"):
            simulate_f1_shot(solver, delivery(2.0, 20.0), settings=object())  # type: ignore[arg-type]

    def test_a_static_delivery_is_refused(self, solver: PlaneStrainMPMSolver) -> None:
        state = IntrusionState(
            delivery(2.0, 20.0).elements, (0.0, 0.0, 0.0), free_surface_height_m=0.0
        )
        with pytest.raises(SolverInputError, match="no in-plane velocity"):
            simulate_f1_shot(solver, state, settings=windowed_settings())

    def test_a_strict_solver_refuses_the_verdict(self, material: SandContinuum) -> None:
        strict = PlaneStrainMPMSolver(
            material=material,
            cell_size_m=0.0001,
            effective_width_m=0.03,
            refusal_policy=RefusalPolicy.STRICT,
        )
        with pytest.raises(OutOfEnvelopeError):
            simulate_f1_shot(strict, delivery(2.0, 20.0), settings=windowed_settings())

    def test_running_off_the_bed_raises_and_names_the_setting(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        with pytest.raises(SolverInputError, match="travel_allowance_m"):
            simulate_f1_shot(
                solver,
                delivery(4.0, 5.0),
                settings=windowed_settings(travel_allowance_m=0.001),
            )


class TestTheTrace:
    """The record is the shape ``simulate_shot`` returns."""

    def test_it_carries_one_sample_per_step(self, windowed: F1ShotResult) -> None:
        assert windowed.shot.n_steps == windowed.run.n_steps

    def test_the_arrays_share_their_leading_axis(self, windowed: F1ShotResult) -> None:
        shot = windowed.shot
        n = shot.n_steps
        assert shot.positions_m.shape == (n, 3)
        assert shot.velocities_m_s.shape == (n, 3)
        assert shot.orientations.shape == (n, 3, 3)
        assert shot.forces_n.shape == (n, 3)
        assert shot.torques_n_m.shape == (n, 3)
        assert shot.sole_depths_m.shape == (n,)
        assert shot.active_areas_m2.shape == (n,)
        assert shot.inertial_fractions.shape == (n,)

    def test_time_starts_at_zero_and_increases(self, windowed: F1ShotResult) -> None:
        assert windowed.shot.times_s[0] == pytest.approx(0.0)
        assert np.all(np.diff(windowed.shot.times_s) > 0.0)

    def test_plane_strain_has_no_out_of_plane_force_or_torque(
        self, windowed: F1ShotResult
    ) -> None:
        assert np.all(windowed.shot.forces_n[:, 1] == 0.0)
        assert np.all(windowed.shot.torques_n_m[:, 0] == 0.0)
        assert np.all(windowed.shot.torques_n_m[:, 2] == 0.0)

    def test_the_orientations_are_rotations(self, windowed: F1ShotResult) -> None:
        for orientation in windowed.shot.orientations[
            :: max(1, windowed.shot.n_steps // 8)
        ]:
            assert orientation @ orientation.T == pytest.approx(np.eye(3), abs=1e-12)

    def test_everything_is_finite(self, windowed: F1ShotResult) -> None:
        assert np.all(np.isfinite(windowed.shot.forces_n))
        assert np.all(np.isfinite(windowed.shot.positions_m))
        assert np.all(np.isfinite(windowed.shot.velocities_m_s))

    def test_it_carries_the_f1_tier_and_verdict(
        self, windowed: F1ShotResult, solver: PlaneStrainMPMSolver
    ) -> None:
        assert windowed.shot.fidelity_tier is solver.fidelity_tier
        assert windowed.shot.verdict.caveats

    def test_the_sole_reference_is_in_the_section_plane(
        self, windowed: F1ShotResult
    ) -> None:
        assert windowed.shot.sole_reference_body_m[1] == pytest.approx(0.0)


class TestEntry:
    """Where the march starts, and why it is placed rather than searched."""

    def test_a_descending_head_starts_clear_of_the_sand(
        self, windowed: F1ShotResult
    ) -> None:
        assert windowed.shot.sole_depths_m[0] == pytest.approx(-0.002, abs=1e-9)

    def test_a_rising_head_starts_where_it_was_delivered(
        self, rising: F1ShotResult
    ) -> None:
        # There is no crossing to back up to, so the delivered pose stands
        # rather than being given an invented descent.
        assert rising.shot.sole_depths_m[0] == pytest.approx(0.005, abs=1e-9)

    def test_the_head_actually_touches_the_sand(self, windowed: F1ShotResult) -> None:
        assert windowed.contacted is True
        assert float(windowed.shot.active_areas_m2.max()) > 0.0


class TestTheTrajectoryIsMarchedNotPrescribed:
    """The §3 claim: nothing tells the head where to go."""

    def test_the_head_slows_down(self, windowed: F1ShotResult) -> None:
        assert windowed.shot.exit_speed_m_s < windowed.shot.entry_speed_m_s

    def test_the_velocity_is_integrated_from_the_reported_force(
        self, windowed: F1ShotResult
    ) -> None:
        # v^{n+1} = v^n + dt F^n / m, exactly. The wrench on the trace is
        # the wrench that moved the head, not a number reported beside it.
        step = windowed.run.time_step_s
        velocities = windowed.shot.velocities_m_s
        predicted = velocities[:-1] + (step / 0.05) * windowed.shot.forces_n[:-1]
        assert predicted == pytest.approx(velocities[1:], abs=1e-12)

    def test_the_head_is_not_driven_at_the_delivered_speed(
        self, windowed: F1ShotResult
    ) -> None:
        speeds = np.linalg.norm(windowed.shot.velocities_m_s, axis=1)
        assert speeds[-1] < 0.95 * speeds[0]

    def test_the_pose_is_the_integral_of_the_velocity(
        self, windowed: F1ShotResult
    ) -> None:
        # x^{n+1} = x^n + dt v^n, exactly, because that is what the swept
        # collision test predicted the body would do.
        step = windowed.run.time_step_s
        positions = windowed.shot.positions_m
        velocities = windowed.shot.velocities_m_s
        predicted = positions[:-1] + step * velocities[:-1]
        assert predicted == pytest.approx(positions[1:], abs=1e-12)

    def test_the_sand_remembers(self, windowed: F1ShotResult) -> None:
        assert windowed.run.divot_depth_m() > 0.0

    def test_the_impulse_matches_the_momentum_the_head_lost(
        self, windowed: F1ShotResult
    ) -> None:
        # The trace's own force history, integrated, is the head's momentum
        # change: the wrench really is what moved the head, not a number
        # reported beside it.
        settings_mass = 0.05
        momentum = settings_mass * (
            windowed.shot.velocities_m_s[-1] - windowed.shot.velocities_m_s[0]
        )
        assert windowed.shot.impulse_n_s == pytest.approx(momentum, rel=0.05)


class TestTermination:
    """Three different endings, reported as three different things."""

    def test_a_windowed_march_is_truncated_not_exited(
        self, windowed: F1ShotResult
    ) -> None:
        assert windowed.truncated is True
        assert windowed.exited is False

    def test_a_rising_head_exits(self, rising: F1ShotResult) -> None:
        assert rising.exited is True
        assert rising.truncated is False
        assert rising.shot.sole_depths_m[-1] <= 0.0

    def test_a_stopped_head_is_not_a_truncation(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        stalled = simulate_f1_shot(
            solver,
            delivery(2.0, 25.0),
            settings=windowed_settings(min_speed_m_s=1.8),
        )
        assert stalled.truncated is False
        assert stalled.exited is False

    def test_require_exit_raises_and_carries_the_partial_trace(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        with pytest.raises(ShotTruncatedError) as excinfo:
            simulate_f1_shot(
                solver,
                delivery(2.0, 25.0),
                settings=windowed_settings(require_exit=True),
            )
        assert excinfo.value.result is not None
        assert "max_time_s" in str(excinfo.value)


class TestReporting:
    """What a caller reads off the result."""

    def test_the_peak_time_lies_inside_the_record(self, windowed: F1ShotResult) -> None:
        peak = windowed.peak_force_time_s()
        assert windowed.shot.times_s[0] <= peak <= windowed.shot.times_s[-1]

    def test_the_peak_time_is_where_the_force_actually_peaked(
        self, windowed: F1ShotResult
    ) -> None:
        magnitude = np.linalg.norm(windowed.shot.forces_n, axis=1)
        index = int(np.argmax(magnitude))
        assert windowed.peak_force_time_s() == pytest.approx(
            windowed.shot.times_s[index]
        )

    def test_the_summary_says_which_march_it_was(self, windowed: F1ShotResult) -> None:
        assert "whole-shot" in windowed.summary()

    def test_the_travel_is_the_horizontal_distance_covered(
        self, windowed: F1ShotResult
    ) -> None:
        expected = abs(
            float(windowed.shot.positions_m[-1][0] - windowed.shot.positions_m[0][0])
        )
        assert windowed.travel_m == pytest.approx(expected)


class TestTheDeclaredPathIsStillThere:
    """The new path is additional. It does not replace the old one."""

    def test_solve_still_answers_the_same_delivery(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        result = solver.solve(delivery(2.0, 25.0))
        assert np.all(np.isfinite(result.wrench.force_n))
        assert result.fidelity_tier is solver.fidelity_tier

    def test_the_declared_approach_keeps_a_constant_speed(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        # The whole point of the comparison: the prescribed approach drives
        # the body at the queried speed for its whole length, so the pose at
        # every step is known in advance. The marched shot's is not.
        setup = solver.prepare(delivery(2.0, 25.0))
        assert setup.section.speed_m_s == pytest.approx(2.0)
        assert setup.approach_distance_m > 0.0

    def test_the_two_paths_reach_different_depths(
        self, solver: PlaneStrainMPMSolver, windowed: F1ShotResult
    ) -> None:
        # The declared approach stops at the queried pose by construction;
        # the marched head keeps going until the sand stops it.
        queried_depth_m = 0.005
        assert windowed.shot.max_sole_depth_m != pytest.approx(
            queried_depth_m, rel=0.05
        )


class TestExtraBodies:
    """A ball marched alongside the head, through §2's ordering."""

    def test_the_ball_is_marched_and_gets_its_own_ledger(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        ball = BallSection.at(
            (0.02, 0.006), radius_m=0.005, n_facets=8, velocity_m_s=(0.0, 0.0)
        )
        shot = simulate_f1_shot(
            solver,
            delivery(2.0, 25.0),
            settings=windowed_settings(max_time_s=0.002),
            extra_bodies=(ball.section,),
        )
        assert len(shot.run.extra_sections) == 1
        assert all(step.n_bodies == 2 for step in shot.run.steps)

    def test_the_head_is_still_the_primary_body(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        ball = BallSection.at(
            (0.02, 0.006), radius_m=0.005, n_facets=8, velocity_m_s=(0.0, 0.0)
        )
        shot = simulate_f1_shot(
            solver,
            delivery(2.0, 25.0),
            settings=windowed_settings(max_time_s=0.002),
            extra_bodies=(ball.section,),
        )
        width = solver.effective_width_m
        for index, step in enumerate(shot.run.steps):
            assert shot.shot.forces_n[index][0] == pytest.approx(
                step.body_contacts[0].force_n_per_m[0] * width
            )
