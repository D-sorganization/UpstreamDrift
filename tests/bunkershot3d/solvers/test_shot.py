"""Time-marching a full shot, and the 50 ms budget (issue #8611).

The acceptance criterion is "runs a full shot in < 50 ms so a 1000-point
DOE is minutes, not weeks".  It is measured here on a real lofted wedge
mesh, not on a toy plate, because the DOE will be run on wedges.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from bunkershot3d.geometry import build_wedge_mesh, get_preset, preset_names
from bunkershot3d.solvers import (
    DRFTSolver,
    EnvelopeStatus,
    FidelityTier,
    HeadKinematics,
    MaterialResponse,
    OutOfEnvelopeError,
    RefusalPolicy,
    ShotResult,
    ShotSettings,
    ShotTruncatedError,
    SolverInputError,
    SurfaceElements,
    simulate_shot,
)

from .conftest import box_mesh

pytestmark = pytest.mark.unit

_HEAD_MASS_KG = 0.30
_DELIVERY_SPEED_M_S = 25.0
_ATTACK_ANGLE_DEG = 6.0
_SHOT_BUDGET_S = 0.050

_STRIKE_WINDOW = ShotSettings(max_time_s=0.010, require_exit=False)
"""A fixed 10 ms window of a strike, for the tests that march the sole plate.

The plate is not a wedge. It is a 20 x 80 x 4 mm slab with no bounce and no
relief, so it buries itself 42 mm and does not bring its sole back above the
surface for 223 ms. The assertions below are about what the sand does to a
body over the first 10 ms of contact -- forces, impulse, the inertial share --
so these shots ask for a window and say so, rather than inheriting the
whole-shot default that requires the head to come out."""


def _entry_velocity(speed_m_s: float = _DELIVERY_SPEED_M_S) -> np.ndarray:
    """A descending blow; +x is rearward in the head frame."""
    angle = math.radians(_ATTACK_ANGLE_DEG)
    return speed_m_s * np.array([-math.cos(angle), 0.0, -math.sin(angle)])


def _wedge_elements(
    n_profile_points: int = 24, n_stations: int = 11
) -> SurfaceElements:
    """A real lofted wedge head, discretised."""
    preset = get_preset(preset_names()[0])
    mesh = build_wedge_mesh(
        preset.geometry,
        n_profile_points=n_profile_points,
        n_stations=n_stations,
    )
    return SurfaceElements.from_mesh(mesh)


@pytest.fixture(scope="module")
def wedge_elements() -> SurfaceElements:
    """Module-scoped so the mesh is lofted once, not once per test."""
    return _wedge_elements()


@pytest.fixture
def sole_body() -> SurfaceElements:
    """A 20 x 80 x 4 mm sole plate in its own body frame."""
    return SurfaceElements.from_mesh(box_mesh(0.020, 0.080, 0.004))


class TestShotTrace:
    """What a shot produces."""

    def test_returns_contiguous_arrays_sharing_one_leading_axis(
        self, solver: DRFTSolver, sole_body: SurfaceElements
    ) -> None:
        shot = simulate_shot(
            solver,
            sole_body,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
            settings=_STRIKE_WINDOW,
        )
        steps = shot.n_steps
        assert steps > 5
        assert shot.times_s.shape == (steps,)
        assert shot.positions_m.shape == (steps, 3)
        assert shot.velocities_m_s.shape == (steps, 3)
        assert shot.forces_n.shape == (steps, 3)
        assert shot.torques_n_m.shape == (steps, 3)
        assert shot.engaged_depths_m.shape == (steps,)
        assert shot.sole_depths_m.shape == (steps,)
        assert shot.orientations.shape == (steps, 3, 3)

    def test_carries_the_tier_and_the_worst_verdict_over_the_trace(
        self, solver: DRFTSolver, sole_body: SurfaceElements
    ) -> None:
        shot = simulate_shot(
            solver,
            sole_body,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
            settings=_STRIKE_WINDOW,
        )
        assert shot.fidelity_tier is FidelityTier.F0
        assert shot.verdict.status is EnvelopeStatus.BEYOND_VALIDATION
        assert "BEYOND_VALIDATION" in shot.summary()

    def test_the_head_decelerates(
        self, solver: DRFTSolver, sole_body: SurfaceElements
    ) -> None:
        shot = simulate_shot(
            solver,
            sole_body,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
            settings=_STRIKE_WINDOW,
        )
        assert shot.entry_speed_m_s == pytest.approx(_DELIVERY_SPEED_M_S, rel=1e-12)
        assert shot.exit_speed_m_s < shot.entry_speed_m_s
        assert shot.peak_force_n > 0.0

    def test_the_impulse_accounts_for_the_momentum_lost(
        self, solver: DRFTSolver, sole_body: SurfaceElements
    ) -> None:
        shot = simulate_shot(
            solver,
            sole_body,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
            settings=_STRIKE_WINDOW,
        )
        momentum_change = _HEAD_MASS_KG * (
            shot.velocities_m_s[-1] - shot.velocities_m_s[0]
        )
        # Trapezoidal against a first-order integrator, so a few percent
        # is the honest agreement, not machine precision. The absolute
        # floor is there for the lateral component, which is zero by the
        # body's own symmetry and would otherwise be compared to noise.
        np.testing.assert_allclose(
            shot.impulse_n_s,
            momentum_change,
            rtol=0.05,
            atol=1e-3 * float(np.linalg.norm(momentum_change)),
        )

    def test_the_head_digs_in(
        self, solver: DRFTSolver, sole_body: SurfaceElements
    ) -> None:
        shot = simulate_shot(
            solver,
            sole_body,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
            settings=_STRIKE_WINDOW,
        )
        assert shot.max_sole_depth_m > 0.001
        assert shot.max_engaged_depth_m > 0.001
        assert shot.contact_duration_s > 0.0

    def test_the_inertial_term_leads_throughout_a_delivery_speed_shot(
        self, solver: DRFTSolver, sole_body: SurfaceElements
    ) -> None:
        shot = simulate_shot(
            solver,
            sole_body,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
            settings=_STRIKE_WINDOW,
        )
        engaged = shot.active_areas_m2 > 0.0
        assert float(shot.inertial_fractions[engaged].min()) > 0.5


class TestPrescribedRotation:
    """The stated idealisation: free translation, prescribed rotation."""

    def test_a_rotating_head_still_produces_a_trace(
        self, solver: DRFTSolver, sole_body: SurfaceElements
    ) -> None:
        shot = simulate_shot(
            solver,
            sole_body,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(
                velocity_m_s=_entry_velocity(),
                angular_velocity_rad_s=(0.0, 30.0, 0.0),
            ),
            settings=_STRIKE_WINDOW,
        )
        assert shot.n_steps > 5
        assert shot.peak_force_n > 0.0

    def test_rotation_changes_the_answer(
        self, solver: DRFTSolver, sole_body: SurfaceElements
    ) -> None:
        still = simulate_shot(
            solver,
            sole_body,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
            settings=_STRIKE_WINDOW,
        )
        spinning = simulate_shot(
            solver,
            sole_body,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(
                velocity_m_s=_entry_velocity(),
                angular_velocity_rad_s=(0.0, 60.0, 0.0),
            ),
            settings=_STRIKE_WINDOW,
        )
        assert spinning.peak_force_n != pytest.approx(still.peak_force_n, rel=1e-6)


class TestRefusalPropagates:
    """A shot is only as answerable as its worst step."""

    def test_a_strict_solver_refuses_the_whole_shot(
        self, material: MaterialResponse, sole_body: SurfaceElements
    ) -> None:
        quasi_static = DRFTSolver(
            material=material,
            dynamic_terms_active=False,
            refusal_policy=RefusalPolicy.STRICT,
        )
        with pytest.raises(OutOfEnvelopeError):
            simulate_shot(
                quasi_static,
                sole_body,
                head_mass_kg=_HEAD_MASS_KG,
                kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
                settings=_STRIKE_WINDOW,
            )

    def test_a_reporting_solver_returns_a_refused_trace(
        self, material: MaterialResponse, sole_body: SurfaceElements
    ) -> None:
        reporting = DRFTSolver(
            material=material,
            dynamic_terms_active=False,
            refusal_policy=RefusalPolicy.REPORT,
        )
        shot = simulate_shot(
            reporting,
            sole_body,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
            settings=_STRIKE_WINDOW,
        )
        assert shot.verdict.status is EnvelopeStatus.REFUSED


class TestPreconditions:
    """Malformed shots are refused before they are integrated."""

    def test_rejects_a_body_that_is_not_surface_elements(
        self, solver: DRFTSolver
    ) -> None:
        with pytest.raises(SolverInputError):
            simulate_shot(
                solver,
                "a wedge",  # type: ignore[arg-type]
                head_mass_kg=_HEAD_MASS_KG,
                kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
            )

    def test_rejects_a_non_positive_head_mass(
        self, solver: DRFTSolver, sole_body: SurfaceElements
    ) -> None:
        with pytest.raises(SolverInputError):
            simulate_shot(
                solver,
                sole_body,
                head_mass_kg=0.0,
                kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
            )

    def test_rejects_a_step_larger_than_the_whole_shot(self) -> None:
        with pytest.raises(SolverInputError):
            ShotSettings(time_step_s=0.1, max_time_s=0.01)

    def test_rejects_an_orientation_that_is_not_a_rotation(self) -> None:
        with pytest.raises(SolverInputError, match="not a rotation"):
            HeadKinematics(
                velocity_m_s=_entry_velocity(), orientation=np.full((3, 3), 0.5)
            )

    def test_rejects_a_non_finite_velocity(self) -> None:
        with pytest.raises(SolverInputError):
            HeadKinematics(velocity_m_s=(float("nan"), 0.0, 0.0))


class TestTheDefaultWindowCoversAWholeShot:
    """Issue #8700: the 10 ms default stopped 0.75 ms before the head cleared."""

    def test_a_nominal_bunker_shot_completes_on_the_defaults(
        self, solver: DRFTSolver, wedge_elements: SurfaceElements
    ) -> None:
        """25 m/s, -6 deg, firm sand -- the condition the demo swept."""
        shot = simulate_shot(
            solver,
            wedge_elements,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
        )
        assert shot.exited
        assert shot.sole_depths_m[-1] <= 0.0
        assert shot.times_s[-1] > 0.0108, (
            "the head does not clear until ~10.8 ms; a window that ends before "
            "that truncates every nominal bunker shot"
        )

    def test_a_window_that_ends_mid_shot_is_reported_by_the_solver(
        self, solver: DRFTSolver, wedge_elements: SurfaceElements
    ) -> None:
        """The complaint belongs to the solver, not to a downstream metric."""
        with pytest.raises(ShotTruncatedError) as excinfo:
            simulate_shot(
                solver,
                wedge_elements,
                head_mass_kg=_HEAD_MASS_KG,
                kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
                settings=ShotSettings(max_time_s=0.005),
            )
        message = str(excinfo.value)
        assert "max_time_s" in message
        assert "0.005" in message
        assert excinfo.value.max_time_s == pytest.approx(0.005)
        assert excinfo.value.time_reached_s == pytest.approx(0.005, abs=2.5e-4)

    def test_the_truncated_trace_is_carried_on_the_error(
        self, solver: DRFTSolver, wedge_elements: SurfaceElements
    ) -> None:
        """A caller debugging the window needs to see what was integrated."""
        with pytest.raises(ShotTruncatedError) as excinfo:
            simulate_shot(
                solver,
                wedge_elements,
                head_mass_kg=_HEAD_MASS_KG,
                kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
                settings=ShotSettings(max_time_s=0.005),
            )
        partial = excinfo.value.result
        assert isinstance(partial, ShotResult)
        assert partial.n_steps > 1
        assert not partial.exited

    def test_a_deliberate_window_is_still_allowed(
        self, solver: DRFTSolver, wedge_elements: SurfaceElements
    ) -> None:
        """A verification study marching a fixed window says so explicitly."""
        shot = simulate_shot(
            solver,
            wedge_elements,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
            settings=ShotSettings(max_time_s=0.005, require_exit=False),
        )
        assert not shot.exited
        assert shot.times_s[-1] == pytest.approx(0.005, abs=2.5e-4)

    def test_the_truncation_guard_survives_python_dash_o(self) -> None:
        """It is a ``raise``, not an ``assert``: ``python -O`` strips those.

        A truncated shot that silently returns under an optimisation flag is
        worse than one that raises, because the caller believes the check ran.
        """
        import os
        import subprocess
        import sys

        source = (
            "import math\n"
            "import numpy as np\n"
            "from bunkershot3d.geometry import build_wedge_mesh, get_preset, "
            "preset_names\n"
            "from bunkershot3d.sand import PlayingCondition, playing_condition\n"
            "from bunkershot3d.solvers import (DRFTSolver, HeadKinematics, "
            "MaterialResponse, RefusalPolicy, ShotSettings, ShotTruncatedError, "
            "SurfaceElements, simulate_shot)\n"
            "preset = get_preset(preset_names()[0])\n"
            "mesh = build_wedge_mesh(preset.geometry, n_profile_points=12, "
            "n_stations=7)\n"
            "solver = DRFTSolver(material=MaterialResponse.from_sand_state("
            "playing_condition(PlayingCondition.FIRM)), "
            "refusal_policy=RefusalPolicy.REPORT)\n"
            "angle = math.radians(6.0)\n"
            "velocity = 25.0 * np.array([-math.cos(angle), 0.0, -math.sin(angle)])\n"
            "try:\n"
            "    simulate_shot(solver, SurfaceElements.from_mesh(mesh), "
            "head_mass_kg=0.30, kinematics=HeadKinematics(velocity_m_s=velocity), "
            "settings=ShotSettings(max_time_s=0.005))\n"
            "except ShotTruncatedError:\n"
            "    print('RAISED')\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        result = subprocess.run(
            [sys.executable, "-O", "-c", source],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert "RAISED" in result.stdout, result.stderr


class TestSoleDepthIsNotEngagedElementDepth:
    """Issue #8701: two different quantities used to share one name."""

    def test_the_sole_depth_is_the_sole_reference_below_the_free_surface(
        self, solver: DRFTSolver, wedge_elements: SurfaceElements
    ) -> None:
        shot = simulate_shot(
            solver,
            wedge_elements,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
        )
        reference_world = np.einsum(
            "nij,j->ni", shot.orientations, shot.sole_reference_body_m
        )
        expected = -(shot.positions_m[:, 2] + reference_world[:, 2])
        np.testing.assert_allclose(shot.sole_depths_m, expected, rtol=0.0, atol=0.0)

    def test_the_sole_descends_monotonically_to_its_deepest_point(
        self, solver: DRFTSolver, wedge_elements: SurfaceElements
    ) -> None:
        """The engaged-element depth is not monotone here; the sole depth is."""
        shot = simulate_shot(
            solver,
            wedge_elements,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
        )
        deepest = int(np.argmax(shot.sole_depths_m))
        descent = np.diff(shot.sole_depths_m[: deepest + 1])
        assert descent.size > 3
        assert float(descent.min()) > 0.0

    def test_the_sole_is_still_buried_when_no_element_is_engaged(
        self, solver: DRFTSolver, wedge_elements: SurfaceElements
    ) -> None:
        """The exact reading that made the old name wrong: 0 while under."""
        shot = simulate_shot(
            solver,
            wedge_elements,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
        )
        disengaged_but_buried = (shot.engaged_depths_m <= 0.0) & (
            shot.sole_depths_m > 0.0
        )
        assert disengaged_but_buried.any(), (
            "this trace no longer exhibits the divergence the issue reports; "
            "the test has stopped measuring what it was written for"
        )

    def test_the_deepest_sole_reading_is_at_least_the_engaged_one(
        self, solver: DRFTSolver, wedge_elements: SurfaceElements
    ) -> None:
        shot = simulate_shot(
            solver,
            wedge_elements,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
        )
        assert shot.max_sole_depth_m >= shot.max_engaged_depth_m > 0.0

    def test_the_ambiguous_names_are_gone(
        self, solver: DRFTSolver, sole_body: SurfaceElements
    ) -> None:
        """``depths_m`` meant neither quantity unambiguously, so it is retired."""
        shot = simulate_shot(
            solver,
            sole_body,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
            settings=_STRIKE_WINDOW,
        )
        assert not hasattr(shot, "depths_m")
        assert not hasattr(shot, "max_depth_m")


class TestFreeFlightLeadIn:
    """The approach the dig-versus-skid discriminator measures (issue #8702)."""

    def test_the_record_starts_above_the_sand(
        self, solver: DRFTSolver, wedge_elements: SurfaceElements
    ) -> None:
        shot = simulate_shot(
            solver,
            wedge_elements,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
        )
        submerged = shot.sole_depths_m > 0.0
        first = int(np.argmax(submerged))
        assert first >= 2, (
            "the delivered path slope is a backward difference across the two "
            "samples before entry, so two free-flight samples are the minimum"
        )
        assert float(shot.active_areas_m2[:first].max()) == 0.0

    def test_the_lead_in_can_be_switched_off(
        self, solver: DRFTSolver, wedge_elements: SurfaceElements
    ) -> None:
        shot = simulate_shot(
            solver,
            wedge_elements,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
            settings=ShotSettings(free_flight_lead_steps=0.0),
        )
        assert shot.sole_depths_m[0] == pytest.approx(0.0, abs=1e-15)


class TestPerformanceBudget:
    """A full shot has to fit in 50 ms so a 1000-point DOE is minutes."""

    def _fastest_shot_s(
        self, solver: DRFTSolver, body: SurfaceElements, repeats: int = 9
    ) -> float:
        """Best of ``repeats`` wall-clock timings, after a warm-up.

        The *minimum*, not the mean or the median, on purpose.  Noise on
        a shared machine only ever adds time, so the best sample is the
        least biased estimate of what the code costs; a mean or median
        measures the load on the box instead.  That is not academic
        here: the same shot measured 15 ms on a quiet interpreter and a
        30 ms median with a large heap resident, while its minimum stayed
        at 17 ms.  Asserting on the median would make this test a
        thermometer.
        """
        kinematics = HeadKinematics(velocity_m_s=_entry_velocity())
        for _ in range(3):
            simulate_shot(
                solver, body, head_mass_kg=_HEAD_MASS_KG, kinematics=kinematics
            )
        best = math.inf
        for _ in range(repeats):
            started = time.perf_counter()
            simulate_shot(
                solver, body, head_mass_kg=_HEAD_MASS_KG, kinematics=kinematics
            )
            best = min(best, time.perf_counter() - started)
        return best

    @pytest.mark.timeout(120)
    def test_a_full_wedge_shot_fits_the_budget(
        self, solver: DRFTSolver, wedge_elements: SurfaceElements
    ) -> None:
        assert len(wedge_elements) > 400, "the budget must be measured on a real mesh"
        fastest_s = self._fastest_shot_s(solver, wedge_elements)
        assert fastest_s < _SHOT_BUDGET_S, (
            f"a full shot took {fastest_s * 1e3:.1f} ms against a "
            f"{_SHOT_BUDGET_S * 1e3:.0f} ms budget; a 1000-point DOE would take "
            f"{fastest_s * 1000 / 60:.1f} minutes"
        )

    @pytest.mark.timeout(120)
    def test_a_thousand_point_design_of_experiments_is_minutes(
        self, solver: DRFTSolver, wedge_elements: SurfaceElements
    ) -> None:
        fastest_s = self._fastest_shot_s(solver, wedge_elements)
        assert fastest_s * 1000 < 300.0

    def test_the_shot_reports_its_own_runtime(
        self, solver: DRFTSolver, sole_body: SurfaceElements
    ) -> None:
        shot = simulate_shot(
            solver,
            sole_body,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
            settings=_STRIKE_WINDOW,
        )
        assert 0.0 < shot.runtime_s < 5.0
