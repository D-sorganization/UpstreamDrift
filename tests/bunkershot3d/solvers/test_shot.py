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
    ShotSettings,
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
        )
        steps = shot.n_steps
        assert steps > 5
        assert shot.times_s.shape == (steps,)
        assert shot.positions_m.shape == (steps, 3)
        assert shot.velocities_m_s.shape == (steps, 3)
        assert shot.forces_n.shape == (steps, 3)
        assert shot.torques_n_m.shape == (steps, 3)
        assert shot.depths_m.shape == (steps,)

    def test_carries_the_tier_and_the_worst_verdict_over_the_trace(
        self, solver: DRFTSolver, sole_body: SurfaceElements
    ) -> None:
        shot = simulate_shot(
            solver,
            sole_body,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
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
        )
        assert shot.max_depth_m > 0.001
        assert shot.contact_duration_s > 0.0

    def test_the_inertial_term_leads_throughout_a_delivery_speed_shot(
        self, solver: DRFTSolver, sole_body: SurfaceElements
    ) -> None:
        shot = simulate_shot(
            solver,
            sole_body,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=_entry_velocity()),
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
        )
        spinning = simulate_shot(
            solver,
            sole_body,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(
                velocity_m_s=_entry_velocity(),
                angular_velocity_rad_s=(0.0, 60.0, 0.0),
            ),
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
        )
        assert 0.0 < shot.runtime_s < 5.0
