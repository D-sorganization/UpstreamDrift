"""The coupled club/shaft/grip boundary and its accounting (issue #9544).

Prescribed rotation stays the named default. What is new is an *optional*
boundary through which the sand wrench on the head can drive the head's
rotation -- a shaft model, a double pendulum, or a test oracle -- and
the ledger a prescribed march owes: the angular impulse the support had
to supply, and the work the driver did holding the delivered rotation.

Every solver here is a stated oracle. The constant-wrench solver has the
closed-form answer ``omega_k = omega_0 + k dt tau / I``, which explicit
Euler reproduces to round-off, so these are identities and not
tolerances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from bunkershot3d.solvers import (
    DRFTSolver,
    FidelityTier,
    HeadKinematics,
    IntrusionState,
    RotationCoupling,
    RotationMode,
    ShotSettings,
    SolverInputError,
    SolverResult,
    SurfaceElements,
    ValidityVerdict,
    Wrench,
    evaluate_envelope,
    simulate_shot,
)
from bunkershot3d.vandv.conservation import (
    ConservationClass,
    prescribed_driver_work,
    support_angular_impulse,
)
from bunkershot3d.vandv.exceptions import VerificationError

from .conftest import box_mesh

pytestmark = pytest.mark.unit

_HEAD_MASS_KG = 0.30
_WINDOW = ShotSettings(time_step_s=1e-3, max_time_s=0.010, require_exit=False)
"""The plate starts 10 mm under the surface and the window ends before it
comes out, so every sample of every oracle shot is an engaged one."""
_ENTRY = HeadKinematics(velocity_m_s=(-5.0, 0.0, -1.0))


def _verdict() -> ValidityVerdict:
    return evaluate_envelope(
        speed_m_s=5.0,
        feature_lengths_m={"plate": 0.02},
        grain_diameter_m=3e-4,
        element_size_m=0.01,
        dynamic_terms_active=False,
    )


@dataclass
class _ConstantWrenchSolver:
    """An oracle: the same force and torque on every engaged step."""

    force_n: tuple[float, float, float]
    torque_n_m: tuple[float, float, float]
    seen: list[IntrusionState] = field(default_factory=list)

    @property
    def fidelity_tier(self) -> FidelityTier:
        return FidelityTier.F0

    def envelope(self, state: IntrusionState) -> ValidityVerdict:
        return _verdict()

    def solve(self, state: IntrusionState) -> SolverResult:
        self.seen.append(state)
        engaged = int((state.element_depths_m() < 0.0).sum())
        wrench = (
            Wrench(self.force_n, self.torque_n_m, state.reference_point_m)
            if engaged
            else Wrench.zero(state.reference_point_m)
        )
        return SolverResult(
            wrench=wrench,
            fidelity_tier=FidelityTier.F0,
            verdict=_verdict(),
            depth_force_n=wrench.force_n,
            inertial_force_n=np.zeros(3),
            n_active_elements=engaged,
            active_area_m2=1e-4 * engaged,
            max_depth_m=0.0,
        )


@dataclass
class _ScalarInertiaCoupling:
    """The simplest coupling: a free head of scalar inertia ``I``.

    ``omega_{n+1} = omega_n + dt tau_n / I``, explicit, matching the
    translational update's ordering step for step.
    """

    inertia_kg_m2: float
    wrenches: list[Wrench] = field(default_factory=list)

    def angular_velocity_after(
        self,
        *,
        time_s: float,
        time_step_s: float,
        wrench: Wrench,
        orientation: np.ndarray,
        angular_velocity_rad_s: np.ndarray,
    ) -> np.ndarray:
        self.wrenches.append(wrench)
        return angular_velocity_rad_s + time_step_s * wrench.torque_n_m / (
            self.inertia_kg_m2
        )


@pytest.fixture
def plate() -> SurfaceElements:
    """A 20 x 80 x 4 mm plate in its own body frame."""
    return SurfaceElements.from_mesh(box_mesh(0.020, 0.080, 0.004))


def _shot(solver, plate, *, coupling=None, **kinematics):
    mode = RotationMode.PRESCRIBED if coupling is None else RotationMode.COUPLED
    return simulate_shot(
        solver,
        plate,
        head_mass_kg=_HEAD_MASS_KG,
        kinematics=HeadKinematics(
            velocity_m_s=(-5.0, 0.0, -1.0), position_m=(0.0, 0.0, -0.01), **kinematics
        ),
        settings=ShotSettings(
            time_step_s=_WINDOW.time_step_s,
            max_time_s=_WINDOW.max_time_s,
            require_exit=False,
            start_at_first_contact=False,
            free_flight_lead_steps=0.0,
            rotation_mode=mode,
        ),
        coupling=coupling,
    )


class TestTheModeIsNamed:
    def test_prescribed_is_the_default_and_is_recorded_on_the_trace(
        self, plate: SurfaceElements
    ) -> None:
        assert ShotSettings().rotation_mode is RotationMode.PRESCRIBED
        shot = _shot(_ConstantWrenchSolver((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), plate)
        assert shot.rotation_mode is RotationMode.PRESCRIBED
        assert shot.angular_velocities_rad_s.shape == (shot.n_steps, 3)

    def test_coupled_mode_needs_a_coupling(self, plate: SurfaceElements) -> None:
        with pytest.raises(SolverInputError, match="coupling"):
            simulate_shot(
                _ConstantWrenchSolver((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                plate,
                head_mass_kg=_HEAD_MASS_KG,
                kinematics=_ENTRY,
                settings=ShotSettings(rotation_mode=RotationMode.COUPLED),
            )

    def test_prescribed_mode_refuses_a_stray_coupling(
        self, plate: SurfaceElements
    ) -> None:
        with pytest.raises(SolverInputError, match="prescribed"):
            simulate_shot(
                _ConstantWrenchSolver((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                plate,
                head_mass_kg=_HEAD_MASS_KG,
                kinematics=_ENTRY,
                settings=ShotSettings(rotation_mode=RotationMode.PRESCRIBED),
                coupling=_ScalarInertiaCoupling(1e-4),
            )

    def test_the_oracle_coupling_satisfies_the_public_protocol(self) -> None:
        assert isinstance(_ScalarInertiaCoupling(1e-4), RotationCoupling)


class TestZeroSandRecovery:
    """With no sand wrench the coupled march is the prescribed march."""

    def test_coupled_and_prescribed_agree_bit_for_bit_without_sand(
        self, plate: SurfaceElements
    ) -> None:
        spin = (0.0, 30.0, 0.0)
        prescribed = _shot(
            _ConstantWrenchSolver((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            plate,
            angular_velocity_rad_s=spin,
        )
        coupling = _ScalarInertiaCoupling(1e-4)
        coupled = _shot(
            _ConstantWrenchSolver((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            plate,
            coupling=coupling,
            angular_velocity_rad_s=spin,
        )
        assert coupled.rotation_mode is RotationMode.COUPLED
        np.testing.assert_array_equal(coupled.orientations, prescribed.orientations)
        np.testing.assert_array_equal(coupled.positions_m, prescribed.positions_m)
        np.testing.assert_array_equal(
            coupled.angular_velocities_rad_s, np.tile(spin, (coupled.n_steps, 1))
        )
        assert all(w.torque_magnitude_n_m == 0.0 for w in coupling.wrenches)


class TestTheWrenchConventionAtTheBoundary:
    def test_the_torque_is_about_the_body_origin_in_the_world_frame(
        self, plate: SurfaceElements
    ) -> None:
        coupling = _ScalarInertiaCoupling(1e-4)
        shot = _shot(
            _ConstantWrenchSolver((-100.0, 0.0, 40.0), (0.3, 0.7, -0.2)),
            plate,
            coupling=coupling,
        )
        # One call per recorded sample, after that sample's solve, in the
        # same place the translational update reads the force.
        assert len(coupling.wrenches) == shot.n_steps
        for k, wrench in enumerate(coupling.wrenches):
            np.testing.assert_array_equal(wrench.reference_point_m, shot.positions_m[k])

    def test_off_centre_levers_shift_the_torque_by_r_cross_f(
        self, plate: SurfaceElements
    ) -> None:
        coupling = _ScalarInertiaCoupling(1e-4)
        _shot(
            _ConstantWrenchSolver((-100.0, 0.0, 40.0), (0.3, 0.7, -0.2)),
            plate,
            coupling=coupling,
        )
        wrench = next(w for w in coupling.wrenches if w.torque_magnitude_n_m > 0.0)
        lever = np.array([0.0, 0.0, -0.02])  # a point 20 mm below the origin
        shifted = wrench.about(wrench.reference_point_m + lever)
        # tau(p + r) = tau(p) + (p - (p + r)) x F = tau(p) - r x F
        expected = wrench.torque_n_m - np.cross(lever, wrench.force_n)
        np.testing.assert_allclose(shifted.torque_n_m, expected, rtol=0.0, atol=1e-15)
        assert not np.allclose(shifted.torque_n_m, wrench.torque_n_m)


class TestConstantTorqueIsExact:
    """``omega_k = omega_0 + k dt tau / I`` to round-off, and the
    orientations are the exponential of the recorded angular velocities."""

    def test_the_recorded_angular_velocity_is_the_closed_form(
        self, plate: SurfaceElements
    ) -> None:
        inertia = 2.5e-4
        torque = np.array([0.0, 0.5, 0.0])
        coupling = _ScalarInertiaCoupling(inertia)
        shot = _shot(
            _ConstantWrenchSolver((-100.0, 0.0, 40.0), tuple(torque)),
            plate,
            coupling=coupling,
        )
        dt = _WINDOW.time_step_s
        engaged = shot.active_areas_m2 > 0.0
        assert engaged.all()  # the plate starts submerged and stays there
        k = np.arange(shot.n_steps)
        expected = np.outer(k * dt / inertia, torque)
        np.testing.assert_allclose(
            shot.angular_velocities_rad_s, expected, rtol=1e-12, atol=1e-12
        )

    def test_the_exit_spin_read_off_the_orientations_is_the_recorded_one(
        self, plate: SurfaceElements
    ) -> None:
        coupling = _ScalarInertiaCoupling(2.5e-4)
        shot = _shot(
            _ConstantWrenchSolver((-100.0, 0.0, 40.0), (0.0, 0.5, 0.0)),
            plate,
            coupling=coupling,
        )
        # The last rotation increment was taken with the last recorded omega.
        np.testing.assert_allclose(
            shot.exit_angular_velocity_rad_s,
            shot.angular_velocities_rad_s[-1],
            rtol=1e-9,
            atol=1e-9,
        )


@dataclass
class _ViscousCoupling:
    """A damped free head: ``I domega/dt = -c omega``, ignoring the sand.

    The closed form is ``omega(t) = omega_0 exp(-c t / I)`` and the swept
    angle ``theta(t) = omega_0 (I / c) (1 - exp(-c t / I))``; the march's
    explicit step is first order in ``dt`` against both.
    """

    rate_1_s: float

    def angular_velocity_after(
        self,
        *,
        time_s: float,
        time_step_s: float,
        wrench: Wrench,
        orientation: np.ndarray,
        angular_velocity_rad_s: np.ndarray,
    ) -> np.ndarray:
        return angular_velocity_rad_s * (1.0 - time_step_s * self.rate_1_s)


class TestTimestepConvergence:
    """The coupled march converges at first order in the step."""

    def test_the_swept_angle_converges_to_the_closed_form(
        self, plate: SurfaceElements
    ) -> None:
        omega_0 = 30.0
        rate = 100.0
        window_s = 0.010
        exact = omega_0 / rate * (1.0 - math.exp(-rate * window_s))
        errors = []
        for dt in (1e-3, 5e-4, 2.5e-4):
            shot = simulate_shot(
                _ConstantWrenchSolver((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                plate,
                head_mass_kg=_HEAD_MASS_KG,
                kinematics=HeadKinematics(
                    velocity_m_s=(-5.0, 0.0, -1.0),
                    position_m=(0.0, 0.0, -0.01),
                    angular_velocity_rad_s=(0.0, omega_0, 0.0),
                ),
                settings=ShotSettings(
                    time_step_s=dt,
                    max_time_s=window_s,
                    require_exit=False,
                    start_at_first_contact=False,
                    free_flight_lead_steps=0.0,
                    rotation_mode=RotationMode.COUPLED,
                ),
                coupling=_ViscousCoupling(rate),
            )
            assert shot.times_s[-1] == pytest.approx(window_s)
            final = shot.orientations[-1]
            swept = math.atan2(final[0, 2], final[0, 0])  # rotation about +y
            errors.append(abs(swept - exact))
        assert errors[0] > errors[1] > errors[2]
        # First order: halving the step roughly halves the error.
        assert 1.5 < errors[0] / errors[1] < 2.5
        assert 1.5 < errors[1] / errors[2] < 2.5


class TestSupportAccounting:
    """What a prescribed march owes: the support's angular impulse and
    the driver's work, both about the body origin."""

    def test_the_support_reacts_the_whole_sand_angular_impulse(
        self, plate: SurfaceElements
    ) -> None:
        torque = np.array([0.3, 0.7, -0.2])
        shot = _shot(
            _ConstantWrenchSolver((-100.0, 0.0, 40.0), tuple(torque)),
            plate,
            angular_velocity_rad_s=(0.0, 30.0, 0.0),
        )
        ledger = support_angular_impulse(shot)
        assert ledger.conservation_class is ConservationClass.ROUND_OFF
        sand = np.trapezoid(shot.torques_n_m, x=shot.times_s, axis=0)
        np.testing.assert_allclose(ledger.sand_n_m_s, sand, rtol=1e-12)
        np.testing.assert_allclose(ledger.support_n_m_s, -sand, rtol=1e-12)
        np.testing.assert_allclose(ledger.support_n_m_s + ledger.sand_n_m_s, 0.0)
        assert ledger.residual == 0.0

    def test_the_driver_work_is_minus_the_sand_torque_work(
        self, plate: SurfaceElements
    ) -> None:
        spin = np.array([0.0, 30.0, 0.0])
        torque = np.array([0.3, 0.7, -0.2])
        shot = _shot(
            _ConstantWrenchSolver((-100.0, 0.0, 40.0), tuple(torque)),
            plate,
            angular_velocity_rad_s=tuple(spin),
        )
        work = prescribed_driver_work(shot)
        # Every engaged step sees the same torque and the same spin, so the
        # power is constant and the integral is closed-form.
        power = float(torque @ spin)
        duration = float(shot.times_s[-1] - shot.times_s[0])
        assert work.sand_torque_work_j == pytest.approx(power * duration, rel=1e-12)
        assert work.driver_work_j == pytest.approx(-power * duration, rel=1e-12)
        assert work.driver_work_j != 0.0

    def test_the_ledger_refuses_a_coupled_trace(self, plate: SurfaceElements) -> None:
        shot = _shot(
            _ConstantWrenchSolver((-100.0, 0.0, 40.0), (0.0, 0.5, 0.0)),
            plate,
            coupling=_ScalarInertiaCoupling(2.5e-4),
        )
        with pytest.raises(VerificationError, match="coupled"):
            support_angular_impulse(shot)
        with pytest.raises(VerificationError, match="coupled"):
            prescribed_driver_work(shot)


class TestOnTheRealSolver:
    def test_a_coupled_wedge_plate_rotates_under_the_sand_torque(
        self, solver: DRFTSolver, plate: SurfaceElements
    ) -> None:
        angle = math.radians(6.0)
        entry = 25.0 * np.array([-math.cos(angle), 0.0, -math.sin(angle)])
        settings = ShotSettings(max_time_s=0.010, require_exit=False)
        prescribed = simulate_shot(
            solver,
            plate,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=entry),
            settings=settings,
        )
        coupling = _ScalarInertiaCoupling(1e-4)
        coupled = simulate_shot(
            solver,
            plate,
            head_mass_kg=_HEAD_MASS_KG,
            kinematics=HeadKinematics(velocity_m_s=entry),
            settings=ShotSettings(
                max_time_s=0.010,
                require_exit=False,
                rotation_mode=RotationMode.COUPLED,
            ),
            coupling=coupling,
        )
        assert prescribed.rotation_mode is RotationMode.PRESCRIBED
        assert coupled.rotation_mode is RotationMode.COUPLED
        assert np.abs(coupled.angular_velocities_rad_s).max() > 0.0
        assert not np.allclose(coupled.orientations[-1], prescribed.orientations[-1])
