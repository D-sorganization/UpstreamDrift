"""Magnus lift must use transverse spin ||w x v_hat|| in every Python flight model.

The Rust kernel and ``_python_fallback_total`` were fixed in #15042. These tests
pin the same law for ``BallFlightSimulator`` (scalar and batch force paths),
``LiftModel``/``MagnusModel`` and the three ODE flight models.

Closed-form reference: F = 0.5 rho A CL(S_t) v^2 with S_t = R w sin(theta)/v and
CL(S) = min(0.26, 0.70 S^0.645) (Penner fit); direction is the unit vector of
w x v. Spin along the flight direction (theta -> 0) must give zero force.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.physics import flight_models as fm
from src.shared.python.physics.aerodynamics import LiftModel, MagnusModel
from src.shared.python.physics.ball_launch_conditions import LaunchConditions
from src.shared.python.physics.ball_simulator import BallFlightSimulator

pytestmark = pytest.mark.unit

_SPEED = 50.0
_RPM = 2500.0
_OMEGA = _RPM * 2.0 * math.pi / 60.0


def _penner_cl(s: float) -> float:
    return min(0.26, 0.70 * s**0.645)


def _tilted_axis(theta: float) -> np.ndarray:
    """Unit spin axis at angle theta from +X (the flight direction)."""
    return np.array([math.cos(theta), -math.sin(theta), 0.0])


class TestBallFlightSimulatorTransverseSpin:
    def _expected(self, sim: BallFlightSimulator, theta: float) -> float:
        r = sim.ball.radius
        s_t = r * _OMEGA * math.sin(theta) / _SPEED
        q = 0.5 * sim.environment.air_density * sim.ball.cross_sectional_area
        return q * _penner_cl(s_t) * _SPEED**2

    @pytest.mark.parametrize("theta_deg", [10.0, 30.0, 60.0, 90.0])
    def test_scalar_magnus_magnitude_uses_transverse_spin(
        self, theta_deg: float
    ) -> None:
        sim = BallFlightSimulator()
        theta = math.radians(theta_deg)
        launch = LaunchConditions(
            velocity=_SPEED,
            launch_angle=0.0,
            spin_rate=_RPM,
            spin_axis=_tilted_axis(theta),
        )
        forces = sim._calculate_forces(np.array([_SPEED, 0.0, 0.0]), launch)
        got = float(np.linalg.norm(forces["magnus"]))
        assert got == pytest.approx(self._expected(sim, theta), rel=1e-9)

    @pytest.mark.parametrize("theta_deg", [10.0, 30.0, 60.0, 90.0])
    def test_batch_matches_closed_form(self, theta_deg: float) -> None:
        sim = BallFlightSimulator()
        theta = math.radians(theta_deg)
        launch = LaunchConditions(
            velocity=_SPEED,
            launch_angle=0.0,
            spin_rate=_RPM,
            spin_axis=_tilted_axis(theta),
        )
        vel = np.array([[_SPEED, _SPEED], [0.0, 0.0], [0.0, 0.0]])
        forces = sim._calculate_forces(vel, launch)
        got = np.linalg.norm(forces["magnus"], axis=0)
        assert got == pytest.approx([self._expected(sim, theta)] * 2, rel=1e-9)

    def test_force_vanishes_continuously_as_spin_aligns_with_velocity(self) -> None:
        sim = BallFlightSimulator()
        mags = []
        for theta in (1e-1, 1e-2, 1e-3, 1e-5):
            launch = LaunchConditions(
                velocity=_SPEED,
                launch_angle=0.0,
                spin_rate=_RPM,
                spin_axis=_tilted_axis(theta),
            )
            f = sim._calculate_forces(np.array([_SPEED, 0.0, 0.0]), launch)
            mags.append(float(np.linalg.norm(f["magnus"])))
        assert mags == sorted(mags, reverse=True)
        # Full-perpendicular force is ~0.3 N; at 1e-5 rad it must be negligible.
        assert mags[-1] < 1e-3 * self._expected(sim, math.pi / 2)


class TestAerodynamicsModelsTransverseSpin:
    @pytest.mark.parametrize("model_cls", [LiftModel, MagnusModel])
    def test_force_magnitude_matches_closed_form(self, model_cls: type) -> None:
        model = model_cls()
        theta = math.radians(25.0)
        v = np.array([_SPEED, 0.0, 0.0])
        spin = _OMEGA * _tilted_axis(theta)
        rho = 1.2
        force = model.calculate(v, spin, rho)
        s_t = model.ball_radius * _OMEGA * math.sin(theta) / _SPEED
        expected = 0.5 * rho * model.ball_area * _penner_cl(s_t) * _SPEED**2
        assert float(np.linalg.norm(force)) == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("model_cls", [LiftModel, MagnusModel])
    def test_near_parallel_spin_gives_near_zero_force(self, model_cls: type) -> None:
        model = model_cls()
        v = np.array([_SPEED, 0.0, 0.0])
        spin = _OMEGA * _tilted_axis(1e-6)
        force = model.calculate(v, spin, 1.2)
        assert float(np.linalg.norm(force)) < 1e-4


class TestOdeFlightModelsTransverseSpin:
    """Early-time lateral acceleration of the ODE models against the closed form.

    A 60 deg launch with pure sidespin (axis +Z) has transverse factor
    sin(theta) = cos(60 deg) = 0.5. Over a 0.02 s window the velocity direction
    and speed change by < 1 %, and drag/gravity act only in the x-z plane, so
    v_y(t) ~= a_lift * t with a_lift = 0.5 rho A CL(S_t) v^2 / m along +Y.
    """

    _T = 0.02
    _LAUNCH = fm.UnifiedLaunchConditions(
        ball_speed=40.0,
        launch_angle=math.radians(60.0),
        spin_rate=1000.0,
        spin_axis_angle=math.pi / 2.0,
    )

    def _lateral_accel(self, model: fm.BallFlightModel) -> float:
        result = model.simulate(self._LAUNCH, max_time=self._T, dt=0.005)
        return float(result.trajectory[-1].velocity[1]) / self._T

    def _transverse_ratio(self) -> float:
        launch = self._LAUNCH
        omega = launch.spin_rate * 2.0 * math.pi / 60.0
        return (
            launch.ball_radius
            * omega
            * math.cos(launch.launch_angle)
            / launch.ball_speed
        )

    def _accel_from_cl(self, cl: float) -> float:
        launch = self._LAUNCH
        area = math.pi * launch.ball_radius**2
        return (
            0.5
            * launch.air_density
            * area
            * cl
            * launch.ball_speed**2
            / launch.ball_mass
        )

    def test_waterloo_penner_lift_uses_transverse_spin(self) -> None:
        c = fm.WATERLOO_PENNER_COEFFICIENTS
        s_t = self._transverse_ratio()
        cl = c.cl0 + c.lift_scale * s_t**c.lift_exponent
        expected = self._accel_from_cl(cl)
        got = self._lateral_accel(fm.WaterlooPennerModel())
        assert got == pytest.approx(expected, rel=0.03)

    def test_macdonald_hanzely_lift_uses_transverse_spin(self) -> None:
        s_t = self._transverse_ratio()
        expected = self._accel_from_cl(min(0.20, 1.7 * s_t))
        got = self._lateral_accel(fm.MacDonaldHanzelyModel(cd=0.225, cl=0.20))
        assert got == pytest.approx(expected, rel=0.03)

    def test_constant_coefficient_lift_uses_transverse_spin(self) -> None:
        spec = fm.ConstantCoefficientSpec(
            name="t",
            description="t",
            reference="t",
            cd=0.225,
            cl=0.20,
            spin_decay=0.0,
        )
        s_t = self._transverse_ratio()
        expected = self._accel_from_cl(min(0.20, 1.7 * s_t))
        got = self._lateral_accel(fm.ConstantCoefficientModel(spec))
        assert got == pytest.approx(expected, rel=0.03)
