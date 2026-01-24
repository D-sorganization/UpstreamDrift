"""Multi-Model Ball Flight Physics Framework.

This module provides a unified interface for multiple golf ball flight models,
enabling comparison and validation across different physics implementations.

Refactored to address MASSIVE DRY violations identified in the
Pragmatic Programmer assessment (2026-01-23).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import cast

import numpy as np
from scipy.integrate import solve_ivp

from src.shared.python.logging_config import get_logger

from .physics_constants import (
    AIR_DENSITY_SEA_LEVEL_KG_M3,
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
    GRAVITY_M_S2,
    MIN_SPEED_THRESHOLD_M_S,
    NUMERICAL_EPSILON,
)

logger = get_logger(__name__)

# Constants (re-exported)
GOLF_BALL_MASS = float(GOLF_BALL_MASS_KG)
GOLF_BALL_RADIUS = float(GOLF_BALL_RADIUS_M)
STD_AIR_DENSITY = float(AIR_DENSITY_SEA_LEVEL_KG_M3)
STD_GRAVITY = float(GRAVITY_M_S2)
MIN_SPEED_THRESHOLD = float(MIN_SPEED_THRESHOLD_M_S)


class FlightModelType(Enum):
    """Available ball flight physics models."""

    WATERLOO_PENNER = "waterloo_penner"
    MACDONALD_HANZELY = "macdonald_hanzely"
    NATHAN = "nathan"
    BALLANTYNE = "ballantyne"
    JCOLE = "jcole"
    ROSPIE_DL = "rospie_dl"
    CHARRY_L3 = "charry_l3"


@dataclass
class UnifiedLaunchConditions:
    """Standardized launch conditions SI units."""

    ball_speed: float
    launch_angle: float
    azimuth_angle: float = 0.0
    spin_rate: float = 2500.0
    spin_axis_angle: float = 0.0
    ball_mass: float = GOLF_BALL_MASS
    ball_radius: float = GOLF_BALL_RADIUS
    air_density: float = STD_AIR_DENSITY
    gravity: float = STD_GRAVITY
    wind_speed: float = 0.0
    wind_direction: float = 0.0

    def get_initial_velocity(self) -> np.ndarray:
        ca, sa = math.cos(self.azimuth_angle), math.sin(self.azimuth_angle)
        cv, sv = math.cos(self.launch_angle), math.sin(self.launch_angle)
        return np.array(
            [self.ball_speed * cv * ca, self.ball_speed * cv * sa, self.ball_speed * sv]
        )

    def get_spin_vector(self) -> np.ndarray:
        omega = self.spin_rate * 2 * math.pi / 60
        backspin = omega * math.cos(self.spin_axis_angle)
        sidespin = omega * math.sin(self.spin_axis_angle)
        return np.array(
            [
                sidespin * math.sin(self.azimuth_angle),
                -backspin,
                sidespin * math.cos(self.azimuth_angle),
            ]
        )

    def get_wind_vector(self) -> np.ndarray:
        return np.array(
            [
                -self.wind_speed * math.cos(self.wind_direction),
                -self.wind_speed * math.sin(self.wind_direction),
                0.0,
            ]
        )


@dataclass
class TrajectoryPoint:
    """Single point in trajectory."""

    time: float
    position: np.ndarray
    velocity: np.ndarray


@dataclass
class FlightResult:
    """Result of simulation."""

    trajectory: list[TrajectoryPoint]
    model_name: str
    carry_distance: float = 0.0
    max_height: float = 0.0
    flight_time: float = 0.0
    landing_angle: float = 0.0
    lateral_deviation: float = 0.0


class BallFlightModel(ABC):
    """Base class for flight models (DRY-optimized)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def reference(self) -> str: ...

    @abstractmethod
    def simulate(
        self, launch: UnifiedLaunchConditions, max_time: float = 10.0, dt: float = 0.01
    ) -> FlightResult: ...

    def _compute_metrics(self, trajectory: list[TrajectoryPoint]) -> FlightResult:
        """Standardized metrics computation (Consolidated for DRY)."""
        if not trajectory:
            return FlightResult([], self.name)

        pos = np.array([p.position for p in trajectory])
        carry = math.sqrt(pos[-1, 0] ** 2 + pos[-1, 1] ** 2)
        max_h = float(np.max(pos[:, 2]))
        time = trajectory[-1].time
        lateral = float(pos[-1, 1])

        angle = 0.0
        if len(trajectory) >= 2:
            v = trajectory[-1].velocity
            v_horiz = math.sqrt(v[0] ** 2 + v[1] ** 2)
            angle = (
                math.degrees(math.atan2(-v[2], v_horiz))
                if v_horiz > MIN_SPEED_THRESHOLD
                else 90.0
            )

        return FlightResult(trajectory, self.name, carry, max_h, time, angle, lateral)

    def _run_ode_simulation(
        self,
        launch: UnifiedLaunchConditions,
        deriv_func: Callable[[float, np.ndarray], np.ndarray],
        max_time: float,
        dt: float,
    ) -> FlightResult:
        """Unified ODE integration loop (Consolidated for DRY)."""
        v0 = launch.get_initial_velocity()
        y0 = np.array([0.0, 0.0, 0.0, v0[0], v0[1], v0[2]])

        def ground_ev(t: float, y: np.ndarray) -> float:
            return float(y[2])

        # Type safe attribute assignment for solve_ivp
        setattr(ground_ev, "terminal", True)  # noqa: B010
        setattr(ground_ev, "direction", -1)  # noqa: B010

        sol = solve_ivp(
            deriv_func,
            (0, max_time),
            y0,
            method="RK45",
            events=ground_ev,
            dense_output=True,
            max_step=0.1,
        )

        t_eval = np.arange(0, sol.t[-1], dt)
        points = [
            TrajectoryPoint(float(t), sol.sol(t)[:3], sol.sol(t)[3:]) for t in t_eval
        ]
        if sol.t[-1] not in t_eval:
            points.append(TrajectoryPoint(float(sol.t[-1]), sol.y[:3, -1], sol.y[3:, -1]))

        return self._compute_metrics(points)


class WaterlooPennerModel(BallFlightModel):
    """Waterloo/Penner model implementation."""

    def __init__(
        self,
        cd0: float = 0.21,
        cd1: float = 0.05,
        cd2: float = 0.02,
        cl0: float = 0.00,
        cl1: float = 0.38,
        cl2: float = 0.08,
        cl_max: float = 0.25,
    ) -> None:
        self.params = (cd0, cd1, cd2, cl0, cl1, cl2, cl_max)

    @property
    def name(self) -> str:
        return "Waterloo/Penner"

    @property
    def description(self) -> str:
        return "Quadratic Cd/Cl from Waterloo tunnel data"

    @property
    def reference(self) -> str:
        return "McPhee et al. (Waterloo)"

    def simulate(
        self, launch: UnifiedLaunchConditions, max_time: float = 10.0, dt: float = 0.01
    ) -> FlightResult:
        cd0, cd1, cd2, cl0, cl1, cl2, cl_max = self.params
        omega_v = launch.get_spin_vector()
        omega_m = np.linalg.norm(omega_v)
        wind_v = launch.get_wind_vector()
        area = math.pi * launch.ball_radius**2

        def derivatives(t: float, y: np.ndarray) -> np.ndarray:
            v_val = cast(np.ndarray, y[3:])
            v_rel = v_val - wind_v
            speed = np.linalg.norm(v_rel)
            if speed < MIN_SPEED_THRESHOLD:
                return np.array(
                    [v_val[0], v_val[1], v_val[2], 0.0, 0.0, -launch.gravity]
                )

            vu = v_rel / speed
            s = (omega_m * launch.ball_radius) / speed
            cd = cd0 + cd1 * s + cd2 * s**2
            cl_val = float(cl0 + cl1 * s + cl2 * s**2)
            cl = min(cl_max, cl_val)

            acc = (
                -(0.5 * launch.air_density * speed**2 * cd * area / launch.ball_mass)
                * vu
            )
            if omega_m > 0:
                cross = np.cross(omega_v / omega_m, vu)
                if np.linalg.norm(cross) > NUMERICAL_EPSILON:
                    acc += (
                        0.5
                        * launch.air_density
                        * speed**2
                        * cl
                        * area
                        / launch.ball_mass
                    ) * (cross / np.linalg.norm(cross))

            acc[2] -= launch.gravity
            return np.array([v_val[0], v_val[1], v_val[2], acc[0], acc[1], acc[2]])

        return self._run_ode_simulation(launch, derivatives, max_time, dt)


class MacDonaldHanzelyModel(BallFlightModel):
    """MacDonald-Hanzely model implementation."""

    def __init__(
        self, cd: float = 0.225, cl: float = 0.20, decay: float = 0.05
    ) -> None:
        self.cd, self.cl, self.decay = cd, cl, decay

    @property
    def name(self) -> str:
        return "MacDonald-Hanzely"

    @property
    def description(self) -> str:
        return "ODE model with exponential spin decay"

    @property
    def reference(self) -> str:
        return "MacDonald & Hanzely (1991)"

    def simulate(
        self, launch: UnifiedLaunchConditions, max_time: float = 10.0, dt: float = 0.01
    ) -> FlightResult:
        omega_0 = launch.spin_rate * 2 * math.pi / 60
        spin_axis = launch.get_spin_vector()
        if np.linalg.norm(spin_axis) > 0:
            spin_axis /= np.linalg.norm(spin_axis)
        wind_v = launch.get_wind_vector()
        area = math.pi * launch.ball_radius**2
        k_drag = 0.5 * launch.air_density * area * self.cd / launch.ball_mass

        def derivatives(t: float, y: np.ndarray) -> np.ndarray:
            v_val = cast(np.ndarray, y[3:])
            v_rel = v_val - wind_v
            speed = np.linalg.norm(v_rel)
            if speed < MIN_SPEED_THRESHOLD:
                return np.array(
                    [v_val[0], v_val[1], v_val[2], 0.0, 0.0, -launch.gravity]
                )

            omega = omega_0 * math.exp(-self.decay * t)
            vu = v_rel / speed
            acc = -k_drag * speed**2 * vu

            if omega > 0:
                cl_eff = self.cl * (omega * launch.ball_radius / speed)
                cross = np.cross(spin_axis, vu)
                if np.linalg.norm(cross) > NUMERICAL_EPSILON:
                    acc += (
                        0.5
                        * launch.air_density
                        * area
                        * cl_eff
                        * speed**2
                        / launch.ball_mass
                    ) * (cross / np.linalg.norm(cross))

            acc[2] -= launch.gravity
            return np.array([v_val[0], v_val[1], v_val[2], acc[0], acc[1], acc[2]])

        return self._run_ode_simulation(launch, derivatives, max_time, dt)
