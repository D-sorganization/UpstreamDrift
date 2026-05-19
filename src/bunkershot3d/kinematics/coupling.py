"""
Couples the granular simulation back to the AffineDrift double-pendulum model.
"""

import numpy as np
from typing import Any


import math
from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumDynamics,
    DoublePendulumState,
    DoublePendulumParameters,
)


class CoupledDoublePendulum:
    """Interface to the AffineDrift double-pendulum for co-simulation."""

    def __init__(self) -> None:
        self.time = 0.0
        self.dt = 0.001
        self.state = DoublePendulumState(
            theta1=0.0, theta2=0.0, omega1=10.0, omega2=15.0
        )
        self.params = DoublePendulumParameters.default()

        self.external_tau1 = 0.0
        self.external_tau2 = 0.0

        def tau1_fn(t: float, state: DoublePendulumState) -> float:
            return self.external_tau1

        def tau2_fn(t: float, state: DoublePendulumState) -> float:
            return self.external_tau2

        self.dynamics = DoublePendulumDynamics(
            parameters=self.params,
            forcing_functions=(tau1_fn, tau2_fn)
        )

    def step(self, dt: float, external_wrench: tuple[np.ndarray, np.ndarray]) -> None:
        """
        Advance the pendulum state under gravity, actuator torques, and external wrench.
        """
        force, torque = external_wrench
        Fx, Fy, Fz = force
        Tx, Ty, Tz = torque

        theta1 = self.state.theta1
        theta2 = self.state.theta2
        l1 = self.params.upper_segment.length_m
        l2 = self.params.lower_segment.length_m

        # Jacobian for mapping Cartesian forces to joint torques
        dx_dt1 = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
        dz_dt1 = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)
        dx_dt2 = l2 * math.cos(theta1 + theta2)
        dz_dt2 = l2 * math.sin(theta1 + theta2)

        self.external_tau1 = Fx * dx_dt1 + Fz * dz_dt1 + Ty
        self.external_tau2 = Fx * dx_dt2 + Fz * dz_dt2 + Ty

        self.state = self.dynamics.step(self.time, self.state, dt)
        self.time += dt

    def get_clubhead_pose(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns position, quat, lin_vel, ang_vel for the clubhead."""
        theta1 = self.state.theta1
        theta2 = self.state.theta2
        omega1 = self.state.omega1
        omega2 = self.state.omega2
        l1 = self.params.upper_segment.length_m
        l2 = self.params.lower_segment.length_m

        x = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
        y = 0.0
        z = -l1 * np.cos(theta1) - l2 * np.cos(theta1 + theta2)
        pos = np.array([x, y, z])

        theta_tot = theta1 + theta2
        qw = np.cos(theta_tot / 2)
        qy = np.sin(theta_tot / 2)
        quat = np.array([qw, 0.0, qy, 0.0])

        vx = l1 * omega1 * np.cos(theta1) + l2 * (omega1 + omega2) * np.cos(theta_tot)
        vy = 0.0
        vz = l1 * omega1 * np.sin(theta1) + l2 * (omega1 + omega2) * np.sin(theta_tot)
        lvel = np.array([vx, vy, vz])

        avel = np.array([0.0, omega1 + omega2, 0.0])

        return pos, quat, lvel, avel

class CoSimulator:
    """Manages the explicit co-simulation between Granular Backend and Double Pendulum."""

    def __init__(self, pendulum: CoupledDoublePendulum, backend_driver: "Any") -> None:  # type: ignore
        self.pendulum = pendulum
        self.backend = backend_driver
        self.coupling_scheme = "Gauss-Seidel"  # explicit staggering

    def step(self, dt_macro: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform one co-simulation step.
        Returns:
            The wrench applied during this step.
        """
        # 1. Query current wrench from granular backend (mocked here)
        # In real implementation, this reads from the backend's current state
        wrench_force = np.array([10.0, 0.0, 5.0])  # Mock force from sand
        wrench_torque = np.array([0.0, 1.0, 0.0])  # Mock torque

        external_wrench = (wrench_force, wrench_torque)

        # 2. Advance pendulum using the constant wrench over dt_macro
        self.pendulum.step(dt_macro, external_wrench)

        # 3. Get new kinematic state
        pos, quat, lvel, avel = self.pendulum.get_clubhead_pose()

        # 4. Push new kinematic constraint to granular backend
        # self.backend.set_kinematic_target(pos, quat, lvel, avel)

        # 5. Advance granular backend
        # self.backend.step(dt_macro)

        return external_wrench
