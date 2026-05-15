"""
Couples the granular simulation back to the AffineDrift double-pendulum model.
"""

import numpy as np
from typing import Any


class MockDoublePendulum:
    """Mock interface to the AffineDrift double-pendulum for testing coupling."""

    def __init__(self) -> None:
        self.time = 0.0
        self.dt = 0.001
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.omega1 = 10.0  # rad/s
        self.omega2 = 15.0  # rad/s

        # Mock arm lengths
        self.l1 = 1.0
        self.l2 = 1.0

    def step(self, dt: float, external_wrench: tuple[np.ndarray, np.ndarray]) -> None:
        """
        Advance the pendulum state under gravity, actuator torques, and external wrench.
        Args:
            dt: Timestep.
            external_wrench: (force, torque) tuple at the clubhead.
        """
        force, torque = external_wrench

        # Mock dynamics: simple integration
        # In a real scenario, this involves solving the ODE with M(q)q'' + C(q,q')q' + g(q) = tau + J^T * F_ext

        # Apply some arbitrary deceleration proportional to external force
        force_mag = np.linalg.norm(force)

        self.omega1 -= force_mag * dt * 0.01
        self.omega2 -= force_mag * dt * 0.01

        self.theta1 += self.omega1 * dt
        self.theta2 += self.omega2 * dt
        self.time += dt

    def get_clubhead_pose(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns position, quat, lin_vel, ang_vel for the clubhead."""
        # Simple planar kinematics for mock
        x = self.l1 * np.sin(self.theta1) + self.l2 * np.sin(self.theta1 + self.theta2)
        y = 0.0
        z = -self.l1 * np.cos(self.theta1) - self.l2 * np.cos(self.theta1 + self.theta2)

        pos = np.array([x, y, z])

        # Quaternion (rotation around Y)
        theta_tot = self.theta1 + self.theta2
        qw = np.cos(theta_tot / 2)
        qy = np.sin(theta_tot / 2)
        quat = np.array([qw, 0.0, qy, 0.0])

        vx = self.l1 * self.omega1 * np.cos(self.theta1) + self.l2 * (
            self.omega1 + self.omega2
        ) * np.cos(theta_tot)
        vy = 0.0
        vz = self.l1 * self.omega1 * np.sin(self.theta1) + self.l2 * (
            self.omega1 + self.omega2
        ) * np.sin(theta_tot)
        lvel = np.array([vx, vy, vz])

        avel = np.array([0.0, self.omega1 + self.omega2, 0.0])

        return pos, quat, lvel, avel


class CoSimulator:
    """Manages the explicit co-simulation between Granular Backend and Double Pendulum."""

    def __init__(self, pendulum: MockDoublePendulum, backend_driver: "Any") -> None:  # type: ignore
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
