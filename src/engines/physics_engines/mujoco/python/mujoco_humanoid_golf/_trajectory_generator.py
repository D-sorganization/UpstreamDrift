from __future__ import annotations

import numpy as np


class TrajectoryGenerator:
    """Generate smooth trajectories for control.

    Useful for generating reference trajectories for controllers.
    """

    @staticmethod
    def minimum_jerk_trajectory(
        start: np.ndarray,
        goal: np.ndarray,
        duration: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate minimum jerk trajectory.

        Minimum jerk trajectories are smooth and human-like.

        Args:
            start: Starting position [n]
            goal: Goal position [n]
            duration: Trajectory duration [s]
            dt: Time step [s]

        Returns:
            Tuple of (positions, velocities, accelerations)
            Each is [num_steps x n]
        """
        if start is None:
            raise ValueError("start must be provided")
        num_steps = int(duration / dt)
        t = np.linspace(0, duration, num_steps)

        # Minimum jerk polynomial
        # s(t) = a_0 + a_1 t + ... + a_5 t^5
        # with boundary conditions:
        # s(0) = 0, ṡ(0) = 0, s̈(0) = 0
        # s(T) = 1, ṡ(T) = 0, s̈(T) = 0

        tau = t / duration  # Normalized time [0, 1]
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
        s_ddot = (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2

        # Interpolate
        positions = (
            start[np.newaxis, :] + (goal - start)[np.newaxis, :] * s[:, np.newaxis]
        )
        velocities = (goal - start)[np.newaxis, :] * s_dot[:, np.newaxis]
        accelerations = (goal - start)[np.newaxis, :] * s_ddot[:, np.newaxis]

        return positions, velocities, accelerations

    @staticmethod
    def quintic_spline(
        waypoints: np.ndarray,
        duration: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate quintic spline through waypoints.

        Args:
            waypoints: Waypoints [num_waypoints x n]
            duration: Total trajectory duration [s]
            dt: Time step [s]

        Returns:
            Tuple of (positions, velocities, accelerations)
        """
        # Simplified: use minimum jerk between consecutive waypoints
        if waypoints is None:
            raise ValueError("waypoints must be provided")
        all_positions = []
        all_velocities = []
        all_accelerations = []

        num_segments = len(waypoints) - 1
        segment_duration = duration / num_segments

        for i in range(num_segments):
            pos, vel, acc = TrajectoryGenerator.minimum_jerk_trajectory(
                waypoints[i],
                waypoints[i + 1],
                segment_duration,
                dt,
            )

            all_positions.append(pos)
            all_velocities.append(vel)
            all_accelerations.append(acc)

        positions = np.vstack(all_positions)
        velocities = np.vstack(all_velocities)
        accelerations = np.vstack(all_accelerations)

        return positions, velocities, accelerations
