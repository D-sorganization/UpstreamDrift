"""Trajectory analysis utilities shared across ball flight simulators.

This submodule provides the TrajectoryAnalysisMixin with methods for computing
carry distance, max height, flight time, landing angle, and apex time from a
trajectory. Extracted from ball_flight_physics.py as part of P1 sprint
decomposition (issue #2486).
"""

from __future__ import annotations

import numpy as np

from src.shared.python.physics.ball_launch_conditions import TrajectoryPoint
from src.shared.python.physics.ball_properties import NUMERICAL_EPSILON


class TrajectoryAnalysisMixin:
    """Mixin providing trajectory analysis methods for ball flight simulators."""

    def calculate_carry_distance(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate total carry distance (horizontal range) in meters."""
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not trajectory:
            return 0.0
        last_pos = trajectory[-1].position
        return float(np.sqrt(last_pos[0] ** 2 + last_pos[1] ** 2))

    def calculate_max_height(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate maximum height achieved in meters."""
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not trajectory:
            return 0.0
        return float(max(p.position[2] for p in trajectory))

    def calculate_flight_time(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate total flight time in seconds."""
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not trajectory:
            return 0.0
        return trajectory[-1].time

    def _calculate_landing_angle(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate landing angle in degrees (positive for descent)."""
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if len(trajectory) < 2:
            return 0.0
        v = trajectory[-1].velocity
        v_horiz = np.linalg.norm(v[:2])
        if v_horiz < NUMERICAL_EPSILON:
            return 90.0
        return float(np.degrees(np.arctan2(-v[2], v_horiz)))

    def _calculate_apex_time(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate time to reach apex in seconds."""
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not trajectory:
            return 0.0
        max_h = -float("inf")
        apex_t = 0.0
        for p in trajectory:
            if p.position[2] > max_h:
                max_h = p.position[2]
                apex_t = p.time
        return apex_t

    def analyze_trajectory(self, trajectory: list[TrajectoryPoint]) -> dict:
        """Generate comprehensive analysis dictionary."""
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not trajectory:
            return {
                "carry_distance": 0.0,
                "max_height": 0.0,
                "flight_time": 0.0,
                "landing_angle": 0.0,
                "apex_time": 0.0,
                "trajectory_points": 0,
            }
        return {
            "carry_distance": self.calculate_carry_distance(trajectory),
            "max_height": self.calculate_max_height(trajectory),
            "flight_time": self.calculate_flight_time(trajectory),
            "landing_angle": self._calculate_landing_angle(trajectory),
            "apex_time": self._calculate_apex_time(trajectory),
            "trajectory_points": len(trajectory),
        }
