"""Swing metrics and kinematics analysis components."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from shared.python.analysis.dataclasses import PeakInfo


class SwingMetricsMixin:
    """Mixin for swing kinematics and metrics.

    Expects the following attributes to be available on the instance:
    - times: np.ndarray
    - joint_positions: np.ndarray
    - club_head_speed: np.ndarray | None
    - dt: float
    """

    # Type hints for anticipated attributes
    times: np.ndarray
    joint_positions: np.ndarray
    club_head_speed: np.ndarray | None
    dt: float

    def compute_range_of_motion(self, joint_idx: int) -> tuple[float, float, float]:
        """Compute range of motion for a joint.

        Args:
            joint_idx: Joint index

        Returns:
            (min_angle, max_angle, rom) in degrees
        """
        if joint_idx >= self.joint_positions.shape[1]:
            return (0.0, 0.0, 0.0)

        angles_deg = np.rad2deg(self.joint_positions[:, joint_idx])
        min_angle = float(np.min(angles_deg))
        max_angle = float(np.max(angles_deg))
        rom = max_angle - min_angle

        return (min_angle, max_angle, rom)

    def compute_tempo(self) -> tuple[float, float, float] | None:
        """Compute swing tempo (backswing:downswing ratio).

        Uses club head speed to identify transition point.

        Returns:
            (backswing_duration, downswing_duration, ratio) or None
        """
        if self.club_head_speed is None or len(self.club_head_speed) < 10:
            return None

        # Find peak club head speed (impact)
        impact_idx = np.argmax(self.club_head_speed)

        # Find transition (minimum speed before impact, after initial movement)
        # Look in first 70% of time before impact
        search_end = int(impact_idx * 0.7)
        if search_end <= 5:
            return None

        # Smooth speed for better transition detection
        segment = self.club_head_speed[:impact_idx]
        window_len = min(11, len(segment))
        if window_len % 2 == 0:
            window_len -= 1
        smoothed_speed = (
            segment if window_len <= 3 else savgol_filter(segment, window_len, 3)
        )

        # Find minimum after initial acceleration
        start_search = 5
        transition_idx = start_search + np.argmin(
            smoothed_speed[start_search:search_end],
        )

        # Compute durations
        backswing_duration = float(self.times[transition_idx] - self.times[0])
        downswing_duration = float(self.times[impact_idx] - self.times[transition_idx])

        ratio = (
            backswing_duration / downswing_duration if downswing_duration > 0 else 0.0
        )

        return (backswing_duration, downswing_duration, ratio)

    def compute_x_factor(
        self,
        shoulder_joint_idx: int,
        hip_joint_idx: int,
    ) -> np.ndarray | None:
        """Compute X-Factor (shoulder-hip rotation difference).

        Args:
            shoulder_joint_idx: Index of shoulder/torso rotation joint
            hip_joint_idx: Index of hip rotation joint

        Returns:
            X-Factor time series (degrees) or None
        """
        if (
            shoulder_joint_idx >= self.joint_positions.shape[1]
            or hip_joint_idx >= self.joint_positions.shape[1]
        ):
            return None

        shoulder_rotation = np.rad2deg(self.joint_positions[:, shoulder_joint_idx])
        hip_rotation = np.rad2deg(self.joint_positions[:, hip_joint_idx])

        return np.asarray(shoulder_rotation - hip_rotation)

    def compute_x_factor_stretch(
        self,
        shoulder_joint_idx: int,
        hip_joint_idx: int,
    ) -> tuple[np.ndarray, float] | None:
        """Compute X-Factor velocity (stretch rate) and peak stretch.

        Args:
            shoulder_joint_idx: Index of shoulder/torso rotation joint
            hip_joint_idx: Index of hip rotation joint

        Returns:
            Tuple of (x_factor_velocity_array, peak_stretch_rate) or None
        """
        x_factor = self.compute_x_factor(shoulder_joint_idx, hip_joint_idx)
        if x_factor is None:
            return None

        # Calculate derivative (finite difference)
        x_factor_velocity = np.gradient(x_factor, self.dt)
        peak_stretch_rate = float(np.max(np.abs(x_factor_velocity)))

        return x_factor_velocity, peak_stretch_rate

    def find_club_head_speed_peak(self) -> PeakInfo | None:
        """Abstract method expected to be implemented by host or other mixin."""
        pass

    def detect_impact_time(self) -> float | None:
        """Detect ball impact time.

        Uses peak club head speed as proxy for impact.

        Returns:
            Impact time in seconds, or None
        """
        peak = self.find_club_head_speed_peak()
        return peak.time if peak else None
