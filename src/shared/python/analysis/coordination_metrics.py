"""Coordination metrics for inter-joint coordination analysis.

Includes coupling angles, continuous relative phase, coordination patterns,
and rolling correlation analysis.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.shared.python.analysis.dataclasses import CoordinationMetrics
from src.shared.python.core.contracts import ensure, require


class CoordinationMetricsMixin:
    """Mixin for coordination analysis between joints.

    Expects the following attributes to be available on the instance:
    - times: np.ndarray
    - joint_positions: np.ndarray
    - joint_velocities: np.ndarray
    - joint_torques: np.ndarray
    - dt: float
    """

    times: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    dt: float

    def compute_coupling_angles(
        self,
        joint_idx_1: int,
        joint_idx_2: int,
    ) -> np.ndarray:
        """Compute Vector Coding coupling angles between two joints.

        The coupling angle represents the direction of the vector between
        successive data points in an Angle-Angle diagram. It quantifies
        the coordination pattern between the two segments.

        Design by Contract:
            Postconditions:
                - all angles in [0, 360)

        Args:
            joint_idx_1: Index of the first joint (x-axis)
            joint_idx_2: Index of the second joint (y-axis)

        Returns:
            Array of coupling angles in degrees [0, 360)
        """
        if (
            joint_idx_1 >= self.joint_velocities.shape[1]
            or joint_idx_2 >= self.joint_velocities.shape[1]
        ):
            return np.array([])

        vel1 = self.joint_velocities[:, joint_idx_1]
        vel2 = self.joint_velocities[:, joint_idx_2]

        gamma_rad: np.ndarray = np.arctan2(vel2, vel1)
        gamma_deg: np.ndarray = np.rad2deg(gamma_rad)

        # Normalize to [0, 360)
        gamma_deg = np.mod(gamma_deg, 360.0)

        result = np.asarray(gamma_deg)
        if len(result) > 0:
            ensure(
                np.all(result >= 0) and np.all(result < 360.0),
                "coupling angles must be in [0, 360)",
            )
        return result

    def compute_coordination_metrics(
        self, joint_idx_1: int, joint_idx_2: int
    ) -> CoordinationMetrics | None:
        """Compute coordination metrics from coupling angles (Vector Coding).

        Classifies coordination into 4 patterns:
        - In-Phase: Both segments rotating in same direction
        - Anti-Phase: Segments rotating in opposite directions
        - Proximal Leading: Proximal segment dominates motion
        - Distal Leading: Distal segment dominates motion

        Standard binning (Chang et al.):
        - In-Phase: 45 +/- 22.5, 225 +/- 22.5
        - Proximal: 0 +/- 22.5, 180 +/- 22.5
        - Distal: 90 +/- 22.5, 270 +/- 22.5
        - Anti-Phase: 135 +/- 22.5, 315 +/- 22.5

        Design by Contract:
            Postconditions:
                - all percentage fields are non-negative
                - percentages sum to approximately 100
                - mean_coupling_angle in [0, 360)
                - coordination_variability >= 0

        Args:
            joint_idx_1: Proximal joint index (X-axis)
            joint_idx_2: Distal joint index (Y-axis)

        Returns:
            CoordinationMetrics object or None
        """
        angles = self.compute_coupling_angles(joint_idx_1, joint_idx_2)
        if len(angles) == 0:
            return None

        binned = np.floor((angles + 22.5) / 45.0) % 8

        counts = np.bincount(binned.astype(int), minlength=8)
        total = len(angles)

        proximal_cnt = counts[0] + counts[4]
        in_phase_cnt = counts[1] + counts[5]
        distal_cnt = counts[2] + counts[6]
        anti_phase_cnt = counts[3] + counts[7]

        # Circular statistics for mean and variability
        angles_rad = np.deg2rad(angles)
        R = (
            np.sqrt(np.sum(np.cos(angles_rad)) ** 2 + np.sum(np.sin(angles_rad)) ** 2)
            / total
        )
        mean_angle_rad = np.arctan2(
            np.sum(np.sin(angles_rad)), np.sum(np.cos(angles_rad))
        )
        mean_angle_deg = np.degrees(mean_angle_rad) % 360.0

        if R < 1.0:
            circ_std = np.sqrt(-2 * np.log(R))
            circ_std_deg = np.degrees(circ_std)
        else:
            circ_std_deg = 0.0

        result = CoordinationMetrics(
            in_phase_pct=float(in_phase_cnt / total * 100),
            anti_phase_pct=float(anti_phase_cnt / total * 100),
            proximal_leading_pct=float(proximal_cnt / total * 100),
            distal_leading_pct=float(distal_cnt / total * 100),
            mean_coupling_angle=float(mean_angle_deg),
            coordination_variability=float(circ_std_deg),
        )

        # Postconditions
        pct_sum = (
            result.in_phase_pct
            + result.anti_phase_pct
            + result.proximal_leading_pct
            + result.distal_leading_pct
        )
        ensure(
            abs(pct_sum - 100.0) < 1e-6,
            "coordination percentages must sum to 100",
            pct_sum,
        )
        ensure(
            result.coordination_variability >= 0,
            "coordination variability must be non-negative",
            result.coordination_variability,
        )
        ensure(
            0.0 <= result.mean_coupling_angle < 360.0,
            "mean coupling angle must be in [0, 360)",
            result.mean_coupling_angle,
        )

        return result

    def compute_phase_angle(self, joint_idx: int) -> np.ndarray:
        """Compute continuous phase angle for a joint.

        Calculated using the arctangent of normalized velocity vs normalized position.
        Phi = arctan2(norm_vel, norm_pos).

        Args:
            joint_idx: Index of the joint

        Returns:
            Array of phase angles in degrees (unwrapped)
        """
        if (
            joint_idx >= self.joint_positions.shape[1]
            or joint_idx >= self.joint_velocities.shape[1]
        ):
            return np.array([])

        pos = self.joint_positions[:, joint_idx]
        vel = self.joint_velocities[:, joint_idx]

        if len(pos) < 2:
            return np.array([])

        min_p, max_p = np.min(pos), np.max(pos)
        range_p = max_p - min_p
        if range_p < 1e-6:
            norm_pos = np.zeros_like(pos)
        else:
            norm_pos = 2 * (pos - min_p) / range_p - 1.0

        max_v = np.max(np.abs(vel))
        norm_vel = np.zeros_like(vel) if max_v < 1e-06 else vel / max_v

        phase = np.arctan2(norm_vel, norm_pos)
        phase_unwrapped = np.unwrap(phase)

        return np.asarray(np.rad2deg(phase_unwrapped))

    def compute_continuous_relative_phase(
        self,
        joint_idx_1: int,
        joint_idx_2: int,
    ) -> np.ndarray:
        """Compute Continuous Relative Phase (CRP) between two joints.

        CRP = Phase1 - Phase2.
        Used to analyze coordination dynamics. Near 0 implies in-phase,
        near 180 implies anti-phase.

        Args:
            joint_idx_1: Proximal joint index
            joint_idx_2: Distal joint index

        Returns:
            Array of CRP values in degrees
        """
        phi1 = self.compute_phase_angle(joint_idx_1)
        phi2 = self.compute_phase_angle(joint_idx_2)

        if len(phi1) == 0 or len(phi2) == 0:
            return np.array([])

        crp = phi1 - phi2
        return np.asarray(crp)

    def compute_correlations(
        self,
        data_type: str = "velocity",
    ) -> tuple[np.ndarray, list[str]]:
        """Compute correlation matrix for joint data.

        Args:
            data_type: Type of data to correlate ('position', 'velocity', 'torque')

        Returns:
            Tuple of (correlation_matrix, labels)
        """
        if data_type == "position":
            data = self.joint_positions
        elif data_type == "torque":
            data = self.joint_torques
        else:
            data = self.joint_velocities

        if data.shape[1] == 0:
            return np.array([]), []

        corr_matrix = np.corrcoef(data.T)
        labels = [f"J{i}" for i in range(data.shape[1])]

        return corr_matrix, labels

    def compute_rolling_correlation(
        self,
        joint_idx_1: int,
        joint_idx_2: int,
        window_size: int = 20,
        data_type: str = "velocity",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute rolling correlation between two joints.

        Design by Contract:
            Preconditions:
                - window_size >= 2
            Postconditions:
                - all correlations in [-1, 1]

        Args:
            joint_idx_1: First joint index
            joint_idx_2: Second joint index
            window_size: Size of rolling window in samples
            data_type: 'position', 'velocity', or 'torque'

        Returns:
            Tuple of (times, correlations). Times correspond to window centers.
        """
        require(window_size >= 2, "window_size must be >= 2", window_size)

        if data_type == "position":
            data = self.joint_positions
        elif data_type == "torque":
            data = self.joint_torques
        else:
            data = self.joint_velocities

        if (
            data.shape[1] == 0
            or joint_idx_1 >= data.shape[1]
            or joint_idx_2 >= data.shape[1]
            or len(data) < window_size
        ):
            return np.array([]), np.array([])

        x = data[:, joint_idx_1]
        y = data[:, joint_idx_2]

        x_windows = sliding_window_view(x, window_shape=window_size)
        y_windows = sliding_window_view(y, window_shape=window_size)

        x_mean = np.mean(x_windows, axis=1, keepdims=True)
        y_mean = np.mean(y_windows, axis=1, keepdims=True)

        x_diff = x_windows - x_mean
        y_diff = y_windows - y_mean

        numerator = np.sum(x_diff * y_diff, axis=1)
        denominator = np.sqrt(np.sum(x_diff**2, axis=1) * np.sum(y_diff**2, axis=1))

        with np.errstate(divide="ignore", invalid="ignore"):
            correlations = numerator / denominator
        correlations[np.isnan(correlations)] = 0.0

        # Postcondition
        if len(correlations) > 0:
            ensure(
                np.all(correlations >= -1.0 - 1e-6)
                and np.all(correlations <= 1.0 + 1e-6),
                "rolling correlations must be in [-1, 1]",
            )

        valid_indices = np.arange(len(correlations)) + window_size // 2
        window_times = self.times[valid_indices]

        return window_times, correlations

    def compute_lag_matrix(
        self,
        data_type: str = "velocity",
        max_lag: float = 0.5,
    ) -> tuple[np.ndarray, list[str]]:
        """Compute time lag matrix between all pairs of joints.

        PERFORMANCE FIX: Uses parallel computation for large joint counts.

        Args:
            data_type: 'position', 'velocity', 'torque'
            max_lag: Maximum lag to search (seconds)

        Returns:
            Tuple of (lag_matrix, labels).
            Matrix[i, j] > 0 means i leads j (j lags i).
        """
        if data_type == "position":
            data = self.joint_positions
        elif data_type == "torque":
            data = self.joint_torques
        else:
            data = self.joint_velocities

        n_joints = data.shape[1]
        if n_joints == 0:
            return np.array([]), []

        fs = 1.0 / self.dt if self.dt > 0 else 0.0
        if fs <= 0:
            return np.zeros((n_joints, n_joints)), []

        try:
            from shared.python.signal_toolkit import (
                signal_processing,  # type: ignore[attr-defined]
            )
        except ImportError:
            return np.zeros((n_joints, n_joints)), []

        lag_matrix = np.zeros((n_joints, n_joints))

        if n_joints > 10:
            import os
            from concurrent.futures import ThreadPoolExecutor

            max_workers = min(os.cpu_count() or 4, 8)

            def compute_lag_pair(i: int, j: int) -> tuple[int, int, float]:
                """Compute lag for a single pair of joints."""
                lag = signal_processing.compute_time_shift(
                    data[:, i], data[:, j], fs, max_lag=max_lag
                )
                return i, j, lag

            pairs = [(i, j) for i in range(n_joints) for j in range(i + 1, n_joints)]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = executor.map(lambda p: compute_lag_pair(p[0], p[1]), pairs)

                for i, j, lag in results:
                    lag_matrix[i, j] = lag
                    lag_matrix[j, i] = -lag
        else:
            for i in range(n_joints):
                for j in range(i + 1, n_joints):
                    lag = signal_processing.compute_time_shift(
                        data[:, i], data[:, j], fs, max_lag=max_lag
                    )
                    lag_matrix[i, j] = lag
                    lag_matrix[j, i] = -lag

        labels = [f"J{i}" for i in range(n_joints)]
        return lag_matrix, labels
