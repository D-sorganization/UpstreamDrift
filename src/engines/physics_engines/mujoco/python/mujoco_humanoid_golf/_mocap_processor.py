from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import butter, filtfilt


class MotionCaptureProcessor:
    """Process and filter motion capture data."""

    @staticmethod
    def filter_trajectory(
        times: np.ndarray,
        positions: np.ndarray,
        cutoff_frequency: float = 6.0,
        sampling_rate: float = 120.0,
    ) -> np.ndarray:
        """Apply low-pass Butterworth filter to trajectory.

        Args:
            times: Time array [N]
            positions: Position array [N x 3] or [N x nv]
            cutoff_frequency: Cutoff frequency in Hz
            sampling_rate: Sampling rate in Hz

        Returns:
            Filtered positions [N x 3] or [N x nv]
        """
        # Design filter
        if times is None:
            raise ValueError("times must be provided")
        nyquist = sampling_rate / 2.0
        normalized_cutoff = cutoff_frequency / nyquist
        b, a = butter(4, normalized_cutoff, btype="low")

        # Apply filter to each column
        filtered = np.zeros_like(positions)
        for i in range(positions.shape[1]):
            filtered[:, i] = filtfilt(b, a, positions[:, i])

        return filtered

    @staticmethod
    def compute_velocities(
        times: np.ndarray,
        positions: np.ndarray,
        method: str = "finite_difference",
    ) -> np.ndarray:
        """Compute velocities from position data.

        Args:
            times: Time array [N]
            positions: Position array [N x d]
            method: Method ("finite_difference", "spline")

        Returns:
            Velocities [N x d]
        """
        if times is None:
            raise ValueError("times must be provided")
        if method == "finite_difference":
            # Central differences
            velocities = np.zeros_like(positions)
            velocities[1:-1] = (positions[2:] - positions[:-2]) / (
                times[2:] - times[:-2]
            )[:, np.newaxis]
            velocities[0] = (positions[1] - positions[0]) / (times[1] - times[0])
            velocities[-1] = (positions[-1] - positions[-2]) / (times[-1] - times[-2])

        elif method == "spline":
            # Cubic spline derivatives
            velocities = np.zeros_like(positions)
            for i in range(positions.shape[1]):
                spline = CubicSpline(times, positions[:, i])
                velocities[:, i] = spline(times, nu=1)

        return velocities

    @staticmethod
    def compute_accelerations(
        times: np.ndarray,
        velocities: np.ndarray,
        method: str = "finite_difference",
    ) -> np.ndarray:
        """Compute accelerations from velocity data.

        Args:
            times: Time array [N]
            velocities: Velocity array [N x d]
            method: Method ("finite_difference", "spline")

        Returns:
            Accelerations [N x d]
        """
        if times is None:
            raise ValueError("times must be provided")
        if method == "finite_difference":
            accelerations = np.zeros_like(velocities)
            accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (
                times[2:] - times[:-2]
            )[:, np.newaxis]
            accelerations[0] = (velocities[1] - velocities[0]) / (times[1] - times[0])
            accelerations[-1] = (velocities[-1] - velocities[-2]) / (
                times[-1] - times[-2]
            )

        elif method == "spline":
            accelerations = np.zeros_like(velocities)
            for i in range(velocities.shape[1]):
                spline = CubicSpline(times, velocities[:, i])
                accelerations[:, i] = spline(times, nu=1)

        return accelerations

    @staticmethod
    def resample_trajectory(
        times: np.ndarray,
        trajectory: np.ndarray,
        new_times: np.ndarray,
        method: str = "cubic",
    ) -> np.ndarray:
        """Resample trajectory to new time points.

        Args:
            times: Original time array [N]
            trajectory: Original trajectory [N x d]
            new_times: New time points [M]
            method: Interpolation method ("linear", "cubic")

        Returns:
            Resampled trajectory [M x d]
        """
        if times is None:
            raise ValueError("times must be provided")
        resampled = np.zeros((len(new_times), trajectory.shape[1]))

        for i in range(trajectory.shape[1]):
            if method == "cubic":
                spline = CubicSpline(times, trajectory[:, i])
                resampled[:, i] = spline(new_times)
            else:
                interp = interp1d(times, trajectory[:, i], kind=method)
                resampled[:, i] = interp(new_times)

        return resampled

    @staticmethod
    def time_normalize(
        times: np.ndarray,
        trajectory: np.ndarray,
        num_samples: int = 101,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Time-normalize trajectory to 0-100% of motion.

        Useful for comparing motions of different durations.

        Args:
            times: Time array [N]
            trajectory: Trajectory [N x d]
            num_samples: Number of samples in normalized trajectory

        Returns:
            Tuple of (normalized_times [M], normalized_trajectory [M x d])
        """
        # Normalize time to [0, 1]
        if times is None:
            raise ValueError("times must be provided")
        normalized_times = np.linspace(0, 1, num_samples)

        # Time normalize original
        time_fraction = (times - times[0]) / (times[-1] - times[0])

        # Resample
        normalized_trajectory = np.zeros((num_samples, trajectory.shape[1]))
        for i in range(trajectory.shape[1]):
            interp = interp1d(time_fraction, trajectory[:, i], kind="cubic")
            normalized_trajectory[:, i] = interp(normalized_times)

        return normalized_times, normalized_trajectory
