"""Comparative analysis module for comparing two golf swings.

This module provides tools to align and compare two sets of swing data,
calculating differences in kinematics, kinetics, and timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from scipy import interpolate

from src.shared.python import signal_processing

if TYPE_CHECKING:
    pass  # pragma: no cover


class RecorderInterface(Protocol):
    """Protocol for a recorder that provides time series data."""

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray | list]:
        """Extract time series for a specific field.

        Args:
            field_name: Name of the field

        Returns:
            Tuple of (times, values)
        """
        ...  # pragma: no cover


@dataclass
class ComparisonMetric:
    """Result of a metric comparison between two swings."""

    name: str
    value_a: float
    value_b: float
    difference: float  # a - b
    percent_diff: float  # (a - b) / mean * 100


@dataclass
class AlignedSignals:
    """Container for two signals aligned on a common time base."""

    times: np.ndarray  # Normalized time (0.0 to 1.0)
    signal_a: np.ndarray
    signal_b: np.ndarray
    error_curve: np.ndarray  # a - b
    rms_error: float
    correlation: float


class ComparativeSwingAnalyzer:
    """Analyzes and compares two recorded swings."""

    def __init__(
        self,
        recorder_a: RecorderInterface,
        recorder_b: RecorderInterface,
        name_a: str = "Swing A",
        name_b: str = "Swing B",
    ) -> None:
        """Initialize with two recorders.

        Args:
            recorder_a: First swing recorder
            recorder_b: Second swing recorder
            name_a: Label for first swing
            name_b: Label for second swing
        """
        self.recorder_a = recorder_a
        self.recorder_b = recorder_b
        self.name_a = name_a
        self.name_b = name_b

    def align_signals(
        self,
        field_name: str,
        num_points: int = 100,
        joint_idx: int | None = None,
    ) -> AlignedSignals | None:
        """Align two signals by normalizing time to 0-100%.

        Args:
            field_name: Name of data field (e.g. 'joint_velocities')
            num_points: Number of points for normalized time base
            joint_idx: Index if field is multidimensional

        Returns:
            AlignedSignals object or None if data missing
        """
        # Get data
        t_a, data_a = self.recorder_a.get_time_series(field_name)
        t_b, data_b = self.recorder_b.get_time_series(field_name)

        # Convert to numpy
        data_a = np.asarray(data_a)
        data_b = np.asarray(data_b)

        if len(t_a) < 2 or len(t_b) < 2:
            return None

        # Handle multidimensional data
        if joint_idx is not None:
            if data_a.ndim > 1:
                if joint_idx >= data_a.shape[1]:
                    return None
                data_a = data_a[:, joint_idx]
            if data_b.ndim > 1:
                if joint_idx >= data_b.shape[1]:
                    return None
                data_b = data_b[:, joint_idx]

        # Normalize time to 0..1
        t_a_norm = (t_a - t_a[0]) / (t_a[-1] - t_a[0])
        t_b_norm = (t_b - t_b[0]) / (t_b[-1] - t_b[0])

        # Common time base
        t_common = np.linspace(0, 1, num_points)

        # Interpolate
        interp_a = interpolate.interp1d(t_a_norm, data_a, kind="linear")
        interp_b = interpolate.interp1d(t_b_norm, data_b, kind="linear")

        sig_a_resampled = interp_a(t_common)
        sig_b_resampled = interp_b(t_common)

        # Compute differences
        error_curve = sig_a_resampled - sig_b_resampled
        rms = float(np.sqrt(np.mean(error_curve**2)))

        # Compute correlation
        if np.std(sig_a_resampled) > 1e-6 and np.std(sig_b_resampled) > 1e-6:
            correlation = float(np.corrcoef(sig_a_resampled, sig_b_resampled)[0, 1])
        else:
            correlation = 0.0

        return AlignedSignals(
            times=t_common,
            signal_a=sig_a_resampled,
            signal_b=sig_b_resampled,
            error_curve=error_curve,
            rms_error=rms,
            correlation=correlation,
        )

    def compare_scalars(
        self, metric_name: str, val_a: float, val_b: float
    ) -> ComparisonMetric:
        """Create comparison metric for two scalar values.

        Args:
            metric_name: Name of metric
            val_a: Value from swing A
            val_b: Value from swing B

        Returns:
            ComparisonMetric object
        """
        diff = val_a - val_b
        mean = (val_a + val_b) / 2.0
        percent = (diff / mean * 100) if abs(mean) > 1e-9 else 0.0

        return ComparisonMetric(
            name=metric_name,
            value_a=val_a,
            value_b=val_b,
            difference=diff,
            percent_diff=percent,
        )

    def compare_peak_speeds(self) -> ComparisonMetric | None:
        """Compare peak club head speeds."""
        _, speed_a = self.recorder_a.get_time_series("club_head_speed")
        _, speed_b = self.recorder_b.get_time_series("club_head_speed")

        if len(speed_a) == 0 or len(speed_b) == 0:
            return None

        max_a = float(np.max(speed_a))
        max_b = float(np.max(speed_b))

        return self.compare_scalars("Peak Club Speed", max_a, max_b)

    def compare_durations(self) -> ComparisonMetric | None:
        """Compare swing durations."""
        t_a, _ = self.recorder_a.get_time_series("club_head_speed")  # use any field
        t_b, _ = self.recorder_b.get_time_series("club_head_speed")

        if len(t_a) < 2 or len(t_b) < 2:
            return None

        dur_a = float(t_a[-1] - t_a[0])
        dur_b = float(t_b[-1] - t_b[0])

        return self.compare_scalars("Swing Duration", dur_a, dur_b)

    def generate_comparison_report(self) -> dict[str, Any]:
        """Generate a dictionary summary of the comparison."""
        metrics = []

        speed_comp = self.compare_peak_speeds()
        if speed_comp:
            metrics.append(speed_comp)

        dur_comp = self.compare_durations()
        if dur_comp:
            metrics.append(dur_comp)

        # Add energy comparison
        _, ke_a = self.recorder_a.get_time_series("kinetic_energy")
        _, ke_b = self.recorder_b.get_time_series("kinetic_energy")
        if len(ke_a) > 0 and len(ke_b) > 0:
            metrics.append(
                self.compare_scalars(
                    "Max Kinetic Energy", float(np.max(ke_a)), float(np.max(ke_b))
                )
            )

        # Add angular momentum comparison
        _, am_a = self.recorder_a.get_time_series("angular_momentum")
        _, am_b = self.recorder_b.get_time_series("angular_momentum")
        if len(am_a) > 0 and len(am_b) > 0:
            am_a = np.asarray(am_a)
            am_b = np.asarray(am_b)
            # Compare max magnitude
            max_am_a = float(np.max(np.linalg.norm(am_a, axis=1)))
            max_am_b = float(np.max(np.linalg.norm(am_b, axis=1)))
            metrics.append(
                self.compare_scalars("Max Angular Momentum", max_am_a, max_am_b)
            )

        # Add CoP path length comparison
        _, cop_a = self.recorder_a.get_time_series("cop_position")
        _, cop_b = self.recorder_b.get_time_series("cop_position")
        if len(cop_a) > 0 and len(cop_b) > 0:
            cop_a = np.asarray(cop_a)
            cop_b = np.asarray(cop_b)

            def path_len(c: np.ndarray) -> float:
                """Calculate path length of a trajectory."""
                return float(np.sum(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1)))

            metrics.append(
                self.compare_scalars(
                    "CoP Path Length", path_len(cop_a), path_len(cop_b)
                )
            )

        report = {"swing_a": self.name_a, "swing_b": self.name_b, "metrics": metrics}
        return report

    def compute_dtw_distance(
        self,
        field_name: str,
        joint_idx: int | None = None,
        radius: int = 10,
    ) -> tuple[float, list[tuple[int, int]]]:
        """Compute Dynamic Time Warping (DTW) distance and alignment path.

        DTW measures similarity between two temporal sequences which may vary in speed.

        Args:
            field_name: Name of data field.
            joint_idx: Index if multidimensional.
            radius: Sakoe-Chiba band radius (constraint window).

        Returns:
            Tuple of (distance, path). Path is list of (i, j) indices.
        """
        # Get data
        _, data_a_raw = self.recorder_a.get_time_series(field_name)
        _, data_b_raw = self.recorder_b.get_time_series(field_name)

        data_a: np.ndarray = np.asarray(data_a_raw)
        data_b: np.ndarray = np.asarray(data_b_raw)

        # Handle dimensions
        if joint_idx is not None:
            if data_a.ndim > 1 and joint_idx < data_a.shape[1]:
                data_a = data_a[:, joint_idx]
            if data_b.ndim > 1 and joint_idx < data_b.shape[1]:
                data_b = data_b[:, joint_idx]

        # Simple Euclidean distance
        # Standardize for scale invariance?
        # Usually DTW is done on normalized data (z-score) to focus on shape.

        if np.std(data_a) > 1e-6:
            data_a = (data_a - np.mean(data_a)) / np.std(data_a)
        if np.std(data_b) > 1e-6:
            data_b = (data_b - np.mean(data_b)) / np.std(data_b)

        # Implementation via centralized signal processing module
        # Note: This uses Squared Euclidean distance (L2) whereas the previous
        # implementation used Absolute Difference (L1). L2 is standard for DTW.
        # This implementation is also potentially Numba-accelerated.
        return signal_processing.compute_dtw_path(data_a, data_b, window=radius)
