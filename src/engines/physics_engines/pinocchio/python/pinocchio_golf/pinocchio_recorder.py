"""Pinocchio simulation recorder and utility classes.

Extracted from gui.py to reduce monolith size.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
from PyQt6 import QtWidgets

from src.shared.python.biomechanics_data import BiomechanicalData


class LogPanel(QtWidgets.QTextEdit):
    """Log panel widget for displaying messages."""

    def __init__(self) -> None:
        """Initialize the log panel."""
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet(
            "background:#111; color:#0F0; font-family:Consolas; font-size:12px;"
        )


class SignalBlocker:
    """Context manager to block signals for a set of widgets."""

    def __init__(self, *widgets: QtWidgets.QWidget) -> None:
        """Initialize with widgets to block."""
        self.widgets = widgets

    def __enter__(self) -> None:
        """Block signals for all widgets."""
        for w in self.widgets:
            w.blockSignals(True)  # noqa: FBT003

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Restore signals for all widgets."""
        for w in self.widgets:
            w.blockSignals(False)  # noqa: FBT003


class PinocchioRecorder:
    """Records time-series data from Pinocchio simulation.

    Implements RecorderInterface for LivePlotWidget.
    """

    def __init__(self, engine: Any = None) -> None:
        """Initialize empty recorder."""
        self.reset()
        self.engine = engine  # Reference for joint names
        self.analysis_config: dict[str, Any] = {}

    def reset(self) -> None:
        """Clear all recorded data."""
        self.frames: list[BiomechanicalData] = []
        self.is_recording = False

    def start_recording(self) -> None:
        """Start recording data."""
        self.is_recording = True
        self.frames = []

    def stop_recording(self) -> None:
        """Stop recording data."""
        self.is_recording = False

    def get_num_frames(self) -> int:
        """Get number of recorded frames."""
        return len(self.frames)

    def record_frame(
        self,
        time: float,
        q: np.ndarray,
        v: np.ndarray,
        tau: np.ndarray | None = None,
        kinetic_energy: float = 0.0,
        potential_energy: float = 0.0,
        club_head_position: np.ndarray | None = None,
        club_head_velocity: np.ndarray | None = None,
        induced_accelerations: dict[str, np.ndarray] | None = None,
        counterfactuals: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Add a frame of data to the recording."""
        if self.is_recording:
            # Simple energy calculation if not provided (approximate)
            total_energy = kinetic_energy + potential_energy

            club_head_speed = 0.0
            if club_head_velocity is not None:
                club_head_speed = float(np.linalg.norm(club_head_velocity))

            frame = BiomechanicalData(
                time=float(time),
                joint_positions=q.copy(),
                joint_velocities=v.copy(),
                joint_torques=tau.copy() if tau is not None else np.zeros_like(v),
                kinetic_energy=kinetic_energy,
                potential_energy=potential_energy,
                total_energy=total_energy,
                club_head_position=club_head_position,
                club_head_velocity=club_head_velocity,
                club_head_speed=club_head_speed,
                induced_accelerations=induced_accelerations or {},
                counterfactuals=counterfactuals or {},
            )
            self.frames.append(frame)

    def set_analysis_config(self, config: dict[str, Any]) -> None:
        """Update analysis configuration."""
        self.analysis_config = config

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray | list]:
        """Extract time series for a specific field."""
        if not self.frames:
            return np.array([]), np.array([])

        times = np.array([f.time for f in self.frames])

        # Handle special counterfactual fields
        if field_name == "ztcf_accel":
            return self.get_counterfactual_series("ztcf_accel")
        if field_name == "zvcf_accel":
            return self.get_counterfactual_series("zvcf_torque")

        values = [getattr(f, field_name, None) for f in self.frames]

        # Handle None values
        if all(v is None for v in values):
            return times, np.array([])

        # Filter out None values
        valid_indices = [i for i, v in enumerate(values) if v is not None]
        if not valid_indices:
            return times, np.array([])

        times = times[valid_indices]
        values = [values[i] for i in valid_indices]

        # Stack into array
        try:
            values_array = np.array(values)
        except (ValueError, TypeError):
            return times, values

        return times, values_array

    def get_induced_acceleration_series(
        self, source_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract time series for a specific induced acceleration source."""
        if not self.frames:
            return np.array([]), np.array([])

        times = []
        values = []

        key = str(source_name)

        for f in self.frames:
            val = f.induced_accelerations.get(key)
            if val is None and isinstance(source_name, int):
                val = f.induced_accelerations.get(str(source_name))

            if val is not None:
                times.append(f.time)
                values.append(val)

        if not values:
            return np.array([]), np.array([])

        return np.array(times), np.array(values)

    def get_counterfactual_series(self, cf_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract time series for a specific counterfactual component."""
        if not self.frames:
            return np.array([]), np.array([])

        times = np.array([f.time for f in self.frames])

        valid_indices = [
            i
            for i, f in enumerate(self.frames)
            if hasattr(f, "counterfactuals") and cf_name in f.counterfactuals
        ]

        if not valid_indices:
            return np.array([]), np.array([])

        filtered_times = times[valid_indices]
        values = [self.frames[i].counterfactuals[cf_name] for i in valid_indices]

        return filtered_times, np.array(values)

    def export_to_dict(self) -> dict[str, Any]:
        """Export all recorded data to a dictionary."""
        if not self.frames:
            return {}

        export_data: dict[str, Any] = {}

        # Get basic time series
        times, positions = self.get_time_series("joint_positions")
        _, velocities = self.get_time_series("joint_velocities")
        _, torques = self.get_time_series("joint_torques")
        _, energies = self.get_time_series("total_energy")

        export_data["time"] = times
        export_data["joint_positions"] = positions
        export_data["joint_velocities"] = velocities
        export_data["joint_torques"] = torques
        export_data["total_energy"] = energies

        # Export Induced
        first_frame = self.frames[0]
        if first_frame.induced_accelerations:
            all_keys: set[str] = set()
            for f in self.frames:
                all_keys.update(f.induced_accelerations.keys())

            for key in all_keys:
                _, vals = self.get_induced_acceleration_series(key)
                if len(vals) > 0:
                    export_data[f"induced_{key}"] = vals

        # Export Counterfactuals
        if first_frame.counterfactuals:
            all_keys_cf: set[str] = set()
            for f in self.frames:
                all_keys_cf.update(f.counterfactuals.keys())

            for key in all_keys_cf:
                _, vals = self.get_counterfactual_series(key)
                if len(vals) > 0:
                    export_data[f"cf_{key}"] = vals

        return export_data
