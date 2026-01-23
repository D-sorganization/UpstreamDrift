"""Simulation telemetry utilities for MuJoCo golf swing experiments.

Provides classes to sample joint states, actuator torques, and interaction forces
for every simulation step. Recorded data can be aggregated into summary reports
that describe peak loads and overall simulation coverage, enabling downstream
parameter optimization and comparison of swing configurations.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class SimulationSample:
    """Container for per-step telemetry values."""

    time: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    controls: np.ndarray
    actuator_torques: dict[str, float]
    constraint_torques: dict[str, float]
    body_forces: dict[str, np.ndarray]
    custom_metrics: dict[str, float] = None  # type: ignore[assignment]


@dataclass
class TelemetryReport:
    """Aggregated view of a simulation run."""

    sample_count: int
    duration_seconds: float
    peak_actuator_torques: dict[str, float]
    peak_constraint_torques: dict[str, float]
    peak_body_forces: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        """Convert report to a serializable dictionary."""

        return {
            "sample_count": self.sample_count,
            "duration_seconds": self.duration_seconds,
            "peak_actuator_torques": self.peak_actuator_torques,
            "peak_constraint_torques": self.peak_constraint_torques,
            "peak_body_forces": self.peak_body_forces,
        }


class TelemetryRecorder:
    """Record telemetry for MuJoCo simulations.

    The recorder captures per-step state, actuator torques, and interaction
    forces (both joint constraints and external body contacts). Each set of
    samples can be converted into a :class:`TelemetryReport` for optimization
    pipelines or stored for post-processing.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        """Initialize the telemetry recorder."""
        self.model = model
        self.samples: list[SimulationSample] = []
        self._current_custom_metrics: dict[str, float] = {}

        # Pre-compute mappings for actuator and joint names to DOF indices.
        self._actuator_dof_map = self._build_actuator_dof_map(model)
        self._joint_names = self._build_joint_name_map(model)
        self._body_names = self._build_body_name_map(model)

    @staticmethod
    def _build_body_name_map(model: mujoco.MjModel) -> list[str]:
        """Build a list of body names from the model."""
        body_names: list[str] = []
        for idx in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, idx)
            body_names.append(name or f"body_{idx}")
        return body_names

    @staticmethod
    def _build_joint_name_map(model: mujoco.MjModel) -> list[str]:
        """Build a list of joint names from the model."""
        joint_names: list[str] = []
        for idx in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, idx)
            joint_names.append(name or f"joint_{idx}")
        return joint_names

    @staticmethod
    def _build_actuator_dof_map(model: mujoco.MjModel) -> dict[int, int]:
        """Build a map from actuator ID to DOF index."""
        actuator_to_dof: dict[int, int] = {}
        joint_transmission_types = {mujoco.mjtTrn.mjTRN_JOINT}
        if hasattr(mujoco.mjtTrn, "mjTRN_JOINTINP"):
            joint_transmission_types.add(mujoco.mjtTrn.mjTRN_JOINTINP)

        for actuator_id in range(model.nu):
            if not TelemetryRecorder._is_joint_target(
                model,
                actuator_id,
                joint_transmission_types,
            ):
                continue

            joint_id = model.actuator_trnid[actuator_id, 0]
            dof_index = model.jnt_dofadr[joint_id]
            if dof_index >= model.nv:
                continue

            actuator_to_dof[actuator_id] = dof_index
        return actuator_to_dof

    @staticmethod
    def _is_joint_target(
        model: mujoco.MjModel,
        actuator_id: int,
        joint_transmission_types: set[int],
    ) -> bool:
        """Check if actuator targets a joint."""
        transmission_type = model.actuator_trntype[actuator_id]
        if transmission_type not in joint_transmission_types:
            return False

        # `actuator_trnid` is only meaningful for joint-backed actuators and
        # is stored as a 2-column array. Guard against corrupted or partially
        # defined entries that can appear in conflicted merges by validating
        # both shape and bounds.
        if model.actuator_trnid.shape[1] == 0:
            return False

        joint_id = model.actuator_trnid[actuator_id, 0]
        return not (joint_id == -1 or joint_id >= model.njnt)

    def reset(self) -> None:
        """Clear captured samples while keeping mappings."""

        self.samples.clear()
        self._current_custom_metrics.clear()

    def add_custom_metric(self, name: str, value: float) -> None:
        """Add a custom metric to be recorded in the next step.

        Args:
            name: Metric identifier
            value: Scalar value
        """
        self._current_custom_metrics[name] = float(value)

    def record_step(self, data: mujoco.MjData) -> None:
        """Capture telemetry for the current simulation state."""

        actuator_torques = self._extract_actuator_torques(data)
        constraint_torques = self._extract_constraint_torques(data)
        body_forces = self._extract_body_forces(data)

        sample = SimulationSample(
            time=float(data.time),
            joint_positions=data.qpos.copy(),
            joint_velocities=data.qvel.copy(),
            controls=data.ctrl.copy(),
            actuator_torques=actuator_torques,
            constraint_torques=constraint_torques,
            body_forces=body_forces,
            custom_metrics=self._current_custom_metrics.copy(),
        )
        self.samples.append(sample)

    def generate_report(self) -> TelemetryReport:
        """Summarize captured telemetry into a report."""

        peak_actuator_torques = self._aggregate_peak_values(
            [sample.actuator_torques for sample in self.samples],
        )
        peak_constraint_torques = self._aggregate_peak_values(
            [sample.constraint_torques for sample in self.samples],
        )
        peak_body_forces = self._aggregate_peak_force_norms(
            [sample.body_forces for sample in self.samples],
        )

        duration = self.samples[-1].time - self.samples[0].time if self.samples else 0.0

        return TelemetryReport(
            sample_count=len(self.samples),
            duration_seconds=duration,
            peak_actuator_torques=peak_actuator_torques,
            peak_constraint_torques=peak_constraint_torques,
            peak_body_forces=peak_body_forces,
        )

    def _extract_actuator_torques(self, data: mujoco.MjData) -> dict[str, float]:
        """Docstring for _extract_actuator_torques."""
        actuator_torques: dict[str, float] = {}
        for actuator_id, dof_index in self._actuator_dof_map.items():
            torque_value = float(data.qfrc_actuator[dof_index])
            actuator_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_id,
            )
            if actuator_name is None:
                actuator_name = f"actuator_{actuator_id}"
            actuator_torques[actuator_name] = torque_value
        return actuator_torques

    def _extract_constraint_torques(self, data: mujoco.MjData) -> dict[str, float]:
        """Docstring for _extract_constraint_torques."""
        constraint_torques: dict[str, float] = {}
        for joint_id, joint_name in enumerate(self._joint_names):
            dof_index = self.model.jnt_dofadr[joint_id]
            constraint_torques[joint_name] = float(data.qfrc_constraint[dof_index])
        return constraint_torques

    def _extract_body_forces(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        """Docstring for _extract_body_forces."""
        forces: dict[str, np.ndarray] = {}
        reshaped = data.cfrc_ext.reshape(-1, 6)
        for idx, name in enumerate(self._body_names):
            force_vector = reshaped[idx]
            if np.linalg.norm(force_vector) > 0.0:
                forces[name] = force_vector.copy()
        return forces

    @staticmethod
    def _aggregate_peak_values(series: Sequence[dict[str, float]]) -> dict[str, float]:
        """Docstring for _aggregate_peak_values."""
        peaks: dict[str, float] = {}
        for entry in series:
            for key, value in entry.items():
                current_peak = peaks.get(key, 0.0)
                candidate = float(abs(value))
                peaks[key] = max(current_peak, candidate)
        return peaks

    @staticmethod
    def _aggregate_peak_force_norms(
        series: Sequence[dict[str, np.ndarray]],
    ) -> dict[str, float]:
        """Docstring for _aggregate_peak_force_norms."""
        peaks: dict[str, float] = {}
        for entry in series:
            for key, value in entry.items():
                force_norm = float(np.linalg.norm(value))
                current_peak = peaks.get(key, 0.0)
                peaks[key] = max(current_peak, force_norm)
        return peaks


def export_telemetry_json(filename: str, data_dict: dict[str, Any]) -> bool:
    """Export telemetry data to JSON."""
    try:
        # Convert numpy arrays to lists for JSON serialization
        serializable_dict = {}
        for key, value in data_dict.items():
            if hasattr(value, "tolist"):
                serializable_dict[key] = value.tolist()
            else:
                serializable_dict[key] = value

        with open(filename, "w") as f:
            json.dump(serializable_dict, f, indent=2)
        return True
    except Exception:
        return False


def export_telemetry_csv(filename: str, data_dict: dict[str, Any]) -> bool:
    """Export telemetry data to CSV."""
    try:
        # Filter for array-like data
        array_data = {}
        max_len = 0

        for key, value in data_dict.items():
            if isinstance(value, list | np.ndarray):
                array_data[key] = value
                max_len = max(max_len, len(value))

        if not array_data:
            return False

        keys = list(array_data.keys())

        # Flatten multi-dimensional arrays into columns
        flat_data = {}
        for key in keys:
            val_array = np.array(array_data[key])
            if val_array.ndim == 1:
                flat_data[key] = val_array
            elif val_array.ndim == 2:
                # 2D array: column for each dimension
                for dim in range(val_array.shape[1]):
                    flat_data[f"{key}_{dim}"] = val_array[:, dim]
            # Skip >2D arrays for CSV

        flat_keys = list(flat_data.keys())

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(flat_keys)

            for i in range(max_len):
                row = []
                for key in flat_keys:
                    val_list = flat_data[key]
                    if i < len(val_list):
                        row.append(val_list[i])
                    else:
                        row.append("")
                writer.writerow(row)
        return True
    except Exception:
        return False
