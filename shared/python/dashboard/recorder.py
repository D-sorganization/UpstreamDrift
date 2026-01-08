"""Generic recorder for PhysicsEngine compatible simulations.

Records state, control, and derived quantities for analysis and plotting.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from shared.python.interfaces import PhysicsEngine

LOGGER = logging.getLogger(__name__)


class GenericPhysicsRecorder:
    """Records simulation data from a PhysicsEngine."""

    def __init__(self, engine: PhysicsEngine, max_samples: int = 100000) -> None:
        """Initialize recorder.

        Args:
            engine: The physics engine instance to record from.
            max_samples: Pre-allocation size for buffers.
        """
        self.engine = engine
        self.max_samples = max_samples
        self.current_idx = 0
        self.is_recording = False
        self.data: dict[str, Any] = {}
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        """Initialize or reset data buffers."""
        self.current_idx = 0
        self.data = {
            "times": [],
            "joint_positions": [],
            "joint_velocities": [],
            "joint_accelerations": [],
            "joint_torques": [],
            "actuator_powers": [],
            "kinetic_energy": [],
            "potential_energy": [],
            "total_energy": [],
            "angular_momentum": [],
            "club_head_position": [],
            "club_head_speed": [],
            "cop_position": [],
            "com_position": [],
            "ground_forces": [],
            # Storage for computed analyses
            "induced_accelerations": {},  # Map source -> (times, accel)
            "counterfactuals": {},  # Map name -> (times, data)
        }

    def start(self) -> None:
        """Start recording."""
        self.is_recording = True
        LOGGER.info("Recording started.")

    def stop(self) -> None:
        """Stop recording."""
        self.is_recording = False
        LOGGER.info("Recording stopped. Recorded %d frames.", self.current_idx)

    def reset(self) -> None:
        """Clear all recorded data."""
        self._reset_buffers()
        LOGGER.info("Recorder reset.")

    def record_step(self, control_input: np.ndarray | None = None) -> None:
        """Record the current state of the engine.

        Args:
            control_input: Optional control vector applied during this step.
        """
        if not self.is_recording:
            return

        # Get State
        q, v = self.engine.get_state()
        t = self.engine.get_time()

        # Handle control input
        if control_input is not None:
            tau = control_input.copy()
        else:
            # Try to get from engine if possible, or assume zero/unknown
            # PhysicsEngine doesn't enforce get_control, but we can check if it has a stored one
            # For now, default to zero if not provided
            tau = np.zeros(len(v))

        # Energies
        try:
            M = self.engine.compute_mass_matrix()
            if M.size > 0:
                ke = 0.5 * v.T @ M @ v
            else:
                ke = 0.0
        except Exception:
            ke = 0.0

        # Store basic data
        self.data["times"].append(t)
        self.data["joint_positions"].append(q.copy())
        self.data["joint_velocities"].append(v.copy())
        # Accel?
        # self.data['joint_accelerations'].append(...)
        self.data["kinetic_energy"].append(ke)
        self.data["joint_torques"].append(tau)

        self.current_idx += 1

    def update_control(self, u: np.ndarray) -> None:
        """Update the last applied control for recording."""
        if self.is_recording and self.data["joint_torques"]:
            # Replace the last placeholder
            self.data["joint_torques"][-1] = u.copy()

    # -------- RecorderInterface Implementation --------

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get time series data for plotting."""
        if field_name not in self.data:
            return np.array([]), np.array([])

        times = np.array(self.data["times"])
        values = self.data[field_name]

        if not values:
            return np.array([]), np.array([])

        return times, np.array(values)

    def get_induced_acceleration_series(
        self, source_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get induced acceleration series."""
        if source_name not in self.data["induced_accelerations"]:
            # Compute on demand if not cached?
            # Doing it post-hoc is expensive (needs re-simulation).
            # For now, return empty if not recorded.
            return np.array([]), np.array([])
        return self.data["induced_accelerations"][source_name]

    def get_counterfactual_series(self, cf_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get counterfactual series."""
        if cf_name not in self.data["counterfactuals"]:
            return np.array([]), np.array([])
        return self.data["counterfactuals"][cf_name]

    # -------- Analysis Computation --------

    def compute_analysis_post_hoc(self) -> None:
        """Compute expensive analysis (ZTCF, Induced Accel) after recording.

        Replays the trajectory (sets state) and computes metrics frame-by-frame.
        """
        LOGGER.info("Computing post-hoc analysis...")
        times = self.data["times"]
        qs = self.data["joint_positions"]
        vs = self.data["joint_velocities"]
        taus = self.data["joint_torques"]

        n_frames = len(times)
        if n_frames == 0:
            return

        ztcf_accels = []
        zvcf_accels = []
        drift_accels = []
        control_accels = []

        for i in range(n_frames):
            q = qs[i]
            v = vs[i]
            tau = taus[i]

            # Set state for computation (without advancing time)
            self.engine.set_state(q, v)
            # Apply recorded torque for consistent state?
            # self.engine.set_control(tau) # Important for ZVCF which uses current tau
            self.engine.set_control(tau)

            # Need to force update
            self.engine.forward()

            # ZTCF
            ztcf = self.engine.compute_ztcf(q, v)
            ztcf_accels.append(ztcf)

            # ZVCF
            zvcf = self.engine.compute_zvcf(q)
            zvcf_accels.append(zvcf)

            # Induced Accel breakdown
            drift = self.engine.compute_drift_acceleration()
            drift_accels.append(drift)

            ctrl_acc = self.engine.compute_control_acceleration(tau)
            control_accels.append(ctrl_acc)

        # Store results
        times_arr = np.array(times)
        self.data["counterfactuals"]["ztcf_accel"] = (times_arr, np.array(ztcf_accels))
        # ZVCF is usually accel, but can be interpreted as torque capacity? No, it's accel.
        self.data["counterfactuals"]["zvcf_accel"] = (times_arr, np.array(zvcf_accels))
        # Also map generic 'ztcf'/'zvcf' for Plotter
        self.data["counterfactuals"]["ztcf"] = (times_arr, np.array(ztcf_accels))
        self.data["counterfactuals"]["zvcf"] = (times_arr, np.array(zvcf_accels))

        self.data["induced_accelerations"]["gravity"] = (
            times_arr,
            np.array(drift_accels),
        )  # Approx label
        self.data["induced_accelerations"]["drift"] = (
            times_arr,
            np.array(drift_accels),
        )
        self.data["induced_accelerations"]["control"] = (
            times_arr,
            np.array(control_accels),
        )
        self.data["induced_accelerations"]["total"] = (
            times_arr,
            np.array(drift_accels) + np.array(control_accels),
        )

        LOGGER.info("Post-hoc analysis complete.")

    def get_data_dict(self) -> dict[str, Any]:
        """Return the raw data dictionary for export."""
        # Convert lists to arrays for export efficiency
        export_data = {}
        for k, v in self.data.items():
            if isinstance(v, list) and v:
                try:
                    export_data[k] = np.array(v)
                except Exception:
                    export_data[k] = v
            else:
                export_data[k] = v

        # Add metadata
        export_data["model_name"] = self.engine.model_name
        return export_data
