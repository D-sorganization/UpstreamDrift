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
        self._buffers_initialized = False
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        """Initialize or reset data buffers.

        Note: Array dimensions are determined on first record_step() call
        when we have access to actual state dimensions.
        """
        self.current_idx = 0
        self._buffers_initialized = False
        self.data = {
            # Scalars (pre-allocated)
            "times": np.zeros(self.max_samples),
            "kinetic_energy": np.zeros(self.max_samples),
            "potential_energy": np.zeros(self.max_samples),
            "total_energy": np.zeros(self.max_samples),
            "club_head_speed": np.zeros(self.max_samples),
            # Arrays (initialized on first record)
            "joint_positions": None,
            "joint_velocities": None,
            "joint_accelerations": None,
            "joint_torques": None,
            "actuator_powers": None,
            "angular_momentum": None,
            "club_head_position": None,
            "cop_position": None,
            "com_position": None,
            "ground_forces": None,
            # Storage for computed analyses
            "induced_accelerations": {},  # Map source -> (times, accel)
            "counterfactuals": {},  # Map name -> (times, data)
        }

    def _initialize_array_buffers(self, q: np.ndarray, v: np.ndarray) -> None:
        """Initialize array buffers with proper dimensions on first record.

        Args:
            q: Position state vector
            v: Velocity state vector
        """
        nq = len(q)
        nv = len(v)

        self.data["joint_positions"] = np.zeros((self.max_samples, nq))
        self.data["joint_velocities"] = np.zeros((self.max_samples, nv))
        self.data["joint_accelerations"] = np.zeros((self.max_samples, nv))
        self.data["joint_torques"] = np.zeros((self.max_samples, nv))
        self.data["actuator_powers"] = np.zeros((self.max_samples, nv))
        self.data["angular_momentum"] = np.zeros((self.max_samples, 3))
        self.data["club_head_position"] = np.zeros((self.max_samples, 3))
        self.data["cop_position"] = np.zeros((self.max_samples, 3))
        self.data["com_position"] = np.zeros((self.max_samples, 3))
        self.data["ground_forces"] = np.zeros((self.max_samples, 3))

        self._buffers_initialized = True
        LOGGER.debug(
            f"Initialized recorder buffers: nq={nq}, nv={nv}, max_samples={self.max_samples}"
        )

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

        # Check buffer capacity
        if self.current_idx >= self.max_samples:
            LOGGER.warning(
                f"Recorder buffer full at {self.max_samples} samples. Stopping recording."
            )
            self.is_recording = False
            return

        # Get State
        q, v = self.engine.get_state()
        t = self.engine.get_time()

        # Initialize array buffers on first record
        if not self._buffers_initialized:
            self._initialize_array_buffers(q, v)

        # Handle control input
        if control_input is not None:
            tau = control_input
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

        # Store basic data using array indexing (no copy needed, direct assignment)
        idx = self.current_idx
        self.data["times"][idx] = t
        self.data["joint_positions"][idx] = q
        self.data["joint_velocities"][idx] = v
        self.data["kinetic_energy"][idx] = ke
        self.data["joint_torques"][idx] = tau

        self.current_idx += 1

    def update_control(self, u: np.ndarray) -> None:
        """Update the last applied control for recording."""
        if self.is_recording and self.current_idx > 0 and self._buffers_initialized:
            # Replace the last recorded control
            self.data["joint_torques"][self.current_idx - 1] = u

    # -------- RecorderInterface Implementation --------

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get time series data for plotting.

        Returns views of the recorded data up to current_idx (no copying).

        Args:
            field_name: Name of the field to retrieve

        Returns:
            Tuple of (times, values) as NumPy array views
        """
        if field_name not in self.data:
            return np.array([]), np.array([])

        values = self.data[field_name]

        # Handle None (uninitialized arrays)
        if values is None or self.current_idx == 0:
            return np.array([]), np.array([])

        # Return views (no copy) up to current_idx
        times = self.data["times"][: self.current_idx]

        # Handle different data types
        if isinstance(values, np.ndarray):
            return times, values[: self.current_idx]
        elif isinstance(values, list):
            # Legacy support for list-based data
            return times, np.array(values[: self.current_idx])
        else:
            # For dict/other types
            return times, values

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

        if not self._buffers_initialized or self.current_idx == 0:
            LOGGER.warning("No data recorded for post-hoc analysis")
            return

        # Use only the recorded portion (up to current_idx)
        n_frames = self.current_idx
        times = self.data["times"][:n_frames]
        qs = self.data["joint_positions"][:n_frames]
        vs = self.data["joint_velocities"][:n_frames]
        taus = self.data["joint_torques"][:n_frames]

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
        """Return the raw data dictionary for export.

        Returns only the recorded portion (up to current_idx) of arrays.
        """
        export_data = {}
        for k, v in self.data.items():
            if isinstance(v, np.ndarray):
                # Export only recorded portion
                export_data[k] = v[: self.current_idx] if v.ndim > 0 else v
            elif isinstance(v, list) and v:
                try:
                    export_data[k] = np.array(v)
                except Exception:
                    export_data[k] = v
            else:
                export_data[k] = v

        # Add metadata
        export_data["model_name"] = self.engine.model_name
        export_data["num_frames"] = self.current_idx
        return export_data
