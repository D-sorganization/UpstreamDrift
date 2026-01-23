"""Generic recorder for PhysicsEngine compatible simulations.

Records state, control, and derived quantities for analysis and plotting.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from shared.python.interfaces import PhysicsEngine

LOGGER = logging.getLogger(__name__)


class GenericPhysicsRecorder:
    """Records simulation data from a PhysicsEngine.

    PERFORMANCE FIX: Uses dynamic buffer sizing with growth factor
    to avoid over-allocation for short recordings.
    """

    def __init__(
        self,
        engine: PhysicsEngine,
        max_samples: int = 100000,
        initial_capacity: int = 1000,
    ) -> None:
        """Initialize recorder.

        Args:
            engine: The physics engine instance to record from.
            max_samples: Maximum allocation size for buffers.
            initial_capacity: Initial buffer size (grows dynamically).
        """
        self.engine = engine
        self.max_samples = max_samples
        # PERFORMANCE FIX: Start with smaller initial capacity
        self.initial_capacity = initial_capacity
        self.current_capacity = initial_capacity
        self.growth_factor = 1.5  # Grow by 50% when needed
        self.current_idx = 0
        self.is_recording = False
        self.data: dict[str, Any] = {}
        self._buffers_initialized = False

        # Real-time analysis configuration
        self.analysis_config = {
            "ztcf": False,
            "zvcf": False,
            "track_drift": False,
            "track_total_control": False,
            "induced_accel_sources": [],  # List of int indices to track individually
        }

        self._reset_buffers()

    def _ensure_capacity(self) -> None:
        """Ensure buffer has capacity for next sample.

        PERFORMANCE FIX: Dynamically grow buffers when needed.
        """
        if self.current_idx >= self.current_capacity:
            # Calculate new capacity
            new_capacity = min(
                int(self.current_capacity * self.growth_factor), self.max_samples
            )

            if new_capacity <= self.current_capacity:
                # Hit max_samples limit
                LOGGER.warning(
                    f"Recorder buffer full at {self.max_samples} samples. "
                    "Stopping recording."
                )
                self.is_recording = False
                return

            LOGGER.debug(
                f"Growing recorder buffers from {self.current_capacity} "
                f"to {new_capacity} samples"
            )

            # Resize all buffers
            for key, arr in self.data.items():
                if isinstance(arr, np.ndarray):
                    # Create new larger array
                    new_shape = list(arr.shape)
                    new_shape[0] = new_capacity
                    new_arr = np.zeros(new_shape, dtype=arr.dtype)
                    # Copy existing data
                    new_arr[: self.current_capacity] = arr
                    self.data[key] = new_arr

            self.current_capacity = new_capacity

    def set_analysis_config(self, config: dict[str, Any]) -> None:
        """Update analysis configuration."""
        self.analysis_config.update(config)
        LOGGER.info(f"Recorder analysis config updated: {self.analysis_config}")

        # If already recording or initialized, ensure buffers exist for new config
        if self._buffers_initialized:
            self._ensure_buffers_allocated()

    def _ensure_buffers_allocated(self) -> None:
        """Allocate buffers for enabled analysis features if missing."""
        # Use existing joint velocity buffer to determine dimensions
        if self.data["joint_velocities"] is None:
            return  # Cannot allocate without knowing dimensions

        nv = self.data["joint_velocities"].shape[1]

        # Allocate ZTCF if missing
        if self.analysis_config["ztcf"] and self.data["ztcf_accel"] is None:
            self.data["ztcf_accel"] = np.zeros((self.max_samples, nv))
            LOGGER.debug("Allocated ZTCF buffer dynamically.")

        # Allocate ZVCF if missing
        if self.analysis_config["zvcf"] and self.data["zvcf_accel"] is None:
            self.data["zvcf_accel"] = np.zeros((self.max_samples, nv))
            LOGGER.debug("Allocated ZVCF buffer dynamically.")

        # Allocate Drift if missing
        if self.analysis_config["track_drift"] and self.data["drift_accel"] is None:
            self.data["drift_accel"] = np.zeros((self.max_samples, nv))
            LOGGER.debug("Allocated Drift buffer dynamically.")

        # Allocate Control if missing
        if (
            self.analysis_config["track_total_control"]
            and self.data["control_accel"] is None
        ):
            self.data["control_accel"] = np.zeros((self.max_samples, nv))
            LOGGER.debug("Allocated Control buffer dynamically.")

        # Allocate Induced Accel sources if missing
        sources = cast(list[int], self.analysis_config["induced_accel_sources"])
        for idx in sources:
            if idx not in self.data["induced_accelerations"]:
                self.data["induced_accelerations"][idx] = np.zeros(
                    (self.max_samples, nv)
                )
                LOGGER.debug(f"Allocated Induced Accel buffer for source {idx}.")

    def _reset_buffers(self) -> None:
        """Initialize or reset data buffers.

        Note: Array dimensions are determined on first record_step() call
        when we have access to actual state dimensions.

        PERFORMANCE FIX: Uses initial_capacity instead of max_samples for allocation.
        """
        self.current_idx = 0
        self.current_capacity = self.initial_capacity
        self._buffers_initialized = False
        self.data = {
            # Scalars (pre-allocated with initial capacity)
            "times": np.zeros(self.current_capacity),
            "kinetic_energy": np.zeros(self.current_capacity),
            "potential_energy": np.zeros(self.current_capacity),
            "total_energy": np.zeros(self.current_capacity),
            "club_head_speed": np.zeros(self.current_capacity),
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
            # Storage for computed analyses (Real-time or Post-hoc)
            "ztcf_accel": None,
            "zvcf_accel": None,
            "drift_accel": None,
            "control_accel": None,
            "induced_accelerations": {},  # Map source_idx -> ndarray
            # Legacy/Post-hoc storage
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

        # PERFORMANCE FIX: Use current_capacity instead of max_samples
        self.data["joint_positions"] = np.zeros((self.current_capacity, nq))
        self.data["joint_velocities"] = np.zeros((self.current_capacity, nv))
        self.data["joint_accelerations"] = np.zeros((self.current_capacity, nv))
        self.data["joint_torques"] = np.zeros((self.current_capacity, nv))
        self.data["actuator_powers"] = np.zeros((self.current_capacity, nv))
        self.data["angular_momentum"] = np.zeros((self.current_capacity, 3))
        self.data["club_head_position"] = np.zeros((self.current_capacity, 3))
        self.data["cop_position"] = np.zeros((self.current_capacity, 3))
        self.data["com_position"] = np.zeros((self.max_samples, 3))
        self.data["ground_forces"] = np.zeros((self.max_samples, 3))

        # Real-time analysis buffers
        if self.analysis_config["ztcf"]:
            self.data["ztcf_accel"] = np.zeros((self.max_samples, nv))
        if self.analysis_config["zvcf"]:
            self.data["zvcf_accel"] = np.zeros((self.max_samples, nv))
        if self.analysis_config["track_drift"]:
            self.data["drift_accel"] = np.zeros((self.max_samples, nv))
        if self.analysis_config["track_total_control"]:
            self.data["control_accel"] = np.zeros((self.max_samples, nv))

        # Individual induced accelerations
        sources = cast(list[int], self.analysis_config["induced_accel_sources"])
        for idx in sources:
            self.data["induced_accelerations"][idx] = np.zeros((self.max_samples, nv))

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

        # PERFORMANCE FIX: Ensure buffer has capacity (dynamic growth)
        self._ensure_capacity()

        if not self.is_recording:
            # _ensure_capacity may have stopped recording if max reached
            return

        # PERFORMANCE FIX: Batch state retrieval (was 3 separate calls)
        # get_full_state() returns q, v, t, and M in a single call
        full_state = self.engine.get_full_state()
        q = full_state["q"]
        v = full_state["v"]
        t = full_state["t"]
        M = full_state.get("M")  # May be None for some engines

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

        # Energies - use pre-computed M from get_full_state()
        if M is not None and M.size > 0:
            try:
                ke = 0.5 * v.T @ M @ v
            except Exception as e:
                LOGGER.warning("Failed to compute kinetic energy: %s", e)
                ke = 0.0
        else:
            ke = 0.0

        # Real-time Analysis Computations
        idx = self.current_idx

        # ZTCF
        if self.analysis_config["ztcf"] and self.data["ztcf_accel"] is not None:
            try:
                self.data["ztcf_accel"][idx] = self.engine.compute_ztcf(q, v)
            except Exception as e:
                LOGGER.warning("Failed to compute ZTCF at frame %d: %s", idx, e)

        # ZVCF
        if self.analysis_config["zvcf"] and self.data["zvcf_accel"] is not None:
            try:
                self.data["zvcf_accel"][idx] = self.engine.compute_zvcf(q)
            except Exception as e:
                LOGGER.warning("Failed to compute ZVCF at frame %d: %s", idx, e)

        # Drift Accel
        if self.analysis_config["track_drift"] and self.data["drift_accel"] is not None:
            try:
                self.data["drift_accel"][idx] = self.engine.compute_drift_acceleration()
            except Exception as e:
                LOGGER.warning(
                    "Failed to compute drift acceleration at frame %d: %s", idx, e
                )

        # Total Control Accel
        if (
            self.analysis_config["track_total_control"]
            and self.data["control_accel"] is not None
        ):
            try:
                self.data["control_accel"][idx] = (
                    self.engine.compute_control_acceleration(tau)
                )
            except Exception as e:
                LOGGER.warning(
                    "Failed to compute control acceleration at frame %d: %s", idx, e
                )

        # PERFORMANCE FIX: Vectorized induced accelerations
        # Instead of calling compute_control_acceleration N times (each solving M\tau),
        # we compute M_inv once and use: accel[src] = M_inv[:, src] * tau[src]
        sources = cast(list[int], self.analysis_config["induced_accel_sources"])
        if sources and M is not None and M.size > 0:
            try:
                # Compute M_inv once for all sources
                M_inv = np.linalg.inv(M)
                for src_idx in sources:
                    if src_idx in self.data["induced_accelerations"]:
                        # Vectorized: M^{-1} * e_src * tau[src] = M_inv[:, src] * tau[src]
                        self.data["induced_accelerations"][src_idx][idx] = (
                            M_inv[:, src_idx] * tau[src_idx]
                        )
            except Exception as e:
                LOGGER.warning(
                    "Failed to compute induced accelerations at frame %d: %s", idx, e
                )
        elif sources:
            # Fallback to original method if M not available
            for src_idx in sources:
                if src_idx in self.data["induced_accelerations"]:
                    try:
                        tau_single = np.zeros_like(tau)
                        tau_single[src_idx] = tau[src_idx]
                        self.data["induced_accelerations"][src_idx][idx] = (
                            self.engine.compute_control_acceleration(tau_single)
                        )
                    except Exception as e:
                        LOGGER.warning(
                            "Failed to compute induced acceleration for source %d: %s",
                            src_idx,
                            e,
                        )

        # Store basic data using array indexing (no copy needed, direct assignment)
        self.data["times"][idx] = t
        self.data["joint_positions"][idx] = q
        self.data["joint_velocities"][idx] = v
        self.data["kinetic_energy"][idx] = ke
        self.data["joint_torques"][idx] = tau

        # Ground Forces (from engine, if supported)
        try:
            grf = self.engine.compute_contact_forces()
            if grf is not None and len(grf) == 3:
                self.data["ground_forces"][idx] = grf
        except Exception as e:
            LOGGER.warning("Failed to compute ground forces at frame %d: %s", idx, e)

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

        values: Any = self.data[field_name]

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
        self, source_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get induced acceleration series."""
        if source_name not in self.data["induced_accelerations"]:
            # Log when parsing/lookup fails for induced acceleration source
            LOGGER.warning(
                "Induced acceleration source '%s' not found in recorded data. "
                "Available sources: %s. Returning empty series.",
                source_name,
                list(self.data["induced_accelerations"].keys()),
            )
            return np.array([]), np.array([])

        # Get data
        data = self.data["induced_accelerations"][source_name]

        # If it's a raw array (real-time buffer), return sliced view
        if isinstance(data, np.ndarray):
            return self.data["times"][: self.current_idx], data[: self.current_idx]

        # If it's already a tuple (post-hoc result), return it
        # Explicitly cast to tuple to satisfy MyPy
        result: tuple[np.ndarray, np.ndarray] = data
        return result

    def get_counterfactual_series(self, cf_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get counterfactual series."""
        if cf_name not in self.data["counterfactuals"]:
            return np.array([]), np.array([])
        # Explicitly cast to tuple to satisfy MyPy
        result: tuple[np.ndarray, np.ndarray] = self.data["counterfactuals"][cf_name]
        return result

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
        export_data: dict[str, Any] = {}
        for k, v in self.data.items():
            if isinstance(v, np.ndarray):
                # Export only recorded portion
                export_data[k] = v[: self.current_idx] if v.ndim > 0 else v
            elif isinstance(v, list) and v:
                try:
                    export_data[k] = np.array(v)
                except Exception as e:
                    LOGGER.debug("Failed to convert list '%s' to numpy array: %s", k, e)
                    export_data[k] = v
            else:
                export_data[k] = v

        # Add metadata
        export_data["model_name"] = self.engine.model_name
        export_data["num_frames"] = self.current_idx
        return export_data
