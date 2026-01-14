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
    """Records simulation data from a PhysicsEngine."""

    # PERFORMANCE: Dynamic buffer sizing constants
    _INITIAL_BUFFER_SIZE = 1000  # Start small to reduce memory footprint
    _BUFFER_GROWTH_FACTOR = 2  # Double buffer size when full
    _MAX_BUFFER_SIZE = 1000000  # Hard cap to prevent runaway memory usage

    def __init__(
        self,
        engine: PhysicsEngine,
        max_samples: int = 100000,
        dynamic_sizing: bool = True,
    ) -> None:
        """Initialize recorder.

        Args:
            engine: The physics engine instance to record from.
            max_samples: Maximum buffer size (hard cap).
            dynamic_sizing: If True, start with small buffers and grow as needed.
                          If False, pre-allocate max_samples (legacy behavior).
        """
        self.engine = engine
        self.max_samples = min(max_samples, self._MAX_BUFFER_SIZE)
        self._dynamic_sizing = dynamic_sizing
        # Current allocated size (may be less than max_samples if dynamic_sizing)
        # Use the smaller of INITIAL_BUFFER_SIZE or max_samples to respect test fixtures
        if dynamic_sizing:
            self._allocated_samples = min(self._INITIAL_BUFFER_SIZE, self.max_samples)
        else:
            self._allocated_samples = self.max_samples
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

        # Performance optimization: Mass matrix caching
        self._cached_mass_matrix: np.ndarray | None = None
        self._cached_mass_matrix_q: np.ndarray | None = None
        self._mass_matrix_cache_tolerance = (
            1e-6  # Re-compute if q changes significantly
        )

        # Performance optimization: Analysis computation frequency
        self._analysis_compute_interval = 1  # Compute every N frames (1 = every frame)

        self._reset_buffers()

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

        PERFORMANCE: Uses dynamic sizing - starts with _allocated_samples
        and grows as needed (see _grow_buffers).

        Note: Array dimensions are determined on first record_step() call
        when we have access to actual state dimensions.
        """
        self.current_idx = 0
        self._buffers_initialized = False
        # Reset allocated size if dynamic sizing
        if self._dynamic_sizing:
            self._allocated_samples = min(self._INITIAL_BUFFER_SIZE, self.max_samples)
        self.data = {
            # Scalars (pre-allocated with current allocated size)
            "times": np.zeros(self._allocated_samples),
            "kinetic_energy": np.zeros(self._allocated_samples),
            "potential_energy": np.zeros(self._allocated_samples),
            "total_energy": np.zeros(self._allocated_samples),
            "club_head_speed": np.zeros(self._allocated_samples),
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

    def _grow_buffers(self) -> bool:
        """Double the buffer size when current allocation is full.

        PERFORMANCE: Dynamic buffer growth reduces initial memory footprint
        while allowing recording to continue without pre-allocating worst-case.

        Returns:
            True if buffers were grown, False if at max capacity.
        """
        if self._allocated_samples >= self.max_samples:
            return False  # At maximum capacity

        new_size = min(
            self._allocated_samples * self._BUFFER_GROWTH_FACTOR, self.max_samples
        )
        
        # Don't grow if we're already at max_samples
        if new_size == self._allocated_samples:
            return False
            
        LOGGER.debug(
            f"Growing recorder buffers: {self._allocated_samples} -> {new_size}"
        )

        # Grow all allocated buffers
        for key, arr in self.data.items():
            if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                if arr.ndim == 1:
                    new_arr = np.zeros(new_size, dtype=arr.dtype)
                    new_arr[: self._allocated_samples] = arr
                else:
                    new_shape = (new_size,) + arr.shape[1:]
                    new_arr = np.zeros(new_shape, dtype=arr.dtype)
                    new_arr[: self._allocated_samples] = arr
                self.data[key] = new_arr
            elif isinstance(arr, dict):
                # Handle induced_accelerations dict
                for src_idx, src_arr in arr.items():
                    if isinstance(src_arr, np.ndarray):
                        new_shape = (new_size,) + src_arr.shape[1:]
                        new_src_arr = np.zeros(new_shape, dtype=src_arr.dtype)
                        new_src_arr[: self._allocated_samples] = src_arr
                        arr[src_idx] = new_src_arr

        self._allocated_samples = new_size
        return True

    def _initialize_array_buffers(self, q: np.ndarray, v: np.ndarray) -> None:
        """Initialize array buffers with proper dimensions on first record.

        PERFORMANCE: Uses _allocated_samples for dynamic sizing.

        Args:
            q: Position state vector
            v: Velocity state vector
        """
        nq = len(q)
        nv = len(v)

        self.data["joint_positions"] = np.zeros((self._allocated_samples, nq))
        self.data["joint_velocities"] = np.zeros((self._allocated_samples, nv))
        self.data["joint_accelerations"] = np.zeros((self._allocated_samples, nv))
        self.data["joint_torques"] = np.zeros((self._allocated_samples, nv))
        self.data["actuator_powers"] = np.zeros((self._allocated_samples, nv))
        self.data["angular_momentum"] = np.zeros((self._allocated_samples, 3))
        self.data["club_head_position"] = np.zeros((self._allocated_samples, 3))
        self.data["cop_position"] = np.zeros((self._allocated_samples, 3))
        self.data["com_position"] = np.zeros((self._allocated_samples, 3))
        self.data["ground_forces"] = np.zeros((self._allocated_samples, 3))

        # Real-time analysis buffers (also use _allocated_samples for dynamic sizing)
        if self.analysis_config["ztcf"]:
            self.data["ztcf_accel"] = np.zeros((self._allocated_samples, nv))
        if self.analysis_config["zvcf"]:
            self.data["zvcf_accel"] = np.zeros((self._allocated_samples, nv))
        if self.analysis_config["track_drift"]:
            self.data["drift_accel"] = np.zeros((self._allocated_samples, nv))
        if self.analysis_config["track_total_control"]:
            self.data["control_accel"] = np.zeros((self._allocated_samples, nv))

        # Individual induced accelerations
        sources = cast(list[int], self.analysis_config["induced_accel_sources"])
        for idx in sources:
            self.data["induced_accelerations"][idx] = np.zeros(
                (self._allocated_samples, nv)
            )

        self._buffers_initialized = True
        LOGGER.debug(
            f"Initialized recorder buffers: nq={nq}, nv={nv}, "
            f"allocated={self._allocated_samples}, max={self.max_samples}"
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

    def _get_cached_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Get mass matrix with caching to avoid redundant O(n³) computations.

        The mass matrix M(q) only depends on configuration q, not velocity.
        We cache it and only recompute when q changes significantly.

        Args:
            q: Current configuration vector

        Returns:
            Mass matrix (n_v, n_v)
        """
        need_recompute = (
            self._cached_mass_matrix is None
            or self._cached_mass_matrix_q is None
            or len(q) != len(self._cached_mass_matrix_q)
            or np.max(np.abs(q - self._cached_mass_matrix_q))
            > self._mass_matrix_cache_tolerance
        )

        if need_recompute:
            self._cached_mass_matrix = self.engine.compute_mass_matrix()
            self._cached_mass_matrix_q = q.copy()

        return self._cached_mass_matrix  # type: ignore[return-value]

    def record_step(self, control_input: np.ndarray | None = None) -> None:
        """Record the current state of the engine.

        Args:
            control_input: Optional control vector applied during this step.
        """
        if not self.is_recording:
            return

        # PERFORMANCE: Dynamic buffer growth - try to grow if at capacity
        if self.current_idx >= self._allocated_samples:
            if self._dynamic_sizing and self._grow_buffers():
                LOGGER.debug(
                    f"Grew buffers to {self._allocated_samples} samples at frame {self.current_idx}"
                )
            else:
                # Can't grow anymore - at max capacity
                LOGGER.warning(
                    f"Recorder buffer full at {self.max_samples} samples. Stopping recording."
                )
                self.is_recording = False
                return

        # PERFORMANCE: Use batched state query if available
        # This reduces 3 separate engine calls to 1
        if hasattr(self.engine, "get_full_state"):
            state = self.engine.get_full_state()
            q = state["q"]
            v = state["v"]
            t = state["t"]
        else:
            # Fallback for engines without batched query
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

        # PERFORMANCE: Use cached mass matrix instead of computing every frame
        try:
            M = self._get_cached_mass_matrix(q)
            if M.size > 0:
                ke = 0.5 * v.T @ M @ v
            else:
                ke = 0.0
        except Exception as e:
            LOGGER.debug("Failed to compute kinetic energy: %s", e)
            ke = 0.0

        # Real-time Analysis Computations
        idx = self.current_idx

        # ZTCF
        if self.analysis_config["ztcf"] and self.data["ztcf_accel"] is not None:
            try:
                self.data["ztcf_accel"][idx] = self.engine.compute_ztcf(q, v)
            except Exception as e:
                LOGGER.debug("Failed to compute ZTCF at frame %d: %s", idx, e)

        # ZVCF
        if self.analysis_config["zvcf"] and self.data["zvcf_accel"] is not None:
            try:
                self.data["zvcf_accel"][idx] = self.engine.compute_zvcf(q)
            except Exception as e:
                LOGGER.debug("Failed to compute ZVCF at frame %d: %s", idx, e)

        # Drift Accel
        if self.analysis_config["track_drift"] and self.data["drift_accel"] is not None:
            try:
                self.data["drift_accel"][idx] = self.engine.compute_drift_acceleration()
            except Exception as e:
                LOGGER.debug(
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
                LOGGER.debug(
                    "Failed to compute control acceleration at frame %d: %s", idx, e
                )

        # PERFORMANCE: Batched Individual Induced Accelerations
        # Instead of N separate engine calls, compute M^{-1} once and use matrix ops
        sources = cast(list[int], self.analysis_config["induced_accel_sources"])
        if sources:
            try:
                # Get cached mass matrix and compute its inverse once
                M = self._get_cached_mass_matrix(q)
                if M.size > 0:
                    # Use solve instead of explicit inverse for numerical stability
                    # For each source, induced_accel = M^{-1} * tau_single
                    # tau_single has only one non-zero element, so this is just
                    # scaling the corresponding column of M^{-1}
                    for src_idx in sources:
                        if src_idx in self.data["induced_accelerations"]:
                            # Create single-source torque vector
                            tau_single = np.zeros_like(tau)
                            tau_single[src_idx] = tau[src_idx]
                            # Use solve for numerical stability: M @ accel = tau
                            accel = np.linalg.solve(M, tau_single)
                            self.data["induced_accelerations"][src_idx][idx] = accel
            except Exception as e:
                LOGGER.debug(
                    "Failed to compute induced accelerations at frame %d: %s", idx, e
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
            LOGGER.debug("Failed to compute ground forces at frame %d: %s", idx, e)

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
            # Compute on demand if not cached?
            # Doing it post-hoc is expensive (needs re-simulation).
            # For now, return empty if not recorded.
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

    def compute_analysis_post_hoc(
        self, subsample: int = 1, key_frames_only: bool = False
    ) -> None:
        """Compute expensive analysis (ZTCF, Induced Accel) after recording.

        PERFORMANCE: Optimized with:
        - Pre-allocated arrays instead of list appending
        - Optional subsampling for long recordings
        - Key frame mode for critical points only
        - Progress logging for long computations

        Args:
            subsample: Compute every N frames (1 = all frames, 10 = every 10th)
            key_frames_only: If True, only compute at detected key frames
                            (impact, transition, peak velocity)
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

        # PERFORMANCE: Determine which frames to compute
        if key_frames_only:
            # Detect key frames based on velocity peaks and transitions
            speed = np.linalg.norm(vs, axis=1)
            from scipy.signal import find_peaks

            peaks, _ = find_peaks(speed, prominence=np.std(speed) * 0.5)
            # Include start, end, peaks, and midpoints
            key_indices = np.unique(
                np.concatenate(
                    [
                        [0, n_frames - 1],
                        peaks,
                        np.linspace(0, n_frames - 1, 20).astype(int),
                    ]
                )
            )
            frame_indices = np.sort(key_indices)
            LOGGER.debug(f"Key frame mode: computing {len(frame_indices)} frames")
        else:
            frame_indices = np.arange(0, n_frames, subsample)

        n_compute = len(frame_indices)
        nv = vs.shape[1]

        # PERFORMANCE: Pre-allocate arrays instead of list appending
        ztcf_accels = np.zeros((n_compute, nv))
        zvcf_accels = np.zeros((n_compute, nv))
        drift_accels = np.zeros((n_compute, nv))
        control_accels = np.zeros((n_compute, nv))
        computed_times = times[frame_indices]

        # Progress logging for long computations
        log_interval = max(1, n_compute // 10)

        for out_idx, frame_idx in enumerate(frame_indices):
            q = qs[frame_idx]
            v = vs[frame_idx]
            tau = taus[frame_idx]

            # Set state for computation (without advancing time)
            self.engine.set_state(q, v)
            self.engine.set_control(tau)

            # Need to force update
            self.engine.forward()

            # Compute all metrics for this frame
            ztcf_accels[out_idx] = self.engine.compute_ztcf(q, v)
            zvcf_accels[out_idx] = self.engine.compute_zvcf(q)
            drift_accels[out_idx] = self.engine.compute_drift_acceleration()
            control_accels[out_idx] = self.engine.compute_control_acceleration(tau)

            # Progress logging
            if out_idx > 0 and out_idx % log_interval == 0:
                LOGGER.debug(
                    f"Post-hoc analysis progress: {out_idx}/{n_compute} frames"
                )

        # Store results
        self.data["counterfactuals"]["ztcf_accel"] = (computed_times, ztcf_accels)
        self.data["counterfactuals"]["zvcf_accel"] = (computed_times, zvcf_accels)
        # Also map generic 'ztcf'/'zvcf' for Plotter
        self.data["counterfactuals"]["ztcf"] = (computed_times, ztcf_accels)
        self.data["counterfactuals"]["zvcf"] = (computed_times, zvcf_accels)

        self.data["induced_accelerations"]["gravity"] = (
            computed_times,
            drift_accels,
        )  # Approx label
        self.data["induced_accelerations"]["drift"] = (
            computed_times,
            drift_accels,
        )
        self.data["induced_accelerations"]["control"] = (
            computed_times,
            control_accels,
        )
        self.data["induced_accelerations"]["total"] = (
            computed_times,
            drift_accels + control_accels,
        )

        LOGGER.info(
            f"Post-hoc analysis complete. Computed {n_compute} frames "
            f"(subsample={subsample}, key_frames_only={key_frames_only})"
        )

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
