"""Generic recorder for PhysicsEngine compatible simulations.

Records state, control, and derived quantities for analysis and plotting.
Integrates GRF analysis and swing-plane wrench decomposition (Issue #761).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from src.shared.python.core.contracts import invariant
from src.shared.python.engine_core.interfaces import PhysicsEngine
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


@invariant(lambda self: self.max_samples > 0, "max_samples must be positive")
@invariant(
    lambda self: self.current_idx <= self.current_capacity,
    "current_idx must not exceed current_capacity",
)
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
                logger.warning(
                    f"Recorder buffer full at {self.max_samples} samples. "
                    "Stopping recording."
                )
                self.is_recording = False
                return

            logger.debug(
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
        logger.info(f"Recorder analysis config updated: {self.analysis_config}")

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
            logger.debug("Allocated ZTCF buffer dynamically.")

        # Allocate ZVCF if missing
        if self.analysis_config["zvcf"] and self.data["zvcf_accel"] is None:
            self.data["zvcf_accel"] = np.zeros((self.max_samples, nv))
            logger.debug("Allocated ZVCF buffer dynamically.")

        # Allocate Drift if missing
        if self.analysis_config["track_drift"] and self.data["drift_accel"] is None:
            self.data["drift_accel"] = np.zeros((self.max_samples, nv))
            logger.debug("Allocated Drift buffer dynamically.")

        # Allocate Control if missing
        if (
            self.analysis_config["track_total_control"]
            and self.data["control_accel"] is None
        ):
            self.data["control_accel"] = np.zeros((self.max_samples, nv))
            logger.debug("Allocated Control buffer dynamically.")

        # Allocate Induced Accel sources if missing
        sources = cast(list[int], self.analysis_config["induced_accel_sources"])
        for idx in sources:
            if idx not in self.data["induced_accelerations"]:
                self.data["induced_accelerations"][idx] = np.zeros(
                    (self.max_samples, nv)
                )
                logger.debug(f"Allocated Induced Accel buffer for source {idx}.")

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
            "ground_moments": None,
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
        self.data["ground_moments"] = np.zeros((self.max_samples, 3))

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
        logger.debug(
            f"Initialized recorder buffers: nq={nq}, nv={nv}, max_samples={self.max_samples}"
        )

    def start(self) -> None:
        """Start recording."""
        self.is_recording = True
        logger.info("Recording started.")

    def stop(self) -> None:
        """Stop recording."""
        self.is_recording = False
        logger.info("Recording stopped. Recorded %d frames.", self.current_idx)

    def reset(self) -> None:
        """Clear all recorded data."""
        self._reset_buffers()
        logger.info("Recorder reset.")

    def record_step(self, control_input: np.ndarray | None = None) -> None:
        """Record the current state of the engine.

        Args:
            control_input: Optional control vector applied during this step.
        """
        if not self.is_recording:
            return

        self._ensure_capacity()

        if not self.is_recording:
            return

        full_state = self.engine.get_full_state()
        q = full_state["q"]
        v = full_state["v"]
        t = full_state["t"]
        M = full_state.get("M")

        if not self._buffers_initialized:
            self._initialize_array_buffers(q, v)

        tau = control_input if control_input is not None else np.zeros(len(v))
        ke = self._compute_kinetic_energy(v, M)

        idx = self.current_idx
        self._record_realtime_analysis(idx, q, v, tau, M)
        self._store_basic_data(idx, t, q, v, ke, tau)
        self._record_ground_forces(idx)

        self.current_idx += 1

    def _compute_kinetic_energy(self, v: np.ndarray, M: np.ndarray | None) -> float:
        """Compute kinetic energy from velocity and mass matrix."""
        if M is not None and M.size > 0:
            try:
                return 0.5 * v.T @ M @ v
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning("Failed to compute kinetic energy: %s", e)
        return 0.0

    def _record_realtime_analysis(
        self,
        idx: int,
        q: np.ndarray,
        v: np.ndarray,
        tau: np.ndarray,
        M: np.ndarray | None,
    ) -> None:
        """Record real-time counterfactual and induced acceleration analysis."""
        if self.analysis_config["ztcf"] and self.data["ztcf_accel"] is not None:
            try:
                self.data["ztcf_accel"][idx] = self.engine.compute_ztcf(q, v)
            except (ValueError, RuntimeError, AttributeError) as e:
                logger.warning("Failed to compute ZTCF at frame %d: %s", idx, e)

        if self.analysis_config["zvcf"] and self.data["zvcf_accel"] is not None:
            try:
                self.data["zvcf_accel"][idx] = self.engine.compute_zvcf(q)
            except (ValueError, RuntimeError, AttributeError) as e:
                logger.warning("Failed to compute ZVCF at frame %d: %s", idx, e)

        if self.analysis_config["track_drift"] and self.data["drift_accel"] is not None:
            try:
                self.data["drift_accel"][idx] = self.engine.compute_drift_acceleration()
            except (ValueError, RuntimeError, AttributeError) as e:
                logger.warning(
                    "Failed to compute drift acceleration at frame %d: %s", idx, e
                )

        if (
            self.analysis_config["track_total_control"]
            and self.data["control_accel"] is not None
        ):
            try:
                self.data["control_accel"][idx] = (
                    self.engine.compute_control_acceleration(tau)
                )
            except (ValueError, RuntimeError, AttributeError) as e:
                logger.warning(
                    "Failed to compute control acceleration at frame %d: %s", idx, e
                )

        self._record_induced_accelerations(idx, tau, M)

    def _record_induced_accelerations(
        self, idx: int, tau: np.ndarray, M: np.ndarray | None
    ) -> None:
        """Record per-source induced accelerations (vectorized when possible)."""
        sources = cast(list[int], self.analysis_config["induced_accel_sources"])
        if sources and M is not None and M.size > 0:
            try:
                M_inv = np.linalg.inv(M)
                for src_idx in sources:
                    if src_idx in self.data["induced_accelerations"]:
                        self.data["induced_accelerations"][src_idx][idx] = (
                            M_inv[:, src_idx] * tau[src_idx]
                        )
            except (ValueError, TypeError, RuntimeError) as e:
                logger.warning(
                    "Failed to compute induced accelerations at frame %d: %s", idx, e
                )
        elif sources:
            for src_idx in sources:
                if src_idx in self.data["induced_accelerations"]:
                    try:
                        tau_single = np.zeros_like(tau)
                        tau_single[src_idx] = tau[src_idx]
                        self.data["induced_accelerations"][src_idx][idx] = (
                            self.engine.compute_control_acceleration(tau_single)
                        )
                    except (ValueError, TypeError, RuntimeError) as e:
                        logger.warning(
                            "Failed to compute induced acceleration for source %d: %s",
                            src_idx,
                            e,
                        )

    def _store_basic_data(
        self,
        idx: int,
        t: float,
        q: np.ndarray,
        v: np.ndarray,
        ke: float,
        tau: np.ndarray,
    ) -> None:
        """Store core state data into recording buffers."""
        self.data["times"][idx] = t
        self.data["joint_positions"][idx] = q
        self.data["joint_velocities"][idx] = v
        self.data["kinetic_energy"][idx] = ke
        self.data["joint_torques"][idx] = tau

    def _record_ground_forces(self, idx: int) -> None:
        """Record ground reaction forces and moments from the engine."""
        try:
            grf = self.engine.compute_contact_forces()
            if grf is not None and len(grf) >= 3:
                self.data["ground_forces"][idx] = grf[:3]
                if len(grf) >= 6:
                    self.data["ground_moments"][idx] = grf[3:6]
        except (ValueError, RuntimeError, AttributeError) as e:
            logger.warning("Failed to compute ground forces at frame %d: %s", idx, e)

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
        if isinstance(values, list):
            # Legacy support for list-based data
            return times, np.array(values[: self.current_idx])
        # For dict/other types
        return times, values

    def get_induced_acceleration_series(
        self, source_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get induced acceleration series."""
        if source_name not in self.data["induced_accelerations"]:
            # Log when parsing/lookup fails for induced acceleration source
            logger.warning(
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
        logger.info("Computing post-hoc analysis...")

        if not self._buffers_initialized or self.current_idx == 0:
            logger.warning("No data recorded for post-hoc analysis")
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

        logger.info("Post-hoc analysis complete.")

    def compute_grf_and_wrench_analysis(
        self, impact_time: float | None = None, fsp_window_ms: float = 100.0
    ) -> dict[str, Any]:
        """Compute GRF analysis and swing-plane wrench decomposition.

        Builds GRFTimeSeries from recorded forces/moments/COP, runs the
        GRFAnalyzer for impulse metrics, fits a Functional Swing Plane (FSP)
        from the clubhead trajectory, and decomposes GRF wrenches into
        global, local, and swing-plane frames.

        Args:
            impact_time: Time of impact [s]. If None, uses time of peak
                vertical force.
            fsp_window_ms: FSP fitting window around impact [ms].

        Returns:
            Dictionary with GRF summary, FSP parameters, and per-frame
            wrench decompositions.
        """
        if not self._buffers_initialized or self.current_idx == 0:
            logger.warning("No data recorded for GRF/wrench analysis")
            return {}

        n = self.current_idx
        times = self.data["times"][:n]
        forces = self.data["ground_forces"][:n]
        moments = self.data["ground_moments"][:n]
        cops = self.data["cop_position"][:n]

        grf_summary = self._run_grf_analysis(times, forces, moments, cops)

        impact_time = self._resolve_impact_time(impact_time, times, forces, n)

        fsp = self._fit_swing_plane(times, impact_time, fsp_window_ms, n)

        wrench_arrays = self._compute_wrench_decomposition(forces, moments, fsp, n)

        result = self._build_grf_wrench_result(grf_summary, fsp, wrench_arrays)

        # Store in recorder data for export
        self.data["grf_analysis"] = result["grf_analysis"]
        self.data["fsp"] = result["fsp"]
        self.data["wrench_swing_plane"] = result["wrench_swing_plane"]

        logger.info(
            "GRF and wrench analysis complete. FSP RMSE=%.4f m",
            fsp.fitting_rmse if fsp else float("nan"),
        )
        return result

    def _run_grf_analysis(
        self,
        times: np.ndarray,
        forces: np.ndarray,
        moments: np.ndarray,
        cops: np.ndarray,
    ) -> Any:
        """Run GRF analysis and return the summary (or None on failure)."""
        from src.shared.python.physics.ground_reaction_forces import (
            FootSide,
            GRFAnalyzer,
            GRFTimeSeries,
        )

        grf_ts = GRFTimeSeries(
            timestamps=times,
            forces=forces,
            moments=moments,
            cops=cops,
            foot_side=FootSide.COMBINED,
        )

        analyzer = GRFAnalyzer()
        analyzer.add_grf_data(grf_ts)

        try:
            return analyzer.analyze(FootSide.COMBINED)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("GRF analysis failed: %s", e)
            return None

    @staticmethod
    def _resolve_impact_time(
        impact_time: float | None,
        times: np.ndarray,
        forces: np.ndarray,
        n: int,
    ) -> float:
        """Determine impact time from peak vertical force if not provided."""
        if impact_time is not None:
            return impact_time
        vertical_forces = forces[:, 2]
        if np.max(np.abs(vertical_forces)) > 0:
            return float(times[np.argmax(np.abs(vertical_forces))])
        return float(times[n // 2])

    def _fit_swing_plane(
        self,
        times: np.ndarray,
        impact_time: float,
        fsp_window_ms: float,
        n: int,
    ) -> Any:
        """Fit a Functional Swing Plane from the clubhead trajectory."""
        from src.shared.python.spatial_algebra.reference_frames import (
            fit_functional_swing_plane,
        )

        clubhead_traj = self.data["club_head_position"][:n]
        if clubhead_traj is not None and np.any(clubhead_traj != 0):
            try:
                return fit_functional_swing_plane(
                    clubhead_traj, times, impact_time, window_ms=fsp_window_ms
                )
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning("FSP fitting failed: %s", e)
        return None

    @staticmethod
    def _compute_wrench_decomposition(
        forces: np.ndarray,
        moments: np.ndarray,
        fsp: Any,
        n: int,
    ) -> dict[str, np.ndarray]:
        """Decompose GRF wrenches into swing-plane components."""
        from src.shared.python.spatial_algebra.reference_frames import (
            ReferenceFrame,
            ReferenceFrameTransformer,
            WrenchInFrame,
        )

        transformer = ReferenceFrameTransformer()
        if fsp is not None:
            transformer.set_swing_plane(fsp)

        wrench_decompositions: list[dict[str, float]] = []
        for i in range(n):
            wrench = WrenchInFrame(
                force=forces[i],
                torque=moments[i],
                frame=ReferenceFrame.GLOBAL,
                body_name="ground",
            )
            if fsp is not None:
                decomp = transformer.get_swing_plane_decomposition(wrench)
            else:
                decomp = {
                    "force_in_plane": 0.0,
                    "force_out_of_plane": 0.0,
                    "force_along_grip": 0.0,
                    "torque_in_plane": 0.0,
                    "torque_out_of_plane": 0.0,
                    "torque_about_grip": 0.0,
                }
            wrench_decompositions.append(decomp)

        decomp_keys = [
            "force_in_plane",
            "force_out_of_plane",
            "force_along_grip",
            "torque_in_plane",
            "torque_out_of_plane",
            "torque_about_grip",
        ]
        return {k: np.array([d[k] for d in wrench_decompositions]) for k in decomp_keys}

    @staticmethod
    def _build_grf_wrench_result(
        grf_summary: Any,
        fsp: Any,
        wrench_arrays: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Assemble the final result dictionary from analysis components."""
        result: dict[str, Any] = {
            "grf_analysis": {},
            "fsp": {},
            "wrench_swing_plane": wrench_arrays,
        }

        if grf_summary is not None:
            result["grf_analysis"] = {
                "peak_vertical_force": grf_summary.peak_vertical_force,
                "peak_horizontal_force": grf_summary.peak_horizontal_force,
                "time_to_peak_vertical": grf_summary.time_to_peak_vertical,
                "cop_trajectory_length": grf_summary.cop_trajectory_length,
                "cop_range_ap": grf_summary.cop_range_ap,
                "cop_range_ml": grf_summary.cop_range_ml,
                "linear_impulse_magnitude": grf_summary.linear_impulse.linear_impulse_magnitude,
                "angular_impulse_magnitude": grf_summary.linear_impulse.angular_impulse_magnitude,
                "duration": grf_summary.linear_impulse.duration,
            }

        if fsp is not None:
            result["fsp"] = {
                "origin": fsp.origin,
                "normal": fsp.normal,
                "in_plane_x": fsp.in_plane_x,
                "in_plane_y": fsp.in_plane_y,
                "grip_axis": fsp.grip_axis,
                "fitting_rmse": fsp.fitting_rmse,
                "fitting_window_ms": fsp.fitting_window_ms,
            }

        return result

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
                except (ValueError, TypeError, RuntimeError) as e:
                    logger.debug("Failed to convert list '%s' to numpy array: %s", k, e)
                    export_data[k] = v
            else:
                export_data[k] = v

        # Add metadata
        export_data["model_name"] = self.engine.model_name
        export_data["num_frames"] = self.current_idx
        return export_data
