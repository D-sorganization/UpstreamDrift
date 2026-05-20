# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

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

from ._recorder_analysis import _AnalysisMixin
from ._recorder_buffers import _BuffersMixin
from ._recorder_playback import _PlaybackMixin
from ._recorder_recording import _RecordingMixin

logger = get_logger(__name__)


@invariant(lambda self: self.max_samples > 0, "max_samples must be positive")
@invariant(
    lambda self: self.current_idx <= self.current_capacity,
    "current_idx must not exceed current_capacity",
)
class GenericPhysicsRecorder(
    _BuffersMixin,
    _RecordingMixin,
    _AnalysisMixin,
    _PlaybackMixin,
):
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
        if engine is None:
            raise ValueError("engine must be provided")
        self.engine = engine
        self.max_samples = max_samples
        self.initial_capacity = initial_capacity
        self.current_capacity = initial_capacity
        self.growth_factor = 1.5
        self.current_idx = 0
        self.is_recording = False
        self.data: dict[str, Any] = {}
        self._buffers_initialized = False

        self.analysis_config = {
            "ztcf": False,
            "zvcf": False,
            "track_drift": False,
            "track_total_control": False,
            "induced_accel_sources": [],
        }

        self._reset_buffers()

    def set_analysis_config(self, config: dict[str, Any]) -> None:
        if config is None:
            raise ValueError("config must be provided")
        self.analysis_config.update(config)
        logger.info(f"Recorder analysis config updated: {self.analysis_config}")

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
        if q is None:
            raise ValueError("q must be provided")
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
        self.is_recording = True
        logger.info("Recording started.")

    def stop(self) -> None:
        self.is_recording = False
        logger.info("Recording stopped. Recorded %d frames.", self.current_idx)

    def reset(self) -> None:
        self._reset_buffers()
        logger.info("Recorder reset.")


__all__ = ["GenericPhysicsRecorder"]
