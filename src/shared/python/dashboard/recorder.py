"""Generic recorder for PhysicsEngine compatible simulations.

Records state, control, and derived quantities for analysis and plotting.
Integrates GRF analysis and swing-plane wrench decomposition (Issue #761).
"""

from __future__ import annotations

from typing import Any

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
        if not (engine is not None):
            raise ValueError("engine must be provided")
        if not (engine is not None):
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
        if not (config is not None):
            raise ValueError("config must be provided")
        if not (config is not None):
            raise ValueError("config must be provided")
        self.analysis_config.update(config)
        logger.info(f"Recorder analysis config updated: {self.analysis_config}")

        if self._buffers_initialized:
            self._ensure_buffers_allocated()

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
