from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "RecorderInterface",
]


@runtime_checkable
class RecorderInterface(Protocol):
    """Protocol for recording and retrieving simulation data.

    Allows different backends (MuJoCo, Drake, Pinocchio) to be visualized
    using the same widgets.
    """

    # The engine associated with this recorder (optional, but useful for joint names)
    engine: Any

    @abstractmethod
    def get_time_series(
        self, field_name: str
    ) -> tuple[np.ndarray, np.ndarray | list[Any]]:
        """Get time series data for a specific field.

        Args:
            field_name: The metric key (e.g. 'joint_positions', 'ztcf_accel').

        Returns:
            Tuple of (times, values).
            times: (N,) array of time timestamps.
            values: (N, D) array of data values.
        """
        ...

    @abstractmethod
    def get_induced_acceleration_series(
        self, source_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get time series for induced acceleration from a specific source.

        Args:
            source_name: Name of the source (e.g. 'gravity') or actuator index.

        Returns:
            Tuple of (times, values).
        """
        ...

    @abstractmethod
    def set_analysis_config(self, config: dict[str, Any]) -> None:
        """Configure which advanced metrics to record/compute.

        Args:
            config: Dictionary of configuration flags (e.g. {'ztcf': True}).
        """
        ...
