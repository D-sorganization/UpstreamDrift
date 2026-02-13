"""Data management and transformation utilities for plotting."""

from __future__ import annotations

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.plotting.base import RecorderInterface

logger = get_logger(__name__)


class DataManager:
    """Manages data retrieval, caching, and common transformations."""

    def __init__(
        self,
        recorder: RecorderInterface,
        joint_names: list[str] | None = None,
        enable_cache: bool = True,
    ) -> None:
        """Initialize data manager.

        Args:
            recorder: Object providing get_time_series(field_name) method
            joint_names: Optional list of joint names.
            enable_cache: If True, cache data fetches to improve performance
        """
        self.recorder = recorder
        self.joint_names = joint_names or []
        self.enable_cache = enable_cache
        self._data_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        if self.enable_cache:
            self._preload_common_data()

    def _preload_common_data(self) -> None:
        """Pre-fetch commonly used data series to cache."""
        common_fields = [
            "joint_positions",
            "joint_velocities",
            "joint_torques",
            "kinetic_energy",
            "potential_energy",
            "total_energy",
            "club_head_speed",
            "club_head_position",
            "angular_momentum",
            "cop_position",
            "com_position",
            "actuator_powers",
            "ground_forces",
        ]

        for field in common_fields:
            try:
                times, values = self.get_series(field)
                if len(times) > 0:
                    self._data_cache[field] = (times, values)
            except (KeyError, RuntimeError, ValueError, OSError) as e:
                # Field may not exist in all recorders
                logger.debug(f"Could not pre-load field '{field}': {e}")

    def get_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get time series data with caching.

        Args:
            field_name: Name of the field to retrieve

        Returns:
            Tuple of (times, values) arrays
        """
        if not self.enable_cache:
            times, values = self.recorder.get_time_series(field_name)
            return np.asarray(times), np.asarray(values)

        # Check cache first
        if field_name in self._data_cache:
            return self._data_cache[field_name]

        # Not in cache, fetch and cache it
        times, values = self.recorder.get_time_series(field_name)
        times_arr, values_arr = np.asarray(times), np.asarray(values)
        if len(times_arr) > 0:
            self._data_cache[field_name] = (times_arr, values_arr)

        return times_arr, values_arr

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._data_cache.clear()
        if self.enable_cache:
            self._preload_common_data()

    def get_joint_name(self, joint_idx: int) -> str:
        """Get human-readable joint name."""
        if 0 <= joint_idx < len(self.joint_names):
            return self.joint_names[joint_idx]
        return f"Joint {joint_idx}"

    def get_aligned_label(self, idx: int, data_dim: int) -> str:
        """Get label aligned with data dimension (handling nq != nv)."""
        if len(self.joint_names) == 0:
            return f"DoF {idx}"

        # If perfect match
        if data_dim == len(self.joint_names):
            return (
                self.joint_names[idx] if idx < len(self.joint_names) else f"DoF {idx}"
            )

        # If mismatch, align from the end (assuming base is at the start)
        offset = max(0, data_dim - len(self.joint_names))
        name_idx = idx - offset

        if 0 <= name_idx < len(self.joint_names):
            return self.joint_names[name_idx]

        return f"DoF {idx}"

    def get_induced_acceleration_series(
        self, source_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get induced acceleration series (uncached for now)."""
        return self.recorder.get_induced_acceleration_series(source_name)

    def get_club_induced_acceleration_series(
        self, source_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get club induced acceleration series (uncached)."""
        if hasattr(self.recorder, "get_club_induced_acceleration_series"):
            return self.recorder.get_club_induced_acceleration_series(source_name)  # type: ignore
        return np.array([]), np.array([])

    def get_counterfactual_series(self, cf_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get counterfactual series (uncached)."""
        return self.recorder.get_counterfactual_series(cf_name)  # type: ignore[attr-defined]
