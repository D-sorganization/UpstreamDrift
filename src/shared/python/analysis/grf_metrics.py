"""Ground Reaction Force (GRF) and Center of Pressure (CoP) metrics analysis module."""

import numpy as np

from src.shared.python.analysis.dataclasses import GRFMetrics
from src.shared.python.core.contracts import ensure


class GRFMetricsMixin:
    """Mixin for computing GRF and CoP metrics."""

    def compute_grf_metrics(self) -> GRFMetrics | None:
        """Compute Ground Reaction Force and Center of Pressure metrics.

        Design by Contract:
            Postconditions:
                - cop_path_length >= 0 (path length is non-negative)
                - cop_max_velocity >= 0 (velocity magnitude is non-negative)
                - cop_x_range >= 0 and cop_y_range >= 0
                - all returned values are finite

        Returns:
            GRFMetrics object or None if data unavailable
        """
        cop_position = getattr(self, "cop_position", None)
        ground_forces = getattr(self, "ground_forces", None)
        dt = getattr(self, "dt", 0.0)

        if cop_position is None or len(cop_position) == 0:
            return None

        # CoP Path Length
        # OPTIMIZATION: Manual slicing is slightly faster than np.diff
        cop_diff = cop_position[1:] - cop_position[:-1]
        # OPTIMIZATION: np.hypot is faster than np.linalg.norm for 2D vectors
        # OPTIMIZATION: Explicit sqrt calculation is faster than np.linalg.norm for 3D
        if cop_diff.shape[1] == 2:
            cop_dist = np.hypot(cop_diff[:, 0], cop_diff[:, 1])
        else:
            cop_dist = np.sqrt(np.sum(cop_diff**2, axis=1))
        path_length = np.sum(cop_dist)

        # CoP Velocity
        # OPTIMIZATION: Use pre-computed distance norm to avoid redundant allocations
        max_vel = np.max(cop_dist) / dt if dt > 0 else 0.0

        # CoP Range
        # Note: Vectorized min/max (axis=0) was found to be slower than separate column access
        x_range = float(np.max(cop_position[:, 0]) - np.min(cop_position[:, 0]))
        y_range = float(np.max(cop_position[:, 1]) - np.min(cop_position[:, 1]))

        # Force metrics
        peak_vertical = None
        peak_shear = None
        # Assuming Z is vertical (index 2)
        if ground_forces is not None and ground_forces.shape[1] >= 3:
            peak_vertical = float(np.max(ground_forces[:, 2]))
            # Shear is magnitude of X and Y forces
            shear = np.hypot(ground_forces[:, 0], ground_forces[:, 1])
            peak_shear = float(np.max(shear))

        result = GRFMetrics(
            cop_path_length=float(path_length),
            cop_max_velocity=float(max_vel),
            cop_x_range=x_range,
            cop_y_range=y_range,
            peak_vertical_force=peak_vertical,
            peak_shear_force=peak_shear,
        )

        # Postconditions
        ensure(
            result.cop_path_length >= 0,
            "CoP path length must be non-negative",
            result.cop_path_length,
        )
        ensure(
            result.cop_max_velocity >= 0,
            "CoP max velocity must be non-negative",
            result.cop_max_velocity,
        )
        ensure(
            result.cop_x_range >= 0,
            "CoP X range must be non-negative",
            result.cop_x_range,
        )
        ensure(
            result.cop_y_range >= 0,
            "CoP Y range must be non-negative",
            result.cop_y_range,
        )
        ensure(
            np.isfinite(result.cop_path_length),
            "CoP path length must be finite",
            result.cop_path_length,
        )

        return result
