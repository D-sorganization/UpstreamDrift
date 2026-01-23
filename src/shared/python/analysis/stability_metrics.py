"""Stability metrics analysis module."""

import numpy as np

from src.shared.python.analysis.dataclasses import StabilityMetrics


class StabilityMetricsMixin:
    """Mixin for computing stability metrics."""

    def compute_stability_metrics(self) -> StabilityMetrics | None:
        """Compute postural stability metrics.

        Requires both CoP and CoM positions.

        Returns:
            StabilityMetrics object or None if data unavailable
        """
        cop_position = getattr(self, "cop_position", None)
        com_position = getattr(self, "com_position", None)

        if (
            cop_position is None
            or com_position is None
            or len(cop_position) != len(com_position)
        ):
            return None

        # Horizontal plane distance (X-Y)
        # Note: Depending on coordinate system, vertical might be Z or Y.
        # MuJoCo standard is Z-up. We assume Z is vertical.
        # CoP is usually 3D on floor (Z=0) or 2D.

        cop_xy = cop_position[:, :2]
        com_xy = com_position[:, :2]

        # OPTIMIZATION: np.hypot is faster for 2D vectors
        diff = cop_xy - com_xy
        dist = np.hypot(diff[:, 0], diff[:, 1])

        # Inclination Angle (Angle between vertical and CoP-CoM vector)
        # Vector P = CoM - CoP
        # If CoP is 2D, assume Z=0
        if cop_position.shape[1] == 2:
            cop_z = np.zeros(len(cop_position))
        else:
            cop_z = cop_position[:, 2]

        vec_temp = com_position - np.column_stack((cop_xy, cop_z))
        vec: np.ndarray = vec_temp

        # Angle with vertical (Z-axis [0, 0, 1])
        # dot(v, k) = |v| * |k| * cos(theta)
        # theta = arccos( v_z / |v| )

        # OPTIMIZATION: Explicit sqrt sum is faster than np.linalg.norm
        vec_norm = np.sqrt(np.sum(vec**2, axis=1))
        # Avoid division by zero
        vec_norm[vec_norm < 1e-6] = 1.0

        cos_theta = vec[:, 2] / vec_norm
        # Clip for numerical stability
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angles_rad = np.arccos(cos_theta)
        angles_deg = np.rad2deg(angles_rad)

        return StabilityMetrics(
            min_com_cop_distance=float(np.min(dist)),
            max_com_cop_distance=float(np.max(dist)),
            mean_com_cop_distance=float(np.mean(dist)),
            peak_inclination_angle=float(np.max(angles_deg)),
            mean_inclination_angle=float(np.mean(angles_deg)),
        )
