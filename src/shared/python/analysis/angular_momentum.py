"""Angular momentum metrics analysis module."""

import numpy as np

from src.shared.python.analysis.dataclasses import AngularMomentumMetrics
from src.shared.python.core.contracts import ensure


class AngularMomentumMetricsMixin:
    """Mixin for computing angular momentum metrics."""

    def compute_angular_momentum_metrics(self) -> AngularMomentumMetrics | None:
        """Compute metrics related to system angular momentum.

        Design by Contract:
            Postconditions:
                - peak_magnitude >= 0
                - mean_magnitude >= 0
                - variability >= 0
                - all component peaks >= 0
                - all values finite

        Returns:
            AngularMomentumMetrics object or None if data unavailable
        """
        angular_momentum = getattr(self, "angular_momentum", None)
        times = getattr(self, "times", None)

        if angular_momentum is None or len(angular_momentum) == 0:
            return None

        # OPTIMIZATION: Explicit sqrt calculation is faster than np.linalg.norm
        mag = np.sqrt(np.sum(angular_momentum**2, axis=1))

        peak_mag = float(np.max(mag))
        peak_idx = int(np.argmax(mag))
        peak_time = float(times[peak_idx]) if times is not None else 0.0

        # Mean
        mean_mag = float(np.mean(mag))

        # Components peaks (absolute)
        peak_lx = float(np.max(np.abs(angular_momentum[:, 0])))
        peak_ly = float(np.max(np.abs(angular_momentum[:, 1])))
        peak_lz = float(np.max(np.abs(angular_momentum[:, 2])))

        # Variability
        std_mag = float(np.std(mag))
        variability = std_mag / mean_mag if mean_mag > 0 else 0.0

        result = AngularMomentumMetrics(
            peak_magnitude=peak_mag,
            peak_time=peak_time,
            mean_magnitude=mean_mag,
            peak_lx=peak_lx,
            peak_ly=peak_ly,
            peak_lz=peak_lz,
            variability=variability,
        )

        # Postconditions
        ensure(
            result.peak_magnitude >= 0,
            "peak_magnitude must be non-negative",
            result.peak_magnitude,
        )
        ensure(
            result.mean_magnitude >= 0,
            "mean_magnitude must be non-negative",
            result.mean_magnitude,
        )
        ensure(
            result.variability >= 0,
            "variability must be non-negative",
            result.variability,
        )
        ensure(result.peak_lx >= 0, "peak_lx must be non-negative", result.peak_lx)
        ensure(result.peak_ly >= 0, "peak_ly must be non-negative", result.peak_ly)
        ensure(result.peak_lz >= 0, "peak_lz must be non-negative", result.peak_lz)

        return result
