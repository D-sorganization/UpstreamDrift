"""Statistical metrics for perturbation analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricStatistics:
    """Summary statistics for a single metric across trials."""

    mean: float | np.ndarray
    std: float | np.ndarray
    cv: float | np.ndarray  # Coefficient of variation (std / mean)
    min_val: float | np.ndarray
    max_val: float | np.ndarray
    median: float | np.ndarray
    iqr: float | np.ndarray
    p5: float | np.ndarray
    p95: float | np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "mean": self._tolist(self.mean),
            "std": self._tolist(self.std),
            "cv": self._tolist(self.cv),
            "min_val": self._tolist(self.min_val),
            "max_val": self._tolist(self.max_val),
            "median": self._tolist(self.median),
            "iqr": self._tolist(self.iqr),
            "p5": self._tolist(self.p5),
            "p95": self._tolist(self.p95),
        }

    @staticmethod
    def _tolist(val: float | np.ndarray) -> float | list[float]:
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val


def compute_metric_statistics(values: np.ndarray) -> MetricStatistics:
    """Compute statistics from an array of metric values over trials.

    Parameters
    ----------
    values : np.ndarray
        Array of values spanning trials. Shape: (n_trials, ...)

    Returns
    -------
    MetricStatistics
        Computed statistics.
    """
    if not (len(values) > 0):
        raise ValueError("Cannot compute statistics on empty values")

    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)

    # Handle cv with zero mean
    if np.isscalar(mean) or mean.ndim == 0:
        cv = std / mean if abs(mean) > 1e-12 else 0.0
    else:
        cv = np.zeros_like(mean)
        mask = np.abs(mean) > 1e-12
        cv[mask] = std[mask] / mean[mask]

    p25, median, p75 = np.percentile(values, [25, 50, 75], axis=0)
    iqr = p75 - p25
    p5, p95 = np.percentile(values, [5, 95], axis=0)

    # Convert to standard types or keep as numpy for vector outputs
    # Let's ensure scalars become floats to avoid json serialization issues later
    def _to_scalar_or_array(arr: Any) -> float | np.ndarray:
        if isinstance(arr, np.ndarray) and arr.ndim == 0:
            return float(arr)
        if isinstance(arr, float | int | np.number):
            return float(arr)
        return arr

    return MetricStatistics(
        mean=_to_scalar_or_array(mean),  # type: ignore[arg-type]
        std=_to_scalar_or_array(std),  # type: ignore[arg-type]
        cv=_to_scalar_or_array(cv),  # type: ignore[arg-type]
        min_val=_to_scalar_or_array(np.min(values, axis=0)),  # type: ignore[arg-type]
        max_val=_to_scalar_or_array(np.max(values, axis=0)),  # type: ignore[arg-type]
        median=_to_scalar_or_array(median),  # type: ignore[arg-type]
        iqr=_to_scalar_or_array(iqr),  # type: ignore[arg-type]
        p5=_to_scalar_or_array(p5),  # type: ignore[arg-type]
        p95=_to_scalar_or_array(p95),  # type: ignore[arg-type]
    )
