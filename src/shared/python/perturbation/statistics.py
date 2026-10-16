"""Statistical metrics for perturbation analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Absolute mean below which the coefficient of variation is undefined.
CV_NEAR_ZERO_MEAN = 1e-12

#: Documented reason string attached when ``cv`` is NaN (NSX-82, sec:var-face).
CV_UNDEFINED_REASON = (
    "cv undefined: |mean| <= 1e-12, so SD/|mean| is undefined or arbitrarily "
    "large for a zero-centred outcome; use wrapped OutcomeDispersion "
    "statistics (loop_nullspace.tex sec:var-face)"
)


@dataclass
class MetricStatistics:
    """Summary statistics for a single metric across trials."""

    mean: float | np.ndarray
    std: float | np.ndarray
    # Coefficient of variation std / mean.  NaN (never 0) where |mean| is
    # near zero; ``cv_reason`` then says why (NSX-82, sec:var-face).
    cv: float | np.ndarray
    min_val: float | np.ndarray
    max_val: float | np.ndarray
    median: float | np.ndarray
    iqr: float | np.ndarray
    p5: float | np.ndarray
    p95: float | np.ndarray
    cv_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "mean": self._tolist(self.mean),
            "std": self._tolist(self.std),
            # strict JSON has no NaN: an undefined cv is serialised as null
            "cv": self._json_safe(self._tolist(self.cv)),
            "cv_reason": self.cv_reason,
            "min_val": self._tolist(self.min_val),
            "max_val": self._tolist(self.max_val),
            "median": self._tolist(self.median),
            "iqr": self._tolist(self.iqr),
            "p5": self._tolist(self.p5),
            "p95": self._tolist(self.p95),
        }

    @staticmethod
    def _json_safe(val: float | list[float]) -> Any:
        if isinstance(val, list):
            return [None if np.isnan(v) else v for v in val]
        return None if np.isnan(val) else val

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
        Computed statistics.  ``cv`` is ``std / mean`` where
        ``|mean| > CV_NEAR_ZERO_MEAN`` and **NaN** elsewhere (it used to be
        silently 0, which made a zero-centred outcome look perfectly robust);
        ``cv_reason`` is then ``CV_UNDEFINED_REASON``, else ``None``.
    """
    if not (len(values) > 0):
        raise ValueError("Cannot compute statistics on empty values")

    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)

    # cv is undefined (NaN, not 0) when the mean is near zero
    if np.isscalar(mean) or mean.ndim == 0:
        undefined = not abs(mean) > CV_NEAR_ZERO_MEAN
        cv = np.nan if undefined else std / mean
    else:
        cv = np.full_like(mean, np.nan, dtype=float)
        mask = np.abs(mean) > CV_NEAR_ZERO_MEAN
        cv[mask] = std[mask] / mean[mask]
        undefined = not bool(mask.all())

    p25, median, p75 = np.percentile(values, [25, 50, 75], axis=0)
    iqr = p75 - p25
    p5, p95 = np.percentile(values, [5, 95], axis=0)

    # Convert to standard types or keep as numpy for vector outputs
    # Let's ensure scalars become floats to avoid json serialization issues later
    def _to_scalar_or_array(arr: Any) -> float | np.ndarray:
        if isinstance(arr, np.ndarray) and arr.ndim == 0:
            return float(arr)
        if isinstance(arr, (float, int, np.number)):
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
        cv_reason=CV_UNDEFINED_REASON if undefined else None,
    )
