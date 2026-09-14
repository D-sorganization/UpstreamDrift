"""Standardized five shared evaluation metrics for tour motion matching.

Re-exports SharedMetrics and compute_shared_metrics from src.shared.python.motion_matching.tour_metrics
for backward compatibility.
"""

from __future__ import annotations

from src.shared.python.motion_matching.tour_metrics import (
    SharedMetrics,
    _pelvis_yaw,
    compute_shared_metrics,
)

__all__ = [
    "SharedMetrics",
    "compute_shared_metrics",
]
