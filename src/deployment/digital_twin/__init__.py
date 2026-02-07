"""Digital Twin Framework for synchronized simulation.

This module provides digital twin capabilities:
- Synchronized simulation mirroring real robot state
- Anomaly detection comparing real vs simulated behavior
- Predictive simulation for planning
- State estimation from sensor data
"""

from __future__ import annotations

from src.deployment.digital_twin.estimator import StateEstimator
from src.deployment.digital_twin.twin import (
    AnomalyReport,
    AnomalyType,
    DigitalTwin,
)

__all__ = [
    "DigitalTwin",
    "AnomalyReport",
    "AnomalyType",
    "StateEstimator",
]
