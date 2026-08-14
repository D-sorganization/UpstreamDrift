"""Designer metrics for BunkerShot3D (issue #8614).

Metrics computed from result artifacts, not solver internals, so they work
identically across fidelity tiers F0-F3.

Modules:
    trajectory: Dig/skid, depth trace, divot profile (entry/max/exit)
    energy: Club KE lost, energy to sand, energy to ball
    force: Peak/mean force, deceleration, contact duration
    twist: Moment about shaft axis and CG — why bounce and relief exist
    forgiveness: Sensitivity of ball launch to input errors
"""

from __future__ import annotations

from .energy import EnergyPartition, compute_energy_partition
from .force import ForceMetrics, compute_force_metrics
from .forgiveness import (
    ForgivenessMetrics,
    SensitivityGradient,
    compute_forgiveness_metrics,
)
from .trajectory import (
    DivotProfile,
    TrajectoryMetrics,
    compute_trajectory_metrics,
)
from .twist import TwistMetrics, compute_twist_metrics

__all__ = [
    "DivotProfile",
    "EnergyPartition",
    "ForceMetrics",
    "ForgivenessMetrics",
    "SensitivityGradient",
    "TrajectoryMetrics",
    "TwistMetrics",
    "compute_energy_partition",
    "compute_force_metrics",
    "compute_forgiveness_metrics",
    "compute_trajectory_metrics",
    "compute_twist_metrics",
]
