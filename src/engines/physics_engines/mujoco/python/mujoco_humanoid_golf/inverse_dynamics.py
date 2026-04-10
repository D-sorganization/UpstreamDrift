"""Inverse dynamics computation for golf swing analysis.

This module provides inverse dynamics solvers for computing required joint
torques from desired motion. Includes:

- Full inverse dynamics for open-chain systems
- Partial inverse dynamics for parallel mechanisms (closed-chain)
- Recursive Newton-Euler algorithm
- Composite rigid body algorithm
- Force decomposition analysis
"""

from __future__ import annotations

from src.shared.python.logging_pkg.logging_config import get_logger

from ._id_core import InverseDynamicsSolver, RecursiveNewtonEuler
from ._id_export import (
    InverseDynamicsAnalyzer,
    _build_inverse_dynamics_csv_row,
    _validate_inverse_dynamics_export_inputs,
    export_inverse_dynamics_to_csv,
)
from ._id_models import (
    ForceDecomposition,
    InducedAccelerationResult,
    InverseDynamicsResult,
)

logger = get_logger(__name__)

__all__ = [
    "ForceDecomposition",
    "InducedAccelerationResult",
    "InverseDynamicsAnalyzer",
    "InverseDynamicsResult",
    "InverseDynamicsSolver",
    "RecursiveNewtonEuler",
    "_build_inverse_dynamics_csv_row",
    "_validate_inverse_dynamics_export_inputs",
    "export_inverse_dynamics_to_csv",
]
