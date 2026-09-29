# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Unified Abstract Interface for Physics Engines.

This module defines the Protocol that all physics engines (MuJoCo, Drake, Pinocchio, etc.)
must adhere to. This ensures that the GUI and Analytics layers can operate
agnostic of the underlying solver.

The PhysicsEngine Protocol is composed of focused sub-protocols defined in
``sub_protocols.py``:

    - **Loadable**: Model loading and identification (3 methods)
    - **Steppable**: Time stepping (3 methods)
    - **Queryable**: State inspection (4 methods)
    - **DynamicsComputable**: Physics computation (7 methods)
    - **SupportsParameterGradients**: Parameter-gradient analysis (2 methods)
    - **CounterfactualComputable**: What-if analysis (2 methods)
    - **Recordable**: Data collection (3 methods, on RecorderInterface)

Consumers that only need a subset of the interface can depend on the
appropriate sub-protocol instead of the full PhysicsEngine.

Design by Contract:
    This interface defines contracts that all implementations must satisfy:

    State Machine:
        UNINITIALIZED -> [load_from_path/load_from_string] -> INITIALIZED
        INITIALIZED -> [reset] -> INITIALIZED (t=0)
        INITIALIZED -> [step] -> INITIALIZED (t+=dt)

    Global Invariants (all implementations must maintain):
        - After initialization: model is loaded and queryable
        - Time is always non-negative
        - State arrays (q, v) have consistent dimensions
        - Mass matrix is always symmetric positive definite
        - Superposition: a_full = a_drift + a_control (Section F)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.shared.python.engine_core._dynamics_interface import DynamicsInterface
from src.shared.python.engine_core._recorder_interface import RecorderInterface
from src.shared.python.engine_core._simulation_interface import SimulationInterface
from src.shared.python.engine_core.sub_protocols import (
    CounterfactualComputable,
    DynamicsComputable,
    Loadable,
    Queryable,
    Recordable,
    Steppable,
    SupportsParameterGradients,
)

# Re-export sub-protocols for convenient access
__all__ = [
    "CounterfactualComputable",
    "DynamicsComputable",
    "Loadable",
    "PhysicsEngine",
    "Queryable",
    "Recordable",
    "RecorderInterface",
    "Steppable",
    "SupportsParameterGradients",
]


@runtime_checkable
class PhysicsEngine(SimulationInterface, DynamicsInterface, Protocol):
    """Protocol defining the required interface for a Golf Modeling Suite physics engine.

    This protocol composes the following sub-protocols:
        - Loadable: model_name, load_from_path, load_from_string
        - Steppable: reset, step, forward
        - Queryable: get_state, set_state, set_control, get_time
        - DynamicsComputable: compute_mass_matrix, compute_bias_forces,
          compute_gravity_forces, compute_inverse_dynamics, compute_jacobian,
          compute_drift_acceleration, compute_control_acceleration
        - CounterfactualComputable: compute_ztcf, compute_zvcf
        - Checkpointable: save_checkpoint, restore_checkpoint, engine_type

    All implementations must be stateless wrappers around a Model/Data pair (or equivalent),
    or manage their own internal state consistently.

    Design by Contract:
        This protocol defines the contract between the simulation framework and
        physics engine implementations. Each method documents its:
        - Preconditions: What must be true before calling
        - Postconditions: What will be true after successful return
        - Invariants: What is preserved by the operation
    """
