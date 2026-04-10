"""Ball flight physics simulation with Magnus effect and drag.

This module implements research-grade ball flight physics including:
- Magnus effect (spin-induced forces)
- Drag forces (Reynolds number dependent)
- Launch angle and velocity effects
- 3D trajectory calculation
- Landing dispersion patterns

Refactored to address DRY and Orthogonality violations (Pragmatic Programmer).

This file is now a backward-compatible coordinator module. The implementation
has been decomposed into focused submodules (P1 sprint, issue #2486):

- ``ball_properties``       — BallProperties dataclass and aerodynamic constants
- ``ball_launch_conditions`` — LaunchConditions, EnvironmentalConditions, TrajectoryPoint
- ``ball_simulator``        — BallFlightSimulator (Rust-kernel backed)
- ``ball_enhanced_simulator`` — EnhancedBallFlightSimulator (toggleable aero + Monte Carlo)

.. deprecated::
    The RK4 integration loop in this module has a Rust kernel equivalent
    in ``upstream_physics`` (via ``rust_kernel.create_integrator_config``).
    New simulation code should use the Rust-backed integrator for native
    performance and WASM parity with the React frontend.

Planned enhancement: implement Environmental Gradient Modeling (wind shear, temperature gradients).
Planned enhancement: implement Hydrodynamic Lubrication (wet ball physics).
Planned enhancement: implement Dimple Geometry Optimization.
Planned enhancement: implement Turbulence Modeling.
Planned enhancement: implement Mud Ball Physics.
"""

from __future__ import annotations

# Re-export everything for backward compatibility.
# All public symbols remain importable from this module.
from src.shared.python.physics.ball_enhanced_simulator import (
    EnhancedBallFlightSimulator,
)
from src.shared.python.physics.ball_launch_conditions import (
    EnvironmentalConditions,
    LaunchConditions,
    TrajectoryPoint,
)
from src.shared.python.physics.ball_properties import (
    MAX_LIFT_COEFFICIENT,
    MIN_SPEED_THRESHOLD,
    NUMERICAL_EPSILON,
    BallProperties,
)
from src.shared.python.physics.ball_simulator import BallFlightSimulator

__all__ = [
    "BallProperties",
    "LaunchConditions",
    "EnvironmentalConditions",
    "TrajectoryPoint",
    "BallFlightSimulator",
    "EnhancedBallFlightSimulator",
    # Constants
    "MIN_SPEED_THRESHOLD",
    "MAX_LIFT_COEFFICIENT",
    "NUMERICAL_EPSILON",
]
