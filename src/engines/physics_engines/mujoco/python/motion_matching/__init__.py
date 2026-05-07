"""MuJoCo motion-matching package (Python side) — canonical forward-sim wrappers.

Houses the forward simulator (``simulate.py``), polynomial-torque driver
(``torque_driver.py``), the model builder with compiled MJCF cache
(``_model_builder.py``), and the visualization sub-package (``viz``). See
``src/engines/physics_engines/mujoco/MUJOCO_PARITY_SPEC.md`` §2 and
``src/engines/CROSS_ENGINE_PARITY_SPEC.md`` (§2.2) for the contract.
"""

from __future__ import annotations

from src.engines.physics_engines.mujoco.python.motion_matching._model_builder import (
    CompiledModel,
    build_model,
    clear_cache,
    load_model,
)

from .fit_swing import (
    FitOptions,
    FitResult,
    MinimizerOptions,
    fit_swing_mujoco,
)
from .jacobians import (
    JacobianCache,
    compute_cost_gradient_analytical,
    compute_qpos_jacobian,
    polynomial_du_dtheta,
)
from .simulate import (
    SimOptions,
    SimOut,
    simulate_with_coefficients,
)
from .torque_driver import (
    POLY_BOUNDS,
    PolynomialTorqueDriver,
    polynomial_torque_bounds,
)

__all__: list[str] = [
    "POLY_BOUNDS",
    "CompiledModel",
    "FitOptions",
    "FitResult",
    "JacobianCache",
    "MinimizerOptions",
    "PolynomialTorqueDriver",
    "SimOptions",
    "SimOut",
    "build_model",
    "clear_cache",
    "compute_cost_gradient_analytical",
    "compute_qpos_jacobian",
    "fit_swing_mujoco",
    "load_model",
    "polynomial_du_dtheta",
    "polynomial_torque_bounds",
    "simulate_with_coefficients",
]
