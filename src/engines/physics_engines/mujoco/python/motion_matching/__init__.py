"""MuJoCo motion-matching package (Python side) — canonical forward-sim wrappers.

Houses the forward simulator (``simulate.py``), polynomial-torque driver
(``torque_driver.py``), and the visualization sub-package (``viz``). See
``src/engines/physics_engines/mujoco/MUJOCO_PARITY_SPEC.md`` §2 and
``src/engines/CROSS_ENGINE_PARITY_SPEC.md`` (§2.2) for the contract.
"""

from __future__ import annotations

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
    "PolynomialTorqueDriver",
    "SimOptions",
    "SimOut",
    "polynomial_torque_bounds",
    "simulate_with_coefficients",
]
