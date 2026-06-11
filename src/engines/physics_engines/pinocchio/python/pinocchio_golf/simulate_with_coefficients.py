"""Compatibility facade for the canonical Pinocchio forward simulator.

This module used to carry an independent polynomial-torque forward
simulator. The maintained implementation is the motion-matching stack
used by production providers and parity tests, so this module now only
re-exports that canonical API.
"""

from __future__ import annotations

from src.engines.physics_engines.pinocchio.python.motion_matching.simulate import (
    CLUBHEAD_FRAME_NAME,
    COEFFS_PER_JOINT,
    GRIP_FRAME_NAME,
    POLY_DEGREE,
    SimOptions,
    SimOut,
    evaluate_polynomial_torque,
    simulate_with_coefficients,
)
from src.engines.physics_engines.pinocchio.python.motion_matching.synthesize import (
    SynthesizeOptions,
    synthesize_target_from_coefficients,
)

__all__ = [
    "CLUBHEAD_FRAME_NAME",
    "COEFFS_PER_JOINT",
    "GRIP_FRAME_NAME",
    "POLY_DEGREE",
    "SimOptions",
    "SimOut",
    "SynthesizeOptions",
    "evaluate_polynomial_torque",
    "simulate_with_coefficients",
    "synthesize_target_from_coefficients",
]
