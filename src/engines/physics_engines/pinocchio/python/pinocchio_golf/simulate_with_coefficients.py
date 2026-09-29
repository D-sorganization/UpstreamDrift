"""Compatibility facade for the canonical Pinocchio forward simulator.

This module used to carry an independent polynomial-torque forward
simulator. The maintained implementation is the motion-matching stack
used by production providers and parity tests, so this module now only
re-exports that canonical API.
"""

from __future__ import annotations

from src.engines.physics_engines.pinocchio.python.simulate_with_coefficients import (
    CANONICAL_COORDINATE_NAMES,
    CLUBHEAD_FRAME_NAME,
    COEFFS_PER_JOINT,
    DEFAULT_GOLFER_URDF,
    GRIP_FRAME_NAME,
    POLY_BOUNDS,
    POLY_DEGREE,
    EngineJointMap,
    SimOptions,
    SimOut,
    SynthesizeOptions,
    evaluate_bernstein_torque,
    evaluate_polynomial_torque,
    get_pinocchio_canonical_joint_map,
    is_pinocchio_available,
    polynomial_torque_bounds,
    simulate_with_coefficients,
    synthesize_target_from_coefficients,
)

__all__ = [
    "CANONICAL_COORDINATE_NAMES",
    "CLUBHEAD_FRAME_NAME",
    "COEFFS_PER_JOINT",
    "DEFAULT_GOLFER_URDF",
    "EngineJointMap",
    "GRIP_FRAME_NAME",
    "POLY_BOUNDS",
    "POLY_DEGREE",
    "SimOptions",
    "SimOut",
    "SynthesizeOptions",
    "evaluate_bernstein_torque",
    "evaluate_polynomial_torque",
    "get_pinocchio_canonical_joint_map",
    "is_pinocchio_available",
    "polynomial_torque_bounds",
    "simulate_with_coefficients",
    "synthesize_target_from_coefficients",
]
