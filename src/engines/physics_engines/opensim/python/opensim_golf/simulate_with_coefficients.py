"""Compatibility facade for the canonical OpenSim forward simulator.

This module used to carry a second implementation of the OpenSim
polynomial-torque simulator. That copy resolved grip and clubhead poses
from body origins and could drift from the maintained motion-matching
implementation. Keep this import surface as a shim only; simulator logic
and pose extraction live in ``motion_matching.simulate`` and
``opensim_golf.fk``.
"""

from __future__ import annotations

from src.engines.physics_engines.opensim.python.motion_matching.simulate import (
    COEFFS_PER_JOINT,
    CLUBHEAD_FRAME_NAME,
    GRIP_FRAME_NAME,
    POLY_DEGREE,
    SimOptions,
    SimOut,
    evaluate_polynomial_torque,
    extract_full_pose,
    simulate_with_coefficients,
)
from src.engines.physics_engines.opensim.python.motion_matching.synthesize import (
    SynthOptions,
    synthesize_target_from_coefficients,
)

__all__ = [
    "COEFFS_PER_JOINT",
    "CLUBHEAD_FRAME_NAME",
    "GRIP_FRAME_NAME",
    "POLY_DEGREE",
    "SimOptions",
    "SimOut",
    "SynthOptions",
    "evaluate_polynomial_torque",
    "extract_full_pose",
    "simulate_with_coefficients",
    "synthesize_target_from_coefficients",
]
