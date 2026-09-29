"""Backward-compatible engine import path for velocity conventions."""

from src.shared.python.engine_core.velocity_conventions import (
    ANGULAR_VELOCITY_UNIT,
    CANONICAL_FLOATING_BASE_CONVENTION,
    CANONICAL_GRAVITY_INERTIAL,
    CANONICAL_VELOCITY_REPRESENTATION,
    LINEAR_GRAVITY_UNIT,
    LINEAR_VELOCITY_UNIT,
    SPATIAL_VECTOR_SIZE,
    FloatingBaseConvention,
    SingleFloatingBodyDynamics,
    VelocityRepresentation,
    convert_floating_base_velocity,
    convert_gravity_vector,
    normalize_floating_base_velocity,
    single_floating_body_h_g,
)

__all__ = [
    "ANGULAR_VELOCITY_UNIT",
    "CANONICAL_FLOATING_BASE_CONVENTION",
    "CANONICAL_GRAVITY_INERTIAL",
    "CANONICAL_VELOCITY_REPRESENTATION",
    "LINEAR_GRAVITY_UNIT",
    "LINEAR_VELOCITY_UNIT",
    "SPATIAL_VECTOR_SIZE",
    "FloatingBaseConvention",
    "SingleFloatingBodyDynamics",
    "VelocityRepresentation",
    "convert_floating_base_velocity",
    "convert_gravity_vector",
    "normalize_floating_base_velocity",
    "single_floating_body_h_g",
]
