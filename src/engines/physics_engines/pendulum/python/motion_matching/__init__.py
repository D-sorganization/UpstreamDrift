"""Pendulum motion matching package."""

from __future__ import annotations

from .adapters import (
    MODEL_ID_ANALYTICAL,
    MODEL_ID_TOOLS,
    DoublePendulumAdapter,
    check_dynamics_parity,
    forward_kinematics_2d,
    params_analytical_to_tools,
    params_tools_to_analytical,
)
from .club_match_matrix import (
    MATCH_SCHEMA,
    PENDULUM_MATCH_MODELS,
    MatchMatrixOutcome,
    PendulumMatchMatrix,
    build_pendulum_match_matrix,
    evidence_payload as pendulum_match_evidence_payload,
)
from .club_pendulum_match import match_club_pendulum
from .provider import PendulumFitSwingProvider
from .torque_optimization import (
    COEFFS_PER_JOINT,
    BernsteinTorqueProfile,
    fit_bounded_double_pendulum,
    integrate_double_pendulum_rollout,
)

__all__ = [
    "COEFFS_PER_JOINT",
    "MATCH_SCHEMA",
    "MODEL_ID_ANALYTICAL",
    "MODEL_ID_TOOLS",
    "PENDULUM_MATCH_MODELS",
    "BernsteinTorqueProfile",
    "DoublePendulumAdapter",
    "MatchMatrixOutcome",
    "PendulumFitSwingProvider",
    "PendulumMatchMatrix",
    "build_pendulum_match_matrix",
    "check_dynamics_parity",
    "fit_bounded_double_pendulum",
    "forward_kinematics_2d",
    "integrate_double_pendulum_rollout",
    "match_club_pendulum",
    "params_analytical_to_tools",
    "params_tools_to_analytical",
    "pendulum_match_evidence_payload",
]
