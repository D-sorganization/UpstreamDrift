"""Headless ZTCF/ZVCF golden evaluation for Agent API (#15021)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from src.shared.python.simulation_backends import (
    EquationsOfMotionProvider,
    GolfModelParams,
    make_backend,
)
from src.shared.python.simulation_backends.ztcf_zvcf import (
    CanonicalDynamicsTrajectory,
    evaluate_ztcf_zvcf_on_canonical_trajectory,
)

ZVCF_DEFINITION_TEXT = (
    "ZVCF is the control component at a fixed state: qddot = solve(M(q), tau), "
    "equivalently M^-1 tau on unconstrained plants; constrained plants carry "
    "M^-1 (tau + J^T (lambda - lambda_0)) per #14515."
)

GOLDEN_TRAJECTORY = CanonicalDynamicsTrajectory(
    t=np.array([0.0, 0.01, 0.02], dtype=float),
    q=np.array([[0.1, -0.2], [0.15, -0.25], [0.2, -0.3]], dtype=float),
    v=np.array([[0.4, -0.1], [0.3, 0.05], [0.2, 0.15]], dtype=float),
    tau=np.array([[1.0, -0.2], [0.5, 0.25], [0.1, 0.4]], dtype=float),
    meta={"golden_case": "agent_api_ztcf_v1", "backend": "ode"},
)


def qualification_banner() -> dict[str, Any]:
    """Qualification banner: ZVCF definition plus model limits for agents."""
    return {
        "zvcf_definition": ZVCF_DEFINITION_TEXT,
        "qualification_state": {
            "backend": "ode",
            "model": "planar_double_pendulum",
            "pointwise_only": True,
            "trajectory_sum_exact": False,
        },
        "model_limits": [
            "Pointwise ZTCF/ZVCF split; not additive over integrated trajectories.",
            "Golden case uses the ODE reference backend (2-DOF planar golfer).",
        ],
    }


def _arrays_to_lists(result: Any) -> dict[str, Any]:
    return {
        "t": np.asarray(result.t, dtype=float).tolist(),
        "ztcf_acceleration": np.asarray(result.ztcf_acceleration, dtype=float).tolist(),
        "zvcf_acceleration": np.asarray(result.zvcf_acceleration, dtype=float).tolist(),
        "drift_acceleration": np.asarray(
            result.drift_acceleration, dtype=float
        ).tolist(),
        "control_acceleration": np.asarray(
            result.control_acceleration, dtype=float
        ).tolist(),
        "convention": result.convention,
        "frame": result.frame,
        "units": result.units,
    }


def execute_golden_ztcf(
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the pinned golden canonical trajectory through sim_run (ODE backend)."""
    del request  # golden inputs are fixed; hash covers the canonical case id only
    provider = cast(
        EquationsOfMotionProvider,
        make_backend("ode", GolfModelParams.default()),
    )
    evaluated = evaluate_ztcf_zvcf_on_canonical_trajectory(provider, GOLDEN_TRAJECTORY)
    return {
        "capability": "counterfactual_ztcf_golden",
        "golden_case_id": "agent_api_ztcf_v1",
        "qualification_banner": qualification_banner(),
        "analysis": _arrays_to_lists(evaluated),
    }
