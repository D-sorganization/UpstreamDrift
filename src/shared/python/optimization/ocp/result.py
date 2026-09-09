"""Adapters from bioptim solutions back to UpstreamDrift result types.

Keeps every ``Solution`` accessor in one place so the OCP builders and the
backend registry never touch bioptim's solution API directly.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from src.shared.python.optimization._swing_kinematics import JOINTS
from src.shared.python.optimization._swing_models import ClubModel, GolferModel
from src.shared.python.optimization.casadi_backend import CasadiSwingResult
from src.shared.python.optimization.ocp._compat import bioptim_version
from src.shared.python.simulation_backends.provenance import ProvenanceStamp

__all__ = [
    "OcpSwingSolution",
    "bioptim_provenance",
    "solution_arrays",
    "solution_to_swing_result",
    "tracking_to_map_estimator_result",
]


@dataclass(frozen=True)
class OcpSwingSolution:
    """One solved swing OCP in plain arrays plus the flagship-shaped result.

    ``time`` is the node grid (``n_nodes``), ``q`` / ``qdot`` are
    ``n_joints x n_nodes``, ``tau`` is ``n_joints x (n_nodes - 1)``.
    """

    result: CasadiSwingResult
    time: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    tau: np.ndarray
    parameters: dict[str, float]
    clubhead_speed: float
    cost: float
    status: int
    iterations: int
    wall_time_s: float
    provenance: ProvenanceStamp | None = None

    @property
    def success(self) -> bool:
        return self.status == 0


def _hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def bioptim_provenance(
    golfer: GolferModel,
    club: ClubModel,
    solver_settings: Mapping[str, object],
    *,
    created_at: str | None = None,
) -> ProvenanceStamp:
    """Stamp a bioptim solve with model/parameter hashes and versions.

    ``git_commit`` is read from ``UD_GIT_COMMIT`` / ``GITHUB_SHA`` when set,
    else ``"unknown"`` (the stamp requires a non-empty string).
    """
    import casadi

    return ProvenanceStamp(
        engine="bioptim",
        engine_version=str(bioptim_version() or "unavailable"),
        model_hash=_hash(
            {"golfer": asdict(golfer), "club": asdict(club), "dofs": JOINTS}
        ),
        param_hash=_hash(dict(solver_settings)),
        git_commit=os.environ.get("UD_GIT_COMMIT")
        or os.environ.get("GITHUB_SHA")
        or "unknown",
        solver_settings={"casadi": str(casadi.__version__), **dict(solver_settings)},
        seed=None,
        created_at=created_at or datetime.now(UTC).isoformat(),
        convention="joint-space, JOINTS order, SI",
        frame="fixed-base swing rig (hip at origin)",
        units={"q": "rad", "qdot": "rad/s", "tau": "N*m", "time": "s"},
    )


def _to_shooting_nodes(values: np.ndarray, n_nodes: int) -> np.ndarray:
    """Subsample direct-collocation sub-points down to the shooting nodes.

    ``decision_states`` returns ``n_shooting * (degree + 1) + 1`` columns
    under ``OdeSolver.COLLOCATION`` and ``n_shooting + 1`` under RK4. The
    flagship decision layout lives on the shooting nodes, so keep every
    ``(degree + 1)``-th column.
    """
    n_columns = values.shape[1]
    if n_columns == n_nodes:
        return values
    stride, remainder = divmod(n_columns - 1, n_nodes - 1)
    if remainder or stride < 1:
        raise ValueError(
            f"cannot map {n_columns} solution columns onto {n_nodes} shooting nodes"
        )
    return values[:, ::stride]


def solution_arrays(
    sol: Any, bioptim: Any, *, n_nodes: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(q, qdot, tau)`` on the shooting-node grid from a ``Solution``.

    Args:
        sol: The solved bioptim ``Solution``.
        bioptim: The bioptim module (avoids a second lazy import).
        n_nodes: Expected number of shooting nodes. Required to collapse a
            direct-collocation solution; RK4 solutions need no hint.
    """
    merge = bioptim.SolutionMerge.NODES
    states = sol.decision_states(to_merge=merge)
    controls = sol.decision_controls(to_merge=merge)
    q = np.asarray(states["q"], dtype=float)
    qdot = np.asarray(states["qdot"], dtype=float)
    tau = np.asarray(controls["tau"], dtype=float)
    if n_nodes is not None:
        q = _to_shooting_nodes(q, n_nodes)
        qdot = _to_shooting_nodes(qdot, n_nodes)
    n_columns = q.shape[1]
    # bioptim may report a control on the terminal node (constant-with-last
    # control type); the flagship layout wants one torque per interval.
    if tau.shape[1] == n_columns:
        tau = tau[:, : n_columns - 1]
    return q, qdot, tau


def solution_to_swing_result(
    sol: Any,
    bioptim: Any,
    *,
    transcription: str,
    x_fallback: np.ndarray,
    n_nodes: int | None = None,
) -> tuple[CasadiSwingResult, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a ``Solution`` into the flagship-layout result.

    Returns ``(result, q, qdot, tau)``. A failed solve keeps ``x_fallback``
    as the decision vector, matching :func:`casadi_backend.solve_swing_casadi`.
    """
    status = int(sol.status)
    q, qdot, tau = solution_arrays(sol, bioptim, n_nodes=n_nodes)
    success = status == 0
    x = (
        np.concatenate([q.flatten(), qdot.flatten()])
        if success
        else np.asarray(x_fallback)
    )
    result = CasadiSwingResult(
        success=success,
        x=x,
        fun=float(np.asarray(sol.cost).ravel()[0]) if success else float("nan"),
        message=f"bioptim IPOPT status {status}",
        iterations=int(getattr(sol, "iterations", 0) or 0),
        torques=tau if success else None,
        transcription=transcription,
    )
    return result, q, qdot, tau


def tracking_to_map_estimator_result(result: Any) -> Any:
    """Convert a :class:`TrackingResult` into a :class:`MapEstimatorResult`.

    Adapts optimal-estimation tracking solves back to the canonical MAP
    estimator result contract.
    """
    from src.shared.python.estimation.map_estimator import MapEstimatorResult

    coefficients = np.concatenate([result.q.flatten(), result.qdot.flatten()])
    residual_values = [val for val in result.marker_rms_m.values() if not np.isnan(val)]
    residual = (
        np.array(residual_values, dtype=float) if residual_values else np.zeros(0)
    )
    gate_report = getattr(result, "gate_report", None)
    return MapEstimatorResult(
        success=bool(result.success),
        coefficients=coefficients,
        parameters=dict(result.parameters),
        residual=residual,
        objective=float(result.cost),
        n_iterations=int(result.iterations),
        message=f"bioptim IPOPT status {result.status}",
        provenance=result.provenance,
        n_non_finite_evaluations=0,
        identifiability=gate_report,
        locked_by_gate=tuple(result.locked_by_gate),
    )
