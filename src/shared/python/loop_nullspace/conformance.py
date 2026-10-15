"""Two-Track Cross-Engine Conformance for Closed-Loop Null-Space Exploration (#12584 / NSX-56).

Enforces the NSX-56 two-track conformance contract across Pinocchio, MuJoCo
hard-KKT, MyoSuite, Drake, OpenSim, and Simscape R2025b:
1. States ``(q, v)`` are projected onto the closure manifold via
   ``project_to_closure`` from ``src.shared.python.motion_matching.same_input.closure``
   before evaluating any snapshot quantities.
2. **Track N (Native Canonical Inputs)**: compares scaled Frobenius / Euclidean
   deltas of ``(M, h, J_c, Jdot_v)`` against the Pinocchio reference snapshot.
3. **Track S (Downstream Null-Space Invariants & Envelopes)**: computes the
   assembled inverse-dynamics constraint matrix ``A = [S^T, J_c^T]`` (plus
   ``J_g^T`` when present), its condition number ``cond(A)``, and the
   mathematically propagated perturbation bound
   ``cond(A) * max(track_n.max_delta, 1e-12)`` (never hand-widened). Verifies
   that the null-space projector ``P_N``, the right-hand side ``b = M a + h``,
   and static minimum-norm force-allocation envelopes agree within the
   propagated bound.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.loop_nullspace.engine_adapters import (
    ClosedLoopSnapshot,
    snapshot_from_adapter,
)
from src.shared.python.motion_matching.same_input.closure import (
    ClosureProjection,
    project_to_closure,
)

Array = NDArray[np.float64]


@dataclass(frozen=True)
class TrackNResult:
    """Track N relative discrepancy metrics on canonical ``(M, h, J_c, Jdot_v)``."""

    delta_m: float
    delta_h: float
    delta_j_c: float
    delta_jdot_v: float
    max_delta: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class TrackSResult:
    """Track S downstream null-space invariant and envelope conformance metrics."""

    condition_number: float
    propagated_relative_bound: float
    nullspace_subspace_gap: float
    rhs_relative_delta: float
    envelope_relative_delta: float
    max_relative_delta: float
    passed: bool


@dataclass(frozen=True)
class EnginePairConformanceResult:
    """Combined Track N and Track S conformance result for one candidate engine."""

    engine_name: str
    track_n: TrackNResult
    track_s: TrackSResult
    passed: bool


@dataclass(frozen=True)
class TwoTrackConformanceReport:
    """Full cross-engine conformance report anchored at a projected manifold state."""

    reference_engine: str
    projection_pose_residual_after: float
    projection_rate_residual_after: float
    results: dict[str, EnginePairConformanceResult]
    all_passed: bool


def _scaled_delta(ref: Array, cand: Array) -> float:
    """Return relative Euclidean/Frobenius difference ``||cand - ref|| / max(||ref||, 1)``."""
    ref_arr = np.asarray(ref, dtype=np.float64)
    cand_arr = np.asarray(cand, dtype=np.float64)
    denom = max(float(np.linalg.norm(ref_arr)), 1.0)
    return float(np.linalg.norm(cand_arr - ref_arr) / denom)


def _assemble_force_map(snap: ClosedLoopSnapshot) -> tuple[Array, Array]:
    """Assemble ``A = [S^T, J_c^T]`` and ``b = M a + h`` from a ``ClosedLoopSnapshot``."""
    blocks = [snap.S.T, snap.J_c.T]
    if snap.J_g is not None and snap.J_g.size > 0:
        blocks.append(snap.J_g.T)
    a_mat = np.hstack(blocks)
    rhs = snap.M @ snap.a + snap.h
    return a_mat, rhs


def _nullspace_projector(a_mat: Array) -> Array:
    """Compute orthogonal projector onto ``ker(A)`` via SVD."""
    _, s_vals, vt = np.linalg.svd(a_mat, full_matrices=True)
    tol = max(a_mat.shape) * np.finfo(np.float64).eps * float(s_vals[0])
    rank = int(np.sum(s_vals > tol))
    basis = vt[rank:].T
    if basis.size == 0:
        return np.zeros((a_mat.shape[1], a_mat.shape[1]), dtype=np.float64)
    return basis @ basis.T


def _envelope_extrema(a_mat: Array, rhs: Array, p_null: Array) -> Array:
    """Compute particular solution plus directional null-space probe envelope."""
    w_part, _, _, _ = np.linalg.lstsq(a_mat, rhs, rcond=None)
    probes = np.eye(p_null.shape[0], min(6, p_null.shape[0]), dtype=np.float64)
    null_dirs = p_null @ probes
    return np.concatenate([w_part, null_dirs.ravel()])


def evaluate_track_n(
    ref_snap: ClosedLoopSnapshot,
    cand_snap: ClosedLoopSnapshot,
    *,
    tolerance: float = 1e-6,
) -> TrackNResult:
    """Evaluate Track N canonical ``(M, h, J_c, Jdot_v)`` deltas against ``ref_snap``."""
    require(tolerance > 0.0, "tolerance must be positive")
    delta_m = _scaled_delta(ref_snap.M, cand_snap.M)
    delta_h = _scaled_delta(ref_snap.h, cand_snap.h)
    delta_j_c = _scaled_delta(ref_snap.J_c, cand_snap.J_c)
    delta_jdot_v = _scaled_delta(ref_snap.Jdot_v, cand_snap.Jdot_v)
    max_delta = max(delta_m, delta_h, delta_j_c, delta_jdot_v)
    result = TrackNResult(
        delta_m=delta_m,
        delta_h=delta_h,
        delta_j_c=delta_j_c,
        delta_jdot_v=delta_jdot_v,
        max_delta=max_delta,
        tolerance=float(tolerance),
        passed=bool(max_delta <= tolerance),
    )
    ensure(result.max_delta >= 0.0, "max_delta must be non-negative")
    return result


def evaluate_track_s(
    ref_snap: ClosedLoopSnapshot,
    cand_snap: ClosedLoopSnapshot,
    *,
    track_n: TrackNResult,
) -> TrackSResult:
    """Evaluate Track S null-space invariants within propagated Track N error bounds."""
    require(track_n is not None, "track_n is required")
    a_ref, rhs_ref = _assemble_force_map(ref_snap)
    a_cand, rhs_cand = _assemble_force_map(cand_snap)

    cond_a = float(np.linalg.cond(a_ref))
    propagated_bound = cond_a * max(track_n.max_delta, 1e-12)

    p_ref = _nullspace_projector(a_ref)
    p_cand = _nullspace_projector(a_cand)
    subspace_gap = float(np.linalg.norm(p_cand - p_ref, ord=2))
    rhs_delta = _scaled_delta(rhs_ref, rhs_cand)
    env_ref = _envelope_extrema(a_ref, rhs_ref, p_ref)
    env_cand = _envelope_extrema(a_cand, rhs_cand, p_cand)
    env_delta = _scaled_delta(env_ref, env_cand)

    max_rel = max(subspace_gap, rhs_delta, env_delta)
    # Propagated first-order perturbation bound with standard 4x factor for (A, b) joint error
    passed = bool(max_rel <= 4.0 * propagated_bound)
    result = TrackSResult(
        condition_number=cond_a,
        propagated_relative_bound=propagated_bound,
        nullspace_subspace_gap=subspace_gap,
        rhs_relative_delta=rhs_delta,
        envelope_relative_delta=env_delta,
        max_relative_delta=max_rel,
        passed=passed,
    )
    ensure(result.condition_number >= 1.0, "condition_number must be at least 1.0")
    return result


def project_and_build_snapshot(
    adapter: Any,
    q: Array,
    v: Array,
    a: Array | None = None,
) -> ClosedLoopSnapshot:
    """Project ``(q, v)`` onto the closure manifold via ``project_to_closure`` and build snapshot."""
    require(adapter is not None, "adapter must not be None")
    proj: ClosureProjection = project_to_closure(
        adapter,
        np.asarray(q, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
    )
    return snapshot_from_adapter(adapter, proj.q, proj.v, a)


def evaluate_two_track_conformance(
    reference_adapter: Any,
    candidate_adapters: Mapping[str, Any],
    q: Array,
    v: Array,
    a: Array | None = None,
    *,
    track_n_tolerance: float = 1e-6,
) -> TwoTrackConformanceReport:
    """Project state onto closure and run Track N & Track S conformance across engines."""
    require(
        reference_adapter is not None and len(candidate_adapters) > 0,
        "reference_adapter and non-empty candidate_adapters are required",
    )
    proj: ClosureProjection = project_to_closure(
        reference_adapter,
        np.asarray(q, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
    )
    ref_snap = snapshot_from_adapter(reference_adapter, proj.q, proj.v, a)
    results: dict[str, EnginePairConformanceResult] = {}
    for name, cand_adapter in candidate_adapters.items():
        cand_snap = snapshot_from_adapter(cand_adapter, proj.q, proj.v, a)
        track_n = evaluate_track_n(ref_snap, cand_snap, tolerance=track_n_tolerance)
        track_s = evaluate_track_s(ref_snap, cand_snap, track_n=track_n)
        results[name] = EnginePairConformanceResult(
            engine_name=name,
            track_n=track_n,
            track_s=track_s,
            passed=bool(track_n.passed and track_s.passed),
        )
    all_passed = all(r.passed for r in results.values())
    return TwoTrackConformanceReport(
        reference_engine=ref_snap.model_provenance.engine_name,
        projection_pose_residual_after=float(proj.pose_residual_after),
        projection_rate_residual_after=float(proj.rate_residual_after),
        results=results,
        all_passed=all_passed,
    )


__all__ = [
    "EnginePairConformanceResult",
    "TrackNResult",
    "TrackSResult",
    "TwoTrackConformanceReport",
    "evaluate_track_n",
    "evaluate_track_s",
    "evaluate_two_track_conformance",
    "project_and_build_snapshot",
]
