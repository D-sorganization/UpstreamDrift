"""Kinematic ensemble over admitted closed-loop trajectory samples (NSX-61, #12586).

Propagates kinematic uncertainty by generating seeded perturbations around a
reference trajectory, reprojecting each candidate onto the dual-grip closure
manifold via ``project_to_closure`` / ``reproject_onto_closure``, gating each
sample through the position and velocity closure tolerances, and computing the
envelope union and intersection across admitted samples with explicit
hypothesis labels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Sequence

import numpy as np

from src.shared.python.core.contracts import ensure, require
from src.shared.python.loop_nullspace.lower_body_loop import (
    FloatArray,
    readonly_array,
    solve_nullspace_envelopes,
)
from src.shared.python.motion_matching.club_only.nullspace_proposals import (
    reproject_onto_closure,
)
from src.shared.python.motion_matching.force_nullspace import ForceNullSpace
from src.shared.python.motion_matching.same_input.closure import project_to_closure

SAMPLING_HYPOTHESIS_ID = "H-KINEMATIC-GAUSSIAN-PERTURBATION-ENSEMBLE"
CLOSURE_ADMISSION_HYPOTHESIS_ID = "H-CLOSURE-MANIFOLD-ADMISSION"


@dataclass(frozen=True)
class AdmittedTrajectorySample:
    """One projected and admitted closed-loop trajectory sample and its static envelope."""

    sample_index: int
    snapshots: tuple[Any, ...]
    q_trajectory: FloatArray
    v_trajectory: FloatArray
    max_pose_residual: float
    max_rate_residual: float
    envelope_lower: FloatArray
    envelope_upper: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshots", tuple(self.snapshots))
        object.__setattr__(
            self, "q_trajectory", readonly_array(self.q_trajectory, "q_trajectory", 2)
        )
        object.__setattr__(
            self, "v_trajectory", readonly_array(self.v_trajectory, "v_trajectory", 2)
        )
        object.__setattr__(
            self,
            "envelope_lower",
            readonly_array(self.envelope_lower, "envelope_lower", 2),
        )
        object.__setattr__(
            self,
            "envelope_upper",
            readonly_array(self.envelope_upper, "envelope_upper", 2),
        )


@dataclass(frozen=True)
class EnsembleEnvelopeBand:
    """Union or intersection envelope band across admitted trajectory samples."""

    lower: FloatArray
    upper: FloatArray
    kind: Literal["UNION", "INTERSECTION"]
    hypothesis_label: str
    hypotheses: tuple[str, ...]
    is_empty: bool

    def __post_init__(self) -> None:
        if self.kind not in ("UNION", "INTERSECTION"):
            raise ValueError(f"Unsupported envelope band kind: {self.kind!r}")
        object.__setattr__(self, "lower", readonly_array(self.lower, "lower", 2))
        object.__setattr__(self, "upper", readonly_array(self.upper, "upper", 2))
        object.__setattr__(self, "hypotheses", tuple(str(h) for h in self.hypotheses))


@dataclass(frozen=True)
class KinematicEnsembleReport:
    """Complete kinematic ensemble report with admitted samples and envelope bands."""

    admitted_samples: tuple[AdmittedTrajectorySample, ...]
    n_requested: int
    n_admitted: int
    rejected_count: int
    seed: int
    perturbation_sigma: float
    closure_tol: float
    envelope_union: EnsembleEnvelopeBand
    envelope_intersection: EnsembleEnvelopeBand
    sampling_hypothesis: str = SAMPLING_HYPOTHESIS_ID


@dataclass(frozen=True)
class _SnapshotLinearClosurePlant:
    """Linearized ClosurePlant adapter around a snapshot's closure Jacobian J_c."""

    q_ref: FloatArray
    j_c: FloatArray

    def closure_pose_residual(self, q: FloatArray) -> FloatArray:
        return np.asarray(self.j_c @ (q - self.q_ref), dtype=np.float64)

    def closure_rate_matrix(self, q: FloatArray) -> FloatArray:
        _ = q
        return self.j_c


def _project_frame_state(
    snap: Any,
    q_pert: FloatArray,
    v_pert: FloatArray,
    closure_tol: float,
    project_fn: Callable[..., Any] | None,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Project one perturbed frame onto closure and return (q_proj, v_proj, pose_res, rate_res)."""
    j_c = readonly_array(snap.J_c, "J_c", 2)
    q_ref = readonly_array(snap.q, "q_ref", 1)

    if project_fn is not None:
        raw_out = project_fn(snap, q_pert, v_pert)
        if len(raw_out) == 4:
            q_p, v_p, p_res, r_res = raw_out
            return (
                readonly_array(q_p, "q_proj", 1),
                readonly_array(v_p, "v_proj", 1),
                float(p_res),
                float(r_res),
            )
        q_p, v_p = raw_out
        q_arr = readonly_array(q_p, "q_proj", 1)
        v_arr = readonly_array(v_p, "v_proj", 1)
        p_res = float(np.max(np.abs(j_c @ (q_arr - q_ref))))
        r_res = float(np.max(np.abs(j_c @ v_arr)))
        return q_arr, v_arr, p_res, r_res

    plant = _SnapshotLinearClosurePlant(q_ref=q_ref, j_c=j_c)
    target_tol = min(float(closure_tol) * 0.1, 1e-10)
    q_reproj = reproject_onto_closure(
        q_pert,
        closure_residual_m=float(np.max(np.abs(plant.closure_pose_residual(q_pert)))),
        tol_m=target_tol,
        plant=plant,
        v=v_pert,
    )
    proj = project_to_closure(plant, q_reproj, v_pert, tolerance=target_tol)
    dq_null = proj.q - q_ref
    curvature_drift = 0.5 * float(np.dot(dq_null, dq_null))
    pose_res = max(float(proj.pose_residual_after), curvature_drift)
    rate_res = float(proj.rate_residual_after)
    return (
        readonly_array(proj.q, "q_proj", 1),
        readonly_array(proj.v, "v_proj", 1),
        pose_res,
        rate_res,
    )


def _frame_static_envelope(
    snap: Any,
    q_proj: FloatArray,
    v_proj: FloatArray,
    torque_bound: float = 200.0,
) -> tuple[FloatArray, FloatArray]:
    """Compute static torque + closure multiplier bounds for one projected frame."""
    S = readonly_array(snap.S, "S", 2)
    J_c = readonly_array(snap.J_c, "J_c", 2)
    j_g_raw = getattr(snap, "J_g", None)
    b_base = readonly_array(snap.generalized_load(), "b", 1)
    dq = q_proj - readonly_array(snap.q, "q", 1)
    dv = v_proj - readonly_array(snap.v, "v", 1)
    m_mat = readonly_array(snap.M, "M", 2)
    b_pert = b_base + 0.5 * (m_mat @ dq) + 0.2 * (m_mat @ dv)

    n_tau = int(S.shape[0])
    m_c = int(J_c.shape[0])
    n_q = n_tau + m_c
    if j_g_raw is not None:
        J_g = readonly_array(j_g_raw, "J_g", 2)
        a_mat = np.hstack([S.T, J_c.T, J_g.T])
    else:
        a_mat = np.hstack([S.T, J_c.T])

    space = ForceNullSpace.from_balance(a_mat, b_pert)
    dim_x = int(a_mat.shape[1])
    rows: list[FloatArray] = []
    rhs: list[float] = []
    for idx in range(dim_x):
        row = np.zeros(dim_x, dtype=np.float64)
        row[idx] = 1.0
        bound = torque_bound if idx < n_tau else torque_bound * 3.0
        rows.extend([row, -row])
        rhs.extend([bound, bound])

    g_x = np.asarray(rows, dtype=np.float64)
    h_x = np.asarray(rhs, dtype=np.float64)
    lo, hi, _signs = solve_nullspace_envelopes(
        space.particular, space.basis, g_x, h_x, n_q
    )
    return lo, hi


def _aggregate_envelopes(
    admitted: Sequence[AdmittedTrajectorySample],
    n_frames: int,
    n_quantities: int,
) -> tuple[EnsembleEnvelopeBand, EnsembleEnvelopeBand]:
    """Compute union and intersection envelope bands across admitted samples."""
    hyps = (SAMPLING_HYPOTHESIS_ID, CLOSURE_ADMISSION_HYPOTHESIS_ID)
    if not admitted:
        zeros = np.zeros((n_frames, n_quantities), dtype=np.float64)
        empty_union = EnsembleEnvelopeBand(
            lower=zeros,
            upper=zeros,
            kind="UNION",
            hypothesis_label="KINEMATIC_ENSEMBLE_UNION_EMPTY",
            hypotheses=hyps,
            is_empty=True,
        )
        empty_inter = EnsembleEnvelopeBand(
            lower=zeros,
            upper=zeros,
            kind="INTERSECTION",
            hypothesis_label="KINEMATIC_ENSEMBLE_INTERSECTION_EMPTY",
            hypotheses=hyps,
            is_empty=True,
        )
        return empty_union, empty_inter

    lowers = np.stack([s.envelope_lower for s in admitted], axis=0)
    uppers = np.stack([s.envelope_upper for s in admitted], axis=0)
    union_lo = np.min(lowers, axis=0)
    union_hi = np.max(uppers, axis=0)
    inter_lo = np.max(lowers, axis=0)
    inter_hi = np.min(uppers, axis=0)
    inter_empty = bool(np.any(inter_lo > inter_hi + 1e-12))

    union_band = EnsembleEnvelopeBand(
        lower=union_lo,
        upper=union_hi,
        kind="UNION",
        hypothesis_label="KINEMATIC_ENSEMBLE_UNION",
        hypotheses=hyps,
        is_empty=False,
    )
    inter_band = EnsembleEnvelopeBand(
        lower=inter_lo,
        upper=inter_hi,
        kind="INTERSECTION",
        hypothesis_label="KINEMATIC_ENSEMBLE_INTERSECTION",
        hypotheses=hyps,
        is_empty=inter_empty,
    )
    return union_band, inter_band


def _sample_trajectory_candidate(
    snapshots: Sequence[Any],
    sample_idx: int,
    rng: np.random.Generator,
    perturbation_sigma: float,
    closure_tol: float,
    project_fn: Callable[..., Any] | None,
) -> AdmittedTrajectorySample | None:
    """Sample and project one candidate trajectory across all frames."""
    q_frames: list[FloatArray] = []
    v_frames: list[FloatArray] = []
    lo_frames: list[FloatArray] = []
    hi_frames: list[FloatArray] = []
    snap_frames: list[Any] = []
    max_pose = 0.0
    max_rate = 0.0

    for snap in snapshots:
        q_base = readonly_array(snap.q, "q", 1)
        v_base = readonly_array(snap.v, "v", 1)
        q_pert = q_base + rng.normal(0.0, float(perturbation_sigma), size=q_base.shape)
        v_pert = v_base + rng.normal(0.0, float(perturbation_sigma), size=v_base.shape)
        try:
            q_proj, v_proj, pose_res, rate_res = _project_frame_state(
                snap, q_pert, v_pert, float(closure_tol), project_fn
            )
        except (ArithmeticError, ValueError):
            return None

        max_pose = max(max_pose, pose_res)
        max_rate = max(max_rate, rate_res)
        if pose_res > closure_tol or rate_res > closure_tol:
            return None

        lo_f, hi_f = _frame_static_envelope(snap, q_proj, v_proj)
        q_frames.append(q_proj)
        v_frames.append(v_proj)
        lo_frames.append(lo_f)
        hi_frames.append(hi_f)
        try:
            snap_frames.append(replace(snap, q=q_proj, v=v_proj))
        except TypeError:
            snap_frames.append(snap)

    return AdmittedTrajectorySample(
        sample_index=sample_idx,
        snapshots=tuple(snap_frames),
        q_trajectory=np.stack(q_frames, axis=0),
        v_trajectory=np.stack(v_frames, axis=0),
        max_pose_residual=max_pose,
        max_rate_residual=max_rate,
        envelope_lower=np.stack(lo_frames, axis=0),
        envelope_upper=np.stack(hi_frames, axis=0),
    )


def kinematic_ensemble(
    base_snapshots: Sequence[Any],
    n: int,
    seed: int,
    *,
    perturbation_sigma: float = 1e-3,
    closure_tol: float = 1e-6,
    project_fn: Callable[..., Any] | None = None,
) -> KinematicEnsembleReport:
    """Generate a seeded kinematic ensemble projected onto the closure manifold."""
    snapshots = tuple(base_snapshots)
    require(len(snapshots) >= 1, "base_snapshots must be non-empty")
    require(int(n) >= 1, "n must be >= 1")
    require(
        perturbation_sigma >= 0.0 and np.isfinite(perturbation_sigma),
        "perturbation_sigma must be finite and >= 0",
    )
    require(
        closure_tol > 0.0 and np.isfinite(closure_tol),
        "closure_tol must be finite and > 0",
    )

    rng = np.random.default_rng(int(seed))
    n_frames = len(snapshots)
    s0 = readonly_array(snapshots[0].S, "S0", 2)
    jc0 = readonly_array(snapshots[0].J_c, "J_c0", 2)
    n_quantities = int(s0.shape[0] + jc0.shape[0])

    admitted: list[AdmittedTrajectorySample] = []
    rejected_count = 0

    for sample_idx in range(int(n)):
        sample = _sample_trajectory_candidate(
            snapshots,
            sample_idx,
            rng,
            float(perturbation_sigma),
            float(closure_tol),
            project_fn,
        )
        if sample is None:
            rejected_count += 1
        else:
            admitted.append(sample)

    union_band, inter_band = _aggregate_envelopes(admitted, n_frames, n_quantities)
    ensure(
        len(admitted) + rejected_count == int(n),
        "admitted + rejected_count must equal requested n",
    )
    return KinematicEnsembleReport(
        admitted_samples=tuple(admitted),
        n_requested=int(n),
        n_admitted=len(admitted),
        rejected_count=rejected_count,
        seed=int(seed),
        perturbation_sigma=float(perturbation_sigma),
        closure_tol=float(closure_tol),
        envelope_union=union_band,
        envelope_intersection=inter_band,
        sampling_hypothesis=SAMPLING_HYPOTHESIS_ID,
    )


__all__ = [
    "CLOSURE_ADMISSION_HYPOTHESIS_ID",
    "SAMPLING_HYPOTHESIS_ID",
    "AdmittedTrajectorySample",
    "EnsembleEnvelopeBand",
    "KinematicEnsembleReport",
    "kinematic_ensemble",
]
