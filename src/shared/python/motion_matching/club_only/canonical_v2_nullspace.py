"""Shared null-space body strategies and joint observability (#15787)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.club_only.nullspace_proposals import (
    NullSpaceAnalysis,
    analyze_grip_jacobian_nullspace,
)

_TORSO = ("torso", "spine", "lumbar", "thorax", "chest", "trunk", "pelvis")
_ARM = ("arm", "shoulder", "elbow", "wrist", "forearm", "hand", "ulnar", "radial")


@runtime_checkable
class CandidateGateEvaluator(Protocol):
    """Evaluates tracking, dynamics, and replay gates for a candidate."""

    def evaluate(
        self,
        q_cand: NDArray[np.float64],
        dq: NDArray[np.float64],
        strategy: str,
    ) -> tuple[bool, tuple[str, ...]]:
        """Return ``(passed, reasons)`` for tracking, dynamics, and replay."""


def _arr(
    val: object, name: str, ndim: int, n_cols: int | None = None
) -> NDArray[np.float64]:
    """Convert to finite float64 ndarray and check shape."""
    arr = np.asarray(val, dtype=np.float64)
    if arr.ndim != ndim or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite {ndim}-D")
    if n_cols is not None and arr.shape[1] != n_cols:
        raise ValueError(f"{name} cols {arr.shape[1]} != {n_cols}")
    return arr.copy()


@dataclass(frozen=True)
class SharedJacobianSpec:
    """Stacked club, grip, and active foot-contact Jacobian specification."""

    j_club: NDArray[np.float64]
    j_grip: NDArray[np.float64]
    j_foot: NDArray[np.float64]
    joint_names: tuple[str, ...]
    singular_tol: float = 1.0e-8

    def __post_init__(self) -> None:
        """Validate finite 2-D Jacobian blocks, joint names, and tolerance."""
        jc = _arr(self.j_club, "j_club", 2)
        nq = int(jc.shape[1])
        jg, jf = _arr(self.j_grip, "j_grip", 2, nq), _arr(self.j_foot, "j_foot", 2, nq)
        names = tuple(str(n) for n in self.joint_names)
        if nq < 1 or jc.shape[0] + jg.shape[0] + jf.shape[0] < 1:
            raise ValueError("stacked shared Jacobian must be non-empty")
        if len(names) != nq or any(not n for n in names) or len(set(names)) != nq:
            raise ValueError("joint_names must have nq unique non-empty names")
        if not np.isfinite(self.singular_tol) or self.singular_tol <= 0.0:
            raise ValueError("singular_tol must be finite and > 0")
        pairs = (("j_club", jc), ("j_grip", jg), ("j_foot", jf), ("joint_names", names))
        for key, val in pairs:
            object.__setattr__(self, key, val)

    def stacked(self) -> NDArray[np.float64]:
        """Return ``J_shared = np.vstack([J_club, J_grip, J_foot])``."""
        return np.vstack([self.j_club, self.j_grip, self.j_foot])


@dataclass(frozen=True)
class JointObservabilityEntry:
    """Per-joint observability split between row space and null space."""

    joint_name: str
    joint_index: int
    observability_score: float
    row_space_norm: float
    null_space_norm: float

    def __post_init__(self) -> None:
        """Validate observability score in [0, 1] and finite norms."""
        lim = 1.0 + 1.0e-12
        norms_ok = (
            0.0 <= self.row_space_norm <= lim and 0.0 <= self.null_space_norm <= lim
        )
        if (
            not self.joint_name
            or self.joint_index < 0
            or not (0.0 <= self.observability_score <= 1.0)
            or not norms_ok
        ):
            raise ValueError("invalid joint observability entry")


@dataclass(frozen=True)
class JointObservabilityReport:
    """Observability report across all joints of a stacked shared Jacobian."""

    rank: int
    null_dim: int
    n_joints: int
    singular_values: NDArray[np.float64]
    entries: tuple[JointObservabilityEntry, ...]


@dataclass(frozen=True)
class StrategyCandidate:
    """One gated null-space body strategy perturbation."""

    candidate_id: str
    strategy: str
    amplitude: float
    delta_q: NDArray[np.float64]
    q_candidate: NDArray[np.float64]
    constraint_residual_norm: float
    passed: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate candidate gate state and residual norm."""
        res = self.constraint_residual_norm
        if not np.isfinite(res) or res < 0.0:
            raise ValueError("constraint_residual_norm must be finite and >= 0")
        if (self.passed and self.rejection_reasons) or (
            not self.passed and not self.rejection_reasons
        ):
            raise ValueError("passed flag must match emptiness of rejection_reasons")


@dataclass(frozen=True)
class StrategyEnvelope:
    """Bounded joint envelope computed strictly over passing candidates."""

    strategy: str
    passing_count: int
    rejected_count: int
    min_q: NDArray[np.float64] | None
    max_q: NDArray[np.float64] | None
    min_delta_q: NDArray[np.float64] | None
    max_delta_q: NDArray[np.float64] | None

    def __post_init__(self) -> None:
        """Validate consistency between passing_count and envelope bounds."""
        arrays = (self.min_q, self.max_q, self.min_delta_q, self.max_delta_q)
        if self.passing_count == 0 and any(a is not None for a in arrays):
            raise ValueError("bounds must be None when passing_count == 0")
        if self.passing_count > 0 and any(a is None for a in arrays):
            raise ValueError("bounds required when passing_count > 0")


@dataclass(frozen=True)
class NullspaceStrategyReport:
    """Complete shared null-space strategy & observability report."""

    observability: JointObservabilityReport
    envelopes: dict[str, StrategyEnvelope]
    candidates: tuple[StrategyCandidate, ...]
    passing_candidates: tuple[StrategyCandidate, ...]
    rejected_candidates: tuple[StrategyCandidate, ...]
    constraint_tol: float


def _obs_from_analysis(
    names: tuple[str, ...], analysis: NullSpaceAnalysis
) -> JointObservabilityReport:
    """Compute per-joint observability entries from an SVD null-space analysis."""
    basis = analysis.basis
    n_null = int(basis.shape[1])
    entries: list[JointObservabilityEntry] = []
    for idx, name in enumerate(names):
        n_norm = float(np.linalg.norm(basis[idx, :])) if n_null > 0 else 0.0
        n_sq = min(1.0, max(0.0, n_norm * n_norm))
        r_sq = min(1.0, max(0.0, 1.0 - n_sq))
        rn, nn = float(np.sqrt(r_sq)), float(np.sqrt(n_sq))
        entries.append(JointObservabilityEntry(name, idx, r_sq, rn, nn))
    sv = analysis.singular_values
    return JointObservabilityReport(
        analysis.rank, n_null, len(names), sv, tuple(entries)
    )


def compute_joint_observability(
    j_club: NDArray[np.floating],
    j_grip: NDArray[np.floating],
    j_foot: NDArray[np.floating],
    joint_names: Sequence[str],
    *,
    singular_tol: float = 1.0e-8,
) -> JointObservabilityReport:
    """Compute per-joint SVD row-space observability and null-space fraction."""
    jc, jg, jf = (
        np.asarray(j_club, float),
        np.asarray(j_grip, float),
        np.asarray(j_foot, float),
    )
    spec = SharedJacobianSpec(jc, jg, jf, tuple(joint_names), singular_tol)
    return _obs_from_analysis(
        spec.joint_names,
        analyze_grip_jacobian_nullspace(spec.stacked(), singular_tol=singular_tol),
    )


def _infer_weights(names: tuple[str, ...]) -> dict[str, NDArray[np.float64]]:
    """Build default DOF weight vectors for ``torso_led`` and ``arm_wrist_led``."""
    nq = len(names)
    t = np.array([float(any(k in n.lower() for k in _TORSO)) for n in names])
    a = np.array([float(any(k in n.lower() for k in _ARM)) for n in names])
    if float(np.sum(t)) == 0.0 or float(np.sum(a)) == 0.0:
        half = max(1, nq // 2)
        t, a = np.zeros(nq), np.zeros(nq)
        t[:half], a[half:] = 1.0, 1.0
    return {"torso_led": t, "arm_wrist_led": a}


def _weighted_dir(
    basis: NDArray[np.float64], w: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """Compute unit direction in ``ker(J_shared)`` maximizing weighted energy."""
    _u, s_vals, vh = np.linalg.svd(w[:, None] * basis, full_matrices=False)
    if s_vals.size == 0 or float(s_vals[0]) <= 1.0e-12:
        return None
    d = np.asarray(basis @ vh[0, :], dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm <= 1.0e-12:
        return None
    d /= norm
    lead = int(np.argmax(np.abs(w * d)))
    return -d if (w * d)[lead] < 0.0 else d


def _scale_bounds(
    q_nom: NDArray[np.float64],
    delta_q: NDArray[np.float64],
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
) -> NDArray[np.float64]:
    """Uniformly scale ``delta_q`` to keep ``q_nom + delta_q`` within bounds."""
    if bounds is None:
        return delta_q
    s = 1.0
    for q0, dq, lo, hi in zip(q_nom, delta_q, bounds[0], bounds[1], strict=True):
        if dq > 1.0e-15:
            s = min(s, max(0.0, float((hi - q0) / dq)))
        elif dq < -1.0e-15:
            s = min(s, max(0.0, float((lo - q0) / dq)))
    return s * delta_q


def _build_envelope(
    strategy: str, cands: Sequence[StrategyCandidate]
) -> StrategyEnvelope:
    """Aggregate min/max joint bounds strictly across passing candidates."""
    ok = [c for c in cands if c.passed]
    rej = len(cands) - len(ok)
    if not ok:
        return StrategyEnvelope(strategy, 0, rej, None, None, None, None)
    qs, dqs = np.vstack([c.q_candidate for c in ok]), np.vstack([c.delta_q for c in ok])
    return StrategyEnvelope(
        strategy, len(ok), rej, qs.min(0), qs.max(0), dqs.min(0), dqs.max(0)
    )


def _qualify_strategy(
    strategy: str,
    direction: NDArray[np.float64] | None,
    q_nom: NDArray[np.float64],
    j_shared: NDArray[np.float64],
    gate: CandidateGateEvaluator,
    amplitudes: tuple[float, ...],
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    tol: float,
) -> list[StrategyCandidate]:
    """Generate and gate all amplitude candidates for a single strategy."""
    cands: list[StrategyCandidate] = []
    for idx, amp in enumerate(amplitudes):
        if direction is None:
            dq, qc, res, passed = np.zeros_like(q_nom), q_nom.copy(), 0.0, False
            reasons: tuple[str, ...] = ("strategy_fully_constrained_in_nullspace",)
        else:
            dq = _scale_bounds(q_nom, amp * direction, bounds)
            qc = q_nom + dq
            res = float(np.linalg.norm(j_shared @ dq))
            gate_ok, gate_reasons = gate.evaluate(qc, dq, strategy)
            rlist = list(gate_reasons)
            if res > tol:
                rlist.append("shared_constraint_tolerance_exceeded")
            if not gate_ok and not rlist:
                rlist.append("gate_rejected")
            passed, reasons = bool(gate_ok) and not rlist, tuple(rlist)
        cid = f"{strategy}:{idx}"
        cands.append(
            StrategyCandidate(cid, strategy, amp, dq, qc, res, passed, reasons)
        )
    return cands


def generate_and_qualify_nullspace_strategies(
    *,
    spec: SharedJacobianSpec,
    q_nominal: NDArray[np.floating],
    gate_evaluator: CandidateGateEvaluator,
    amplitudes: Sequence[float] = (-0.05, 0.05),
    strategy_weights: Mapping[str, Sequence[float]] | None = None,
    q_bounds: tuple[NDArray[np.floating], NDArray[np.floating]] | None = None,
    constraint_tol: float = 1.0e-8,
) -> NullspaceStrategyReport:
    """Compute ``ker(J_shared)``, generate gated body strategies, and report."""
    if not np.isfinite(constraint_tol) or constraint_tol <= 0.0:
        raise ValueError("constraint_tol must be finite and > 0")
    amps = tuple(float(a) for a in amplitudes)
    if not amps or any(not np.isfinite(a) for a in amps):
        raise ValueError("amplitudes must be non-empty and finite")
    nq = len(spec.joint_names)
    q_nom = _arr(q_nominal, "q_nominal", 1)
    if q_nom.size != nq:
        raise ValueError(f"q_nominal length {q_nom.size} != {nq}")
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    if q_bounds is not None:
        lo, hi = _arr(q_bounds[0], "q_lo", 1), _arr(q_bounds[1], "q_hi", 1)
        shape_ok = lo.shape == q_nom.shape and hi.shape == q_nom.shape
        if not shape_ok or np.any(lo > hi) or np.any(q_nom < lo) or np.any(q_nom > hi):
            raise ValueError("q_bounds invalid or does not contain q_nominal")
        bounds = (lo, hi)
    j_shared = spec.stacked()
    analysis = analyze_grip_jacobian_nullspace(j_shared, singular_tol=spec.singular_tol)
    if analysis.basis.shape[1] < 1:
        raise ValueError("empty null-space basis; no underdetermined body DOFs")
    obs = _obs_from_analysis(spec.joint_names, analysis)
    wmap = _infer_weights(spec.joint_names)
    if strategy_weights is not None:
        wmap = {
            str(k): _arr(v, f"weights[{k}]", 1) for k, v in strategy_weights.items()
        }
    all_cands: list[StrategyCandidate] = []
    envs: dict[str, StrategyEnvelope] = {}
    gate, tol = gate_evaluator, constraint_tol
    for strat, w in wmap.items():
        if w.shape != (nq,) or np.any(w < 0.0):
            raise ValueError(f"invalid strategy weights for {strat!r}")
        d = _weighted_dir(analysis.basis, w)
        sc = _qualify_strategy(strat, d, q_nom, j_shared, gate, amps, bounds, tol)
        all_cands.extend(sc)
        envs[strat] = _build_envelope(strat, sc)
    ok = tuple(c for c in all_cands if c.passed)
    bad = tuple(c for c in all_cands if not c.passed)
    return NullspaceStrategyReport(obs, envs, tuple(all_cands), ok, bad, tol)
