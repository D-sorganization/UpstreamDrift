"""Pareto fronts (epsilon-constraint) and certified affine probes (NSX-25, #12568).

Provides:
- :class:`AffinePredicate`: Validated affine predicate over trajectory-coupled
  null-space coordinates (rejects non-affine predicates at construction).
- :class:`Certificate`: Certified probe answer (``HOLDS``, ``FAILS``, or
  ``UNDECIDED``) with JSON-serializable witness trajectories.
- :func:`certify`: Single trajectory-coupled LP min/max verification of an
  affine predicate over a frame window.
- :func:`trajectory_envelope`: Per-frame lower/upper bounds over the
  trajectory-coupled feasible set (with optional inter-frame rate coupling).
- :func:`pareto_front`: True epsilon-constraint Pareto frontier solving
  ``min f_1(z) s.t. f_2(z) <= eps_k`` across ``eps_grid``.
- :func:`weight_sweep_tradeoffs_shim`: Deprecation shim wrapping legacy weight
  sweeps with an explicit warning naming weight sweeps as non-Pareto on LPs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal
import warnings

import numpy as np
from scipy.optimize import linprog

from src.shared.python.motion_matching.force_nullspace import (
    ForceConstraints,
    ForceNullSpace,
    TorqueDistributionTradeoff,
    explore_torque_tradeoffs,
)

CertificateStatus = Literal["HOLDS", "FAILS", "UNDECIDED"]
PredicateSense = Literal[">=", "<="]

_LP_TOL = 1e-8


class WeightSweepDeprecationWarning(DeprecationWarning, UserWarning):
    """Warning emitted when legacy weight-sweep tradeoffs are called in place of Pareto fronts."""


def _readonly_1d(arr: Any, name: str) -> np.ndarray:
    if callable(arr):
        raise TypeError(f"{name} must be a finite 1-D array, not a callable")
    out = np.array(arr, dtype=float, copy=True)
    if out.ndim != 1 or out.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    out.setflags(write=False)
    return out


def _readonly_2d(arr: Any, cols: int, name: str) -> np.ndarray:
    out = np.array(arr, dtype=float, copy=True)
    if out.ndim != 2 or out.shape[1] != cols:
        raise ValueError(f"{name} must have shape (m, {cols})")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class FrameFeasibleSet:
    """Polyhedral feasible set for a single frame in null-space coordinates."""

    A_ub: np.ndarray
    b_ub: np.ndarray
    A_eq: np.ndarray | None = None
    b_eq: np.ndarray | None = None
    lb: np.ndarray | None = None
    ub: np.ndarray | None = None
    hypotheses: tuple[str, ...] = ("H-ACTUATOR-BOUNDS",)

    def __post_init__(self) -> None:
        a_ub = np.array(self.A_ub, dtype=float, copy=True)
        if a_ub.ndim != 2:
            raise ValueError("A_ub must be a 2-D array")
        b_ub = np.array(self.b_ub, dtype=float, copy=True)
        if b_ub.ndim != 1 or b_ub.shape[0] != a_ub.shape[0]:
            raise ValueError("b_ub length must match A_ub row count")
        a_ub.setflags(write=False)
        b_ub.setflags(write=False)
        object.__setattr__(self, "A_ub", a_ub)
        object.__setattr__(self, "b_ub", b_ub)

    def dimension(self) -> int:
        """Return the decision variable dimension for this frame."""
        if self.A_ub.shape[1] > 0:
            return int(self.A_ub.shape[1])
        if self.lb is not None:
            return int(np.asarray(self.lb).size)
        if self.ub is not None:
            return int(np.asarray(self.ub).size)
        raise ValueError("Cannot infer frame dimension from empty constraints")


@dataclass(frozen=True)
class AffinePredicate:
    """Certified probe affine predicate: sum_{t in W} w_t^T z_t (>= or <=) rhs_threshold."""

    name: str
    weights_by_frame: Mapping[int, np.ndarray]
    rhs_threshold: float
    sense: PredicateSense = ">="
    is_affine: bool = True

    def __post_init__(self) -> None:
        if not self.is_affine or callable(self.weights_by_frame):
            raise ValueError(
                "Non-affine predicates are rejected; certified probes require affine predicates"
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Predicate name must be a non-empty string")
        if self.sense not in (">=", "<="):
            raise ValueError(
                f"Invalid predicate sense {self.sense!r}; expected '>=' or '<='"
            )
        if not math.isfinite(float(self.rhs_threshold)):
            raise ValueError("rhs_threshold must be finite")
        if (
            not isinstance(self.weights_by_frame, Mapping)
            or len(self.weights_by_frame) == 0
        ):
            raise ValueError(
                "weights_by_frame must be a non-empty mapping of frame -> weights"
            )

        cleaned: dict[int, np.ndarray] = {}
        for frame_idx, w in self.weights_by_frame.items():
            if not isinstance(frame_idx, int) or frame_idx < 0:
                raise ValueError(
                    f"Frame index must be a non-negative int, got {frame_idx!r}"
                )
            cleaned[frame_idx] = _readonly_1d(w, f"weights_by_frame[{frame_idx}]")
        object.__setattr__(self, "weights_by_frame", cleaned)


@dataclass(frozen=True)
class LinearObjective:
    """Linear objective f(z) = offset + sum_t w_t^T z_t over trajectory variables."""

    name: str
    weights_by_frame: Mapping[int, np.ndarray]
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Objective name must be a non-empty string")
        if not math.isfinite(float(self.offset)):
            raise ValueError("Objective offset must be finite")
        if (
            not isinstance(self.weights_by_frame, Mapping)
            or len(self.weights_by_frame) == 0
        ):
            raise ValueError("weights_by_frame must be a non-empty mapping")
        cleaned = {
            int(k): _readonly_1d(v, f"weights_by_frame[{k}]")
            for k, v in self.weights_by_frame.items()
        }
        object.__setattr__(self, "weights_by_frame", cleaned)


@dataclass(frozen=True)
class Certificate:
    """Result of certifying an affine predicate over the trajectory-coupled feasible set."""

    status: CertificateStatus
    predicate_name: str
    worst_value: float
    margin: float
    witness_z: tuple[np.ndarray, ...] | None
    reason: str
    hypotheses: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize certificate and its witness trajectory to a JSON-compatible dict."""
        witness_list = (
            [np.asarray(w, dtype=float).tolist() for w in self.witness_z]
            if self.witness_z is not None
            else None
        )
        return {
            "status": self.status,
            "predicate_name": self.predicate_name,
            "worst_value": float(self.worst_value),
            "margin": float(self.margin),
            "witness_z": witness_list,
            "reason": self.reason,
            "hypotheses": list(self.hypotheses),
        }


@dataclass(frozen=True)
class TrajectoryEnvelope:
    """Per-frame lower and upper bounds over the trajectory-coupled feasible set."""

    lower: np.ndarray
    upper: np.ndarray
    feasible: bool
    rate_limit: float | None
    hypotheses: tuple[str, ...]
    label: str = "TRAJECTORY_COUPLED"


@dataclass(frozen=True)
class ParetoFront:
    """Epsilon-constraint Pareto frontier over the trajectory-coupled feasible set."""

    eps_values: np.ndarray
    primary_values: np.ndarray
    secondary_values: np.ndarray
    feasible_mask: np.ndarray
    witnesses: tuple[tuple[np.ndarray, ...] | None, ...]
    hypotheses: tuple[str, ...]


@dataclass(frozen=True)
class _CoupledLP:
    """Stacked trajectory-coupled LP representation."""

    dims: tuple[int, ...]
    offsets: tuple[int, ...]
    total_dim: int
    a_ub: np.ndarray
    b_ub: np.ndarray
    a_eq: np.ndarray | None
    b_eq: np.ndarray | None
    bounds: list[tuple[float | None, float | None]]
    hypotheses: tuple[str, ...]


def _extract_frame_lp(fs: Any) -> FrameFeasibleSet:
    if isinstance(fs, FrameFeasibleSet):
        return fs
    if hasattr(fs, "to_lp"):
        lp_data = fs.to_lp()
        return _extract_frame_lp(lp_data)
    a_ub = np.asarray(getattr(fs, "A_ub", getattr(fs, "A_ineq", None)), dtype=float)
    b_ub = np.asarray(getattr(fs, "b_ub", getattr(fs, "b_ineq", None)), dtype=float)
    a_eq_raw = getattr(fs, "A_eq", None)
    b_eq_raw = getattr(fs, "b_eq", None)
    a_eq = np.asarray(a_eq_raw, dtype=float) if a_eq_raw is not None else None
    b_eq = np.asarray(b_eq_raw, dtype=float) if b_eq_raw is not None else None
    lb = getattr(fs, "lb", None)
    ub = getattr(fs, "ub", None)
    bounds_attr = getattr(fs, "bounds", None)
    if bounds_attr is not None and lb is None and ub is None:
        lb, ub = bounds_attr
    hyps = tuple(getattr(fs, "hypotheses", ("H-ACTUATOR-BOUNDS",)))
    return FrameFeasibleSet(
        A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, lb=lb, ub=ub, hypotheses=hyps
    )


def _build_rate_limit_rows(
    dims: tuple[int, ...], offsets: tuple[int, ...], total_dim: int, rate_limit: float
) -> tuple[list[np.ndarray], list[float]]:
    if rate_limit <= 0.0 or not math.isfinite(rate_limit):
        raise ValueError("rate_limit must be a positive finite float")
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for t in range(len(dims) - 1):
        if dims[t] != dims[t + 1]:
            raise ValueError(
                "rate_limit requires identical variable dimension across frames"
            )
        d = dims[t]
        o_curr = offsets[t]
        o_next = offsets[t + 1]
        for j in range(d):
            r_pos = np.zeros(total_dim, dtype=float)
            r_pos[o_next + j] = 1.0
            r_pos[o_curr + j] = -1.0
            rows.append(r_pos)
            rhs.append(float(rate_limit))

            r_neg = np.zeros(total_dim, dtype=float)
            r_neg[o_next + j] = -1.0
            r_neg[o_curr + j] = 1.0
            rows.append(r_neg)
            rhs.append(float(rate_limit))
    return rows, rhs


def _assemble_coupled_lp(
    feasible_sets: Sequence[Any], rate_limit: float | None = None
) -> _CoupledLP:
    if len(feasible_sets) == 0:
        raise ValueError("feasible_sets must be non-empty")
    frames = [_extract_frame_lp(fs) for fs in feasible_sets]
    dims = tuple(f.dimension() for f in frames)
    offsets_list: list[int] = []
    running = 0
    for d in dims:
        offsets_list.append(running)
        running += d
    offsets = tuple(offsets_list)
    total_dim = running

    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []
    eq_rows: list[np.ndarray] = []
    eq_rhs: list[float] = []
    bounds: list[tuple[float | None, float | None]] = []
    seen_hyps: list[str] = []

    for idx, f in enumerate(frames):
        for h in f.hypotheses:
            if h not in seen_hyps:
                seen_hyps.append(h)
        off = offsets[idx]
        d = dims[idx]
        if f.A_ub.shape[0] > 0:
            block = np.zeros((f.A_ub.shape[0], total_dim), dtype=float)
            block[:, off : off + d] = f.A_ub
            ub_rows.extend(list(block))
            ub_rhs.extend(float(v) for v in f.b_ub)
        if f.A_eq is not None and f.b_eq is not None and f.A_eq.shape[0] > 0:
            eq_block = np.zeros((f.A_eq.shape[0], total_dim), dtype=float)
            eq_block[:, off : off + d] = _readonly_2d(f.A_eq, d, "A_eq")
            eq_rows.extend(list(eq_block))
            eq_rhs.extend(float(v) for v in f.b_eq)
        lb_vec = (
            np.asarray(f.lb, dtype=float) if f.lb is not None else np.full(d, -np.inf)
        )
        ub_vec = (
            np.asarray(f.ub, dtype=float) if f.ub is not None else np.full(d, np.inf)
        )
        for j in range(d):
            lo = None if not np.isfinite(lb_vec[j]) else float(lb_vec[j])
            hi = None if not np.isfinite(ub_vec[j]) else float(ub_vec[j])
            bounds.append((lo, hi))

    if rate_limit is not None:
        r_rows, r_rhs = _build_rate_limit_rows(dims, offsets, total_dim, rate_limit)
        ub_rows.extend(r_rows)
        ub_rhs.extend(r_rhs)
        if "H-TORQUE-RATE-LIMIT" not in seen_hyps:
            seen_hyps.append("H-TORQUE-RATE-LIMIT")

    a_ub_mat = np.vstack(ub_rows) if ub_rows else np.zeros((0, total_dim), dtype=float)
    b_ub_vec = np.array(ub_rhs, dtype=float) if ub_rhs else np.zeros(0, dtype=float)
    a_eq_mat = np.vstack(eq_rows) if eq_rows else None
    b_eq_vec = np.array(eq_rhs, dtype=float) if eq_rhs else None

    return _CoupledLP(
        dims=dims,
        offsets=offsets,
        total_dim=total_dim,
        a_ub=a_ub_mat,
        b_ub=b_ub_vec,
        a_eq=a_eq_mat,
        b_eq=b_eq_vec,
        bounds=bounds,
        hypotheses=tuple(seen_hyps),
    )


def _solve_coupled_lp(
    coupled: _CoupledLP,
    c: np.ndarray,
    extra_row: np.ndarray | None = None,
    extra_rhs: float | None = None,
) -> tuple[str, float, tuple[np.ndarray, ...] | None, str]:
    a_ub = coupled.a_ub
    b_ub = coupled.b_ub
    if extra_row is not None and extra_rhs is not None:
        a_ub = (
            np.vstack([a_ub, extra_row.reshape(1, -1)])
            if a_ub.size
            else extra_row.reshape(1, -1)
        )
        b_ub = np.append(b_ub, float(extra_rhs))

    res = linprog(
        c=c,
        A_ub=a_ub if a_ub.shape[0] > 0 else None,
        b_ub=b_ub if b_ub.shape[0] > 0 else None,
        A_eq=coupled.a_eq,
        b_eq=coupled.b_eq,
        bounds=coupled.bounds,
        method="highs",
    )
    if res.success and res.x is not None:
        sol = np.asarray(res.x, dtype=float)
        witness = tuple(
            _readonly_1d(
                sol[coupled.offsets[t] : coupled.offsets[t] + coupled.dims[t]], f"z_{t}"
            )
            for t in range(len(coupled.dims))
        )
        return "OPTIMAL", float(res.fun), witness, "Optimal solution found"
    status_code = int(getattr(res, "status", -1))
    msg = str(getattr(res, "message", "LP solver did not converge"))
    if status_code == 2:
        return (
            "INFEASIBLE",
            float("nan"),
            None,
            f"Trajectory feasible set is empty: {msg}",
        )
    if status_code == 3:
        return (
            "UNBOUNDED",
            float("nan"),
            None,
            f"Objective is unbounded over feasible set: {msg}",
        )
    return (
        "NUMERICAL",
        float("nan"),
        None,
        f"Solver numerical status ({status_code}): {msg}",
    )


def _resolve_window_indices(
    n_frames: int,
    predicate: AffinePredicate,
    window: Sequence[int] | slice | None,
) -> tuple[int, ...]:
    if window is None:
        indices = tuple(sorted(predicate.weights_by_frame.keys()))
    elif isinstance(window, slice):
        indices = tuple(range(*window.indices(n_frames)))
    else:
        indices = tuple(int(i) for i in window)
    for idx in indices:
        if idx < 0 or idx >= n_frames:
            raise ValueError(f"Window frame index {idx} out of range [0, {n_frames})")
    active = tuple(i for i in indices if i in predicate.weights_by_frame)
    if not active:
        raise ValueError("No predicate frame weights overlap with the requested window")
    return active


def certify(
    feasible_sets: Sequence[Any],
    predicate: AffinePredicate,
    window: Sequence[int] | slice | None = None,
    *,
    rate_limit: float | None = None,
) -> Certificate:
    """Certify an affine predicate over the trajectory-coupled feasible set via a single LP."""
    if not isinstance(predicate, AffinePredicate):
        raise TypeError("predicate must be an AffinePredicate instance")
    coupled = _assemble_coupled_lp(feasible_sets, rate_limit=rate_limit)
    active_frames = _resolve_window_indices(len(coupled.dims), predicate, window)

    c = np.zeros(coupled.total_dim, dtype=float)
    for t in active_frames:
        w_t = predicate.weights_by_frame[t]
        if w_t.size != coupled.dims[t]:
            raise ValueError(
                f"Predicate weight dim ({w_t.size}) at frame {t} does not match frame dim ({coupled.dims[t]})"
            )
        off = coupled.offsets[t]
        c[off : off + coupled.dims[t]] += w_t

    lp_c = c if predicate.sense == ">=" else -c
    status, obj_val, witness, reason = _solve_coupled_lp(coupled, lp_c)
    if status != "OPTIMAL" or witness is None:
        return Certificate(
            status="UNDECIDED",
            predicate_name=predicate.name,
            worst_value=float("nan"),
            margin=float("nan"),
            witness_z=None,
            reason=reason,
            hypotheses=coupled.hypotheses,
        )

    worst_val = obj_val if predicate.sense == ">=" else -obj_val
    margin = (
        worst_val - predicate.rhs_threshold
        if predicate.sense == ">="
        else predicate.rhs_threshold - worst_val
    )
    holds = margin >= -_LP_TOL
    return Certificate(
        status="HOLDS" if holds else "FAILS",
        predicate_name=predicate.name,
        worst_value=float(worst_val),
        margin=float(margin),
        witness_z=witness,
        reason=(
            "Predicate holds across all feasible trajectories"
            if holds
            else f"Counterexample violates predicate ({worst_val:.6g} vs {predicate.rhs_threshold:.6g})"
        ),
        hypotheses=coupled.hypotheses,
    )


def trajectory_envelope(
    feasible_sets: Sequence[Any],
    quantities: Sequence[np.ndarray] | None = None,
    *,
    rate_limit: float | None = None,
) -> TrajectoryEnvelope:
    """Compute per-frame lower and upper bounds over the trajectory-coupled feasible set."""
    coupled = _assemble_coupled_lp(feasible_sets, rate_limit=rate_limit)
    n_frames = len(coupled.dims)
    d0 = coupled.dims[0]
    q_dirs = (
        [np.eye(d0, dtype=float)[i] for i in range(d0)]
        if quantities is None
        else [_readonly_1d(q, "quantity") for q in quantities]
    )
    n_q = len(q_dirs)
    lower = np.full((n_frames, n_q), np.nan, dtype=float)
    upper = np.full((n_frames, n_q), np.nan, dtype=float)
    any_feasible = True

    for t in range(n_frames):
        off = coupled.offsets[t]
        dt_dim = coupled.dims[t]
        for q_idx, q_vec in enumerate(q_dirs):
            if q_vec.size != dt_dim:
                raise ValueError(
                    "Quantity direction dimension must match frame dimension"
                )
            c = np.zeros(coupled.total_dim, dtype=float)
            c[off : off + dt_dim] = q_vec
            st_min, val_min, _, _ = _solve_coupled_lp(coupled, c)
            st_max, val_max_neg, _, _ = _solve_coupled_lp(coupled, -c)
            if st_min == "OPTIMAL" and st_max == "OPTIMAL":
                lower[t, q_idx] = val_min
                upper[t, q_idx] = -val_max_neg
            else:
                any_feasible = False

    lower.setflags(write=False)
    upper.setflags(write=False)
    return TrajectoryEnvelope(
        lower=lower,
        upper=upper,
        feasible=any_feasible,
        rate_limit=rate_limit,
        hypotheses=coupled.hypotheses,
    )


def _objective_vector(coupled: _CoupledLP, obj: Any) -> tuple[np.ndarray, float]:
    c = np.zeros(coupled.total_dim, dtype=float)
    if isinstance(obj, LinearObjective):
        for t, w_t in obj.weights_by_frame.items():
            off = coupled.offsets[t]
            c[off : off + coupled.dims[t]] += w_t
        return c, float(obj.offset)
    if isinstance(obj, Mapping):
        for t, w_t in obj.items():
            vec = _readonly_1d(w_t, f"objective[{t}]")
            off = coupled.offsets[int(t)]
            c[off : off + coupled.dims[int(t)]] += vec
        return c, 0.0
    vec = _readonly_1d(obj, "objective")
    if vec.size == coupled.total_dim:
        return vec.copy(), 0.0
    if vec.size == coupled.dims[0]:
        for t in range(len(coupled.dims)):
            off = coupled.offsets[t]
            c[off : off + coupled.dims[t]] = vec
        return c, 0.0
    raise ValueError("Objective dimension does not match trajectory feasible set")


def pareto_front(
    feasible_sets: Sequence[Any],
    primary_objective: Any,
    secondary_objective: Any = None,
    eps_grid: Sequence[float] | np.ndarray | None = None,
    *,
    rate_limit: float | None = None,
) -> ParetoFront:
    """Compute the epsilon-constraint Pareto frontier min f_1(z) s.t. f_2(z) <= eps_k."""
    if (
        eps_grid is None
        and isinstance(primary_objective, tuple)
        and len(primary_objective) == 2
    ):
        eps_grid = secondary_objective
        primary_objective, secondary_objective = primary_objective
    if eps_grid is None or secondary_objective is None:
        raise ValueError(
            "pareto_front requires primary_objective, secondary_objective, and eps_grid"
        )

    eps_arr = _readonly_1d(eps_grid, "eps_grid")
    coupled = _assemble_coupled_lp(feasible_sets, rate_limit=rate_limit)
    c1, off1 = _objective_vector(coupled, primary_objective)
    c2, off2 = _objective_vector(coupled, secondary_objective)

    n_pts = eps_arr.size
    prim_vals = np.full(n_pts, np.nan, dtype=float)
    sec_vals = np.full(n_pts, np.nan, dtype=float)
    feas_mask = np.zeros(n_pts, dtype=bool)
    witnesses: list[tuple[np.ndarray, ...] | None] = []

    for k, eps_val in enumerate(eps_arr):
        status, f1_raw, wit, _ = _solve_coupled_lp(
            coupled, c1, extra_row=c2, extra_rhs=float(eps_val) - off2
        )
        if status == "OPTIMAL" and wit is not None:
            feas_mask[k] = True
            prim_vals[k] = f1_raw + off1
            stacked_z = np.concatenate(wit)
            sec_vals[k] = float(c2 @ stacked_z) + off2
            witnesses.append(wit)
        else:
            witnesses.append(None)

    prim_vals.setflags(write=False)
    sec_vals.setflags(write=False)
    feas_mask.setflags(write=False)
    return ParetoFront(
        eps_values=eps_arr,
        primary_values=prim_vals,
        secondary_values=sec_vals,
        feasible_mask=feas_mask,
        witnesses=tuple(witnesses),
        hypotheses=coupled.hypotheses,
    )


def weight_sweep_tradeoffs_shim(
    space: ForceNullSpace,
    constraints: ForceConstraints,
    *,
    lead_indices: Sequence[int],
    trail_indices: Sequence[int],
    grip_indices: Sequence[int] | None = None,
    ground_indices: Sequence[int] | None = None,
    joint_velocities: np.ndarray | None = None,
) -> TorqueDistributionTradeoff:
    """Shim over explore_torque_tradeoffs emitting a weight-sweep deprecation warning."""
    warnings.warn(
        "explore_torque_tradeoffs performs a heuristic weight sweep (which misses "
        "non-supported points on linear/polyhedral fronts) rather than a true "
        "epsilon-constraint Pareto front; use loop_nullspace.probes.pareto_front instead.",
        WeightSweepDeprecationWarning,
        stacklevel=2,
    )
    return explore_torque_tradeoffs(
        space,
        constraints,
        lead_indices=lead_indices,
        trail_indices=trail_indices,
        grip_indices=grip_indices,
        ground_indices=ground_indices,
        joint_velocities=joint_velocities,
    )


explore_torque_tradeoffs_shim = weight_sweep_tradeoffs_shim

__all__ = [
    "AffinePredicate",
    "Certificate",
    "CertificateStatus",
    "FrameFeasibleSet",
    "LinearObjective",
    "ParetoFront",
    "PredicateSense",
    "TrajectoryEnvelope",
    "WeightSweepDeprecationWarning",
    "certify",
    "explore_torque_tradeoffs_shim",
    "pareto_front",
    "trajectory_envelope",
    "weight_sweep_tradeoffs_shim",
]
