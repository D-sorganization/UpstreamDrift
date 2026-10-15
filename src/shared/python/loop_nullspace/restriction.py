"""Reaction-restriction sensitivity (WELD6 -> SHAFT_FREE5 -> BALL3) on a fixed welded trajectory (NSX-26, #12569).

Evaluates how static torque and split envelopes depend on which grip reaction
components are allowed when holding the welded kinematics ``(q, v, a)`` fixed.
Every result carries ``label = "REACTION_RESTRICTION_SENSITIVITY"`` to distinguish
reaction-subspace restriction from true alternative closure kinematics (NSX-72).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Any, Literal

import numpy as np

from src.shared.python.loop_nullspace.envelopes import (
    EnvelopeSeries,
    static_envelopes,
)
from src.shared.python.loop_nullspace.feasible_set import (
    FeasibleSet,
    FloatArray,
    _readonly_1d,
    _readonly_2d,
)
from src.shared.python.motion_matching.force_nullspace import ForceNullSpace

RestrictionKind = Literal["WELD6", "SHAFT_FREE5", "BALL3"]

REACTION_RESTRICTION_SENSITIVITY: str = "REACTION_RESTRICTION_SENSITIVITY"

RESTRICTION_ROW_INDICES: dict[str, tuple[int, ...]] = {
    "WELD6": (0, 1, 2, 3, 4, 5),
    "SHAFT_FREE5": (0, 1, 2, 4, 5),
    "BALL3": (0, 1, 2),
}

_HIERARCHY_ORDER: dict[str, int] = {
    "WELD6": 0,
    "SHAFT_FREE5": 1,
    "BALL3": 2,
}


def _normalize_restriction_kind(kind: Any) -> RestrictionKind:
    """Validate and normalize ``kind`` to ``'WELD6' | 'SHAFT_FREE5' | 'BALL3'``."""
    raw = str(kind.value) if isinstance(kind, Enum) else str(kind)
    if raw == "WELD6":
        return "WELD6"
    if raw == "SHAFT_FREE5":
        return "SHAFT_FREE5"
    if raw == "BALL3":
        return "BALL3"
    raise ValueError(
        f"Unsupported restriction kind {kind!r}; expected one of {tuple(RESTRICTION_ROW_INDICES)}"
    )


@dataclass(frozen=True)
class RestrictedFrameResult:
    """Per-frame outcome of a reaction-restriction evaluation (NSX-26)."""

    frame_index: int
    kind: RestrictionKind
    selected_rows: tuple[int, ...]
    snapshot: Any
    feasible_set: FeasibleSet
    is_empty: bool
    conflict_set: tuple[str, ...]
    label: str = REACTION_RESTRICTION_SENSITIVITY

    def __post_init__(self) -> None:
        if self.label != REACTION_RESTRICTION_SENSITIVITY:
            raise ValueError(
                f"RestrictedFrameResult.label must be {REACTION_RESTRICTION_SENSITIVITY!r}"
            )


@dataclass(frozen=True)
class RestrictionReport:
    """Trajectory-level reaction-restriction sensitivity report (NSX-26)."""

    kinds: tuple[RestrictionKind, ...]
    envelopes_by_kind: dict[str, EnvelopeSeries]
    frame_results_by_kind: dict[str, tuple[RestrictedFrameResult, ...]]
    empty_frames_by_kind: dict[str, tuple[int, ...]]
    conflict_sets_by_kind: dict[str, dict[int, tuple[str, ...]]]
    nesting_holds: bool
    max_nesting_violation: float
    label: str = REACTION_RESTRICTION_SENSITIVITY

    def __post_init__(self) -> None:
        if self.label != REACTION_RESTRICTION_SENSITIVITY:
            raise ValueError(
                f"RestrictionReport.label must be {REACTION_RESTRICTION_SENSITIVITY!r}"
            )


def restrict(snapshot: Any, kind: RestrictionKind | Any) -> Any:
    """Select canonical grip reaction rows for ``kind`` from a 6-row WELD6 ``snapshot``."""
    norm_kind = _normalize_restriction_kind(kind)
    if not hasattr(snapshot, "J_c") or not hasattr(snapshot, "Jdot_v"):
        raise TypeError("snapshot must provide J_c and Jdot_v arrays")

    j_c = _readonly_2d(snapshot.J_c, "snapshot.J_c")
    jdot_v = _readonly_1d(snapshot.Jdot_v, "snapshot.Jdot_v")
    if j_c.shape[0] != 6 or jdot_v.shape[0] != 6:
        raise ValueError(
            f"restrict() requires a 6-row WELD6 closure block, got J_c.shape={j_c.shape}"
        )

    topology = getattr(snapshot, "topology", None)
    if topology is not None and hasattr(topology, "closure_kind"):
        cur_kind_raw = topology.closure_kind
        cur_kind_str = (
            str(cur_kind_raw.value)
            if isinstance(cur_kind_raw, Enum)
            else str(cur_kind_raw)
        )
        if cur_kind_str != "WELD6":
            raise ValueError(
                f"restrict() requires snapshot closure_kind=='WELD6', got {cur_kind_str!r}"
            )

    rows = RESTRICTION_ROW_INDICES[norm_kind]
    row_list = list(rows)
    new_j_c = _readonly_2d(j_c[row_list, :], "restricted_J_c")
    new_jdot_v = _readonly_1d(jdot_v[row_list], "restricted_Jdot_v")

    if topology is not None and hasattr(topology, "closure_kind"):
        cur_ck = topology.closure_kind
        target_ck = type(cur_ck)(norm_kind) if isinstance(cur_ck, Enum) else norm_kind
        new_topo = replace(topology, closure_kind=target_ck)
        return replace(
            snapshot,
            J_c=new_j_c,
            Jdot_v=new_jdot_v,
            topology=new_topo,
        )
    return replace(snapshot, J_c=new_j_c, Jdot_v=new_jdot_v)


def _snapshot_generalized_load(snapshot: Any) -> FloatArray:
    """Return generalized inverse-dynamics load ``b = M @ a + h`` from ``snapshot``."""
    if hasattr(snapshot, "generalized_load") and callable(snapshot.generalized_load):
        return _readonly_1d(snapshot.generalized_load(), "b")
    m_mat = _readonly_2d(snapshot.M, "snapshot.M")
    a_vec = _readonly_1d(snapshot.a, "snapshot.a")
    h_vec = _readonly_1d(snapshot.h, "snapshot.h")
    return _readonly_1d(m_mat @ a_vec + h_vec, "b")


def _build_restricted_feasible_set(
    restricted_snap: Any,
    kind: RestrictionKind,
    hypothesis_set: Any,
    capacity_bounds: Any,
) -> FeasibleSet:
    """Assemble balance matrix ``A = [S^T, J_c^T, (J_g^T)]`` and build :class:`FeasibleSet`."""
    s_mat = _readonly_2d(restricted_snap.S, "snapshot.S")
    j_c = _readonly_2d(restricted_snap.J_c, "snapshot.J_c")
    blocks: list[np.ndarray] = [s_mat.T, j_c.T]
    j_g = getattr(restricted_snap, "J_g", None)
    if j_g is not None:
        blocks.append(_readonly_2d(j_g, "snapshot.J_g").T)

    matrix_a = np.hstack(blocks)
    rhs_b = _snapshot_generalized_load(restricted_snap)
    n_tau = int(s_mat.shape[0])

    try:
        fns = ForceNullSpace.from_balance(matrix_a, rhs_b)
    except ValueError:
        dim_x = int(matrix_a.shape[1])
        return FeasibleSet(
            x_particular=np.zeros(dim_x, dtype=np.float64),
            null_basis=np.zeros((dim_x, 0), dtype=np.float64),
            n_tau=n_tau,
            infeasible_reason_hypothesis_ids=(f"H-CLOSURE-RESTRICTION-{kind}",),
        )

    decomp_dict = {
        "x_particular": fns.particular,
        "null_basis": fns.basis,
        "dimensions": {"n_tau": n_tau},
    }
    return FeasibleSet.from_hypotheses(
        decomp_dict,
        hypothesis_set,
        capacity_bounds=capacity_bounds,
    )


def _check_envelope_nesting(
    kinds: Sequence[RestrictionKind],
    envelopes_by_kind: dict[str, EnvelopeSeries],
    empty_frames_by_kind: dict[str, tuple[int, ...]],
    tol: float,
) -> tuple[bool, float]:
    """Verify monotonicity ``F(BALL3) <= F(SHAFT_FREE5) <= F(WELD6)`` on feasible frames."""
    ordered = sorted(kinds, key=lambda k: _HIERARCHY_ORDER[k])
    max_violation = 0.0

    for idx in range(len(ordered) - 1):
        coarse_k = ordered[idx]
        fine_k = ordered[idx + 1]
        coarse_env = envelopes_by_kind[coarse_k]
        fine_env = envelopes_by_kind[fine_k]
        coarse_empty = set(empty_frames_by_kind[coarse_k])
        fine_empty = set(empty_frames_by_kind[fine_k])

        # If coarse is empty on frame t, fine must also be empty on frame t
        if fine_empty < coarse_empty:
            max_violation = max(max_violation, float("inf"))
            continue

        c_lo = coarse_env.lower_matrix
        c_hi = coarse_env.upper_matrix
        f_lo = fine_env.lower_matrix
        f_hi = fine_env.upper_matrix
        n_frames = int(c_lo.shape[0])

        for t_idx in range(n_frames):
            if t_idx in fine_empty or t_idx in coarse_empty:
                continue
            for q_idx in range(c_lo.shape[1]):
                lo_c, lo_f = float(c_lo[t_idx, q_idx]), float(f_lo[t_idx, q_idx])
                hi_c, hi_f = float(c_hi[t_idx, q_idx]), float(f_hi[t_idx, q_idx])
                if math.isfinite(lo_c) and math.isfinite(lo_f):
                    max_violation = max(max_violation, lo_c - lo_f)
                if math.isfinite(hi_c) and math.isfinite(hi_f):
                    max_violation = max(max_violation, hi_f - hi_c)

    return bool(max_violation <= tol), float(max_violation)


def restriction_sensitivity(
    trajectory_snapshots: Any,
    kinds: Sequence[RestrictionKind | Any] = ("WELD6", "SHAFT_FREE5", "BALL3"),
    quantities: Any = None,
    *,
    capacity_bounds: Any = None,
    hypothesis_set: Any = None,
    tol: float = 1e-8,
) -> RestrictionReport:
    """Evaluate static torque envelopes across ``WELD6 -> SHAFT_FREE5 -> BALL3`` restrictions."""
    snaps = (
        tuple(trajectory_snapshots.snapshots)
        if hasattr(trajectory_snapshots, "snapshots")
        else tuple(trajectory_snapshots)
    )
    if not snaps:
        raise ValueError("trajectory_snapshots must be non-empty")

    norm_kinds = tuple(_normalize_restriction_kind(k) for k in kinds)
    if not norm_kinds:
        raise ValueError("kinds must be non-empty")

    envelopes_by_kind: dict[str, EnvelopeSeries] = {}
    frame_results_by_kind: dict[str, tuple[RestrictedFrameResult, ...]] = {}
    empty_frames_by_kind: dict[str, tuple[int, ...]] = {}
    conflict_sets_by_kind: dict[str, dict[int, tuple[str, ...]]] = {}

    for kind in norm_kinds:
        rows = RESTRICTION_ROW_INDICES[kind]
        f_results: list[RestrictedFrameResult] = []
        f_sets: list[FeasibleSet] = []
        empty_indices: list[int] = []
        conflicts: dict[int, tuple[str, ...]] = {}

        for t_idx, snap in enumerate(snaps):
            r_snap = restrict(snap, kind)
            fs = _build_restricted_feasible_set(
                r_snap, kind, hypothesis_set, capacity_bounds
            )
            empty = fs.is_empty(tol=tol)
            c_set = fs.conflict_set(tol=tol) if empty else ()
            if empty:
                empty_indices.append(t_idx)
                conflicts[t_idx] = c_set

            f_sets.append(fs)
            f_results.append(
                RestrictedFrameResult(
                    frame_index=t_idx,
                    kind=kind,
                    selected_rows=rows,
                    snapshot=r_snap,
                    feasible_set=fs,
                    is_empty=empty,
                    conflict_set=c_set,
                    label=REACTION_RESTRICTION_SENSITIVITY,
                )
            )

        envelopes_by_kind[kind] = static_envelopes(
            f_sets, quantities=quantities, zero_tol=tol
        )
        frame_results_by_kind[kind] = tuple(f_results)
        empty_frames_by_kind[kind] = tuple(empty_indices)
        conflict_sets_by_kind[kind] = conflicts

    nesting_ok, max_viol = _check_envelope_nesting(
        norm_kinds, envelopes_by_kind, empty_frames_by_kind, tol
    )
    return RestrictionReport(
        kinds=norm_kinds,
        envelopes_by_kind=envelopes_by_kind,
        frame_results_by_kind=frame_results_by_kind,
        empty_frames_by_kind=empty_frames_by_kind,
        conflict_sets_by_kind=conflict_sets_by_kind,
        nesting_holds=nesting_ok,
        max_nesting_violation=max_viol,
        label=REACTION_RESTRICTION_SENSITIVITY,
    )


__all__ = [
    "REACTION_RESTRICTION_SENSITIVITY",
    "RESTRICTION_ROW_INDICES",
    "RestrictedFrameResult",
    "RestrictionKind",
    "RestrictionReport",
    "restrict",
    "restriction_sensitivity",
]
