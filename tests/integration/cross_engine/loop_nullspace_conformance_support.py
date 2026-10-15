"""Support for the NSX-56 two-track closed-loop conformance harness (#12584).

Track N compares the canonical snapshot arrays ``M, h, J_c, Jdot_v`` between a
reference engine and another engine.  Track S compares quantities *derived*
from those arrays (constrained acceleration, grip-wrench multiplier, kinetic
energy, power, closure singular values, grip-wrench envelope).  The Track S
tolerance is never entered by hand: it is first-order perturbation theory
applied to the Track N differences actually measured (see :func:`kkt_bound`).

Agreement between engines is not physical validation: two engines built from
one specification can share a modelling error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    InputProvenance,
)
from src.shared.python.loop_nullspace.snapshot import snapshot_from_adapter

EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support"
)
SPEC_PATH = EVIDENCE / "full_body_spec_hipcal_scaled.json"
CHECK_NAME = "closed_loop_snapshot_conformance"
TRACK_N_METRICS = ("M", "h", "J_c", "Jdot_v")
#: Half-width of the normalised actuator box used for the grip-wrench envelope.
ENVELOPE_EFFORT = 1.0


def scaled_error(x: np.ndarray, ref: np.ndarray) -> float:
    """``max|x - ref| / max(1, max|ref|)``: relative with a unit floor."""
    return float(np.max(np.abs(x - ref))) / max(1.0, float(np.max(np.abs(ref))))


@dataclass(frozen=True)
class NamedSnapshot:
    """A snapshot with its arrays permuted into a reference coordinate order."""

    snapshot: ClosedLoopSnapshot
    M: np.ndarray
    h: np.ndarray
    J_c: np.ndarray
    Jdot_v: np.ndarray
    v: np.ndarray
    actuated: frozenset[str]
    loop_support: frozenset[str]


def to_reference_order(
    snapshot: ClosedLoopSnapshot, reference_order: tuple[str, ...]
) -> NamedSnapshot:
    """Permute ``snapshot`` into ``reference_order`` by coordinate name.

    Raises ``ValueError`` if the coordinate vocabularies differ.
    """
    own = list(snapshot.topology.coordinate_order)
    if sorted(own) != sorted(reference_order):
        raise ValueError("snapshots must share one coordinate vocabulary")
    perm = [own.index(name) for name in reference_order]
    ix = np.ix_(perm, perm)
    topo = snapshot.topology
    return NamedSnapshot(
        snapshot=snapshot,
        M=snapshot.M[ix],
        h=snapshot.h[perm],
        J_c=snapshot.J_c[:, perm],
        Jdot_v=snapshot.Jdot_v,
        v=snapshot.v[perm],
        actuated=frozenset(own[i] for i in topo.actuated_indices),
        loop_support=frozenset(own[i] for i in topo.loop_support_indices),
    )


def build_snapshot(adapter, named_q, named_v, source: str) -> ClosedLoopSnapshot:
    """Snapshot of ``adapter`` at the named (projected) state, zero target ``a``."""
    order = adapter.coordinate_order
    return snapshot_from_adapter(
        adapter,
        np.array([named_q[n] for n in order]),
        np.array([named_v[n] for n in order]),
        np.zeros(adapter.nv),
        input_provenance=InputProvenance(kind="test", source=source),
    )


def track_n_errors(ref: NamedSnapshot, other: NamedSnapshot) -> dict[str, float]:
    """Scaled Track N difference for each canonical array."""
    return {
        metric: scaled_error(getattr(other, metric), getattr(ref, metric))
        for metric in TRACK_N_METRICS
    }


# -- Track S -------------------------------------------------------------


def seeded_effort(names: tuple[str, ...], actuated: frozenset[str]) -> np.ndarray:
    """One effort by coordinate name (zero off the actuated set), reference order."""
    rng = np.random.default_rng(12584)
    draw = rng.normal(size=len(names))
    return np.array([d if n in actuated else 0.0 for n, d in zip(names, draw)])


def kkt_matrix(snap: NamedSnapshot) -> np.ndarray:
    """``[[M, -J^T], [J, 0]]`` of the rigid closed-loop acceleration problem."""
    m = snap.J_c.shape[0]
    return np.block([[snap.M, -snap.J_c.T], [snap.J_c, np.zeros((m, m))]])


def kkt_solve(snap: NamedSnapshot, tau: np.ndarray) -> np.ndarray:
    """Return ``[a; lambda]`` solving ``M a - J^T lam = tau - h``, ``J a = -Jdot v``."""
    rhs = np.concatenate((tau - snap.h, -snap.Jdot_v))
    return np.linalg.solve(kkt_matrix(snap), rhs)


def kkt_bound(
    ref: NamedSnapshot, other: NamedSnapshot, solution_norm: float, rhs_scale: float
) -> float:
    """First-order bound on ``|x_other - x_ref|_2`` from the measured differences.

    With ``K x = r``: ``|dx| <= |K^-1| (|dr| + |dK||x|) / (1 - |K^-1||dK|)``.
    ``dK`` and ``dr`` are the measured spectral/Euclidean differences of the
    two snapshots; ``rhs_scale`` multiplies ``dr`` (``1`` for the offset
    solution, ``0`` for the linear effort response).
    """
    k_ref, k_other = kkt_matrix(ref), kkt_matrix(other)
    inv_norm = 1.0 / float(np.linalg.svd(k_ref, compute_uv=False)[-1])
    d_k = float(np.linalg.norm(k_other - k_ref, 2))
    d_r = float(
        np.linalg.norm(np.concatenate((other.h - ref.h, other.Jdot_v - ref.Jdot_v)))
    )
    denom = 1.0 - inv_norm * d_k
    if denom <= 0.0:
        raise ValueError("snapshots differ too much for a perturbation bound")
    return inv_norm * (rhs_scale * d_r + d_k * solution_norm) / denom


def effort_response(snap: NamedSnapshot, actuated_idx: list[int]) -> np.ndarray:
    """``d[a; lambda]/d tau_act``: the KKT solve of the unit actuated efforts."""
    n = snap.M.shape[0]
    m = snap.J_c.shape[0]
    rhs = np.zeros((n + m, len(actuated_idx)))
    rhs[actuated_idx, np.arange(len(actuated_idx))] = 1.0
    return np.linalg.solve(kkt_matrix(snap), rhs)


def multiplier_envelope(
    snap: NamedSnapshot, tau: np.ndarray, actuated_idx: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form box envelope of the grip wrench over ``|tau_act| <= 1``.

    ``lambda`` is affine in the actuated effort, so the extremes over the box
    are ``lambda0 +- sum_j |Lambda_kj|`` (exact, no optimisation involved).
    """
    n = snap.M.shape[0]
    centre = kkt_solve(snap, tau)[n:]
    spread = ENVELOPE_EFFORT * np.abs(effort_response(snap, actuated_idx)[n:]).sum(1)
    return centre - spread, centre + spread
