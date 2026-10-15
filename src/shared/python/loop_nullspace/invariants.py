"""Invariant quantities with validity gates on a closed-loop trajectory (NSX-17).

Each frame of an admitted trajectory yields five quantities, every one a
:class:`QuantityRecord` that is either ``INVARIANT`` or ``UNAVAILABLE`` with a
``reason`` and all-NaN values; nothing is computed outside its gate:

``net_club_wrench``
    Bare-club Newton-Euler wrench about the grip origin (NSX-16).  Gate: scaled
    ``Phi_ddot`` small and the bare-club keywords supplied.
``loop_power``
    ``sum_i tau_i v_i`` over the actuated coordinates of the loop support.  From
    ``b = S^T tau + J_c^T lambda (+ J_g^T f_g)`` this equals ``v^T b`` less the
    torque power outside the support, so it does not depend on ``lambda`` when
    ``J_c v = 0``.  Gate: scaled ``Phi_dot`` small; with ground contact the
    contact power ``(J_g v)^T f_g`` must be zero (stationary contacts) and no
    ground load may reach an actuated coordinate outside the support.
``base_centroidal_rows``
    ``J_{g,b}^T f_obs`` on the base rows (unactuated coordinates outside the loop
    support), the external wrench balanced by the ground load.  Gate: floating
    base, ground-contact Jacobian and an observed GRF, and the base balance
    ``b_b - J_{g,b}^T f_obs`` must be absorbable by ``J_{c,b}^T lambda``.
    Invariant for a ground load held at the observed value, not for free
    ``f_g``.  The centroidal-momentum map is not part of the snapshot, so only
    the base generalised-force rows are reported.
``tangent_pairing``
    ``N_c^T S^T tau`` with ``J_c N_c = 0`` (metric-free; orthonormal ``N_c``).
    Gate: as ``loop_power``'s ``Phi_dot`` gate, and ``J_g N_c = 0`` when a
    free ground load is present.
``min_metric_torque``
    ``P_{c,W} tau``, the minimum-``W``-norm torque equivalent to ``tau``.
    Invariant *given the declared convention* ``W`` (its label is carried in
    ``conventions``); never "the torque".  No further gate.

Non-claims: invariant means "independent of the hand split under the stated
model", not validated; records stay ``UNVALIDATED``.  The gate tolerances come
from an :class:`AdmissionGates` (the default is stricter than the admission
the trajectory may have passed with).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.admission import (
    AdmissionGates,
    AdmittedTrajectory,
)
from src.shared.python.loop_nullspace.contracts import ClosedLoopSnapshot
from src.shared.python.loop_nullspace.decomposition import (
    BALANCE_TOL,
    RANK_RTOL,
    AllocationMetric,
    Decomposition,
)
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    QuantityRecord,
)
from src.shared.python.loop_nullspace.grip_split import net_club_wrench

__all__ = [
    "INVARIANT_NAMES",
    "FrameInvariants",
    "InvariantReport",
    "compute_invariants",
    "min_metric_equivalent",
    "tangent_basis",
]

INVARIANT_NAMES = (
    "net_club_wrench",
    "loop_power",
    "base_centroidal_rows",
    "tangent_pairing",
    "min_metric_torque",
)
_GENERALISED_FORCE = "generalised force"
_JG_NC_TOL = 1e-9
_GRF_CONVENTION = "C-GRF-OBSERVED: ground load held at the observed value"


# ----------------------------------------------------------- value objects --


@dataclass(frozen=True)
class FrameInvariants:
    """The invariant records of one frame, keyed by :data:`INVARIANT_NAMES`."""

    index: int
    records: dict[str, QuantityRecord]


def _record_dict(rec: QuantityRecord) -> dict[str, Any]:
    unavailable = rec.allocation_dependence is AllocationDependence.UNAVAILABLE
    return {
        "values": None if unavailable else rec.values.tolist(),
        "units": rec.units,
        "allocation_dependence": rec.allocation_dependence.value,
        "validation_status": rec.validation_status.value,
        "conventions": list(rec.conventions),
        "reason": rec.reason,
        "model": rec.model_provenance.name,
        "input": rec.input_provenance.source,
    }


@dataclass(frozen=True)
class InvariantReport:
    """Per-frame invariant records plus the gate they were checked against."""

    frames: tuple[FrameInvariants, ...]
    gates: AdmissionGates
    metric_label: str

    def records(self, name: str) -> tuple[QuantityRecord, ...]:
        """The record ``name`` of every frame (``ValueError`` for unknown names)."""
        if name not in INVARIANT_NAMES:
            raise ValueError(f"unknown invariant {name!r}; expected {INVARIANT_NAMES}")
        return tuple(f.records[name] for f in self.frames)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form (unavailable values are ``None``, never NaN)."""
        return {
            "metric_label": self.metric_label,
            "gates": {
                "phi_dot_tol": self.gates.phi_dot_tol,
                "phi_ddot_tol": self.gates.phi_ddot_tol,
                "length_scale_m": self.gates.length_scale_m,
                "time_scale_s": self.gates.time_scale_s,
            },
            "frames": [
                {
                    "index": f.index,
                    "records": {k: _record_dict(v) for k, v in f.records.items()},
                }
                for f in self.frames
            ],
        }


# ------------------------------------------------------------- primitives --


def tangent_basis(snapshot: ClosedLoopSnapshot, decomposition: Decomposition) -> Array:
    """Orthonormal ``N_c`` (``n x (n - rank J_c)``) with ``J_c N_c = 0``.

    The rank is the decomposition's ``rank_J_c``.  The basis is the trailing
    right-singular block of ``J_c`` (unique only up to rotation within the
    tangent space).
    """
    rank = decomposition.rank_J_c
    _, _, vt = np.linalg.svd(snapshot.closure_rows())
    return np.asarray(vt[rank:].T)


def min_metric_equivalent(
    torque: Array, decomposition: Decomposition, metric: AllocationMetric
) -> Array:
    """``P_{c,W} tau``: the minimum-``W``-norm member of ``tau + range(T)``.

    ``T = N[:n_a]`` is the actuated-torque part of the null space, so the result
    is the same for every ``tau`` in one equivalence class.  Raises
    ``ValueError`` if shapes disagree with the decomposition.
    """
    n_a = decomposition.n_actuated
    tau = np.asarray(torque, dtype=np.float64)
    if tau.shape != (n_a,) or metric.W.shape != (n_a, n_a):
        raise ValueError(
            f"torque and metric must match the {n_a} actuated torques, got "
            f"{tau.shape} and {metric.W.shape}"
        )
    basis = decomposition.N[:n_a]
    if basis.shape[1] == 0:
        return tau.copy()
    chol = np.linalg.cholesky(0.5 * (metric.W + metric.W.T))
    left, sv, right_t = np.linalg.svd(chol.T @ basis, full_matrices=False)
    # N is unit in scaled coordinates, so a torque effect below this absolute
    # floor is the multiplier gauge (numerical noise), exactly as in decompose().
    floor = RANK_RTOL * float(np.linalg.norm(chol, 2)) * decomposition.scales.moment
    keep = sv > floor
    w = -(right_t[keep].T @ ((left[:, keep].T @ (chol.T @ tau)) / sv[keep]))
    return np.asarray(tau + basis @ w)


# ----------------------------------------------------------------- records --


def _make(
    snap: ClosedLoopSnapshot,
    values: Array,
    units: str,
    conventions: tuple[str, ...],
) -> QuantityRecord:
    return QuantityRecord(
        values=values,
        units=units,
        allocation_dependence=AllocationDependence.INVARIANT,
        model_provenance=snap.model_provenance,
        input_provenance=snap.input_provenance,
        conventions=conventions,
    )


def _unavailable(
    snap: ClosedLoopSnapshot, shape: tuple[int, ...], units: str, reason: str
) -> QuantityRecord:
    return QuantityRecord(
        values=np.full(shape, np.nan),
        units=units,
        allocation_dependence=AllocationDependence.UNAVAILABLE,
        model_provenance=snap.model_provenance,
        input_provenance=snap.input_provenance,
        reason=reason,
    )


@dataclass(frozen=True)
class _Gate:
    """Scaled closure residuals of one frame against the gate tolerances."""

    phi_dot: float
    phi_ddot: float
    contact_speed: float
    gates: AdmissionGates

    def failure(self, level: str) -> str | None:
        value, tol = getattr(self, level), self.gates.tolerance(level)
        if value > tol:
            return f"scaled {level} {value:.3e} exceeds the gate {tol:.1e}"
        return None

    def convention(self, level: str) -> str:
        return f"gate: scaled {level} <= {self.gates.tolerance(level):.1e}"


def _gate_of(snap: ClosedLoopSnapshot, gates: AdmissionGates) -> _Gate:
    jac = snap.closure_rows()
    length, time = gates.length_scale_m, gates.time_scale_s
    phi_dot = float(np.max(np.abs(jac @ snap.v), initial=0.0)) * time / length
    resid = jac @ snap.a + snap.Jdot_v
    phi_ddot = float(np.max(np.abs(resid), initial=0.0)) * time**2 / length
    speed = 0.0
    if snap.J_g is not None:
        speed = float(np.max(np.abs(snap.J_g @ snap.v), initial=0.0)) * time / length
    return _Gate(phi_dot, phi_ddot, speed, gates)


def _net_wrench(
    snap: ClosedLoopSnapshot, gate: _Gate, club_ne: Mapping[str, Any] | None
) -> QuantityRecord:
    units = "N,N,N,N m,N m,N m"
    if club_ne is None:
        return _unavailable(
            snap, (6,), units, "club_ne (bare-club Newton-Euler keywords) not supplied"
        )
    failed = gate.failure("phi_ddot")
    if failed:
        return _unavailable(snap, (6,), units, f"phi_ddot gate: {failed}")
    return net_club_wrench(snap, club_ne)


def _support_split(snap: ClosedLoopSnapshot) -> tuple[list[int], list[int]]:
    topo = snap.topology
    support = set(topo.loop_support_indices)
    inside = [i for i in topo.actuated_indices if i in support]
    outside = [i for i in topo.actuated_indices if i not in support]
    return inside, outside


def _loop_power(
    snap: ClosedLoopSnapshot, dec: Decomposition, gate: _Gate
) -> QuantityRecord:
    failed = gate.failure("phi_dot")
    if failed:
        return _unavailable(snap, (1,), "W", f"phi_dot gate: {failed}")
    inside, outside = _support_split(snap)
    if snap.J_g is not None:
        if gate.contact_speed > gate.gates.stance_velocity_tol:
            return _unavailable(
                snap,
                (1,),
                "W",
                f"contact power: scaled |J_g v| {gate.contact_speed:.3e} exceeds "
                f"{gate.gates.stance_velocity_tol:.1e} (contacts not stationary)",
            )
        if outside and np.max(np.abs(snap.J_g[:, outside])) > 0.0:
            return _unavailable(
                snap,
                (1,),
                "W",
                "a free ground load reaches an actuated coordinate outside the "
                "loop support, so the support power depends on f_g",
            )
    index = {c: k for k, c in enumerate(snap.topology.actuated_indices)}
    tau = dec.torque_particular
    power = sum(float(tau[index[i]] * snap.v[i]) for i in inside)
    return _make(snap, np.array([power]), "W", (gate.convention("phi_dot"),))


def _base_rows(snap: ClosedLoopSnapshot) -> list[int]:
    support = set(snap.topology.loop_support_indices)
    return [i for i in snap.topology.unactuated_indices if i not in support]


def _absorbable(jc_t: Array, rhs: Array, moment_scale: float) -> bool:
    """Whether ``rhs`` lies in the range of ``jc_t`` (scaled residual test)."""
    scaled = rhs / moment_scale
    if jc_t.shape[1] == 0 or not np.any(jc_t):
        resid = float(np.linalg.norm(scaled))
    else:
        left, sv, _ = np.linalg.svd(jc_t / moment_scale, full_matrices=False)
        keep = left[:, : int(np.count_nonzero(sv > RANK_RTOL * max(sv[0], 1.0)))]
        resid = float(np.linalg.norm(scaled - keep @ (keep.T @ scaled)))
    return resid <= BALANCE_TOL * max(1.0, float(np.linalg.norm(scaled)))


def _base_centroidal(snap: ClosedLoopSnapshot, dec: Decomposition) -> QuantityRecord:
    base = _base_rows(snap)
    units = _GENERALISED_FORCE
    shape = (len(base),)
    if not base:
        return _unavailable(
            snap,
            shape,
            units,
            "no floating base: no unactuated coordinate lies outside the loop support",
        )
    if snap.J_g is None:
        return _unavailable(snap, shape, units, "no ground contact (J_g absent)")
    if snap.grf_observed is None:
        return _unavailable(snap, shape, units, "no observed ground reaction force")
    external = snap.J_g[:, base].T @ snap.grf_observed
    rhs = snap.generalized_load()[base] - external
    if not _absorbable(snap.closure_rows()[:, base].T, rhs, dec.scales.moment):
        return _unavailable(
            snap,
            shape,
            units,
            "observed GRF is inconsistent with the base balance: the residual "
            "b_base - J_g,base^T f_obs is not absorbable by the closure",
        )
    return _make(snap, external, units, (_GRF_CONVENTION,))


def _tangent_pairing(
    snap: ClosedLoopSnapshot, dec: Decomposition, gate: _Gate
) -> QuantityRecord:
    basis = tangent_basis(snap, dec)
    shape = (basis.shape[1],)
    failed = gate.failure("phi_dot")
    if failed:
        return _unavailable(snap, shape, _GENERALISED_FORCE, f"phi_dot gate: {failed}")
    if snap.J_g is not None and basis.shape[1]:
        scale = max(1.0, float(np.max(np.abs(snap.J_g))))
        if np.max(np.abs(snap.J_g @ basis)) > _JG_NC_TOL * scale:
            return _unavailable(
                snap,
                shape,
                _GENERALISED_FORCE,
                "J_g N_c is not zero: N_c^T tau depends on the free ground load",
            )
    value = basis.T @ (snap.actuated_rows().T @ dec.torque_particular)
    return _make(
        snap,
        value,
        _GENERALISED_FORCE,
        (gate.convention("phi_dot"), "C-TANGENT-BASIS: orthonormal SVD null space"),
    )


def _min_metric(
    snap: ClosedLoopSnapshot, dec: Decomposition, metric: AllocationMetric | None
) -> QuantityRecord:
    tau = dec.torque_particular
    if metric is not None:
        tau = min_metric_equivalent(tau, dec, metric)
    return _make(snap, tau, "N m", (dec.metric_label,))


def _frame(
    snap: ClosedLoopSnapshot,
    dec: Decomposition,
    club_ne: Mapping[str, Any] | None,
    gates: AdmissionGates,
    metric: AllocationMetric | None,
) -> dict[str, QuantityRecord]:
    gate = _gate_of(snap, gates)
    return {
        "net_club_wrench": _net_wrench(snap, gate, club_ne),
        "loop_power": _loop_power(snap, dec, gate),
        "base_centroidal_rows": _base_centroidal(snap, dec),
        "tangent_pairing": _tangent_pairing(snap, dec, gate),
        "min_metric_torque": _min_metric(snap, dec, metric),
    }


# ------------------------------------------------------------------ public --


def compute_invariants(
    admitted: AdmittedTrajectory,
    decomposition: Sequence[Decomposition],
    *,
    club_ne: Sequence[Mapping[str, Any]] | None = None,
    gates: AdmissionGates | None = None,
    metric: AllocationMetric | None = None,
) -> InvariantReport:
    """Invariant quantities of every frame of ``admitted``, each behind its gate.

    ``decomposition`` is one :class:`Decomposition` per frame (for example
    ``decompose_trajectory``); ``club_ne`` is the optional per-frame bare-club
    Newton-Euler keyword set (``adapter.club_newton_euler``).  ``gates`` holds
    the tolerances; by default :class:`AdmissionGates` defaults with the
    trajectory's length and time scales.  ``metric``, when given, must carry the
    decompositions' metric label and ``P_{c,W} tau`` is re-evaluated with it.

    Raises:
        TypeError: ``admitted`` is not an ``AdmittedTrajectory`` or an element of
            ``decomposition`` is not a ``Decomposition``.
        ValueError: lengths disagree with the frame count or ``metric`` does not
            match the decompositions.

    Postcondition: every record is ``INVARIANT`` or ``UNAVAILABLE`` with a
    reason; an ``UNAVAILABLE`` record has all-NaN values.
    """
    if not isinstance(admitted, AdmittedTrajectory):
        raise TypeError(
            "compute_invariants requires an AdmittedTrajectory, got "
            f"{type(admitted).__name__}"
        )
    snaps = admitted.snapshots
    if isinstance(decomposition, Decomposition) or not all(
        isinstance(d, Decomposition) for d in decomposition
    ):
        raise TypeError(
            "decomposition must be a sequence of Decomposition (one per frame)"
        )
    if len(decomposition) != len(snaps):
        raise ValueError(
            f"one Decomposition per frame is required: {len(decomposition)} for "
            f"{len(snaps)} frames"
        )
    if club_ne is not None and len(club_ne) != len(snaps):
        raise ValueError(
            f"club_ne needs one mapping per frame: {len(club_ne)} for {len(snaps)}"
        )
    label = decomposition[0].metric_label
    if metric is not None and metric.label != label:
        raise ValueError(
            f"metric label {metric.label!r} does not match the decompositions' "
            f"metric {label!r}"
        )
    used = gates or dataclasses.replace(
        AdmissionGates(),
        length_scale_m=admitted.report.gates.length_scale_m,
        time_scale_s=admitted.report.gates.time_scale_s,
    )
    frames = tuple(
        FrameInvariants(
            k,
            _frame(s, d, None if club_ne is None else club_ne[k], used, metric),
        )
        for k, (s, d) in enumerate(zip(snaps, decomposition, strict=True))
    )
    return InvariantReport(frames, used, label)
