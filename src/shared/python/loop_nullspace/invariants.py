"""Allocation-invariant quantities and scaled validity gates (#12562, NSX-17).

Implements Rule R5 of Epic #12593:
1. net_bare_club_wrench: bare-club Newton-Euler wrench in C-FRAME-GRIP (INVARIANT).
2. summed_loop_power: (S v)^T tau = v^T (b - J_g^T f_g) over loop support, gated on
   ||J_c v|| / v_char <= max_scaled_closure_vel.
3. base_centroidal_wrench: unactuated floating-base rows (M a + h)_u - J_{g,u}^T f_g
   when unactuated rows exist and J_{c,u} == 0.
4. outside_loop_torques: actuated joints with ||row_i(J_c^T)|| == 0.
5. tangent_generalized_load: N_c^T tau = N_c^T b where J_c S^T N_c = 0 (C-TANGENT-ORTHONORMAL).
6. metric_projected_torque: P_{c,W} tau tagged with declared metric_convention_id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg

from src.shared.python.loop_nullspace.contracts import (
    AllocationDependence,
    ClosedLoopSnapshot,
    FloatArray,
    QuantityRecord,
    _readonly_float_array,
    _validate_nonempty_str,
)
from src.shared.python.loop_nullspace.grip_split import net_club_wrench

TANGENT_ORTHONORMAL_CONVENTION_ID = "C-TANGENT-ORTHONORMAL"
DEFAULT_METRIC_CONVENTION_ID = "C-METRIC-IDENTITY"


@dataclass(frozen=True)
class InvariantGates:
    """Scaled kinematic validity gates for allocation-invariant quantities (R5)."""

    max_scaled_closure_vel: float = 1e-4
    max_scaled_closure_acc: float = 1e-3
    characteristic_speed_m_s: float = 20.0

    def __post_init__(self) -> None:
        for name in (
            "max_scaled_closure_vel",
            "max_scaled_closure_acc",
            "characteristic_speed_m_s",
        ):
            val = float(getattr(self, name))
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {val}")
            object.__setattr__(self, name, val)


def _record_to_dict(record: QuantityRecord) -> dict[str, Any]:
    """Serialize a QuantityRecord to a JSON-compatible dictionary."""
    vals = [
        None if np.isnan(float(x)) else float(x)
        for x in np.asarray(record.values, dtype=np.float64).ravel()
    ]
    return {
        "values": vals,
        "shape": list(record.values.shape),
        "units": record.units,
        "allocation_dependence": record.allocation_dependence.value,
        "conventions": list(record.conventions),
        "hypotheses": list(record.hypotheses),
        "validation_status": record.validation_status.value,
        "reason": record.reason,
        "model_provenance": {
            "spec_sha256": record.model_provenance.spec_sha256,
            "inertial_source": record.model_provenance.inertial_source,
            "bare_club_bench_id": record.model_provenance.bare_club_bench_id,
            "engine_name": record.model_provenance.engine_name,
        },
        "input_provenance": {
            "kinematic_source": record.input_provenance.kinematic_source,
            "derivative_method": record.input_provenance.derivative_method,
            "time_sync_status": record.input_provenance.time_sync_status,
        },
    }


@dataclass(frozen=True)
class InvariantReport:
    """Data-determined invariant report with per-quantity validity gate status (R5)."""

    net_bare_club_wrench: QuantityRecord
    summed_loop_power: QuantityRecord
    base_centroidal_wrench: QuantityRecord
    outside_loop_torques: QuantityRecord
    tangent_generalized_load: QuantityRecord
    metric_projected_torque: QuantityRecord

    def to_dict(self) -> dict[str, Any]:
        """Serialize all six invariant QuantityRecord channels to a JSON-ready dict."""
        return {
            "net_bare_club_wrench": _record_to_dict(self.net_bare_club_wrench),
            "summed_loop_power": _record_to_dict(self.summed_loop_power),
            "base_centroidal_wrench": _record_to_dict(self.base_centroidal_wrench),
            "outside_loop_torques": _record_to_dict(self.outside_loop_torques),
            "tangent_generalized_load": _record_to_dict(self.tangent_generalized_load),
            "metric_projected_torque": _record_to_dict(self.metric_projected_torque),
        }


def _unavailable_record(
    snapshot: ClosedLoopSnapshot,
    *,
    units: str,
    reason: str,
    conventions: tuple[str, ...] = (),
    size: int = 1,
) -> QuantityRecord:
    """Build an UNAVAILABLE QuantityRecord with all-NaN values and explicit reason."""
    return QuantityRecord(
        values=np.full(max(size, 1), np.nan, dtype=np.float64),
        units=units,
        allocation_dependence=AllocationDependence.UNAVAILABLE,
        conventions=conventions,
        hypotheses=(),
        model_provenance=snapshot.model_provenance,
        input_provenance=snapshot.input_provenance,
        reason=reason,
    )


def _invariant_record(
    snapshot: ClosedLoopSnapshot,
    *,
    values: FloatArray,
    units: str,
    conventions: tuple[str, ...] = (),
) -> QuantityRecord:
    """Build an INVARIANT QuantityRecord with read-only values and provenance."""
    return QuantityRecord(
        values=values,
        units=units,
        allocation_dependence=AllocationDependence.INVARIANT,
        conventions=conventions,
        hypotheses=(),
        model_provenance=snapshot.model_provenance,
        input_provenance=snapshot.input_provenance,
    )


def _effective_generalized_load(snapshot: ClosedLoopSnapshot) -> FloatArray:
    """Return b_eff = M a + h - J_g^T f_g when observed GRF is available, else M a + h."""
    b_vec = snapshot.generalized_load()
    if snapshot.J_g is not None and snapshot.grf_observed is not None:
        return np.asarray(
            b_vec - snapshot.J_g.T @ snapshot.grf_observed, dtype=np.float64
        )
    return b_vec


def _compute_summed_loop_power(
    snapshot: ClosedLoopSnapshot,
    b_eff: FloatArray,
    gates: InvariantGates,
) -> QuantityRecord:
    """Compute summed loop power (S v)^T tau = v^T (b - J_g^T f_g) gated on ||J_c v|| / v_char."""
    scaled_vel_res = float(np.linalg.norm(snapshot.J_c @ snapshot.v)) / float(
        gates.characteristic_speed_m_s
    )
    if scaled_vel_res > gates.max_scaled_closure_vel:
        return _unavailable_record(
            snapshot,
            units="W",
            reason="closure_velocity_residual_exceeds_gate",
            size=1,
        )
    support = np.asarray(snapshot.topology.loop_support_indices, dtype=np.int64)
    power = (
        float(np.dot(snapshot.v[support], b_eff[support]))
        if support.size > 0
        else float(np.dot(snapshot.v, b_eff))
    )
    return _invariant_record(
        snapshot,
        values=np.array([power], dtype=np.float64),
        units="W",
    )


def _compute_base_centroidal_wrench(
    snapshot: ClosedLoopSnapshot,
    b_eff: FloatArray,
) -> QuantityRecord:
    """Compute unactuated floating-base rows (M a + h)_u - J_{g,u}^T f_g when J_{c,u} == 0."""
    unactuated = snapshot.unactuated_rows()
    if unactuated.size == 0:
        return _unavailable_record(
            snapshot,
            units="N,N*m",
            reason="no_unactuated_floating_base_rows",
            size=1,
        )
    j_cu = snapshot.J_c[:, unactuated]
    if float(np.linalg.norm(j_cu)) > 1e-12:
        return _unavailable_record(
            snapshot,
            units="N,N*m",
            reason="closure_couples_unactuated_rows",
            size=int(unactuated.size),
        )
    return _invariant_record(
        snapshot,
        values=np.asarray(b_eff[unactuated], dtype=np.float64),
        units="N,N*m",
    )


def _compute_outside_loop_torques(
    snapshot: ClosedLoopSnapshot,
    b_eff: FloatArray,
) -> QuantityRecord:
    """Extract actuated joint torques whose closure Jacobian rows vanish."""
    j_ca_t = snapshot.S @ snapshot.J_c.T
    tau_p = snapshot.S @ b_eff
    row_norms = np.linalg.norm(j_ca_t, axis=1)
    outside_mask = row_norms <= 1e-12
    if not np.any(outside_mask):
        return _unavailable_record(
            snapshot,
            units="N*m",
            reason="all_actuated_joints_in_loop_support",
            size=1,
        )
    return _invariant_record(
        snapshot,
        values=np.asarray(tau_p[outside_mask], dtype=np.float64),
        units="N*m",
    )


def _validate_metric_matrix(metric_W: FloatArray | None, n_tau: int) -> FloatArray:
    """Validate or default the (n_tau, n_tau) symmetric positive-definite metric W."""
    if metric_W is None:
        return np.eye(n_tau, dtype=np.float64)
    w_arr = _readonly_float_array(metric_W, "metric_W", shape=(n_tau, n_tau))
    if not np.allclose(w_arr, w_arr.T, atol=1e-10):
        raise ValueError("metric_W must be symmetric")
    sym_w = 0.5 * (w_arr + w_arr.T)
    try:
        np.linalg.cholesky(sym_w)
    except np.linalg.LinAlgError as exc:
        raise ValueError("metric_W must be symmetric positive definite") from exc
    return sym_w


def _compute_tangent_and_metric_projections(
    snapshot: ClosedLoopSnapshot,
    b_eff: FloatArray,
    metric_W: FloatArray | None,
    metric_convention_id: str,
) -> tuple[QuantityRecord, QuantityRecord]:
    """Compute metric-free tangent load N_c^T tau and metric-projected torque P_{c,W} tau."""
    conv_id = _validate_nonempty_str(metric_convention_id, "metric_convention_id")
    j_ca = snapshot.J_c @ snapshot.S.T
    tau_base = snapshot.S @ b_eff
    n_tau = int(tau_base.shape[0])
    w_mat = _validate_metric_matrix(metric_W, n_tau)

    n_c = scipy.linalg.null_space(j_ca, rcond=1e-10)
    if n_c.shape[1] == 0:
        tangent_rec = _unavailable_record(
            snapshot,
            units="N*m",
            reason="zero_dimensional_tangent_subspace",
            conventions=(TANGENT_ORTHONORMAL_CONVENTION_ID,),
            size=1,
        )
        metric_rec = _invariant_record(
            snapshot,
            values=np.zeros(n_tau, dtype=np.float64),
            units="N*m",
            conventions=(conv_id,),
        )
        return tangent_rec, metric_rec

    # Canonical sign orientation of orthonormal basis columns of N_c
    for col_idx in range(n_c.shape[1]):
        col = n_c[:, col_idx]
        pivot = int(np.argmax(np.abs(col)))
        if col[pivot] < 0.0:
            n_c[:, col_idx] = -col

    tangent_vals = n_c.T @ tau_base
    w_inv_nc = np.linalg.solve(w_mat, n_c)
    gram = n_c.T @ w_inv_nc
    proj_tau = w_inv_nc @ np.linalg.solve(gram, tangent_vals)

    tangent_rec = _invariant_record(
        snapshot,
        values=tangent_vals,
        units="N*m",
        conventions=(TANGENT_ORTHONORMAL_CONVENTION_ID,),
    )
    metric_rec = _invariant_record(
        snapshot,
        values=proj_tau,
        units="N*m",
        conventions=(conv_id,),
    )
    return tangent_rec, metric_rec


def compute_invariants(
    snapshot: ClosedLoopSnapshot,
    *,
    metric_W: np.ndarray | None = None,
    metric_convention_id: str = DEFAULT_METRIC_CONVENTION_ID,
    gates: InvariantGates | None = None,
) -> InvariantReport:
    """Compute all six allocation-invariant quantities with validity gates (R5)."""
    if not isinstance(snapshot, ClosedLoopSnapshot):
        raise TypeError("snapshot must be a ClosedLoopSnapshot instance")
    active_gates = InvariantGates() if gates is None else gates
    if not isinstance(active_gates, InvariantGates):
        raise TypeError("gates must be an InvariantGates instance")

    b_eff = _effective_generalized_load(snapshot)
    club_wrench = net_club_wrench(snapshot)
    loop_power = _compute_summed_loop_power(snapshot, b_eff, active_gates)
    base_wrench = _compute_base_centroidal_wrench(snapshot, b_eff)
    outside_tau = _compute_outside_loop_torques(snapshot, b_eff)
    tangent_load, metric_tau = _compute_tangent_and_metric_projections(
        snapshot, b_eff, metric_W, metric_convention_id
    )
    return InvariantReport(
        net_bare_club_wrench=club_wrench,
        summed_loop_power=loop_power,
        base_centroidal_wrench=base_wrench,
        outside_loop_torques=outside_tau,
        tangent_generalized_load=tangent_load,
        metric_projected_torque=metric_tau,
    )


__all__ = [
    "DEFAULT_METRIC_CONVENTION_ID",
    "TANGENT_ORTHONORMAL_CONVENTION_ID",
    "InvariantGates",
    "InvariantReport",
    "compute_invariants",
]
