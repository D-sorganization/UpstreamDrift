"""Unit tests for NSX-17 invariant quantities with validity gates (#12562)."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from src.shared.python.loop_nullspace.contracts import (
    AllocationDependence,
    ClosedLoopSnapshot,
    LoopTopology,
)
from src.shared.python.loop_nullspace.fixtures import (
    planar_golfer_snapshot,
    spatial_two_arm_club_snapshot,
)
from src.shared.python.loop_nullspace.invariants import (
    InvariantGates,
    InvariantReport,
    compute_invariants,
)

pytestmark = pytest.mark.unit


def _perturb_snapshot_with_multiplier(
    snapshot: ClosedLoopSnapshot, delta_lambda: np.ndarray
) -> ClosedLoopSnapshot:
    """Apply a closure multiplier perturbation to the snapshot's apparent bias h.

    Wait: in ClosedLoopSnapshot, b = M a + h = S^T tau + J_c^T lambda + J_g^T f_g.
    Wait: how does a closure multiplier perturbation affect tau vs snapshot?
    Let's check both perturbing h along J_c^T delta_lambda (when testing tangent projection)
    and passing/perturbing tau -> tau - S J_c^T delta_lambda!
    """
    delta_b = snapshot.J_c.T @ delta_lambda
    return replace(snapshot, h=snapshot.h - delta_b)


def test_invariants_unchanged_under_random_closure_multiplier_perturbations() -> None:
    snapshot, _ = planar_golfer_snapshot(actuation="full")
    base_report = compute_invariants(snapshot)

    assert (
        base_report.net_bare_club_wrench.allocation_dependence
        is AllocationDependence.INVARIANT
    )
    assert (
        base_report.summed_loop_power.allocation_dependence
        is AllocationDependence.INVARIANT
    )
    assert (
        base_report.tangent_generalized_load.allocation_dependence
        is AllocationDependence.INVARIANT
    )
    assert (
        base_report.metric_projected_torque.allocation_dependence
        is AllocationDependence.INVARIANT
    )

    rng = np.random.default_rng(12562)
    m_c = snapshot.closure_rows()
    for _ in range(10):
        delta_lambda = rng.normal(scale=25.0, size=m_c)
        perturbed_snapshot = _perturb_snapshot_with_multiplier(snapshot, delta_lambda)
        perturbed_report = compute_invariants(perturbed_snapshot)

        np.testing.assert_allclose(
            perturbed_report.net_bare_club_wrench.values,
            base_report.net_bare_club_wrench.values,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            perturbed_report.summed_loop_power.values,
            base_report.summed_loop_power.values,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            perturbed_report.tangent_generalized_load.values,
            base_report.tangent_generalized_load.values,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            perturbed_report.metric_projected_torque.values,
            base_report.metric_projected_torque.values,
            atol=1e-10,
        )


def test_summed_loop_power_gate_violation_returns_unavailable_nan() -> None:
    snapshot, _ = planar_golfer_snapshot(actuation="full")
    # Inject a velocity that violates J_c v = 0 beyond the gate
    v_bad = np.array([5.0, 5.0, 5.0, -5.0, -5.0], dtype=np.float64)
    bad_snapshot = replace(snapshot, v=v_bad)

    gates = InvariantGates(
        max_scaled_closure_vel=1e-4,
        max_scaled_closure_acc=1e-3,
        characteristic_speed_m_s=20.0,
    )
    report = compute_invariants(bad_snapshot, gates=gates)
    power_rec = report.summed_loop_power

    assert power_rec.allocation_dependence is AllocationDependence.UNAVAILABLE
    assert power_rec.reason == "closure_velocity_residual_exceeds_gate"
    assert np.all(np.isnan(power_rec.values))


def test_base_centroidal_wrench_fixed_vs_floating_base() -> None:
    fixed_snap, _ = spatial_two_arm_club_snapshot(floating_base=False)
    fixed_rep = compute_invariants(fixed_snap)
    assert (
        fixed_rep.base_centroidal_wrench.allocation_dependence
        is AllocationDependence.UNAVAILABLE
    )
    assert fixed_rep.base_centroidal_wrench.reason == "no_unactuated_floating_base_rows"
    assert np.all(np.isnan(fixed_rep.base_centroidal_wrench.values))

    float_snap, _ = spatial_two_arm_club_snapshot(floating_base=True, include_feet=True)
    float_rep = compute_invariants(float_snap)
    assert (
        float_rep.base_centroidal_wrench.allocation_dependence
        is AllocationDependence.INVARIANT
    )
    assert np.all(np.isfinite(float_rep.base_centroidal_wrench.values))
    assert float_rep.base_centroidal_wrench.values.shape == (6,)

    # When closure couples into unactuated rows (e.g., planar_two_arm_underactuated)
    under_snap, _ = planar_golfer_snapshot(actuation="club_unactuated")
    under_rep = compute_invariants(under_snap)
    assert (
        under_rep.base_centroidal_wrench.allocation_dependence
        is AllocationDependence.UNAVAILABLE
    )
    assert under_rep.base_centroidal_wrench.reason == "closure_couples_unactuated_rows"
    assert np.all(np.isnan(under_rep.base_centroidal_wrench.values))


def test_outside_loop_torques_available_and_unavailable() -> None:
    snapshot, _ = planar_golfer_snapshot(actuation="full")
    rep_all_in_loop = compute_invariants(snapshot)
    assert (
        rep_all_in_loop.outside_loop_torques.allocation_dependence
        is AllocationDependence.UNAVAILABLE
    )
    assert np.all(np.isnan(rep_all_in_loop.outside_loop_torques.values))

    # Add an actuated joint with zero column in J_c (outside the loop)
    J_c_zero_col = snapshot.J_c.copy()
    J_c_zero_col[:, 2] = 0.0
    topo_outside = LoopTopology(
        closure_kind=snapshot.topology.closure_kind,
        closing_side=snapshot.topology.closing_side,
        holding_side=snapshot.topology.holding_side,
        actuated_indices=snapshot.topology.actuated_indices,
        unactuated_indices=snapshot.topology.unactuated_indices,
        loop_support_indices=(0, 1, 3, 4),
        coordinate_order=snapshot.topology.coordinate_order,
    )
    snap_outside = replace(snapshot, J_c=J_c_zero_col, topology=topo_outside)
    rep_outside = compute_invariants(snap_outside)
    assert (
        rep_outside.outside_loop_torques.allocation_dependence
        is AllocationDependence.INVARIANT
    )
    np.testing.assert_allclose(
        rep_outside.outside_loop_torques.values,
        np.array([snap_outside.generalized_load()[2]]),
        atol=1e-12,
    )


def test_metric_projected_torque_convention_and_metric_dependence() -> None:
    snapshot, _ = planar_golfer_snapshot(actuation="full")
    rep_id = compute_invariants(
        snapshot,
        metric_W=np.eye(5, dtype=np.float64),
        metric_convention_id="C-METRIC-IDENTITY",
    )
    W_cap = np.diag([1.0, 4.0, 9.0, 2.0, 0.5])
    rep_cap = compute_invariants(
        snapshot,
        metric_W=W_cap,
        metric_convention_id="C-METRIC-CAPACITY",
    )

    assert "C-METRIC-IDENTITY" in rep_id.metric_projected_torque.conventions
    assert "C-METRIC-CAPACITY" in rep_cap.metric_projected_torque.conventions
    assert not np.allclose(
        rep_id.metric_projected_torque.values,
        rep_cap.metric_projected_torque.values,
    )

    # Both metric projections are invariant to closure multiplier perturbations
    delta_lambda = np.array([15.0, -22.0], dtype=np.float64)
    perturbed = _perturb_snapshot_with_multiplier(snapshot, delta_lambda)
    rep_cap_pert = compute_invariants(
        perturbed,
        metric_W=W_cap,
        metric_convention_id="C-METRIC-CAPACITY",
    )
    np.testing.assert_allclose(
        rep_cap_pert.metric_projected_torque.values,
        rep_cap.metric_projected_torque.values,
        atol=1e-10,
    )


def test_invariant_report_json_serializable() -> None:
    snapshot, _ = spatial_two_arm_club_snapshot(floating_base=True, include_feet=True)
    report = compute_invariants(snapshot)
    assert isinstance(report, InvariantReport)
    payload = report.to_dict()
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert "net_bare_club_wrench" in decoded
    assert "summed_loop_power" in decoded
    assert "base_centroidal_wrench" in decoded
    assert "outside_loop_torques" in decoded
    assert "tangent_generalized_load" in decoded
    assert "metric_projected_torque" in decoded
