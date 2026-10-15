"""Unit tests for closed-loop ForceNullSpace decomposition (NSX-15, #12560)."""

from __future__ import annotations

import json
import numpy as np
import pytest

from src.shared.python.loop_nullspace.constrained_dynamics import (
    constrained_acceleration,
)
from src.shared.python.loop_nullspace.contracts import (
    AllocationDependence,
    ClosedLoopSnapshot,
    ClosureKind,
    LoopTopology,
    QuantityRecord,
)
from src.shared.python.loop_nullspace.decomposition import (
    AllocationMetric,
    Decomposition,
    ZeroSideResult,
    decompose,
    min_metric_torque,
    zero_side_solve,
)
from src.shared.python.loop_nullspace.fixtures import (
    load_golden_fixtures,
    planar_golfer_snapshot,
    spatial_two_arm_club_snapshot,
    straight_arm_snapshot,
)

pytestmark = pytest.mark.unit


def _floating_base_two_arm_oracle() -> tuple[ClosedLoopSnapshot, dict[str, object]]:
    """Construct the 7-DOF planar floating-base two-arm snapshot from NSX-00 golden."""
    fixtures = load_golden_fixtures()
    raw = fixtures["floating_base_two_arm"]
    base_snap, _ = planar_golfer_snapshot()
    topology = LoopTopology(
        closure_kind=ClosureKind.PLANAR_POINT2,
        closing_side="right",
        holding_side="left",
        actuated_indices=(3, 4, 5, 6),
        unactuated_indices=(0, 1, 2),
        loop_support_indices=(3, 4, 5, 6),
        coordinate_order=("bx", "bz", "bpitch", "q_L1", "q_L2", "q_R1", "q_R2"),
    )
    n_v = 7
    S = np.zeros((4, n_v), dtype=np.float64)
    for row_idx, col_idx in enumerate(topology.actuated_indices):
        S[row_idx, col_idx] = 1.0
    M = np.eye(n_v, dtype=np.float64)
    h = np.zeros(n_v, dtype=np.float64)
    b = np.asarray(raw["b"], dtype=np.float64)
    J_c = np.asarray(raw["J_c"], dtype=np.float64)
    J_g = np.asarray(raw["J_g"], dtype=np.float64)
    snap = ClosedLoopSnapshot(
        q=np.zeros(n_v, dtype=np.float64),
        v=np.zeros(n_v, dtype=np.float64),
        a=b.copy(),
        M=M,
        h=h,
        S=S,
        J_c=J_c,
        Jdot_v=np.zeros(2, dtype=np.float64),
        grip_frame=base_snap.grip_frame,
        bare_club=base_snap.bare_club,
        welded_hand_inertias=base_snap.welded_hand_inertias,
        model_provenance=base_snap.model_provenance,
        input_provenance=base_snap.input_provenance,
        topology=topology,
        J_g=J_g,
    )
    return snap, raw


def test_dimension_counts_match_goldens_across_all_four_oracles() -> None:
    """Verify R2 dimension counts against NSX-12 goldens for all four oracle fixtures."""
    snap_full, golden_full = planar_golfer_snapshot(actuation="full")
    snap_under, golden_under = planar_golfer_snapshot(actuation="club_unactuated")
    snap_float, raw_float = _floating_base_two_arm_oracle()
    snap_sing, golden_sing = straight_arm_snapshot()

    cases = [
        (snap_full, golden_full.dimensions, golden_full.x_particular),
        (snap_under, golden_under.dimensions, golden_under.x_particular),
        (snap_float, raw_float["dimensions"], np.asarray(raw_float["x_particular"])),
        (snap_sing, golden_sing.dimensions, golden_sing.x_particular),
    ]
    for snap, expected_dims, expected_xp in cases:
        decomp = decompose(snap)
        assert isinstance(decomp, Decomposition)
        assert decomp.rank_A == expected_dims["rank_A"]
        assert decomp.nullity == expected_dims["equality_nullity"]
        assert decomp.multiplier_gauge == expected_dims["multiplier_gauge"]
        assert (
            decomp.attainable_torque_variation
            == expected_dims["attainable_torque_variation"]
        )
        assert decomp.dimensions == expected_dims
        np.testing.assert_allclose(decomp.x_particular, expected_xp, atol=1e-11)
        # Verify null basis is in kernel of A
        A_mat = decomp.matrix_A
        np.testing.assert_allclose(A_mat @ decomp.null_basis, 0.0, atol=1e-12)


def test_straight_arm_rank_drop_and_spatial_weld_decomposition() -> None:
    """Verify straight-arm rank drop (multiplier_gauge=1) and spatial WELD6 decomposition."""
    snap_sing, _ = straight_arm_snapshot()
    decomp_sing = decompose(snap_sing)
    assert decomp_sing.multiplier_gauge == 1
    assert decomp_sing.attainable_torque_variation == 1
    assert decomp_sing.nullity == 2

    snap_spatial, golden_spatial = spatial_two_arm_club_snapshot(
        floating_base=True, include_feet=True
    )
    decomp_spatial = decompose(snap_spatial)
    assert decomp_spatial.dimensions == golden_spatial.dimensions
    # Per-frame diagnostics must be JSON-serializable and wrapped in QuantityRecord
    payload = json.dumps(decomp_spatial.to_dict())
    assert "rank_A" in payload
    assert isinstance(decomp_spatial.singular_values_record, QuantityRecord)
    assert (
        decomp_spatial.singular_values_record.allocation_dependence
        is AllocationDependence.INVARIANT
    )


def test_zero_side_solve_three_cases_and_inconsistent_residual() -> None:
    """Verify R3 three-case classification and exact INCONSISTENT residual norm."""
    fixtures = load_golden_fixtures()
    zs_goldens = fixtures.get("zero_side_cases", {})

    # 1. CONSISTENT_UNIQUE: planar 5-DOF full actuation, right arm [3, 4] (2x2 rank 2)
    snap_full, _ = planar_golfer_snapshot(actuation="full")
    res_unique = zero_side_solve(snap_full, (3, 4))
    assert isinstance(res_unique, ZeroSideResult)
    assert res_unique.case == "CONSISTENT_UNIQUE"
    assert res_unique.nullity == 0
    assert res_unique.residual_norm == pytest.approx(0.0, abs=1e-12)
    assert res_unique.lambda_solution is not None

    # 2. CONSISTENT_NONUNIQUE: planar 5-DOF with PLANAR_WELD3 (m_c=3) on 2-DOF right arm
    snap_weld3, _ = planar_golfer_snapshot(
        actuation="full", closure_kind=ClosureKind.PLANAR_WELD3
    )
    res_nonunique = zero_side_solve(snap_weld3, "right")
    assert res_nonunique.case == "CONSISTENT_NONUNIQUE"
    assert res_nonunique.nullity == 1
    assert res_nonunique.residual_norm == pytest.approx(0.0, abs=1e-12)
    assert res_nonunique.kernel_basis.shape == (3, 1)

    # 3. INCONSISTENT: overdetermined 3-DOF side against 2-D point closure
    # Matches NSX-00 golden: J_s3_T = [[1, 0], [0, 1], [1, 1]], tau_s3 = [1, 2, 6]
    topology_3 = LoopTopology(
        closure_kind=ClosureKind.PLANAR_POINT2,
        closing_side="right",
        holding_side="left",
        actuated_indices=(0, 1, 2),
        unactuated_indices=(),
        loop_support_indices=(0, 1, 2),
        coordinate_order=("q_R1", "q_R2", "q_R3"),
    )
    J_c_3 = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    b_3 = np.array([1.0, 2.0, 6.0], dtype=np.float64)
    snap_inc = ClosedLoopSnapshot(
        q=np.zeros(3),
        v=np.zeros(3),
        a=b_3,
        M=np.eye(3),
        h=np.zeros(3),
        S=np.eye(3),
        J_c=J_c_3,
        Jdot_v=np.zeros(2),
        grip_frame=snap_full.grip_frame,
        bare_club=snap_full.bare_club,
        welded_hand_inertias=snap_full.welded_hand_inertias,
        model_provenance=snap_full.model_provenance,
        input_provenance=snap_full.input_provenance,
        topology=topology_3,
    )
    res_inc = zero_side_solve(snap_inc, (0, 1, 2))
    assert res_inc.case == "INCONSISTENT"
    assert res_inc.lambda_solution is None
    expected_res_norm = float(
        zs_goldens.get("inconsistent_overdetermined", {}).get(
            "residual_norm", np.sqrt(3.0)
        )
    )
    assert res_inc.residual_norm == pytest.approx(expected_res_norm, abs=1e-12)
    np.testing.assert_allclose(
        res_inc.residual_vector, np.array([-1.0, -1.0, 1.0]), atol=1e-12
    )

    # Also verify underactuated club + right arm is classified INCONSISTENT
    snap_under, _ = planar_golfer_snapshot(actuation="club_unactuated")
    res_under = zero_side_solve(snap_under, "right")
    assert res_under.case == "INCONSISTENT"


def test_forward_kkt_constrained_acceleration_invariance_over_nullspace() -> None:
    """Verify constrained_acceleration(tau_p + N_tau z) reproduces snapshot.a to 1e-9 scaled."""
    snap, _ = planar_golfer_snapshot(actuation="full")
    decomp = decompose(snap)
    n_tau = snap.nv()
    tau_p = decomp.x_particular[:n_tau]
    N_tau = decomp.null_basis[:n_tau, :]
    # Consistent closure drift for snapshot.a is Jdot_v_consistent = -J_c @ a
    jdot_v_consistent = -(snap.J_c @ snap.a)

    rng = np.random.default_rng(12560)
    scale_a = max(float(np.linalg.norm(snap.a)), 1.0)
    for _ in range(10):
        z = rng.normal(scale=10.0, size=(decomp.nullity,))
        tau_z = tau_p + N_tau @ z
        a_kkt, _ = constrained_acceleration(
            snap.M, snap.h, tau_z, snap.J_c, jdot_v_consistent
        )
        scaled_err = float(np.linalg.norm(a_kkt - snap.a)) / scale_a
        assert scaled_err <= 1e-9


def test_min_metric_torque_matches_closed_form_minv_golden() -> None:
    """Verify min_metric_torque with M^-1 metric matches closed-form golden and rejects Gauss label."""
    snap, golden = planar_golfer_snapshot(actuation="full")
    W_minv = np.linalg.inv(snap.M)
    metric = AllocationMetric(convention_id="C-METRIC-MINV", W=W_minv)
    rec = min_metric_torque(snap, metric)

    assert isinstance(rec, QuantityRecord)
    assert rec.allocation_dependence is AllocationDependence.INVARIANT
    assert rec.conventions == ("C-METRIC-MINV",)
    assert golden.tau_min_Minv is not None
    np.testing.assert_allclose(rec.values, golden.tau_min_Minv, atol=1e-12)

    with pytest.raises(ValueError, match="Gauss"):
        AllocationMetric(convention_id="C-GAUSS-PRINCIPLE", W=W_minv)
