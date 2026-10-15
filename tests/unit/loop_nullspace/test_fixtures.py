"""Unit tests for analytic closed-loop oracle fixtures (#12557, NSX-12)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.contracts import ClosureKind
from src.shared.python.loop_nullspace.fixtures import (
    analytic_envelope_case,
    audit_golfer_constraints_rank,
    planar_golfer_snapshot,
    spatial_two_arm_club_snapshot,
    straight_arm_snapshot,
)

pytestmark = pytest.mark.unit


def test_golfer_constraints_rank_2_audit() -> None:
    audit = audit_golfer_constraints_rank()
    assert audit.full_shape == (4, 8)
    assert audit.rank == 2
    assert audit.independent_rows == (0, 1)
    assert audit.redundant_rows == (2, 3)
    assert audit.zero_columns == (3, 6)
    assert audit.projection_residual < 1e-12
    assert audit.jacobian_projection_residual < 1e-12
    assert audit.independent_jacobian.shape == (2, 8)
    assert np.linalg.matrix_rank(audit.independent_jacobian) == 2


def test_planar_golfer_snapshot_full_actuation_reproduces_goldens() -> None:
    snapshot, golden = planar_golfer_snapshot(actuation="full")
    assert snapshot.topology.closure_kind is ClosureKind.PLANAR_POINT2
    assert snapshot.nv() == 5
    assert snapshot.nc() == 2
    assert snapshot.nu() == 0

    b = snapshot.generalized_load()
    A = np.hstack([snapshot.S.T, snapshot.J_c.T])
    np.testing.assert_allclose(b, golden.b, atol=1e-12)
    np.testing.assert_allclose(A, golden.A, atol=1e-12)
    np.testing.assert_allclose(snapshot.J_c, golden.J_c, atol=1e-12)
    np.testing.assert_allclose(snapshot.Jdot_v, golden.Jdot_v, atol=1e-12)
    np.testing.assert_allclose(snapshot.M, golden.M, atol=1e-12)
    np.testing.assert_allclose(snapshot.h, golden.h, atol=1e-12)

    np.testing.assert_allclose(A @ golden.x_particular, b, atol=1e-12)
    np.testing.assert_allclose(A @ golden.null_basis, np.zeros((5, 2)), atol=1e-12)
    assert golden.dimensions["equality_nullity"] == 2
    assert golden.dimensions["multiplier_gauge"] == 0
    assert golden.dimensions["attainable_torque_variation"] == 2
    assert golden.lambda_zero_feasible is True

    M_inv = np.linalg.inv(snapshot.M)
    Lambda_c = np.linalg.inv(snapshot.J_c @ M_inv @ snapshot.J_c.T)
    P_Minv = np.eye(5) - snapshot.J_c.T @ Lambda_c @ snapshot.J_c @ M_inv
    tau_min_Minv = P_Minv @ b
    a_ztcf = (
        -M_inv @ P_Minv @ snapshot.h
        - M_inv @ snapshot.J_c.T @ Lambda_c @ snapshot.Jdot_v
    )
    assert golden.tau_min_Minv is not None
    assert golden.a_ztcf is not None
    np.testing.assert_allclose(tau_min_Minv, golden.tau_min_Minv, atol=1e-12)
    np.testing.assert_allclose(a_ztcf, golden.a_ztcf, atol=1e-12)


def test_planar_golfer_snapshot_club_unactuated_and_weld3() -> None:
    snapshot, golden = planar_golfer_snapshot(actuation="club_unactuated")
    assert snapshot.nv() == 5
    assert snapshot.nc() == 2
    assert snapshot.nu() == 1
    assert snapshot.topology.unactuated_indices == (2,)
    assert golden.lambda_zero_feasible is False

    b = snapshot.generalized_load()
    A = np.hstack([snapshot.S.T, snapshot.J_c.T])
    np.testing.assert_allclose(A, golden.A, atol=1e-12)
    np.testing.assert_allclose(b, golden.b, atol=1e-12)
    np.testing.assert_allclose(A @ golden.x_particular, b, atol=1e-12)
    np.testing.assert_allclose(A @ golden.null_basis, np.zeros((5, 1)), atol=1e-12)
    assert golden.dimensions["equality_nullity"] == 1
    assert golden.dimensions["rank_Jc_u"] == 1
    assert golden.dimensions["attainable_torque_variation"] == 1

    snap_w3, gold_w3 = planar_golfer_snapshot(
        actuation="full", closure_kind="PLANAR_WELD3"
    )
    assert snap_w3.topology.closure_kind is ClosureKind.PLANAR_WELD3
    assert snap_w3.nc() == 3
    A_w3 = np.hstack([snap_w3.S.T, snap_w3.J_c.T])
    np.testing.assert_allclose(
        A_w3 @ gold_w3.x_particular, snap_w3.generalized_load(), atol=1e-12
    )
    np.testing.assert_allclose(
        A_w3 @ gold_w3.null_basis,
        np.zeros((snap_w3.nv(), gold_w3.null_basis.shape[1])),
        atol=1e-12,
    )


def test_straight_arm_snapshot_rank_deficient() -> None:
    snapshot, golden = straight_arm_snapshot()
    assert snapshot.nv() == 4
    assert snapshot.nc() == 2
    assert np.linalg.matrix_rank(snapshot.J_c) == 1
    assert golden.dimensions["rank_Jc"] == 1
    assert golden.dimensions["multiplier_gauge"] == 1
    assert golden.dimensions["equality_nullity"] == 2
    assert golden.dimensions["attainable_torque_variation"] == 1

    A = np.hstack([snapshot.S.T, snapshot.J_c.T])
    b = snapshot.generalized_load()
    np.testing.assert_allclose(A, golden.A, atol=1e-12)
    np.testing.assert_allclose(b, golden.b, atol=1e-12)
    np.testing.assert_allclose(A @ golden.x_particular, b, atol=1e-12)
    np.testing.assert_allclose(A @ golden.null_basis, np.zeros((4, 2)), atol=1e-12)


def test_spatial_two_arm_club_snapshot_fixed_and_floating() -> None:
    snapshot, golden = spatial_two_arm_club_snapshot(
        floating_base=False, include_feet=False
    )
    assert snapshot.topology.closure_kind is ClosureKind.WELD6
    assert snapshot.nv() == 6
    assert snapshot.nc() == 6
    assert snapshot.nu() == 0
    assert abs(snapshot.bare_club.mass_kg - 0.313) <= 1e-12
    hand_mass = sum(h.mass_kg for h in snapshot.welded_hand_inertias)
    assert abs(hand_mass - 1.1082) <= 1e-12

    A = np.hstack([snapshot.S.T, snapshot.J_c.T])
    b = snapshot.generalized_load()
    np.testing.assert_allclose(A @ golden.x_particular, b, atol=1e-12)
    np.testing.assert_allclose(
        A @ golden.null_basis,
        np.zeros((snapshot.nv(), golden.null_basis.shape[1])),
        atol=1e-12,
    )

    snap_fb, gold_fb = spatial_two_arm_club_snapshot(
        floating_base=True, include_feet=True
    )
    assert snap_fb.nv() == 12
    assert snap_fb.nu() == 6
    assert snap_fb.nc() == 6
    assert snap_fb.J_g is not None
    assert snap_fb.J_g.shape == (6, 12)
    assert len(snap_fb.foot_contacts) == 2
    A_fb = np.hstack([snap_fb.S.T, snap_fb.J_c.T, snap_fb.J_g.T])
    np.testing.assert_allclose(
        A_fb @ gold_fb.x_particular, snap_fb.generalized_load(), atol=1e-12
    )
    np.testing.assert_allclose(
        A_fb @ gold_fb.null_basis,
        np.zeros((snap_fb.nv(), gold_fb.null_basis.shape[1])),
        atol=1e-12,
    )


def test_analytic_envelope_case_closed_form_bounds() -> None:
    snapshot, golden = analytic_envelope_case()
    assert snapshot.nv() == 5
    assert snapshot.nc() == 2
    assert golden.z_min is not None
    assert golden.z_max is not None
    assert golden.chebyshev_center is not None
    assert golden.chebyshev_radius is not None
    assert golden.tau_min is not None
    assert golden.tau_max is not None
    assert golden.lambda_min is not None
    assert golden.lambda_max is not None

    np.testing.assert_allclose(golden.z_min, np.array([-2.0, 1.0]), atol=1e-12)
    np.testing.assert_allclose(golden.z_max, np.array([4.0, 5.0]), atol=1e-12)
    assert golden.sign_class == ("INDETERMINATE", "POSITIVE")
    np.testing.assert_allclose(
        golden.chebyshev_center, np.array([1.0, 3.0]), atol=1e-12
    )
    assert abs(golden.chebyshev_radius - 2.0) <= 1e-12

    # Check corners of z in [-2, 4] x [1, 5] reproduce tau_min/tau_max and lambda_min/lambda_max
    corners = [
        np.array([z0, z1], dtype=np.float64) for z0 in (-2.0, 4.0) for z1 in (1.0, 5.0)
    ]
    x_corners = np.stack([golden.x_particular + golden.null_basis @ z for z in corners])
    np.testing.assert_allclose(x_corners[:, :5].min(axis=0), golden.tau_min, atol=1e-12)
    np.testing.assert_allclose(x_corners[:, :5].max(axis=0), golden.tau_max, atol=1e-12)
    np.testing.assert_allclose(
        x_corners[:, 5:].min(axis=0), golden.lambda_min, atol=1e-12
    )
    np.testing.assert_allclose(
        x_corners[:, 5:].max(axis=0), golden.lambda_max, atol=1e-12
    )
