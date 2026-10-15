#!/usr/bin/env python3
"""Generate deterministic SymPy golden fixtures for NSX-00 (#12549)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import sympy as sp

OUTPUT_PATH = Path(__file__).resolve().with_name("golden_fixtures.json")


def _to_float_list(matrix: sp.Matrix, *, keep_2d: bool = False) -> list[Any]:
    """Convert a SymPy Matrix to nested Python floats (1-D if column vector unless keep_2d)."""
    if matrix.cols == 1 and not keep_2d:
        return [float(sp.N(val, 20)) for val in matrix]
    return [
        [float(sp.N(matrix[r, c], 20)) for c in range(matrix.cols)]
        for r in range(matrix.rows)
    ]


def _derive_planar_two_arm_fixed_base() -> dict[str, Any]:
    """Derive R2 dimension counts and null-space on a 5-DOF planar two-arm + club loop."""
    # Generalized coordinates: q = [q_L1, q_L2, q_club, q_R1, q_R2] (n_v = 5)
    # Full actuation: S = I_5 (n_tau = 5, n_u = 0)
    # Closure constraint (PLANAR_POINT2, m_c = 2): trail wrist meets grip point
    J_c = sp.Matrix(
        [
            [1, sp.Rational(1, 2), sp.Rational(1, 4), -1, sp.Rational(-1, 2)],
            [0, sp.Rational(3, 4), sp.Rational(1, 2), 0, sp.Rational(-3, 4)],
        ]
    )
    S_T = sp.eye(5)
    A = sp.Matrix.hstack(S_T, J_c.T)
    b = sp.Matrix(
        [
            sp.Rational(12, 5),
            sp.Rational(-7, 5),
            sp.Rational(3, 5),
            sp.Rational(9, 5),
            sp.Rational(-4, 5),
        ]
    )

    # Particular solution via exact Moore-Penrose pseudoinverse A^+ b
    A_pinv = A.T * (A * A.T).inv()
    x_p = A_pinv * b

    # Exact rational null-space basis columns
    null_vecs = A.nullspace()
    N = sp.Matrix.hstack(*null_vecs)

    # Symmetric positive-definite mass matrix M and bias h
    M = sp.Matrix(
        [
            [sp.Rational(5, 2), sp.Rational(1, 2), sp.Rational(1, 5), 0, 0],
            [sp.Rational(1, 2), sp.Rational(3, 2), sp.Rational(1, 4), 0, 0],
            [sp.Rational(1, 5), sp.Rational(1, 4), sp.Rational(4, 5), 0, 0],
            [0, 0, 0, sp.Rational(2, 1), sp.Rational(2, 5)],
            [0, 0, 0, sp.Rational(2, 5), sp.Rational(6, 5)],
        ]
    )
    h = sp.Matrix(
        [
            sp.Rational(1, 2),
            sp.Rational(-1, 4),
            sp.Rational(1, 10),
            sp.Rational(3, 10),
            sp.Rational(-1, 5),
        ]
    )
    Jdot_v = sp.Matrix([sp.Rational(1, 8), sp.Rational(-1, 12)])

    M_inv = M.inv()
    Lambda_c = (J_c * M_inv * J_c.T).inv()
    # Minimum-M^{-1}-norm torque (convention C-METRIC-MINV):
    # minimizes tau^T M^{-1} tau subject to tau + J_c^T lambda = b
    P_Minv = sp.eye(5) - J_c.T * Lambda_c * J_c * M_inv
    tau_min_Minv = P_Minv * b

    # Constrained ZTCF acceleration (tau = 0):
    # M a_ztcf + h = J_c^T lambda_ztcf, J_c a_ztcf + Jdot_v = 0
    a_ztcf = (
        -M_inv * (sp.eye(5) - J_c.T * Lambda_c * J_c * M_inv) * h
        - M_inv * J_c.T * Lambda_c * Jdot_v
    )

    return {
        "description": "5-DOF planar two-arm + club loop with full actuation and 2-D point closure (PLANAR_POINT2)",
        "closure_kind": "PLANAR_POINT2",
        "A": _to_float_list(A),
        "b": _to_float_list(b),
        "J_c": _to_float_list(J_c),
        "Jdot_v": _to_float_list(Jdot_v),
        "M": _to_float_list(M),
        "h": _to_float_list(h),
        "x_particular": _to_float_list(x_p),
        "null_basis": _to_float_list(N, keep_2d=True),
        "tau_min_Minv": _to_float_list(tau_min_Minv),
        "a_ztcf": _to_float_list(a_ztcf),
        "lambda_zero_feasible": True,
        "dimensions": {
            "n_v": 5,
            "n_tau": 5,
            "n_u": 0,
            "m_c": 2,
            "m_g": 0,
            "dim_x": 7,
            "rank_A": 5,
            "equality_nullity": 2,
            "multiplier_gauge": 0,
            "rank_Jc": 2,
            "rank_Jc_u": 0,
            "attainable_torque_variation": 2,
            "feasible_set_dim": 2,
        },
    }


def _derive_planar_two_arm_underactuated() -> dict[str, Any]:
    """Derive R2 dimension counts when club hinge angle (DOF 2) is unactuated."""
    J_c = sp.Matrix(
        [
            [1, sp.Rational(1, 2), sp.Rational(1, 4), -1, sp.Rational(-1, 2)],
            [0, sp.Rational(3, 4), sp.Rational(1, 2), 0, sp.Rational(-3, 4)],
        ]
    )
    # Actuated DOFs: [0, 1, 3, 4]; DOF 2 (club angle) is unactuated
    S_T = sp.Matrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    A = sp.Matrix.hstack(S_T, J_c.T)
    b = sp.Matrix(
        [
            sp.Rational(12, 5),
            sp.Rational(-7, 5),
            sp.Rational(3, 5),
            sp.Rational(9, 5),
            sp.Rational(-4, 5),
        ]
    )

    A_pinv = A.T * (A * A.T).inv()
    x_p = A_pinv * b
    null_vecs = A.nullspace()
    N = sp.Matrix.hstack(*null_vecs)

    # Unactuated row is index 2: J_{c,u} = J_c[:, 2] (2x1, rank 1)
    J_c_u = J_c[:, 2]
    rank_Jc = J_c.rank()
    rank_Jc_u = J_c_u.rank()

    return {
        "description": "5-DOF planar two-arm + unactuated club angle (n_tau=4, n_u=1, m_c=2)",
        "closure_kind": "PLANAR_POINT2",
        "actuated_indices": [0, 1, 3, 4],
        "unactuated_indices": [2],
        "A": _to_float_list(A),
        "b": _to_float_list(b),
        "J_c": _to_float_list(J_c),
        "x_particular": _to_float_list(x_p),
        "null_basis": _to_float_list(N, keep_2d=True),
        "lambda_zero_feasible": False,
        "dimensions": {
            "n_v": 5,
            "n_tau": 4,
            "n_u": 1,
            "m_c": 2,
            "m_g": 0,
            "dim_x": 6,
            "rank_A": 5,
            "equality_nullity": 1,
            "multiplier_gauge": 0,
            "rank_Jc": int(rank_Jc),
            "rank_Jc_u": int(rank_Jc_u),
            "attainable_torque_variation": int(rank_Jc - rank_Jc_u),
            "feasible_set_dim": 1,
        },
    }


def _derive_floating_base_two_arm() -> dict[str, Any]:
    """Derive R2 dimension counts for a planar floating base (n_u=3) + 4 arm DOFs + 2 feet."""
    # DOFs 0, 1, 2: unactuated planar base (x, z, pitch)
    # DOFs 3, 4, 5, 6: actuated upper-limb joints
    S_T = sp.Matrix.vstack(sp.zeros(3, 4), sp.eye(4))
    # Internal hand-to-hand closure does not depend on base pose directly
    J_c = sp.Matrix(
        [
            [0, 0, 0, 1, sp.Rational(1, 2), -1, sp.Rational(-1, 2)],
            [0, 0, 0, 0, sp.Rational(3, 4), 0, sp.Rational(-3, 4)],
        ]
    )
    # Two planar point feet on ground (m_g = 4) acting on base DOFs
    J_g = sp.Matrix(
        [
            [1, 0, sp.Rational(-1, 2), 0, 0, 0, 0],
            [0, 1, sp.Rational(-1, 3), 0, 0, 0, 0],
            [1, 0, sp.Rational(1, 2), 0, 0, 0, 0],
            [0, 1, sp.Rational(1, 3), 0, 0, 0, 0],
        ]
    )
    A = sp.Matrix.hstack(S_T, J_c.T, J_g.T)
    b = sp.Matrix(
        [
            sp.Rational(3, 2),
            sp.Rational(19, 2),
            sp.Rational(1, 2),
            2,
            -1,
            1,
            sp.Rational(-1, 2),
        ]
    )

    A_pinv = A.T * (A * A.T).inv()
    x_p = A_pinv * b
    null_vecs = A.nullspace()
    N = sp.Matrix.hstack(*null_vecs)

    return {
        "description": "7-DOF planar floating-base model (n_u=3, n_tau=4, m_c=2 hand weld, m_g=4 two-foot contact)",
        "closure_kind": "PLANAR_POINT2",
        "actuated_indices": [3, 4, 5, 6],
        "unactuated_indices": [0, 1, 2],
        "A": _to_float_list(A),
        "b": _to_float_list(b),
        "J_c": _to_float_list(J_c),
        "J_g": _to_float_list(J_g),
        "x_particular": _to_float_list(x_p),
        "null_basis": _to_float_list(N, keep_2d=True),
        "base_centroidal_wrench_invariant": _to_float_list(b[:3, :]),
        "dimensions": {
            "n_v": 7,
            "n_tau": 4,
            "n_u": 3,
            "m_c": 2,
            "m_g": 4,
            "dim_x": 10,
            "rank_A": 7,
            "equality_nullity": 3,
            "multiplier_gauge": 0,
            "rank_Jc": 2,
            "rank_Jc_u": 0,
            "attainable_torque_variation": 2,
            "feasible_set_dim": 3,
        },
    }


def _derive_straight_arm_rank_deficient() -> dict[str, Any]:
    """Derive R2 dimension counts at a straight-arm singularity where rank(J_c) < m_c."""
    # 4-DOF planar two-arm colinear along X-axis: first row of J_c vanishes!
    J_c = sp.Matrix(
        [
            [0, 0, 0, 0],
            [1, sp.Rational(1, 2), -1, sp.Rational(-1, 2)],
        ]
    )
    S_T = sp.eye(4)
    A = sp.Matrix.hstack(S_T, J_c.T)
    b = sp.Matrix([1, sp.Rational(1, 2), -1, sp.Rational(-1, 2)])

    A_pinv = A.T * (A * A.T).inv()
    x_p = A_pinv * b
    null_vecs = A.nullspace()
    N = sp.Matrix.hstack(*null_vecs)

    rank_Jc = J_c.rank()
    m_c = J_c.rows

    return {
        "description": "4-DOF straight-arm singular configuration where m_c=2 but rank(J_c)=1 (multiplier_gauge=1)",
        "closure_kind": "PLANAR_POINT2",
        "A": _to_float_list(A),
        "b": _to_float_list(b),
        "J_c": _to_float_list(J_c),
        "x_particular": _to_float_list(x_p),
        "null_basis": _to_float_list(N, keep_2d=True),
        "dimensions": {
            "n_v": 4,
            "n_tau": 4,
            "n_u": 0,
            "m_c": int(m_c),
            "m_g": 0,
            "dim_x": 6,
            "rank_A": 4,
            "equality_nullity": 2,
            "multiplier_gauge": int(m_c - rank_Jc),
            "rank_Jc": int(rank_Jc),
            "rank_Jc_u": 0,
            "attainable_torque_variation": int(rank_Jc),
            "feasible_set_dim": 2,
        },
    }


def _derive_zero_side_cases() -> dict[str, Any]:
    """Derive R3 three-case zero-side solve examples."""
    # Case 1: CONSISTENT_UNIQUE (n_side = 2, m_c = 2, rank = 2)
    J_s1_T = sp.Matrix([[1, 0], [sp.Rational(1, 2), sp.Rational(3, 4)]])
    lam1 = sp.Matrix([2, 4])
    tau_s1 = J_s1_T * lam1

    # Case 2: CONSISTENT_NONUNIQUE (n_side = 2, m_c = 3 planar weld on 2-DOF arm, rank = 2, ker = 1)
    J_s2_T = sp.Matrix([[1, 0, 1], [0, 1, 1]])
    tau_s2 = sp.Matrix([3, 5])
    J_s2_T_pinv = J_s2_T.T * (J_s2_T * J_s2_T.T).inv()
    lam2_min_norm = J_s2_T_pinv * tau_s2

    # Case 3: INCONSISTENT (n_side = 3 trail arm DOFs, m_c = 2 point closure, rank = 2)
    J_s3_T = sp.Matrix(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )
    tau_s3 = sp.Matrix([1, 2, 6])  # [1, 2, 3] is in range; +3 orthogonal component
    J_s3_T_pinv = (J_s3_T.T * J_s3_T).inv() * J_s3_T.T
    lam3_ls = J_s3_T_pinv * tau_s3
    tau_s3_residual = (sp.eye(3) - J_s3_T * J_s3_T_pinv) * tau_s3
    res_norm = sp.sqrt(tau_s3_residual.dot(tau_s3_residual))

    return {
        "consistent_unique": {
            "status": "CONSISTENT_UNIQUE",
            "J_side_T": _to_float_list(J_s1_T),
            "tau_id_side": _to_float_list(tau_s1),
            "lambda_solution": _to_float_list(lam1),
            "kernel_dim": 0,
            "residual_vector": [0.0, 0.0],
            "residual_norm": 0.0,
        },
        "consistent_nonunique": {
            "status": "CONSISTENT_NONUNIQUE",
            "J_side_T": _to_float_list(J_s2_T),
            "tau_id_side": _to_float_list(tau_s2),
            "lambda_min_norm": _to_float_list(lam2_min_norm),
            "kernel_basis": _to_float_list(J_s2_T.nullspace()[0]),
            "kernel_dim": 1,
            "residual_vector": [0.0, 0.0],
            "residual_norm": 0.0,
        },
        "inconsistent_overdetermined": {
            "status": "INCONSISTENT",
            "J_side_T": _to_float_list(J_s3_T),
            "tau_id_side": _to_float_list(tau_s3),
            "lambda_least_squares": _to_float_list(lam3_ls),
            "kernel_dim": 0,
            "residual_vector": _to_float_list(tau_s3_residual),
            "residual_norm": float(sp.N(res_norm, 20)),
        },
    }


def _derive_grasp_map_6x6() -> dict[str, Any]:
    """Derive R4 6x6 grasp-map kernel K, 12x12 split map T, and worked example."""
    # Half-grip separation h = 0.04 m (8 cm between hand grip points along u_hat = [1, 0, 0]^T)
    h = sp.Rational(1, 25)  # 0.04 m
    u_skew = sp.Matrix([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    I3 = sp.eye(3)
    Z3 = sp.zeros(3, 3)

    # Grasp map G (6x12) mapping [F_L; tau_L; F_R; tau_R] -> [F_net; M_M] at midpoint r_M
    # r_L - r_M = -h u_hat, r_R - r_M = +h u_hat
    G = sp.Matrix.vstack(
        sp.Matrix.hstack(I3, Z3, I3, Z3),
        sp.Matrix.hstack(-h * u_skew, I3, h * u_skew, I3),
    )

    # Particular right inverse G_p (12x6): symmetric split of net wrench at midpoint
    half = sp.Rational(1, 2)
    G_p = sp.Matrix.vstack(
        sp.Matrix.hstack(half * I3, Z3),
        sp.Matrix.hstack(Z3, half * I3),
        sp.Matrix.hstack(half * I3, Z3),
        sp.Matrix.hstack(Z3, half * I3),
    )

    # Kernel basis K (12x6) mapping split coordinates s = [s_1..s_6]^T -> [W_L; W_R]:
    # s_1..s_3 (N): internal force difference 1/2 (F_R - F_L) paired with compensating symmetric free moment -h (u_hat x s_f)
    # s_4..s_6 (N*m): differential free moment 1/2 (tau_R - tau_L)
    K = sp.Matrix.vstack(
        sp.Matrix.hstack(-I3, Z3),
        sp.Matrix.hstack(-h * u_skew, -I3),
        sp.Matrix.hstack(I3, Z3),
        sp.Matrix.hstack(-h * u_skew, I3),
    )

    assert (G * K).is_zero_matrix
    assert (G * G_p - sp.eye(6)).is_zero_matrix

    T_12x12 = sp.Matrix.hstack(G_p, K)
    T_12x12_inv = T_12x12.inv()

    # Worked numeric example
    W_net = sp.Matrix([120, -40, 80, 6, -18, 24])
    s_vec = sp.Matrix([35, -15, 10, 2, -3, 4])
    W_stacked = T_12x12 * sp.Matrix.vstack(W_net, s_vec)
    W_L = W_stacked[:6, :]
    W_R = W_stacked[6:, :]

    return {
        "half_grip_separation_m": float(h),
        "frame_axes": {
            "u_hat": [1.0, 0.0, 0.0],
            "v_hat": [0.0, 1.0, 0.0],
            "w_hat": [0.0, 0.0, 1.0],
        },
        "split_coordinate_names": [
            "s1_axial_internal_force_n",
            "s2_transverse_v_force_shear_couple_n",
            "s3_transverse_w_force_shear_couple_n",
            "s4_axial_twist_diff_moment_nm",
            "s5_bending_v_diff_moment_nm",
            "s6_bending_w_diff_moment_nm",
        ],
        "G": _to_float_list(G),
        "G_particular": _to_float_list(G_p),
        "K": _to_float_list(K),
        "T_12x12": _to_float_list(T_12x12),
        "T_12x12_inv": _to_float_list(T_12x12_inv),
        "worked_example": {
            "net_wrench": _to_float_list(W_net),
            "split_coordinates": _to_float_list(s_vec),
            "wrench_left": _to_float_list(W_L),
            "wrench_right": _to_float_list(W_R),
        },
    }


def _derive_bare_club_partition() -> dict[str, Any]:
    """Derive R4 & NSX-01 bare-club mass and CoM partition from composite Clubface Vector."""
    club_solids = {
        "Clubhead": sp.Rational(198, 1000),
        "Rigid Shaft": sp.Rational(65, 1000),
        "Grip": sp.Rational(50, 1000),
    }
    hand_solids = {
        "LHand": sp.Rational(5491, 10000),
        "LHandStandoff": sp.Rational(10, 1000),
        "RHand": sp.Rational(5491, 10000),
    }
    bare_mass = sum(club_solids.values(), sp.Integer(0))
    hands_mass = sum(hand_solids.values(), sp.Integer(0))
    composite_mass = bare_mass + hands_mass

    return {
        "spec_body_name": "solid_reference:GolfSwing3D_Kinetic/Club/Clubface Vector",
        "club_solids_kg": {k: float(v) for k, v in club_solids.items()},
        "welded_hand_solids_kg": {k: float(v) for k, v in hand_solids.items()},
        "bare_club_mass_kg": float(bare_mass),
        "welded_hands_mass_kg": float(hands_mass),
        "composite_mass_kg": float(composite_mass),
        "hand_mass_fraction_of_composite": float(sp.N(hands_mass / composite_mass, 20)),
    }


def _derive_c_plus_minus_and_envelope() -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive R6 c_+/c_- identity and NSX-12/NSX-22 analytic static envelope golden."""
    # Planar 5-DOF fixed-base loop: J_c v = 0
    J_c = sp.Matrix(
        [
            [1, sp.Rational(1, 2), sp.Rational(1, 4), -1, sp.Rational(-1, 2)],
            [0, sp.Rational(3, 4), sp.Rational(1, 2), 0, sp.Rational(-3, 4)],
        ]
    )
    # Pick v in ker(J_c)
    v_null = J_c.nullspace()
    v = 2 * v_null[0] - v_null[1]
    assert (J_c * v).is_zero_matrix

    b = sp.Matrix(
        [
            sp.Rational(12, 5),
            sp.Rational(-7, 5),
            sp.Rational(3, 5),
            sp.Rational(9, 5),
            sp.Rational(-4, 5),
        ]
    )
    z1 = sp.Matrix([sp.Rational(1, 2), sp.Rational(-3, 4)])
    z2 = sp.Matrix([sp.Rational(-5, 4), sp.Rational(7, 4)])
    tau1 = b - J_c.T * z1
    tau2 = b - J_c.T * z2

    p1 = [v[i] * tau1[i] for i in range(5)]
    p2 = [v[i] * tau2[i] for i in range(5)]
    P_net_1 = sum(p1, sp.Integer(0))
    P_net_2 = sum(p2, sp.Integer(0))
    assert sp.simplify(P_net_1 - P_net_2) == 0

    c_plus = sp.Rational(5, 4)
    c_minus = sp.Rational(1, 4)
    lhs = sum(
        (c_plus * sp.Max(pi, 0) + c_minus * sp.Max(-pi, 0) for pi in p1),
        sp.Integer(0),
    )
    rhs = (
        sp.Rational(1, 2)
        * (c_plus + c_minus)
        * sum((sp.Abs(pi) for pi in p1), sp.Integer(0))
        + sp.Rational(1, 2) * (c_plus - c_minus) * P_net_1
    )
    assert sp.simplify(lhs - rhs) == 0

    c_pm_golden = {
        "c_plus": float(c_plus),
        "c_minus": float(c_minus),
        "v_admissible": _to_float_list(v),
        "net_power_at_z1": float(sp.N(P_net_1, 20)),
        "net_power_at_z2": float(sp.N(P_net_2, 20)),
        "lhs_asymmetric_sum": float(sp.N(lhs, 20)),
        "rhs_decomposed_sum": float(sp.N(rhs, 20)),
    }

    # Analytic 2-D static envelope case: z in [-2, 4] x [1, 5]
    # Coordinate 0 spans [-2, 4] -> INDETERMINATE sign
    # Coordinate 1 spans [1, 5]  -> POSITIVE sign
    # Chebyshev ball in [-2, 4] x [1, 5] has center [1, 3] and radius min(3, 2) = 2
    envelope_golden = {
        "z_min": [-2.0, 1.0],
        "z_max": [4.0, 5.0],
        "sign_class": ["INDETERMINATE", "POSITIVE"],
        "chebyshev_center": [1.0, 3.0],
        "chebyshev_radius": 2.0,
    }
    return c_pm_golden, envelope_golden


def build_golden_fixtures() -> dict[str, Any]:
    """Build the complete NSX-00 golden fixtures dictionary."""
    c_pm_golden, envelope_golden = _derive_c_plus_minus_and_envelope()
    return {
        "schema_version": "1.0.0",
        "issue": 12549,
        "generator": "docs/research/loop_nullspace/derivations/generate_goldens.py",
        "fixtures": {
            "planar_two_arm_fixed_base": _derive_planar_two_arm_fixed_base(),
            "planar_two_arm_underactuated": _derive_planar_two_arm_underactuated(),
            "floating_base_two_arm": _derive_floating_base_two_arm(),
            "straight_arm_rank_deficient": _derive_straight_arm_rank_deficient(),
            "zero_side_cases": _derive_zero_side_cases(),
            "grasp_map_6x6": _derive_grasp_map_6x6(),
            "bare_club_partition": _derive_bare_club_partition(),
            "c_plus_minus_identity": c_pm_golden,
            "analytic_envelope_case": envelope_golden,
        },
    }


def main() -> int:
    """Regenerate golden_fixtures.json deterministically."""
    payload = build_golden_fixtures()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    sys.stdout.write(f"Wrote {OUTPUT_PATH}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
