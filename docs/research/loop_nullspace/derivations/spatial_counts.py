"""Spatial two-arm club chain: exact dimension counts (NSX-00 fixture (c)).

Only dimension counts are pinned, not spatial dynamics.  The instance is a
*generic rational geometry*: joint axes are rational unit vectors (Pythagorean
quadruples) and joint centres are rational points, so every Jacobian is an exact
rational matrix and every rank below is an exact sympy rank.  The geometry is
not a closed golf posture; the counts are properties of the structure (a weld
between two points of one kinematic tree) and of generic joint geometry.

Structure
---------
* Optional floating root: coordinates ``(v_o, omega_0)`` are the linear velocity
  of the root origin and the angular velocity, both in the world frame.
* Lead arm and trail arm: 7 revolute joints each (shoulder 3, elbow 2, wrist 2).
* The bare club is rigidly attached to the lead hand (holding hand).  The trail
  hand closes the loop onto a grip frame of the club: closure kind ``WELD6``
  with ``J_c = J_trail(p*) - J_lead(p*)`` evaluated at the common grip point.
* Optional ground contact: two *point* feet rigidly attached to the root (the
  legs are frozen in this count fixture), each contributing a 3-component
  contact force.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

import _common as cm

R = sp.Rational

AXES = [
    (R(1), R(0), R(0)),
    (R(0), R(1), R(0)),
    (R(0), R(0), R(1)),
    (R(2, 3), R(2, 3), R(1, 3)),
    (R(1, 3), R(-2, 3), R(2, 3)),
    (R(2, 7), R(3, 7), R(6, 7)),
    (R(6, 7), R(-2, 7), R(3, 7)),
    (R(-3, 7), R(6, 7), R(2, 7)),
    (R(1, 9), R(4, 9), R(8, 9)),
    (R(4, 9), R(1, 9), R(-8, 9)),
]
P_STAR = sp.Matrix([R(3, 5), R(1, 10), R(-1, 2)])
ROOT = sp.Matrix([0, 0, 0])
FEET = [sp.Matrix([R(1, 5), R(1, 10), R(-1)]), sp.Matrix([R(-3, 10), R(-1, 5), R(-1)])]
# Lower-body variant (NSX-60): a second point per foot, offset in the ground plane.
FOOT_SECOND_OFFSET = [
    sp.Matrix([R(1, 5), R(1, 5), 0]),
    sp.Matrix([R(1, 5), R(-1, 10), 0]),
]
FEET4 = [
    FEET[0],
    FEET[0] + FOOT_SECOND_OFFSET[0],
    FEET[1],
    FEET[1] + FOOT_SECOND_OFFSET[1],
]


def _skew(v: sp.Matrix) -> sp.Matrix:
    return sp.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _arm_columns(shoulder: sp.Matrix, offset: int) -> sp.Matrix:
    """Twist columns (lin; ang) of the end point ``P_STAR`` for a 7-joint arm."""
    elbow = shoulder + sp.Matrix([R(1, 10), 0, R(-3, 10)])
    wrist = elbow + sp.Matrix([R(1, 5), R(1, 10), R(-1, 4)])
    centres = [shoulder] * 3 + [elbow] * 2 + [wrist] * 2
    cols = []
    for k, c in enumerate(centres):
        a = sp.Matrix(AXES[(offset + k) % len(AXES)])
        cols.append(sp.Matrix.vstack(a.cross(P_STAR - c), a))
    return sp.Matrix.hstack(*cols)


def _root_block(point: sp.Matrix) -> sp.Matrix:
    top = sp.Matrix.hstack(sp.eye(3), -_skew(point - ROOT))
    bottom = sp.Matrix.hstack(sp.zeros(3), sp.eye(3))
    return sp.Matrix.vstack(top, bottom)


def build(floating: bool, feet: bool, points_per_foot: int = 1) -> dict[str, Any]:
    lead = _arm_columns(sp.Matrix([R(-1, 10), 0, 0]), 0)
    trail = _arm_columns(sp.Matrix([R(1, 10), 0, 0]), 3)
    root = _root_block(P_STAR) if floating else sp.zeros(6, 0)
    # closure: trail hand twist minus club-grip (= lead hand) twist at p*
    j_c = sp.Matrix.hstack(root - root if floating else root, -lead, trail)
    n = j_c.cols
    n_root = 6 if floating else 0
    unact = list(range(n_root))
    if feet:
        j_g = sp.Matrix.vstack(
            *[
                sp.Matrix.hstack(
                    sp.Matrix.hstack(sp.eye(3), -_skew(f - ROOT)), sp.zeros(3, 14)
                )
                for f in (FEET if points_per_foot == 1 else FEET4)
            ]
        )
    else:
        j_g = sp.zeros(0, n)
    return {
        "J_c": j_c,
        "J_g": j_g,
        "n": n,
        "unactuated": unact,
        "lead": -lead,
        "trail": trail,
    }


def counts(floating: bool, feet: bool, points_per_foot: int = 1) -> dict[str, Any]:
    st = build(floating, feet, points_per_foot)
    j_c, j_g, n = st["J_c"], st["J_g"], st["n"]
    u = st["unactuated"]
    m, n_g = j_c.rows, j_g.rows
    n_a = n - len(u)
    j_hat = sp.Matrix.vstack(j_c, j_g) if n_g else j_c
    a_cols = [i for i in range(n) if i not in u]
    s_t = sp.zeros(n, n_a)
    for r, i in enumerate(a_cols):
        s_t[i, r] = 1
    a_mat = sp.Matrix.hstack(s_t, j_c.T, j_g.T) if n_g else sp.Matrix.hstack(s_t, j_c.T)
    rank_a = a_mat.rank()
    rank_jc = j_c.rank()
    rank_jcu = j_c[:, u].rank() if u else 0
    rank_ghat = j_hat.rank()
    rank_ghat_u = j_hat[:, u].rank() if u else 0
    nullity = n_a + m + n_g - rank_a
    out = {
        "n": n,
        "n_actuated": n_a,
        "n_unactuated": len(u),
        "m": m,
        "n_g": n_g,
        "dim_x": n_a + m + n_g,
        "rank_J_c": rank_jc,
        "rank_J_cu_T": rank_jcu,
        "rank_J_g": j_g.rank() if n_g else 0,
        "rank_stacked_J_c_J_g": rank_ghat,
        "rank_stacked_unactuated_columns": rank_ghat_u,
        "rank_A": rank_a,
        "equality_nullity": nullity,
        "multiplier_gauge": m - rank_jc,
        "attainable_torque_variation_fg_fixed": rank_jc - rank_jcu,
        "gauge_total_fg_free": (m + n_g) - rank_ghat,
        "attainable_torque_variation_fg_free": rank_ghat - rank_ghat_u,
        "nullity_fg_fixed": m - rank_jcu,
        "lead_chain_rank_J_c_T": st["lead"].rank(),
        "trail_chain_rank_J_c_T": st["trail"].rank(),
        "lead_chain_equations": st["lead"].cols,
        "trail_chain_equations": st["trail"].cols,
    }
    if not n_g:
        assert (
            out["equality_nullity"]
            == out["multiplier_gauge"] + out["attainable_torque_variation_fg_fixed"]
        )
    assert (
        out["equality_nullity"]
        == out["gauge_total_fg_free"] + out["attainable_torque_variation_fg_free"]
    )
    if u:
        bal = sp.Matrix.hstack(j_c[:, u].T, j_g[:, u].T) if n_g else j_c[:, u].T
        left_null = bal.T.nullspace()
        out["balance_rank"] = bal.rank()
        out["balance_rows"] = bal.rows
        out["consistency_conditions"] = bal.rows - bal.rank()
        out["consistency_left_null_vectors"] = [cm.exact_mat(v.T)[0] for v in left_null]
    return out


def generate() -> dict[str, Any]:
    variants = {
        "fixed_base_arms_only": counts(False, False),
        "floating_root_no_feet": counts(True, False),
        "floating_root_two_point_feet": counts(True, True),
        "floating_root_four_point_feet": counts(True, True, 2),
    }
    foot_axis = (FEET[0] - FEET[1]).T
    return cm.dump(
        "spatial_two_arm_counts",
        {
            "description": (
                "Exact dimension counts for the spatial two-arm club chain, WELD6, "
                "2 x 7-joint arms, optional 6-DOF floating root and two point feet."
            ),
            "coordinate_order": "root(6, optional) | lead arm (7) | trail arm (7)",
            "geometry_note": "generic rational geometry; counts are structural, not a golf posture",
            "foot_axis_vector": cm.exact_mat(foot_axis)[0],
            "variants": variants,
        },
    )
