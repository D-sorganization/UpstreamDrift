"""Native-to-canonical closure Jacobian conversion (R8), NSX-02 golden.

Canonical convention (pinned in NSX-00):

* rows ``[fx fy fz tx ty tz]`` = ``[lin; ang]`` of the relative twist
  ``V_a - V_b`` (closing hand ``a`` relative to the club grip frame ``b``);
* components expressed in the grip frame (rotation ``R_g``: world <- grip);
* the linear rows are the velocity of the *reference point* ``P_g``, the
  closing-hand grip point;
* ``+J_c^T lambda`` is the generalised force the closure applies to the tree, so
  the closing hand exerts ``-lambda`` on the club.

A native Jacobian is the relative twist with a different row order, frame,
reference point and sign.  Everything is rational so the conversion is exact.
Conversion matrix ``C`` with ``J_c = C J_native``; multipliers transform
contragrediently, ``lambda_native = C^T lambda_c``, which preserves the power
``lambda . (J v)``.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

import _common as cm

R = sp.Rational


def skew(v: sp.Matrix) -> sp.Matrix:
    return sp.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def blockdiag_rot(rot: sp.Matrix) -> sp.Matrix:
    return sp.diag(rot, rot)


def point_shift(delta: sp.Matrix) -> sp.Matrix:
    """Rows for the velocity of a point displaced by ``delta`` on the same body:
    ``lin' = lin + omega x delta = lin - [delta]x ang``."""
    return sp.Matrix.vstack(
        sp.Matrix.hstack(sp.eye(3), -skew(delta)),
        sp.Matrix.hstack(sp.zeros(3), sp.eye(3)),
    )


SWAP = sp.Matrix.vstack(
    sp.Matrix.hstack(sp.zeros(3), sp.eye(3)), sp.Matrix.hstack(sp.eye(3), sp.zeros(3))
)

NATIVE = {
    "pinocchio": {
        "row_order": "lin_ang",
        "frame": "LOCAL",
        "relative_sign": 1,
        "point": "frame_origin",
    },
    "drake": {
        "row_order": "ang_lin",
        "frame": "LOCAL",
        "relative_sign": -1,
        "point": "frame_origin",
    },
    "mujoco": {
        "row_order": "lin_ang",
        "frame": "WORLD",
        "relative_sign": 1,
        "point": "site_xpos",
    },
    "opensim": {
        "row_order": "lin_ang",
        "frame": "WORLD",
        "relative_sign": -1,
        "point": "placement",
    },
}


def native_from_world(name: str, j_w: sp.Matrix, r_f: sp.Matrix) -> sp.Matrix:
    """Forward model (independent of the converter): world relative twist at the
    native reference point to the native Jacobian layout."""
    cfg = NATIVE[name]
    j = j_w * cfg["relative_sign"]
    if cfg["frame"] == "LOCAL":
        j = blockdiag_rot(r_f.T) * j
    if cfg["row_order"] == "ang_lin":
        j = SWAP * j
    return j


def conversion_matrix(
    name: str, r_f: sp.Matrix, r_g: sp.Matrix, p_native: sp.Matrix, p_g: sp.Matrix
) -> sp.Matrix:
    """``C`` with ``J_c = C J_native``: undo the native layout, then express in the grip frame."""
    return _compose(name, r_f, r_g, p_native, p_g)


def _compose(
    name: str, r_f: sp.Matrix, r_g: sp.Matrix, p_native: sp.Matrix, p_g: sp.Matrix
) -> sp.Matrix:
    cfg = NATIVE[name]
    inv = sp.eye(6)
    if cfg["row_order"] == "ang_lin":
        inv = SWAP * inv  # SWAP is an involution
    if cfg["frame"] == "LOCAL":
        inv = blockdiag_rot(r_f) * inv
    inv = inv * cfg["relative_sign"]  # sign is its own inverse
    return blockdiag_rot(r_g.T) * point_shift(p_g - p_native) * inv


def generate() -> dict[str, Any]:
    rz = sp.Matrix([[R(3, 5), R(-4, 5), 0], [R(4, 5), R(3, 5), 0], [0, 0, 1]])
    rx = sp.Matrix([[1, 0, 0], [0, R(5, 13), R(-12, 13)], [0, R(12, 13), R(5, 13)]])
    r_f = rz * rx
    r_g = sp.Matrix([[R(3, 5), R(4, 5), 0], [0, 0, 1], [R(4, 5), R(-3, 5), 0]])
    for r in (r_f, r_g):
        assert r.T * r == sp.eye(3) and r.det() == 1
    p0 = sp.Matrix([R(1, 5), R(-1, 10), R(3, 5)])
    p_g = p0 + sp.Matrix([R(1, 20), R(1, 10), R(-1, 20)])
    j_w = sp.Matrix(
        [
            [1, 0, 2, -1],
            [0, 1, -1, 2],
            [2, 1, 0, 1],
            [R(1, 2), 0, 1, 0],
            [0, R(3, 2), 0, -1],
            [1, -1, R(1, 3), 2],
        ]
    )
    v = sp.Matrix([R(1, 2), -1, 2, R(3, 4)])
    lam_c = sp.Matrix([3, -2, 1, R(1, 2), -1, 4])
    canonical = blockdiag_rot(r_g.T) * point_shift(p_g - p0) * j_w
    out: dict[str, Any] = {}
    for name in NATIVE:
        j_n = native_from_world(name, j_w, r_f)
        c = conversion_matrix(name, r_f, r_g, p0, p_g)
        assert (c * j_n - canonical).applyfunc(sp.simplify) == sp.zeros(6, 4), name
        lam_n = c.T * lam_c
        assert (lam_n.T * j_n * v - lam_c.T * canonical * v).applyfunc(
            sp.simplify
        ) == sp.zeros(1, 1), name
        out[name] = {
            "native_convention": NATIVE[name],
            "J_native": cm.exact_mat(j_n),
            "C_native_to_canonical": cm.exact_mat(c),
            "lambda_native_for_lambda_canonical": cm.exact_mat(lam_n.T)[0],
        }
    return cm.dump(
        "closure_convention",
        {
            "description": "One analytic relative twist expressed through each native convention.",
            "J_world_relative_twist_at_P0": cm.exact_mat(j_w),
            "P0_native_reference_point_world": cm.exact_mat(p0.T)[0],
            "R_native_local_frame_world": cm.exact_mat(r_f),
            "R_grip_world_from_frame": cm.exact_mat(r_g),
            "P_canonical_reference_point_world": cm.exact_mat(p_g.T)[0],
            "J_canonical": cm.exact_mat(canonical),
            "v": cm.exact_mat(v.T)[0],
            "lambda_canonical": cm.exact_mat(lam_c.T)[0],
            "power_lambda_dot_Jv": str((lam_c.T * canonical * v)[0, 0]),
            "natives": out,
            "drift_rule": (
                "On a state with J_c v = 0 the drift Jdot_v transforms with the same matrix C; "
                "off the velocity manifold it differs by terms proportional to J_c v, "
                "which is why admission (R7) precedes allocation."
            ),
        },
    )
