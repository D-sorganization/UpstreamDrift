"""Grip frame, 6x12 grasp map, 12x6 kernel and the 6x6 closure-to-split map.

All identities are verified symbolically (sympy, exact), then a rational worked
example is evaluated end to end: bare-club Newton-Euler -> net wrench ->
split coordinates -> hand wrenches -> recovered split.

Conventions pinned here (NSX-00, "Grip Frame and Grasp Map"):

* Hand wrenches are *exerted by the hand on the club* (ADR-0052).  Hand ``h``
  contributes a force ``f_h`` at its grip point ``r_h`` and a free couple
  ``c_h``; the stacked hand load is ``y = (f_L, c_L, f_R, c_R)`` (12-vector).
* Grip frame ``(u, v, w)``: origin ``o = (r_L + r_R)/2``; ``u`` = unit shaft axis
  pointing from the grip toward the clubhead; ``v`` = normalised projection of
  the clubface normal ``n`` onto the plane orthogonal to ``u``; ``w = u x v``.
* In this frame ``rho_R = d = -rho_L`` with ``d = (delta, 0, 0)`` because the
  grip points are, by definition, the projections of the hands onto the shaft
  axis.  The derivations below keep ``d`` a *general* vector so the algebra does
  not rely on that; the physical names of the coordinates do.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

import _common as cm

R = sp.Rational


def skew(v: sp.Matrix) -> sp.Matrix:
    return sp.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def grasp_map(d: sp.Matrix) -> sp.Matrix:
    """``G (6x12)`` mapping ``(f_L, c_L, f_R, c_R)`` to the net wrench about the
    midpoint, with ``rho_L = -d`` and ``rho_R = +d``."""
    i3, z3 = sp.eye(3), sp.zeros(3)
    g_left = sp.Matrix.vstack(sp.Matrix.hstack(i3, z3), sp.Matrix.hstack(-skew(d), i3))
    g_right = sp.Matrix.vstack(sp.Matrix.hstack(i3, z3), sp.Matrix.hstack(skew(d), i3))
    return sp.Matrix.hstack(g_left, g_right)


def split_basis(d: sp.Matrix) -> sp.Matrix:
    """``K (12x6)``; column ``k`` realises unit split coordinate ``s_k``."""
    i3, z3 = sp.eye(3), sp.zeros(3)
    return sp.Matrix.vstack(
        sp.Matrix.hstack(-i3, z3),
        sp.Matrix.hstack(-skew(d), -i3),
        sp.Matrix.hstack(i3, z3),
        sp.Matrix.hstack(-skew(d), i3),
    )


def differential_map() -> sp.Matrix:
    """``Phi (6x12)``: ``s = ((f_R - f_L)/2, (c_R - c_L)/2)``."""
    i3, z3 = sp.eye(3), sp.zeros(3)
    return sp.Matrix.vstack(
        sp.Matrix.hstack(-i3 / 2, z3, i3 / 2, z3),
        sp.Matrix.hstack(z3, -i3 / 2, z3, i3 / 2),
    )


def equal_split_inverse() -> sp.Matrix:
    """``G# (12x6)``: the particular right inverse with ``s = 0`` (origin at the
    midpoint): both hands carry half the net force and half the net moment."""
    i3 = sp.eye(3)
    return sp.Matrix.vstack(i3 / 2, sp.zeros(3, 3), i3 / 2, sp.zeros(3, 3)).row_join(
        sp.Matrix.vstack(sp.zeros(3, 3), i3 / 2, sp.zeros(3, 3), i3 / 2)
    )


def transport(t: sp.Matrix) -> sp.Matrix:
    """Wrench transport ``T(t)`` from reference point ``o`` to ``o + t``:
    ``(f, m_o) -> (f, m_o - t x f)``."""
    return sp.Matrix.vstack(
        sp.Matrix.hstack(sp.eye(3), sp.zeros(3)), sp.Matrix.hstack(-skew(t), sp.eye(3))
    )


def symbolic_identities() -> dict[str, bool]:
    dx, dy, dz = sp.symbols("d_x d_y d_z", real=True)
    d = sp.Matrix([dx, dy, dz])
    g, k, phi, gs = (
        grasp_map(d),
        split_basis(d),
        differential_map(),
        equal_split_inverse(),
    )
    t = sp.Matrix(sp.symbols("t_x t_y t_z", real=True))
    s = sp.Matrix(sp.symbols("s1:7", real=True))
    w = sp.Matrix(sp.symbols("w1:7", real=True))
    y = gs * w + k * s
    # grasp map about the shifted origin o + t: rho' = rho - t, G(o + t) = T(t) G(o)
    g_t = sp.Matrix.hstack(
        sp.Matrix.vstack(
            sp.Matrix.hstack(sp.eye(3), sp.zeros(3)),
            sp.Matrix.hstack(skew(-d - t), sp.eye(3)),
        ),
        sp.Matrix.vstack(
            sp.Matrix.hstack(sp.eye(3), sp.zeros(3)),
            sp.Matrix.hstack(skew(d - t), sp.eye(3)),
        ),
    )
    z = sp.zeros
    return {
        "G_K_equals_0": (g * k).applyfunc(sp.expand) == z(6, 6),
        "Phi_K_equals_I": (phi * k).applyfunc(sp.expand) == sp.eye(6),
        "G_Gsharp_equals_I": (g * gs).applyfunc(sp.expand) == sp.eye(6),
        "Phi_Gsharp_equals_0": (phi * gs).applyfunc(sp.expand) == z(6, 6),
        "stack_inverse_is_Gsharp_K": (
            (sp.Matrix.hstack(gs, k) * sp.Matrix.vstack(g, phi)).applyfunc(sp.expand)
            == sp.eye(12)
        ),
        "G_y_equals_w": (g * y - w).applyfunc(sp.expand) == z(6, 1),
        "Phi_y_equals_s": (phi * y - s).applyfunc(sp.expand) == z(6, 1),
        "G_origin_shift_is_wrench_transport": (g_t - transport(t) * g).applyfunc(
            sp.expand
        )
        == z(6, 12),
        "K_independent_of_origin_shift": (g_t * k).applyfunc(sp.expand) == z(6, 6),
        "rank_G_is_6": g.rank() == 6,
        "rank_K_is_6": k.rank() == 6,
    }


def closing_hand_map() -> dict[str, bool]:
    """``s = sigma Q(sigma d) y_h + s0`` for the closing hand ``h`` (sigma = +1
    for R, -1 for L), verified symbolically against the general ``y(w, s)``."""
    dsym = sp.Matrix(sp.symbols("d_x d_y d_z", real=True))
    s = sp.Matrix(sp.symbols("s1:7", real=True))
    f = sp.Matrix(sp.symbols("f_x f_y f_z", real=True))
    m = sp.Matrix(sp.symbols("m_x m_y m_z", real=True))
    w = sp.Matrix.vstack(f, m)
    y = equal_split_inverse() * w + split_basis(dsym) * s
    i3, z3 = sp.eye(3), sp.zeros(3)

    def q_of(dv: sp.Matrix) -> sp.Matrix:
        return sp.Matrix.vstack(
            sp.Matrix.hstack(i3, z3), sp.Matrix.hstack(skew(dv), i3)
        )

    y_r = y[6:12, :]
    y_l = y[0:6, :]
    s0_r = sp.Matrix.vstack(-f / 2, -(m + dsym.cross(f)) / 2)
    s0_l = sp.Matrix.vstack(f / 2, (m - dsym.cross(f)) / 2)
    ok_r = (s - (q_of(dsym) * y_r + s0_r)).applyfunc(sp.expand) == sp.zeros(6, 1)
    ok_l = (s - (-q_of(-dsym) * y_l + s0_l)).applyfunc(sp.expand) == sp.zeros(6, 1)
    inv_ok = (q_of(dsym) * q_of(-dsym)).applyfunc(sp.expand) == sp.eye(6)
    return {
        "closing_R_affine_map": ok_r,
        "closing_L_affine_map": ok_l,
        "Q_inverse_is_Q_of_minus_d": inv_ok,
    }


def named_columns(delta: sp.Rational) -> dict[str, Any]:
    """The 12x6 kernel for axial hands ``d = (delta, 0, 0)`` with physical names."""
    k = split_basis(sp.Matrix([delta, 0, 0]))
    names = [
        "s1_axial_pull_positive_squeeze_negative",
        "s2_transverse_shear_couple_v",
        "s3_transverse_shear_couple_w",
        "s4_twist_couple_about_shaft",
        "s5_bending_couple_about_v",
        "s6_bending_couple_about_w",
    ]
    return {
        "delta_m": str(delta),
        "row_order": [
            "f_Lx",
            "f_Ly",
            "f_Lz",
            "c_Lx",
            "c_Ly",
            "c_Lz",
            "f_Rx",
            "f_Ry",
            "f_Rz",
            "c_Rx",
            "c_Ry",
            "c_Rz",
        ],
        "column_names": names,
        "K": cm.exact_mat(k),
        "G": cm.exact_mat(grasp_map(sp.Matrix([delta, 0, 0]))),
    }


def _world_loads(rot: sp.Matrix, *loads: sp.Matrix) -> dict[str, list[str]]:
    """Hand loads (f_L, c_L, f_R, c_R) rotated from the grip frame to the world."""
    names = ("f_L", "c_L", "f_R", "c_R")
    return {n: cm.exact_mat((rot * x).T)[0] for n, x in zip(names, loads, strict=True)}


def worked_example() -> dict[str, Any]:
    u = sp.Matrix([R(3, 5), 0, R(4, 5)])
    n_face = sp.Matrix([1, 0, 0])
    v_raw = n_face - (n_face.dot(u)) * u
    v = v_raw / sp.sqrt(v_raw.dot(v_raw))
    v = v.applyfunc(sp.simplify)
    w = u.cross(v)
    rot = sp.Matrix.hstack(u, v, w)  # world <- frame (columns are the frame axes)
    assert rot.T * rot == sp.eye(3) and rot.det() == 1
    delta = R(3, 50)
    r_m = sp.Matrix([R(1, 10), R(1, 5), 1])
    r_l, r_r = r_m - delta * u, r_m + delta * u
    # bare-club rigid body, all quantities in the grip frame
    m_c = R(313, 1000)
    com_f = sp.Matrix([R(3, 4), 0, 0])  # COM position from o, frame components
    inertia = sp.diag(R(1, 10000), R(13, 500), R(13, 500))
    omega_f = sp.Matrix([5, 12, -8])
    alpha_f = sp.Matrix([30, -40, 50])
    a_world = sp.Matrix([30, -10, 20])
    g_world = sp.Matrix([0, 0, R(-981, 100)])
    a_f = rot.T * a_world
    g_f = rot.T * g_world
    force_f = m_c * (a_f - g_f)
    moment_f = (
        inertia * alpha_f + omega_f.cross(inertia * omega_f) + com_f.cross(force_f)
    )
    w_net = sp.Matrix.vstack(force_f, moment_f)
    s_true = sp.Matrix([40, -15, 25, R(4, 5), R(-1, 2), R(3, 10)])
    d_f = sp.Matrix([delta, 0, 0])
    y = equal_split_inverse() * w_net + split_basis(d_f) * s_true
    g = grasp_map(d_f)
    phi = differential_map()
    assert (g * y - w_net).applyfunc(sp.simplify) == sp.zeros(6, 1)
    assert (phi * y - s_true).applyfunc(sp.simplify) == sp.zeros(6, 1)
    f_l, c_l, f_r, c_r = y[0:3, :], y[3:6, :], y[6:9, :], y[9:12, :]
    # closing hand R: lambda = -(f_R, c_R) at its own grip point
    lam = -sp.Matrix.vstack(f_r, c_r)
    q_r = sp.Matrix.vstack(
        sp.Matrix.hstack(sp.eye(3), sp.zeros(3)), sp.Matrix.hstack(skew(d_f), sp.eye(3))
    )
    s0_r = sp.Matrix.vstack(-force_f / 2, -(moment_f + d_f.cross(force_f)) / 2)
    s_from_lambda = (q_r * (-lam) + s0_r).applyfunc(sp.simplify)
    assert s_from_lambda == s_true
    s_scaled = sp.diag(*(([R(1, 100)] * 3) + ([R(1, 5)] * 3))) * s_true
    world_forces = _world_loads(rot, f_l, c_l, f_r, c_r)
    return {
        "frame": {
            "shaft_axis_world": cm.exact_mat(u.T)[0],
            "face_normal_world": cm.exact_mat(n_face.T)[0],
            "v_world": cm.exact_mat(v.T)[0],
            "w_world": cm.exact_mat(w.T)[0],
            "R_world_from_frame": cm.exact_mat(rot),
            "origin_world": cm.exact_mat(r_m.T)[0],
            "r_L_world": cm.exact_mat(r_l.T)[0],
            "r_R_world": cm.exact_mat(r_r.T)[0],
            "half_separation_delta_m": str(delta),
        },
        "bare_club": {
            "mass_kg": str(m_c),
            "com_from_origin_frame_m": cm.exact_mat(com_f.T)[0],
            "inertia_about_com_frame_kg_m2": cm.exact_mat(inertia),
            "omega_frame_rad_s": cm.exact_mat(omega_f.T)[0],
            "alpha_frame_rad_s2": cm.exact_mat(alpha_f.T)[0],
            "a_com_world_m_s2": cm.exact_mat(a_world.T)[0],
            "gravity_world_m_s2": cm.exact_mat(g_world.T)[0],
        },
        "net_wrench_frame": {
            "force_N": cm.exact_mat(force_f.T)[0],
            "moment_about_origin_Nm": cm.exact_mat(moment_f.T)[0],
            "force_N_float": cm.vec(force_f),
            "moment_Nm_float": cm.vec(moment_f),
        },
        "split_coordinates": {
            "s": cm.exact_mat(s_true.T)[0],
            "units": ["N", "N", "N", "Nm", "Nm", "Nm"],
            "characteristic_scales": {"force_N": "100", "moment_Nm": "5"},
            "s_scaled": cm.exact_mat(s_scaled.T)[0],
        },
        "hand_loads_frame": {
            "f_L": cm.exact_mat(f_l.T)[0],
            "c_L": cm.exact_mat(c_l.T)[0],
            "f_R": cm.exact_mat(f_r.T)[0],
            "c_R": cm.exact_mat(c_r.T)[0],
        },
        "hand_loads_world": world_forces,
        "closing_hand": {
            "hand": "R",
            "lambda_frame": cm.exact_mat(lam.T)[0],
            "s_from_lambda_equals_s": True,
            "Q": cm.exact_mat(q_r),
            "Q_inverse": cm.exact_mat(q_r.inv()),
            "s0": cm.exact_mat(s0_r.T)[0],
        },
        "round_trip_s_recovered": cm.exact_mat((phi * y).T)[0],
        "net_wrench_recovered": cm.exact_mat((g * y).T)[0],
    }


def generate() -> dict[str, Any]:
    ids = symbolic_identities()
    clo = closing_hand_map()
    assert all(ids.values()), ids
    assert all(clo.values()), clo
    return cm.dump(
        "grasp_map_worked_example",
        {
            "symbolic_identities_verified": ids,
            "closing_hand_identities_verified": clo,
            "axial_matrices": named_columns(R(3, 50)),
            "worked_example": worked_example(),
        },
    )
