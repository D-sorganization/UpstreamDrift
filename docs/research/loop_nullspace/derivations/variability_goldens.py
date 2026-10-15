"""Variability-sensitivity goldens (NSX-80, #12709): exact sympy derivations.

Reference section: ``loop_nullspace.tex``, "Variability Sensitivity".

Two models are used, both small enough for sympy to finish in seconds.

1. **Redundantly actuated planar pendulum** (exact rational data).  A point mass
   ``m`` at ``q = L (sin th, -cos th)`` on the circle ``Phi(q) = (|q|^2 - L^2)/2 = 0``
   with two world-frame actuator forces ``u in R^2`` and one reaction
   ``lambda``::

       m a + h(q) = u + J(q)^T lambda,   h = (0, m g),   J = q^T,   H = grad^2 Phi = I.

   ``x = (u, lambda)`` has 3 unknowns for 2 equations, so the allocation family
   ``x = x_p + N z`` has one parameter: ``N = (-q, 1)`` (``z`` shifts the
   reaction by ``z`` and the radial actuator force by ``-z q``), the nominal
   motion is unchanged, and the *feed-forward* ``u(z)`` is what a perturbed
   swing replays.  The tangent-bundle linearization is derived from the
   differentiated KKT system (reactions re-solved, ``delta lambda`` eliminated)
   and compared with an independent reduced-coordinate derivation.
2. **Planar two-arm oracle**: the geometric preload block
   ``P^T (sum_r lambda_r grad^2 Phi_r) P`` of the NSX-00 planar golfer at its
   generic configuration, for two reaction allocations.

Nothing here imports repository production code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as cm  # noqa: E402

R = sp.Rational
NAME = "variability_sensitivity"

# Nominal data (exact).  sin th = 3/5, cos th = 4/5.
PEND = {
    "m": R(2),
    "L": R(1),
    "g": R(10),
    "sin_theta": R(3, 5),
    "cos_theta": R(4, 5),
    "omega": R(1),
    "u_particular": (R(7), R(5)),
}
# Frozen sensitivities and declared noise (hypotheses, not measurements).
NOISE = {
    "dt": R(1, 10),
    "c_outcome": (R(2, 5), R(1)),  # y = c_theta * dtheta + c_omega * domega
    "sigma0": (R(1, 5), R(3, 10)),
    "k": (R(1, 10), R(1, 4)),
    "P0_diag": (R(1, 100), R(1, 25)),
}
ZS = (R(0), R(1, 2), R(-3, 2))  # allocation samples for the golden table

TH, OM, ZZ, KAP = sp.symbols("theta omega z kappa", real=True)
M_, L_, G_ = sp.symbols("m L g", positive=True)
U1, U2 = sp.symbols("u1 u2", real=True)


def _vec(*a: Any) -> sp.Matrix:
    return sp.Matrix(list(a))


# --------------------------------------------------------------------------- #
# Tangent-bundle linearization of the pendulum                                 #
# --------------------------------------------------------------------------- #


def linearization(include_preload: bool = True) -> dict[str, Any]:
    """Symbolic ``(F_t, G_t)`` in tangent coordinates ``(delta th, delta om)``.

    The differentiated KKT system is solved for ``(delta a, delta lambda)``; the
    tangent acceleration ``alpha`` (``a = L alpha t - L om^2 e_r``) is read off.
    ``include_preload=False`` omits ``sum_r lambda_r H_r delta q`` (the error the
    specification forbids) and exists so tests can show that the golden would
    change.  Returns symbolic ``F`` (2x2), ``G`` (1x2 for the ``om`` row),
    the allocation map ``u(z)``, ``lambda(z)``.
    """
    s, c = sp.sin(TH), sp.cos(TH)
    q = L_ * _vec(s, -c)
    t_hat = _vec(c, s)
    e_r = _vec(s, -c)
    grav = _vec(0, M_ * G_)
    u_p = _vec(U1, U2)
    # Nominal reaction fixed by the radial balance (a = L alpha t - L om^2 e_r).
    lam_p = (-M_ * L_ * OM**2 - M_ * G_ * c - (e_r.T * u_p)[0, 0]) / L_
    lam = lam_p + ZZ
    u = u_p - q * ZZ
    # Perturbations on the tangent bundle.
    d_th, d_om = sp.symbols("d_theta d_omega")
    d_u = sp.Matrix(sp.symbols("du1 du2"))
    d_q = L_ * t_hat * d_th
    d_v = L_ * t_hat * d_om - L_ * OM * e_r * d_th
    jmat = q.T  # J = q^T (1x2); H = I
    alpha = sp.symbols("alpha")
    a_nom = L_ * alpha * t_hat - L_ * OM**2 * e_r
    # Unknowns: delta a (2), delta lambda (1).
    da = sp.Matrix(sp.symbols("da1 da2"))
    dlam = sp.Symbol("dlam")
    preload = lam * d_q if include_preload else sp.zeros(2, 1)
    # R1: M da + dh/dq dq - du - J^T dlam - (lam H) dq = 0 (M, h_q, H constant).
    r1 = M_ * da - d_u - jmat.T * dlam - preload
    # R2: J da + (H a).dq + 2 v.H.dv = 0.
    v_nom = L_ * OM * t_hat
    r2 = (jmat * da)[0, 0] + (a_nom.T * d_q)[0, 0] + 2 * (v_nom.T * d_v)[0, 0]
    sol = sp.solve([*list(r1), r2], [*list(da), dlam], dict=True)[0]
    da_s = _vec(sol[da[0]], sol[da[1]])
    # delta(t . a) = t.da - (e_r . a) d_th ;  L d_alpha = t.da + L om^2 d_th
    d_alpha = (t_hat.T * da_s)[0, 0] / L_ + OM**2 * d_th
    d_alpha = sp.expand(sp.simplify(d_alpha.subs(alpha, 0)))
    f_row = [sp.simplify(d_alpha.coeff(d_th)), sp.simplify(d_alpha.coeff(d_om))]
    g_row = [sp.simplify(d_alpha.coeff(d_u[i])) for i in range(2)]
    f_mat = sp.Matrix([[0, 1], f_row])
    g_mat = sp.Matrix([[0, 0], g_row])
    return {
        "F": f_mat,
        "G": g_mat,
        "u_of_z": u,
        "lambda_of_z": lam,
        "lambda_particular": lam_p,
        "d_alpha": d_alpha,
    }


def reduced_coordinate_reference() -> sp.Matrix:
    """Independent derivation: ``alpha(th, om; u) = (t.u - m g sin th)/(m L)``.

    ``u`` is held fixed (feed-forward), the reaction does not appear; the
    allocation enters only through the replayed ``u(z)``.
    """
    s, c = sp.sin(TH), sp.cos(TH)
    t_hat = _vec(c, s)
    q = L_ * _vec(s, -c)
    u = _vec(U1, U2) - q * ZZ
    alpha = ((t_hat.T * u)[0, 0] - M_ * G_ * s) / (M_ * L_)
    # u holds the *nominal-point* value: differentiate t(th) only (u fixed).
    u_fixed = u.subs(TH, sp.Symbol("th0"))
    alpha_fixed = ((t_hat.T * u_fixed)[0, 0] - M_ * G_ * s) / (M_ * L_)
    return sp.Matrix([[0, 1], [sp.diff(alpha_fixed, TH), 0]])


def nominal_subs() -> dict[sp.Symbol, sp.Expr]:
    p = PEND
    th0 = sp.asin(p["sin_theta"])
    return {
        M_: p["m"],
        L_: p["L"],
        G_: p["g"],
        OM: p["omega"],
        U1: p["u_particular"][0],
        U2: p["u_particular"][1],
        TH: th0,
    }


def _rat(expr: sp.Expr) -> sp.Expr:
    return sp.nsimplify(sp.simplify(expr), rational=True)


def _eval(mat: sp.Matrix, **extra: sp.Expr) -> sp.Matrix:
    sub = nominal_subs()
    sub.update({sp.Symbol(k): v for k, v in extra.items()})
    return mat.subs(sub).applyfunc(lambda e: _rat(sp.simplify(e)))


def _trig_exact(expr: sp.Expr) -> sp.Expr:
    """Replace sin/cos of the nominal angle by their exact rationals."""
    p = PEND
    sub = {
        sp.sin(TH): p["sin_theta"],
        sp.cos(TH): p["cos_theta"],
    }
    return sp.simplify(expr.subs(sub))


# --------------------------------------------------------------------------- #
# Saltation matrix                                                             #
# --------------------------------------------------------------------------- #


def saltation_matrix(
    f_minus: sp.Matrix, f_plus: sp.Matrix, grad_g: sp.Matrix, reset_jac: sp.Matrix
) -> sp.Matrix:
    """``S = R_x + (f+ - R_x f-) grad_g^T / (grad_g^T f-)``."""
    denom = (grad_g.T * f_minus)[0, 0]
    return reset_jac + (f_plus - reset_jac * f_minus) * grad_g.T / denom


def event_projection(f_minus: sp.Matrix, grad_g: sp.Matrix) -> sp.Matrix:
    """``Pi = I - f- grad_g^T / (grad_g^T f-)``: event-state timing correction.

    Maps a perturbation of the state at a *fixed time* to the perturbation of
    the state *at the event* (``grad_g^T Pi = 0``, ``Pi f- = 0``).
    """
    denom = (grad_g.T * f_minus)[0, 0]
    return sp.eye(f_minus.rows) - f_minus * grad_g.T / denom


def saltation_closed_form_check() -> dict[str, Any]:
    """Independent check of ``S`` on a solvable flow.

    State ``(s, w)``; pre-event flow ``s' = w, w' = a_m`` (constant), event
    ``g = s - s_e``, reset ``(s, w) -> (s, kappa w)``, post-event flow with
    constant acceleration ``a_p``.  ``D = d x(T)/d x0`` at a fixed later time
    ``T`` must equal ``Phi_plus(T - t*) S Phi_minus(t*)``.
    """
    s0, w0, s_e, t_big = sp.symbols("s0 w0 s_e T", real=True)
    a_m, a_p, kap = sp.symbols("a_m a_p kappa", real=True)
    tau = sp.Symbol("tau", positive=True)
    # event time: s0 + w0 t + a_m t^2/2 = s_e, positive root.
    disc = sp.sqrt(w0**2 + 2 * a_m * (s_e - s0))
    t_star = (-w0 + disc) / a_m
    w_ev = w0 + a_m * t_star
    s_post = s_e + kap * w_ev * (t_big - t_star) + a_p * (t_big - t_star) ** 2 / 2
    w_post = kap * w_ev + a_p * (t_big - t_star)
    x_t = sp.Matrix([s_post, w_post])
    d_mat = x_t.jacobian(sp.Matrix([s0, w0]))
    # Saltation formula at the event.
    s_ev = s_e
    f_m = sp.Matrix([w_ev, a_m])
    f_p = sp.Matrix([kap * w_ev, a_p])
    r_x = sp.Matrix([[1, 0], [0, kap]])
    s_mat = saltation_matrix(f_m, f_p, sp.Matrix([1, 0]), r_x)
    phi_m = sp.Matrix([[1, t_star], [0, 1]]) + sp.Matrix(
        [[a_m * t_star**2 / 2 * 0, 0], [0, 0]]
    )  # constant acceleration: d(s,w)/d(s0,w0) = [[1, t],[0, 1]]
    phi_p = sp.Matrix([[1, t_big - t_star], [0, 1]])
    pred = phi_p * s_mat * phi_m
    diff = (d_mat - pred).applyfunc(sp.simplify)
    # Without the timing term the identity must fail.
    pred_no_timing = phi_p * r_x * phi_m
    diff_no_timing = (d_mat - pred_no_timing).applyfunc(sp.simplify)
    del s_ev, tau
    # State at the event itself: d x(t*)/d x0 = Pi Phi_minus(t*).
    x_ev = sp.Matrix([s_e, w_ev])
    d_ev = x_ev.jacobian(sp.Matrix([s0, w0]))
    pi_mat = event_projection(f_m, sp.Matrix([1, 0]))
    diff_ev = (d_ev - pi_mat * phi_m).applyfunc(sp.simplify)
    diff_ev_fixed_time = (d_ev - phi_m).applyfunc(sp.simplify)
    return {
        "identity_holds": bool(diff == sp.zeros(2, 2)),
        "identity_holds_without_timing_term": bool(diff_no_timing == sp.zeros(2, 2)),
        "event_state_identity_holds": bool(diff_ev == sp.zeros(2, 2)),
        "event_state_identity_holds_fixed_time": bool(
            diff_ev_fixed_time == sp.zeros(2, 2)
        ),
    }


# --------------------------------------------------------------------------- #
# Variance model                                                               #
# --------------------------------------------------------------------------- #


def variance_model(include_preload: bool) -> dict[str, Any]:
    """One discrete step ``dx1 = A dx0 + B w`` and the outcome variance in ``z``.

    ``Var(z) = c^T A P0 A^T c + sum_i b_i^2 (sigma0_i^2 + k_i^2 u_i(z)^2)`` with
    ``b = B^T c``.  ``A = I + dt F`` (explicit Euler, declared), ``B = dt G``.
    """
    lin = linearization(include_preload=include_preload)
    sub = nominal_subs()
    f_n = lin["F"].subs(sub)
    g_n = lin["G"].subs(sub)
    f_n = f_n.applyfunc(lambda e: _trig_exact(sp.simplify(e)))
    g_n = g_n.applyfunc(lambda e: _trig_exact(sp.simplify(e)))
    f_n = f_n
    dt = NOISE["dt"]
    a_mat = sp.eye(2) + dt * f_n
    b_mat = dt * g_n
    c_vec = _vec(*NOISE["c_outcome"])
    p0 = sp.diag(*NOISE["P0_diag"])
    bvec = (b_mat.T * c_vec)[:2, :]
    u_z = lin["u_of_z"].subs(sub).applyfunc(_trig_exact)
    sig0 = NOISE["sigma0"]
    kk = NOISE["k"]
    var_init = (c_vec.T * a_mat * p0 * a_mat.T * c_vec)[0, 0]
    var_input = sum(
        (bvec[i] ** 2) * (sig0[i] ** 2 + kk[i] ** 2 * u_z[i] ** 2) for i in range(2)
    )
    var = sp.expand(var_init + var_input)
    return {"F": f_n, "G": g_n, "u": u_z, "b": bvec, "var": var}


def var_with_k(include_preload: bool, k: tuple[sp.Expr, sp.Expr]) -> sp.Expr:
    """Variance polynomial in ``z`` for given signal-dependence ``k``."""
    saved = dict(NOISE)
    try:
        NOISE["k"] = k
        return variance_model(include_preload)["var"]
    finally:
        NOISE.update(saved)


def closed_form_zstar() -> dict[str, Any]:
    """Minimum-variance ``z*`` (M1 only: frozen sensitivities, no preload)."""
    mdl = variance_model(include_preload=False)
    var = mdl["var"]
    d1 = sp.diff(var, ZZ)
    z_star = sp.solve(d1, ZZ)[0]
    # Formula from the specification: z* = sum b_i^2 k_i^2 u_p,i q_i / sum b_i^2 k_i^2 q_i^2
    q_n = _vec(PEND["sin_theta"], -PEND["cos_theta"]) * PEND["L"]
    up = _vec(*PEND["u_particular"])
    kk = NOISE["k"]
    b = mdl["b"]
    num = sum(b[i] ** 2 * kk[i] ** 2 * up[i] * q_n[i] for i in range(2))
    den = sum(b[i] ** 2 * kk[i] ** 2 * q_n[i] ** 2 for i in range(2))
    return {
        "var_poly": var,
        "z_star": z_star,
        "z_star_formula": sp.nsimplify(num / den),
        "second_derivative": sp.diff(var, ZZ, 2),
        "var_at_zstar": var.subs(ZZ, z_star),
    }


def additive_flat_case() -> dict[str, Any]:
    """``k = 0`` and no preload: ``d Var / d z`` is identically zero."""
    var_flat = var_with_k(False, (R(0), R(0)))
    var_preload_additive = var_with_k(True, (R(0), R(0)))
    return {
        "gradient_no_preload": sp.diff(var_flat, ZZ),
        "gradient_with_preload": sp.diff(var_preload_additive, ZZ),
        "var_with_preload_additive": var_preload_additive,
    }


# --------------------------------------------------------------------------- #
# Planar oracle: geometric preload block                                       #
# --------------------------------------------------------------------------- #


def golfer_preload() -> dict[str, Any]:
    """``P^T (sum_r lambda_r H_r) P`` for two allocations of the planar golfer."""
    import planar_golfer as pg

    mdl = pg.model()
    cfg = pg.generic_configuration()
    q, v, a = cfg["q"], cfg["v"], cfg["a"]
    ev = pg.evaluate(q, v)
    qv = list(pg.Q)
    hess = [sp.hessian(mdl["phi"][r], qv) for r in range(2)]
    f_h = [sp.lambdify(qv, h_r, "numpy", cse=True) for h_r in hess]
    h_num = [np.array(f(*map(float, q)), dtype=float) for f in f_h]
    jc = ev["Jc"]
    b = ev["M"] @ a + ev["h"]  # all coordinates actuated: tau = b - Jc^T lam
    # tangent basis P of ker J_c (orthonormal, SVD)
    _, s, vt = np.linalg.svd(jc)
    p_mat = vt[int((s > cm.RANK_RTOL).sum()) :].T
    out: dict[str, Any] = {
        "closure_rows": 2,
        "tangent_dimension": int(p_mat.shape[1]),
        "hessian_symmetry_error": max(float(np.abs(h - h.T).max()) for h in h_num),
        "allocations": [],
    }
    lam0 = np.array([4.0, -3.0])
    for label, dz in (("z0", np.zeros(2)), ("z1", np.array([2.5, 1.5]))):
        lam = lam0 + dz
        k_geom = sum(lam[r] * h_num[r] for r in range(2))
        proj = p_mat @ p_mat.T  # basis-independent projector onto ker J_c
        kt = proj @ k_geom @ proj
        eig = np.sort(np.linalg.eigvalsh(p_mat.T @ k_geom @ p_mat))
        tau = b - jc.T @ lam
        out["allocations"].append(
            {
                "label": label,
                "lambda": [float(x) for x in lam],
                "tau": [float(x) for x in tau],
                "K_geom_projected": cm.mat(sp.Matrix(kt)),
                "K_geom_tangent_eigenvalues": [float(x) for x in eig],
                "K_geom_projected_frobenius": float(np.linalg.norm(kt)),
            }
        )
    k0 = np.array(out["allocations"][0]["K_geom_projected"])
    k1 = np.array(out["allocations"][1]["K_geom_projected"])
    out["preload_changes_with_allocation"] = bool(np.linalg.norm(k1 - k0) > 1e-3)
    return out


# --------------------------------------------------------------------------- #
# Golden                                                                       #
# --------------------------------------------------------------------------- #


def _poly_coeffs(expr: sp.Expr) -> list[str]:
    poly = sp.Poly(sp.expand(expr), ZZ)
    return [str(c) for c in poly.all_coeffs()]


def _model_block() -> dict[str, Any]:
    return {
        "equation": "m a + h(q) = u + J(q)^T lambda; Phi = (|q|^2 - L^2)/2",
        "allocation": "x = (u, lambda) = x_p + N z, N = (-q, 1)",
        "parameters": {k: str(v) for k, v in PEND.items() if k != "u_particular"},
        "u_particular": [str(x) for x in PEND["u_particular"]],
        "noise_declared": {
            "dt": str(NOISE["dt"]),
            "c_outcome": [str(x) for x in NOISE["c_outcome"]],
            "sigma0": [str(x) for x in NOISE["sigma0"]],
            "k": [str(x) for x in NOISE["k"]],
            "P0_diag": [str(x) for x in NOISE["P0_diag"]],
        },
    }


def _linearization_block() -> dict[str, Any]:
    sub = nominal_subs()
    with_m2 = linearization(include_preload=True)
    without_m2 = linearization(include_preload=False)
    ref = reduced_coordinate_reference()
    # Symbolic cross-check: KKT with preload equals reduced coordinates.
    kkt_vs_reduced = sp.simplify(
        with_m2["F"][1, 0] - sp.simplify(ref[1, 0].subs(sp.Symbol("th0"), TH))
    )
    q_sym = L_ * _vec(sp.sin(TH), -sp.cos(TH))
    g_times_n_zero = bool(sp.simplify((with_m2["G"][1, :] * (-q_sym))[0, 0]) == 0)
    omega, m = PEND["omega"], PEND["m"]
    lam_p_n = _trig_exact(with_m2["lambda_particular"].subs(sub))
    lam_p_n = sp.nsimplify(lam_p_n)
    table = []
    for z in ZS:
        lam_z = lam_p_n + z
        f10_with = sp.nsimplify(_trig_exact(with_m2["F"][1, 0].subs(sub)).subs(ZZ, z))
        f10_without = sp.nsimplify(
            _trig_exact(without_m2["F"][1, 0].subs(sub)).subs(ZZ, z)
        )
        u_z = with_m2["u_of_z"].subs(sub).applyfunc(_trig_exact).subs(ZZ, z)
        table.append(
            {
                "z": str(z),
                "lambda": str(lam_z),
                "u": [str(x) for x in u_z],
                "F10_with_preload": str(f10_with),
                "F10_without_preload": str(f10_without),
                "F10_expected_omega2_plus_lambda_over_m": str(omega**2 + lam_z / m),
            }
        )
    g_nom = with_m2["G"].subs(sub).applyfunc(_trig_exact)
    g_nom = g_nom.applyfunc(sp.nsimplify)

    return {
        "symbolic_F10_with_preload": str(sp.simplify(with_m2["F"][1, 0])),
        "kkt_equals_reduced_coordinate_symbolic": bool(kkt_vs_reduced == 0),
        "F01": "1",
        "F11": str(with_m2["F"][1, 1]),
        "G_nominal_om_row": cm.exact_mat(g_nom[1, :]),
        "G_times_allocation_direction_is_zero": g_times_n_zero,
        "lambda_particular": str(lam_p_n),
        "allocation_table": table,
    }


def _saltation_block() -> dict[str, Any]:
    # Saltation on the pendulum nominal point (event: x-coordinate hits d).
    t_n = _vec(PEND["cos_theta"], PEND["sin_theta"])
    q_n = PEND["L"] * _vec(PEND["sin_theta"], -PEND["cos_theta"])
    u_p = _vec(*PEND["u_particular"])
    alpha_n = ((t_n.T * u_p)[0, 0] - PEND["m"] * PEND["g"] * PEND["sin_theta"]) / (
        PEND["m"] * PEND["L"]
    )
    del q_n
    f_minus = _vec(PEND["omega"], alpha_n)
    grad_g = _vec(PEND["L"] * PEND["cos_theta"], 0)
    salt: dict[str, Any] = {
        "event": "g = L sin(theta) - d (ball-plane crossing at x = d)",
        "f_minus": cm.exact_mat(f_minus),
        "grad_g": cm.exact_mat(grad_g),
        "cases": [],
    }
    for kap_v in (R(1), R(7, 10)):
        r_x = sp.diag(1, kap_v)
        f_plus = _vec(kap_v * PEND["omega"], alpha_n)
        s_mat = saltation_matrix(f_minus, f_plus, grad_g, r_x)
        salt["cases"].append(
            {
                "kappa": str(kap_v),
                "f_plus": cm.exact_mat(f_plus),
                "reset_jacobian": cm.exact_mat(r_x),
                "S": cm.exact_mat(s_mat),
                "S_f_minus_equals_f_plus": bool(
                    (s_mat * f_minus - f_plus).is_zero_matrix
                ),
                "S_differs_from_reset_jacobian": bool(s_mat != r_x),
            }
        )
    pi_n = event_projection(f_minus, grad_g)
    salt["event_projection"] = {
        "Pi": cm.exact_mat(pi_n),
        "grad_g_T_Pi_is_zero": bool((grad_g.T * pi_n).is_zero_matrix),
        "Pi_f_minus_is_zero": bool((pi_n * f_minus).is_zero_matrix),
        "Pi_differs_from_identity": bool(pi_n != sp.eye(2)),
    }
    salt["independent_closed_form_check"] = saltation_closed_form_check()

    return salt


def _min_variance_block() -> dict[str, Any]:
    zres = closed_form_zstar()
    var_m1 = zres["var_poly"]
    var_m2 = variance_model(include_preload=True)["var"]
    z_stars_m2 = sp.Poly(sp.diff(var_m2, ZZ), ZZ).all_coeffs()
    f_wo = variance_model(include_preload=False)
    f_wi = variance_model(include_preload=True)

    return {
        "model": (
            "dx1 = (I + dt F) dx0 + dt G w; y = c^T dx1; "
            "Var(z) = c^T A P0 A^T c + sum_i b_i^2 (sigma0_i^2 + k_i^2 u_i(z)^2)"
        ),
        "b": [str(x) for x in zres_b()],
        "var_poly_coeffs_descending_M1": _poly_coeffs(var_m1),
        "z_star": str(zres["z_star"]),
        "z_star_closed_form": str(zres["z_star_formula"]),
        "z_star_matches_closed_form": bool(
            sp.simplify(zres["z_star"] - zres["z_star_formula"]) == 0
        ),
        "second_derivative": str(zres["second_derivative"]),
        "var_at_zstar": str(zres["var_at_zstar"]),
        "dvar_dz_coeffs_with_preload_M1_M2": [str(c) for c in z_stars_m2],
        "F10_frozen_without_preload": str(f_wo["F"][1, 0]),
        "F10_with_preload_at_lambda_p": str(f_wi["F"][1, 0]),
    }


def _flat_block() -> dict[str, Any]:
    flat = additive_flat_case()
    return {
        "k": ["0", "0"],
        "gradient_no_preload": str(flat["gradient_no_preload"]),
        "gradient_no_preload_is_zero": bool(flat["gradient_no_preload"] == 0),
        "gradient_with_preload": str(flat["gradient_with_preload"]),
        "gradient_with_preload_is_zero": bool(flat["gradient_with_preload"] == 0),
    }


def generate() -> dict[str, Any]:
    """Assemble the golden payload (see the module docstring)."""
    return cm.dump(
        NAME,
        {
            "description": (
                "NSX-80 variability-sensitivity goldens: redundantly actuated "
                "planar pendulum (exact) and planar-golfer preload block."
            ),
            "model": _model_block(),
            "linearization": _linearization_block(),
            "saltation": _saltation_block(),
            "min_variance": _min_variance_block(),
            "additive_noise_flat": _flat_block(),
            "golfer_preload": golfer_preload(),
        },
    )


def zres_b() -> sp.Matrix:
    return variance_model(include_preload=False)["b"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if file differs")
    args = parser.parse_args()
    payload = generate()
    target = cm.GOLDENS_DIR / f"{NAME}.json"
    if args.check:
        stored = (
            json.loads(target.read_text(encoding="utf-8")) if target.exists() else None
        )
        fresh = json.loads(json.dumps(payload, allow_nan=False))
        if stored != fresh:
            sys.stdout.write(f"STALE {target}\n")
            return 1
        return 0
    sys.stdout.write(f"{cm.write_json(NAME, payload)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
