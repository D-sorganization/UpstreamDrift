"""Planar two-arm golfer: symbolic Lagrangian model and closed-loop goldens.

Model (NSX-00, section "Planar Two-Arm Fixture"; not a golfer model):

* Five generalized coordinates ``q = (q1, q2, q3, q4, q5)``.  ``q1`` is the lead
  shoulder, ``q2`` the lead elbow, ``q3`` the club angle relative to the lead
  forearm (the lead hand holds the club rigidly through a pin at the wrist, so
  ``q3`` is the "club angle"), ``q4`` the trail shoulder and ``q5`` the trail
  elbow.  All angles are measured from the downward vertical, counter-clockwise
  positive, so a segment of absolute angle ``a`` points along
  ``u(a) = (sin a, -cos a)``.
* The trail hand closes the loop onto a grip point on the club, ``g`` metres
  from the lead hand along the club axis.  ``Phi(q) = H_R - G = 0`` is a point
  (pin) closure of two rows: closure kind ``PLANAR_POINT2``.  The multiplier
  ``lambda`` is the planar force (N) the closure exerts on the trail hand, rows
  ``[fx, fy]`` in the world frame.
* The shoulders are fixed in the world; gravity acts along ``-y``.

Everything numeric below is derived from sympy expressions; nothing is read from
the repository's production code.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np
import sympy as sp

import _common as cm

R = sp.Rational
N_Q = 5
LEAD, TRAIL, CLUB = [0, 1, 2], [3, 4], [2]

PARAMS = {
    "shoulder_separation_m": R(1, 5),
    "upper_arm_length_m": R(3, 10),
    "forearm_length_m": R(7, 25),
    "upper_arm_mass_kg": R(2),
    "forearm_mass_kg": R(8, 5),
    "club_mass_kg": R(313, 1000),
    "club_length_m": R(1),
    "club_com_from_lead_hand_m": R(13, 20),
    "trail_grip_from_lead_hand_m": R(1, 5),
    "gravity_m_s2": R(981, 100),
}

Q = sp.symbols("q1:6", real=True)
W = sp.symbols("w1:6", real=True)


def _u(angle: sp.Expr) -> sp.Matrix:
    return sp.Matrix([sp.sin(angle), -sp.cos(angle)])


@functools.lru_cache(maxsize=1)
def model() -> dict[str, Any]:
    """Symbolic M(q), h(q, w), Phi(q), J_c(q), Jdot_v(q, w)."""
    p = PARAMS
    d, lu, lf = (
        p["shoulder_separation_m"],
        p["upper_arm_length_m"],
        p["forearm_length_m"],
    )
    mu, mf, mc = p["upper_arm_mass_kg"], p["forearm_mass_kg"], p["club_mass_kg"]
    lc, cc, g_off = (
        p["club_length_m"],
        p["club_com_from_lead_hand_m"],
        p["trail_grip_from_lead_hand_m"],
    )
    grav = p["gravity_m_s2"]
    q1, q2, q3, q4, q5 = Q
    a1, a2, a3 = q1, q1 + q2, q1 + q2 + q3
    b1, b2 = q4, q4 + q5
    s_l, s_r = sp.Matrix([-d / 2, 0]), sp.Matrix([d / 2, 0])
    e_l = s_l + lu * _u(a1)
    h_l = e_l + lf * _u(a2)
    e_r = s_r + lu * _u(b1)
    h_r = e_r + lf * _u(b2)
    grip = h_l + g_off * _u(a3)
    bodies = [
        (mu, mu * lu**2 / 12, s_l + (lu / 2) * _u(a1), a1),
        (mf, mf * lf**2 / 12, e_l + (lf / 2) * _u(a2), a2),
        (mc, mc * lc**2 / 12, h_l + cc * _u(a3), a3),
        (mu, mu * lu**2 / 12, s_r + (lu / 2) * _u(b1), b1),
        (mf, mf * lf**2 / 12, e_r + (lf / 2) * _u(b2), b2),
    ]
    qv = sp.Matrix(Q)
    mass = sp.zeros(N_Q, N_Q)
    potential = 0
    for m_i, i_i, com, ang in bodies:
        jv = com.jacobian(qv)
        jw = sp.Matrix([ang]).jacobian(qv)
        mass += m_i * jv.T * jv + i_i * jw.T * jw
        potential += m_i * grav * com[1]
    mass = mass.applyfunc(sp.expand)
    wv = sp.Matrix(W)
    d_mass = [mass.diff(q) for q in Q]
    mdot = sum((d_mass[k] * W[k] for k in range(N_Q)), sp.zeros(N_Q, N_Q))
    quad = sp.Matrix([(wv.T * d_mass[i] * wv)[0, 0] / 2 for i in range(N_Q)])
    h_vec = mdot * wv - quad + sp.Matrix([potential.diff(q) for q in Q])
    phi = h_r - grip
    jc = phi.jacobian(qv)
    jdot = sum((jc.diff(Q[k]) * W[k] for k in range(N_Q)), sp.zeros(2, N_Q))
    return {
        "M": mass,
        "h": h_vec,
        "phi": phi,
        "Jc": jc,
        "Jdot_v": jdot * wv,
        "positions": {"H_L": h_l, "H_R": h_r, "G": grip, "E_L": e_l, "E_R": e_r},
    }


@functools.lru_cache(maxsize=1)
def _numeric() -> dict[str, Callable[..., Any]]:
    mdl = model()
    args = (*Q, *W)
    return {
        "M": sp.lambdify(Q, mdl["M"], "numpy", cse=True),
        "h": sp.lambdify(args, mdl["h"], "numpy", cse=True),
        "phi": sp.lambdify(Q, mdl["phi"], "numpy", cse=True),
        "Jc": sp.lambdify(Q, mdl["Jc"], "numpy", cse=True),
        "Jdot_v": sp.lambdify(args, mdl["Jdot_v"], "numpy", cse=True),
    }


def evaluate(q: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    f = _numeric()
    qa, va = list(map(float, q)), list(map(float, v))
    return {
        "M": np.array(f["M"](*qa), dtype=float),
        "h": np.array(f["h"](*qa, *va), dtype=float).reshape(-1),
        "phi": np.array(f["phi"](*qa), dtype=float).reshape(-1),
        "Jc": np.array(f["Jc"](*qa), dtype=float),
        "Jdot_v": np.array(f["Jdot_v"](*qa, *va), dtype=float).reshape(-1),
    }


def _solve_closure(q_fixed: dict[int, float], guess: dict[int, float]) -> np.ndarray:
    """Solve Phi(q) = 0 for the unknown coordinates at 50-digit precision."""
    import mpmath as mp

    mp.mp.dps = cm.WORKING_DPS
    mdl = model()
    unknown = sorted(guess)
    subs_fixed = {Q[i]: sp.Float(str(q_fixed[i]), cm.WORKING_DPS) for i in q_fixed}
    eqs = [
        sp.lambdify([Q[i] for i in unknown], e.subs(subs_fixed), "mpmath")
        for e in mdl["phi"]
    ]
    sol = mp.findroot(lambda *x: [e(*x) for e in eqs], [guess[i] for i in unknown])
    out = np.zeros(N_Q)
    for i, val in q_fixed.items():
        out[i] = val
    for k, i in enumerate(unknown):
        out[i] = float(sol[k])
    return out


def generic_configuration() -> dict[str, np.ndarray]:
    """A closed, non-singular configuration with a velocity/acceleration pair
    that satisfies the position-, velocity- and acceleration-level closure."""
    q = _solve_closure({0: 0.10, 1: 1.40, 2: -0.30}, {3: 0.20, 4: 1.00})
    # lead joint rates chosen, trail rates solved from J_c v = 0
    v = _complete_rates(q, np.zeros(N_Q), {0: 2.0, 1: -1.5, 2: 3.0}, [3, 4])
    a = _complete_rates(q, v, {0: -4.0, 1: 6.0, 2: 10.0}, [3, 4], acceleration=True)
    return {"q": q, "v": v, "a": a}


def _complete_rates(
    q: np.ndarray,
    v: np.ndarray,
    chosen: dict[int, float],
    solved: list[int],
    *,
    acceleration: bool = False,
) -> np.ndarray:
    ev = evaluate(q, v)
    out = np.zeros(N_Q)
    for i, val in chosen.items():
        out[i] = val
    rhs = -ev["Jdot_v"] if acceleration else np.zeros(2)
    rhs = rhs - ev["Jc"][:, list(chosen)] @ np.array(list(chosen.values()))
    out[solved] = np.linalg.solve(ev["Jc"][:, solved], rhs)
    return out


# --------------------------------------------------------------------------- #
# Diagnostics shared by the fixtures                                           #
# --------------------------------------------------------------------------- #


def _rank(arr: np.ndarray) -> tuple[int, list[float]]:
    if arr.size == 0:
        return 0, []
    s = np.linalg.svd(arr, compute_uv=False)
    tol = cm.RANK_RTOL * max(float(s[0]), 1.0)
    return int((s > tol).sum()), [float(x) for x in s]


def zero_side(jc: np.ndarray, b: np.ndarray, idx: list[int]) -> dict[str, Any]:
    """Three-case classification of ``J_{c,side}^T lambda = b_side`` (R3)."""
    js_t = jc[:, idx].T  # n_side x m
    b_s = b[idx]
    rank, sv = _rank(js_t)
    pinv = np.linalg.pinv(js_t, rcond=cm.RANK_RTOL)
    lam = pinv @ b_s
    resid_vec = b_s - js_t @ lam
    resid = float(np.linalg.norm(resid_vec))
    consistent = resid <= 1e-9 * (1.0 + float(np.linalg.norm(b_s)))
    nullity = js_t.shape[1] - rank
    if not consistent:
        case = "INCONSISTENT"
    elif nullity == 0:
        case = "CONSISTENT_UNIQUE"
    else:
        case = "CONSISTENT_NONUNIQUE"
    return {
        "indices": idx,
        "n_equations": js_t.shape[0],
        "n_unknowns": js_t.shape[1],
        "rank": rank,
        "nullity": nullity,
        "singular_values": sv,
        "case": case,
        "residual": resid if not consistent else 0.0,
        "lambda_min_residual_min_norm": [float(x) for x in lam],
    }


def min_metric_torque(
    jc: np.ndarray, b: np.ndarray, actuated: list[int], w_a: np.ndarray
) -> dict[str, Any]:
    """Minimum ``tau_a^T W tau_a`` over the equality family (any rank)."""
    unact = [i for i in range(N_Q) if i not in actuated]
    jca, jcu = jc[:, actuated], jc[:, unact]
    m = jc.shape[0]
    if unact:
        lam_p = np.linalg.pinv(jcu.T, rcond=cm.RANK_RTOL) @ b[unact]
        _, s, vt = np.linalg.svd(jcu.T)
        rk = int((s > cm.RANK_RTOL * max(float(s[0]), 1.0)).sum())
        kmat = vt[rk:].T
    else:
        lam_p, kmat = np.zeros(m), np.eye(m)
    t0 = b[actuated] - jca.T @ lam_p
    bmat = jca.T @ kmat
    z = np.linalg.pinv(bmat.T @ w_a @ bmat, rcond=1e-12) @ (bmat.T @ w_a @ t0)
    lam = lam_p + kmat @ z
    tau = b[actuated] - jca.T @ lam
    # minimum-norm representative of lambda (removes multiplier gauge only)
    gauge_free = np.linalg.pinv(jc.T, rcond=cm.RANK_RTOL) @ (jc.T @ lam)
    return {
        "tau_actuated": [float(x) for x in tau],
        "lambda_min_norm": [float(x) for x in gauge_free],
        "objective": float(tau @ w_a @ tau),
    }


def kkt_acceleration(
    mass: np.ndarray,
    h: np.ndarray,
    s_rows: list[int],
    tau: np.ndarray,
    jc: np.ndarray,
    jdv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Constrained acceleration from the KKT system (min-norm lambda if singular)."""
    n, m = mass.shape[0], jc.shape[0]
    gen = np.zeros(n)
    gen[s_rows] = tau
    kkt = np.block([[mass, -jc.T], [jc, np.zeros((m, m))]])
    rhs = np.concatenate([gen - h, -jdv])
    sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    return sol[:n], sol[n:]


def _dimension_counts(jc: np.ndarray, actuated: list[int]) -> dict[str, Any]:
    n = jc.shape[1]
    m = jc.shape[0]
    unact = [i for i in range(n) if i not in actuated]
    n_a, n_u = len(actuated), len(unact)
    s_mat = np.zeros((n_a, n))
    for r, i in enumerate(actuated):
        s_mat[r, i] = 1.0
    a_mat = np.hstack([s_mat.T, jc.T])
    rank_a, sv_a = _rank(a_mat)
    rank_jc, sv_jc = _rank(jc)
    rank_jcu, _ = _rank(jc[:, unact].T) if unact else (0, [])
    nullity = n_a + m - rank_a
    gauge = m - rank_jc
    attainable = rank_jc - rank_jcu
    return {
        "n": n,
        "n_actuated": n_a,
        "n_unactuated": n_u,
        "m": m,
        "n_g": 0,
        "dim_x": n_a + m,
        "rank_J_c": rank_jc,
        "rank_J_cu_T": rank_jcu,
        "rank_A": rank_a,
        "equality_nullity": nullity,
        "multiplier_gauge": gauge,
        "attainable_torque_variation": attainable,
        "nullity_equals_gauge_plus_attainable": bool(nullity == gauge + attainable),
        "singular_values_A": sv_a,
        "singular_values_J_c": sv_jc,
        "rank_method": f"SVD, relative tolerance {cm.RANK_RTOL:g} of max(sigma_max, 1)",
    }


def _curvature(jc: np.ndarray, actuated: list[int], b: np.ndarray) -> dict[str, Any]:
    """Positive-curvature check of ``1/2 tau^T tau`` on the retained space.

    The torque variation per multiplier is ``-J_{c,a}^T`` restricted to the
    multiplier directions that keep the unactuated rows balanced.  Gauge
    directions (``J_c^T dlam = 0``) are quotiented out.
    """
    unact = [i for i in range(N_Q) if i not in actuated]
    jca, jcu = jc[:, actuated], jc[:, unact]
    m = jc.shape[0]
    if unact:
        _, s, vt = np.linalg.svd(jcu.T)
        rk = int((s > cm.RANK_RTOL * max(float(s[0]), 1.0)).sum())
        kmat = vt[rk:].T
    else:
        kmat = np.eye(m)
    n_tau = -(jca.T @ kmat)
    gram = n_tau.T @ n_tau
    eig = np.linalg.eigvalsh(gram) if gram.size else np.array([])
    tol = cm.RANK_RTOL * max(float(eig.max()) if eig.size else 1.0, 1.0)
    return {
        "family_dimension": int(kmat.shape[1]),
        "retained_dimension": int((eig > tol).sum()),
        "gram_eigenvalues_identity_metric": [float(x) for x in eig],
        "strictly_convex_on_family": bool(eig.size and (eig > tol).all()),
    }


def fixture(
    name: str,
    cfg: dict[str, np.ndarray],
    actuated: list[int],
    description: str,
    ev: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    q, v, a = cfg["q"], cfg["v"], cfg["a"]
    ev = ev if ev is not None else evaluate(q, v)
    mass, h, jc, jdv = ev["M"], ev["h"], ev["Jc"], ev["Jdot_v"]
    b = mass @ a + h
    s_mat = np.zeros((len(actuated), N_Q))
    for r, i in enumerate(actuated):
        s_mat[r, i] = 1.0
    minv = np.linalg.inv(mass)
    unact = [i for i in range(N_Q) if i not in actuated]
    out: dict[str, Any] = {
        "description": description,
        "model": {
            "closure_kind": "PLANAR_POINT2",
            "closure_rows": ["fx", "fy"],
            "coordinate_order": [
                "lead_shoulder",
                "lead_elbow",
                "club_angle",
                "trail_shoulder",
                "trail_elbow",
            ],
            "parameters": {k: str(v_) for k, v_ in PARAMS.items()},
            "actuated_indices": actuated,
            "unactuated_indices": unact,
            "sides": {"lead": LEAD, "trail": TRAIL, "club_joint": CLUB},
        },
        "state": {
            "q": cm.vec(sp.Matrix(q)),
            "v": cm.vec(sp.Matrix(v)),
            "a": cm.vec(sp.Matrix(a)),
            "phi": [float(x) for x in ev["phi"]],
            "J_c_v": [float(x) for x in jc @ v],
            "J_c_a_plus_Jdot_v": [float(x) for x in jc @ a + jdv],
        },
        "matrices": {
            "M": cm.mat(sp.Matrix(mass)),
            "h": cm.vec(sp.Matrix(h)),
            "S": cm.mat(sp.Matrix(s_mat)),
            "J_c": cm.mat(sp.Matrix(jc)),
            "Jdot_v": cm.vec(sp.Matrix(jdv)),
            "b": cm.vec(sp.Matrix(b)),
        },
        "dimensions": _dimension_counts(jc, actuated),
        "unactuated_residual_at_lambda_zero": float(np.linalg.norm(b[unact]))
        if unact
        else 0.0,
        "zero_side": {
            k: zero_side(jc, b, idx)
            for k, idx in (("lead", LEAD), ("trail", TRAIL), ("club_joint", CLUB))
        },
        "curvature": _curvature(jc, actuated, b),
        "min_metric_torque": {},
    }
    for label, w_full in (("minv", minv), ("identity", np.eye(N_Q))):
        w_a = w_full[np.ix_(actuated, actuated)]
        res = min_metric_torque(jc, b, actuated, w_a)
        tau = np.array(res["tau_actuated"])
        a_kkt, lam_kkt = kkt_acceleration(mass, h, actuated, tau, jc, jdv)
        res["metric"] = (
            "W = (M^-1) restricted to actuated rows and columns"
            if label == "minv"
            else "W = identity on actuated torques"
        )
        res["a_from_kkt"] = [float(x) for x in a_kkt]
        res["a_error_inf"] = float(np.max(np.abs(a_kkt - a)))
        res["lambda_from_kkt_min_norm"] = [float(x) for x in lam_kkt]
        out["min_metric_torque"][label] = res
    if not unact:
        lam_c = np.linalg.pinv(jc @ minv @ jc.T, rcond=cm.RANK_RTOL)
        closed = (np.eye(N_Q) - jc.T @ lam_c @ jc @ minv) @ b
        out["min_metric_torque"]["minv"]["closed_form_tau"] = [float(x) for x in closed]
        out["min_metric_torque"]["minv"]["closed_form_error_inf"] = float(
            np.max(
                np.abs(
                    closed - np.array(out["min_metric_torque"]["minv"]["tau_actuated"])
                )
            )
        )
        out["min_metric_torque"]["minv"]["lambda_closed_form"] = [
            float(x) for x in (lam_c @ jc @ minv @ b)
        ]
    return out


def generate_full() -> dict[str, Any]:
    cfg = generic_configuration()
    return cm.dump(
        "planar_golfer_full",
        fixture(
            "full",
            cfg,
            [0, 1, 2, 3, 4],
            "Planar two-arm golfer, every coordinate actuated (S = I).",
        ),
    )


def generate_club_unactuated() -> dict[str, Any]:
    cfg = generic_configuration()
    return cm.dump(
        "planar_golfer_club_unactuated",
        fixture(
            "club_unactuated",
            cfg,
            [0, 1, 3, 4],
            "Same state as planar_golfer_full; the club angle q3 carries no actuator.",
        ),
    )


def straight_configuration_exact() -> dict[str, Any]:
    """Fully collinear configuration with exact (rational) kinematics.

    Both shoulders, both arms and the club lie on the world x-axis:
    ``q = (pi/2, 0, 0, pi/2, 0)``.  Every column of ``J_c`` is then parallel to
    ``e_y`` and ``rank J_c = 1 < m = 2`` (a closed-loop singularity).  The state
    is at rest (``v = 0``) so that ``Jdot_v = 0`` and the acceleration-level
    closure is the homogeneous ``J_c a = 0``.
    """
    mdl = model()
    q_exact = {Q[0]: sp.pi / 2, Q[1]: 0, Q[2]: 0, Q[3]: sp.pi / 2, Q[4]: 0}
    w_zero = dict.fromkeys(W, 0)
    exact = {
        "M": mdl["M"].subs(q_exact).applyfunc(sp.simplify),
        "h": mdl["h"].subs({**q_exact, **w_zero}).applyfunc(sp.simplify),
        "phi": mdl["phi"].subs(q_exact).applyfunc(sp.simplify),
        "Jc": mdl["Jc"].subs(q_exact).applyfunc(sp.simplify),
        "Jdot_v": mdl["Jdot_v"].subs({**q_exact, **w_zero}).applyfunc(sp.simplify),
    }
    a_free = [R(1), R(-1), R(2), R(-2)]
    jc = exact["Jc"]
    a5 = -(jc[1, :4] * sp.Matrix(a_free))[0, 0] / jc[1, 4]
    a_exact = sp.Matrix([*a_free, sp.simplify(a5)])
    return {"q": q_exact, "exact": exact, "a": a_exact}


def generate_straight_arm() -> dict[str, Any]:
    st = straight_configuration_exact()
    ex = st["exact"]
    q = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, 0.0])
    cfg = {"q": q, "v": np.zeros(N_Q), "a": np.array(cm.vec(st["a"]))}
    ev = {
        k: np.array(cm.mat(v) if v.cols > 1 else cm.vec(v), dtype=float)
        for k, v in ex.items()
    }
    ev["h"], ev["phi"], ev["Jdot_v"] = (
        np.asarray(ev[k]).reshape(-1) for k in ("h", "phi", "Jdot_v")
    )
    out = {}
    for label, act in (("full", [0, 1, 2, 3, 4]), ("club_unactuated", [0, 1, 3, 4])):
        fx = fixture(
            f"straight_{label}",
            cfg,
            act,
            f"Collinear straight-arm configuration, {label}.",
            ev=ev,
        )
        s_exact = sp.zeros(len(act), N_Q)
        for r, i in enumerate(act):
            s_exact[r, i] = 1
        a_exact = sp.Matrix(st["a"])
        a_mat = s_exact.T.row_join(ex["Jc"].T)
        unact = [i for i in range(N_Q) if i not in act]
        jcu_t = ex["Jc"][:, unact].T if unact else sp.zeros(0, 2)
        b_exact = ex["M"] * a_exact + ex["h"]
        fx["exact"] = {
            "rank_J_c": ex["Jc"].rank(),
            "rank_J_cu_T": jcu_t.rank() if unact else 0,
            "rank_A": a_mat.rank(),
            "J_c": cm.exact_mat(ex["Jc"]),
            "M": cm.exact_mat(ex["M"]),
            "h": cm.exact_mat(ex["h"]),
            "a": cm.exact_mat(a_exact),
            "b": cm.exact_mat(b_exact),
        }
        dm = fx["dimensions"]
        assert (dm["rank_J_c"], dm["rank_J_cu_T"], dm["rank_A"]) == (
            fx["exact"]["rank_J_c"],
            fx["exact"]["rank_J_cu_T"],
            fx["exact"]["rank_A"],
        ), "numeric rank disagrees with exact rank"
        out[label] = fx
    return cm.dump("planar_golfer_straight_arm", {"variants": out})
