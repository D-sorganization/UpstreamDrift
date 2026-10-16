"""Planar five-bar tangent state space (NSX-84b, #12817): exact sympy derivation.

Reference section: ``loop_nullspace.tex``, "Tangent State Space for Several Degrees of Freedom".

Model.  A planar five-bar: base joints ``A = (0, 0)`` and ``B = (d, 0)``; two
two-link arms with absolute proximal angles ``a1, a2`` and relative elbow angles
``b1, b2``; the arm tips meet in one point ``E`` (closure ``Phi = E_1 - E_2 =
0``, two rows).  Generalized coordinates ``q = (a1, b1, a2, b2)`` (``n_v = 4``),
two closure rows, so the tangent dimension is ``k = 2``.  Each arm carries a
point mass ``m1`` at the elbow and ``m2`` at the tip.  All four joints are
torque-actuated and ``tau`` is the feed-forward load held fixed.

Independent route (no KKT system, no production code).  The configuration is
parametrised by the actuated angles ``s = (a1, a2)``; the passive angles come
from the closed-form two-circle intersection, ``q = gamma(s)``.  The reduced
Euler-Lagrange equation ``M_r(s) s'' = Q^T tau + dL/ds - d_s(M_r s') s'``
(``Q = d gamma / d s``) is linearised in ``(ds, ds')`` and then mapped to the
moving-frame coordinates ``zeta = (xi, eta) = (P^T dq, P^T dv)`` with the
time-dependent map ``zeta = T(s, s') (ds, ds')``::

    zeta' = (T F_r T^{-1} + T' T^{-1}) zeta,    T' = d_s T s' + d_{s'} T s''.

The frame ``P(q)`` is the Gram-Schmidt orthonormalisation of
``(Pi e_1, Pi e_3)`` with ``Pi = I - J^T (J J^T)^{-1} J`` (smooth wherever
``rank J = 2``).  The golden also stores the moving-frame connection pieces
``Gamma_v = P^T d_q P[v]`` (skew) and the normal-offset term, and the
literal ``k = 1`` -style connection ``C_ij = (d_q P_i [P e_j]) . a`` so a test
can show the latter is *not* the correct ``k > 1`` generalisation.

Nothing here imports repository production code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as cm  # noqa: E402

NAME = "planar_five_bar_tangent_state_space"
R = sp.Rational
DPS = 40

PARAMS = {
    "d": R(1),
    "L1": R(1),
    "L2": R(3, 2),
    "m1": R(1),
    "m2": R(4, 5),
    "g": R(10),
}
S0 = (R(11, 10), R(19, 10))  # actuated angles a1, a2 at the nominal state
SD0 = (R(7, 10), R(-2, 5))  # actuated rates
TAU0 = (R(1, 2), R(-3, 10), R(1, 5), R(1, 10))  # feed-forward joint load

A1, A2, D1, D2 = sp.symbols("a1 a2 sd1 sd2", real=True)
S_VEC = sp.Matrix([A1, A2])
SD_VEC = sp.Matrix([D1, D2])
TAU = sp.Matrix(sp.symbols("tau1:5", real=True))
QS = sp.Matrix(sp.symbols("q0:4", real=True))


def _unit(angle: sp.Expr) -> sp.Matrix:
    return sp.Matrix([sp.cos(angle), sp.sin(angle)])


def _tips(q: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]:
    """Elbow and tip positions of both arms as functions of ``q``."""
    p = PARAMS
    base_b = sp.Matrix([p["d"], 0])
    e1 = p["L1"] * _unit(q[0])
    t1 = e1 + p["L2"] * _unit(q[0] + q[1])
    e2 = base_b + p["L1"] * _unit(q[2])
    t2 = e2 + p["L2"] * _unit(q[2] + q[3])
    return e1, t1, e2, t2


def passive_angles() -> sp.Matrix:
    """``q = gamma(s)``: closed-form two-circle intersection of the elbows."""
    p = PARAMS
    e1 = p["L1"] * _unit(A1)
    e2 = sp.Matrix([p["d"], 0]) + p["L1"] * _unit(A2)
    mid = (e1 + e2) / 2
    dv = e2 - e1
    dn = sp.sqrt(dv.dot(dv))
    half = sp.sqrt(p["L2"] ** 2 - dn**2 / 4)
    nrm = sp.Matrix([-dv[1], dv[0]]) / dn
    tip = mid + half * nrm  # tip above the base line
    b1 = sp.atan2(tip[1] - e1[1], tip[0] - e1[0]) - A1
    b2 = sp.atan2(tip[1] - e2[1], tip[0] - e2[0]) - A2
    return sp.Matrix([A1, b1, A2, b2])


def closure_jacobian_sym() -> sp.Matrix:
    _, t1, _, t2 = _tips(QS)
    return (t1 - t2).jacobian(QS)


def frame_from_jacobian(jac: mp.matrix) -> mp.matrix:
    """Gram-Schmidt of ``(Pi e_1, Pi e_3)``, ``Pi = I - J^T (J J^T)^-1 J``.

    Evaluated in ``mpmath`` arithmetic (the symbolic 2 x 2 inverse of the trig
    Jacobian is needlessly slow); shape ``(4, 2)``.
    """
    proj = mp.eye(4) - jac.T * mp.inverse(jac * jac.T) * jac
    w1, w2 = proj[:, 0], proj[:, 2]
    p1 = w1 / mp.sqrt(sum(x * x for x in w1))
    res = w2 - (sum(p1[i] * w2[i] for i in range(4))) * p1
    p2 = res / mp.sqrt(sum(x * x for x in res))
    out = mp.matrix(4, 2)
    for i in range(4):
        out[i, 0], out[i, 1] = p1[i], p2[i]
    return out


def reduced_dynamics(gamma: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """``(M_r, Q, rhs0)`` with ``M_r s'' = Q^T tau + rhs0`` (EL equation)."""
    p = PARAMS
    e1, t1, e2, t2 = _tips(gamma)
    masses = [(e1, p["m1"]), (t1, p["m2"]), (e2, p["m1"]), (t2, p["m2"])]
    m_r = sp.zeros(2, 2)
    pot = 0
    for pos, mass in masses:
        jp = pos.jacobian(S_VEC)
        m_r += mass * jp.T * jp
        pot += mass * p["g"] * pos[1]
    q_mat = gamma.jacobian(S_VEC)
    kin_grad = sp.Matrix(
        [(SD_VEC.T * m_r.diff(s_i) * SD_VEC)[0, 0] / 2 for s_i in S_VEC]
    )
    pot_grad = sp.Matrix([sp.diff(pot, s_i) for s_i in S_VEC])
    inertia_term = (m_r * SD_VEC).jacobian(S_VEC) * SD_VEC
    rhs0 = kin_grad - pot_grad - inertia_term
    return m_r, q_mat, rhs0


def _lam(args: list[sp.Symbol], expr: sp.Matrix) -> Any:
    return sp.lambdify(args, expr, modules="mpmath", cse=True)


def _mpf(x: Any) -> Any:
    return mp.mpf(str(sp.N(x, DPS)))


class Reduced:
    """Closed-form pieces of the reduced model, lambdified to ``mpmath``.

    Symbolic: ``gamma(s)``, ``Q = d gamma/d s``, ``d_s(Q s')``, ``M_r`` and the
    Euler-Lagrange right-hand side, and the ambient frame ``P(q)``.  The outer
    derivatives (Jacobian of ``s''`` and the time derivative of ``T``) are taken
    numerically in 40-digit arithmetic (``mp.jacobian`` / ``mp.diff``), whose
    truncation error is far below ``1e-25``.
    """

    def __init__(self) -> None:
        mp.mp.dps = DPS
        gamma = passive_angles()
        q_mat = gamma.jacobian(S_VEC)
        m_r, _, rhs0 = reduced_dynamics(gamma)
        v_amb = q_mat * SD_VEC
        vjac = v_amb.jacobian(S_VEC)
        sd_args = [A1, A2, D1, D2]
        self.gamma = _lam([A1, A2], gamma)
        self.q_mat = _lam([A1, A2], q_mat)
        self.vjac = _lam(sd_args, vjac)
        self.m_r = _lam([A1, A2], m_r)
        self.rhs0 = _lam(sd_args, rhs0)
        self._jac = _lam(list(QS), closure_jacobian_sym())

    def accel(self, s: list[Any], sd: list[Any], tau: Any) -> mp.matrix:
        """``s'' = M_r^{-1} (Q^T tau + rhs0)``."""
        rhs = mp.matrix(self.q_mat(*s)).T * tau + mp.matrix(self.rhs0(*s, *sd))
        return mp.lu_solve(mp.matrix(self.m_r(*s)), rhs)

    def frame(self, *q: Any) -> mp.matrix:
        """Ambient frame ``P(q)``."""
        return frame_from_jacobian(mp.matrix(self._jac(*q)))

    def p_of_s(self, s: list[Any]) -> mp.matrix:
        return self.frame(*mp.matrix(self.gamma(*s)))

    def t_map(self, s: list[Any], sd: list[Any]) -> mp.matrix:
        """``zeta = T (ds, ds')`` with ``T = blkdiag(P^T, P^T) d(q, v)/d(s, s')``."""
        q_m = mp.matrix(self.q_mat(*s))
        v_jac = mp.matrix(self.vjac(*s, *sd))
        p_t = self.p_of_s(s).T
        t1 = p_t * q_m
        t3 = p_t * v_jac
        out = mp.zeros(4, 4)
        for i in range(2):
            for j in range(2):
                out[i, j] = t1[i, j]
                out[2 + i, 2 + j] = t1[i, j]
                out[2 + i, j] = t3[i, j]
        return out


def build() -> dict[str, Any]:
    """Compute the golden quantities at the nominal state (mpmath precision)."""
    red = Reduced()
    s0 = [_mpf(x) for x in S0]
    sd0 = [_mpf(x) for x in SD0]
    tau = mp.matrix([_mpf(x) for x in TAU0])
    sdd0 = red.accel(s0, sd0, tau)

    def acc_of(*x: Any) -> mp.matrix:
        return red.accel([x[0], x[1]], [x[2], x[3]], tau)

    x0 = mp.matrix(s0 + sd0)
    f_acc_jac = mp.jacobian(acc_of, x0)  # (2, 4): [d/ds | d/ds']
    f_r = mp.zeros(4, 4)
    for i in range(2):
        f_r[i, 2 + i] = 1
        for j in range(4):
            f_r[2 + i, j] = f_acc_jac[i, j]
    q_m = mp.matrix(red.q_mat(*s0))
    g_r = mp.inverse(mp.matrix(red.m_r(*s0))) * q_m.T  # ds''/dtau (2 x 4)

    def t_of_time(t: Any) -> mp.matrix:
        s_t = [s0[i] + sd0[i] * t for i in range(2)]
        sd_t = [sd0[i] + sdd0[i] * t for i in range(2)]
        return red.t_map(s_t, sd_t)

    t_num = t_of_time(0)
    t_dot = mp.matrix(4, 4)
    for i in range(4):
        for j in range(4):
            t_dot[i, j] = mp.diff(lambda t, i=i, j=j: t_of_time(t)[i, j], 0)
    gam = mp.matrix(red.gamma(*s0))
    v0 = q_m * mp.matrix(sd0)
    vjac0 = mp.matrix(red.vjac(*s0, *sd0))
    acc_amb = q_m * sdd0 + vjac0 * mp.matrix(sd0)
    d_p = []
    for i in range(4):
        d_p.append(_partial_frame(red, gam, i))
    return {
        "gamma": gam,
        "Q": q_m,
        "v": v0,
        "a": acc_amb,
        "sdd": sdd0,
        "F_r": f_r,
        "G_r": g_r,
        "T": t_num,
        "T_dot": t_dot,
        "P": red.p_of_s(s0),
        "dP": d_p,
    }


def _partial_frame(red: Reduced, gam: mp.matrix, i: int) -> mp.matrix:
    """``dP_amb/dq_i`` at ``gamma(s0)`` (40-digit central difference)."""
    out = mp.matrix(4, 2)
    for r in range(4):
        for c in range(2):

            def entry(x: Any, r: int = r, c: int = c) -> Any:
                q = [gam[k] + (x if k == i else 0) for k in range(4)]
                return red.frame(*q)[r, c]

            out[r, c] = mp.diff(entry, 0)
    return out


def _mp_list(m: mp.matrix) -> list[Any]:
    rows, cols = m.rows, m.cols
    return [[float(m[i, j]) for j in range(cols)] for i in range(rows)]


def frame_quantities(res: dict[str, Any]) -> dict[str, Any]:
    """Frame-coordinate ``F``, ``G`` and the connection pieces from ``build()``."""
    t_inv = res["T"] ** -1
    f_frame = res["T"] * res["F_r"] * t_inv + res["T_dot"] * t_inv
    p_mat, v, acc = res["P"], res["v"], res["a"]
    # G: eta' += T1 (ds'' / dtau) dtau, T1 = P^T Q (upper-left 2x2 block of T).
    t1 = res["T"][0:2, 0:2]
    g_low = t1 * res["G_r"]
    g_frame = mp.zeros(4, 4)
    for i in range(2):
        for j in range(4):
            g_frame[2 + i, j] = g_low[i, j]
    d_v = _dp(res["dP"], v)  # d_q P [v], 4 x 2
    gamma_v = p_mat.T * d_v  # P^T d_q P [v], 2 x 2 (skew)
    omega = p_mat.T * v
    conn_literal = mp.zeros(2, 2)
    offset_term = mp.zeros(2, 2)
    normal = mp.eye(4) - p_mat * p_mat.T
    for j in range(2):
        w = p_mat[:, j]
        d_w = _dp(res["dP"], w)
        for i in range(2):
            col_i = d_w[:, i]
            conn_literal[i, j] = sum(col_i[r] * acc[r] for r in range(4))
        o_col = normal * (d_w * omega)
        for i in range(2):
            offset_term[i, j] = sum(d_v[r, i] * o_col[r] for r in range(4))
    return {
        "F": f_frame,
        "G": g_frame,
        "Gamma_v": gamma_v,
        "offset_term": offset_term,
        "connection_literal": conn_literal,
        "omega": omega,
    }


def _dp(dpq: list[mp.matrix], w: mp.matrix) -> mp.matrix:
    """``d_q P [w]`` from the four partials ``dP/dq_i``."""
    out = mp.zeros(4, 2)
    for i in range(4):
        out += dpq[i] * w[i]
    return out


def generate() -> dict[str, Any]:
    mp.mp.dps = DPS
    res = build()
    fq = frame_quantities(res)
    p = PARAMS
    closure = _closure_residual(res["gamma"])
    payload = {
        "model": {
            "description": (
                "planar five-bar, q = (a1, b1, a2, b2), tips coincide; point masses "
                "m1 at the elbows and m2 at the tips; torque on all four joints"
            ),
            "parameters": {k: str(v) for k, v in p.items()},
            "frame": "P = Gram-Schmidt(Pi e_1, Pi e_3), Pi = I - J^T (J J^T)^-1 J",
        },
        "state": {
            "s": [str(x) for x in S0],
            "s_dot": [str(x) for x in SD0],
            "tau": [str(x) for x in TAU0],
            "q": [float(x) for x in res["gamma"]],
            "v": [float(x) for x in res["v"]],
            "closure_residual": closure,
        },
        "nominal": {
            "acceleration": [float(x) for x in res["a"]],
            "P": _mp_list(res["P"]),
            "omega": [float(x) for x in fq["omega"]],
        },
        "F": _mp_list(fq["F"]),
        "G": _mp_list(fq["G"]),
        "Gamma_v": _mp_list(fq["Gamma_v"]),
        "offset_term": _mp_list(fq["offset_term"]),
        "connection_literal": _mp_list(fq["connection_literal"]),
        "eigenvalues_F": _eigs(fq["F"]),
    }
    return cm.dump(NAME, payload)


def _closure_residual(gamma: mp.matrix) -> float:
    q = [sp.Float(str(gamma[i]), DPS) for i in range(4)]
    _, t1, _, t2 = _tips(sp.Matrix(q))
    return float(sp.N((t1 - t2).norm(), 20))


def _eigs(mat_f: mp.matrix) -> list[list[float]]:
    vals = mp.eig(mat_f, left=False, right=False)
    return sorted([float(mp.re(v)), float(mp.im(v))] for v in vals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = json.loads(json.dumps(generate(), allow_nan=False))
    target = cm.GOLDENS_DIR / f"{NAME}.json"
    if args.check:
        stored = json.loads(target.read_text(encoding="utf-8"))
        if stored != payload:
            sys.stdout.write(f"STALE {target}\n")
            return 1
        return 0
    sys.stdout.write(f"{cm.write_json(NAME, payload)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
