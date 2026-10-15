"""The c-plus/c-minus gross-power identity (R6) checked on the planar fixture.

Identity.  With ``p_i = tau_i v_i`` and ``[x]_+ = max(x, 0)``::

    c_+ [p]_+ + c_- [-p]_+ = (c_+ + c_-)/2 |p| + (c_+ - c_-)/2 p.

Summed over joints with *uniform* ``c_+, c_-`` the second term is
``(c_+ - c_-)/2 P_net`` with ``P_net = v^T tau = v^T (b - J_c^T lam)`` which is
independent of ``lam`` whenever ``J_c v = 0``.  The minimiser over ``lam`` of
the uniform objective therefore does not depend on the ratio ``c_-/c_+``.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

import _common as cm
import planar_golfer as pg


def _objective(
    lam: np.ndarray,
    jc: np.ndarray,
    b: np.ndarray,
    v: np.ndarray,
    cp: np.ndarray,
    cm_: np.ndarray,
) -> float:
    tau = b - jc.T @ lam
    p = tau * v
    return float(np.sum(cp * np.maximum(p, 0.0) + cm_ * np.maximum(-p, 0.0)))


def _vertex_minimum(
    jc: np.ndarray, b: np.ndarray, v: np.ndarray, cp: np.ndarray, cm_: np.ndarray
) -> dict[str, Any]:
    """Minimum of a convex piecewise-linear function of ``lam`` in R^2.

    The minimum is attained at an intersection of two lines ``p_i(lam) = 0`` (the
    arrangement is coercive: the rows of ``J_c^T`` span R^2).  All minimising
    vertices are returned; a flat optimal edge shows up as two or more of them,
    which is why the pure l1 objective needs a strictly convex partner (R6).
    """
    lines = [i for i in range(len(b)) if abs(v[i]) > 0]
    cand = []
    for i, j in itertools.combinations(lines, 2):
        a = np.array([jc[:, i], jc[:, j]])
        if abs(np.linalg.det(a)) < 1e-12:
            continue
        lam = np.linalg.solve(a, np.array([b[i], b[j]]))
        cand.append((_objective(lam, jc, b, v, cp, cm_), lam))
    best = min(c[0] for c in cand)
    tol = 1e-9 * max(1.0, abs(best))
    optimal = sorted([float(x) for x in c[1]] for c in cand if c[0] <= best + tol)
    return {"value": best, "optimal_vertices": optimal, "unique": len(optimal) == 1}


def _same_set(a: list[list[float]], b: list[list[float]]) -> bool:
    return len(a) == len(b) and all(
        np.allclose(x, y, atol=1e-8) for x, y in zip(a, b, strict=True)
    )


def generate() -> dict[str, Any]:
    base = pg.generic_configuration()
    # Same closed configuration as planar_golfer_full, different lead rates and
    # accelerations chosen so that negative power is unavoidable (min sum|p| > |P_net|).
    v_ = pg._complete_rates(base["q"], np.zeros(5), {0: -2.0, 1: -1.0, 2: 4.0}, [3, 4])
    a_ = pg._complete_rates(
        base["q"], v_, {0: 5.0, 1: -3.0, 2: 2.0}, [3, 4], acceleration=True
    )
    cfg = {"q": base["q"], "v": v_, "a": a_}
    ev = pg.evaluate(cfg["q"], cfg["v"])
    jc, v = ev["Jc"], cfg["v"]
    b = ev["M"] @ cfg["a"] + ev["h"]
    assert np.linalg.norm(jc @ v) < 1e-12
    rng = np.random.default_rng(20261014)
    samples = []
    p_net_ref = float(v @ b)
    max_dev = 0.0
    max_ident_err = 0.0
    for lam in rng.uniform(-30.0, 30.0, size=(6, 2)):
        tau = b - jc.T @ lam
        p = tau * v
        p_net = float(p.sum())
        max_dev = max(max_dev, abs(p_net - p_net_ref))
        for cp_s, cm_s in ((1.0, 1.0), (1.0, 0.25), (3.0, 0.75)):
            lhs = float(np.sum(cp_s * np.maximum(p, 0) + cm_s * np.maximum(-p, 0)))
            rhs = (
                0.5 * (cp_s + cm_s) * float(np.abs(p).sum())
                + 0.5 * (cp_s - cm_s) * p_net
            )
            max_ident_err = max(max_ident_err, abs(lhs - rhs))
        samples.append(
            {
                "lambda": [float(x) for x in lam],
                "P_net": p_net,
                "sum_abs_p": float(np.abs(p).sum()),
            }
        )
    n = len(b)
    uniform = {}
    for label, (cp_s, cm_s) in {
        "1_1": (1.0, 1.0),
        "1_0.25": (1.0, 0.25),
        "3_0.75": (3.0, 0.75),
    }.items():
        cp, cm_ = np.full(n, cp_s), np.full(n, cm_s)
        res = _vertex_minimum(jc, b, v, cp, cm_)
        res["c_plus"], res["c_minus"] = cp_s, cm_s
        uniform[label] = res
    cp_h = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    cm_h = np.array([0.2, 1.0, 0.2, 1.0, 0.2])
    hetero = _vertex_minimum(jc, b, v, cp_h, cm_h)
    hetero["c_plus"], hetero["c_minus"] = cp_h.tolist(), cm_h.tolist()
    ref = uniform["1_1"]["optimal_vertices"]
    same = [_same_set(ref, u["optimal_vertices"]) for u in uniform.values()]
    return cm.dump(
        "c_pm_identity",
        {
            "fixture": "planar_golfer_full (same q, rates and accelerations given here)",
            "q": [float(x) for x in cfg["q"]],
            "v": [float(x) for x in v],
            "a": [float(x) for x in cfg["a"]],
            "b": [float(x) for x in b],
            "J_c": [[float(x) for x in r] for r in jc],
            "min_sum_abs_p": uniform["1_1"]["value"],
            "P_net_reference": p_net_ref,
            "P_net_max_deviation_over_lambda": max_dev,
            "identity_max_error": max_ident_err,
            "samples": samples,
            "uniform_weights_minimiser": uniform,
            "uniform_minimisers_coincide": all(same),
            "heterogeneous_weights_minimiser": hetero,
            "heterogeneous_differs_from_uniform": not _same_set(
                ref, hetero["optimal_vertices"]
            ),
            "min_value_relation": (
                "min F(c+,c-) = (c+ + c-)/2 * min sum|p| + (c+ - c-)/2 * P_net for uniform weights"
            ),
        },
    )
