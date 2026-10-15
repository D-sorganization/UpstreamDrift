"""Analytic envelope cases (NSX-00 fixture (d)): closed-form torque bounds.

The envelopes are computed with *exact rational arithmetic* by enumerating the
vertices of the (low-dimensional) feasible polytope.  No numerical LP solver is
involved, so these values can check the production LP/SOCP code without sharing
any of its failure modes.

Base problem (E1)
-----------------
Three fully actuated joints, a two-row closure (``m = 2``)::

    tau(lam) = tau_ID - J_c^T lam,   J_c^T = [[1, 0], [0, 1], [1, 1]],
    tau_ID   = (1/2, -1/4, 1),       |tau_i| <= c_i,  c = (1/2, 3/4, 1/2).

The feasible set in ``lam`` is a polygon: the rectangle from joints 1 and 2 cut
by the band from joint 3.
"""

from __future__ import annotations

import itertools
from fractions import Fraction as F
from typing import Any

import sympy as sp

import _common as cm

JCT = [[F(1), F(0)], [F(0), F(1)], [F(1), F(1)]]
TAU_ID = [F(1, 2), F(-1, 4), F(1)]
CAP = [F(1, 2), F(3, 4), F(1, 2)]
NAMES = ["tau_1", "tau_2", "tau_3", "lam_1", "lam_2"]


def _fstr(x: F) -> str:
    return str(x)


def _solve(rows: list[list[F]], rhs: list[F]) -> list[F] | None:
    """Exact Gaussian elimination for a square system; ``None`` if singular."""
    n = len(rows)
    a = [r[:] + [b] for r, b in zip(rows, rhs, strict=True)]
    for c in range(n):
        piv = next((r for r in range(c, n) if a[r][c] != 0), None)
        if piv is None:
            return None
        a[c], a[piv] = a[piv], a[c]
        pv = a[c][c]
        a[c] = [x / pv for x in a[c]]
        for r in range(n):
            if r != c and a[r][c] != 0:
                f = a[r][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c], strict=True)]
    return [a[r][n] for r in range(n)]


def vertices(g: list[list[F]], h: list[F]) -> list[list[F]]:
    """All vertices of ``{x : g x <= h}`` (exact), possibly empty."""
    d = len(g[0])
    found: list[list[F]] = []
    for sub in itertools.combinations(range(len(g)), d):
        x = _solve([g[i] for i in sub], [h[i] for i in sub])
        if x is None:
            continue
        if all(
            sum(gi * xi for gi, xi in zip(g[k], x, strict=True)) <= h[k]
            for k in range(len(g))
        ):
            if x not in found:
                found.append(x)
    return found


def _affine(coeff: list[F], const: F, x: list[F]) -> F:
    return const + sum(c * xi for c, xi in zip(coeff, x, strict=True))


def _polygon_rows(
    tau_id: list[F], cap: list[F]
) -> tuple[list[list[F]], list[F], list[str]]:
    """``-c_i <= tau_i - (J^T lam)_i <= c_i`` as ``G lam <= h`` with row labels."""
    g: list[list[F]] = []
    h: list[F] = []
    labels: list[str] = []
    for i in range(3):
        # tau_i - j.lam <= c  ->  -j.lam <= c - tau_i
        g.append([-v for v in JCT[i]])
        h.append(cap[i] - tau_id[i])
        labels.append(f"cap_{i + 1}_upper")
        g.append(list(JCT[i]))
        h.append(cap[i] + tau_id[i])
        labels.append(f"cap_{i + 1}_lower")
    return g, h, labels


def _quantities() -> list[tuple[str, list[F], F]]:
    q: list[tuple[str, list[F], F]] = []
    for i in range(3):
        q.append((NAMES[i], [-v for v in JCT[i]], TAU_ID[i]))
    q.append(("lam_1", [F(1), F(0)], F(0)))
    q.append(("lam_2", [F(0), F(1)], F(0)))
    return q


SIGN_CASES = [
    (F(0), F(0)),
    (F(0), F(1)),
    (F(-1), F(0)),
    (F(1, 4), F(1)),
    (F(-1), F(-1, 4)),
    (F(-1), F(1)),
]


def _sign_class(lo: F, hi: F) -> str:
    if lo == 0 and hi == 0:
        return "ZERO"
    if lo > 0:
        return "POSITIVE"
    if hi < 0:
        return "NEGATIVE"
    return "INDETERMINATE"


def static_envelope() -> dict[str, Any]:
    g, h, labels = _polygon_rows(TAU_ID, CAP)
    verts = vertices(g, h)
    env = {}
    for name, coeff, const in _quantities():
        vals = [_affine(coeff, const, v) for v in verts]
        lo, hi = min(vals), max(vals)
        env[name] = {
            "lower": _fstr(lo),
            "upper": _fstr(hi),
            "lower_float": float(lo),
            "upper_float": float(hi),
            "status_lower": "BOUNDED",
            "status_upper": "BOUNDED",
            "sign_class": _sign_class(lo, hi),
            "argmin_vertex": [_fstr(c) for c in verts[vals.index(lo)]],
            "argmax_vertex": [_fstr(c) for c in verts[vals.index(hi)]],
        }
    return {
        "J_c_T": [[_fstr(v) for v in r] for r in JCT],
        "tau_ID": [_fstr(v) for v in TAU_ID],
        "capacity": [_fstr(v) for v in CAP],
        "rows": [
            {"label": lab, "g": [_fstr(v) for v in row], "h": _fstr(hh)}
            for lab, row, hh in zip(labels, g, h, strict=True)
        ],
        "vertices": [[_fstr(c) for c in v] for v in sorted(verts)],
        "envelopes": env,
        "chebyshev": {
            "euclidean": _chebyshev(g, h, [[F(1), F(0)], [F(0), F(1)]]),
            "weighted_diag_1_4": _chebyshev(g, h, [[F(1), F(0)], [F(0), F(4)]]),
        },
        "polygon_area": _area(verts),
        "sign_class_cases": [
            {"lower": _fstr(lo), "upper": _fstr(hi), "sign_class": _sign_class(lo, hi)}
            for lo, hi in SIGN_CASES
        ],
        "sign_class_rule": (
            "ZERO iff lower = upper = 0; POSITIVE iff lower > 0; NEGATIVE iff upper < 0; "
            "otherwise INDETERMINATE (a bound that touches zero is INDETERMINATE)"
        ),
    }


def _area(verts: list[list[F]]) -> str:
    cx = sum(v[0] for v in verts) / len(verts)
    cy = sum(v[1] for v in verts) / len(verts)
    import math

    ordered = sorted(
        verts, key=lambda v: math.atan2(float(v[1] - cy), float(v[0] - cx))
    )
    area = F(0)
    for a, b in zip(ordered, ordered[1:] + ordered[:1], strict=True):
        area += a[0] * b[1] - b[0] * a[1]
    return _fstr(abs(area) / 2)


def _chebyshev(g: list[list[F]], h: list[F], w: list[list[F]]) -> dict[str, Any]:
    """Largest inscribed ball in the metric ``|z|_W^2 = z^T W z``.

    ``max r  s.t.  a_i.z + r*sqrt(a_i^T W^-1 a_i) <= b_i``; solved exactly by
    enumerating the vertices of the 3-variable LP (``z1, z2, r``).
    """
    winv = sp.Matrix(
        [[sp.Rational(x.numerator, x.denominator) for x in r] for r in w]
    ).inv()
    norms = []
    for row in g:
        a = sp.Matrix([sp.Rational(x.numerator, x.denominator) for x in row])
        norms.append(sp.sqrt((a.T * winv * a)[0, 0]))
    rows = [
        [sp.Rational(x.numerator, x.denominator) for x in row] + [nrm]
        for row, nrm in zip(g, norms, strict=True)
    ]
    rhs = [sp.Rational(x.numerator, x.denominator) for x in h]
    rows.append([sp.Integer(0), sp.Integer(0), sp.Integer(-1)])
    rhs.append(sp.Integer(0))
    feasible_vertices: list[sp.Matrix] = []
    for sub in itertools.combinations(range(len(rows)), 3):
        a = sp.Matrix([rows[i] for i in sub])
        if a.det() == 0:
            continue
        x = a.LUsolve(sp.Matrix([rhs[i] for i in sub])).applyfunc(sp.simplify)
        if all(
            (sum(rows[k][j] * x[j] for j in range(3)) - rhs[k]).evalf(40) <= 1e-30
            for k in range(len(rows))
        ):
            feasible_vertices.append(x)
    radius = max(feasible_vertices, key=lambda v: float(v[2]))[2]
    optimal = {
        (str(sp.simplify(v[0])), str(sp.simplify(v[1])))
        for v in feasible_vertices
        if abs(float((v[2] - radius).evalf(40))) < 1e-30
    }
    assert len(optimal) == 1, "Chebyshev center must be unique for the golden"
    x = next(
        v for v in feasible_vertices if abs(float((v[2] - radius).evalf(40))) < 1e-30
    )
    return {
        "center": [str(sp.simplify(x[0])), str(sp.simplify(x[1]))],
        "radius": str(sp.simplify(x[2])),
        "center_float": [float(x[0]), float(x[1])],
        "radius_float": float(x[2]),
        "center_unique": True,
    }


# --------------------------------------------------------------------------- #
# Trajectory coupling (two frames)                                             #
# --------------------------------------------------------------------------- #


def trajectory_coupled() -> dict[str, Any]:
    """Two frames with ``|lam_k(2) - lam_k(1)| <= delta``; static vs coupled."""
    tau2 = [F(0), F(0), F(1, 2)]
    delta = F(1, 8)
    g1, h1, lab1 = _polygon_rows(TAU_ID, CAP)
    g2, h2, lab2 = _polygon_rows(tau2, CAP)
    g: list[list[F]] = []
    h: list[F] = []
    for row, hh in zip(g1, h1, strict=True):
        g.append(row + [F(0), F(0)])
        h.append(hh)
    for row, hh in zip(g2, h2, strict=True):
        g.append([F(0), F(0)] + row)
        h.append(hh)
    for k in range(2):
        for sign in (1, -1):
            r = [F(0)] * 4
            r[k] = F(-sign)
            r[2 + k] = F(sign)
            g.append(r)
            h.append(delta)
    verts = vertices(g, h)
    static2 = vertices(g2, h2)
    out: dict[str, Any] = {
        "delta": _fstr(delta),
        "tau_ID_frame_2": [_fstr(v) for v in tau2],
    }
    for idx, name in (
        (2, "lam_1_frame_2"),
        (3, "lam_2_frame_2"),
        (0, "lam_1_frame_1"),
        (1, "lam_2_frame_1"),
    ):
        vals = [v[idx] for v in verts]
        base = [
            v[idx - 2] if idx >= 2 else v[idx]
            for v in (static2 if idx >= 2 else vertices(g1, h1))
        ]
        out[name] = {
            "coupled_lower": _fstr(min(vals)),
            "coupled_upper": _fstr(max(vals)),
            "static_lower": _fstr(min(base)),
            "static_upper": _fstr(max(base)),
        }
    out["coupled_is_subset_of_static"] = all(
        F(out[k]["static_lower"]) <= F(out[k]["coupled_lower"])
        and F(out[k]["coupled_upper"]) <= F(out[k]["static_upper"])
        for k in out
        if isinstance(out[k], dict)
    )
    out["coupled_strictly_narrower_somewhere"] = any(
        F(out[k]["coupled_lower"]) > F(out[k]["static_lower"])
        or F(out[k]["coupled_upper"]) < F(out[k]["static_upper"])
        for k in out
        if isinstance(out[k], dict)
    )
    return out


# --------------------------------------------------------------------------- #
# Unbounded and infeasible certificates                                        #
# --------------------------------------------------------------------------- #


def unbounded_case() -> dict[str, Any]:
    """Only joint 1 is capacity limited; the second multiplier is free."""
    g = [[F(-1), F(0)], [F(1), F(0)]]
    h = [CAP[0] - TAU_ID[0], CAP[0] + TAU_ID[0]]
    ray = [F(0), F(1)]
    assert all(sum(gi * ri for gi, ri in zip(row, ray, strict=True)) <= 0 for row in g)
    lam1 = vertices_1d(g, h)
    return {
        "rows": [
            {"g": [_fstr(v) for v in r], "h": _fstr(hh)}
            for r, hh in zip(g, h, strict=True)
        ],
        "recession_ray": [_fstr(v) for v in ray],
        "statuses": {
            "lam_1": {
                "lower": _fstr(lam1[0]),
                "upper": _fstr(lam1[1]),
                "status": "BOUNDED",
            },
            "lam_2": {
                "status_lower": "UNBOUNDED",
                "status_upper": "UNBOUNDED",
                "lower": None,
                "upper": None,
            },
            "tau_3": {
                "status_lower": "UNBOUNDED",
                "status_upper": "UNBOUNDED",
                "lower": None,
                "upper": None,
            },
        },
        "note": "ray certificate: g.ray <= 0 and c.ray != 0 for lam_2 and tau_3",
    }


def vertices_1d(g: list[list[F]], h: list[F]) -> tuple[F, F]:
    lo, hi = F(-(10**9)), F(10**9)
    for row, hh in zip(g, h, strict=True):
        if row[0] > 0:
            hi = min(hi, hh / row[0])
        elif row[0] < 0:
            lo = max(lo, hh / row[0])
    return lo, hi


def _feasible(g: list[list[F]], h: list[F]) -> bool:
    """Exact feasibility of ``g x <= h`` by Fourier-Motzkin elimination."""
    rows = [(list(r), hh) for r, hh in zip(g, h, strict=True)]
    d = len(g[0])
    for var in range(d):
        pos = [r for r in rows if r[0][var] > 0]
        neg = [r for r in rows if r[0][var] < 0]
        zero = [r for r in rows if r[0][var] == 0]
        new = zero[:]
        for rp, hp in pos:
            for rn, hn in neg:
                a, b = rp[var], -rn[var]
                new.append(
                    (
                        [b * x + a * y for x, y in zip(rp, rn, strict=True)],
                        b * hp + a * hn,
                    )
                )
        rows = new
    return all(hh >= 0 for _, hh in rows)


def infeasible_case() -> dict[str, Any]:
    """Contradictory hypotheses with an inclusion-minimal conflict set."""
    rows = [
        ("H-TORQUE-CAPACITY:joint1:upper", [F(1), F(0)], F(3, 4)),
        ("H-TORQUE-CAPACITY:joint1:lower", [F(-1), F(0)], F(-1, 4)),
        ("H-GRIP-PRELOAD:axial_nonpositive", [F(1), F(0)], F(0)),
        ("H-TORQUE-CAPACITY:joint2:upper", [F(0), F(1)], F(3, 4)),
        ("H-TORQUE-CAPACITY:joint2:lower", [F(0), F(-1)], F(5, 4)),
    ]
    g = [r[1] for r in rows]
    h = [r[2] for r in rows]
    assert not _feasible(g, h)
    iis = []
    for size in range(1, len(rows) + 1):
        for sub in itertools.combinations(range(len(rows)), size):
            if any(set(prev) <= set(sub) for prev in iis):
                continue
            if not _feasible([g[i] for i in sub], [h[i] for i in sub]):
                iis.append(sub)
    farkas = [F(0), F(1), F(1), F(0), F(0)]
    combo = [sum(farkas[i] * g[i][j] for i in range(5)) for j in range(2)]
    assert combo == [0, 0] and sum(farkas[i] * h[i] for i in range(5)) < 0
    return {
        "rows": [
            {"label": lab, "g": [_fstr(v) for v in r], "h": _fstr(hh)}
            for lab, r, hh in rows
        ],
        "feasible": False,
        "inclusion_minimal_conflict_sets": [[rows[i][0] for i in sub] for sub in iis],
        "farkas_multipliers": [_fstr(v) for v in farkas],
        "farkas_rhs": _fstr(sum(farkas[i] * h[i] for i in range(5))),
    }


def generate() -> dict[str, Any]:
    return cm.dump(
        "analytic_envelope",
        {
            "description": "Exact envelopes of a 2-D multiplier polygon; infeasible and unbounded certificates.",
            "static": static_envelope(),
            "trajectory_coupled": trajectory_coupled(),
            "unbounded": unbounded_case(),
            "infeasible": infeasible_case(),
        },
    )
