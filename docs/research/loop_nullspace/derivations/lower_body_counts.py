"""Lower-body loop: ground-reaction observation counts (NSX-60, exact sympy).

The spatial two-arm chain of :mod:`spatial_counts` with a floating root and
rigid point feet has the balance ``A x = b``, ``A = [S^T J_c^T J_g^T]``,
``x = (tau, lambda, f_g)``.  A ground-reaction observation is a linear map
``E`` of the contact loads, ``E f_g = g``.  With ``N`` a basis of the null
space of ``A`` (exact, over the rationals), embedding ``E_x = [0 0 E]`` gives

    nullity after observation = nullity(A) - rank(E_x N),

the exact limit of vanishing measurement uncertainty.  ``rank(E_x N)`` is the
dimension of the contact-load null directions the observation can see.

Observation maps are stated through the point forces they sum, so the same
selector applies to the numeric fixture:

* ``net_wrench``: total force and the moment about the root origin; the root
  rows already fix it, so it adds no information (rank 0).
* ``net_force``: total force only (rank 0 for the same reason).
* ``normal_load_per_foot``: the vertical load of each foot.
* ``force_per_foot``: the three force components of each foot.
* ``all_point_forces``: every point force (the identity on ``f_g``).
* ``vertical_per_foot``: a per-foot vertical-cell plate (OF-11, #13067) whose
  force-sensing plane passes through the contacts: per foot the normal load
  and the two CoP moment rows ``sum_k (r_k . t_i)(n . f_k)``.
* ``vertical_per_foot_sensing_depth``: the same plate with its sensing plane a
  depth ``h = 1/50`` below the contacts.  The cells balance moments about that
  plane, so each CoP row gains the shear lever term
  ``- sum_k (r_k . n)(t_i . f_k)``, with ``r_k = p_k - c'`` and ``c'`` the CoP on
  the sensing plane.

The CoPs of the vertical selectors are physically realisable: the contact
point for a point foot and the point one third of the way from its first to
its second contact for a two-point foot, dropped onto the sensing plane.  For
those two selectors the payload also records the rank of the shear rows of the
null basis before and after the observation.

Nothing here imports repository production code.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

import _common as cm
import spatial_counts as sc


def _skew(v: sp.Matrix) -> sp.Matrix:
    return sc._skew(v)


def _block_sum(n_pts: int, members: list[int], comp: int | None) -> sp.Matrix:
    """Rows summing the point forces of ``members`` (all three components, or one)."""
    comps = [comp] if comp is not None else [0, 1, 2]
    rows = []
    for c in comps:
        row = sp.zeros(1, 3 * n_pts)
        for k in members:
            row[0, 3 * k + c] = 1
        rows.append(row)
    return sp.Matrix.vstack(*rows)


def _net_wrench(points: list[sp.Matrix]) -> sp.Matrix:
    """Total force and moment about the root origin of the point forces."""
    n = len(points)
    force = _block_sum(n, list(range(n)), None)
    moment = sp.zeros(3, 3 * n)
    for k, p in enumerate(points):
        moment[:, 3 * k : 3 * k + 3] = _skew(p - sc.ROOT)
    return sp.Matrix.vstack(force, moment)


def selectors(points_per_foot: int) -> dict[str, sp.Matrix]:
    """Exact observation selectors ``E`` (rows x ``3 n_points``)."""
    pts = sc.FEET if points_per_foot == 1 else sc.FEET4
    n = len(pts)
    feet = [list(range(i * points_per_foot, (i + 1) * points_per_foot)) for i in (0, 1)]
    out = {
        "net_wrench": _net_wrench(pts),
        "net_force": _block_sum(n, list(range(n)), None),
        "normal_load_per_foot": sp.Matrix.vstack(*[_block_sum(n, f, 2) for f in feet]),
        "force_per_foot": sp.Matrix.vstack(*[_block_sum(n, f, None) for f in feet]),
        "all_point_forces": sp.eye(3 * n),
    }
    return out


#: Depth of the force-sensing plane below the contacts (OF-11 golden variant).
SENSING_DEPTH = sp.Rational(1, 50)
#: Fraction along a two-point foot (first to second contact) of its golden CoP.
COP_FRACTION = sp.Rational(1, 3)


def _foot_cop(
    pts: list[sp.Matrix], members: list[int], depth: sp.Rational
) -> sp.Matrix:
    """Realisable CoP of a foot, on a sensing plane ``depth`` below its contacts."""
    cop = pts[members[0]]
    if len(members) == 2:
        cop = cop + COP_FRACTION * (pts[members[1]] - pts[members[0]])
    return cop - sp.Matrix([0, 0, depth])


def _vertical_rows(
    pts: list[sp.Matrix], members: list[int], cop: sp.Matrix
) -> sp.Matrix:
    """Load row and the two CoP moment rows of one foot (ground normal is z)."""
    n = len(pts)
    rows = sp.zeros(3, 3 * n)
    for k in members:
        r = pts[k] - cop
        rows[0, 3 * k + 2] = 1
        for i in (0, 1):
            rows[1 + i, 3 * k + 2] = r[i]
            rows[1 + i, 3 * k + i] = -r[2]
    return rows


def vertical_selector(
    points_per_foot: int, depth: sp.Rational
) -> tuple[sp.Matrix, list]:
    """Per-foot vertical-cell selector and the CoP used for each foot."""
    pts = sc.FEET if points_per_foot == 1 else sc.FEET4
    feet = [list(range(i * points_per_foot, (i + 1) * points_per_foot)) for i in (0, 1)]
    cops = [_foot_cop(pts, f, depth) for f in feet]
    blocks = [_vertical_rows(pts, f, c) for f, c in zip(feet, cops, strict=True)]
    return sp.Matrix.vstack(*blocks), cops


def _shear_rank(basis: sp.Matrix, lead: int, n_pts: int) -> int:
    """Rank of the tangential (x, y) contact-load rows of a null basis."""
    if basis.cols == 0:
        return 0
    rows = [lead + 3 * k + c for k in range(n_pts) for c in (0, 1)]
    return basis.extract(rows, list(range(basis.cols))).rank()


def _null_basis(a_mat: sp.Matrix) -> sp.Matrix:
    vectors = a_mat.nullspace()
    return sp.Matrix.hstack(*vectors) if vectors else sp.zeros(a_mat.cols, 0)


def _entry(e_sel: sp.Matrix, null: sp.Matrix, lead: int) -> dict[str, Any]:
    """Exact rank accounting of one observation selector."""
    e_x = sp.Matrix.hstack(sp.zeros(e_sel.rows, lead), e_sel)
    reduction = (e_x * null).rank()
    return {
        "rows": e_sel.rows,
        "rank_selector": e_sel.rank(),
        "reduction": reduction,
        "nullity_observed": null.cols - reduction,
        "selector": cm.exact_mat(e_sel),
    }


def variant(points_per_foot: int) -> dict[str, Any]:
    st = sc.build(True, True, points_per_foot)
    j_c, j_g, n, u = st["J_c"], st["J_g"], st["n"], st["unactuated"]
    a_cols = [i for i in range(n) if i not in u]
    s_t = sp.zeros(n, len(a_cols))
    for r, i in enumerate(a_cols):
        s_t[i, r] = 1
    a_mat = sp.Matrix.hstack(s_t, j_c.T, j_g.T)
    n_a, m, n_g = len(a_cols), j_c.rows, j_g.rows
    null = _null_basis(a_mat)
    nullity = null.cols
    assert nullity == n_a + m + n_g - a_mat.rank()
    observations: dict[str, Any] = {}
    for name, e_sel in selectors(points_per_foot).items():
        observations[name] = _entry(e_sel, null, n_a + m)
    n_pts = j_g.rows // 3
    for name, depth in (
        ("vertical_per_foot", sp.Integer(0)),
        ("vertical_per_foot_sensing_depth", SENSING_DEPTH),
    ):
        e_sel, cops = vertical_selector(points_per_foot, depth)
        entry = _entry(e_sel, null, n_a + m)
        e_x = sp.Matrix.hstack(sp.zeros(e_sel.rows, n_a + m), e_sel)
        kept = _null_basis(e_x * null)
        entry.update(
            {
                "sensing_depth": cm.exact_mat(sp.Matrix([[depth]]))[0][0],
                "cop_per_foot": [cm.exact_mat(c.T)[0] for c in cops],
                "shear_rank_free": _shear_rank(null, n_a + m, n_pts),
                "shear_rank_observed": _shear_rank(null * kept, n_a + m, n_pts),
            }
        )
        observations[name] = entry
    return {
        "n_actuated": n_a,
        "m": m,
        "n_g": n_g,
        "dim_x": n_a + m + n_g,
        "nullity_free": nullity,
        "multiplier_gauge": m - j_c.rank(),
        "foot_groups": [
            list(range(i * points_per_foot, (i + 1) * points_per_foot)) for i in (0, 1)
        ],
        "observations": observations,
    }


def generate() -> dict[str, Any]:
    return cm.dump(
        "lower_body_observation",
        {
            "description": (
                "Exact nullity reduction rank(E_x N) of a ground-reaction "
                "observation E f_g = g on the spatial two-arm chain with a "
                "floating root and rigid point feet (one or two points per "
                "foot).  Selectors are exact rationals over the stacked point "
                "forces (fx fy fz per point)."
            ),
            "geometry_note": "generic rational geometry; ground normal is world z",
            "variants": {
                "two_point_feet": variant(1),
                "four_point_feet": variant(2),
            },
        },
    )
