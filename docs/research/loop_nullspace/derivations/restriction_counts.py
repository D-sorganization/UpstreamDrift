"""Reaction-restriction oracles (NSX-26, exact sympy).

On a fixed welded trajectory the closure rows of ``WELD6`` may be restricted to
the components a looser hand-club joint can transmit.  In the canonical grip
frame (rows ``fx fy fz tx ty tz``, first axis ``u`` the shaft axis):

* ``WELD6``       keeps rows ``0 1 2 3 4 5``;
* ``SHAFT_FREE5`` drops the twist moment about the shaft, row ``3`` (``tx``);
* ``BALL3``       keeps the three force rows ``0 1 2``.

Two things are pinned here.

1. *Dimension counts* of the spatial two-arm chain of :mod:`spatial_counts`
   (all four NSX-00/NSX-60 variants) for each kind, from the same exact
   Jacobian with only the listed rows kept.  Counts are structural: they do
   not depend on which physical axis the shaft is, only on which rows remain.

2. *A torque oracle* on the fixed-base 14-DOF chain (every coordinate
   actuated, so ``tau = b - J_r^T lambda_r`` is solvable for any ``b``).  With
   ``b = J_c^T lambda_0`` for a chosen rational ``lambda_0``, the minimum
   Euclidean-norm torque over the allowed reactions is

       tau* = (I - P_r) b,   P_r = A (A^T A)^-1 A^T,   A = J_r^T,

   exactly, with ``|tau*|^2 = b^T (I - P_r) b``.  ``tau* = 0`` exactly when
   ``lambda_0`` has no component in a dropped row (``J_c`` has full row rank,
   so the columns of ``J_c^T`` are independent).  Because the allowed reaction
   set shrinks along the nesting, ``|tau*|`` is non-decreasing.

The max-abs entry of ``tau*`` bounds the smallest symmetric capacity box that
admits a solution from above (``tau*`` is feasible), and ``|tau*|_2 / sqrt(n)``
bounds it from below (any feasible torque has ``|tau|_2 <= sqrt(n) |tau|_inf``).
Nothing here imports repository production code.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

import _common as cm
import spatial_counts as sc

R = sp.Rational

KINDS: dict[str, dict[str, Any]] = {
    "WELD6": {"rows": [0, 1, 2, 3, 4, 5], "components": "fx fy fz tx ty tz"},
    "SHAFT_FREE5": {"rows": [0, 1, 2, 4, 5], "components": "fx fy fz ty tz"},
    "BALL3": {"rows": [0, 1, 2], "components": "fx fy fz"},
}
VARIANTS = {
    "fixed_base_arms_only": (False, False, 1),
    "floating_root_no_feet": (True, False, 1),
    "floating_root_two_point_feet": (True, True, 1),
    "floating_root_four_point_feet": (True, True, 2),
}
LAMBDA_0 = {
    "twist": [R(1, 2), R(-1, 3), R(1, 5), R(7, 4), R(0), R(0)],
    "bend": [R(1, 2), R(-1, 3), R(1, 5), R(0), R(3, 2), R(-2, 3)],
    "generic": [R(1, 2), R(-1, 3), R(1, 5), R(7, 4), R(3, 2), R(-2, 3)],
}
KEPT_COUNT_KEYS = (
    "m",
    "n",
    "n_actuated",
    "n_unactuated",
    "n_g",
    "dim_x",
    "rank_A",
    "equality_nullity",
    "rank_J_c",
    "rank_J_cu_T",
    "rank_J_g",
    "multiplier_gauge",
    "attainable_torque_variation_fg_fixed",
    "attainable_torque_variation_fg_free",
    "gauge_total_fg_free",
    "nullity_fg_fixed",
)


def _counts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, (floating, feet, per_foot) in VARIANTS.items():
        out[name] = {}
        for kind, spec in KINDS.items():
            full = sc.counts(floating, feet, per_foot, spec["rows"])
            out[name][kind] = {k: full[k] for k in KEPT_COUNT_KEYS}
            if "balance_rank" in full:
                out[name][kind]["balance_rank"] = full["balance_rank"]
                out[name][kind]["consistency_conditions"] = full[
                    "consistency_conditions"
                ]
    return out


def _oracle_case(j_c: sp.Matrix, lam0: list[Any]) -> dict[str, Any]:
    b = j_c.T * sp.Matrix(lam0)
    n = j_c.cols
    per_kind: dict[str, Any] = {}
    for kind, spec in KINDS.items():
        a = sp.Matrix.hstack(*[j_c[r, :].T for r in spec["rows"]])
        proj = a * (a.T * a).inv() * a.T
        tau = (sp.eye(n) - proj) * b
        norm_sq = sp.nsimplify((tau.T * tau)[0, 0])
        inf = max(abs(x) for x in tau)
        per_kind[kind] = {
            "tau_star": cm.exact_mat(tau.T)[0],
            "min_tau_norm_sq": str(norm_sq),
            "min_tau_norm_sq_float": cm.to_float(norm_sq),
            "tau_star_inf": cm.to_float(inf),
            "capacity_lower_bound": cm.to_float(sp.sqrt(norm_sq) / sp.sqrt(n)),
        }
    return {
        "lambda_0": [str(x) for x in lam0],
        "b": cm.exact_mat(b.T)[0],
        "kinds": per_kind,
    }


def generate() -> dict[str, Any]:
    st = sc.build(False, False)
    j_c = st["J_c"]
    return cm.dump(
        "restriction_counts",
        {
            "description": (
                "Reaction-restriction oracles on the spatial two-arm chain: "
                "row selections, exact dimension counts per kind and variant, "
                "and the exact minimum-norm torque of the fixed-base chain."
            ),
            "kinds": KINDS,
            "nesting": ["WELD6", "SHAFT_FREE5", "BALL3"],
            "counts": _counts(),
            "oracle": {
                "description": (
                    "fixed_base_arms_only, J_c rows [fx fy fz tx ty tz], every "
                    "coordinate actuated, M = I, a = 0, h = J_c^T lambda_0"
                ),
                "J_c": cm.exact_mat(j_c),
                "cases": {k: _oracle_case(j_c, v) for k, v in LAMBDA_0.items()},
            },
        },
    )
