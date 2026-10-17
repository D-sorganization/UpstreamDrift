"""Shared fixtures for the ground-contact and force-plate null-space tests (NSX-60, OF-11)."""

from __future__ import annotations

import dataclasses

import numpy as np

from src.shared.python.loop_nullspace.contracts import ClosedLoopSnapshot

__all__ = ["SPATIAL_VARIANT", "VARIANTS", "make_consistent"]

VARIANTS = {
    "two_point_feet": {"foot_points": 1},
    "four_point_feet": {"foot_points": 2},
}
SPATIAL_VARIANT = {
    "two_point_feet": "floating_root_two_point_feet",
    "four_point_feet": "floating_root_four_point_feet",
}


def make_consistent(snap: ClosedLoopSnapshot) -> ClosedLoopSnapshot:
    """Move the root acceleration so the contact balance has a solution.

    Two point feet are a rank-5 contact map, so a random ``a`` violates the one
    consistency condition (NSX-00 table).  Adding ``alpha M_uu^-1 l`` to the root
    acceleration, with ``l`` the left null vector, restores it and does not
    touch the closure (``J_c`` has zero root columns).
    """
    u = list(snap.topology.unactuated_indices)
    g_u = snap.J_g[:, u].T
    left, sv, _ = np.linalg.svd(g_u)
    keep = int(np.count_nonzero(sv > 1e-10 * sv[0]))
    if keep == g_u.shape[0]:
        return snap
    ell = left[:, keep]
    b_u = snap.generalized_load()[u]
    alpha = -float(ell @ b_u) / float(ell @ ell)
    a = np.array(snap.a)
    a[u] += alpha * np.linalg.solve(snap.M[np.ix_(u, u)], ell)
    return dataclasses.replace(snap, a=a)
