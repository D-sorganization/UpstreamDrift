"""Thin Drake adapter around the canonical engine-agnostic cost function.

Per cross-engine §2.3 the swing-matching cost lives in
``src/shared/python/motion_matching/cost.py`` and must not be duplicated
per-engine. This module is the **adapter**: it converts a Drake
:class:`~src.engines.physics_engines.drake.python.motion_matching.simulate.SimOut`
into the shared :class:`~src.shared.python.motion_matching.cost.SimOutput`
schema and (optionally) tacks on Drake-specific constraint-residual
penalties returned by the auto-diff fit driver.

By design this file stays **well under 100 LOC**. Anything fatter than a
schema mapping plus a residual-norm sum belongs in the shared module.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.cost import (
    CostBreakdown,
    CostOptions,
    SimOutput,
)
from src.shared.python.motion_matching.cost import (
    compute_cost as _shared_compute_cost,
)

from .simulate import SimOut

__all__ = [
    "compute_cost_drake",
    "drake_simout_to_shared",
]


def drake_simout_to_shared(sim_out: SimOut) -> SimOutput:
    """Map a Drake :class:`SimOut` to the shared :class:`SimOutput`.

    The shared cost reads the **butt** anchor; the Drake forward-sim
    samples the equivalent body as ``grip`` (per cross-engine §2.2 the
    canonical name is ``grip``, but the existing MATLAB cost was authored
    against ``butt`` — we map field-by-field rather than rename either
    side).
    """
    return SimOutput(
        butt=np.asarray(sim_out.grip, dtype=np.float64),
        clubhead=np.asarray(sim_out.clubhead, dtype=np.float64),
        club_quat=np.asarray(sim_out.club_quat, dtype=np.float64),
        time=np.asarray(sim_out.time, dtype=np.float64),
        tau=np.asarray(sim_out.tau, dtype=np.float64),
        omega=np.asarray(sim_out.qd, dtype=np.float64),
    )


def compute_cost_drake(
    theta: NDArray[np.float64],
    target: ClubTarget,
    sim_fn,
    opts: CostOptions | None = None,
    *,
    constraint_residuals: Sequence[NDArray[np.float64]] | None = None,
) -> tuple[float, CostBreakdown]:
    """Drake-flavoured wrapper around :func:`compute_cost`.

    ``sim_fn`` is a callable ``theta -> SimOut`` (typically a closure
    over :func:`simulate_with_coefficients`). We adapt it to the shared
    cost's ``theta -> SimOutput`` contract here.

    ``constraint_residuals`` is an optional list of residual arrays
    produced by ``MathematicalProgram`` constraints in the auto-diff
    driver; their squared L2 norms are added to the returned scalar
    (and to the regularizer breakdown bucket).
    """
    cost_opts = opts if opts is not None else CostOptions()

    def _shared_sim_fn(theta_inner: NDArray[np.float64]) -> SimOutput:
        return drake_simout_to_shared(sim_fn(theta_inner))

    j, breakdown = _shared_compute_cost(theta, target, _shared_sim_fn, cost_opts)
    if constraint_residuals:
        penalty = float(sum(np.sum(r * r) for r in constraint_residuals))
        j = j + penalty
        breakdown = CostBreakdown(
            position=breakdown.position,
            orientation=breakdown.orientation,
            impact_anchor=breakdown.impact_anchor,
            body_marker=breakdown.body_marker,
            regularizer=breakdown.regularizer + penalty,
            total=j,
        )
    return j, breakdown
