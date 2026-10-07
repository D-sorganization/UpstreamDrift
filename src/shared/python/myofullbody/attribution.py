"""Attribute the reserve actuators of a MyoFullBody run to their causes (#11689).

Four measurements, each independent of the pose mapping, answer *why* a group
needs reserve:

* :func:`waterfall` re-solves every frame with passive force removed, then with
  three times the active capacity as well, then with unlimited capacity.  The
  drops are the share due to passive muscle force (poses beyond the validated
  range), force capacity and what is left (muscle geometry: no muscle can
  generate that component of the effort at that pose).
* :func:`kinematic_floor` lets *any* generalised force act on every MyoFullBody
  coordinate.  What remains is effort at spec coordinates that no MyoFullBody
  motion corresponds to (``Phi`` rows), i.e. a structural model difference.
* :func:`capacity_audit` compares the peak effort of every coordinate with the
  largest torque the muscles could produce on it alone.

These are diagnostics on the receipt, never inputs to qualification.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Any

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.myofullbody import redundancy

Array = np.ndarray
# (name, active-capacity scale, passive scale, reserve weight or None for default)
Variant = tuple[str, float, float, float | None]
WATERFALL: tuple[Variant, ...] = (
    ("baseline", 1.0, 1.0, None),
    ("passive_off", 1.0, 0.0, None),
    ("capacity_x3_passive_off", 3.0, 0.0, None),
    ("capacity_unlimited", 1.0e3, 0.0, None),
)


_SHARED: dict[str, Any] = {}


def _scaled(values: Array, factor: float, n_scaled: int | None) -> Array:
    """``values`` times ``factor`` for the first ``n_scaled`` entries only."""
    if n_scaled is None:
        return factor * values
    return np.concatenate([factor * values[:n_scaled], values[n_scaled:]])


def _solve_variant(spec: Variant) -> tuple[str, Array, bool]:
    """Solve every shared frame for one variant (see :data:`Variant`)."""
    name, scale, passive_scale, own_weight = spec
    bases, tau = _SHARED["bases"], _SHARED["tau"]
    weight = _SHARED["weight"] if own_weight is None else own_weight
    n_scaled = _SHARED["n_scaled"]
    out, ok = [], True
    for b, t in zip(bases, tau, strict=True):
        sol = redundancy.solve_frame(
            _scaled(b.active, scale, n_scaled),
            _scaled(b.passive, scale * passive_scale, n_scaled),
            b.moment,
            t,
            reserve_weight=weight,
        )
        out.append(sol.reserve)
        ok = ok and sol.success
    return name, np.array(out), ok


def solve_variants(
    bases: Sequence[redundancy.FrameBasis],
    tau: Array,
    variants: Sequence[Variant],
    weight: float,
    workers: int = 1,
    n_scaled: int | None = None,
) -> list[tuple[str, Array, bool]]:
    """Reserves of every frame for each variant of :data:`Variant`.

    ``n_scaled`` limits the capacity and passive scaling to the first entries of
    each basis (the muscles, ahead of any appended torque actuators).
    ``workers > 1`` runs the variants in forked processes that share the bases
    without copying them.

    Raises:
        ValueError: if ``tau`` does not have one row per basis.
    """
    require(tau.shape[0] == len(bases), "tau needs one row per frame basis")
    _SHARED.update(bases=bases, tau=tau, weight=weight, n_scaled=n_scaled)
    try:
        if workers > 1:
            context = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                return list(pool.map(_solve_variant, variants))
        return [_solve_variant(v) for v in variants]
    finally:
        _SHARED.clear()


def reserve_ratios(
    bases: Sequence[redundancy.FrameBasis],
    tau: Array,
    groups: dict[str, list[int]],
    columns: list[int],
    variants: Sequence[Variant],
    weight: float,
    workers: int = 1,
    n_scaled: int | None = None,
) -> list[dict[str, Any]]:
    """Per variant: reserve RMS over effort RMS of every group and convergence."""
    solved = solve_variants(bases, tau, variants, weight, workers, n_scaled)
    out = []
    for (name, scale, passive, own), (_, res, ok) in zip(variants, solved, strict=True):
        metrics = redundancy.group_reserve_metrics(res, tau, groups, columns)
        out.append(
            {
                "name": name,
                "force_scale": scale,
                "passive_scale": passive,
                "reserve_weight": weight if own is None else own,
                "converged": ok,
                "reserve_over_effort_rms": {
                    g: round(m["reserve_over_effort_rms"], 4)
                    for g, m in metrics.items()
                },
            }
        )
    return out


def waterfall(
    bases: Sequence[redundancy.FrameBasis],
    tau: Array,
    groups: dict[str, list[int]],
    columns: list[int],
    weight: float,
    workers: int = 1,
    n_scaled: int | None = None,
) -> dict[str, dict[str, float]]:
    """Reserve RMS over effort RMS per group for each :data:`WATERFALL` variant."""
    rows = reserve_ratios(
        bases, tau, groups, columns, WATERFALL, weight, workers, n_scaled
    )
    out: dict[str, dict[str, float]] = {g: {} for g in groups}
    for row in rows:
        for g in groups:
            out[g][row["name"]] = row["reserve_over_effort_rms"].get(g, 0.0)
    return out


def kinematic_floor(
    bases: Sequence[redundancy.FrameBasis],
    tau: Array,
    groups: dict[str, list[int]],
    columns: list[int],
) -> dict[str, float]:
    """Reserve over effort RMS if every MyoFullBody coordinate could take any torque.

    Per frame the effort ``tau`` is fitted by ``Phi_c^T f`` over all MyoFullBody
    generalised forces ``f``; the residual cannot be produced by any muscle.
    """
    require(tau.shape[0] == len(bases), "tau needs one row per frame basis")
    residual = np.empty_like(tau)
    for i, (b, t) in enumerate(zip(bases, tau, strict=True)):
        p = b.phi[:, columns]
        f, *_ = np.linalg.lstsq(p.T, t, rcond=None)
        residual[i] = t - p.T @ f
    metrics = redundancy.group_reserve_metrics(residual, tau, groups, columns)
    return {g: metrics.get(g, {}).get("reserve_over_effort_rms", 0.0) for g in groups}


def capacity_audit(
    bases: Sequence[redundancy.FrameBasis], tau: Array, names: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Per coordinate: peak effort against the best single-coordinate capacity.

    The capacity at a frame is the active force of every muscle whose moment arm
    has the sign of the demand, summed with its moment arm.  It ignores that the
    same muscles load other coordinates, so it is an *upper* bound.
    """
    require(tau.shape == (len(bases), len(names)), "tau must be (frames, names)")
    out: dict[str, dict[str, float]] = {}
    for c, name in enumerate(names):
        demand = np.abs(tau[:, c])
        cap = np.empty(len(bases))
        for i, b in enumerate(bases):
            sign = 1.0 if tau[i, c] >= 0.0 else -1.0
            cap[i] = float(np.sum(np.maximum(sign * b.moment[:, c], 0.0) * b.active))
        ratio = cap / np.maximum(demand, 1e-9)
        out[name] = {
            "peak_demand_nm": float(demand.max()),
            "capacity_at_peak_demand_nm": float(cap[int(np.argmax(demand))]),
            "median_capacity_over_demand": float(np.median(ratio)),
            "frames_demand_exceeds_capacity": float((demand > cap).mean()),
        }
    return out


PHASES = ("backswing", "downswing", "impact_window", "follow_through")


def phase_breakdown(
    steps: Array,
    keys: dict[str, int],
    reserve: Array,
    tau: Array,
    groups: dict[str, list[int]],
    columns: list[int],
    impact_half_width: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Share of effort and reserve energy (sum of squares) per swing phase.

    Phases: ``backswing`` (before the top), ``downswing`` (top to the start of the
    impact window), ``impact_window`` (within ``impact_half_width`` steps of
    impact) and ``follow_through`` (after the window).  ``ratio`` is the reserve
    RMS over the effort RMS inside the phase.

    Raises:
        ValueError: if ``impact_half_width`` is negative or shapes disagree.
    """
    require(impact_half_width >= 0, "impact_half_width must be non-negative")
    require(reserve.shape == tau.shape, "reserve and tau must have the same shape")
    top, impact = keys["top"], keys["impact"]
    lo, hi = impact - impact_half_width, impact + impact_half_width
    masks = {
        "backswing": steps < top,
        "downswing": (steps >= top) & (steps < lo),
        "impact_window": (steps >= lo) & (steps <= hi),
        "follow_through": steps > hi,
    }
    where = {c: i for i, c in enumerate(columns)}
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, cols in groups.items():
        idx = [where[c] for c in cols if c in where]
        if not idx:
            continue
        eff = np.sum(tau[:, idx] ** 2, axis=1)
        res = np.sum(reserve[:, idx] ** 2, axis=1)
        out[name] = {}
        for phase in PHASES:
            m = masks[phase]
            e, r = float(eff[m].sum()), float(res[m].sum())
            out[name][phase] = {
                "frames": int(m.sum()),
                "effort_share": e / float(eff.sum()) if eff.sum() > 0 else 0.0,
                "reserve_share": r / float(res.sum()) if res.sum() > 0 else 0.0,
                "ratio": float(np.sqrt(r / e)) if e > 0 else 0.0,
            }
    return out


def attribution_receipt(
    bases: Sequence[redundancy.FrameBasis],
    tau: Array,
    groups: dict[str, list[int]],
    columns: list[int],
    names: Sequence[str],
    weight: float,
    workers: int = 1,
    n_scaled: int | None = None,
) -> dict[str, Any]:
    """The receipt block combining the three diagnostics."""
    return {
        "waterfall_reserve_over_effort_rms": waterfall(
            bases, tau, groups, columns, weight, workers, n_scaled
        ),
        "kinematic_floor_reserve_over_effort_rms": kinematic_floor(
            bases, tau, groups, columns
        ),
        "capacity_audit": capacity_audit(bases, tau, names),
        "definition": (
            "waterfall: baseline, then passive force removed, then x3 and unlimited "
            "active capacity (what is left is muscle geometry); kinematic floor: "
            "effort at spec coordinates that no MyoFullBody generalised force can "
            "produce; capacity audit: peak effort versus the best single-coordinate "
            "active capacity.  Diagnostics only; none enters qualification."
        ),
    }
