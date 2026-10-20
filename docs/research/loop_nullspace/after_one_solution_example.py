"""After-one-solution workflow on generic fixtures (NSE-UD-4a, #13464).

Run from the repository root::

    PYTHONPATH=. python3 docs/research/loop_nullspace/after_one_solution_example.py

Companion to ``after_one_solution.md``.  Three small, self-contained settings:

* the generic planar two-arm fixture (``loop_nullspace.fixtures``), decomposed
  once into ``x = x_p + N z`` and given a *declared* box on the reduced
  coordinates ``z`` (an invented feasible set, not a golfer);
* the circle ``F(u) = |u|^2 - 1`` and the cross ``F(u) = u1 u2`` as the smallest
  smooth and rank-dropping nonlinear solution sets;
* a three-muscle segment fibre ``G = [40 40 0]`` N m with ``b = 80`` N m, whose
  true fibre is one-dimensional although ``dim ker G = 2``.

No spatial (pinocchio) fixture is used.  Nothing here is a result about any
golfer.  :func:`run_example` returns plain Python numbers and strings so that
``tests/unit/loop_nullspace/test_after_one_solution_example.py`` can check each
one against an independent statement of the property.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.decomposition import (
    Decomposition,
    decompose,
    dimension_accounting,
)
from src.shared.python.loop_nullspace.envelopes import (
    static_envelopes,
    x_functional_quantities,
)
from src.shared.python.loop_nullspace.evidence import (
    Array,
    InputProvenance,
    ModelProvenance,
)
from src.shared.python.loop_nullspace.feasible_rows import FeasibleRow
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.fixtures import (
    generic_planar_config,
    planar_golfer_snapshot,
)
from src.shared.python.loop_nullspace.functional_identifiability import (
    attribution_quantities,
    classify_functional,
    functional_record,
)
from src.shared.python.loop_nullspace.hit_and_run import TrajectoryDomain, hit_and_run
from src.shared.python.loop_nullspace.manifold_continuation import (
    ManifoldSingularityError,
    local_chart,
    predictor_corrector_step,
)
from src.shared.python.loop_nullspace.muscle_fiber import (
    CocontractionIndex,
    activation_fiber,
)
from src.shared.python.loop_nullspace.muscle_frame import MuscleFrame, TendonKind

SEED = 13464
HALF_WIDTH = 1.0  # declared box |z_i| <= 1 on every reduced coordinate
INVARIANCE_TOL = 1e-8  # declared tolerance of the numerical label
CHART_TOL = 1e-10
MODEL = ModelProvenance("planar_two_arm_golfer_numpy", "after-one-solution-example")
SOURCE = InputProvenance("synthetic", "generic_planar_config", "declared box on z")


# ----------------------------------------------------------- planar fibre
def _decomposition() -> Decomposition:
    snap = planar_golfer_snapshot(generic_planar_config(), actuation="full").snapshot
    return decompose(snap, weights=np.eye(snap.topology.n_actuated))


def _box(dec: Decomposition, half: float) -> FeasibleSet:
    """Declared box ``|z_i| <= half``; the decomposition is attached to the set."""
    rows = []
    for i in range(dec.nullity):
        e = np.zeros(dec.nullity)
        e[i] = 1.0
        rows.append(FeasibleRow(e, half, f"B:{i}:up", "H-BOX", "torque"))
        rows.append(FeasibleRow(-e, half, f"B:{i}:lo", "H-BOX", "torque"))
    return dataclasses.replace(
        FeasibleSet.from_rows(dec.nullity, rows), decomposition=dec
    )


def _balance_determined_selector(dec: Decomposition, seed: int) -> Array:
    """A selector in ``range(A^T)``: the part of a random vector orthogonal to ``N``."""
    w = np.random.default_rng(seed).standard_normal(dec.dim_x)
    coef = np.linalg.lstsq(dec.N, w, rcond=None)[0]
    return w - dec.N @ coef


def _fibre_section(dec: Decomposition) -> dict[str, Any]:
    snap = planar_golfer_snapshot(generic_planar_config(), actuation="full").snapshot
    counts = dimension_accounting(snap)
    return {
        "dim_x": int(dec.dim_x),
        "rank_A": int(dec.rank_A),
        "nullity": int(dec.nullity),
        "accounting_rank": int(counts.rank_A),
    }


def _identifiability_section(
    dec: Decomposition, box: FeasibleSet, draws: Array
) -> dict[str, Any]:
    radius = HALF_WIDTH * float(np.sqrt(dec.nullity))  # encloses the box about z = 0
    invariant_sel = _balance_determined_selector(dec, 0)
    family_sel = np.zeros(dec.dim_x)
    family_sel[dec.n_actuated] = 1.0  # a closure multiplier coordinate
    invariant = classify_functional(dec, invariant_sel, tol=INVARIANCE_TOL)
    family = classify_functional(
        dec,
        family_sel,
        tol=INVARIANCE_TOL,
        feasible_set=box,
        enclosing_radius=radius,
    )
    assert family.bound is not None
    x0 = dec.x(np.zeros(dec.nullity))
    deviations = [float(family_sel @ (dec.x(z) - x0)) for z in draws]
    records = {
        res.status.value: functional_record(
            res,
            [0.0],
            units="N",
            model=MODEL,
            source=SOURCE,
            hypotheses=("H-BOX",),
        ).allocation_dependence.name.lower()
        for res in (invariant, family)
    }
    return {
        "invariant_status": invariant.status.value,
        "invariant_residual": float(invariant.residual),
        "family_status": family.status.value,
        "family_residual": float(family.residual),
        "annihilation_norm": float(family.annihilation_norm),
        "enclosing_radius": radius,
        "deviation_bound": float(family.bound.deviation_bound),
        "width_bound": float(family.bound.width_bound),
        "max_sampled_deviation": float(np.max(np.abs(deviations))),
        "invariant_record": records["numerically_invariant"],
        "family_record": records["family_dependent"],
    }


def _envelope_section(
    dec: Decomposition, box: FeasibleSet, draws: Array
) -> dict[str, Any]:
    sel = np.zeros(dec.dim_x)
    sel[dec.n_actuated] = 1.0
    factory = x_functional_quantities(
        ["multiplier"], sel[None, :], units="N", scale=1.0
    )
    series = static_envelopes([box], factory)
    values = [float(sel @ dec.x(z)) for z in draws]
    return {
        "status": series.status_lower[0][0].value,
        "lower": float(series.lower[0, 0]),
        "upper": float(series.upper[0, 0]),
        "sampled_min": min(values),
        "sampled_max": max(values),
    }


def _attribution_section(dec: Decomposition, box: FeasibleSet) -> dict[str, Any]:
    owner = np.arange(dec.dim_x) % 2
    partition = {f"src{j}": np.diag((owner == j).astype(float)) for j in range(2)}
    task_map = np.stack([_balance_determined_selector(dec, 10 + r) for r in range(2)])
    selectors = {name: task_map @ e for name, e in partition.items()}
    factory = attribution_quantities(
        ("a0", "a1"), selectors, partition=partition, units="m/s^2", scale=1.0
    )
    series = static_envelopes([box], factory)
    total = classify_functional(dec, task_map[0], tol=INVARIANCE_TOL)
    sources = [
        classify_functional(dec, sel[0], tol=INVARIANCE_TOL).status.value
        for sel in selectors.values()
    ]
    error = float(np.max(np.abs(sum(partition.values()) - np.eye(dec.dim_x))))
    return {
        "total_status": total.status.value,
        "source_statuses": sources,
        "partition_sum_error": error,
        "n_quantities": len(series.names),
        "bounded": all(s.value == "bounded" for s in series.status_lower[0]),
    }


# ------------------------------------------------------ nonlinear chart
def _circle_f(u: Array) -> Array:
    return np.array([u @ u - 1.0])


def _circle_j(u: Array) -> Array:
    return 2.0 * u.reshape(1, -1)


def _cross_f(u: Array) -> Array:
    return np.array([u[0] * u[1]])


def _cross_j(u: Array) -> Array:
    return np.array([[u[1], u[0]]])


def _chart_section() -> dict[str, Any]:
    scale = np.ones(2)
    chart = local_chart(
        _circle_f, _circle_j, np.array([1.0, 0.0]), variable_scale=scale, tol=CHART_TOL
    )
    step = predictor_corrector_step(
        chart,
        np.array([0.0, 1.0]),
        0.3,
        _circle_f,
        _circle_j,
        tol=CHART_TOL,
        max_iter=20,
    )
    new = step.chart.u_base
    cross = local_chart(
        _cross_f, _cross_j, np.array([1.0, 0.0]), variable_scale=scale, tol=CHART_TOL
    )
    try:
        predictor_corrector_step(
            cross,
            np.array([-1.0, 0.0]),
            1.0,
            _cross_f,
            _cross_j,
            tol=CHART_TOL,
            max_iter=20,
        )
        raises = False
    except ManifoldSingularityError:
        raises = True
    return {
        "rank": int(chart.rank),
        "iterations": int(step.iterations),
        "residual_after_step": float(abs(_circle_f(new)[0])),
        "moved_distance": float(np.linalg.norm(new - chart.u_base)),
        "crossing_raises": raises,
    }


# ------------------------------------------------------------ muscle fibre
def _segment_frame() -> MuscleFrame:
    """``G = [40 40 0]``: the third muscle has a zero moment arm."""
    return MuscleFrame(
        names=("flexor", "brachialis", "bystander"),
        coordinates=("joint",),
        moment_arms=np.array([[0.04], [0.04], [0.0]]),
        activation_grid=np.array([0.0, 1.0]),
        force_grid=np.array([np.zeros(3), np.full(3, 1000.0)]),
        max_isometric_force=np.full(3, 1000.0),
        tendon=TendonKind.RIGID,
        engine="example",
        model_provenance=MODEL,
        input_provenance=SOURCE,
    )


def _muscle_section() -> dict[str, Any]:
    fibre = activation_fiber(_segment_frame(), np.array([80.0]))
    bystander = CocontractionIndex("bystander", np.array([0.0, 0.0, 1.0]))
    samples = fibre.sample_result(50, rng=np.random.default_rng(SEED))
    lo, hi = fibre.index_range(bystander)
    level = fibre.level_subfibre_dimension(bystander, 0.5 * (lo + hi))
    return {
        "nullity": int(fibre.nullity),
        "hull_dimension": int(fibre.hull_dimension),
        "samples_vary": bool(np.ptp(samples.samples[:, 2]) > 0.0),
        "is_singleton": bool(samples.is_singleton),
        "level_subfibre_dimension": int(level),
    }


# -------------------------------------------------------------------- run
def run_example() -> dict[str, Any]:
    """Run every section and return the reported numbers (deterministic).

    Postconditions: the result has the keys ``fibre``, ``identifiability``,
    ``envelope``, ``attribution``, ``chart`` and ``muscle``; hit-and-run draws are
    seeded with :data:`SEED`.
    """
    dec = _decomposition()
    box = _box(dec, HALF_WIDTH)
    domain = TrajectoryDomain([box], np.ones((1, 1)))
    chain = hit_and_run(domain, np.zeros(domain.dimension), n_draws=200, seed=SEED)
    return {
        "fibre": _fibre_section(dec),
        "identifiability": _identifiability_section(dec, box, chain.coefficients),
        "envelope": _envelope_section(dec, box, chain.coefficients),
        "attribution": _attribution_section(dec, box),
        "chart": _chart_section(),
        "muscle": _muscle_section(),
    }


def main() -> None:
    """Write the section results to stdout (a CLI entry point)."""
    for section, values in run_example().items():
        sys.stdout.write(f"[{section}]\n")
        for key, value in values.items():
            sys.stdout.write(f"  {key} = {value}\n")


if __name__ == "__main__":
    main()
