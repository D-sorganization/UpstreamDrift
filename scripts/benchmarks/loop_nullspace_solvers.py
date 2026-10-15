"""Benchmark the loop null-space convex backends (NSX-20, #12563).

Not run in CI.  Usage (needs the ``loop-nullspace`` extra)::

    python3 scripts/benchmarks/loop_nullspace_solvers.py --rows 120 --cols 160

Reports, for a sequence of perturbed objectives on one constraint set, the
simplex iterations and wall time of cold ``solve_lp`` calls versus a reused
``LPSession`` (basis reuse), plus one SOCP solve.  Timing is diagnostic only.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src" / "shared" / "python"))
sys.path.insert(0, str(_ROOT))

from loop_nullspace.solvers import (  # noqa: E402
    LPSession,
    SOCBlock,
    SolveStatus,
    solve_lp,
    solve_socp,
)


def _problem(rows: int, cols: int, seed: int):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(rows, cols))
    x0 = rng.uniform(0.0, 1.0, cols)
    b = a @ x0 + rng.uniform(0.0, 0.5, rows)
    return a, b, [(0.0, 2.0)] * cols, rng.normal(size=cols), rng


def bench_lp(rows: int, cols: int, steps: int, seed: int) -> dict[str, float]:
    """Cold one-shot solves versus a basis-reusing session."""
    a, b, bounds, c0, rng = _problem(rows, cols, seed)
    objectives = [c0 + 0.02 * rng.normal(size=cols) for _ in range(steps)]
    t0 = time.perf_counter()
    cold = [solve_lp(c, None, None, a, b, bounds) for c in objectives]
    t_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    with LPSession(None, None, a, b, bounds) as session:
        session.solve(c0)
        warm = [session.solve(c) for c in objectives]
    t_warm = time.perf_counter() - t0
    if any(r.status is not SolveStatus.OPTIMAL for r in cold + warm):
        raise RuntimeError("benchmark LP did not solve to optimality")
    return {
        "cold_iterations": float(sum(r.iterations for r in cold)),
        "warm_iterations": float(sum(r.iterations for r in warm)),
        "cold_seconds": t_cold,
        "warm_seconds": t_warm,
    }


def bench_socp(dim: int, seed: int) -> dict[str, float]:
    """min t s.t. ||M x - r||_2 <= t, 0 <= x <= 1 (variables are x then t)."""
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(dim, dim))
    r = rng.normal(size=dim)
    a_cone = -np.hstack([np.zeros((1, dim)), np.ones((1, 1))])
    a_cone = np.vstack([a_cone, -np.hstack([m, np.zeros((dim, 1))])])
    b_cone = np.concatenate([[0.0], -r])
    c = np.zeros(dim + 1)
    c[-1] = 1.0
    bounds = [(0.0, 1.0)] * dim + [(None, None)]
    t0 = time.perf_counter()
    res = solve_socp(
        c, None, None, None, None, [SOCBlock(a_cone, b_cone)], bounds=bounds
    )
    return {
        "status_optimal": float(res.status is SolveStatus.OPTIMAL),
        "iterations": float(res.iterations),
        "seconds": time.perf_counter() - t0,
    }


def main() -> int:
    """Run the benchmarks and print one line per metric."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--cols", type=int, default=160)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--socp-dim", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name, table in (
        ("lp", bench_lp(args.rows, args.cols, args.steps, args.seed)),
        ("socp", bench_socp(args.socp_dim, args.seed)),
    ):
        for key, value in table.items():
            print(f"{name}.{key} = {value:.6g}")  # noqa: T201 - CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
