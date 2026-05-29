# ruff: noqa: T201
"""Throughput comparison: pure-Python vs Rust batched RL kernels.

Implements the UD#5216 acceptance benchmark: 1000 muscles × 1000 RL
steps, comparing the existing pure-Python pipeline against the
``upstream_muscle`` Rust kernel.

The Python baseline uses the same source-of-truth code in
``src/shared/python/biomechanics/{hill_muscle,activation_dynamics,multi_muscle}.py``
so the comparison is apples-to-apples. The Rust kernel goes through the
``rust_muscle`` facade.

Usage::

    python rust_core/upstream-muscle/benches/compare_python_batch.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.biomechanics.activation_dynamics import (  # noqa: E402
    ActivationDynamics,
)
from src.shared.python.biomechanics.hill_muscle import (  # noqa: E402
    HillMuscleModel,
    MuscleParameters,
    MuscleState,
)
from src.shared.python.biomechanics.rust_muscle import (  # noqa: E402
    is_rust_available,
    step_full,
)

N_MUSCLES = 1000
N_STEPS = 1000
DT = 0.001


def _make_inputs():
    rng = np.random.default_rng(42)
    u = rng.uniform(0.0, 1.0, size=(N_STEPS, N_MUSCLES))
    a0 = rng.uniform(0.0, 0.5, size=N_MUSCLES)
    l_ce = np.full(N_MUSCLES, 0.10)
    v_ce = rng.normal(0.0, 0.02, size=N_MUSCLES)
    params = np.column_stack(
        [
            rng.uniform(500.0, 1500.0, size=N_MUSCLES),  # F_max
            np.full(N_MUSCLES, 0.10),  # l_opt
            np.full(N_MUSCLES, 0.20),  # l_slack
            np.full(N_MUSCLES, 10.0),  # v_max
            np.zeros(N_MUSCLES),  # pennation_angle
            np.full(N_MUSCLES, 0.05),  # damping
            np.full(N_MUSCLES, 0.56),  # force_length_width
        ]
    )
    moment_arms = rng.normal(0.0, 0.03, size=(10, N_MUSCLES))
    return u, a0, l_ce, v_ce, params, moment_arms


def _run_python(u, a0, l_ce, v_ce, params, moment_arms) -> float:
    """Loop the existing pure-Python pipeline for N_STEPS steps."""
    # Build per-muscle objects.
    dyn = ActivationDynamics(tau_act=0.010, tau_deact=0.040, min_activation=0.001)
    models: list[HillMuscleModel] = []
    for i in range(N_MUSCLES):
        p = MuscleParameters(
            F_max=float(params[i, 0]),
            l_opt=float(params[i, 1]),
            l_slack=float(params[i, 2]),
            v_max=float(params[i, 3]),
            pennation_angle=float(params[i, 4]),
            damping=float(params[i, 5]),
        )
        models.append(HillMuscleModel(p, force_length_width=float(params[i, 6])))
    a = a0.copy()
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        u_row = u[step]
        for i in range(N_MUSCLES):
            a[i] = dyn.update(float(u_row[i]), float(a[i]), DT)
        forces = np.empty(N_MUSCLES)
        for i in range(N_MUSCLES):
            state = MuscleState(
                activation=float(a[i]),
                l_CE=float(l_ce[i]),
                v_CE=float(v_ce[i]),
                l_MT=0.0,
            )
            forces[i] = models[i].compute_force(state)
        _tau = moment_arms @ forces  # noqa: F841
        # Sink to keep it honest.
        if step == N_STEPS - 1:
            _ = float(_tau.sum())
    return time.perf_counter() - t0


def _run_rust(u, a0, l_ce, v_ce, params, moment_arms) -> float:
    a = a0.copy()
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        a, _tau = step_full(u[step], a, l_ce, v_ce, params, moment_arms, DT)
    return time.perf_counter() - t0


def main() -> int:
    print(f"Rust kernel available: {is_rust_available()}")
    print(
        f"Benchmark config: M={N_MUSCLES} muscles, N_STEPS={N_STEPS}, dt={DT * 1000:.1f} ms"
    )

    inputs = _make_inputs()

    print("\nRunning Rust path...")
    t_rust = _run_rust(*inputs)
    print(f"  Rust : {t_rust:.4f} s ({1e3 * t_rust / N_STEPS:.4f} ms/step)")

    print("\nRunning Python path (this is the slow one)...")
    t_py = _run_python(*inputs)
    print(f"  Python: {t_py:.4f} s ({1e3 * t_py / N_STEPS:.4f} ms/step)")

    speedup = t_py / t_rust if t_rust > 0 else float("inf")
    print(f"\nSpeedup: {speedup:.1f}x  (target: >= 20x)")
    if speedup >= 20.0:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
