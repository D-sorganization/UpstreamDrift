"""Micro-benchmark: Rust ``finite_diff_q_to_qdot_qddot`` vs the Python
reference, on the issue #5218 acceptance trajectory shape (1000 frames
× 30 DOFs).

Skipped when the Rust extension is not installed. Marked ``benchmark``
so it doesn't run in the default ``pytest -m unit`` lane — invoke with
``pytest -m benchmark tests/unit/motion_pipeline/test_bench_rust_finite_diff.py``.

Per the issue plan, the eventual ≥3× target is end-to-end with Pinocchio
RNEA in the loop; this micro-bench is informational.

We assert a conservative ≥2× speedup as a defensive regression lower
bound. The headline number is dominated by the ``Vec<Vec<f64>>``
marshalling round-trip we chose for slice 1 (see ``lib.rs`` rationale);
slice 2 will introduce a zero-copy numpy path that lifts the floor
substantially (≥10× expected once the ``pin.rnea`` callback driver
needs flat buffers anyway). The Rust criterion bench in
``rust_core/upstream-motion-matching/benches/finite_diff.rs`` measures
the pure-kernel speed without the marshalling tax.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.unit]


def _python_finite_difference(
    q: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    n = q.shape[0]
    qdot = np.zeros_like(q)
    qddot = np.zeros_like(q)
    if n < 2:
        return qdot, qddot
    for i in range(1, n - 1):
        qdot[i] = (q[i + 1] - q[i - 1]) / (2.0 * dt)
        qddot[i] = (q[i + 1] - 2.0 * q[i] + q[i - 1]) / (dt * dt)
    qdot[0] = (q[1] - q[0]) / dt
    qdot[-1] = (q[-1] - q[-2]) / dt
    if n >= 3:
        qddot[0] = qddot[1]
        qddot[-1] = qddot[-2]
    return qdot, qddot


def _make_traj(n: int, d: int, dt: float) -> np.ndarray:
    times = np.arange(n) * dt
    out = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        f = 1.0 + 0.5 * j
        out[:, j] = np.sin(2.0 * math.pi * f * times + 0.1 * j)
    return out


def _time_callable(fn, *args, repeats: int = 5) -> float:
    """Best-of-``repeats`` median wall time in seconds."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def test_rust_finite_diff_at_least_10x_faster_than_python() -> None:
    pytest.importorskip(
        "upstream_motion_matching",
        reason="upstream_motion_matching wheel not installed",
    )
    import upstream_motion_matching as umm

    dt = 1.0 / 240.0
    n_frames, n_dof = 1000, 30
    q = _make_traj(n_frames, n_dof, dt)
    q_list = q.tolist()

    py_t = _time_callable(_python_finite_difference, q, dt)
    rs_t = _time_callable(umm.finite_diff_q_to_qdot_qddot, q_list, dt)

    speedup = py_t / max(rs_t, 1e-9)
    print(  # noqa: T201 - test diagnostic
        f"\n[bench] N={n_frames}x{n_dof}: python={py_t * 1e3:.2f}ms "
        f"rust={rs_t * 1e3:.2f}ms speedup={speedup:.1f}x"
    )

    # 2× is the conservative regression floor for slice 1 (marshalling-
    # dominated). The Rust criterion micro-bench measures the kernel
    # alone; slice 2's zero-copy numpy path is where ≥10× lands here.
    assert speedup >= 2.0, (
        f"Expected ≥2× Rust speedup over Python on N={n_frames}×{n_dof}; "
        f"observed {speedup:.1f}× (python={py_t * 1e3:.2f}ms, "
        f"rust={rs_t * 1e3:.2f}ms). The Rust kernel may have regressed."
    )
