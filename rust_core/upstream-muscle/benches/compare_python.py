"""Python baseline microbenchmark for the Hill-curve scalar functions.

Mirrors `benches/hill.rs`: 10_000 calls to `f_l`/`f_v`/`f_t` per batch.
Run alongside `cargo bench -p upstream-muscle` to see the GIL-release /
native-math win vs the pure-Python source-of-truth.

Usage::

    cd <repo-root>
    python rust_core/upstream-muscle/benches/compare_python.py

If `upstream_muscle` is installed (via `maturin develop --features python`),
also reports its per-batch wall time for an apples-to-apples comparison.

Slice 1 of UD#5216.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.biomechanics.hill_muscle import (  # noqa: E402
    HillMuscleModel,
    MuscleParameters,
)

BATCH = 10_000
ITERS = 50


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")


def _bench(label: str, fn, xs: list[float]) -> float:
    # Warm-up.
    for x in xs[:100]:
        fn(x)

    best = float("inf")
    for _ in range(ITERS):
        t0 = time.perf_counter()
        s = 0.0
        for x in xs:
            s += fn(x)
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    _emit(f"{label:>32s}: {best * 1e6:9.1f} us / batch  (sum={s:.6e})")
    return best


def main() -> int:
    muscle = HillMuscleModel(
        MuscleParameters(F_max=1000.0, l_opt=0.15, l_slack=0.20, v_max=10.0)
    )

    xs_l = _linspace(0.5, 1.5, BATCH)
    xs_v = _linspace(-1.0, 1.0, BATCH)
    xs_t = _linspace(0.9, 1.4, BATCH)

    _emit(f"Batch size: {BATCH}, iterations: {ITERS} (best-of)")
    _emit("-- Python source (HillMuscleModel) --")
    py_l = _bench("python f_l", muscle.force_length_active, xs_l)
    py_v = _bench("python f_v", muscle.force_velocity, xs_v)
    py_t = _bench("python f_t", muscle.tendon_force, xs_t)

    try:
        import upstream_muscle as um  # type: ignore[import-not-found]
    except ImportError:
        _emit(
            "\n(install with `maturin develop --features python` from "
            "rust_core/upstream-muscle to also bench the Rust extension.)"
        )
        return 0

    _emit("-- Rust extension (upstream_muscle) --")
    rs_l = _bench("rust f_l (PyO3)", um.f_l, xs_l)
    rs_v = _bench("rust f_v (PyO3)", um.f_v, xs_v)
    rs_t = _bench("rust f_t (PyO3)", um.f_t, xs_t)

    _emit("\n-- Speedup (python / rust) --")
    _emit(f"  f_l: {py_l / rs_l:6.2f}x")
    _emit(f"  f_v: {py_v / rs_v:6.2f}x")
    _emit(f"  f_t: {py_t / rs_t:6.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
