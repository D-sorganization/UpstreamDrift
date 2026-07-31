# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Benchmarking suite for performance regression testing.

These tests measure execution time of critical code paths and fail if
they exceed generous upper bounds.  The bounds are intentionally loose
to avoid flaky CI failures -- they catch order-of-magnitude regressions
rather than fine-grained performance changes.

Each benchmark uses *median* timing over multiple independent trials to
reduce sensitivity to transient CPU load, OS scheduling, and thermal
throttling.  Thresholds are set high enough to pass on slow CI machines
(observed up to ~44ms for batch ID on Windows) while still catching
genuine regressions.

Budget rule of thumb: limits are set ~3-10x the typical observed wall
time on a modern dev machine so CI runners (which can be 2-3x slower
under load) do not flake.

Run with:
    python3 -m pytest tests/test_benchmarks.py -v --tb=short
"""

from __future__ import annotations

import logging
import statistics
import time

import numpy as np

from movement_optimizer.models import BodyModel, make_squat_config
from movement_optimizer.trajectory import SolutionCache, TrajectoryOptimizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Number of independent timing trials for each benchmark.  The *median*
# trial is compared against the threshold so that a single slow outlier
# (e.g. OS context-switch) cannot cause a failure.
# ---------------------------------------------------------------------------
_BENCHMARK_TRIALS = 5


def _measure_ms(fn: object, iterations: int, trials: int = _BENCHMARK_TRIALS) -> float:
    """Return the *median* per-call time in milliseconds.

    Runs *fn* for *iterations* calls in each of *trials* independent
    timing windows, then returns ``statistics.median`` of the per-call
    times.  Using the median rather than a single measurement makes the
    benchmark resilient to transient system noise.
    """
    times: list[float] = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()  # type: ignore[operator]
        elapsed = time.perf_counter() - start
        times.append(elapsed / iterations * 1000)  # ms per call
    return statistics.median(times)


class TestInverseDynamicsBenchmark:
    """Benchmark the inverse dynamics hot path."""

    def test_single_inverse_dynamics_speed(self, default_body: BodyModel):
        """Single-pose inverse dynamics should complete in < 2ms (median)."""
        dyn, _, _, _ = make_squat_config(default_body, 60.0)
        q = np.array([0.1, -0.5, 0.3])
        qd = np.array([0.5, -0.3, 0.2])
        qdd = np.array([1.0, -1.0, 0.5])

        # Warmup -- multiple calls to stabilise JIT / CPU caches
        for _ in range(10):
            dyn.inverse_dynamics(q, qd, qdd)

        per_call_ms = _measure_ms(
            lambda: dyn.inverse_dynamics(q, qd, qdd), iterations=1000
        )
        assert (
            per_call_ms < 2.0
        ), f"Single ID call took {per_call_ms:.3f}ms median (limit: 2ms)"

    def test_batch_inverse_dynamics_speed(self, default_body: BodyModel):
        """Batch inverse dynamics (100 timesteps) should complete in < 50ms (median).

        The threshold is intentionally generous (observed ~5-44ms across
        platforms) to avoid flaky failures on slow CI runners while still
        catching order-of-magnitude regressions.
        """
        dyn, _, _, _ = make_squat_config(default_body, 60.0)
        n = 100
        rng = np.random.default_rng(42)
        q = rng.uniform(-1.5, 1.5, (n, 3))
        qd = rng.uniform(-5, 5, (n, 3))
        qdd = rng.uniform(-10, 10, (n, 3))

        # Warmup -- multiple calls to stabilise JIT / CPU caches
        for _ in range(5):
            dyn.inverse_dynamics_batch(q, qd, qdd)

        per_call_ms = _measure_ms(
            lambda: dyn.inverse_dynamics_batch(q, qd, qdd), iterations=100
        )
        assert (
            per_call_ms < 50.0
        ), f"Batch ID (N=100) took {per_call_ms:.3f}ms median (limit: 50ms)"


class TestMassMatrixBenchmark:
    def test_mass_matrix_speed(self, default_body: BodyModel):
        """Mass matrix computation should complete in < 1ms (median)."""
        dyn, _, _, _ = make_squat_config(default_body, 60.0)
        q = np.array([0.1, -0.5, 0.3])

        # Warmup
        for _ in range(10):
            dyn.mass_matrix(q)

        per_call_ms = _measure_ms(lambda: dyn.mass_matrix(q), iterations=1000)
        assert (
            per_call_ms < 1.0
        ), f"Mass matrix took {per_call_ms:.3f}ms median (limit: 1ms)"


class TestForwardKinematicsBenchmark:
    def test_forward_kinematics_speed(self, default_body: BodyModel):
        """Forward kinematics should complete in < 1ms (median)."""
        dyn, _, _, _ = make_squat_config(default_body, 60.0)
        q = np.array([0.1, -0.5, 0.3])

        # Warmup
        for _ in range(10):
            dyn.forward_kinematics(q)

        per_call_ms = _measure_ms(lambda: dyn.forward_kinematics(q), iterations=1000)
        assert per_call_ms < 1.0, f"FK took {per_call_ms:.3f}ms median (limit: 1ms)"


class TestBodyModelBenchmark:
    def test_body_model_construction_speed(self):
        """BodyModel construction should complete in < 2ms (median)."""
        # Warmup
        for _ in range(10):
            BodyModel(75.0, 1.75)

        per_call_ms = _measure_ms(lambda: BodyModel(75.0, 1.75), iterations=1000)
        assert (
            per_call_ms < 2.0
        ), f"BodyModel init took {per_call_ms:.3f}ms median (limit: 2ms)"


# ===========================================================================
# Extended benchmarks (issue #409): add coverage for the optimizer hot path,
# end-to-end optimisation, the solution cache, and grid-size scaling so that
# accidental performance regressions surface in CI.
#
# Wall-clock budgets are intentionally generous (about 3-10x the observed
# typical time on a modern dev machine, ~50% headroom for slow CI runners).
# All RNGs are seeded for determinism.
# ===========================================================================


def _best_of(fn, trials: int = 3) -> float:
    """Return the *minimum* wall-clock time (seconds) over *trials* runs.

    The minimum is more stable than the mean for end-to-end SLSQP timing
    because it filters transient CPU contention without inflating the
    estimate.
    """
    times: list[float] = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def _make_squat_optimizer(
    body: BodyModel,
    *,
    n_waypoints: int = 6,
    n_eval: int = 20,
    n_starts: int = 1,
) -> TrajectoryOptimizer:
    """Construct a small squat optimiser used by the perf benchmarks."""
    dyn, qs, qe, qb = make_squat_config(body, 60.0)
    return TrajectoryOptimizer(
        body,
        dyn,
        "squat",
        60.0,
        qs,
        qe,
        qb,
        duration=2.0,
        n_waypoints=n_waypoints,
        n_eval=n_eval,
        n_starts=n_starts,
    )


class TestBatchDynamicsScaling:
    """Lagrangian batch inverse dynamics should be near-linear in N."""

    def test_batch_id_typical_grid_under_budget(self, default_body: BodyModel):
        """Batch ID at a typical optimiser grid (N=60) should be < 25ms median.

        Observed ~0.1ms on a modern dev machine; budget is ~250x to absorb
        slow CI runners and the NumPy fallback path when the Rust extension
        is unavailable.
        """
        dyn, _, _, _ = make_squat_config(default_body, 60.0)
        n = 60
        rng = np.random.default_rng(42)
        q = rng.uniform(-1.5, 1.5, (n, 3))
        qd = rng.uniform(-2.0, 2.0, (n, 3))
        qdd = rng.uniform(-5.0, 5.0, (n, 3))

        for _ in range(5):
            dyn.inverse_dynamics_batch(q, qd, qdd)

        per_call_ms = _measure_ms(
            lambda: dyn.inverse_dynamics_batch(q, qd, qdd), iterations=200
        )
        logger.info("batch ID (N=%d) median %.4f ms", n, per_call_ms)
        assert (
            per_call_ms < 25.0
        ), f"Batch ID (N={n}) took {per_call_ms:.3f}ms median (limit: 25ms)"

    def test_batch_id_scales_subquadratic(self, default_body: BodyModel):
        """Doubling N should not multiply batch-ID time by more than 4x.

        The vectorised implementation is O(N), so ~2x is expected; we allow
        up to 4x to absorb cache effects, BLAS thresholds, and CI noise.
        """
        dyn, _, _, _ = make_squat_config(default_body, 60.0)
        rng = np.random.default_rng(42)

        def time_batch(n: int) -> float:
            q = rng.uniform(-1.5, 1.5, (n, 3))
            qd = rng.uniform(-2.0, 2.0, (n, 3))
            qdd = rng.uniform(-5.0, 5.0, (n, 3))
            for _ in range(5):
                dyn.inverse_dynamics_batch(q, qd, qdd)
            return _measure_ms(
                lambda: dyn.inverse_dynamics_batch(q, qd, qdd), iterations=200
            )

        t_small = time_batch(50)
        t_large = time_batch(200)
        logger.info("batch ID scaling: N=50 %.4fms, N=200 %.4fms", t_small, t_large)
        # N grows 4x; the cost should grow no more than 4x (linear) in the
        # vectorised path.  Use a generous 8x bound for CI noise and to
        # tolerate fixed-overhead-dominated regimes.
        assert t_large < max(0.5, 8.0 * t_small), (
            f"Batch ID scaling regressed: N=50 took {t_small:.4f}ms, "
            f"N=200 took {t_large:.4f}ms (limit: 8x)"
        )


class TestOptimizerCostHotPath:
    """The optimiser cost function is called hundreds of times per solve."""

    def test_compute_cost_under_budget(self, default_body: BodyModel):
        """A single _compute_cost call at n_eval=20 should be < 5ms median.

        Observed ~0.3ms; the budget is ~15x to absorb CI variance.  This
        is the dominant inner-loop cost driver during SLSQP iterations.
        """
        opt = _make_squat_optimizer(default_body, n_eval=20)
        x0 = opt._initial_guess().flatten()

        for _ in range(5):
            opt._compute_cost(x0)

        per_call_ms = _measure_ms(lambda: opt._compute_cost(x0), iterations=200)
        logger.info("_compute_cost (n_eval=20) median %.4f ms", per_call_ms)
        assert (
            per_call_ms < 5.0
        ), f"_compute_cost took {per_call_ms:.3f}ms median (limit: 5ms)"


class TestEndToEndOptimizer:
    """End-to-end optimiser timing for a small problem."""

    def test_small_problem_under_10s(self, default_body: BodyModel):
        """A small squat optimisation should complete in < 10 seconds.

        Observed ~0.05-0.1s on a modern dev machine; budget is intentionally
        very generous (>100x) so this benchmark only triggers on
        catastrophic regressions (e.g. accidental algorithmic change,
        deadlock, or fall back to a serial code path).
        """
        opt = _make_squat_optimizer(default_body, n_eval=20, n_starts=2)
        elapsed = _best_of(lambda: opt.optimize(), trials=2)
        logger.info("end-to-end optimize (n_eval=20, n_starts=2) %.3f s", elapsed)
        assert (
            elapsed < 10.0
        ), f"Optimizer took {elapsed:.2f}s for a small problem (limit: 10s)"


class TestOptimizerScaling:
    """Optimiser wall time should scale sub-quadratically with grid size."""

    def test_grid_doubling_subquadratic(self, default_body: BodyModel):
        """n_eval=20 should be no more than ~6x slower than n_eval=10.

        The cost function is O(N) and SLSQP iteration count is roughly
        constant, so we expect ~2x scaling.  In practice, the relationship
        is noisy because SLSQP convergence depends on the discretisation
        (sometimes a coarser grid takes more iterations).  We use the
        minimum of multiple trials and require the larger grid to be no
        more than 6x slower -- well below quadratic (which would be 4x for
        a 2x size increase but with noise can drift higher).
        """

        def run_at(n_eval: int) -> float:
            # Build a fresh optimiser per trial so we don't reuse cached state.
            return _best_of(
                lambda n=n_eval: _make_squat_optimizer(
                    default_body, n_eval=n, n_starts=1
                ).optimize(),
                trials=3,
            )

        t_small = run_at(10)
        t_large = run_at(20)
        logger.info(
            "optimizer scaling: n_eval=10 %.3fs, n_eval=20 %.3fs", t_small, t_large
        )
        # Floor plus absolute cap prevent division blow-up when both runs are
        # very fast (sub-second) and scheduler noise dominates the ratio.
        baseline = max(t_small, 0.05)
        limit = max(6.0 * baseline, 1.0)
        assert t_large < limit, (
            f"Optimizer scaling regressed: n_eval=10 took {t_small:.3f}s, "
            f"n_eval=20 took {t_large:.3f}s (limit: {limit:.3f}s)"
        )


class TestSolutionCacheBenchmark:
    """Cache hits must be dramatically faster than running the optimiser."""

    def test_cache_hit_much_faster_than_miss(self, default_body: BodyModel):
        """A cache hit should be at least 100x faster than a full solve.

        Observed ratio is ~2000x.  Setting the lower bound at 100x leaves
        headroom for CI noise while still catching a regression where the
        cache no longer short-circuits the solver (e.g. accidental
        invalidation, hash collision, or call-through bug).
        """
        opt = _make_squat_optimizer(default_body, n_eval=20, n_starts=1)

        # Cache miss = full optimise.  Use the minimum of two runs to
        # absorb a single transient slow run on shared CI.
        t_miss = _best_of(lambda: opt.optimize(), trials=2)
        result = opt.optimize()

        cache = SolutionCache()
        seg_mults = {"lower_leg": 1.0, "upper_leg": 1.0, "torso": 1.0}
        cache.put("squat", 75.0, 1.75, seg_mults, 60.0, 2.0, 1.0, result)

        # Warm the lookup path before timing.
        for _ in range(10):
            cache.get("squat", 75.0, 1.75, seg_mults, 60.0, 2.0, 1.0)

        per_hit_ms = _measure_ms(
            lambda: cache.get("squat", 75.0, 1.75, seg_mults, 60.0, 2.0, 1.0),
            iterations=1000,
        )
        per_hit_s = per_hit_ms / 1000.0
        ratio = t_miss / per_hit_s if per_hit_s > 0 else float("inf")
        logger.info(
            "cache miss %.4fs vs hit %.6fs (ratio %.0fx)", t_miss, per_hit_s, ratio
        )

        # Absolute upper bound on a single hit lookup so we catch the case
        # where a hit becomes unexpectedly expensive (e.g. deep copy added
        # to the get path).
        assert per_hit_ms < 1.0, f"Cache hit took {per_hit_ms:.4f}ms (limit: 1ms)"
        # Lower bound on speedup: the optimiser must be far slower than a
        # lookup, otherwise the cache provides no real value.
        assert ratio > 100.0, (
            f"Cache speedup only {ratio:.1f}x (miss={t_miss:.3f}s, "
            f"hit={per_hit_s * 1000:.4f}ms; limit: 100x)"
        )
