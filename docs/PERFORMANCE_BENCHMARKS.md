# Performance Benchmarks

This document describes the performance benchmark / regression-detection
setup added in issue #3510. It lives next to the implementation in
`tests/benchmarks/`.

## Goals

- Catch large (>=5x) performance regressions in critical hot paths.
- Stay dependency-light so the suite runs in the default CI lane.
- Keep the bar low for adding new benchmarks: write a test, run it once
  to capture a baseline, commit the updated `baseline.json`.

This is a **regression detector**, not a microbenchmark harness. For
fine-grained statistical benchmarking use the existing pytest-benchmark
files (`tests/benchmarks/test_physics_benchmarks.py`,
`tests/benchmarks/test_dynamics_benchmarks.py`,
`tests/benchmarks/test_performance_baseline.py`).

## Layout

| Path                                             | Purpose                                                                  |
| ------------------------------------------------ | ------------------------------------------------------------------------ |
| `tests/benchmarks/test_regression_benchmarks.py` | CI-runnable regression tests (uses `time.perf_counter`).                 |
| `tests/benchmarks/regression_helpers.py`         | Helpers: `measure_median_seconds`, `assert_within_regression_threshold`. |
| `tests/benchmarks/baseline.json`                 | Checked-in median-per-call baselines (seconds).                          |
| `tests/benchmarks/test_physics_benchmarks.py`    | Existing pytest-benchmark suite for physics functions (unchanged).       |
| `tests/benchmarks/test_dynamics_benchmarks.py`   | Existing pytest-benchmark suite for spatial dynamics (unchanged).        |
| `tests/benchmarks/test_performance_baseline.py`  | Existing pytest-benchmark micro baselines (unchanged).                   |

All tests in this suite are marked with `pytest.mark.benchmark` (already
registered in `pyproject.toml`).

## Running locally

```bash
python3 -m pytest tests/benchmarks/test_regression_benchmarks.py \
    -m benchmark --timeout=120 -v
```

To run the full pytest-benchmark suite alongside the regression checks:

```bash
python3 -m pytest tests/benchmarks/ -m benchmark --timeout=120 -v
```

## How regression detection works

For each tracked operation:

1. The test runs the operation `N` times after a small warmup, using
   `time.perf_counter`.
2. The median per-call wall-clock time is computed (median is more
   robust to outliers from JIT/GC/scheduler noise than mean).
3. The result is compared against the value stored under that key in
   `tests/benchmarks/baseline.json`, multiplied by
   `DEFAULT_REGRESSION_MULTIPLIER` (currently `5.0`).
4. If the measurement exceeds that threshold, the test fails with a
   diagnostic showing measured vs baseline vs threshold.
5. **If the key is missing from `baseline.json`, the test passes**,
   so adding a new benchmark does not break CI on first run.

The 5x multiplier is intentionally generous. Hot paths benchmarked here
involve numpy / pydantic, both of which have non-trivial run-to-run
variance on shared CI runners. We prefer occasional under-detection of
small regressions over flaky CI noise. Tighten the multiplier in
`regression_helpers.py` once baselines stabilize across runners.

## Current baseline (captured 2026-04-30)

| Benchmark                            | Median per call |
| ------------------------------------ | --------------- |
| `drag_force_calculation`             | ~4 us           |
| `aerodynamics_engine_compute_forces` | ~130 us         |
| `ball_flight_force_step`             | ~27 us          |
| `simulation_request_model_validate`  | ~1.6 us         |

Baselines were captured on Python 3.11.15 on the development self-hosted
runner. Runner-to-runner variation is expected; the 5x threshold absorbs
it.

## Adding a new benchmark

1. Add a test in `tests/benchmarks/test_regression_benchmarks.py`:

   ```python
   def test_my_hot_path_regression() -> None:
       median = measure_median_seconds(my_func, arg1, arg2, iterations=500)
       assert_within_regression_threshold("my_hot_path", median)
   ```

2. Run it locally to record the median; copy the value (with a small
   safety margin) into `tests/benchmarks/baseline.json` under
   `measurements`.
3. Commit the test and the updated `baseline.json` together.

## Updating an existing baseline

Re-baseline only when an intentional change moves the measurement:

1. Run `python3 -m pytest tests/benchmarks/test_regression_benchmarks.py
-m benchmark -v` and capture the printed measured value (or the
   failure diagnostic).
2. Update the value in `tests/benchmarks/baseline.json`.
3. Reference the rationale in the commit message (which optimization,
   which regression, etc.).

## CI integration

The `tests` job in `.github/workflows/ci-standard.yml` runs the
regression suite on the Python 3.11 matrix entry as a separate step
(see `Performance Regression Benchmarks`). The `baseline.json` file is
uploaded as a workflow artifact (`performance-baseline`) so future runs
can compare deltas if we add automated tracking later.

The benchmark suite is excluded from the main test step via
`--ignore=tests/benchmarks` and `-m "not ... benchmark ..."` so it does
not double-run in the coverage lane.

## Follow-up TODOs

- Hook the artifact uploader into a delta-comparison job that posts a PR
  comment summarising regressions vs the previous run.
- Tighten `DEFAULT_REGRESSION_MULTIPLIER` from 5x once we have multi-week
  baselines that show typical noise envelope.
- Consider auto-baselining in nightly runs (write the JSON back via a
  PR-creating workflow) so baselines drift with intentional perf work.
