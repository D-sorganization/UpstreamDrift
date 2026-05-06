# Visualization — Option 4

Most of Option 4's visualization reuses the shared views in [shared/VISUALIZATION_SPEC.md](../shared/VISUALIZATION_SPEC.md) — when the adapter is driving `system_identification` or `dataset_generator`, the viz is whatever those consumers produce.

What is **specific** to Option 4 is operational: latency, throughput, and cache behaviour. These are the views that justify the high setup cost and tell us whether the adapter is healthy.

## Adapter-specific views

### 1. Latency histogram per call type

**Purpose.** Detect regressions in MATLAB-Engine bridge cost. Helps distinguish solver-bound runs from marshalling-bound runs.

**What it shows.** A four-panel histogram, one panel per call category, on a log-x axis (microseconds → seconds):

```
+-----------------------+-----------------------+
| engine_start (slow)   | load_from_path        |
| 10-30 s               | 1-3 s after first     |
+-----------------------+-----------------------+
| set_param/setVariable | simulate              |
| 5-20 ms               | 50-200 ms             |
+-----------------------+-----------------------+
| logsout_extract       | (overlay) cached path |
| 5-20 ms               | <1 ms                 |
+-----------------------+-----------------------+
```

**Data source.** `SimscapeAdapter._timing_log: list[TimingEvent]` — populated automatically when `adapter.timing_enabled = True` (default off; on during dev). Each `TimingEvent` has `(call_type: str, duration_s: float, timestamp_utc: str, was_cache_hit: bool)`.

**Where it lives.** `option4_python_bridge/visualization/latency_histogram.py`. Entry point:

```python
def plot_latency_histogram(
    timing_log: list[TimingEvent],
    out_path: Path | None = None,
) -> matplotlib.figure.Figure: ...
```

**Acceptance.** Passing a known-good timing log produces a 4-panel figure with each panel labelled and a vertical line at the assumption-doc target latency.

### 2. Throughput vs pool size

**Purpose.** Calibrate `SimscapeAdapterPool.pool_size` for the deployment host. Detect license-pool ceiling and matlab-engine startup overhead at the elbow.

**What it shows.** Line plot of total simulations per second on the y axis vs. pool size on the x axis (1, 2, 4, 8, 16). Annotated with:

- Theoretical maximum (`pool_size × per_call_throughput`).
- Measured throughput (data points).
- Vertical dashed line at the host's MATLAB license count (above which throughput plateaus).

```
sims/s ^
       |          ------------- theoretical (linear)
   80  |       ./.
       |     ./.
   60  |   ./.--- measured (sub-linear)
       | ./.   <-- elbow at license_count
   40  |/.
       +------------------------> pool_size
       1   2   4   8   16
```

**Data source.** Run `benchmark_pool_throughput.py` (in `option4_python_bridge/visualization/`) which iterates pool sizes, runs N=100 simulations at each, records elapsed time. Ships sample fixture data committed under `option4_python_bridge/visualization/fixtures/throughput_sample.json` so the plot can be reviewed without running benchmarks.

**Where it lives.** `option4_python_bridge/visualization/throughput.py`:

```python
def plot_throughput_vs_pool_size(
    measurements: list[ThroughputMeasurement],
    license_count: int | None = None,
    out_path: Path | None = None,
) -> matplotlib.figure.Figure: ...
```

### 3. Cache hit rate over a typical fit run

**Purpose.** Verify the cache is doing its job during a real `system_identification` run. A well-behaved fit should converge to a steady cache miss rate (each iteration explores new coefficients) but should also show a small persistent hit rate from line searches and gradient finite differences.

**What it shows.** Two-line plot over iteration number:

- Cumulative hit rate (`hits / (hits + misses)`) — starts at 0, asymptotes upward.
- Per-iteration hit rate (rolling mean over 50 calls) — noisy, useful for spotting "the optimizer rediscovered a region" events.

**Data source.** `SimscapeAdapter.cache_stats()` returns a `CacheStats(hits, misses, evictions, current_size)` snapshot. The fit loop polls this at each iteration end.

**Where it lives.** `option4_python_bridge/visualization/cache_hit_rate.py`:

```python
def plot_cache_hit_rate(
    snapshots: list[tuple[int, CacheStats]],   # (iteration, stats)
    out_path: Path | None = None,
) -> matplotlib.figure.Figure: ...
```

**Acceptance.** A 1000-iteration `system_identification` run on a fixture target produces a hit rate that monotonically increases (modulo rolling-mean noise) and ends > 5%.

## Reused views (shared)

The following views come from [shared/VISUALIZATION_SPEC.md](../shared/VISUALIZATION_SPEC.md) and Option 4 reuses them as-is:

- **Live trajectory overlay** — `q_meas` vs `q_sim` per joint over time. Driven by the adapter's `simulate_with_coefficients` output.
- **Position/orientation error timecourse** — `‖r_sim − r_meas‖` and geodesic angle `d_geo` per timestep.
- **Final summary card** — `final_rmse_m`, `final_total_work_J`, fit duration, target metadata.

No Option-4-specific code is needed for these; the adapter just produces the data and hands it to the shared visualizers.

## Operational dashboard (optional, recommended)

If a production deployment runs many fits per day, a small dashboard combining the three Option-4-specific views is helpful. Suggested layout:

```
+-------------------------------------------------------------+
| SimscapeAdapter dashboard                                   |
+----------------------+---------------+----------------------+
|  Latency histogram   |  Throughput   |  Cache hit rate      |
|  (last 1k calls)     |  vs pool size |  (running fit)       |
|                      |               |                      |
|  [4-panel hist]      |  [line plot]  |  [2-line plot]       |
+----------------------+---------------+----------------------+
|  Engine processes alive: 4 / 4                              |
|  License pool free:    2 / 6                                |
|  Total sims today:    14,237                                |
|  Cache size / max:    1024 / 1024 (full, evicting)          |
+-------------------------------------------------------------+
```

This is a single file: `option4_python_bridge/visualization/dashboard.py`. It is **not** required for issues #036–#040 acceptance; flag for a follow-up issue when the bridge is in real-world use.
