# Native Window Parallel Performance Handoff

## Result and Scope

ControlTower completed exactly one sequential six-window pass and one two-worker
spawn pass on the unchanged run19 initial candidate. Sequential wall time was
20.18865 seconds; parallel wall time including startup, imports, input/output
serialization, IPC and pool shutdown was 10.99769 seconds: **1.83572 times faster,
45.53% less wall time**. Shared input/chart setup took 0.54910 seconds separately.

Every returned marker sample, primal state sample, full marker Jacobian and full
state Jacobian was exactly equal between passes. This includes endpoint states
and Jacobians. RHS counts also match. All 189 degree-six effort directions were
included; the last five windows additionally include the original 42 chart
directions. No optimization, physics change, acceptance change, or fit reset was
performed. The parent agent owns any production integration.

| Absolute Window (s) | Sequential Wall (s) | Worker Wall (s) | RHS Evaluations |
| ------------------- | ------------------: | --------------: | --------------: |
| 0–0.2               |              4.6002 |          4.8315 |            9869 |
| 0.2–0.4             |              4.6767 |          5.0095 |            9833 |
| 0.4–0.6             |              4.6793 |          4.6533 |            9833 |
| 0.6–0.7             |              2.3684 |          2.3549 |            4913 |
| 0.7–0.8             |              2.6253 |          2.5872 |            5429 |
| 0.8–0.85            |              1.2373 |          1.2061 |            2471 |

Workers had per-process peak RSS of 222.22 and 222.09 MiB; sequential peak RSS
was 221.97 MiB. These are Linux high-water marks, not simultaneous aggregate
memory. The prototype retains both result sets for comparison and serialization;
its memory footprint is not a production cache specification. Host: WSL2 Linux,
16 logical CPUs, Python 3.12.3, OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1.

## Reproduction and Evidence

`window_benchmark.py` calls unchanged `replay_marker_sensitivities`, shared
`sampled_shooting_windows` and shared `retract_node`. It extracts only the exact
chart setup statements from the hash-verified original run19b driver; it never
executes that driver's CLI, replay preparation or optimizer. Snapshot1 physical
nodes are the saved continuous initial nodes; zero-chart retraction is checked
against them within 1e-10 before both passes receive identical fixed arrays.
This extraction is a study adapter, not a proposed production API.

Source candidate canonical SHA256:
`c89597f6ccc00eadcbcab3008b83fdc7eb30b25efa442aab5ecf69273b67d049`.
Native model SHA256:
`b817fea76407f71b29e42aeb188890df53d2d86874c7e5875d14d353f4a1a248`.
Bernstein basis duration remains 0.8 seconds; windows end at 0.85 seconds.
Report records model, capture, candidate, config, snapshot, driver, runner and
sensitivity-source identities. No marker resampling or time remapping occurs.

Use a **new output path**, preserving this completed study:

```powershell
scp docs/development/native_parallel_performance/window_benchmark.py controltower:C:/Users/diete/window_benchmark_10021.py
ssh controltower wsl -e env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=/home/dieterolson/native-ms-pilot-9967-19 /home/dieterolson/simscape-pinocchio-9967/.venv/bin/python /mnt/c/Users/diete/window_benchmark_10021.py --run /mnt/c/Users/diete/native-ms-fit-9967-19 --model /mnt/c/Users/diete/native_geometry_spec_9967.json --target /mnt/c/Users/diete/driver_marker_payload_9967.json --driver /mnt/c/Users/diete/run_native_ms_recentered_9967_19b.py --output /mnt/c/Users/diete/NEW-OUTPUT-DIRECTORY
```

The completed run is `C:/Users/diete/native-window-performance-10021-01` on
ControlTower, copied intact to
`C:/Users/diete/Repositories/simscape-tour-checkpoints/native-window-performance-10021-01`
on this machine. Its 136,841,161-byte `arrays.npz` holds both complete outputs
and all prepared clock/state/tangent inputs; SHA256 is
`07f594daf3d7059167785c6c0cbb6edc5f048e5067e0109fdc924048db98358d`.
The large NPZ stays in those two checkpoint locations, outside Git.
`report.json` is the readable receipt. `raw-study-inputs.zip` preserves original
report bytes, runner/test, snapshot/config/candidate, model, target and original
driver without adding another copy of the large arrays to Git.

TDD: comparator tests first failed with the missing module, then two tests
passed. They reject changed elements in every returned array family, shape/key
differences and nonfinite data. Ruff passes. Test command:

```powershell
python3 -m pytest docs/development/native_parallel_performance/test_window_benchmark.py --noconftest -q --tb=short
```

## Recommended Next Step

This supports implementing a bounded, optional batched window evaluator with
two persistent isolated workers, deterministic window ordering, immutable input
identities and explicit failure propagation. Keep retraction, residual/defect
assembly and optimizer coordination in the parent process; workers should only
evaluate independent windows. Reuse the shared solver, cache and sensitivity
contracts; do not create a second fitter. Compare assembled residuals and full
Jacobians against sequential execution before a bounded solver trial. Validate
failure cleanup and preserve the sequential fallback.

One sequential-then-parallel sample does not quantify timing variability,
cache-order effects, contention, or whole-solver speedup. Dense solver work,
node retraction, acceptance replay and bookkeeping remain serial. Do not infer
a 25-minute solve becomes 13.6 minutes without measuring the actual fraction
spent in independent windows. Pool lifetime, cache-hit behavior, cancellation,
worker errors and reproducibility across repeated evaluations remain unqualified.
No additional benchmark sweep or optimization is authorized by this study.
