# Native Window Parallel Performance Handoff

## Executor Implementation Checkpoint

The optional engine-neutral executor is now implemented in
`src/shared/python/motion_matching/native_window_executor.py`. The shared
multiple-shooting solver now exposes an optional `segmented_forward_batch`
boundary. It receives only ordered cache misses and returns results in that
order; parent-owned cache, residual/defect assembly, analytic Jacobians and the
sequential default remain unchanged. The worker-side transport now lives in
`native_sensitivity_batch.py`: private runner-created payloads reconstruct and
validate the model/candidate in every spawned worker, then return primal markers,
endpoint state and sensitivity arrays. A native driver has not yet bound this
seam to a persistent executor, so this is not a native solver qualification or a
fit speed claim. No runtime19/runtime20 or sensitivity source was changed, and
no optimizer was launched.

## Trusted Worker Fixed-Input Qualification

`qualify_native_worker.py` ran once in a fresh isolated ControlTower runtime
`/home/dieterolson/native-ms-pilot-9967-21`, copied from immutable runtime20
before adding only the new worker and executor modules. It used the archived
run19 initial candidate and input array receipt
`07f594daf3d7059167785c6c0cbb6edc5f048e5067e0109fdc924048db98358d`.
No optimizer, model, existing runtime, or prior output was modified.

Sequential execution took21.12844s; one two-worker batch including startup,
IPC and shutdown took11.14284s. Every one of six windows matched both the new
sequential worker result and archived sequential result exactly for markers,
full states, marker Jacobians and state Jacobians. The readable receipt is
`executor-worker-report.json`; the exact runner/source/report bytes are in
`raw-worker-qualification.zip`. This qualifies trusted worker transport and
full raw sensitivity output only. It does not yet qualify
`fit_multiple_shooting` residual/constraint assembly with this executor,
repeated solver cache behavior, or a whole-fit speedup.

Discovery found a per-call `ProcessPoolExecutor` in
`src/shared/python/sidekick/process_calculators/multi_param_analysis.py`, tied to
calculator/UI parameter handling. No reusable persistent motion-window executor
exists there. This new boundary uses Python's standard executor directly and
reuses the caller's existing result contract, without duplicating physics,
retraction, fitting, or residual assembly.

The API is `NativeWindowExecutor(evaluate, workers=0)` with optional `workers=2`.
The evaluator must be a module-level picklable callable accepting immutable
`bytes`; its result type is preserved. Requests are **trusted internal transport
only**. The shared module does not encode or decode payloads. Never expose the
qualification script's pickle adapter to untrusted inputs. `evaluate(requests)`
returns an ordered tuple, independent of completion order. The two-worker pool
persists across batches. Explicit sequential mode is the default; failures never
silently retry in another mode.

Use a context manager or `close()`. Any evaluation/submission failure cancels
queued tasks, waits for running tasks, closes the pool, and raises
`WindowEvaluationError` with `window_index` and the original exception as cause.
Use after failure/close is rejected. Shutdown is idempotent. There is no forced
termination of a hung physics evaluation; external supervision remains necessary
for a true hang. Own the executor in one coordinating thread. Set
OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 before construction and preserve
those values while workers start. Use the application's guarded main entry point.

### TDD and Native Qualification

Five executor tests first failed on the missing module, then passed. They cover
sequential/two-worker repeated ordered batches, explicit original-cause errors,
cleanup without surviving owned children, reuse rejection, immutable bytes and
thread-limit/configuration contracts. Together with the two earlier comparator
tests, seven pass. Ruff and direct mypy on the new shared module pass.

One final native sequential pass took 21.30037 seconds and one executor pass
took 11.18647 seconds, including startup/IPC/shutdown. All full marker/state
Jacobians and all primal marker/state arrays match both that sequential pass
and the historical study **exactly**. Concatenated states/markers with shared
boundary samples removed also match exactly. RHS counts match. This checks
ordering implications for assembly; it does **not** qualify the solver's actual
residual/defect assembly, cache integration, cancellation API, or whole-solve
speed. Native repeated-batch timing is not claimed; repeated lifecycle behavior
is covered by deterministic unit functions.

Raw completed output: ControlTower
`C:/Users/diete/native-window-executor-10021-01/report.json`, copied to local
`C:/Users/diete/Repositories/simscape-tour-checkpoints/native-window-executor-10021-01`.
Readable copy: `executor-report.json`. `raw-executor-qualification.zip` retains
exact report, executor, qualifier and tests. The earlier large input/output NPZ
and raw study archive remain unchanged at the locations below. No native
qualification process remains active after this checkpoint.

```powershell
python3 -m pytest tests/unit/motion_matching/test_native_window_executor.py docs/development/native_parallel_performance/test_window_benchmark.py --noconftest -q --tb=short
python3 -m ruff check src/shared/python/motion_matching/native_window_executor.py tests/unit/motion_matching/test_native_window_executor.py docs/development/native_parallel_performance/qualify_executor.py
python3 -m mypy src/shared/python/motion_matching/native_window_executor.py --follow-imports=silent
scp src/shared/python/motion_matching/native_window_executor.py controltower:C:/Users/diete/native_window_executor_10021.py
scp docs/development/native_parallel_performance/qualify_executor.py controltower:C:/Users/diete/qualify_executor_10021.py
# Requires the original archived window_benchmark_10021.py beside the qualifier.
# Always select a NEW output path; do not overwrite the completed receipt.
ssh controltower wsl -e env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=/home/dieterolson/native-ms-pilot-9967-19 /home/dieterolson/simscape-pinocchio-9967/.venv/bin/python /mnt/c/Users/diete/qualify_executor_10021.py --executor /mnt/c/Users/diete/native_window_executor_10021.py --study /mnt/c/Users/diete/native-window-performance-10021-01 --run /mnt/c/Users/diete/native-ms-fit-9967-19 --model /mnt/c/Users/diete/native_geometry_spec_9967.json --output /mnt/c/Users/diete/NEW-EXECUTOR-OUTPUT
```

The next agent should first read the root master turnover for current run state.
Then implement the optional solver batch boundary with TDD, keeping sequential
behavior as default. Construct one immutable request per independent window
after parent-owned node retraction, evaluate only cache misses in a persistent
pool, and restore original window ordering before existing residual/Jacobian
assembly. Qualify identical assembled objective, constraints and Jacobians,
including repeated candidates/cache hits and worker-error cleanup. Only after
root review should a separately recorded bounded fitting trial be considered.
Do not restart completed run19 or interrupt an unrelated live fit.

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
