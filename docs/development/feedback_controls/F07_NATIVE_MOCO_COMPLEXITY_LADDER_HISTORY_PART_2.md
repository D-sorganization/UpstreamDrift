# F07 Native Moco Complexity Ladder Historical Record — Part 2

Continuation of the earlier runs and saved-solution diagnostics in [Part 1](F07_NATIVE_MOCO_COMPLEXITY_LADDER_HISTORY_PART_1.md).

## Bounded Run With Source-Derived Control Bounds

One additional serial ladder was authorized at exact harness HEAD
`5a41243536c72ee06e456a6cc3692d1e5428b75d`, with the same 30-second limit per
rung, sampled 1 GiB soft RSS stop threshold, single-thread environment,
solver `parallel=0`, and stop-on-first-nonpass policy. The system-Python
supervisor was Python 3.13.5 with psutil 7.0.0; the isolated worker was Python
3.12.10 with OpenSim 4.6-2026-06-22-85aaf64. Only the `plain` rung ran; no
later rung started.

The solver reached `Solve_Succeeded` after 82 iterations with objective
`4.9211583953665505e-6` and solver duration 4.948 seconds. The staged source
was unchanged at SHA-256
`6a098517bdba09b19b3eb99e4da7e795f27d10be8f42c0fca04fee20ce67e143`. The
request bound both muscle controls to their native `[0.01, 1.0]` limits. All
21 persisted flexor and extensor controls are at or above 0.01: flexor range
`0.01000507697736689` to `0.08944407793890985`; extensor range
`0.010018064141171771` to `0.01006776583851479`. The first five controls were
also within their source limits. Solver equality-constraint violation was
`2.9318385382209183e-10` (configured tolerance `1e-7`), overall NLP error was
`8.593230782349099e-7`, and persisted solution round-trip error was at most 3
float64 ULP.

Independent fresh replay failed the unchanged criteria for speeds,
activations, and flexor fiber length. It used the persisted time-only control
law, and the replay object recorded exact equality between all applied control
samples and saved control samples (maximum absolute sample error 0). The first
mismatch appears at 0.0005 seconds: flexor activation error
`1.620052036322081e-4`, flexor fiber-length error `3.130300580411749e-7` m,
and base speed error `2.1240543793358277e-9` rad/s. Maxima were
`4.278119184863939e-6` rad/s speed error at 0.01 s (limit `1e-7`),
`4.2278841340708667e-4` activation error at 0.001 s (limit `1e-8`), and
`5.394625915713935e-6` m fiber error at 0.01 s (limit `1e-7`). Coordinate
error peaked at `1.524035136979407e-8` rad at 0.01 s (limit `2e-7`). The
request target is `0.30000558574001657` rad; optimized endpoint was
`0.3000055857400166` rad and replay endpoint was `0.300005600980368` rad, a
`+1.524035142530522e-8` rad target residual inside the unchanged endpoint
limit. All ten initial named states matched exactly.

Source-derived control limits remove the below-minimum control condition, but
the same replay gates still fail. This shows the prior bounds mismatch was not
sufficient to explain divergence; it does not establish another cause. Mesh
transcription/discretization remains a candidate to test separately from
control interpolation. No physical qualification or production admission is
inferred.

The worker completed in 13.636 seconds total supervisor elapsed time and
sampled peak process-tree RSS was 227,188,736 bytes. Recorded stage times were
0.662 s source build/equilibration, 2.081 s forward witness, 0.032 s solver
preflight, 4.964 s solve, 0.130 s solution export/control staging, 0.090 s
replay input preflight, and 0.007 s saved-input reload. The worker exited 1
after writing the failed replay audit, as required by fail-closed acceptance.
A post-run scan found no remaining native ladder worker process. No timeout or
memory stop occurred.

Artifacts are retained under
`experiments/native_moco_ladder_nativebounds_5a412435/plain/`:

| Artifact                            | SHA-256                                                            |
| ----------------------------------- | ------------------------------------------------------------------ |
| `request.json`                      | `b21492734171be2163f585447425a779f79d44a2020a7e2c2e03815328c073f3` |
| `solution.sto`                      | `558ceea7fd5fdc1f507dbd2042812385fbc186d516ff2cf8923013a4212f19b1` |
| `time_only_controls.sto`            | `9bc9b709d3a199152c17d11ec182ff98991fddeb066d101f1c0e0e99fc7c030a` |
| `solution_serialization_audit.json` | `aa15a6e8c1b5bf6c3086f083f47c688567258fd6ea5c2b788b61bc343d070fdd` |
| `fresh_replay.json`                 | `ba5f9f309beabe10216e0c0d27167e15bfecd4670827aaabf3f8fd9d2f81bee6` |
| `worker.stdout.log`                 | `7f702fbf36fe9fb8763800b40d8d87ba5191b530a439fc20bd41bf577c36ed63` |
| `worker.stderr.log`                 | `47cb62f37b764b3995f99fc4f6af557ce945feb20226f2b03c618e8fe3d76aea` |
| `progress.json`                     | `5b1a30c4564291c706804844d883aa823d1254cd4abd4d0b33d66a420f5f8049` |

The worker source SHA-256 recorded in the request is
`f8fc7c19ff7835d8f5ebe569f7e0f67186143a8b79fa4733cb8dfac7f7d84e06`; the
harness/builder SHA-256 is
`62ed9de2d82713ed36be87decfc83a4a3e63c9f2886ad4573e85d9fd0ed77a07`. The
state-error matrix SHA-256 is
`a2eec5558da62a25088b31069757fa96a33f3be49476cba83c71c407eedf2b57`.

## Mesh-Refinement Candidate Configuration

The next diagnostic candidate is a single `plain` rung with 20 Hermite-Simpson
mesh intervals over the same 0.01-second horizon. It keeps the exact source,
source-derived `[0.01, 1.0]` muscle control limits, named cold states,
constant-0.05 forward witness, target construction, ten-state replay checks,
all acceptance tolerances, solver parallel setting 0, and saved time-only
piecewise-linear replay law. The finer mesh changes the discretization and
number of optimized mesh samples; it does not switch the control law to a
different interpolant. No endpoint, horizon, activation, or integration
tolerance is changed.

The worker now accepts an explicit mesh interval count while retaining 10 as
its default, and the request records both requested and read-back native solver
settings. The solver is explicitly configured as `hermite-simpson` with
`interpolate_control_mesh_interior_points=true`; request creation fails if the
native getter values do not exactly match those settings. A native setup test
constructs both the unchanged 10-interval baseline and a 20-interval candidate
without solving, checking transcription, midpoint interpolation, parallelism,
and mesh count. A separate contract test verifies the CLI's 20-interval flag
and keeps the 10-interval default flag-free. Expected exported solution size is
41 rows for the 20-interval mesh, versus 21 rows in the historical 10-interval
runs. This candidate configuration was committed before its separately
authorized run, whose result follows.

## Bounded 20-Interval Plain-Rung Run

One plain-rung run was authorized at exact HEAD
`e1548c51790b71e4db2b70ba26feb8a56bbaaba1`, with the same 30-second
per-rung wall deadline, sampled 1 GiB soft worker-tree RSS threshold, stop-on-
first-nonpass policy, single-thread environment, solver parallel 0, and all
previous acceptance tolerances. The unchanged source was
`6a098517bdba09b19b3eb99e4da7e795f27d10be8f42c0fca04fee20ce67e143`. The
request confirms the native solver read back 20 intervals,
`hermite-simpson`, midpoint interpolation true, and parallel 0; replay remains
piecewise linear with the same 0.01-second horizon, target
`0.30000558574001657` rad, initial q 0.3, named cold state, and source-derived
muscle bounds `[0.01, 1.0]`.

The optimizer converged (`Solve_Succeeded`) in 122 iterations with objective
`4.1654698156426359e-6`, solver duration 18.044235 seconds, equality
constraint violation `2.0773524207241822e-9`, and overall NLP error
`1.2395367043207645e-7`. The solution contains the expected 41 rows from 0 to
0.01 seconds at 0.00025-second spacing; its final q is
`0.3000055857400166` rad, `5.551115123125783e-17` rad from target. Controls
are within the native lower limit: flexor range
`0.010002529403274341` to `0.1066043472144142`; extensor range
`0.01000010113709679` to `0.010015522473752789`. Solution serialization
round-trip remains within 3 ULP.

The supervisor reached its 30-second deadline at 30.228 seconds while the
worker was in the fresh native replay call. Peak sampled process-tree RSS was
219,119,616 bytes; the supervisor terminated and reaped the owned process
tree, and a post-run scan found no remaining native ladder worker. The request,
solution, saved controls, serialization audit, source, and logs are present,
but no `fresh_replay.json`, receipt, or replay-state error metrics were
produced. Therefore this is a successful mesh-20 optimizer solve followed by
an incomplete replay diagnostic, not a passing rung. The run does not establish
whether refinement improves independent replay agreement. The lower objective
than the 10-interval result is not directly comparable evidence of replay
quality because the optimized mesh sample count also changed. No second run or
later rung was started.

Stage measurements were 4.190 seconds source build/state equilibration,
0.967 seconds forward witness, 0.589 seconds preflight and solver setup,
18.068 seconds solve, 0.137 seconds solution/control export, 0.175 seconds
replay-input preflight, and 0.008 seconds reload. The progress file contains no
completed `fresh_native_replay` stage because that call had not returned before
the parent deadline.

Retained files under
`experiments/native_moco_mesh20_e1548c5179/plain/` have these SHA-256 values:

| Artifact                            | SHA-256                                                            |
| ----------------------------------- | ------------------------------------------------------------------ |
| `request.json`                      | `86570790ab4fce02df7b8a1c899b0755a7518c36c541fa366a2dd467a6a8f1a9` |
| `solution.sto`                      | `a13dfe3a7e793efff177a1c2cb64dbb055cd57e1704498a99e9fa3e80340578f` |
| `time_only_controls.sto`            | `dced78b02cea4d36af31d6f08b881d9feaa87cff1ac5a87d2bebfa5d592a1796` |
| `solution_serialization_audit.json` | `61b09b3d1d0efcf311d335231e63df14b836365028f58e7c6e536a39c9685f8f` |
| `progress.json`                     | `f9cc524d1334175f843e927803b8fe7758adf26ee58b3192f2bf1c2d4a039844` |
| `worker.stdout.log`                 | `04ef384237a2618fa74cefb36b20535496792415b178e5c6aa7230a8fd6e1d63` |
| `worker.stderr.log`                 | `11b47947f5283764fe1aabf32eb6563f9ca10af8f54d8e63c53429f1f3a9f579` |

The worker source hash in the request is
`98a375933f0e64fac364e2223eb803743000120d2ff6a3bf1d3b309d09b94a43`, the
harness/builder hash is
`55afad2bbc57ad8b471ba1addd3cecec53b1aea275420151876941f04c8d6ae2`, and
the replay mechanics profile hash is
`189c5d07c24b6b8c993aedba8cb794dbe82925bb6b368fef98be1327c2057fbf`.

## Retained-Input Replay Diagnostic Attempt

After the mesh-20 solve timed out during its integrated replay stage, an authorized no-optimizer diagnostic was launched against the saved `plain/` source, request, solution, and controls. The first launch used the supervisor log directory as the worker output directory, although `run_saved_replay_diagnostic()` creates its output with `exist_ok=False`. It reached the 30-second `wall_timeout` with peak sampled process-tree RSS 154,619,904 bytes, but wrote no worker audit or stage record. That launch did not establish whether replay began.

A corrected diagnostic was authorized at exact HEAD `a5a0fcdb8ecc7a75227f771550dd91e9cb933311`, with a new, non-existing `native_audit/` child directory separate from the supervisor logs. It retained the 30-second deadline, sampled 1 GiB soft process-tree RSS threshold, one-thread environment, and owned-tree cleanup. The durable worker stage is `native_replay_started`: OpenSim and the replay provider imported, the saved inputs loaded, and the fresh native replay call began. The call did not return before the wall deadline, so no state-error audit was written. This is an incomplete replay diagnostic, not a measured replay failure or a passing replay.

The supervisor terminated and verified cleanup of its owned process tree. A wrapper bookkeeping typo then prevented it from writing the timeout result and sampled peak RSS; those fields are recorded as unavailable rather than inferred. No optimizer ran, no deadline or acceptance limit changed, and no retry or later rung was started. The retained `invocation.json` binds the exact command, non-secret environment, source/request/solution/control hashes, and worker/harness source hashes. `native_audit/diagnostic_progress.json` records the final stage. The supervisor stdout and stderr files are empty.

| Diagnostic artifact                                                         | SHA-256                                                            |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `diagnostic_retained_replay_a5a0fcdb/invocation.json`                       | `d8d4bdcf047f97b6d92e219203399ae9d5af2d6564c2dfe9a0e59bf384bbff29` |
| `diagnostic_retained_replay_a5a0fcdb/native_audit/diagnostic_progress.json` | `7bf7945fa5c2d8e4aadd6761549393ef1b996fbc8d308ffc54e91c0730a6cf23` |
| `diagnostic_retained_replay_a5a0fcdb/supervisor_result.json`                | Recorded after timeout; sampled peak RSS unavailable               |

The diagnostic worker now fsyncs its stage before/after imports, saved-input loading, native replay, and audit writing. Its contract test checks durable initial stage output and a distinct worker-output child under the supervisor log directory. This instrumentation does not extend the deadline or change replay acceptance.

## Read-Only Comparison of Saved 10- and 20-Interval Solutions

The source-bound 10-interval and 20-interval solution files cover the same
0–0.01 s horizon, with 21 nodes at 0.5 ms and 41 nodes at 0.25 ms. At their
common 10-interval times, clock values differ by at most
$1.74\times10^{-18}$ s. The base-coordinate values differ by at most
$5.66\times10^{-11}$ rad and dependent-coordinate values by
$1.07\times10^{-10}$ rad. Base and follower speed differences are at most
$3.23\times10^{-8}$ and $3.84\times10^{-8}$ rad/s; muscle-state differences
at common times are at most $1.36\times10^{-4}$ activation and
$2.93\times10^{-7}$ m fiber length.

The saved control traces differ more. Flexor commands differ by up to
0.0352284 at common knots; ranges are 0.01000508–0.08944408 at 10 intervals
and 0.01000253–0.10660435 at 20 intervals. Extensor traces differ by up to
$5.2244\times10^{-5}$, with ranges 0.01001806–0.01006777 and
0.01000010–0.01001552. Both remain within the source-derived [0.01, 1.0]
bounds. The 10-interval replay has a retained error audit and fails speed,
activation and fiber-length limits. The 20-interval run has no replay audit;
its separate replay timeout supplies no state-error values. This comparison
shows the refined optimization selected a different control trace, but does not
identify why the 10-interval replay disagreed or predict a 20-interval replay
result.

The corrected diagnostic manifest binds worker source SHA-256
`cb5adcb83548bc261e7e672194ac643d551c5810c6245d4601a6f821d27d11a1` and the
unchanged mesh-20 source/request/solution/control hashes listed above. The
supervisor wrapper mistakenly read `_WorkerTimeout.reason` instead of its
`stop_reason`, after process-tree termination had completed. Thus the durable
worker stage and wall-timeout reason are known, but the wrapper did not persist
the sampled peak RSS. No replay acceptance or qualification is inferred.

## Completed Retained-Input Replay Diagnostic at Harness `3e195b0e`

After adding scoped `faulthandler.dump_traceback_later(10, repeat=True)` around
the retained replay call, with cancellation in `finally`, exactly one
no-optimizer diagnostic was authorized at commit
`3e195b0e863a869cd5e0e66259203346ccb11508`. The source, request, solution and
saved-control inputs were re-hashed before launch and matched the mesh-20
identities above. The fresh worker reached `audit_written` and exited 0. The
existing process-tree supervisor recorded a 210,214,912-byte sampled peak,
below its unchanged 1 GiB soft threshold, and verified owned-tree cleanup. A
post-run process scan found no matching Python worker. The stack trace file is
empty because replay completed before the first 10-second dump. Supervisor
stdout/stderr were retained. No solver or optimizer was invoked.

The full 41-knot saved-input replay **failed** its unchanged per-unit limits.
The target endpoint predicate passed: optimized target
`0.30000558574001657` rad, replay endpoint `0.30000558954459644` rad, residual
`3.804579873367686e-9` rad. Maximum errors were `3.804579817856535e-9` rad for
coordinates (limit `2e-7`), `1.0189039163566779e-6` rad/s for speeds (limit
`1e-7`), `1.1531595562106617e-4` for activations (limit `1e-8`), and
`1.2169215853896631e-6` m for fiber lengths (limit `1e-7`). The failed-unit
peak errors occurred at or near the final knot for speed and fiber length, and
at the initial replay step for activation. The audit retains timewise named
state arrays and hashes; its SHA-256 is
`ad6f872d5f385f67a8ff60e014cd9d330f7311f4b2b4f49c7407e9bf5653f1db`.

The saved flexor/extensor control samples were applied exactly at every audit
sample (`max_applied_control_sample_error = 0`). This establishes knot-value
identity, not equality of the continuous control law between knots. The request
declares Hermite-Simpson transcription with control-mesh interior interpolation
enabled, while fresh replay uses piecewise-linear interpolation of the saved
knots. A read-only OpenSim 4.6 property inspection of the exact source found
both Millard muscles retain activation dynamics, default activation `0.05`,
activation time constant `0.01` s, deactivation time constant `0.04` s, and
source excitation bounds `[0.01, 1.0]`. This is a candidate numerical
transcription/control-representation difference, not a demonstrated cause:
the optimized/replay activation divergence is visible at the first 0.25 ms
sample even though applied controls agree there, and this diagnostic did not
retain an independent continuous-input integration trace. Solver feasibility
and physical qualification remain unestablished.

The tool-observed end-to-end command wall time was 7.6 seconds; the supervisor
receipt does not separately time native replay, so no native replay duration is
claimed. Its stdout reports 300 realizations, 47 accepted / 63 attempted
Runge-Kutta-Merson steps, and 63 integrator iterations. The trace file is empty
(SHA-256 of the empty file); this is consistent with completion before 10 s,
but does not give a stage duration.

The complete retained invocation is under
`experiments/native_moco_mesh20_e1548c5179/diagnostic_retained_replay_fauldump_3e195b0e/`.
The invocation manifest SHA-256 is
`525831d63e32ace8b2eea67d009f4d727097bd059d604f9a270b0dfc41586edf`; supervisor
result is `4c188ed0d80e8d7ac8de02672376103a401d5b2bf239e7a676b6f69d679df659`,
progress is `fe4caa35f2d72feae8bcc044e853012ee786f23f076abaeee7f11ed42d921864`,
and stdout is `bb98c739736ebde04ab5e593340edd39fd6e4e11dc0452345fc0dceabb5bbe9d`.
The empty stderr and stack files both hash to
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. The
worker and harness source hashes are respectively
`826bb8311a0fdf9e9612b7bcd146a33bc6915529264a2bd8db4fa4da747673af` and
`ff19221e4bc8823e5c87e5007108b88d799a9148953fb9e3b8e22839cc4d0c2a`. The
retained source, request, solution and controls stayed byte-identical to their
pre-run hashes. No tolerance changed, and no additional native run is implied.

## Completed Offline Activation-Law Analysis

Initial screening treated adjacent output rows as Hermite-Simpson cells; that
estimate was not a valid cell residual. The 41 exported rows cover 20 cells.
Each cell uses endpoint, midpoint and endpoint rows over 0.5 ms. The corrected offline
analysis in `experiments/native_moco_mesh20_e1548c5179/activation_residual_analysis.json`
uses that three-row layout, the OpenSim 4.6 first-order activation equation,
and a segmented SciPy DOP853 integration that restarts at every saved
piecewise-linear excitation interval. It runs without importing OpenSim,
constructing a model or replaying the native trajectory. The analysis code is
`scripts/opensim/analyze_retained_activation_residual.py`; ten pure regression
tests are in `tests/opensim/test_retained_activation_residual.py` (11 tests
after adding the per-cell restart and branch-crossing regression).

The flexor Hermite-Simpson midpoint-state and Simpson dynamics residual maxima
are `1.1397e-10` and `2.0774e-9`; extensor maxima are `1.6274e-11` and
`1.7293e-10`. The exported values therefore satisfy the reconstructed
discrete collocation equations closely. The flexor activation in the optimized
trajectory nevertheless differs from the independent segmented scalar trace
by up to `1.15316e-4` at 0.00025 s, and by `-8.10477e-5` at the final time.
Fresh native replay agrees with the segmented trace to `2.57503e-8` maximum
absolute flexor activation difference and `1.98e-15` for extensor. The saved
control midpoint differs from linear endpoint interpolation by no more than
`5.21e-18` (flexor) or `6.94e-18` (extensor). The independent integration's
change between tolerances `(1e-12, 1e-14)` and `(1e-13, 1e-15)` is at most
`3.78e-13` for flexor and zero for extensor; this is a sensitivity estimate,
not a rigorous error bound.

The source XML omits `activation_time_constant`,
`deactivation_time_constant`, `minimum_activation` and
`ignore_activation_dynamics`. The calculation uses OpenSim 4.6 source defaults
`0.01 s`, `0.04 s`, `0.01`, and `false`, respectively; these values were not
read from the installed binary during this offline analysis. The law and Moco
transcription references are recorded in the JSON. Exact source/request/solution/
control/replay-audit SHA-256 values are also recorded there; the report digest
is recorded below after adding the per-cell allocation.

The evidence separates a small reconstructed Hermite-Simpson defect from the
larger optimized-state versus continuous-replay difference. It does not
identify its cause, show physiological validity, change replay thresholds,
or qualify either synthetic solve. In particular, sampled control identity
does not establish whole-capture muscular prediction. Any follow-up numerical
experiment requires separate review and authorization; no solver, optimizer,
native replay, mesh refinement or tolerance change is authorized by this
analysis.

A second offline-only calculation allocates the error per full Hermite-Simpson
cell. Each 0.5 ms cell's scalar integration starts at that cell's saved
optimized left activation and traverses its two saved piecewise-linear input
half-intervals; a separate trace accumulates continuously from the initial
state across all cells. For flexor, the largest local midpoint discrepancy is
`1.1531598e-4` in the first cell and the largest local endpoint discrepancy is
`9.4146359e-5`. The continuously accumulated scalar trace ends `-8.1047709e-5`
from the optimized activation. The first-cell saved excitation is
`[0.10660435, 0.06055026, 0.01449617]` and optimized activation is
`[0.05000000, 0.05130361, 0.05131032]`; the sampled chord's excitation-minus-
activation changes sign in its second half at the linearly estimated time
`0.0003001872 s`. This is a sample-based branch-crossing estimate, not a
continuous event location. The largest local tolerance sensitivity was
`3.7749e-13` between the two DOP853 settings. These comparisons localize the
discrepancy numerically but do not establish why it occurs. The extensor local
endpoint discrepancy maximum is `1.7235e-10`. All inputs retain the same
source/request/solution/control/replay hashes; the new report SHA-256 is
`1195cdf39855accfd761e41410d8b23121cafbb28f0407d69e3a479d1e136116`.

## Bounded Nonuniform Refinement Design

The five public synthetic parent inputs are retained byte-for-byte under
`experiments/native_moco_mesh20_e1548c5179/`: `plain/source.osim`,
`plain/request.json`, `plain/solution.sto`, `plain/time_only_controls.sto`, and
`diagnostic_retained_replay_fauldump_3e195b0e/native_audit/fresh_replay_diagnostic.json`.
The test `test_committed_activation_analysis_reproduces_retained_parent_bindings`
recomputes the offline analysis from these files. The current plan binds their
hashes and the analysis report before creating a child attempt.

The pure one-pass planner uses an indicative `1e-8` absolute local activation
indicator. In the retained case it marks full Hermite-Simpson cell 0 and inserts
normalized mesh point `0.025` (physical time `0.00025 s`), making 21 intervals.
The generic planner accepts up to four flagged parent cells and caps the child
mesh at 24 intervals. It rejects stale source, request, solution, control or
replay-audit bytes, malformed meshes and cap overflow without dropping marked
cells. Repeating the same attempt ID collides with its existing output path,
preventing an automatic second attempt.

The child configuration uses Moco's nonuniform normalized mesh, carries state
initialization from the saved parent trajectory, and seeds each control by
piecewise-linear interpolation of the persisted control STO at every child
endpoint and midpoint. It does not resample controls. It preserves the parent
source, horizon, initial-state declaration, target, effort objective, native
control bounds, Hermite-Simpson scheme, midpoint-control policy, solver limits
and replay thresholds. The existing isolated-worker 30-second wall and 1 GiB
sampled soft RSS controls remain in effect; attempt provenance keeps parent and
child hashes, worker command, timeout or failure status, elapsed time and
sampled peak when present.

This committed code and its pure tests are a refinement plan only. No refined
native solve or replay has run, and no native API/time-grid acceptance is
claimed. A smaller local activation indicator is not a guarantee of convergence
or replay acceptance and does not qualify fiber lengths, kinematics, or
physiology. Any later child still has to pass the unchanged independent replay
checks and receive a separate review before execution.

The post-solve grid audit reads the seeded grid from the child request,
leaving the retained plan identity immutable. Before solver setup, the worker
recomputes and compares the complete canonical plan, including source hashes,
child mesh and preserved contract; changing a plan field while keeping its
attempt ID is rejected. Pure regressions cover JSON reload, plan tampering and
the distinction between plan data and request-only seed data.

The saved-solution guess path exports its states as an OpenSim
`TimeSeriesTable` before `insertStatesTrajectory`, matching the installed 4.6
binding. It then restores each declared initial state value exactly and checks
the readback, so interpolation roundoff cannot alter the fixed initial state.
A fake-SWIG regression covers both the accepted table signature and exact
initial-state restoration; no native child build or solve was run.

An installed OpenSim 4.6 binding smoke then loaded the committed parent
solution and control STOs, inserted an exported state `TimeSeriesTable` into a
native data-bearing `MocoTrajectory` with 43 time rows, and seeded both control
channels through a stub solver. Exact initial values were read back. The run
used the existing process-tree supervisor (30-second wall limit, 1 GiB sampled
RSS threshold; 140 MB peak) and made no model initialization, study solve, or
replay call. An earlier smoke-test attempt populated an empty trajectory with
per-state setters and crashed inside the SWIG setter; the data-bearing native
constructor fixed that test setup. This API smoke does not validate an actual
Moco solver guess or child solve.

The first corrected-constructor probe then exposed a separate SWIG vector-iteration defect: iteration over the native 43-element time vector attempted index 43 and failed. The worker now reads native vectors by explicit indices bounded by the reported size; the same installed-SDK seed smoke passed after that correction. Both failed probe outputs remain preserved in their original experiment locations. The successful smoke still uses a stub solver and does not establish actual solver acceptance.

The first authorized preparation-only wrapper attempt stopped before OpenSim import because its supervisor argument pointed at a nonexistent nested directory. A corrected wrapper was then invoked once from commit `37439d18d0413c135680968d512a03e278a82cf2`, using a fresh output folder. It launched the SDK interpreter, but the worker exited with code 1 before its start marker or OpenSim import because that SDK environment does not contain `psutil`, which the wrapper imported to record its PID. No source staging, witness replay, model initialization, or solver setup occurred. The supervisor sampled 19,951,616 bytes peak RSS and observed no active process tree on return; worker-PID cleanup could not be independently recorded because the start marker was never written. The immutable receipt is `experiments/native_moco_refinement_prep_37439d18_20261011T071340Z/supervisor_result.json`; it binds the exact HEAD, wrapper/worker/harness hashes, and all five retained parent input hashes before and after (identical). The earlier first-attempt wrapper files were overwritten during correction and are not byte-preserved; no original hashes are claimed. That first psutil-related attempt was not retried.

Review of the custom mesh path found that OpenSim's uniform `get_num_mesh_intervals()` getter can retain its unrelated default while `setMesh()` receives the requested nonuniform grid. For refinement requests, the receipt now labels that getter as ignored and separately records the normalized points and requested interval count; exact seeded native guess clock validation remains the grid evidence at preparation, with the exported solution grid audit reserved for a later solve. A fake solver regression retains a uniform getter value of 100 while accepting the custom 21-interval request, and checks that the configured points are not substituted by the uniform count.

The replacement stdlib-only child wrapper passed its supervised `--stub` validation with all five retained-input hashes unchanged, accurate witness/report paths, and no OpenSim import or model construction. The one separately authorized native preparation attempt used byte-identical wrapper files and commit `4a2df8866ff12a4e76fbf3b1b9a0b70860f874ad`. It imported the SDK, staged the retained source with SHA-256 `6a098517bdba09b19b3eb99e4da7e795f27d10be8f42c0fca04fee20ce67e143`, loaded the model, initialized its system, and equilibrated muscles while deriving the declared cold start. The helper returns `(path, declaration)`, but refinement staging treated that tuple as the declaration; `dataclasses.replace` then raised `TypeError: replace() should be called on dataclass instances`. The attempt exited 1 with 175,058,944 bytes sampled peak RSS. The process-tree supervisor returned after observing no active owned processes. All five retained parent input hashes are identical before and after. No fixed-excitation forward witness, constrained replay bundle, Moco study/solver setup, solve, optimizer, or child solution replay ran. The immutable receipt and traceback are `experiments/native_moco_refinement_prep_4a2df886_20261011T072232Z/supervisor_result.json`. The source fix now unpacks the documented helper tuple and asserts its returned path matches the staged source; a pure regression reproduces that tuple contract. This attempt was not retried.

The next preparation-only attempt ran from `edc346b1302629c09f9a74058da5f4dbc05392d5` using system Python for the process-tree supervisor and the pinned OpenSim SDK child. It passed staging, the existing 10 ms fixed-excitation forward witness, model/study setup, and native warm-seed construction; the native seed clock was validated as 43 Hermite-Simpson points. It then failed the strict requested-versus-read-back solver-settings check in `_validate_actual_solver_settings` before `request.json` was written. The bound settings were Hermite-Simpson transcription, interpolated controls, parallelism zero, and a custom 21-interval grid with normalized points from 0 to 1 in 0.05 increments. The exact native readback is unavailable because this attempt did not persist the request before validation; the failure does not identify which field differed. The supervisor reports exit code 1, sampled peak RSS 203,116,544 bytes, verified owned-tree cleanup, and all five parent-input hashes unchanged. Its immutable receipt is `experiments/native_moco_refinement_prep_edc346b130_root_20261011T072745Z/supervisor_result.json`; source, witness STO files, stage progress, and wrapper/harness/worker hashes are retained in that folder. No optimizer or child solution replay ran. A pure regression now ensures each future attempt writes the bound request and native settings readback before strict validation, so a mismatch remains diagnosable without weakening the comparison. The edc346 attempt was not retried; the distinct follow-up below exposed the in-memory representation mismatch.

The separate preparation-only attempt at `c6c232e12dffa9ab6fd48bdb6052cdcaf7f9c35d` reached the same native 43-point seeded guess and again stopped in solver-settings validation, with code 1, 204,349,440 bytes sampled peak RSS, verified process-tree cleanup, and all five retained parent-input hashes unchanged. Its persisted `plain/request.json` shows the requested custom mesh and actual settings are semantically identical, and the JSON-reloaded request passes `_validate_actual_solver_settings`. The failure therefore occurred only on the in-memory path: the planner holds `normalized_mesh_points` as a tuple, while the actual-settings readback makes a list. Exact ordered numeric values match; Python's tuple/list comparison fails. `_refinement_request_fields` now canonicalizes the tuple to a list at the request boundary, preserving exact points and keeping changed points subject to strict failure. The native run remains preparation-only: no optimizer or child solution replay ran. Its immutable receipt is `experiments/native_moco_refinement_prep_c6c232e12d_root_20261011T073310Z/supervisor_result.json`. The distinct follow-up below records the next wrapper-level check.

The root-run attempt at `05593eb896c3186a0c9edba1572d8b359c90bc74` completed core preparation, including strict settings validation and a 43-point native seeded guess, but the external wrapper then repeated a tuple-versus-list comparison of the actual setMesh points against the reviewed plan and exited 1. Its receipt is `experiments/native_moco_refinement_prep_05593eb896_root_20261011T073710Z/supervisor_result.json`; sampled peak RSS was 203,845,632 bytes, owned-tree cleanup was verified, and all five parent input hashes stayed unchanged. The request records validation status `passed`, exact 21-interval custom points, source and witness hashes, cold state and control bounds. The separate `core_preparation_audit.json` records 13 successful offline checks tying those saved values to the refinement plan and witness bytes. A shared `_validate_preparation_request_bindings` helper now checks strict settings, exact list-canonicalized mesh values, the exact 43-point native seed clock, and interpolation policy; both wrapper stub and native paths use it. The wrapper candidate applies that same contract in stub mode, preventing a stub pass from bypassing the native binding assertions. No solve, optimizer, or child replay ran, and this follow-up has not been rerun.

The post-solve mesh audit now canonicalizes only the container representation at its JSON boundary: planner points may be a tuple and request points a list, but their ordered numeric values must match exactly. The native seed clock must still equal the planned Hermite-Simpson grid exactly. A regression exercises that tuple-to-list boundary and rejects changed mesh values, preventing a child from failing after solve/export solely because of container type.

A dedicated supervisor candidate prepares exactly one `plain` rung through the existing `run_ladder` process-tree supervisor. It validates and byte-copies the exact plan `experiments/native_moco_refinement_prep_05593eb896_root_20261011T073710Z/refinement_plan.json` (SHA-256 `746f731ce8f2c4002dcdc87c4dd346f98d5bc7ba00b2297c0da63f2b47df2770`) and invokes the worker with 21 intervals, one 30-second wall limit, a sampled 1 GiB RSS soft threshold, and single-thread environment settings. The wrapper records the actual HEAD, harness/worker/planner/supervisor hashes, bounded OpenSim runtime fingerprint, interpreter identity, the five retained input hashes and analysis-report hash before and after, result and output digests, and verified process-tree cleanup. Any input, report, code, runtime, interpreter, or plan mutation invalidates the outcome. Invocation and result records use the shared atomic, fsynced JSON writer. It cannot retry or advance to another rung. Its stub exercises the same plan validation and command construction without launching OpenSim. The command and hashes remain a review candidate; no native child solve/export/replay was launched from this candidate. Any later failed solve, serialization audit, or fresh-replay rejection must retain its complete evidence and remain failed under the existing tolerances.

The first authorized plain 21-interval child attempt ran from `cecfcd98f4035c44bfead453672c4694057c4405` in `experiments/native_moco_refinement_solve_cecfcd98_root_20261011T080500Z/`. It completed the fixed-excitation witness, source/state setup, and the optimizer; Ipopt reported `EXIT: Optimal Solution Found` after 77 iterations with objective `4.6865422780214056e-06`. The worker then exited 1 during post-solve clock auditing, before writing time-only controls or performing fresh replay. The attempt is therefore a failed child execution, not a solved-and-replayed pass or physical qualification. The unchanged 30-second / sampled 1 GiB supervisor observed 20.193 seconds and 207,364,096 bytes peak RSS; owned-tree cleanup and all source, plan, runtime, interpreter, analysis, and five parent-input hashes were verified unchanged.

Offline comparison of `plain/mesh_grid_audit.json` shows the plan and persisted solution each contain 43 clock entries, with endpoints exactly `0.0` and `0.01 s`. Nine entries differ bitwise from the Python planned grid; every difference is exactly one float64 ULP, with maximum absolute difference `8.673617379884035e-19 s`. The remaining 34 entries are bitwise equal. The saved solution’s independent serialization audit passes its separate 8-ULP bound: time max 1 ULP, states max 3 ULP (absolute `5.551115123125783e-17`), and controls max 3 ULP (absolute `6.938893903907228e-18`). Thus the observed block is a post-export floating-point clock-representation mismatch; it is not evidence of a changed interval count or an optimizer failure. The exact serialized arrays are retained and no time-only controls or replay audit exist for this child.

Immutable attempt receipt SHA-256 is `72bbe202491f49eaffdd9baaa87ddd85e5b01c88d4fe91328790d1f4e3ef13e8`; mesh audit `93403801b94986fd51fe1136d3c21cb1a3e1595f90ad6787c2694a4aec8a15b5`; serialization audit `da7a3b732714a3b5bf219f9ab73168325cf7d0882f45fac911d735f81b940886`; solution STO `0ba31fd0f6e6859ab4e8b20ed82cb7d9f9eb1092b8e279b0e3954990440b8996`; request `ba01112a4ea142d1b93e89a27b5f381829889fd3799aa613eb8e2efdd737174c`; source `6a098517bdba09b19b3eb99e4da7e795f27d10be8f42c0fca04fee20ce67e143`; progress `3631734345e15ffce200ec908528cb0eabad30fd36a920b2417c06966003c317`; stdout `154decd8d4fd868aa7daae81871f23a5904e9cb1ecdbb847db61e49422a43f5a`; stderr `e95c9e6466f4a27443d7ff67b088f915e2f11f00b5b21cc3a3a3ca7fc8b41baa`.

No new one-ULP grid acceptance is proposed. Offline comparison against the persisted 43-row native grid tested three arithmetic constructions: normalized midpoint then horizon scaling (the existing Python plan), midpoint of horizon-scaled endpoints, and scaled left endpoint plus half a scaled interval. The first differs at 9 rows by one ULP; each alternate differs at 11 rows and reaches two ULPs. All retain the exact 0 and 0.01 second endpoints. These candidates do not reveal the SDK's precise midpoint arithmetic, and the independent 8-ULP STO round-trip contract remains separate. The strict grid failure is preserved.

For a diagnostic that does not repeat the optimizer, `prepare_solution_controls_snapshot` copies the failed attempt's source, request, and solution byte-for-byte into a new directory and derives a separate time-only control STO from the persisted solution's exact time and declared control columns. The request's source control bounds determine which columns qualify as controls and the allowed values; exact control-name selection excludes muscle activation/fiber state columns. Its manifest binds the three copied files, the failed mesh and serialization audits, and the derived STO, and labels the result diagnostic-only/unqualified. The input snapshot is `experiments/native_moco_refinement_solve_cecfcd98_root_20261011T080500Z/solution_control_replay_inputs_final/`; its manifest SHA-256 is `e9100b2682fec486d83abcbc8bf514d5ba8850a1afcfab93a2f8c7744308c7a5`.

The one authorized retained-input replay ran from commit `3bc787e36d2002776198e70506ab74d86c28da6d` using the snapshot above. The worker returned code 0 within the existing 30-second / 1-GiB sampled soft limit; peak sampled tree RSS was 213,254,144 bytes, cleanup was verified, and all before/after source, request, solution, audit, code, runtime, interpreter, and snapshot hashes were unchanged. It invoked no optimizer. The fresh model used the persisted solution's 43 timestamps and controls; both applied control channels matched saved samples exactly (maximum sample error zero), and the replay endpoint met the existing endpoint predicate.

The complete replay audit is `experiments/native_moco_refinement_solve_cecfcd98_root_20261011T080500Z/solution_control_replay_diagnostic_3bc787e36d_root/retained_replay/native_audit/fresh_replay_diagnostic.json` (SHA-256 `dfcb7498c0aecd03928d82cade03c7bf1d867be1c88c44c2c2dd469c917ca97d`). It failed only the activation state limit: maximum absolute activation error was `5.078110452592238e-5` at `0.0007500000000000001 s`, above the unchanged `1e-8` threshold. Maximum coordinate error was `6.577771660687404e-11 rad`, speed error `3.859223773427893e-8 rad/s`, and fiber-length error `7.369955239144854e-8 m`; those unit limits and the endpoint predicate passed. This remains a diagnostic of the already grid-rejected child, not ordinary rung acceptance. It neither repairs the strict 9-of-43 one-ULP grid mismatch nor explains the activation discrepancy; no causal or scientific qualification claim follows.

The offline analyzer was generalized to use the exact saved time grid for nonuniform Hermite-Simpson cells. It retains `interval_count` as 42 adjacent row spans and adds the separate 21-cell count, saved cell widths, saved midpoint times, and mean duration; the first two cells are 0.00025 s and the remaining 19 are 0.0005 s (mean `0.0004761904761904762 s`). It validates finite, strictly increasing times, endpoint/midpoint row structure, and agreement between the saved cell count and the request. No planned grid is reconstructed, so the separate strict one-ULP rejection remains unchanged.

The source-bound offline result is `experiments/native_moco_refinement_solve_cecfcd98_root_20261011T080500Z/solution_control_replay_diagnostic_3bc787e36d_root/retained_replay/offline_activation_residual_analysis.json`, SHA-256 `36c4866850ef7ed5ca1202cedd920d373b12d25245f5a42e847a1a79a84436d7`. It imports no OpenSim SDK. For flexor, scalar integration from the optimized initial state differs from optimized activation by at most `5.078935403843299e-5`; replay differs from that scalar trajectory by at most `4.016980001408932e-8`. Per-cell integration restarted from each optimized left state reaches a maximum midpoint/endpoint difference of `9.262298642974398e-5`, versus `5.078935403843299e-5` for accumulated integration. The maximum Simpson dynamics defect is `3.5819897541253427e-9`. Flexor activation parameters use explicit XML values when present and otherwise OpenSim 4.6 source-tag defaults (0.01 s activation, 0.04 s deactivation, 0.01 minimum); the installed binary was not queried. These scalar/defect comparisons do not explain the failed replay or change qualification.

After the separate analyzer hardening and nonuniform-grid update, I regenerated
the offline analysis to a new file using the final source bytes, the existing
immutable snapshot, and the same fresh-replay audit. The prior report and its
source identity remain preserved. New report SHA-256 is
`d78b4efbfa66a01de23ad6403971ab2dca7e7171ccac6c4f7809f6cdaf3cd45d`; analyzer
SHA-256 is `31925492d691b40e42181b311f818b0b88038027cea2bd5641898310b5d650ca`.
It confirms 21 cells/43 rows. Flexor optimized-versus-scalar maximum error is
`5.078935403843299e-5`, replay-versus-scalar is `4.016980001408932e-8`,
maximum local-cell optimized-state difference is `9.262298642974398e-5`, and
maximum Simpson dynamics defect is `3.5819897541253427e-9`. Sampled chord
branch-crossing estimates lie in cells 0 and 2; saved controls are linear at
midpoints to about `7e-18`. These are local indicators, not proof of continuous
collocation error or causality. No native SDK was run, no thresholds or source
states changed, and the result remains unqualified. The final report path is
`experiments/native_moco_refinement_solve_cecfcd98_root_20261011T080500Z/solution_control_replay_diagnostic_3bc787e36d_root/retained_replay/offline_activation_residual_analysis_finalsource_4e40b764.json`.

A follow-up exact-input analysis additionally checks each stored middle row against the arithmetic midpoint of its adjacent saved endpoints within four ULPs at the local time-coordinate scale. The retained child differs by at most three such ULPs. This check validates endpoint/midpoint structure only; it does not reconstruct or accept the planned clock and leaves the separate 9-of-43 one-ULP planned-grid rejection intact. The follow-up report is `experiments/native_moco_refinement_solve_cecfcd98_root_20261011T080500Z/solution_control_replay_diagnostic_3bc787e36d_root/retained_replay/offline_activation_residual_analysis_midpointvalidated.json`, SHA-256 `fc6d1bf759c428e79d973b302ae81c96deadaf33d523d707623051352613ea93`.

Following the midpoint validator merge, I regenerated the offline report using
the final integrated analyzer and unchanged retained inputs. Analyzer SHA-256:
`c247a4cfd5369f829b5224c551713f7697e74bf5f430fcffbc59bd8ab1d9218b`; report
SHA-256: `fa913993a25903120c27d02d529bd67579475935021a70f54723ae269e20832e`.
The 43 saved rows represent 21 cells. The arithmetic midpoint check permits at
most four local ULPs and the retained midpoint offsets are at most three; it
accepts the persisted clock without reconstructing the planned grid. The strict
9-of-43 planned-grid rejection remains unchanged. Metrics agree with the prior
final-source offline report. This run used only saved files and did not launch
OpenSim, a solver, or replay; scientific qualification remains unestablished.
The new report path is
`experiments/native_moco_refinement_solve_cecfcd98_root_20261011T080500Z/solution_control_replay_diagnostic_3bc787e36d_root/retained_replay/offline_activation_residual_analysis_midpointvalidated_source_c247a4cf.json`.
