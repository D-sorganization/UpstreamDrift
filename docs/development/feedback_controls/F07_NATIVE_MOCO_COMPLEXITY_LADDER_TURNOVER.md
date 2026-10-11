# F07 Native Moco Complexity Ladder Turnover

## Bounded Run at Frozen Harness `856803317e0ccb26bd4ba43348244ac735c01f15`

One serial run was authorized with a 30-second ceiling per rung, a sampled
1 GiB process-tree RSS stop threshold, and stop-on-first-nonpass. The verified
supervisor was Python 3.13; the isolated worker used the repository's dedicated
OpenSim 4.6 SDK interpreter. The supervisor stopped after the first `plain`
rung. It did not start an optimizer or launch any later rung.

The worker failed after 2.974 seconds while flushing the staged `source.osim`:
`os.fsync()` received a read-only Windows file descriptor and raised
`OSError: [Errno 9] Bad file descriptor`. The supervisor observed a sampled
peak process-tree RSS of 184,315,904 bytes (about 176 MiB), then completed its
owned process-tree cleanup. This was a harness staging failure; it provides no
solve, export, replay, feasibility, or physiological evidence.

The immutable run artifacts are under
`experiments/native_moco_ladder_856803317e/plain/`:

| Artifact            | SHA-256                                                            |
| ------------------- | ------------------------------------------------------------------ |
| `source.osim`       | `6a098517bdba09b19b3eb99e4da7e795f27d10be8f42c0fca04fee20ce67e143` |
| `worker.stderr.log` | `64a0208343c2dbdb2c2cd53cb0a1cd32ea2e92563fbd6204cba81f34065f450b` |
| `worker.stdout.log` | `b106a55e233b4ced5aa9f6f0f38b10362308dce084af1d24df98de7f0a87f9c9` |

Review also found that the planned moving-path-point spline samples
`[-0.01, 0, 0.01]` at knots `[-1, 0, 1]` were collinear, so they did not
represent a nonlinear moving guide. The first run stopped before that rung and
does not validate it. A follow-up changes the monotone bounded sample law to
`[-0.008, 0, 0.012]`, checks nonzero curvature from the native-loaded path
point at two coordinate neighborhoods, and confirms that a collinear native
source is rejected. It also flushes the staged model through a writable file
descriptor. These corrections require a fresh frozen revision and a separate
explicit run authorization; this turnover does not authorize another solve.

## Bounded Run at Corrected Harness `9bbbc81cf7d4f4569d3cf08474a22bc180fc3f37`

After the writable-descriptor and native curvature fixes passed, one fresh
serial run was authorized with the same 30-second rung timeout, 1 GiB sampled
tree-RSS stop threshold, and stop-on-first-nonpass policy. The `plain` rung
staged and initialized its native source, constructed the solver, and entered
`MocoCasADiSolver.solve()`. Ipopt was actively iterating through restoration
steps when the worker was stopped at 30.049 seconds. Peak sampled
process-tree RSS was 210,210,816 bytes (about 200.5 MiB). The supervisor
terminated and reaped its owned worker tree; a post-run process scan found no
worker module process. No later rung launched and no retry was made.

`progress.json` records 0.191 seconds wall / 0.031 seconds CPU for source
build and state equilibration, then 0.062 seconds wall / 0.063 seconds CPU for
preflight and solver setup. The plain request has a 0.01-second horizon, ten
mesh intervals, initial coordinate 0.3, and final target 0.300005. The log
shows two muscle controls, an unactuated anchor, and restoration iterates
through at least iteration 231; the bounded log ends mid-iteration at timeout.
The worker produced no solution, saved controls, replay, or success receipt.
This is a solver nonconvergence diagnostic for this synthetic fixture, not a
feasibility or replay result. Higher-complexity rungs remain untested.

Artifacts are retained under
`experiments/native_moco_ladder_9bbbc81cf7/plain/`:

| Artifact            | SHA-256                                                            |
| ------------------- | ------------------------------------------------------------------ |
| `source.osim`       | `6a098517bdba09b19b3eb99e4da7e795f27d10be8f42c0fca04fee20ce67e143` |
| `request.json`      | `0be895bd98b3d0a7c224448b9ac1a9533668efe2332b1567402f1fa45a6f489a` |
| `progress.json`     | `4b248ceae071c5653f5533a75e267fdd31b1b3dd10308d68eacb19746f734cbf` |
| `worker.stderr.log` | `11b47947f5283764fe1aabf32eb6563f9ca10af8f54d8e63c53429f1f3a9f579` |
| `worker.stdout.log` | `6b55b2c3bd87642533f44c48906ed7c1fdf98a561b0d5bff138ff6edae5320a7` |

The repeated plain-source hash confirms that the corrected harness changed the
execution wrapper and MPP rung law, not the baseline model. Read-only OpenSim
4.6 evaluation of the plain source at q=0.3, zero speed, and equilibrated
activation 0.05 gives flexor moment arm +0.0224157198710 m/rad, extensor
moment arm 0, and q acceleration +0.1117164843 rad/s². The follower coordinate
acceleration is +0.2034137907 rad/s²; the fixed anchor is 0. The flexor and
extensor fiber states are 0.0954183220m and 0.0854558028m, respectively, with
0.1m optimal fiber length for both. The 5e-6 rad target over 0.01s is near a
constant-acceleration estimate from this one initial evaluation, which does
not prove reachability under full muscle dynamics.

The Moco setup assigns every non-activation state an interval of initial
value ±1. Its logged fiber-length ranges are [-0.904582, 1.095418]m and
[-0.914544, 1.085456]m, so they include negative, nonphysical lengths;
activation bounds are [0.01, 0.15]. The follow-up harness now constructs a separate synthetic positive-fiber-bound
witness from fixed 0.05 excitations, retains full source/profile/state/control
provenance, and reloads saved trajectories as the Moco guess. Its forward
integration and persisted-input checks pass under native tests; this correction
happened after the recorded timeout and was not part of that solve. Any future
solve still requires a separately reviewed bounded plan and independent fresh
replay. This remains unqualified synthetic software work.

Any new optimizer attempt needs a separately reviewed bounded plan and fresh
authorization; do not increase the timeout or automatically rerun this ladder.

## Bounded Run at Witness Checkpoint `8d14ce84cd89bb6b32577eb158320fce72896489`

One fresh serial invocation was authorized with a 30-second per-rung ceiling,
solver parallelism 0, native thread limits set to 1, 1 GiB sampled process-tree
RSS stop threshold, and stop-on-first-nonpass. The system Python 3.13 supervisor
used `psutil 7.0.0`; the worker used OpenSim `4.6-2026-06-22-85aaf64` from the
dedicated SDK interpreter. The `plain` rung stopped after 3.369 seconds with
`worker_failed` and peak sampled tree RSS 209,031,168 bytes (about 199.3 MiB).
The fixed-excitation native forward witness completed in 0.745 seconds wall /
0.641 seconds CPU, after source build/equilibration took 0.193 seconds wall /
0.094 seconds CPU. No Moco solver was constructed or run; no optimization,
optimized trajectory, saved-controls comparison, per-unit optimized/replay
error report, or fresh optimized-solution replay exists. No later rung launched
and no retry was made. A process query after termination found no Python ladder
worker.

Failure occurred while serializing the complete request: the cold named state
was exposed as an immutable `mappingproxy`, which `json.dump` cannot encode.
The staged source hash was `6a098517bdba09b19b3eb99e4da7e795f27d10be8f42c0fca04fee20ce67e143`.
The authorized worker and harness hashes were respectively
`f248255db92cbb4b5e0e7a32970ec3072cc811bc0a30db42052119e53b3ff864` and
`3cfa7341cd8dbcf5aa8330367c28bd38b3f5351c4c41ef418c59297a4f217922`.
The incomplete `request.json` bytes hash to
`fbad04a56dfe5c639003060c100e3cd0f2398da721a8426c26d806987e8413c6` and are
not a valid request. The complete request serialization now has a regression
that round-trips the whole rung configuration, exact cold state and units,
native profile, and its profile digest. Solver-setup stage timing now starts
after witness construction. This correction is a new checkpoint and has not
been run as a ladder. Synthetic feasibility remains unqualified.

Run files remain locally retained under
`experiments/native_moco_ladder_8d14ce84/plain/`; existing repository ignore
rules exclude worker logs. Their hashes are:

| Artifact                  | SHA-256                                                            |
| ------------------------- | ------------------------------------------------------------------ |
| `source.osim`             | `6a098517bdba09b19b3eb99e4da7e795f27d10be8f42c0fca04fee20ce67e143` |
| incomplete `request.json` | `fbad04a56dfe5c639003060c100e3cd0f2398da721a8426c26d806987e8413c6` |
| `progress.json`           | `346a67c71a282cce4988a02a430a9b720f93aa5deefccf909f6cce064693b7a1` |
| `worker.stderr.log`       | `920734f0dc5d744bf8dc74fe8d6831c02ab5004c7c535d9d3eab0345bb1bf373` |
| `worker.stdout.log`       | `e7d2618ebf54c04d3da82be34647918e6534f9480e6d6261e36221e53331db36` |

## Bounded Run at Corrected Checkpoint `5d8fe120773ca22df114b0d78b56e86bf2789934`

One fresh invocation was authorized with 30 seconds per rung, solver
parallelism 0, native thread limits 1, a 1 GiB sampled process-tree RSS stop,
and stop-on-first-nonpass. The Python 3.13.5 supervisor used `psutil 7.0.0`;
the worker used OpenSim `4.6-2026-06-22-85aaf64`. The `plain` rung's Moco solve
succeeded in 5.904 seconds wall / 5.656 seconds CPU after 51 Ipopt iterations,
with objective `3.175796576239704e-6` and a target endpoint of
`0.30000558574001657` rad. The worker then failed after 10.642 seconds total
while validating the saved controls clock. Peak sampled process-tree RSS was
209,035,264 bytes (about 199.3 MiB). No optimized-solution replay or per-unit
state-error comparison ran, and no later rung launched. A post-run process
query found no Python ladder worker. No retry was made.

The persisted `solution.sto` and `time_only_controls.sto` each contain 21 rows.
Their serialized clocks compare exactly (maximum absolute difference 0), as do
both saved control channels versus the corresponding columns in
`solution.sto` (maximum difference 0). Reloading `solution.sto` as a
`MocoTrajectory` also gives a clock exactly equal to the controls STO. Thus the
failure is the in-memory `MocoSolution.getTimeMat()` versus a STO round-trip;
that in-memory vector was not retained, so its exact mismatch or ULP error
cannot be recovered from this run. No tolerance is relaxed. The next fix should
make the persisted solution trajectory authoritative only after checking the
in-memory clock, states, and controls against it under a predeclared maximum of
8 float64 ULP per element. The audit records maximum absolute and ULP errors
per array, channel names, and hashes of both in-memory and persisted arrays;
the request fixes the ULP budget before the solve. The persisted 0 and 0.01
second endpoints remain exact, and the separately saved replay-input controls
must exactly match the persisted solution clock and control values. Tests now
accept a one-ULP perturbation and reject truncated arrays, changed controls,
non-finite data, and nine-ULP corruption. A read-only SDK check of the retained
persisted solution confirmed identical 21-row solution/control clocks and
control values; it cannot recover the original in-memory arrays. This TDD fix
has not been run as a ladder and requires review before another solve attempt.

At the authorized checkpoint, worker and harness SHA-256 values were
`3ff8275bdea8057c68a7220fdcb89933102ca584a5c9d058b8cd9f895219e30c` and
`d6374348fbee7acdcb44410facebac7bdf4ea1d139511a439cb66de3b37b6a4b`.
The request binds source
`6a098517bdba09b19b3eb99e4da7e795f27d10be8f42c0fca04fee20ce67e143` and native
profile `189c5d07c24b6b8c993aedba8cb794dbe82925bb6b368fef98be1327c2057fbf`.
Run files remain locally retained under
`experiments/native_moco_ladder_5d8fe120/plain/`; worker logs are excluded by
the repository's existing ignore rule:

| Artifact                 | SHA-256                                                            |
| ------------------------ | ------------------------------------------------------------------ |
| `request.json`           | `cdb44cdce3df26ccd161eade7f18f6da7d2b52de5e44b0e4d9a103e4543389de` |
| `progress.json`          | `9a1e4bba67309458f8ffe0f9fed7c275041b29979ba979c067659034e9c493db` |
| `solution.sto`           | `f38c5c8c4cd94d8a009ad043fff9bd7d36c21e4f03062092398e8e18ef5d1e5c` |
| `time_only_controls.sto` | `b5c65c34067a17e5964bf4e82d1e34268c7c40d747597657a2c98a70737ceb95` |
| `worker.stderr.log`      | `913716ed6d8041846013525c3f0172f4743a18bbc86aab5ae0827bd4a64a8608` |
| `worker.stdout.log`      | `a20d5039bd6859d95c7c8ee5a960a0ef42f11ef456063ff4ee84904507662f59` |

The ladder remains a synthetic software feasibility diagnostic. Even a future
successful solve and fresh replay would not qualify muscle physiology, the
captured swing, independent excitation parity, or production readiness.

The serialization-fidelity gate now measures exact ordered IEEE-754 binary64
representable-value distance, canonicalizing `-0.0` and `+0.0` to distance
zero. This avoids treating a spacing-normalized delta as an exact ULP count at
binade boundaries. Pure tests cover positive and negative power-of-two
crossings, the 8-step acceptance/9-step rejection boundary, and subnormals.
This correction has not been applied to the retained solve or used for another
optimizer run; the earlier failed run remains immutable evidence.

## Bounded Run at Exact-ULP Checkpoint `a1a2767e612188c6994952bb7b6aff655d78f7cd`

One serial run was authorized at this exact committed checkpoint with the
existing 30-second per-rung wall limit, solver parallelism 0, native thread
limits 1, sampled 1 GiB process-tree RSS threshold, and stop-on-first-nonpass.
The Python 3.13.5 supervisor used psutil 7.0.0; its isolated worker used
Python 3.12.10 and OpenSim `4.6-2026-06-22-85aaf64`. The `plain` solve
succeeded in 5.149 seconds wall / 5.063 seconds CPU after 51 iterations, with
objective `3.175796576239704e-6`. The optimized base-coordinate endpoint was
`0.3000055857400166` rad for the requested `0.30000558574001657` rad target
stored in `request.json`.

The corrected persisted-trajectory audit passed before replay. Exact ordered
float64 distances were 1 ULP for time, 2 for states, and 3 for controls; their
maximum absolute errors were respectively `8.673617379884035e-19` seconds,
`5.551115123125783e-17`, and `6.938893903907228e-18`. All are within the
predeclared 8-ULP maximum. Saved controls matched the persisted solution's
clock and control values exactly, and the exact 0 and 0.01 second endpoint
checks passed. The independent fresh native replay ran, but failed the
unchanged per-unit state agreement limits: maximum coordinate error was
`1.5042460832503934e-8` rad (limit `2e-7`), speed error
`4.243097622228572e-6` rad/s (limit `1e-7`), activation error
`4.2118714943707725e-4` (limit `1e-8`), and fiber-length error
`5.393972289921556e-6` m (limit `1e-7`). The replay endpoint was not retained
because state comparison failed before endpoint validation. No success receipt
was written, no later rung launched, and no retry was made. The supervisor
recorded peak sampled worker-tree RSS of 227,803,136 bytes and returned after
9.936 seconds; a post-run process scan found no remaining ladder worker.

The round-trip array identities and errors were:

| Array    |      Max absolute error | Max ULP | In-memory SHA-256                                                  | Persisted SHA-256                                                  |
| -------- | ----------------------: | ------: | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| Time     | `8.673617379884035e-19` |       1 | `7162018ce4581b12cc5188c7dced3235df6f0a7a8392156d3783408a28cc189c` | `65b6042fb2ab0a5d3d5d256e77bf58d47efbabd4861c20cc2dcf0f7c9247fdb9` |
| States   | `5.551115123125783e-17` |       2 | `7dfd03b88dac2a24e6cbf6204ef51573dbcc9b8a38e006dda56bef3c75c29b78` | `01ed31a109026c7c45af14186d6b8483a96e33da1a06a4ad54e10406ef3157a2` |
| Controls | `6.938893903907228e-18` |       3 | `1ab4ed66df04af4dd7f0b319d2392774fd20d6b2b4d445d8e4491df26256e772` | `0059dc7ca34b5d5f11d7e8e226e480f4d7efd2b99535829447a357c7d30cd3c9` |

This run confirms a successful synthetic solve and bounded STO round-trip,
but not successful fresh replay. It provides no physiological, captured-swing,
or production-readiness qualification. All run files remain locally retained
under `experiments/native_moco_ladder_a1a2767e61/plain/`:

| Artifact                                 | SHA-256                                                            |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `request.json`                           | `85023064b3e1ca8c8c03c73ded091fdff19633284b61337ae35129b8f2b0536e` |
| `progress.json`                          | `502a61d373a3f800877cbe3a1686f6cd7b0612010068f7ec6262c4c41bcdb5cc` |
| `solution_serialization_audit.json`      | `c178a5366a235e9db88415b135e5b216c951852c678bd49d94dfaaeabe42ab38` |
| `solution.sto`                           | `9c96137a20e5ac014ed5e90685967e7fb85a6e563cc870986e02a4367aaac851` |
| `time_only_controls.sto`                 | `b5c65c34067a17e5964bf4e82d1e34268c7c40d747597657a2c98a70737ceb95` |
| `source.osim`                            | `6a098517bdba09b19b3eb99e4da7e795f27d10be8f42c0fca04fee20ce67e143` |
| `worker.stderr.log`                      | `d2ad61fb9f2d516bbbff5266800e1ad8270cce4f3ee20289b9d51c9903fe8772` |
| `worker.stdout.log`                      | `48ba1cd5a89ded9433c6adb4075087403f7bfb85933a240ec1e60a20d4436f73` |
| `witness/witness_states.sto`             | `8524ace75d72795445b7a8bb7095cf63f41af9709452afee5285398c779f5f10` |
| `witness/witness_time_only_controls.sto` | `db0c14e4fbfb1fe9678fcb5df75b6e027ee1462922ff6dff0134dd6fde6f3e72` |

The executed worker and harness source hashes were
`b1a95f9aae245d6d04a9b4b91aae07fd29a3f43543797c0ca83eb8e2f137fe43` and
`62ed9de2d82713ed36be87decfc83a4a3e63c9f2886ad4573e85d9fd0ed77a07`.

## Bounded Saved-Solution Replay Diagnosis

The request's full-precision target is `0.30000558574001657` rad; the stored
solution endpoint is `0.3000055857400166` rad, not a `0.300005` rad target.
The optimized and fresh-replay first rows both exactly match all ten declared
cold-start states. The independent controller-off replay below used only the
retained source, request, persisted solution, and saved time-only controls; it
did not construct a Moco study or invoke an optimizer.

The replay completed under a 30-second ceiling and sampled peak worker-tree RSS
of 209,420,288 bytes in 5.083 seconds. It reproduced the failed acceptance
status. Its complete 21-by-10 replay state matrix, 21-sample clock, absolute
per-state/per-time difference matrix, per-unit maxima and times, endpoint
residuals, and source/request/solution/control bindings are retained in
`experiments/native_moco_ladder_replay_a1a2767e61/diagnostic/fresh_replay_diagnostic.json`.
The artifact SHA-256 is
`a9fd0c8dac9dc4065841bebc4dcafb8ea07cf613d6b5b9274a843e9901240b14`; its
replayed-state matrix SHA-256 is
`c1253c5670193a1e249b5f107c9742f1a7553e9eb18c7abe2f945e126970ac2b`, and the
absolute-error matrix SHA-256 is
`bf5a0a163b62c1c939812ae8bfb74befa4a579a52c5094b35f5f25285eb0e8b2`. The
worker exited 1 after persisting the failed audit because the diagnostic entry
point also invoked the fail-closed acceptance check; the supervisor did not
timeout, and a process scan found no remaining worker. This exit does not
change the retained audit's explicit `failed` status.

The diagnostic worker source SHA-256 was
`8ce2ae5fcd56096b48a037eaf33a2ef2bfe77230e3d0a10bde3de4e07d83f6e2`; its
stdout and stderr hashes are `3492492046068ec50b66497cb6c2a24de666c9d621c3cdeee97dbc467bc04311`
and `e708e31ca6f1000beb84394c0f7b1642a87651abdb6aa5c2a58151524d9625f9`.
The request, solution, controls, and source hashes in the diagnostic match the
original run: `85023064b3e1ca8c8c03c73ded091fdff19633284b61337ae35129b8f2b0536e`,
`9c96137a20e5ac014ed5e90685967e7fb85a6e563cc870986e02a4367aaac851`,
`b5c65c34067a17e5964bf4e82d1e34268c7c40d747597657a2c98a70737ceb95`, and
`6a098517bdba09b19b3eb99e4da7e795f27d10be8f42c0fca04fee20ce67e143`.

All named-state errors are zero at time 0. At 0.0005 seconds, flexor
activation already differs by `1.202676736851635e-4`; it peaks at
`4.2118714943707725e-4` at 0.001 seconds and remains `3.6461925940936557e-4`
different at 0.01 seconds. Base-coordinate speed error grows from
`1.8516945245736823e-9` rad/s at 0.0005 seconds to `4.243097622228572e-6`
rad/s at the final sample. Flexor fiber-length error grows from
`3.2634289141852957e-7` m at 0.0005 seconds to `5.393972289921556e-6` m at
0.01 seconds. Extensor activation and fiber-length maxima are only
`9.995776859006433e-9` and `4.90473939240843e-10` respectively. Thus the main
divergence begins after the identical initial sample and is concentrated in
the flexor activation, then its fiber length and base speed.

The replay endpoint is `0.30000560078247746` rad, with target residual
`+1.5042460888015086e-8` rad, within the unchanged `2e-7` endpoint limit. The
Moco solve reports an equality-constraint violation of
`2.0528726148464835e-9`, below the configured `1e-7` optimizer constraint
tolerance. That scalar optimizer feasibility result does not guarantee
componentwise agreement with forward integration: the observed speed,
activation, and flexor fiber errors exceed their separate replay limits. No
quantitative attribution to the solver tolerance is established.

The saved solution and control tables have identical 21-knot clocks, with
exactly matching control samples. The constrained replay provider adds only a
`PrescribedController` using `PiecewiseLinearFunction`; its integration path
checks sampled applied excitations against saved knots with zero relative and
`1e-12` absolute tolerance, and that check completed. The Moco problem fixes
all ten initial continuous states to the declared values, bounds controls in
`[0, 1]`, has no source controller, and uses a 10-mesh-interval Moco grid whose
solution contains 21 rows. These checks make an initial-state, source identity,
or gross control-sample mismatch unlikely. The retained replay does not record
the applied-control sample matrix, so exact rather than tolerance-bounded
applied-control equality cannot be asserted. Whether the remaining divergence
comes from the Moco transcription's internal control/state interpolation or
from its discretized muscle-state defects is unresolved; no threshold or model
was changed to force agreement.

### Read-Only Control-Bound and Midpoint Check

The persisted controls were inspected without another replay or solve. The
first five saved knots are:

| Time (s) |                  Flexor |              Extensor |
| -------: | ----------------------: | --------------------: |
|        0 |     0.09544759030496992 | 0.0008312441911413515 |
|   0.0005 |     0.04772462501999535 | 0.0007238851585008595 |
|    0.001 | 0.000001659735020775346 | 0.0006165261258603676 |
|   0.0015 |    0.005345077920641525 | 0.0006317446484905896 |
|    0.002 |     0.01068849610626228 | 0.0006469631711208116 |

The native OpenSim 4.6 model reports `min_control=0.01` and `max_control=1.0`
for both muscles; its XML contains those same limits and no activation-limit
fields. The Moco problem instead sets control bounds to `[0, 1]`. Over the
21 saved samples, flexor controls fall below the source minimum at 18 samples
(minimum `1.659735020775346e-6`); extensor controls fall below it at all 21
samples (range `0.0006165261258603676` to `0.0008868723599153841`). This is a
confirmed declaration mismatch and a candidate replay-divergence mechanism;
the retained evidence does not establish how the muscle's native minimum
control affects force or activation during this replay.

The 21 rows represent 10 mesh intervals and their midpoints. At the first five
interior samples, each saved midpoint equals the average of its neighboring
mesh-endpoint controls to within `2.7e-18` for either muscle. The OpenSim 4.6
Python binding exposes `interpolate_control_mesh_interior_points`; a
read-only `MocoCasADiSolver` instance reports `True`. The ladder's solver setup
does not override this setting, and the SDK documentation identifies the
setting as the default recommended linear interpolation. The API does not
expose a setting named `enforce_control_midpoints`. The realized control
matrix from the prior replay was not retained, although its in-process sample
check passed within `1e-12` absolute tolerance. These findings do not attribute
the state mismatch to either source minimum-control handling or collocation.

The original `a1a2767e61` solve remains historical: its request explicitly
used `[0, 1]` control bounds. A follow-up harness change now derives each
channel's Moco bounds from the unchanged source muscle's `min_control` and
`max_control`, binds the bounds and source hash into the request, and rejects
nonfinite, reversed, out-of-range, or incomplete channel declarations before
solver setup. The synthetic constant `0.05` witness must lie inside those
source-derived intervals. A native SDK regression uses altered test-only
source bounds `[0.02, 0.8]` and verifies those exact values reach the Moco
binding calls. This addresses a proven configuration mismatch, but is not
evidence that the mismatch caused the earlier replay errors. The earlier
replay artifact has no persisted applied-control matrix; the code records such
samples for future runs, but this historical artifact is not being rerun or
rewritten. No new optimizer run has been authorized yet.

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
`9e0ff02cb67ebb90eb9d6c7dcfe1f0de98b15d7911f618317460265148f25a23`.
