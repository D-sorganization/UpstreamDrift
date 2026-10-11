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
`0.3000055857400166` rad for the requested `0.300005` rad target.

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
