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

The ladder remains a synthetic software feasibility diagnostic. Even a future
successful solve and fresh replay would not qualify muscle physiology, the
captured swing, independent excitation parity, or production readiness.
