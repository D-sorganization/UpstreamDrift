# Lower-Cost Agent Execution and Turnover

## Explicit Pause and Engine Transfer State

The user explicitly requested stopping this agent and transferring to a cheaper
agent. No further work or optimizer should be launched by the outgoing agent.
Root checkpoint before this pause note is9fa847afc, pushed and clean; this
pause document's checkpoint is SELF. Run20 PID2348439 is absent and its session
exited0. All three delegated agents are completed; all their reported jobs ended.
No claim is made about unrelated Gemini processes owned by other agents.

| Lane               | Saved State                                                                            | First Remaining Task                                                                           |
| ------------------ | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Pinocchio / Shared | Root branch; run20 independently rejected and fully archived                           | Integrate tested executor with solver/cache and prove equivalence                              |
| MuJoCo             | Native rigid adapter and bundle factory qualified on baseline; integrated root         | Qualify final fitted candidate when available; stock mj_step remains separate                  |
| Drake              | Native rigid adapter qualified on baseline; extended scalar-rate discrepancy preserved | Resolve coordinate-rate qualification scientifically; develop reviewed smooth-path initializer |
| Simscape           | R2025b baseline evidence preserved; Gemini owns its worktree                           | Coordinate source identities and replay eventual final same-input candidate in R2025b          |
| OpenSim            | Epic10003 and OS-0..6 plan only; branch docs/10003-opensim-matching-epic at9ab8165bc   | OS-0 runtime and native-model inventory, not immediate fitting                                 |

OpenSim's own HANDOFF has older run16 status; its pointer to the TOP root native
checkpoint is authoritative for current cross-engine state. Do not restart its
historical run17 suggestion. Its status showed a metadata/line-ending refresh
with no content diff; no user content was discarded. Drake and MuJoCo worktrees
have no pending content changes. All relevant branch commits are pushed.

## Copy-Paste Assignment

Continue the golf matching program in
C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native on
feat/9967-native-simscape-pinocchio. Read AGENTS.md, CLAUDE.md, this full turnover,
and the TOP of NATIVE_PORT_CHECKPOINT_20260911.md before doing anything.
Check actual Git state and coordinate/transfer the issue9967 lease.

The outgoing agent is paused. No optimizer is live; run20 is completed,
independently audited, unaccepted, and archived. Do not restart it, repeat
completed engine ports, or treat baseline equivalence as full-swing acceptance.
The final goal remains the full1.8138888889 s C3D swing driven continuously by
one global sixth-order effort polynomial per channel, with native engine
behavior consistent with Simscape R2025b.

Third Task's integration and fixed-input full assembly receipt are complete.
The guarded run20 derivative driver submitted the existing tested batch seam to
the shared solver, and workers=0/2 returned the same two-evaluation frozen
0.85 s result byte-for-byte except timing telemetry. Read
`../native_parallel_performance/batched-solver-qualification-9967-24.json` and
the performance handoff before using it. Preserve original identities and
immutable runtimes; do not build another fitter. Next, follow the reviewed
smooth native closure-feasible target-path/torque-initializer stages. If a
mathematical choice is unresolved, report evidence and request review before a
broad expensive run.

The path stage now has an engine-owned residual probe:
`NativePinocchioModel.closure_residuals`. Use it as the closure oracle for
constrained marker poses and trajectory checks. It probes with zero primitive
efforts and never projects, resets, or modifies forward replay state. Start by
qualifying it against a known constrained ControlTower state; then use local
marker-pose solves as seeds for a smooth trajectory. Do not infer trajectory
feasibility from independent poses alone.

The ControlTower single-state qualification is complete in
`native_evidence/closure_probe_9967_22/receipt.json`: direct constrained
dynamics and the probe agree exactly at the archived run19 candidate for both
its qd and zero qd. The next required qualification is therefore a _trajectory_
test across smoothed constrained marker-pose seeds, including pose, rate and
acceleration closure. Do not skip that test for a torque fit.

Use `NativeConstrainedPoseOracle` with the existing shared
`fit_marker_pose`; it is the only permitted bridge from the native model to
the static constrained-pose stage. Pass the exact candidate coordinate order,
captured marker-body attachments and offsets. Save one independent receipt per
frame, including observed mask, marker RMS, SLSQP status and weld residual. The
adapter does not make independent frames a valid trajectory.

The initial-frame smoke receipt is recorded in
`native_evidence/marker_pose_seed_9967_23/receipt.json`: 25 observed markers,
2.89808e-13 m RMS and5.54112e-13 m weld residual. It uses the archived run19
initial state, so it verifies native identity and adapter wiring only. Next use
the same immutable candidate/model/attachment identity for sparse transition
frames, warm-starting each solve from the preceding feasible pose and recording
any reachability or branch failure before trajectory smoothing.

Use TDD, DbC, LoD and DRY. Keep all acceptance gates and missing-marker masks.
Never silently change geometry, add feedback/state resets, or relax criteria.
Use one bounded fit at a time after qualification; independently forward replay
returned controls from original q0/qd0. Preserve all raw hashes and test receipts.
Keep canonical and engine handoffs/development log current in each implementation
commit, push through normal checks, and coordinate material changes on issue9964.
OpenSim implementation is separate under epic10003 and begins at OS-0.

## Current Override: Run 20 Is Finished

Run20 session43223 exited0; independent final replay and bound audit are complete.
No optimizer is live. The older live-run instructions below are retained for
reproduction only. Do NOT restart the run or repeat the completed audit.
Read the top native checkpoint and `native_evidence/ms_fit_9967_20/` for results:
whole29.941228 mm, terminal98.417488 mm, continuity1.202905e-4; unaccepted.
Third Task's shared solver/cache seam is complete in commit81e1784c4: use
`MultipleShootingOptions.segmented_forward_batch` to submit only ordered cache
misses. Two new TDD tests prove repeated candidates do not resubmit cached
windows, analytic Jacobian assembly remains parent-owned, and malformed batch
results fail. The trusted native worker adapter now exists in
`native_sensitivity_batch.py`; the runner needs a bounded binding and fixed-input
ControlTower qualification before it may use workers=2. The final candidate
19193da3... is a documented experimental seed, not an accepted swing.
No new fit is running or preselected.

## Mission and Current Truth

Continue the existing native golf matching program; do not rebuild the model
or restart completed epics. Deliver a repeatable full 1.8138888889 s C3D swing
using one global degree-six polynomial per effort channel. MATLAB execution,
save and acceptance release is R2025b. Cubic stitching may be an initializer
experiment only; it is not the final required control representation.

Native Pinocchio, MuJoCo and Drake baseline adapters already exist and have
same-input qualification evidence through the specified 0.8 s baseline. They
preserve 27 scalar coordinates, 31 solids, original inertias/gravity, and the
6D right-hand grip closure. URDF requires its sidecar. MuJoCo and Drake use
actual engine dynamics with a custom rigid constrained solve and shared DOP853;
stock mj_step and Drake SAP are not qualified equivalents.

The full swing is NOT matched. Run19 reached 0.85 s, with whole RMS 30.791102 mm
and terminal RMS 99.988650 mm. It is unaccepted. A current experiment, run20,
is live; its callback cost alone cannot establish improvement. A cheaper agent
can execute the bounded tasks below. A guaranteed successful full-swing route
has not been established; do not conceal that by relaxing acceptance criteria.

## Start Here: Ownership and Exact Workspace

Repository: D-sorganization/UpstreamDrift. Root worktree:
`C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native`.
Branch: `feat/9967-native-simscape-pinocchio`. Source/evidence checkpoint before
this turnover: `aa85fd004` (includes c3044946a studies). This document's commit
is SELF. PR: not created. Do not push to main or discard unrelated changes.

Read, in order:

1. This worktree's AGENTS.md and CLAUDE.md, then agent_context/README.md and
   USAGE.md for any implementation discovery.
2. TOP of NATIVE_PORT_CHECKPOINT_20260911.md: authoritative live-job state.
3. MULTI_ENGINE_NATIVE_PROGRAM.md: engine scopes and ownership.
4. Relevant lane HANDOFF.md under drake_native_matching,
   mujoco_native_matching or native_parallel_performance.
5. Existing shared source and tests before adding an interface.

```powershell
Set-Location C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native
git status --short
git log -5 --oneline
Set-Location C:/Users/diete/Repositories/Repository_Management
python3 -m scripts.check_agent_claim --repo UpstreamDrift --issue 9967
```

Coordinate with the active owner. If the lease is your resumed session, renew
it; if another agent holds it, ask that agent to transfer the lane before edits.
Use the documented post_agent_lease CLI and your real session id. Do not create
a competing optimizer. User expressly authorized this program across lanes.

## First Task: Observe the Existing Run

Run20 is the only optimizer. Codex session43223 is convenient within this
conversation; cross-thread handoff must use the host/process/output identity.
Host: controltower. WSL distro: ControlTower-Runner. PID2348439. At the last
inspection it was live at 10:05 elapsed, evaluation27 cost30.01935414.

```powershell
ssh controltower wsl -d ControlTower-Runner -- ps -p 2348439 -o pid,etime,args
ssh controltower wsl -d ControlTower-Runner -- tail -n 1 /mnt/c/Users/diete/native-ms-fit-9967-20/evaluations.jsonl
ssh controltower wsl -d ControlTower-Runner -- ls /mnt/c/Users/diete/native-ms-fit-9967-20
```

Verify args as well as PID: PIDs can be reused. A timeout is not termination.
If the process is absent, inspect returned.json and actual logs; absence alone
is not success. Never launch this command as a restart while the process lives:
`run_native_ms_bound_continuation_9967_20.py`. Preserve existing output files.
The immutable runtime is `/home/dieterolson/native-ms-pilot-9967-20` and Python
is `/home/dieterolson/simscape-pinocchio-9967/.venv/bin/python`.

Run20 uses horizon0.85, nodes0.2/0.4/0.6/0.7/0.8/0.85, basis duration0.8,
max_nfev100, max_iterations60, equality_tolerance1e-7 and node_bound0.05.
It widens exactly71 saturated correction intervals +/-2 to +/-4. Other bounds
and run19 variable scales remain unchanged. No physical actuator limit was
inferred. Source b5b1c382... and reconstructed seed b9d610b4... are distinct.
Strict source-state rate parity FAILED; own derivatives and charts passed.
Do not relabel this an exact restart or a single-factor experiment.

## Second Task: Independent Terminal Audit

When run20 terminates, copy its entire raw directory to a NEW local checkpoint
path under `C:/Users/diete/Repositories/simscape-tour-checkpoints`. Preserve raw
JSON bytes before formatters. Record SHA256 for model, target, returned candidate,
config, scripts, runtime manifest and any snapshot selected for analysis.

Existing independent marker replay tool on ControlTower:
`C:/Users/diete/benchmark_native_marker_visual_9967.py`. Verify its CLI with
--help before invocation; it accepts --model, --candidate, --target, --output
and --trajectory-output. Use the immutable runtime20 PYTHONPATH and explicit
single-thread environment. Inputs are native_geometry_spec_9967.json,
driver_marker_payload_9967.json and run20/returned-candidate.json. Write NEW
output names, such as independent-replay-final.json and
independent-trajectory-final.npz, inside the completed run directory only if
these names do not already exist. The replay must start from original q0/qd0.

Example command structure (replace NEW names only after checking absence):

```powershell
ssh controltower wsl -d ControlTower-Runner -- env PYTHONPATH=/home/dieterolson/native-ms-pilot-9967-20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/dieterolson/simscape-pinocchio-9967/.venv/bin/python /mnt/c/Users/diete/benchmark_native_marker_visual_9967.py --model /mnt/c/Users/diete/native_geometry_spec_9967.json --candidate /mnt/c/Users/diete/native-ms-fit-9967-20/returned-candidate.json --target /mnt/c/Users/diete/driver_marker_payload_9967.json --output /mnt/c/Users/diete/native-ms-fit-9967-20/NEW-replay.json --trajectory-output /mnt/c/Users/diete/native-ms-fit-9967-20/NEW-trajectory.npz
```

Adapt the existing read-only bound audit
`simscape-tour-checkpoints/audit_native_constrained_bounds_9967_19.py` to a NEW
run20-specific script. Change its explicit run/output paths and verify it reads
actual run20 config bounds/scales and initial-candidate.json; never assume all
bounds are +/-2. Do not change dynamics, candidate or output on failure.
Archive the exact modified audit script with its receipt.

Generate an overlay using existing
`native_evidence/reproduction/plot_native_marker_replay.py --trajectory ... --output ...`.
Set MPLBACKEND=Agg and repo-root PYTHONPATH as needed. Open the generated image
with view_image. Report whole, early, terminal, club and yaw metrics, per-marker
error, closure, full scaled continuity defects, segmented-to-continuous terminal
gap, solver convergence and every gate. A decreasing optimization cost alone
is not a reason to extend the horizon.

## Acceptance: Do Not Improvise

| Quantity                              | Current Gate                                   |
| ------------------------------------- | ---------------------------------------------- |
| Whole Observed Marker RMS             | <=25 mm                                        |
| Early 0–0.6 s RMS                     | <=12 mm                                        |
| Terminal Marker RMS                   | <=35 mm                                        |
| Terminal Club RMS                     | <=60 mm                                        |
| Wrapped Pelvis Yaw Percentage Error   | <=5%                                           |
| Maximum Full Scaled Continuity Defect | <=1e-4                                         |
| Optimizer                             | Converged, not iteration/evaluation limit      |
| Continuous Replay                     | Finite, original initial state, observed masks |

Yaw uses wrapped angle error divided by max(abs(target angle in degrees),1),
times100. Preserve existing implementation. The pointwise terminal gap is
also reported; do not silently turn it into a new acceptance gate. Strict
same-input scalar-rate comparison at1e-4 is a separate engine/restart gate.
Its documented failures near the shoulder chart remain failures even when
physical marker differences are tiny. Do not lower accuracy or rename a failed
qualification to make a report green.

All654 capture frames remain in scope. Last two frames lack club observations;
missing metrics must be null/None, not zero. Never trim the swing to hide error.
The final model must be continuously forward driven, without target state
resets, prescribed motion, hidden PD corrections or independent segment clocks.

## Third Task: Integrate the Parallel Executor

The MuJoCo agent completed native_window_executor.py and five tests in
621efab7a, integrated into this root checkpoint SELF. Root reran all five tests:
pass. Native isolated qualification reports 21.30037 s sequential versus
11.18647 s executor, with exact primal and full Jacobian agreement. Read the
performance HANDOFF and executor-report.json. No solver integration is done;
that remains the next bounded implementation. Do not recreate the executor. The prior benchmark already found 20.18865 s versus
10.99769 s for six windows, with every primal/Jacobian entry identical. It is
one sample, not a promised full-fit speedup.

Use the existing shared multiple-shooting evaluator and cache. Keep retraction,
residual/defect assembly and SLSQP in the parent process. Workers evaluate
independent windows only. Submit immutable internal requests, preserve ordering,
keep absolute time and the global basis, propagate indexed failures, clean up
workers, and retain sequential fallback. Never deserialize untrusted pickle.

Required sequential implementation gates:

1. Review executor tests and reproduce their red/green evidence; ensure repeat
   batches, errors, cancellation/cleanup and closed-use contracts are covered.
2. Add failing integration tests showing assembled residuals, gradients and
   complete continuity/marker Jacobians are identical to sequential evaluation.
   Test cache hits do not unnecessarily submit another batch.
3. Implement only the optional execution seam; reuse existing calculations.
4. Run focused tests, Ruff and mypy, then normal repository commit/push hooks.
5. Stage a NEW immutable runtime. Run one fixed-input native equivalence test,
   including assembled output and repeated batches, before optimizing.
6. Only after run20 is terminal and independently classified, choose one bounded
   follow-on trial. Freeze its inputs for a performance comparison; if changing
   seed/bounds too, explicitly state that fit quality is a separate experiment.

## Fourth Task: Better Initialization Requires Judgment

Read drake_native_matching/REACTION_IDENTIFICATION_FEASIBILITY.md. The closed
loop permits eliminating reaction forces by projecting dynamics into null(J).
This is not ordinary tree inverse dynamics and does not set reactions to zero.
The known-baseline study identified189 sextic controls and reproduced markers
closely, but failed strict scalar-rate reconstruction. It is a diagnostic only.

The next prerequisite is a smooth native kinematic trajectory matching C3D
while satisfying position, velocity AND acceleration closure, with a continuous
coordinate branch. Independent static poses and an Euler-angle spline do not
suffice. Existing static continuation attains about35.8 mm at0.85 s, but naive
interpolation violates closure. Do not spend a large solve budget identifying
torques from an unqualified noisy path.

Implement trajectory fitting in small TDD stages: synthetic closed-loop path
with known derivatives; observed-marker masks; continuous charts through the
shoulder region; constraint checks between samples; only then actual target
segments. Obtain review of the mathematical formulation before a broad C3D run.
Use the existing effort design and Bernstein conversion for identification,
report rank/conditioning and held-out residuals, then forward replay the single
polynomial from original q0/qd0. It may seed the existing forward optimizer;
it never replaces forward acceptance. Exact early polynomial values across a
continuous interval would fix the polynomial everywhere: use explicit weighted
priors/tolerances, not an impossible demand for independent late coefficients.

## Stop and Escalate Conditions

Pause dependent work and report exact evidence if model/capture identity differs,
noisy target derivatives violate closure, native APIs or coefficient conventions
are ambiguous, Jacobian checks fail, a worker changes outputs, a model length
change is proposed, or the optimizer plateaus despite justified bounds. Do not
start another blind long solve, add a hidden controller, relax gates, fabricate
physical limits or edit a live runtime. Preserve diagnostics and propose one
specific next test. A completed budget is a checkpoint, not scientific success.

Geometry calibration creates a new model identity and requires mass/inertia,
marker-frame and native parity requalification. Baseline parity does not certify
new fitted controls; verify the final candidate in native MuJoCo, Drake and
R2025b with identical inputs. OpenSim remains a separate staged epic#10003;
read its owned worktree HANDOFF before OS-0 implementation.

## Saving and Coordination at Every Step

Use conventional incremental commits on this topic branch. Refresh the top
native checkpoint, relevant lane HANDOFF, canonical docs/development/HANDOFF.md
and DL-#9967 in DEVELOPMENT_LOG.md in implementation commits. Record exact
commands, outcomes, process identities, source/config hashes, raw artifact paths,
known failures and one executable next action. Keep original ZIP bytes tracked
explicitly if ignored; verify git ls-files contains intended archives. Never
commit **pycache**, environments, credentials or giant duplicate trajectories.

After normal push checks pass, record HEAD and clean/dirty status. Copy the
current native checkpoint to the existing owned Gemini breadcrumb:
`C:/Users/diete/.gemini/antigravity/brain/9efc50d9-1e1a-4d78-a883-cf8c5e9f7a59/CODEX_NATIVE_PORT_CHECKPOINT_20260911.md`.
Post material coordination to UpstreamDrift issue9964 using gh and a body file.
Do not edit Gemini's source or runtime. Its diffstep candidate currently lacks
sufficient provenance for a same-input comparison; ask its producer for those
identities instead of silently translating it.

Before handing off, give the user the exact commit, current best independently
replayed candidate and inspected visual, active-job handles, unresolved gates,
and the next bounded task. Leave the full goal active until actually achieved.
