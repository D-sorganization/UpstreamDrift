# Matching Review and Execution Handoff

## Decision and Evidence Boundary

Review at 22:02 PDT on September 11, 2026. Inspected Gemini commit
`58df3af67b5cef1cdea335253df672cd1c8771d0`, its receipts and tests, and the
actual runner scripts on DeskComputer. No fitting MATLAB worker was observed
on DeskComputer at inspection; only MATLABConnector was listed. Recheck before
launching. The full matching goal is incomplete. R2025b is mandatory.

There is real progress in early tracking, candidate preservation and native
geometry reconstruction. The latest optimization is a regression, and the
claimed multiple-shooting experiment was not actually multiple shooting.
Fix execution identity and acceptance before spending another long run.

## Current Candidate Ledger

Numbers below are reported native replay metrics, inspected in committed
receipts, not a fresh independent simulation performed during this review.

| Candidate       | Horizon |  Whole RMS |  Early RMS | Terminal RMS | Club Cluster Terminal RMS | Gates |
| --------------- | ------: | ---------: | ---------: | -----------: | ------------------------: | ----: |
| Run 06 eval 79  |  0.75 s |  23.859 mm |   9.852 mm |    94.690 mm |                115.979 mm |   3/5 |
| 0.80 s eval 559 |  0.80 s |  43.228 mm |   9.851 mm |   195.214 mm |                283.236 mm |   2/5 |
| Deliverable 3   |  0.80 s | 265.583 mm | 175.207 mm |   400.411 mm |                522.925 mm |   0/5 |

Do not compare different horizons as if they were identical objectives.
The club metric aggregates six shaft/head markers selected by label, and is
not necessarily the physical clubhead-center error. Preserve the legacy gate,
but add explicit shaft, head-cluster and clubhead-center metrics.

The cold-replay script labels the last logged evaluation (775) as the optimizer
final. That is incorrect candidate selection: finite-difference evaluations
are not necessarily the returned optimum. The original report's `final_stage`
has whole RMS 44.042 mm, terminal 200.541 mm and club 274.539 mm; the receipt's
209.196 mm result belongs to the last evaluated parameter vector. Reconcile
returned parameters, converted native coefficients and saved MAT replay by hash.

## Findings Requiring Correction

1. **The executed runner is still single shooting.**
   `DeskComputer:C:/Users/diete/SimscapeTour9921/run_ms_optimization_080s.py`
   extracts a 0.60 s state but never uses it as a decision variable or segment
   initial state. Its optimizer has only 189 effort variables, calls
   `run_continuous_forward(x, 0.80)` and has no defect residuals. It never calls
   the new `fit_multiple_shooting`. The receipt therefore does not test the
   advertised architecture. Archive it as a failed single-shooting experiment.
2. **Warm-start time scaling changes the motion.** The same eval-79 Bernstein
   controls, originally defined on 0.75 s, are passed to
   `bernstein_to_simscape(..., duration_s=0.60)` and then 0.80. This changes
   physical effort versus absolute time. Separate `basis_duration_s` from
   integration end time. Re-express coefficients algebraically when changing
   basis duration; prove equality of effort and its derivatives before replay.
   The original 0.80 s fit records basis duration 0.80, despite prose claiming
   all existing candidates use a full-swing 1.814 s basis.
3. **Acceptance is incomplete in the new library.**
   `src/shared/python/motion_matching/multi_shooting_fit.py` accepts optimizer
   success plus small defects, irrespective of continuous replay marker error
   or NaNs. `segmented_rmse_m` is hard-coded zero. Terminal/pelvis options and
   `state_dim` are unused. Its two tests exercise validation and constant
   acceleration, not a nonconstant sextic, golf closure, or native restart.
4. **Segment boundary semantics are unsafe.** The callback receives observation
   times rather than explicit start/end times. Off-grid nodes and empty windows
   cannot reliably integrate to the declared boundary. Both windows include
   shared observations. Raw subtraction mixes m, rad, m/s and rad/s and is not
   suitable for arbitrary manifold coordinates. Native closure and velocity
   consistency at shooting nodes are not enforced by the generic fitter.
5. **Remote residual and bounds need repair.** The early-retention residual is
   appended only above a threshold, changing residual dimension across it.
   Always append a fixed-size hinge term. The runner applies a common +/-150
   range to force and torque controls, mixes mm penalties with m residuals,
   duplicates metrics, and checkpoints history only at termination. Use declared
   physical scales, shared metrics and incremental immutable candidate saves.
6. **The transition explanation is not established.** The diagnostic compares
   velocities from two different candidates at 0.75 s. That does not demonstrate
   a discontinuity in either trajectory. It performs no constrained kinematic
   floor fit, sensitivity-rank study or controlled bounds experiment. Global
   polynomial coupling is real, but it does not prove sextic infeasibility or
   unavoidable ringing. `xtol` is termination, not a proof of optimality.
7. **Check native joint singularities.** LSInputY is about -1.8 rad near the
   transition and reaches about -3.486 rad at 0.80 s in the diagnostic. Examine
   the complete native trajectory and angular-velocity Jacobian, including
   branches, before interpreting large coordinate rates as body whiplash.
   Native Gimbal joints are singular at second rotation +/-90 degrees:
   [MathWorks Gimbal Joint](https://www.mathworks.com/help/sm/ref/gimbaljoint.html).
   This is a plausible diagnostic lead, not a proven cause in this run.
8. **Cold replay provenance is incomplete.** The runner enables FastRestart,
   starts one MATLAB engine for all candidates and does not record/assert its
   release. Obtain an independent fresh-process, FastRestart-off replay for
   acceptance, with model/workspace/source/coefficient hashes. Do not silently
   relabel the existing receipts as invalid numerics; qualify their limits.

## Engine Equivalence Status

The native-derived Pinocchio implementation in this worktree preserves all 27
scalar coordinates, 31 uncommented solids and the cut right-hand weld restored
as a 6D constraint. Actual ControlTower Pinocchio execution matches the saved
initial 25-marker Simscape fixture with RMS 2.895e-13 m. A zero-effort,
zero-velocity free-fall invariant passes with maximum acceleration discrepancy
8.786e-13. These are initial FK and invariant checks only: no full native
force parity or continuous swing equivalence has been established.

Legacy cross-engine animations are not equivalence certificates. Prior review
identified missing forearm rotations, differing torso/spine order and reduced
or missing grip closure in surrogate models. MuJoCo, Drake and Pinocchio must
share native geometry, inertia, gravity, coordinate conventions, both grips,
initial state and physically equivalent efforts. Do not independently fit
different physics and claim the inputs are interchangeable.

## Next Agent Execution Prompt

You own completion of epics #9921 and #9964, coordinating the native port under
#9967. Continue from these files and existing candidate archives. Deliver a
repeatable, continuous forward-dynamics match using one global degree-six
polynomial per native input channel, with no target-state resets or tracking
controller in final acceptance. Maintain TDD, DbC, LoD and DRY. Use MATLAB
R2025b explicitly. Do not report completion from unit tests, animations or
optimizer termination alone. Execute the following stages in order.

### Stage 1: Repair Identity and Reproduce the Baseline

- Read applicable AGENTS.md/CLAUDE.md and current claims. Coordinate ownership
  with Gemini before changing its files; use an isolated checkout.
- Preserve eval79 and eval559. Store coordinate names, force/torque units,
  polynomial degree, basis family, time origin, basis duration, coefficient
  ordering, geometry, marker attachments, q/qd, input transforms and source
  hashes in one shared candidate schema. Reject missing identity fields.
- Add failing tests proving horizon changes do not change tau(t), nonconstant
  sextic evaluation uses absolute time in every segment, and exported MATLAB
  A..G coefficients agree with the source basis. Extend existing conversion
  helpers instead of introducing a second conversion implementation.
- Convert the preserved polynomial to a fixed full-capture basis using the
  exact capture end time, not rounded 1.814. Note that extension preserves an
  old polynomial but does not make its extrapolation a valid full-swing match.
- Replay eval79 and eval559 in fresh R2025b processes, no FastRestart. Compare
  actual returned optimum separately from last logged perturbation. Repeat
  selected inputs to measure numerical noise and tighter-tolerance sensitivity.
  Gate this stage on reproducibility before optimizing.

### Stage 2: Measure the Transition Feasibility

- Fit native closure-consistent poses to measured markers over 0.55-0.85 s
  with fixed geometry and attachments; measure achievable kinematic error.
  Use smooth pose/velocity regularization and multiple starts. This is a
  feasibility diagnostic and node initialization, never final forward motion.
- Export raw same-time q, qd, body angular velocities, joint limits, closure
  residuals and constraint-Jacobian singular values. Inspect gimbal rate-map
  conditioning and solver warnings. Do not diagnose from interpolated Euler
  rates alone. Establish units and angle unwrapping explicitly.
- Compare effort-bound activity and scaled sensitivity singular values.
  Use controlled perturbations at several sizes above native replay noise.
  Separate poor geometry, chart singularity, input bounds, insufficient basis
  flexibility and optimizer conditioning; report measured evidence for each.
- If geometry cannot fit, optimize bounded segment lengths in an outer loop,
  regenerate inertia and all engine artifacts, and freeze each geometry during
  an inner torque solve. Keep marker offsets fixed or explicitly regularized;
  unrestricted offsets must not hide geometry errors.

### Stage 3: Qualify a Fast Native-Equivalent Engine

- Continue native_assembly.py/native_spec.py and Pinocchio native_model.py.
  Compare native frame poses at multiple exact raw configurations through the
  transition; avoid comparing FK of interpolated q with interpolated markers.
- Audit active damping, limits, spring forces and compiled topology. Prove
  upstream input-to-generalized-effort mapping by virtual work and native
  single-channel force/torque pulses. Compare qdd at identical q, qd and efforts,
  then short free rollouts with step/tolerance convergence and closure checks.
- Implement constraint-consistent integration. A tree ABA simulation alone
  does not preserve the two-hand loop. Use actual native engine execution,
  not mocks, for equivalence gates. Derive tolerances from solver convergence
  and keep parity error well below the marker-fit error budget.
- Export the same canonical spec into MuJoCo and Drake, retaining forearm
  joints, native bushing order and grip closure. Verify mass/COM/inertia, FK,
  pulse response and continuous same-input rollouts independently. Declare
  any approximation as a separate model variant. Do not block native Simscape
  diagnostic work on finishing all three ports.

### Stage 4: Execute Real Multiple Shooting

- Fix the acceptance, metrics, contracts and boundary issues above with red
  tests: poor/NaN continuous replay must reject; nonzero segmented RMS must be
  reported; empty/off-grid windows handled; invalid dimensions/bounds rejected;
  all required early/terminal/club/yaw gates exercised; closure-inconsistent
  nodes rejected; final native execution demonstrably consumes the new solver.
- Optimize theta plus independent intermediate states with closure-consistent
  reconstruction. Specify segment start/end separately from observation times.
  Use physically scaled state differences and sparse defect Jacobians. Enforce
  defects using equality constraints or a documented augmented-Lagrangian
  continuation. A fixed penalty alone is not proof of continuity.
- Start with [0,0.60] and [0.60,0.80], inserting a 0.70 node if justified by
  conditioning. Every segment uses the same absolute-time sextic. Connect
  physical state, including velocities and any additional native dynamic state.
  Record actual initial/end states to prove restart fidelity.
- Preserve early RMS <=12 mm while reducing terminal and club errors. Current
  gates are whole <=25 mm, terminal <=35 mm, club cluster <=60 mm and pelvis
  yaw <5% by the legacy definition. Also report absolute yaw degrees; percentage
  is ill-conditioned near zero. Do not weaken gates to declare completion.
- Compare equal-compute-budget runs from the same correctly transformed seed:
  repaired single shooting versus actual multiple shooting. Record simulation
  calls, wall time, best feasible metrics, defects and continuous replay gaps.
  Exploit qualified fast-engine derivatives/sparsity; use remote batches that
  own whole optimizations rather than SSH round trips for every residual.

### Stage 5: Full Horizon and Degree Refinement

- Advance only from independently replayed candidates, initially in small
  transition increments. Preserve the best feasible candidate separately from
  best average error and optimizer-returned candidate. Never overwrite them.
- Piecewise C2 cubics may diagnose attainable motion and initialize a solution.
  A spline is generally not exactly representable by one sextic: project it
  using weighted effort/derivative fitting, then re-optimize the resulting
  global sextic against forward marker error. Torque approximation alone is
  not acceptance. Exact preservation of a polynomial over a nonzero interval
  leaves no freedom to change its later part; early retention must be a measured
  tolerance, not a promise to freeze the entire first 0.60 s exactly.
- Use exact degree elevation for cubic-to-sextic warm starts; preserve basis
  metadata for any later order. Prefer direct sextic optimization first because
  existing useful sextic seeds already exist. Higher order is an explicit
  comparative experiment, not a substitute for meeting the sextic requirement.
- Audit observations throughout the nominal ~1.813889 s capture. Prior audit
  found missing markers after ~1.233333 s; revalidate masks. Simulate the entire
  requested horizon, but label unobserved intervals as unvalidated extrapolation.
  Never manufacture a full-swing marker match where observations are absent.

### Stage 6: Deliver and Preserve

- Produce a single replay command per engine consuming the same candidate
  package. Native R2025b final acceptance must be one unsegmented run from the
  initial state, without feedback or measured-state injection.
- Save q/qd/efforts, predictions, observation masks, per-marker and time-window
  metrics, closure error, effort extrema, model/input/source hashes, release,
  solver settings, timings, terminal status and best candidate at each milestone.
- Create synchronized target/model overlays and error-versus-time plots with
  millimeter units, full time axis and missing-data shading. Separate C3D fit
  from engine-to-engine parity. Label each visual with exact candidate/hash.
- Commit incremental code/tests/docs; push topic branches through normal policy.
  Do not alter another agent's dirty work or kill unverified remote processes.
  Update one CURRENT checkpoint pointing to immutable runs, then communicate
  the exact checkpoint and remaining work to the other agent after each stage.

## Locations and Resume Commands

- Review/native checkout:
  `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native`, branch
  `feat/9967-native-simscape-pinocchio`.
- Gemini checkout:
  `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour`, branch
  `feat/9921-simscape-tour-matching`; reviewed commit `58df3af67`.
- Desk runtime:
  `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour-runtime`.
- Native runs: `DeskComputer:C:/Users/diete/SimscapeTour9921`.
- Preserved reviewed remote scripts:
  `C:/Users/diete/Repositories/simscape-tour-checkpoints/review_run_ms_optimization_080s_20260911.py`
  and `review_cold_replay_evals_20260911.py`. These are audit snapshots, not
  endorsed runners. Source SHA256 values are in the companion review manifest.
- ControlTower native Pin environment: WSL `ControlTower-Runner`,
  `/home/dieterolson/simscape-pinocchio-9967/.venv/bin/python`.
- MATLAB executable on both hosts:
  `C:/Program Files/MATLAB/R2025b/bin/matlab.exe`. Assert release inside the job.
- Geometry archive and native receipts:
  `C:/Users/diete/Repositories/simscape-tour-checkpoints`.
- Native check runner:
  `docs/development/simscape_tour_matching/native_evidence/reproduction/check_native_pinocchio_pose.py`.
  Arguments: `--module <native_model.py> --spec <native_geometry_spec_9967.json>
--seed <initial_velocity_seed_qualified_r2025b.json> --output <new.json>
--check-free-fall`. Run with actual Pinocchio 4.1.0, not a stub.

The existing initial-FK/free-fall source files were uncommitted when this review
began. They are preserved in this checkpoint; those checks are not full dynamics
qualification. Native graph/solid/transform/assembly/spec tests passed (16).

## References

- Gemini receipts at commit 58df3af67 under
  `docs/development/simscape_tour_matching/native_evidence/reproduction/`:
  `cold_replay_deliverable1_receipt.json`, `controlled_diagnostic_report.json`,
  `deliverable3_080s_receipt.json`.
- [Drake MultipleShooting API](https://drake.mit.edu/doxygen_cxx/classdrake_1_1planning_1_1trajectory__optimization_1_1_multiple_shooting.html)
  describes state/input decision-variable infrastructure; it does not certify
  this repository's implementation or guarantee convergence.

This document supersedes older instructions to wait on completed inventory
jobs and any claim that Deliverable 3 demonstrated successful multiple shooting.
