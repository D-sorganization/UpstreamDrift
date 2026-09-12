# Pinocchio Matching Execution Plan

## Program Scope and Current State

The active program covers native R2025b Simscape equivalence, a faster
Pinocchio matching oracle on ControlTower, repeatable C3D fitting with global
degree-six effort profiles, cross-engine interchange, and the newly requested
OpenSim matching workstream. OpenSim planning is committed in a separate
worktree under [Epic #10003](https://github.com/D-sorganization/UpstreamDrift/issues/10003);
implementation awaits the requested user check-in. The detailed plan and
next-agent prompt are in `docs/development/opensim_tour_matching/EPIC_10003.md`
and `HANDOFF.md` on branch `docs/10003-opensim-matching-epic`, commit 0517dea90.
This extends the working scope without replacing or completing the native
Pinocchio objective. Read `NATIVE_PORT_CHECKPOINT_20260911.md` for current
evidence and `REVIEW_AND_EXECUTION_HANDOFF_20260911_2200.md` for the matching
strategy and known Gemini runner defects.

Native geometry, initial mass/COM, 27 stationary input responses, and six
moving-state accelerations through 0.80 seconds have been compared with actual
R2025b execution. The essential fix is depth-first joint insertion, committed
as d03983d08. Those checks do not establish continuous rollout or full-swing
matching. Existing native JSON preserves a tree plus explicit grip closure;
plain URDF cannot preserve that entire execution contract by itself.

## Rules for Each Assigned Stage

One owner and one narrow stage at a time per checkout. Check the issue lease,
read AGENTS.md/CLAUDE.md, inspect current Git state and live remote processes,
and use an isolated branch. Do not alter another agent's files or restart a
job based only on stale documents. The parent program is #9964 and native
matching is #9921; native-port history and coordination are under #9967.

Before new behavior, write the red test; then implement minimally and run the
same test green. Use public shared APIs. Validate shape, units, coordinate
identity, finite values, bounds and time coverage at boundaries. Keep engine
internals within adapters. Avoid duplicating polynomial evaluation, C3D
loading, marker projection, metrics or optimizer logic. Run targeted tests,
Ruff lint/format and normal repository hooks. A successful mocked test cannot
replace actual native execution evidence.

After every stage, commit code/tests/docs, preserve immutable run artifacts,
update the checkpoint and post the exact branch/commit/receipt to coordination.
Record failed experiments too. Every turnover must say what is verified,
what remains unverified, whether any process is actually live, and the next
specific command. Do not close an epic based on a smaller stage's success.

## P1: Continuous Forward Replay Qualification

Inputs: production `native_model.py`, native geometry spec, native saved
polynomial coefficients and raw native q/qd clock. Reproduction diagnostics
live under `native_evidence/reproduction`. Shared `continuous_forward.py`
integrates an explicit Euclidean state derivative once from t=0; an engine
adapter owns closed-loop accelerations and post-run constraint checks.

1. Start with 0.60 s, then 0.70, 0.75 and 0.80 s. Use one initial q/qd, the
   exact saved polynomial time origin and highest-power-first native inputs.
   Map world forces into the hip base; keep native coordinate identity after
   depth-first internal ordering. Do not inject measured intermediate states.
2. Red tests must cover a nonconstant sextic analytic trajectory, malformed
   clocks/states, nonfinite derivative output, failed/partial integration,
   coefficient-order mismatch and horizon extension without retiming effort.
3. Compare q/qd, projected marker positions, closure pose/velocity and effort
   extrema over identical times. Repeat with tighter tolerances and smaller
   maximum step; quantify numerical convergence rather than choosing gates
   from the worst run. State angle chart/unwrap conventions explicitly.
4. If integration stalls near a gimbal singularity, save the last valid state,
   conditioning and solver status. Investigate a valid internal coordinate
   representation with proven virtual-work mapping. Do not change the native
   physical model or declare a truncated trajectory accepted.

Done: continuous same-input replay agrees within a documented numerical error
budget well below the C3D fitting budget, with small closure residuals and
step convergence. Output a compact receipt and raw trajectory artifact hashes.

## P2: Candidate and Engine Adapter Contract

Inputs: P1-qualified integrator and existing shared candidate/target helpers.
Expose a stable forward oracle consuming one candidate identity and requested
times and returning native marker predictions, q/qd and diagnostics.

Red tests: swapped coordinate names, missing force frame, absent geometry hash,
coefficient order/basis ambiguity, invalid initial closure, NaN prediction,
missing observation coverage and an unsupported passive-parameter change must
fail explicitly. Require immutable model, capture, attachment, initial-state,
input and source identities. Keep optimizer parameters separate from engine
primitive effort vectors. Cache only with all those identities in the key.

Done: one replay command consumes a versioned candidate package on ControlTower
and produces the same qualified result as P1; unsupported physics never fall
back to zero/default values. Preserve the actual dependency freeze, including
the newly installed SciPy runtime version, in the receipt.

## P3: Tree and Closure Interchange

Reuse native geometry/spec APIs and existing model-generation infrastructure.
Export a URDF tree plus an explicit sidecar containing weld closure frames,
native primitive names/order, generalized effort mapping, initial state,
passive parameters, gravity and provenance. Preserve all 27 native coordinates
and forearm rotations. Do not interpret them as 27 unconstrained DOFs.

Red tests: reload must preserve mass/COM/inertia, poses, closure, input power
and pulse responses; removal of a grip constraint or a forearm joint fails.
Validate the exported tree with actual Pinocchio first, then engine-specific
MuJoCo/Drake importers. Each engine gets its own actual pulse/rollout evidence;
a common XML file is not equivalence proof.

Done: exported/reloaded Pinocchio reproduces P1/P2 and artifacts explicitly
describe every engine approximation or unsupported feature. OpenSim uses the
same source/target identity where applicable but must not inherit an unproven
native-equivalence claim for a different musculoskeletal model.

## P4: Benchmark and Optimizer Integration

Profile complete residual evaluation, including input mapping, integration,
marker projection and metrics, on the actual ControlTower environment. Compare
against the same native R2025b candidate and clock; record warm-up, median and
tail time, simulation failures and thread settings. Do not quote speedup from
bare FK or acceleration calls as end-to-end optimization speedup.

Use the shared prefix/multiple-shooting code after repairing the acceptance and
time-basis defects described in the review. Red tests must reject poor
unsegmented replay even with small shooting defects. Check explicit absolute
segment boundaries, constant residual dimension, scaled state defects and
closure-consistent node states. Audit actual runner imports and decision-vector
contents so another single-shooting run cannot be mislabeled multiple shooting.

Done: an equal-budget comparison from the same preserved seed demonstrates
working optimizer integration, feasible candidate preservation and native
replay agreement. Keep best feasible, best average and returned optimum separate.

## P5: Fit and Extend the Swing

Start from preserved eval79/eval559 with correct polynomial basis conversion.
Diagnose transition kinematic floor and coordinate conditioning before changing
bounds or geometry. Optimize a global sextic directly; optional piecewise
cubics are initialization/feasibility tools whose projected sextic must be
re-optimized and independently replayed. Exact preservation over the first
0.60 s would fix the entire polynomial; use an explicit early-error tolerance.

Keep current gates visible: early <=12 mm, whole <=25 mm, terminal <=35 mm,
club-cluster <=60 mm and legacy yaw <5%, alongside absolute yaw degrees.
Do not weaken gates or ignore late errors to report success. Any marker gaps
must remain masked; the nominal full capture extends beyond the previously
observed marker coverage. Separate a full physical rollout from a validated
match over measured observations.

Done: a reproducible candidate satisfies the declared observed-motion gates
and passes a fresh unsegmented R2025b validation. Save error-versus-time plots,
target/model overlays, missing-data shading and exact candidate/hash labels.
The full program is not complete until the requested final horizon and each
named engine's actual qualification are addressed explicitly.

## Turnover Template

- Scope, owner, issue, branch, worktree and last committed revision.
- Current stage and exact pass/fail evidence; source/model/capture/input hashes.
- Raw artifacts and compact receipt locations on both local and remote hosts.
- Actual live process handle, command and start time, or explicit terminal exit.
- Tests run and any remaining failures; avoid a generic statement of green CI.
- Next command, expected output, numerical/physical acceptance gate and reason
  to stop or investigate instead of blindly increasing computation.
- Changes communicated to other agents and any user checkpoint still pending.
