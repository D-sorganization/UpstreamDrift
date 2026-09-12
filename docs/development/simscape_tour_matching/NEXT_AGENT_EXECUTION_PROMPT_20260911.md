# Execution Prompt: Finish Four-Engine Tour Swing Matching

You are taking over a critical scientific implementation task. Execute it through measured acceptance, not just planning or long-running jobs. Read REVIEW_AND_TURNOVER_20260911.md and review_800ms_remote_receipt_20260911.json first. The final deliverable is the tour-average driver capture matched by 3D forward dynamics with a SINGLE global degree-six polynomial per actuation channel, and equivalent same-input behavior in Simscape R2025b, MuJoCo, Drake and Pinocchio. Then make the workflow reusable for other C3Ds, initial guesses and polynomial degrees. The current motion is not fully matched; do not repeat the claim that cross-engine physics is fully established merely because #9965–#9969 merged.

## 0. Immediate Checkpoint and First Experiments

The latest inspected revision is 9bf2a4f8d. The 0.80 s run is FINISHED, not active:
775 logged evaluations, final whole RMS 44.042 mm, terminal 200.541 mm, club
terminal 274.539 mm, 2/5 gates. Eval 559 has the best logged average 43.228 mm
but lacks an independent gate audit. Run 06 Eval 79 remains a useful 0.75 s
exploratory seed (23.859 mm whole, 94.690 mm terminal, 115.979 mm club terminal).
Do not compare unlike horizons as records or treat xtol as motion acceptance.

Your first deliverables, in order:

1. Refresh process/status and ownership; cold-replay Eval 79 and the 0.80 s
   final, then reconstruct/replay Eval 559 if its exact coefficients exist.
   Independently recompute gates using the same target masks and metric definition.
2. Produce one diagnostic report from controlled experiments: native kinematic
   marker-error floor with fixed geometry, actual rollout state/velocity error,
   solver repeatability, scaled directional derivatives, active bounds and
   Jacobian rank. If the kinematic floor fails the gates, repair geometry,
   marker attachments or constraints before further torque search. A feasible
   kinematic target alone does not prove dynamic reachability.
3. Compare full 189-coefficient sextic versus existing restricted search under
   the same wall-time budget and objective. Log early/transition/terminal/club
   tradeoffs. If derivative checks fail, fix simulation resolution/scaling
   first. If stable derivatives still stall, implement multiple shooting as
   specified below; do not blindly run another horizon ladder.
4. Export the evaluated R2025b physical model and qualify Pinocchio on
   ControlTower before surrogate fitting. URDF plus manifest must preserve
   loop constraints, inertias, frames, efforts and initial state. Start with
   FK and independent force pulses; no four-engine certificate from mocks.

Keep the native Simscape lane and engine qualification lane separately owned.
Request explicit bounded delegation if multiple agents are needed. A smaller
agent can execute tested run/replay recipes once these contracts are settled;
the remaining modeling and numerical diagnosis is not merely computation.

## 1. Establish Ownership and Reproducible State

- Read applicable AGENTS.md, CLAUDE.md, Repository_Management coordination rules and this turnover. Inspect current git status and remote issue updates. Confirm ownership with the active Gemini session; use the fleet lease mechanism before fixes. Work in an isolated topic branch when someone else owns the active checkout. Do not overwrite their source or runtime.
- Recheck DeskComputer/ControlTower actual processes, command lines, start times, heartbeat and exit receipts. Never identify a job by PID alone. At review, run 04 was terminal with no native process; do not restart a duplicate based on stale handoff text.
- Pin exact source, dependency versions, MATLAB release, model hashes, C3D hash, solver settings and actuator names per run. A completed candidate package must contain its own parameters, basis, coefficient order, physical clock, state/geometry identity, selected evaluation ID, gate report and independently reproduced trajectory.
- Preserve failed attempts and selected intermediate candidates. Never silently replace a selected candidate with the optimizer's final trial. Keep raw results outside git when large, with archive hashes and durable receipts in version control.

## 2. Repair Acceptance Before Launching More Fits

- Write failing tests for: a candidate with good whole-window RMS but bad terminal RMS; a selected trial different from final optimizer output; equality/breach at every threshold; missing cold replay; wrong source/state/basis identity; invalid C3D samples; missing required native engines; benchmark input differing from baseline theta.
- Candidate 528 is the real regression example: 24.312024 mm whole, 9.726861 mm early, 103.309516 mm terminal and 124.636260 mm terminal club RMS. Packaging must not label it accepted. It may be an explicitly unaccepted exploratory seed.
- Retain existing gates unless the user explicitly changes them: early <=12 mm, whole <=25 mm, terminal <=35 mm, club terminal <=60 mm, current documented yaw requirement. Centralize thresholds and definitions. Report transition-window RMS, p95, max, velocity error, per-marker/body errors and all failures. Use valid masks; missing markers begin later in this capture. Freeze capture event times from evidence, not assumed 0.75/1.2 s labels.
- Use a consistent wrapped-angle yaw residual and state the reference frame. Avoid sin-only angle penalties that admit 180-degree aliases. A percent of absolute yaw is frame-dependent and near-zero unstable; document this and propose a fixed angular acceptance criterion, without silently changing the user's gate.
- Audit actual force/torque profiles for all 27 channels (24 torques and 3 forces); do not call these all independent DOFs. Reject silent clipping and unactuated/missing channels.

## 3. Diagnose and Repair Transition in Simscape

- Use the best 0.6 s cubic and the explicit Candidate 75/528 packages as separate starting points. Keep geometry, marker attachments and initial state fixed during each torque experiment. Any geometry change starts a newly qualified identity; never change link lengths with time to hide residuals.
- Before more search, test repeatability and directional finite differences at the same candidate with relative steps 1e-5, 1e-4 and 1e-3 plus appropriately tighter solver tolerances. Use actual parameter scales and record physical effort perturbations. Check whether derivatives stabilize, identify near-null/redundant actuation directions, active bounds and Jacobian conditioning. Compare weighted objective and unweighted physical metrics. Save nfev, njev, optimality, cost, termination reason and wall time. TRF max_nfev is not a count of every finite-difference simulation.
- Keep final coefficients in a numerically conditioned degree-six basis with an explicit fixed full-capture clock from the actual last sample (approximately 1.813889 s). Exact change of basis is allowed; there is no physical difference between equivalent power and Bernstein curves. Protect the early motion with measured residuals, not a belief that SVD torque directions leave motion invariant.
- First repair 0.70–0.75 s using transition/terminal/club residuals plus early retention. Compare full sextic and restricted SVD/high-order searches on the same objective and budget. Avoid over-restricting later authority or arbitrary large objective weights without sensitivity analysis. Bound actual force/torque profiles and document regularization units and scaling.
- Require improvement across a Pareto set of early, transition, terminal and club errors. Do not increase the horizon solely because average RMS passes. A short failed-horizon diagnostic is allowed but remains explicitly failed.
- If stable derivatives still stall, implement constrained multiple shooting using the SAME single global polynomial across every window. Optimize independent feasible boundary states and impose matching state/velocity and loop constraints; handle all dynamic states needed for restart, not only visible q/qd. Initial boundary states may be seeded from smooth native kinematics. Eliminate shooting defects to declared tolerances, then validate with one continuous t0 simulation. No measured-pose resets in accepted motion; no inverse-dynamics prerequisite; no stitched-polynomial approximation as final evidence.
- Continue toward the full capture only after each gate, with checkpointed refinement. If a global sextic appears limiting, document reproducible multistart/convergence evidence and achieved tradeoffs; do not assert mathematical impossibility from one local failure.

## 4. Establish Actual Four-Engine Physics Parity

- Inspect new wrappers and main-branch changes before implementing. Reuse existing public abstractions; do not build another parallel simulator. Canonical physical authority must be reconciled against actual R2025b model data, not just a similarly named YAML.
- Publish one manifest for topology, geometry, mass/COM/inertia frames and units, gravity, damping/friction, actuator transmission/signs, joint limits, contact settings, dual-grip loop frames, q/velocity mapping, initial state and marker attachments. A 25-DOF or 15-actuator model is not equivalent merely by selecting 27 source rows. Prove any coordinate reduction and virtual-work-consistent force mapping.
- Test coefficient conversions at t=0 and several nonzero times. Distinguish highest-power-first Simscape A..G, lowest-power-first powers and Bernstein controls. Preserve the polynomial time origin/horizon independently of rollout duration. Map base forces and angular efforts with correct frames and gear gains; never silently discard unmapped channels.
- Implement/verify loop closure in actual execution: appropriate MuJoCo equality constraints and measured compliance, Drake-supported loop constraints for the chosen solver, Pinocchio constrained forward dynamics with constraint-consistent integration. Ordinary tree ABA or a one-hand-only club is not the target mechanism. Native dependency/version capability checks must be explicit.
- Qualification ladder: (a) same-pose FK and marker geometry; (b) identical applied effort at sampled times; (c) gravity and independent actuator-pulse acceleration/constraint response; (d) 20–50 ms, 100 ms, 0.6 s, transition and full-capture same-input rollouts. Compare physically mapped velocities/accelerations and loop residuals, not raw incompatible coordinate arrays. Include free-base and both-arm excitations; three gentle torso channels over 20 ms are insufficient.
- Export a true R2025b reference fixture with source/run/model/initial-state/theta hashes and explicit native exit success. The NPZ loader must verify provenance and actual theta equality. Do not synthesize a surrogate baseline and label it Simscape. Assess existing fixture provenance before making accusations.
- Acceptance job must run all four native engines and fail/incomplete on missing dependencies. A matrix with only baseline-versus-itself is never a four-engine pass. Keep optional unit-test skips separate. Retain time alignment, sampling masks, input equality, grip <5 mm and clubhead <10 mm parity criteria, and documented joint/constraint/energy checks. Energy balance includes base-force work, damping/contact work and constraint-work assumptions; do not apply a torque-only identity blindly.

## 5. Accelerate Only After Qualification

- Benchmark one qualifying fast engine, initially MuJoCo if its loop/mapping parity passes. Then run small multistart searches and independent finite-difference batches in isolated processes. Recheck candidates and nearby perturbation sensitivities in Simscape, limiting surrogate use to regions with measured agreement. Do not search four non-equivalent models and call different input solutions parity.
- Allocate DeskComputer to native fit/replay, ControlTower R2025b to derivative diagnostics or independent verification, and available Linux/native hosts to qualified fast-engine work. Confirm availability/licensing before scheduling. Use persistent workers/model compilation caches with tested input invalidation. Do not concurrently mutate a shared MATLAB engine/model workspace.
- Once physical parity is established, use the same canonical sextic on all four engines for final acceptance. Separate capture-fit accuracy from engine-to-engine accuracy: all engines agreeing on a bad swing is still failure.

## 6. TDD, Contracts, Handoff and Completion

- TDD for substantive behavior; record red/green for gate enforcement, selected-candidate serialization, coefficient conversion, state/force mapping, loop preservation and no-reset replay. DbC validates shapes/units/finiteness/domain/identity; LoD uses public engine APIs; DRY centralizes canonical metadata and metrics. Do not substitute tests of mocks for actual physics acceptance.
- Every checkpoint updates a concise status ledger: candidate hash, source/engine/solver identities, exact run command, host/process state, fit/gate metrics, best accepted versus best exploratory candidate, archive location/hash and next action. Correct stale active/certified text. Use conventional commits and normal hooks; push topic branches and PRs without changing collaborators' files.
- Deliver a reusable C3D-to-candidate configuration/CLI, accepted coefficients, native replay artifacts for all engines, synchronized marker overlays, per-phase residual/force/constraint plots, a native all-engine test report, and reproducible cold-run instructions. Degree and starting-point settings must be explicit. No credentials or host-specific secrets in git.
- The goal is complete only when the full capture meets agreed motion gates in native forward dynamics and all engines meet same-input parity, with no missing-engine skips. If an external dependency blocks work, name the exact blocker and leave every independent completed improvement committed and resumable. Do not mark completion because a run launched, a PR merged, or the optimization budget ended.

Start now with Section 0 and the new parity suite, correct stale acceptance/run records, and record a concrete run/ownership plan on #9921/#9964. Candidate 528 remains a historical regression fixture; do not mistake it for the latest seed. Work incrementally through the above sequence until scientific acceptance is achieved.
