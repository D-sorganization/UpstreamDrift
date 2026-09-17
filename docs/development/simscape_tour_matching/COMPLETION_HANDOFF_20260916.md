# Simscape Matching Completion Handoff

## Status and Evidence Authority

Reviewed 2026-09-16. Native fit source checkpoint: 2f0460d25; fetched main
at d2aafa43c. The latest committed native experiment found is run102. No newer
accepted full swing or archived 0.90-second fit was found in the inspected
branches. Remote running-process state was not checked. Do not infer inactivity
from old handoffs. The objective remains incomplete.

Read this document before RUN101_REVIEW_AND_TURNOVER.md, which remains useful
background but has been superseded for task ordering. The old Simscape worktree
at feat/9921-simscape-tour-matching still contains superseded run100-era status;
do not restart from its uncorrected force/geometry replay scripts.

Canonical evidence: native_evidence/two_window_fit_9967_102/ and
native_evidence/refinement_audit_101/ under this directory.
Run102 candidate identity: ee4a908737f4cf3065bb6e64a056dfab71926c43ac40c0e85574fc8531576c82.

- Coverage: 307 frames, 0–0.85 s of the required 654 frames / 1.813888889 s.
- R2025b overall RMS about20.267 mm; early9.995 mm; terminal40.30135 mm;
  club8.389 mm; yaw0.610%. Terminal **fails** the35 mm gate.
- Recorded maximum Euclidean Simscape–Pinocchio difference:0.0605214 mm.
- Run102 returned accepted=false, optimizer_converged=false,
  improved_uninterrupted=false; physical evaluation budget exhausted, fallback
  returned. Zero active bounds is not convergence or unrestricted motion.
- Run101 refinement reduces MATLAB/Pinocchio marker disagreement to0.004523 mm;
  independent Pinocchio half-step change is0.00004693 mm. These support prefix
  marker accuracy, not full-horizon velocity/effort accuracy. MATLAB refinement
  changes native qd by0.202; assess physical body rates before claiming rate parity.
- The independently reproduced23.67696 mm rigid-cluster lower bound relaxes
  connected joints and welds. It neither proves35 mm feasible nor proves the
  observed40.30 mm plateau irreducible. Hub's47.664 mm cluster floor alone
  cannot justify modifying anatomy or removing markers.

## Newly Identified Reproduction and Derivative Dependencies

**Fresh-checkout execution is not yet self-contained.** Run102's driver imports
native_node_chart, native_replay and native_sensitivity from the Pinocchio package.
Those three files are absent from tracked HEAD and the inspected origin/main.
The launcher relies on frozen remote runtime78. A committed driver and source
hashes do not substitute for installable source. Recover exact qualified sources
from recorded runtime/source archives and historical commits; reconcile existing
providers before restoring missing modules. Never overwrite the frozen runtime.
A clean-runtime import and baseline replay is the first engineering deliverable.

**Coordinate with issue10260 / PR10263 before off-closure pose optimization.**
The other agent reproduced that closure_position_linearization used the velocity
Jacobian for the finite weld-position residual, valid at zero error but wrong
away from closure. Commit67d99f4e3 applies the SE(3) log derivative and branch guard.
At review, PR10263 is open, conflicting and has a failing equivalence check; it
is not a qualified merged dependency. Do not take ownership of the other agent's
active lease or blindly merge its branch. Agree on a pinned corrected provider,
then run real-engine directional tests and downstream chart/retraction tests in
the actual Pinocchio4.1.0 runtime. Preserve velocity/acceleration conventions.
This is a dependency for trustworthy feasibility derivatives, not evidence that
historical closed-loop trajectories are invalid or the plateau's cause is known.

## Execution Packages and Exit Criteria

### A. Restore a Reproducible Baseline

Reconcile latest main, historical fit sources and frozen runtime manifests on a
new implementation branch. Add only needed providers with their contracts/tests;
retain the old evidence unchanged. Build one config-driven entry point reusing
shared code: capture/model/candidate, horizon, shooting nodes, polynomial basis,
solver settings and output location are explicit inputs. Stop copying 700-line
numbered scripts. Separate polynomial basis duration from replay horizon.

Exit: from a clean checkout/environment, import all providers, validate all input
hashes and reproduce run102's uninterrupted prefix within established numerical
tolerances. Record the exact command, dependency versions and time budget. Test
identity restart, coefficient ordering/units, missing inputs, no double force
rotation, and horizon changes that preserve the original polynomial function.
Do not require MATLAB on this laptop: use explicit R2025b on DeskComputer or
ControlTower, and the recorded Pinocchio WSL environment.

### B. Establish What the Fixed Model Can Reach

After qualifying the finite-weld derivative, use existing shared constrained-pose
providers to fit the terminal frame with original geometry/attachments, all valid
markers, physical joint constraints, weld closure and yaw requirement. Initialize
from integrated run101/run102 plus a bounded set of alternatives. Save the pose,
body transforms, closure and per-marker residuals. A passing feasible pose is an
upper-bound witness for kinematics only. Failed local starts do not prove global
infeasibility. Recover any already-executed feasibility probes before rerunning.

At run102 inspect equality-projected sensitivities, feasible gradients, scaling,
accepted steps and budget/line-search history. Test a few predicted directions by
replay. If early-preserving useful directions require lower sextic coefficients,
qualify a small selected subset; do not add independent per-window torques.

Exit: a saved articulated feasibility witness or an honestly bounded inconclusive
result, plus one justified next fitting change. No new neck joint, dropped marker,
softened weld or changed threshold without an explicit model-variant decision.

### C. Deliver a Real Extended-Horizon Fit

Start with0.90 s, adapting to recovered extrapolation evidence. Add an integrated
node near0.85 s only if needed. Preserve original initial state, absolute time and
one continuous global degree-six profile per actuator. Test same-input transfer
before optimization. Use a bounded trial, record predicted versus actual change,
and independently replay its returned candidate without intermediate resets.
Keep early retention and report errors at0.85 s as well as the new endpoint.
Do not spend indefinitely polishing0.85 s before learning about the downswing.

Exit: a genuinely fitted0.90 s result, all gates graded PASS/FAIL, independent
R2025b replay and an explicit next decision for0.95 s. Rejected results remain
exploratory. A reproducible numerical blocker is a valid report, not completion.
Extend incrementally toward1.813888889 s only after each run yields useful evidence.
Use refined R2025b ode15s RelTol1e-7, AbsTol1e-10, MaxStep1/2880 s for promoted
checks; qualify settings again when a new horizon/candidate warrants it.

### D. Turn the Experiment Into a Usable Product

Deliver a repeatable workflow: validate target -> load restart/config -> fit ->
independent replay -> evaluate -> report. Each run needs a manifest with raw and
canonical hashes, versions, geometry, marker masks, frame/effort conventions,
polynomial degree/basis, initial state, original clock, budgets and terminal status.
Output directories are immutable; resume uses a new directory and parent identity.
Reject missing/nonfinite/shape-invalid inputs. Reuse shared providers and write
behavioral tests first (TDD, DbC, LoD, DRY), including failed-run/resume paths.

A report must show tested horizon and acceptance status prominently, with per-body
and per-marker errors over time, terminal/early/yaw/effort/closure metrics and
engine differences. Include synchronized 3D capture/model playback, timeline
scrubbing, residual vectors and torque plots. Generate a portable static summary
and animation from the same saved arrays; screenshots must not imply full-swing
acceptance for a prefix. Integrate with the existing replay/launcher work after
agreeing its data contract, not by creating another viewer or solver stack.

Exit: another agent can reproduce one candidate and its report with documented
commands from a clean checkout, then start a different C3D/restart or polynomial
configuration without editing source. Degree-six remains the acceptance target;
configurability does not authorize silently changing it.

## Completion, Ownership and Reporting

Final matching acceptance requires all654 frames, original-state uninterrupted
forward dynamics, global sextic inputs, all established marker/yaw/closure/effort
criteria and independent MATLAB R2025b replay. Preserve the existing27-coordinate
model; anthropometric/full-body models are separate variants. MuJoCo, Drake,
OpenSim and alternate-coordinate acceptance remain separately evidenced and must
not be inferred from Pinocchio–Simscape marker agreement.

Epic9921 and issue9967 are closed at review. Reconcile the full-goal epic or create
an explicitly linked continuation issue before implementation; completion of a
harness issue is not completion of the swing. Check leases and other agents' work.
Update main HANDOFF, DEVELOPMENT_LOG and this plan at each tested checkpoint.
Keep failed-gate reporting factual; do not call budget exhaustion convergence.

## Copy-Ready Agent Prompt

Resume the tour-average Simscape matching goal. Read AGENTS.md, CLAUDE.md,
docs/development/HANDOFF.md and
COMPLETION_HANDOFF_20260916.md in docs/development/simscape_tour_matching.
Use run102 as an exploratory baseline, not an accepted swing. Coordinate issue10260
and PR10263; do not duplicate their finite-weld derivative correction.

First recover a clean, installable runtime for the missing native fitting providers
and reproduce the archived prefix. Then qualify corrected off-closure derivatives,
produce a constrained terminal feasibility result and diagnose feasible optimizer
directions. Execute one bounded0.90 s fit with preserved global sextic inputs and
original-state replay, validate in R2025b, and decide the next horizon from evidence.
Build the shared config-driven execution/report path as part of this work, with
portable synchronized visuals, immutable manifests and tested resume behavior.
No full-goal closure until all654 frames and physical/engine gates pass. Commit
incrementally, preserve historical evidence and leave exact next commands in the
handoff after every checkpoint. Start by publishing the small work-package plan
and success criteria; proceed without launching a blind long optimization ladder.
