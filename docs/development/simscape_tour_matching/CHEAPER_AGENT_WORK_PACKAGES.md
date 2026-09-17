# Cheaper-Agent Execution Packages

## Start Here

The current work is deliberately paused for handoff. No new optimization was
started. The completed static curve audit is terminal (exit 0): 304 solves at
38 times through 1.813888889 s, 143.43 seconds on ControlTower. Read
PROGRESS_REVIEW_20260917.md, AGENT_RESUME_PROMPT.md and the evidence README before
any execution. Do not repeat the source restoration or terminal audit.

Branch: feat/10285-native-saved-replay; draft PR #10287. Latest committed checkpoint
before this document: 308a7a2f2; use the branch tip containing this file for handoff.
Local worktree: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native.
Run git status and check issue leases before changes. Three pre-existing status-only
modifications had empty git diff --numstat; do not discard or stage them blindly.

## Package A: Matching Diagnosis and Calibration

Give this package to the lead numerical agent first. Its next bounded deliverable
is an evidence-backed calibration decision, not an overnight optimization run.

1. Read native_evidence/static_curve_20260917/README.md and summary.json. The
   fixed-model static errors are 39.76 mm at 0.85 s, 41.91 mm at 0.90 s and
   46.29 mm at 1.05 s with no active local bounds. No new dynamic fit exists.
2. Investigate only the flagged 1.25–1.55 s static-search cases before interpreting
   their spikes: active local bounds and changed validity masks make them weaker
   evidence. Save the original rows. Test smaller time steps, alternative seeds
   and declared expanded numerical boxes; do not call these anatomical limits.
3. Audit head/back and upper-arm marker assignments against authoritative source
   exports. Head and back currently share Hub. BackLeft–HeadSide target distance
   varies by about 110 mm over the capture; a single fixed rigid pair cannot
   reproduce all those distances. This can reflect real relative motion, target
   processing or attachment/model assumptions; do not assume which without checks.
4. Use shared marker_calibration, rigidity and constrained_marker_pose providers.
   Compare fixed attachment/length calibration on declared training frames with
   held-out frames. Preserve original offsets. Report parameter deltas, validity,
   all-marker error, club/yaw and closure. Do not calibrate per frame.
5. Deliver a short decision report: whether a coherent fixed-geometry variant helps,
   what remains unexplained, and whether to proceed to control fitting. Model
   topology changes, including adding a neck, require explicit review of the
   equivalence/model requirements. Do not silently alter the Simscape reference.
6. Only after that decision, follow the trajectory derivative and global-sextic
   continuation stages in AGENT_RESUME_PROMPT.md. Start with a bounded 0.90 s
   experiment; preserve early-motion gates and independent R2025b replay.

Acceptance for this package: reproducible commands/inputs, saved witnesses and
residual vectors, held-out validation, tests for any new behavior, no baseline
mutation and a clear next action. A failed local solve or optimization budget
expiry is not proof of infeasibility or convergence.

## Package B: Saved-Run Product Polish

This can proceed in a separate worktree with an issue lease for epic #10285.
Avoid Package A's numerical sources and fixture files.

Extend the existing tour_matching_viewer and SimulationDataStore. Load
native_evidence/simscape_returned102.replay.json using the existing verified
manifest path. Deliver a shared catalog with provenance and accepted/rejected
status, capsule/cylinder views with persistent cameras, marker error overlays,
and effort display that explicitly marks unavailable tau. Add speed controls
only through the existing source-time clock. Preserve pause, seek and restart.

First useful increment: a catalog entry for the verified run102 manifest that
opens the existing viewer, plus an obvious link/action to inspect its report and
actual cylinder animation. Then add multi-angle cylinders using shared rendering
providers. Do not invent a new viewer or infer native coordinate order from names.
For MATLAB reruns, use the explicit R2025b executable and a new output directory;
show the model/candidate/input identities and preserve failure reports.

Acceptance: focused corruption/path/hash/missing-effort tests; user can discover,
open and inspect the saved run without finding an arbitrary NPZ; normal-font visual
QA; no claims that playback reruns physics. Commit each usable increment.

## Package C: Native Counterfactual Qualification

Use epic #10286 and SAVED_RUNS_AND_COUNTERFACTUALS_HANDOFF.md. Start with one native
constrained engine, not four half-working adapters. Reuse native constraintDynamics
and authoritative AffineDrift/WSCG definitions. Establish saved-state intervention
and reaction/wrench contracts with units, frames, points of application and signs.
ZTCF removes declared controls at achieved state; ZVCF uses q unchanged, v=0 and
u=0. Recompute constraints/reactions for each intervention. Label any separate
control-preserved zero-velocity analysis distinctly.

First deliverable: a tested native provider exposing physically qualified reaction
wrenches and actual/ZTCF/ZVCF records, with no mutation of the source run. Validate
constraint equations, state restoration and wrench-frame transforms. Only then
connect to shared playback and port one engine at a time. Never use unconstrained
ABA or qdd-times-length placeholder arrows for the closed-loop native model.

## Shared Checkpoint Contract

Use TDD, DbC, LoD and DRY. Lease issues; separate worktrees; no competing edits to
shared providers. The numerical lead coordinates model changes. Every handoff must
name the branch/commit, tests, live job handles (or confirmed none), exact next
command, output paths and unresolved gates. Save incremental receipts during remote
runs. Never restart a job because a polling call timed out; inspect the live process.

Useful focused checks from repository root:

```text
python -m pytest tests/unit/motion_matching/test_constrained_marker_pose.py tests/motion_matching/test_rigidity.py -q --no-cov
python -m ruff check <changed Python files>
python -m ruff format --check <changed Python files>
python -m pre_commit run mypy --hook-stage pre-push --files <changed Python files>
```

Use the repository's Python command/environment conventions on each host. Do not
bypass commit/push hooks. Do not edit files while hooks are running. Update
HANDOFF.md and DEVELOPMENT_LOG.md in every implementation checkpoint. Report
partial scope honestly; full-swing matching and all-engine product acceptance
remain outstanding.

## Copyable Coordinator Prompt

Continue from the branch tip containing CHEAPER_AGENT_WORK_PACKAGES.md on
feat/10285-native-saved-replay / PR #10287. Read the current handoff and execute
Package A first in bounded, tested increments. Do not repeat completed audits.
Treat the static curve as diagnostic evidence, not a matched full swing. Preserve
R2025b, original gates, the global-sextic requirement and all original artifacts.
If parallel agents are available, assign Packages B and C separate worktrees and
leases with no overlapping source ownership. Save exact commands and receipts,
commit tested checkpoints and maintain turnover documents. Seek review for model
topology/equivalence decisions; do not silently weaken the requested end product.
