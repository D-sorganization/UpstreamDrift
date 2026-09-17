# Simscape Matching Progress Review and Execution Turnover

## Latest Verified State: Local Terminal Feasibility Is the Priority

This section supersedes the next-step orders and running-job statements below.
All audits launched by this review have finished. No new dynamic candidate was
produced. Run102 remains a rejected 0–0.85 s prefix of 654 frames / 1.813888889 s.

| Evidence                              | Terminal Marker RMS | Qualification                      |
| ------------------------------------- | ------------------: | ---------------------------------- |
| Saved R2025b run102                   |         40.30135 mm | Actual forward replay; fails 35 mm |
| Static pose, exact target yaw         |         40.37778 mm | Fixed geometry and attachments     |
| Best tested static pose within 5% yaw |         39.76310 mm | Fixed geometry; fails 35 mm        |
| Static pose with unrestricted yaw     |         39.20167 mm | Also fails yaw (15.70349%)         |

Twenty-four local constrained solves converged with tight weld closure. No local
bound is active at the best allowed pose. Multiple starts and two search radii
support a repeatable local result, **not a global error lower bound**. The best
allowed pose remains only about 0.54 mm below the saved dynamic terminal error.
Largest residuals are HeadSide (80.40 mm), LUArmHigh (78.99 mm), HeadFront
(76.16 mm), HeadTop (51.87 mm), and WaistLeft (51.43 mm). This supports checking
fixed geometry and attachment calibration before a large additional torque budget.
No markers, acceptance gates, geometry, or baseline candidates were changed.

The restored shared constrained_marker_pose solver has five passing unit tests
(RED missing import before restoration, GREEN afterward), Ruff and pinned mypy.
A separate native chart audit passes closure directional derivatives at 0, 0.6,
and 0.85 s, both closed and with a 0.1 rad wrist perturbation, at three step sizes.
Maximum error is 1.68e-8; nonzero retraction derivative error is below 4.68e-10.
These sampled checks do not qualify full trajectory sensitivities or reactions.
All native audits consume the isolated finite-weld provider from #10263 commit
0db3c295a7f668789afbb4666f60912294f81cce; the checkout provider is not overwritten.

Evidence, exact executed scripts, receipts, local search definitions, and an
all-marker comparison plot are in native_evidence/terminal_feasibility_20260917.
Read its README.md and the rewritten AGENT_RESUME_PROMPT.md for the executable
handoff order. Search boxes are numerical bounds, not certified anatomical limits.
The fixed-model terminal study must not be relabeled a successful swing match.

## Previous Checkpoint: Restoration and Derivative Audit Complete

Both remote audits are terminal. No optimization is running from this review.
The fitter startup audit passes in 90.73 s: relative segmented/uninterrupted
objective discrepancy 5.06e-14, marker difference 9.50e-12 m, scaled defect
3.36e-11, projected rank 42. It uses the original run102 configuration and run101
restart, not a new candidate. Evidence: native_evidence/restoration_audit_20260917.

PR #10263 now has head 0db3c295a7f668789afbb4666f60912294f81cce, is open and
reported mergeable=true at this check. Earlier conflict status below is historical.
A separate real Pinocchio 4.1.0 native-model audit compares the current and PR
providers at run102 initial/terminal states, wrist offsets 0/+0.1/-0.4 rad,
three finite-difference steps (1e-5/1e-6/1e-7) and three directions per step.
The old provider passes closed states but fails off-closure cases, with maximum
absolute directional error 0.0507–0.2724. The PR provider passes every tested
case with maximum error below 8e-9 (rtol 2e-6, atol 2e-7). The archived audit
contains its exact executed script, corrected provider and receipt.

This qualifies the finite **pose** derivative on these native states. It is not
an acceleration/reaction qualification or proof that this error caused the
optimizer plateau. NativeNodeChart uses trajectory finite differences; check
that downstream path separately. The source checkout still has the original
provider; consume the qualified owner's change rather than duplicating it.
Coordination evidence: GitHub issue #10260 comment 5708242678.

Next sequence: pin/consume #10263; qualify native chart/retraction and preserve
the velocity Jacobian for acceleration constraints; restore only the required
articulated-pose helpers/tests from the verified source archive; obtain a feasible
terminal witness and test projected control directions; then run a bounded 0.90 s
fit. Baseline/source recovery is now done and should not be repeated blindly.
The rejected 0.85 s candidate and all original acceptance gates remain unchanged.

Fitter-dependency checkpoint: native_effort_penalty and shooting_schedule are
restored with exact integral/boundary tests. All historical run102 driver imports
resolve locally; 22 focused tests and the repository-pinned mypy hook pass.
All six original driver inputs were recovered with matching recorded hashes in
native_evidence/run102_original_inputs.tar.gz. Their directory layout is retained.
The first push was rejected by the pinned typing hook; the row-array typing
compatibility correction is now tested. Local mypy alone was insufficient.

A fresh audit-only run is executing at /home/dieterolson/native-fit-audit-20260917-01
on ControlTower / ControlTower-Runner. Source snapshot and qualified baseline
folder are separate. Read audit_output/receipt.json and verify its process before
resuming; do not start a duplicate from an old status file. This is the original
run102 restart/configuration (run101 seed), not a new swing fit. The current
receipt manifest is native_evidence/run102_fitter_restoration.json.

**Verified numerical restoration:** the new source snapshot plus pinned Tools
commit 1ac89c18e reproduces all run102 marker samples exactly (maximum difference
0 m) in 13.38 s on Pinocchio 4.1.0/Python 3.12.3. Position/rate closure maxima
are 3.16e-12 / 1.78e-11. Receipt: native_evidence/runtime78_restoration.json.
This resolves the replay reproduction blocker, not the full fitter, off-closure
derivatives, full horizon, torque logging or MATLAB acceptance. The old
source-recovery paragraphs below are historical checkpoints, not pending work.

To repeat: initialize the pinned Tools submodule in a checkout with these restored
modules. In a new output directory copy runtime78_original_model.bin as model.bin,
two_window_fit_9967_102/returned-candidate.json as candidate.json and
two_window_fit_9967_102/returned-replay.npz as reference.npz. Set PYTHONPATH to
the checkout, OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1; use the recorded
Pinocchio Python environment to execute native_evidence/reproduction/
run102_restored_baseline.py by its absolute path. Always use a new directory.
The script writes receipt_restored_verified.json. The exact tested remote folder
is /home/dieterolson/native-clean-replay-20260917-01; do not replace its results.

Next restore/reconcile the fitter's native_effort_penalty and shooting_schedule
dependencies with their tests, check any further transitive imports, then qualify
#10263 before finite-weld feasibility work. Do not expand restoration to unrelated
manifold/full-body providers merely because they are present in the archive.

Restoration checkpoint: the complete missing-file audit identified 50 historical
source/test paths, all recovered with hashes matching run102. Nine source modules
and four regression files needed by replay/node-chart/sensitivity imports are now
restored; 28 local tests, lint, formatting and nine-module mypy checks pass. The
remaining recovered files are archived, not silently installed. See
native_evidence/runtime78_restoration.json and runtime78_missing_sources.tar.gz.
The exact original model bytes are preserved as runtime78_original_model.bin;
the tracked JSON is semantically identical but has a different formatting hash.
A fresh source snapshot was uploaded to ControlTower at
/home/dieterolson/native-clean-replay-20260917-01 for independent baseline replay.
Do not call the restored fitter fully reproducible until that numerical result
and the remaining driver dependencies have been checked.

Follow-on recovery: the three missing fitting modules have been retrieved read-only
from ControlTower runtime78 and their bytes match run102's recorded hashes. See
native_evidence/runtime78_recovered_sources.zip and its JSON receipt. They are
archival recovery, not yet installed providers or a clean-runtime replay. Restore
with tests and dependency reconciliation; do not rerun the blind source search.
Remote WSL login shell startup fails on a stale cargo environment path; direct
`wsl -d ControlTower-Runner -- <command>` or non-login `sh -c` worked.

## Review Scope and Verdict

Reviewed on 2026-09-17 UTC (2026-09-16 local). Working HEAD b15ba9474;
fetched main dd243d392. Inspected current GitHub prerequisite PRs, archived
run102 configuration/reports/MAT arrays, and the unfinished local replay tests.
No remote process inventory or new physics run was performed. No newer native
fit was found in the inspected evidence history/main; that does not exclude
uncommitted work on another host.

The approach has demonstrated repeatable prefix dynamics and close cross-engine
marker agreement. It has not demonstrated optimizer convergence or full-swing
tracking. More compute with the unchanged run102 setup is not the next justified
step. The immediate deliverable is a reproducible, derivative-qualified bounded
optimization experiment that improves independently replayed motion.

## Verified Baseline

Evidence directory: `native_evidence/two_window_fit_9967_102/`.
Read `returned.json`, `config.json`, and `qualified_candidate_replay.json` together.

| Measurement                       | Recorded Result                   | Meaning                                                                 |
| --------------------------------- | --------------------------------- | ----------------------------------------------------------------------- |
| Coverage                          | 307 frames, 0–0.85 s              | Required capture is 654 frames, 1.813888889 s                           |
| R2025b whole RMS                  | 20.26736 mm                       | Passes 25 mm gate                                                       |
| Early RMS                         | 9.99539 mm                        | Passes 12 mm gate                                                       |
| Terminal RMS                      | 40.30135 mm                       | Fails 35 mm gate                                                        |
| Club cluster RMS                  | 8.38853 mm                        | Passes 60 mm gate                                                       |
| Pelvis yaw error                  | 0.60998%                          | Passes 5% gate                                                          |
| Pinocchio–Simscape marker maximum | 0.0605214 mm                      | Same-input prefix agreement only                                        |
| Optimizer outcome                 | 60 physical evaluations; fallback | accepted=false; optimizer_converged=false; improved_uninterrupted=false |

The two-window segmented/uninterrupted terminal replay gap is about
1.50e-7 m. Thus resetting between windows is not concealing a large motion gap
in this returned candidate. This does not prove optimization derivatives correct.
Run102 varies controls starting at index 4, uses a 0.85 s basis duration and a
node at 0.6 s. Its limited control subspace, scaling, local conditioning and
line-search behavior need diagnosis before increasing budgets.

The current saved MATLAB file contains finite q, qd, qdd and omega (each
307 by 27), and finite marker predictions (307 by 25 by 3). **All 8,289 tau
entries are nonfinite.** This is missing exported effort evidence, not proof
that the applied simulation input was nonfinite. Preserve the original file.
A report may reconstruct commanded inputs from the candidate only if clearly
labeled as reconstructed; measured joint efforts and constraint reactions need
separate logging, units, frames and qualification.

## Why Progress Is Stalled

1. A small prefix improvement has been confused with completion of the fitting
   problem. Run102 did not improve the uninterrupted objective enough to pass
   its acceptance logic; zero active bounds is not convergence.
2. Reproduction depends on frozen runtime78. The driver imports native_node_chart,
   native_replay and native_sensitivity, absent from tracked main in this review.
   Recover qualified source before running a copied historical launcher.
3. The off-closure weld derivative correction is not integrated. PR #10263 remains
   open and reports mergeable=false at head cced84fa4903dbc0ad6011c04227f7fdb39f5372.
   Coordinate its owner. The finite residual Jacobian and velocity constraint
   Jacobian have different roles; do not replace both indiscriminately.
4. The 23.67696 mm independent rigid-cluster bound relaxes articulation and welds.
   It cannot establish that a connected model can achieve the terminal gate.
5. Product and alternative full-body work are useful but do not advance the
   native fit by themselves. The #10254 full-body receipts use different models
   and must not be ranked against this native 27-coordinate result.

These are evidenced blockers and hypotheses to test, not a proven single cause
of the transition plateau.

## Ordered Work Packages

### 1. Reproduce From an Installable Checkout

Check other agents and remote jobs before launching anything. Recover the exact
missing providers from runtime78/source archives, reconcile shared APIs and
commit them with contracts and tests. Preserve the frozen directory. Use one
config-driven command instead of another numbered driver. Pin dependencies,
model/capture/candidate hashes, initial state, clock, basis and solver settings.

Exit: a fresh environment imports every provider and reproduces run102's
uninterrupted trajectory within declared numerical tolerances, with exact
commands and wall time recorded. Test coefficient order/units, identity restart,
force-frame routing, missing data, immutable outputs and horizon changes.

### 2. Qualify Derivatives and Identify a Useful Direction

Coordinate #10260/#10263, qualify the corrected finite-weld residual derivative
on the actual Pinocchio runtime, then test chart/retraction and shooting defects.
Use centered directional differences over several step sizes at feasible and
off-closure states. Check marker, yaw, effort and defect derivatives separately.

Compute scaled equality-projected gradients and log predicted versus actual
objective change for a few feasible perturbations. Record accepted/rejected
steps and line-search evaluations. A small gradient in the currently restricted
coefficient space does not establish optimality over all sextic coefficients.
If useful directions require lower coefficients, release a small selected subset
with early-motion retention constraints and test its effect.

In parallel with cheap diagnostics, solve the terminal articulated marker fit
with the same geometry, attachments, weld, valid markers and joint limits.
Save feasible poses and residuals. A passing pose proves kinematic feasibility
only; failed local starts are inconclusive. Do not silently remove head markers,
add a neck, soften the loop or relax the 35 mm criterion.

Exit: derivative checks plus a justified fitting change, or a reproducible
specific blocker. Do not launch a long unchanged optimization ladder.

### 3. Fit and Validate a Bounded 0.90 s Extension

First perform a same-input extension with the original polynomial function.
Keep basis duration separate from simulation duration: changing normalized time
without transforming coefficients changes the already-fitted input. Warm-start
shooting nodes from integrated states; add a node near 0.85 s only if needed.
Use one global degree-six input per actuator throughout all windows.

Select and record a bounded evaluation budget after timing the baseline. Record
all candidates and restart lineage. Compare both old-prefix and new-horizon
metrics. Independently replay the returned candidate from the original initial
state with no node resets. Check R2025b with the explicit executable on
DeskComputer/ControlTower. For promoted checks use the qualified refined settings
as a starting point: ode15s, RelTol 1e-7, AbsTol 1e-10, MaxStep 1/2880 s;
recheck numerical convergence when the candidate/horizon changes.

Exit: a real 0.90 s fitting result, PASS/FAIL gates, uninterrupted replay,
per-marker residuals and a decision for 0.95 s. Rejected candidates remain
exploratory. Continue incrementally toward 1.813888889 s based on evidence.

Piecewise cubics may provide an exploratory seed, but generally cannot be
converted exactly to one sextic. Any projection must be followed by optimization
and forward replay of the actual global sextic. Do not spend the main budget
perfecting a torque representation that the final product cannot use.

### 4. Deliver the Polished Reusable Workflow

Keep #10285 replay/catalog and #10286 counterfactual analysis separate from the
critical fitting path. Reuse shared store, solver, FK and viewer providers.
The minimum matching product is a config-driven validate/fit/replay/report path,
immutable run manifests, resume from another candidate, and an honest report
showing coverage, acceptance, residuals over time, torque provenance, loop error,
solver convergence, synchronized capture/model animation and multiple views.

Distinguish commanded input, generalized effort, spatial wrench and constraint
reaction. Do not infer all-engine equivalence from marker-only prefix agreement.
Full acceptance requires original-state uninterrupted forward dynamics through
all 654 frames, global sextic inputs, established physical/marker gates and
independent MATLAB R2025b verification. Save every tested checkpoint and exact
next command. Use TDD, DbC, LoD and DRY throughout.

## Local Replay Work in Progress

This review changes documentation only. Three unfinished files predate it:
`src/shared/python/simulation_store/replay_bundle.py`,
`tests/unit/tools/test_saved_simscape_bundle.py`, and the additional capsule test
in `tests/unit/tools/test_tour_matching_viewer_core.py`.
The focused suite currently has 13 passes and 3 failures:

- Importer rejects the real archive because it assumes all tau entries finite.
  Model unavailable optional signals explicitly; never fill with zero or relax
  finiteness requirements for required motion states.
- A malformed-state fixture fails during MAT writing because nested field names
  exceed 31 characters. Correct fixture serialization before claiming the
  intended loader rejection is tested.
- Capsule-start frame test exposes the viewer's follower/body-frame mismatch.
  Apply the physical-frame conversion consistently without breaking the public
  follower-frame FK contract. The standalone saved-run renderer already applies
  that conversion; do not assume the GUI is qualified because its GIF is correct.

Reproduce with:

```powershell
python -m pytest tests/unit/tools/test_saved_simscape_bundle.py tests/unit/tools/test_tour_matching_viewer_core.py -q --no-cov --tb=short --disable-warnings
```

These files are not a completed importer feature and are excluded from this
documentation commit. Preserve them for the replay owner; do not stage all files.
Existing preview: `visuals_returned102/simscape_returned102_cylinders.gif`.
It displays saved R2025b states, explicitly a rejected 0.85 s prefix.

## Copy-Ready Prompt

Resume native Simscape tour-average matching. Read AGENTS.md, CLAUDE.md,
docs/development/HANDOFF.md and
simscape_tour_matching/PROGRESS_REVIEW_20260917.md under docs/development,
then COMPLETION_HANDOFF_20260916.md for supporting provenance.
Recheck current commits, leases and remote jobs; do not overwrite historical
runtime/evidence. Start from run102 as a rejected baseline, not a finished swing.

Execute packages 1–3 in order: restore an installable fitting runtime and
reproduce the prefix; coordinate and qualify the finite-weld derivative fix;
diagnose feasible optimization directions and articulated terminal feasibility;
run one bounded 0.90 s fit with globally shared sextic inputs and independently
replay in MATLAB R2025b. Preserve geometry, markers, closure, initial state and
absolute clock. Extend only from measured outcomes. Record every gate and budget
exit honestly. Do not substitute full-body IK, viewer work or closed issues for
native forward-dynamics progress.

Make the fitting workflow configurable, tested and resumable, reusing existing
providers. Save incremental commits and immutable manifests with exact next
commands, hashes, versions, results and remaining blockers. Fix missing torque
logging before claiming force/effort analysis. Keep replay UI polish on #10285
and counterfactual work on #10286 so they do not consume the fitting critical
path. Hand back a reproducible numerical result and synchronized visual report,
then the evidence-based next horizon. Full completion requires all 654 frames
and independent R2025b acceptance, not merely successful execution.
