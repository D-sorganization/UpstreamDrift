# Simscape Matching Progress Review and Execution Turnover

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
