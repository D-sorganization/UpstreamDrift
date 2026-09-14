# Validation and Release Gates

## Test-Driven Development

Every behavior starts with a failing test that demonstrates the missing or wrong
result. Record the red command and failure in the PR, implement the smallest
change, run green, then refactor. Test public outcomes, invalid inputs and real
boundaries, not just function existence or mocked success. Use unit tests without
engines, integration tests with explicit optional-dependency markers, and real
scientific runs separately. Never silently replace a missing engine with a mock.

DbC tests exercise preconditions, postconditions and invariants, including
mutation after construction. LoD review rejects deep engine/object access in
orchestration. DRY review checks existing camera, FK, contact, state, shape,
control and persistence facilities before adding utilities.

## Required Test Families

| Family         | Positive Cases                                  | Adversarial Cases                                                         |
| -------------- | ----------------------------------------------- | ------------------------------------------------------------------------- |
| Contracts      | Immutable valid records and lossless round trip | NaN, wrong units/shapes, unknown version, mutation, duplicate IDs         |
| Timing         | PTS, synchronization offset/drift, slow motion  | Nonmonotonic PTS, cuts, duplicate telecine frames, unknown film speed     |
| Projection     | Analytic landmarks and calibrated masks         | Mirror, crop, distortion, behind-camera, wrong pixel convention           |
| Masks          | Body/club separation, partial visibility        | Empty/unknown masks, occlusion, cast shadow, spectators, blurred club     |
| Initialization | Known pose/scale/camera and nonzero velocity    | Depth alternatives, missing address, left/right ambiguity                 |
| Dynamics       | Known controls and continuous replay            | State resets, ghost root forces, bad grip/contact, early termination      |
| Objective      | Known shifts increase residuals                 | Body IoU hides club error, offscreen geometry, all-invalid pixels         |
| Uncertainty    | Calibrated coverage on withheld synthetic truth | Multimodal ambiguity, wrong prior, unknown time/scale, out-of-domain clip |
| Integration    | Canonical/native round trips, result reload     | Missing engines, unsupported body models, fake metre RMSE, stale cache    |
| Product        | Import, correct, fit, cancel, resume, export    | Dependency failure, corrupt session, partial run labeled complete         |

## Gate Profiles

The numbers below are **initial development targets**, not achieved performance
or established scientific accuracy. ST-01 must preregister a versioned profile
after a feasibility pilot, before holdout evaluation. Any adjustment requires
reason, affected claims, reviewer and new version; never loosen a gate merely
to make a failing candidate pass. Final release remains blocked until the
measurement tolerances and physical limits are justified for the selected model.

| Gate                           | Initial Target and Evidence                                                                                                                                                       |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| G0: Input Integrity            | 100% hash/frame/camera/time references valid; all invalid inputs fail deterministically                                                                                           |
| G1: Synthetic Projection       | Known landmark projection error <= 0.5 px; binary reference-mask IoU >= 0.99 on unoccluded fixtures                                                                               |
| G2: Synthetic Recovery         | Median body IoU >= 0.95; 95th-percentile normalized contour error <= 0.01; held-out world joint RMSE <= 0.02 m when identifiable                                                  |
| G3: Modern Reference           | Initial target median body IoU >= 0.90 and independent joint RMSE <= 0.05 m; stratify by phase, view, joint and visibility; separate club metrics required                        |
| G4: Dynamics Replay            | Full requested interval; zero intermediate state resets and zero undeclared root actuation; profile-specific contact, grip, joint, control and energy-balance limits all pass     |
| G5: Robustness and Uncertainty | Nominal 90% intervals evaluated with empirical coverage and interval width; report multimodal/sensitivity sets honestly; all intentionally unidentifiable cases downgrade/abstain |
| G6: Historical Pilot           | Every result traces to reviewed frames, source rights and physical-time assumptions; no SI kinetics qualification with unknown time/scale; degradation-class results reported     |
| G7: Product and Reproduction   | Saved bundle replays within pinned numeric tolerances; cancellation/resume works; PyQt/React capability parity and no missing-dependency fake success                             |

For G2/G3, normalized contour error uses distance divided by image diagonal.
Report pixel error too. Score body and club separately. ST-01 must lock clubhead
and shaft error limits appropriate to resolution and exposure; missing club
evidence is a limitation, not a passing zero. Always report worst phases and
failed frames, not just favorable medians.

G4 limits must include maximum penetration, friction-cone residual, foot slip,
grip closure, joint-limit exceedance, peak torque and power, and actuator rate.
Use SI values plus normalized values where helpful. Account for contact
dissipation and external work when checking energy; a driven dissipative swing
does not conserve mechanical energy. Compare integration-step refinement and
independent replay state discrepancies with pinned tolerances.

## Benchmark Design

1. Analytic tiny fixtures establish projection and loss correctness independent
   of the production renderer.
2. Synthetic known full-body rollouts establish end-to-end recovery and known
   ambiguous cases. Evaluate both matched and mismatched visual/dynamics models
   to expose the inverse-crime risk of generating and fitting identical models.
3. Modern simultaneous calibrated video and independent reference measurements
   test actual reconstruction. Reserve cameras, subjects and sessions; references
   used for evaluation cannot initialize the silhouette-only benchmark.
4. Downsample, blur, occlude, interlace, re-time, crop and move the camera on
   reference data. Measure degradation curves and abstention, not one score.
5. Historical pilot tests workflow and conditional claims; archive footage
   without ground truth cannot establish absolute 3D accuracy by itself.

Compare silhouette-only, keypoint-only, combined, kinematic-only, physics-fitted,
single-view and multiview variants with equal budgets. Include fixed/wrong camera
and shape ablations. Report uncertainty calibration and coverage separately
from accuracy. Keep test-set identities and hashes immutable before tuning.

## Evidence Receipt

Each gate run must save commit, dependency/engine/model/asset hashes, environment,
seed, exact command, objective weights, camera/time hypotheses, split membership,
per-frame/per-view/per-phase metrics, exclusion reasons, candidate IDs, raw audit
values, profile version and pass/fail reasons. Distinguish optimizer output from
fresh replay. Save source/mask/render contour overlays and residual maps at
address, takeaway, transition, impact window and follow-through plus worst frames.

## Commands and Governance

Planned implementation commands, runnable after test modules exist:

```bash
python3 -m pytest tests/unit/shadow_tracker -n auto --timeout=60
python3 -m pytest tests/integration/shadow_tracker -m "not live_simulation" --timeout=60
python3 -m pytest tests/integration/shadow_tracker -m live_simulation --timeout=60
python3 -m ruff check src/shared/python/shadow_tracker tests/unit/shadow_tracker tests/integration/shadow_tracker
python3 -m ruff format --check src/shared/python/shadow_tracker tests/unit/shadow_tracker tests/integration/shadow_tracker
```

Run affected existing consumers and repository CI/type/coverage gates as required
by `CLAUDE.md`; these focused tests do not waive them. Scientific batch jobs may
need a separately declared timeout/runner. For Simscape, use MATLAB R2025b
explicitly and its test/evidence harness. Apply design-manual governance before
shipping public scientific pathways. Software correctness, scientific
qualification, and publication approval are separate statuses.
