# Work Packages

## Execution Rules

GitHub issues are the task authority; [roadmap.json](roadmap.json) stores IDs and
dependencies. Each work package may require several narrow PRs. An agent takes
one bounded slice, not the whole epic. No later milestone may treat an open
dependency as implemented. Sizes are rough effort classes: S = a small PR,
M = several focused PRs, L = research or multiple integration slices; they are
not delivery promises. “Small Agent” means low-context implementation following
frozen contracts, with review. Research decisions require a senior reviewer.

Every implementation package includes red/green evidence, contract/invalid-input
tests, LoD/DRY review, changed-boundary tests, provenance and handoff updates.

## ST-00: Establish the Planning Baseline

- **Outcome:** A new agent can find intent, interfaces, risks, gates and work.
- **Depends On:** None. **Size:** M. **Owner:** Planning Agent.
- **Scope:** This document set, reserved source/test homes, epic and children,
  repository discovery links, SPEC and current handoff.
- **Acceptance:** All local links resolve; roadmap IDs/dependencies are valid;
  every package has a test-first acceptance plan; no runtime readiness claims.
- **Turnover:** State precisely which files exist and which modules are planned.
  Close only this setup issue through the planning PR, never the parent epic.

## ST-01: Qualify Feasibility and Freeze the Benchmark Profile

- **Outcome:** Expensive work starts with a representable model and measurable gates.
- **Depends On:** ST-00. **Size:** L. **Owner:** Dynamics/Scientific Reviewer.
- **Scope:** Model capability inventory, head/limb/club/grip fidelity, initial
  engine benchmark, result/rollout adapter design and gate-profile registration.
- **First Tests:** Reject upper-body-only qualification; verify native state
  mapping; replay known controls twice with a frozen ground plane.
- **Acceptance:** Measured pilot evidence, chosen engine/config hashes, exact
  G2–G5 limits including club/contact/replay, holdout policy and reviewer rationale.
  Identify unsupported capabilities and separate follow-up issues as needed.
- **Do Not:** Infer readiness from class names, mock success or prior epic closure.
- **Next Slice:** Add a real model-capability receipt before tuning any solver.

## ST-02: Implement Immutable Evidence Contracts

Dispatch update: [ST-D7](CONTRACT_FREEZE.md) permits image-only Packets A/C/B
(#10137/#10139/#10138) independently of scientific qualification. Full ST-02
completion still depends on ST-01 for dynamics/provider/result contracts.
See [Ready Tasks](READY_TASKS.md); this is not a waiver of any physics gate.

- **Outcome:** Every frame, mask, camera and result has unambiguous meaning.
- **Depends On:** ST-01. **Size:** M. **Owner:** Small Agent + API Reviewer.
- **Scope:** `contracts.py`, serialization fixtures, typed provider interfaces;
  reuse existing camera/state primitives and shared validation helpers.
- **First Tests:** Valid record round trip; malformed dimensions, NaN, unknown
  schema, duplicate timestamps/IDs, invalid transforms and mutation all fail.
- **Acceptance:** Versioned schemas, explicit timing/units/unknowns, legacy
  camera conversion tests, bounded asset paths, and zero optional engine imports.
- **Do Not:** Implement the fitter or invent a third canonical coordinate system.
- **Next Slice:** Frame/mask record tests followed by the minimum implementation.

## ST-03: Preserve Source Video, Shots, Timing and Capture Evidence

- **Outcome:** Modern and archive clips enter one auditable observation pipeline.
- **Depends On:** ST-02. **Size:** M. **Owner:** Small Agent + Video Reviewer.
- **Scope:** `ingestion.py`; existing decoder/Capture Rig adapters, source catalog
  and timestamp/transform provenance. Local assets first.
- **First Tests:** Variable PTS, cut, mirror/crop, slow motion, telecine duplicates,
  synchronization offsets/drift, corrupt media, different-swing fusion rejection.
- **Acceptance:** Original/processed frame identity and time round trips; source
  checksums and permissions metadata; unknown physical time survives import.
- **Do Not:** Download archives in bulk, interpolate observations or assume FPS.
- **Next Slice:** Tiny redistributable local video fixture and PTS preservation.

## ST-04: Produce and Review Body/Club Silhouettes

- **Outcome:** The intended golfer's outline is available with correction history.
- **Depends On:** ST-03. **Size:** M. **Owner:** Vision Agent + Reviewer.
- **Scope:** `segmentation.py`; manual-mask baseline, optional pinned model
  provider, occlusion/identity tracking and revision-aware cache invalidation.
- **First Tests:** Empty vs unknown masks, body/club separation, identity loss,
  occlusion, manual edit persistence and downstream hash invalidation.
- **Acceptance:** Curated gold-mask agreement and correction-time report;
  historical blur/clothing/club cases; lazy missing-provider errors; reproducible
  model/license/checkpoint manifest. No unqualified automatic mask confidence.
- **Do Not:** Fit dynamics or count restored/generated frames as observations.
- **Next Slice:** Deterministic manual provider before a neural dependency.

## ST-05: Render Calibrated Silhouettes and Compute Residuals

- **Outcome:** Any candidate pose can be compared fairly with every valid image.
- **Depends On:** ST-02. **Size:** M. **Owner:** Geometry Agent + Reviewer.
- **Scope:** `projection.py`, body-shape renderer adapter and body/club losses.
- **First Tests:** Analytic projection, crop/distortion/mirror parity, occlusion,
  offscreen body, all-invalid mask, known contour shift and thin-club mismatch.
- **Acceptance:** G1 passes; renderer comparison documented; residuals normalized
  and valid-pixel aware; same geometry binding as the dynamics skeleton.
- **Do Not:** Use skeleton screenshots as ground-truth rasterization or duplicate
  camera math already present in pose estimation/estimation services.
- **Next Slice:** Known primitive in a known camera, independent pixel oracle.

## ST-06: Fit Subject Shape and Initial-State Hypotheses

- **Outcome:** The model starts in a plausible image-consistent pose and velocity.
- **Depends On:** ST-04, ST-05. **Size:** L. **Owner:** Estimation Reviewer.
- **Scope:** `initialization.py`; fixed morphology, camera/scale/handedness
  hypotheses, short-window velocity and optional kinematic warm start.
- **First Tests:** Recover known pose from calibrated views; ambiguous monocular
  scene retains alternatives; missing address and nonzero velocity are handled.
- **Acceptance:** Separate visual/inertial parameters, bounded priors, observed
  vs inferred labels, reproducible candidates and sensitivity to camera/scale.
- **Do Not:** Assume a single address frame identifies depth or zero velocity.
- **Next Slice:** Fit initial pose with known camera/shape before releasing both.

## ST-07: Connect a Real Full-Body Forward Rollout

- **Outcome:** A candidate executes from one initial state with auditable physics.
- **Depends On:** ST-02, ST-05. **Size:** L. **Owner:** Dynamics Reviewer.
- **Scope:** Engine-local adapter; decouple existing full-body rollout from
  mandatory marker scoring; canonical/native mappings and realized-control audit.
- **First Tests:** Existing tour rollout regression; independent repeat replay;
  state/velocity round trip; detect mid-run resets and undeclared root forces.
- **Acceptance:** Real full-body engine; full interval, control basis/units and
  ground calibration preserved; no fake `TourCapture`; contacts/grip auditable;
  explicit unsupported/missing-engine failure. G4 profile exercised.
- **Do Not:** Rewrite dynamics, change existing fitted models or access engine
  private members from the new shared orchestration layer.
- **Next Slice:** Characterize current rollout behavior before boundary extraction.

## ST-08: Optimize Controls Against the Silhouette Sequence

- **Outcome:** Forward dynamics, not per-frame pose injection, explains the swing.
- **Depends On:** ST-06, ST-07. **Size:** L. **Owner:** Optimization Reviewer.
- **Scope:** `fitting.py`; bounded staged control optimization, objective
  breakdown, budgets, checkpoints and fresh replay scoring.
- **First Tests:** Known-control synthetic recovery; lower image loss cannot
  override failed physics; interrupted and divergent runs preserve honest status.
- **Acceptance:** G2 and G4; full continuous pre-impact proof then explicit
  impact/follow-through qualification; real engine evidence; compare equal-budget
  kinematic/keypoint/silhouette baselines; no success from solver status alone.
- **Do Not:** Train a new policy, increase control freedom or loosen gates without
  a reviewed decision and benchmark evidence.
- **Next Slice:** Recover one low-dimensional known control perturbation first.

## ST-09: Quantify Ambiguity and Implement Evidence Gates

- **Outcome:** Users can distinguish stable conclusions from unsupported guesses.
- **Depends On:** ST-08. **Size:** L. **Owner:** Scientific Reviewer.
- **Scope:** `evaluation.py`; candidate families, sensitivity, gate reasons,
  uncertainty coverage and abstention.
- **First Tests:** Two different 3D poses with indistinguishable silhouette;
  unknown time blocks SI kinetics; hidden-club case does not return zero error.
- **Acceptance:** G5; held-out coverage/width, camera/timing/mass/contact ablations,
  per-quantity confidence and explicit separation of posterior vs sensitivity
  ranges. Failed physical candidates cannot be exported as validated.
- **Do Not:** Call optimizer curvature or multistart spread calibrated confidence.
- **Next Slice:** Deterministic evidence-status rules and their negative tests.

## ST-10: Qualify Modern and Historical Footage Workflows

- **Outcome:** Real footage is processed with measured yield and defensible limits.
- **Depends On:** ST-03, ST-04, ST-09. **Size:** L. **Owner:** Data/Scientific Reviewer.
- **Scope:** Reference cohort, degradation harness, reviewed historical pilot,
  deduplication and resumable archive catalog.
- **First Tests:** Same-film re-encoding cannot cross split; source rights/identity
  unknown remains flagged; impossible synchronization cannot enable multiview.
- **Acceptance:** G3/G6, locked holdouts, failed-case inventory and source lineage;
  measured annotation and compute cost; modern reference timing and calibration.
- **Do Not:** Claim archive ground truth or select only easy successful clips.
- **Next Slice:** Ground-truth modern clip with archive-like degradation and abstention.

## ST-11: Integrate Review, Jobs and Model Exports

- **Outcome:** Users can review, correct, fit, replay and export within UpstreamDrift.
- **Depends On:** ST-09. **Size:** M. **Owner:** Product Agent + API Reviewer.
- **Scope:** `artifacts.py`, `service.py`, existing job/video/capture presenters,
  canonical exports and PyQt/React capability registry updates.
- **First Tests:** Save/reload preserves uncertainty; correction invalidates fits;
  cancel/resume cannot become complete; missing backend gives an actionable error.
- **Acceptance:** G7; source/mask/render overlays, worst-frame navigation,
  uncertainty/assumptions visible, typed API schema, export provenance and parity.
- **Do Not:** Register an unfinished ready tile or present generated markers as mocap.
- **Next Slice:** Headless bundle round trip before UI wiring.

## ST-12: Qualify Advertised Engines and Release

- **Outcome:** Shadow Tracker ships with reproducible and bounded scientific claims.
- **Depends On:** ST-10, ST-11. **Size:** L. **Owner:** Release/Scientific Reviewer.
- **Scope:** Engine conformance matrix, independent replay, performance profile,
  scientific manual/registry, documentation and release evidence.
- **First Tests:** Supported-engine declarations require real receipts; wrong
  conventions, model hashes or MATLAB release fail qualification.
- **Acceptance:** All G0–G7 for the released profile, documented unsupported
  engines, independent review and complete swing coverage; user guide and
  agent turnover; R2025b evidence for Simscape if advertised.
- **Do Not:** Block a first-engine release on unadvertised engines, or imply
  identical contact results across materially different physical models.
- **Delivery:** Delivered in #10135. Implemented `src/shared/python/shadow_tracker/engine_matrix.py`,
  with full engine conformance checking, fail-closed MATLAB R2025b validation for Simscape,
  model hash verification, independent replay auditing, performance profiling, complete
  G0–G7 gate validation across all six core swing phases, scientific calculation registry,
  and deterministic release evidence inventory generation. Verified by
  `tests/unit/shadow_tracker/test_engine_matrix.py`.
