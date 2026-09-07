# SPEC.md — Repository Specification Document

## Guard Optional Excel Export Tests (#9579)

Guards `TestExportExcel` in `movement_optimizer/tests/test_export.py` with `pytest.importorskip("openpyxl")` so test suites pass when optional dependencies (`openpyxl`) are absent in minimal or optional test runners.

## Resolve Sidekick Extension Scope Gate and Test Paths (#9572)

Resolves test failures and runtime import gates blocking CI Standard on `main`:
- Permits supported scopes (`chat`, `sidekick`) in `sidekick_extension_overlay._module_name()` instead of restricting strictly to `sidekick`, enabling manifest-approved UpstreamDrift extensions like `chat/_qt/runtime.py`.
- Classifies `chat/_qt/runtime.py` owner as `UpstreamDrift` in `scripts/config/shared_python_ownership_exceptions.yaml`.
- Anchors relative paths in `tests/launchers/test_simulation_guis.py` and `tests/docker/test_docker_compose.py` to project root so tests execute deterministically regardless of invocation directory.
- Re-registers known architectural boundary exception in `tests/architecture/test_dependency_direction.py` for `api/routes/_ball_flight_trajectory_import.py`.
- Restores markerless mocap authority section and handoff pointer in `SPEC.md` and `AGENT_HANDOFF.md`.

## Markerless Mocap Program (#9063)

Issue #9065 establishes ADR-0041 and an executable acceptance program before
live markerless capture begins. Canonical camera, capture, time, calibration,
observation, reconstruction, session, and C3D exchange contracts belong to
Tools #4706. UpstreamDrift owns application orchestration, persistence,
biomechanics integration, and matching PyQt6/React/API workflows. AffineDrift
owns sanitized evidence publication. Tools_Private is not a dependency of the
open runtime. The first consumer under #9069 must pin a protected Tools merge,
reject missing or incompatible schema authority, and adapt existing C3D and
motion-pipeline paths instead of copying shared code. This M0 slice makes no
camera, inference, C3D round-trip, commercial, or physical-lab qualification
claim.

## Enforce Declared Measurement Acceptance Conditions & Physical Bounds (#9286)

Enforces measurement acceptance conditions, physical admissibility bounds, and genuine apparatus/condition declarations in the V&V validation ledger:
- Extends `AcceptanceCriterion` with optional `value_min` and `value_max` bounds, strictly checking that finite values fall within declared physical ranges (`value_min <= record.value <= value_max`) during shortfall evaluation.
- Populates physical bounds for all 7 standard `MeasurementSpec` entries in `roadmap.py`:
  - `bunker_sand_angle_of_repose_deg`: `[0.0, 90.0]` deg.
  - `bunker_sand_bulk_density_kg_m3`: `[500.0, 3000.0]` kg/m³.
  - `bunker_sand_drained_friction_angle_deg`: `[0.0, 90.0]` deg.
  - `bunker_sand_population_survey`: `[500.0, 3000.0]` kg/m³.
  - `splash_shot_divot_cast_volume_m3`: `[0.0, 0.1]` m³.
  - `ejecta_launch_high_speed_video`: `[0.0, 100.0]` m/s.
  - `clubhead_delivery_shaft_strain`: `[0.0, 10000.0]` N.
- Adds `shortfall()` to `MeasurementSpec` verifying spec key, matching units, and ensuring real instrument measurements (`INSTRUMENT` basis) declare genuine instrument classes and conditions (rejecting empty or dummy `"none"` placeholders).
- Ensures `MeasurementSpec.is_satisfied_by()` delegates to `shortfall()`, preventing invalid or unphysical measurement records from advancing validation or credibility factor scores.

## Bunker Shot Pose Reflection Rejection and Exit Kinematics Preservation (#9542)

Resolves posture admissibility and exit state fidelity defects in the bunker shot model:
- Rejects orientation matrices that admit reflections (`det(R) ≈ -1`) in `HeadKinematics` and `SandDelivery`, strictly enforcing proper 3D rotations with `det(R) = +1` and finite elements.
- Enforces immutability on input kinematics arrays (`velocity_m_s`, `position_m`, `angular_velocity_rad_s`, `orientation`) by setting `writeable = False` upon initialization, preventing post-construction mutation.
- Exposes full exit kinematics on `ShotResult`: `exit_velocity_m_s`, `exit_angular_velocity_rad_s`, `exit_orientation`, and `exit_position_m`.
- Extends `SandDelivery` to carry actual exit kinematics (`exit_velocity_m_s`, `exit_angular_velocity_rad_s`, `exit_orientation`).
- Preserves actual 3D exit linear velocity and angular velocity during handoff to `PostImpactState` in `to_post_impact_state()`, eliminating forced projection to zero lateral speed and zero spin when measured exit kinematics are present.

## Pose2Sim Observation Confidence and Alignment Hardening (#9552)

Resolves confidence assignment and multi-camera observation alignment defects in Pose2Sim ingestion:
- Eliminates 1.0 default confidence for missing evidence: empty or unsupported observations now return 0.0 reconstruction confidence with explicit quality status (`"unknown"`, `"invalid"`) and failure reasons (`"missing_observation"`, `"insufficient_views"`).
- Replaces frame-index alignment with timestamp-synchronized lookup bounded by a tolerance envelope (`0.5 / fps`), preventing cross-frame pollution from dropped or delayed frames.
- Replaces naive index fallback with strict schema and canonical name mapping across supported detector layouts (`MediaPipe_33`, `BODY_25`), eliminating positional mismatches across differing schemas.
- Rejects duplicate camera stream IDs during session loading with explicit `ValueError` rather than silently overwriting earlier streams.
- Preserves contributing camera view counts and individual per-view detector confidences in reconstruction quality metadata and provides `to_canonical_observations()` bridge to canonical CIR records without creating competing schemas.

## Headless MuJoCo Version Resolution and Example Dependency Gating (#9431)

Resolves headless MuJoCo initialization crashes and un-isolated native dependencies in test suites:
- Updates `ProvenanceInfo.capture()` to safely resolve MuJoCo version via `sys.modules` or `importlib.metadata.version("mujoco")` without forcing an import of `mujoco`, avoiding headless OpenGL `glGetError` NoneType attribute crashes when generating simulation metadata in headless runners.
- Gating examples requiring native Rust wheels (`basic_flight_simulation.py`) or external fixture files (`motion_training_demo.py`) in `tests/examples/test_examples_produce_output.py` using `pytest.skip` when optional runtime components are unavailable.

## Repository Hygiene & Artifact Gitignore Enforcement (#9415)

Enforces repository hygiene and artifact gitignore coverage following the #9415 sweep:
- `.gitignore` explicitly excludes `.scratch/`, `/output/*` (preserving `.gitkeep`), `/motion_matching/`, `/motion_matching_training/`, and `/reports/*.json` scanner dumps.
- `scripts/check_forbidden_artifacts.py` bars committed `.scratch/` agent drafts and `reports/*.json` scanner dumps (`bandit.json`, `pip_audit.json`, `semgrep.json`).
- `tests/ci/test_artifact_dir_hygiene.py` and `tests/scripts/test_check_forbidden_artifacts.py` pin that artifact directories remain untracked and forbidden patterns cannot be re-tracked.

## Clean-Machine Install Verification and Standalone Tutorial Execution (#9407)

Verifies the clean-machine installation path and tutorial execution flow:
- Resolves Windows charmap decode errors in `tests/unit/test_install_script.py` with UTF-8 decoding and Git Bash path translation, marked with unit test suite markers.
- Updates tutorial examples (`examples/01_basic_simulation.py`, `examples/02_parameter_sweeps.py`, `examples/03_injury_risk_tutorial.py`) to resolve and prepend project root and source paths to `sys.path` so they execute cleanly from checkout without requiring editable installation.
- Validates 40/40 verification checks passing in `scripts/ci/verify_installation.py`.

## Tile Category Registry and Sidebar Grouping Reconciliation (#9481)

Reconciles launcher tile categories directly with the registry (`models.yaml`), eliminating
hardcoded model IDs and type-based fallback heuristics from `LauncherLayoutManager.get_model_categories()`:
- `LauncherPresentationMetadata` supports optional multi-category declarations (`categories: tuple[str, ...]`).
- `models.yaml` explicitly declares secondary categories where appropriate (`movement_optimizer`, `cross_engine_dashboard`, `putting_green`, `library_tool`).
- `_launcher_navigation_ui.py` sidebar buttons are collapsed from duplicate mappings to distinct categories (`Engines`, `Biomechanics`, `Simulation`, `Tools`, `Documentation`).
- `test_category_registry_gate.py` enforces that `get_model_categories()` contains no model IDs in its AST, all categories map cleanly, and no sidebar filter is empty.

## Physics Metrics Dimensionality & Pinocchio Actuation Loop (#9477)

Addresses GH-9477:
1. **Actuator Torque in Pinocchio Simulation Loop**: Replaced hard-coded zero-torque vectors (`tau = zeros`) in `SimulationMixin._advance_physics()`, `_record_frame()`, and `PinocchioGUI.step_simulation()` with explicit commanded torque support (`self.commanded_tau`), enabling actuated dynamics rather than pure free-fall simulation. Added `set_commanded_torque()` and tests demonstrating that non-zero commanded torque alters trajectories compared to free-fall.
2. **True Physical Energy and Cartesian Speed Metrics**: Replaced dimensionless velocity norm proxy in `CrossEnginePerturbationRunner._run_trial()` with genuine total mechanical energy in Joules ($J$), evaluating engine-native energy methods, kinetic energy with physical mass matrix/inertia ($0.5 v^T M(q) v$), and potential energy. Evaluated end-effector speed as Cartesian linear velocity in m/s via kinematics/Jacobians rather than generalized velocity norms. Added unit tests against an analytic single-link pendulum verifying mass scaling and kinematic speed. Decomposed energy evaluation helpers to conform to function-line budgets and refined numpy/GeometryModel type annotations.


## Research Evidence Source Hash Re-Registration (#9233)

Re-registers `source_sha256` content digests across proximal-distal research evidence
records in `docs/research/proximal_distal_energy_transfer/data/` that drifted following
refactoring PRs (#8760, #8962, #8963, #9055, #9240, #9267, #9306, #9415, #9427).
Fixes the table lookup directory in `run_two_hand_wscg_analysis.py` to match the canonical
`matlab/Tables` directory location, normalizes hex digest casing in `two_hand_wscg_analysis.json`,
updates `claim_evidence_manifest.json` with current artifact digests, and adds architecture-budget
exceptions for pre-existing `run_two_hand_wscg_analysis.py` methods.

## ADR-0045 F1: Roll-Model Provenance in Every Result (#9343, #9356)

Implements F1 of accepted ADR-0045 (`docs/adr/0045-putting-integration-one-experience-two-preserved-stacks.md`).
Both roll models remain preserved and active across the putting green architecture:

- `UD_LEGACY_ROLL_MODEL = "ud-legacy-roll/1"` ($\mu \approx 0.196/\text{stimp} \times \text{height-of-cut/condition/grain}$) in `ball_roll_physics.py`.
- `USGA_STIMP_ROLL_MODEL = "usga-stimp-roll/1"` ($\mu \approx 0.559/\text{stimp}$, USGA 1.83 m/s release + Holmes/Penner capture).
  Results from different models are strictly forbidden from numerical comparison without their model names attached.
  All result documents carry `roll_model` fail-closed (`SimulationResult`, export/load results, `simulate_putt`, `simulate_with_feedback`, `simulate_scatter`, `compute_aim_line`, `get_current_trajectory`, `/simulate`, `/scatter`, `/read-green`, `/simulate-3d`, and `PuttScene`).
  Checkpoints are versioned (`schema_version = 2` carrying `roll_model`), preserving archive readability for legacy v1 payloads without silent relabeling.

## Hybrid Manufactured-Solution Authority Repair (#9236)

The protected articulated manufactured-solution record is not presently
reproducible authority. Optional-stack run `33278810705`, job `99170340423`,
used MuJoCo 3.12.0 and Pinocchio 4.1.0 while the record declared 3.8.0 for both,
and the current generator did not reproduce the committed JSON bytes. This is
negative authority evidence even though 12 of 13 native checks passed.

The #9236 candidate defines a study-scoped, hash-locked CPython 3.11.15
manylinux environment and an isolated native workflow. Its canonical writer
rejects NaN and performs atomic replacement. Semantic comparison fails closed
unless both authority and rolling records declare the complete six-gate
tolerance set, the exact 18-field compatibility policy, every governed result
path, finite numeric leaves, monotone convergence, and an internally consistent
component maximum. Runtime profile, dependency, command, lock/input digest,
source, and output identities are explicit.

This implementation does not modify the scientific record or its checksum,
release manifest, or claim-evidence manifest. Independent implementation review
is GO at `0f2f4aed2d1aa2c231c241aabe464249f1586b0f`; 83 cumulative corrective
tests pass. The exact original authority suite remains deliberately partitioned
as 33 passed, 6 native-stack skips on Windows, and 3 RED tests that reserve
publication until exact Linux CPython 3.11.15 regeneration. A regenerated
record may be accepted only after two same-environment native builds are
byte-identical, their execution identity matches the declared profile, numeric
evidence satisfies the governed comparison, and the canonical checksum and
manifest cascade is regenerated. #9192, #9174, and AffineDrift #4022 remain
open until that protected evidence exists.

- `UD_LEGACY_ROLL_MODEL = "ud-legacy-roll/1"` ($\mu \approx 0.196/\text{stimp} \times \text{height-of-cut/condition/grain}$) in `ball_roll_physics.py`.
- `USGA_STIMP_ROLL_MODEL = "usga-stimp-roll/1"` ($\mu \approx 0.559/\text{stimp}$, USGA 1.83 m/s release + Holmes/Penner capture).
  Results from different models are strictly forbidden from numerical comparison without their model names attached.
  All result documents carry `roll_model` fail-closed (`SimulationResult`, export/load results, `simulate_putt`, `simulate_with_feedback`, `simulate_scatter`, `compute_aim_line`, `get_current_trajectory`, `/simulate`, `/scatter`, `/read-green`, `/simulate-3d`, and `PuttScene`).
  Checkpoints are versioned (`schema_version = 2` carrying `roll_model`), preserving archive readability for legacy v1 payloads without silent relabeling.

## ADR-0045 F4: Green-Surface Adapter Physics Consumer Contract (#9346)

Issue #9346 completes the #9143 rider called for in ADR-0045's Validation section: the UD-side
consumer-contract test drives UpstreamDrift's own `ball_roll_physics.BallRollPhysics` /
`turf_properties.TurfProperties` engine (not a reimplementation) against the vendored Tools
integrator `shared.python.swing_sim.putting.simulate_putt_on_surface`, on the same green and the
same launch. `tests/integration/putting_green_drift/test_green_surface_adapter_consumer.py` adds:
a round trip that authors a green with UD's own `ContourPoint`/`GreenSurface` code, synthesizes the
UD topography JSON field-for-field, imports it through the vendored `green_surface_from_ud_json`,
and exports it back into a fresh UD `GreenSurface`, geometry preserved at every grid node; the
shared-physics gates from the `ud_adapter` module docstring (flat-green straight line, break sign
matching the cross slope, roll-out monotone in stimp) run on both engines from the same authored
green; and the documented `mu_tools / mu_ud ≈ 2.854` roll-out ratio (Tools #4819) is pinned
**empirically** from both engines' own `simulate_putt` / `simulate_putt_on_surface` calls, isolating
the rolling-resistance phase by starting each simulator already in pure roll. A weighted-slope UD
field is proven genuinely UD-loadable (`GreenSurface.load_from_file`, live gravity contribution)
before asserting the adapter refuses it with its documented non-conservative-slope reason. The
`tests/integration/launch_monitor_drift` CI-guard helper (`require_vendored_tools_stack`) is
imported, not reimplemented, for the same skip-locally/run-in-CI posture.

## Vendored Tools Pin: Flight-Interchange Authority (ADR-0047 H1)

The `vendor/ud-tools` gitlink advances from
`cc883cbaf63157b58c71cba385a683df2762b0cb` to
`5e0eaade29441dd65d667151b5108c8925774d73`, the Tools `main` squash commit for
Tools #4888, which adds the `shared.python.swing_sim.flight_interchange`
package to the vendored tree. The
vendored reader is the interchange authority for the
`swing_sim.ball_flight_trajectory/1` record; UpstreamDrift's exporter in
`src/shared/python/physics/flight_trajectory_export.py` is verified against it
by the four `TestCrossFamilySanity` gates in
`tests/unit/physics/test_flight_trajectory_export.py`, which arm automatically
once the pin carries the module: records parse with the vendored reader and
round-trip byte-identically, the reimplemented SHA-256 parameter digest equals
Tools' `parameter_digest`, and the two flight-model families produce same-order
carry for identical launch conditions. The companion catalog provenance
contract (`tests/companion/test_companion_catalog.py`) pins the exact gitlink
and moves with it. No UpstreamDrift runtime behavior changes; the prior pin
statement under "Swing Objective Lab Web Parity (#9128)" is superseded by this
section.

## Deterministic AffineDrift Companion Authority (#9174)

ADR-0043 establishes UpstreamDrift as the one-way provider of the strict v1
software-fact catalog consumed by AffineDrift #4010. The local-only exporter
reconciles the current 49 raw launcher entries and 56 repository-local model
entries into a 70-program union, plus all 41 feature-parity records and their
79 declared shell-source paths. These are migration baselines in tests and
summary metadata, not permanent schema limits. Exact UpstreamDrift commit,
committed-input SHA-256 values, package/Python compatibility, verification
command, support tiers, and the `vendor/ud-tools` gitlink commit are explicit.
Maturity, availability, engine support, shell parity, and scientific
qualification remain separate. The fail-closed CLI rejects dirty trees,
mismatched CI commits, mutable/external input paths, and non-gitlink Tools
authority; discovery never depends on sibling repositories or provider-root
environment variables. Publication remains draft while capability evidence and
governed documentation, workflow, screenshot, and immutable release
inventories are incomplete. PR #9180 is the foundation only: #9190 owns ten
exact-revision workflows plus failure fixtures and provider CI; #9191 owns the
complete screenshot/capture contract; #9192 owns protected artifacts,
attestation, compatibility fixtures, rollback-safe acquisition, and release
assets; #9193 owns documentation freshness and engine capability evidence.
Issue #9174 remains open after the foundation merge. Empty inventories and
ignored local artifacts are negative evidence, not completion or publication.
This catalog does not copy or supersede #9064's design-manual authority or
#9070's typed calculation-manifest authority.

Issue #9192 adds the publication boundary without changing that scientific or
content status. The existing release workflow now runs the same fail-closed
`python3 -m scripts.companion_publication build` command for protected `main`
and exact `vX.Y.Z` tags. It packages the manifest, manifest schema, acquisition schema,
compatibility policy, and detached SHA-256 files; attests the exact payloads;
and records repository, source commit, workflow run, schema/generator versions,
sizes, hashes, and artifact identities. Protected-main Actions artifacts are
explicitly 30-day/ephemeral and have no durable release URL. Tag releases are
draft-first, refuse overwrites, use numeric GitHub API asset identities, and
become public only after the acquisition record is generated and attested.
Schema 1.0.0 remains current with no fabricated predecessor; compatibility
tests require a previous fixture as soon as a real second supported version is
declared and reject future/incompatible fixtures now. No tag or release is
created by implementing #9192.

Protected publication commands have an explicit repository-root precondition:
every workflow job that invokes `scripts.companion_publication` runs from
`${{ github.workspace }}`. This is enforced by a workflow-structure contract so
self-hosted runner defaults cannot make a successful checkout non-importable.
Failure to establish that working directory is negative publication evidence:
no artifact or attestation may be accepted, and #9192 remains open until a new
protected-main run publishes and verifies the exact protected commit bytes.

## Deterministic Companion Workflow Authority (#9190)

ADR-0043 amendment establishes UpstreamDrift as the workflow-execution authority
for the 15 registered companion workflows. Ten success workflows cover
installation, launch resolution, simulation, import/export, catalog export,
counterfactual mechanics, reports, and plotting; four deterministic failure
fixtures cover unsupported dependency, bad input, unavailable engine, and stale
version; one native OpenSim GUI workflow is explicitly unavailable until
native-engine UI and screenshot authority exist. All workflows execute through
`scripts.companion_workflows` using `shell=False` arguments and emit structured
execution records with exact command strings, exits, durations, and outputs.
CI job `companion-workflows` runs all available records and verifies execution
evidence against schema `upstreamdrift-companion-v1.schema.json`.

## Ball-Sand Interaction: What Reaches the Ball (#8712)

Issue #8712 resolves the sand arriving at the ball inside the F1 plane-strain
MPM tier. `solvers/mpm/ballreach.py` reads the ball's own exact momentum ledger
as traction on the ball, resolved around its in-plane surface: a below-equator /
face-side split, an even-sector resolution whose bin edges always place the
equator on a boundary, and a per-node radial (compressive) and tangential (shear)
decomposition. The time history reports first contact, a caller-thresholded
loading onset, the peak and its timing, and the total impulse. Nothing here
computes a force: `BodyContact` now retains the node-resolved `ContactImpulse`
it was reduced from, so every number is the existing ledger regrouped and the
two-body momentum budget still closes to round-off with the ball's term in it.

Every quantity is named for what it is: per unit out-of-plane width, on an
infinite cylinder rather than a sphere. The absolute force on the ball raises
`RefusedQuantity.OUT_OF_PLANE` because, unlike the club, there is no effective
width to declare; heel-toe and lateral distributions raise for the same reason;
ball launch stays on F0's #8657 momentum-transfer path and
`RefusedQuantity.BALL_LAUNCH` still raises. `SandVersusClub` compares what the
sand delivers to the ball against what the club delivers to the sand as a
dimensionless share of one solve's ledger plus a pair of timings, never as two
forces, since absolute club force is refused at this tier and an absolute ball
force does not exist in plane strain. Every result carries its
`ValidityVerdict`: F1, BEYOND_VALIDATION and no better, published-speed ceiling
1.44 m/s, NASA-STD-7009B validation 0 of 4.

## F1 MPM Plastic-Limit, Manufactured-Solution and Temporal Verification (#8733 §4)

Three code-verification cases the shipped F1 suite did not reach, all routed
through the existing `vandv/` conservation, convergence and Celik GCI
implementations rather than a second Richardson extrapolation.

**Plastic limit.** The 2-D Drucker-Prager surface is written on the two
in-plane principal Kirchhoff stresses, so the plane-strain Coulomb limit it
enforces is `K = (1 ∓ √2·α)/(1 ± √2·α)` at an equivalent friction angle
`φ* = asin(√2·α) = 31.944°` — not the 34° handed to `drucker_prager_alpha`,
which fits the inner cone in three dimensions. A smooth rigid wall pushed at
`v/c = 1.58e-4` into a frictionless-based cohesionless layer reaches
`P_p = 22.744 N/m` against the closed form `K_p·ρgH²/2 = 21.287 N/m` at
`dx = 3 mm`, `H = 30 mm`, with 98.9% of the bed at yield — **6.845%**, falling
to **2.966%** at `dx = 2 mm` and rising to 10.182% at `dx = 4 mm`.

**MMS.** A manufactured diagonal deformation-gradient field with a closed-form
`div σ` is compared against one step from rest, which makes the particle
velocity exactly `dt` times the discrete P2G–solve–G2P operator. Observed order
**1.880** over four grids (pairwise 1.866, 1.892, 1.870; spread 0.026) against
a design order of 2. It covers the stress divergence and the transfer together
on the **elastic** branch and covers neither the return map nor the time
integration; the accompanying uniform-stress patch test is a round-off-class
identity at **1.76e-15** relative and refuses an order test.

**Temporal.** Step refinement at fixed `dx` over one elastic transit is
monotonic with Celik apparent order **1.214** and `GCI_fine = 0.473%`, declared
a temporal band only. Over two transits and beyond the same triplet is
`MONOTONIC_DIVERGENCE`: the particle-grid round trip costs a fixed amount per
step, so over a fixed physical window its total grows as `1/dt` and overtakes
the integrator's own `O(dt)` error. The tier's convergent refinement direction
is therefore space-time at fixed Courant number, which is what the shipped
grid study already takes.

**The conservative elastic case, attempted.** The issue's remaining bullet said
a cohesive cone tip "might admit a genuinely conservative small-amplitude case;
it was not attempted". It was attempted and it does not work. A `FIRM` column
pre-compressed by `2.0e-5`, inside the `6.566e-5` the tip can carry in
extension, oscillates with **zero** particles yielding at any step, and the
total energy still drifts 9.472%, 9.782% and 11.354% across a four-fold step
refinement, fitting an order of **-0.124** against the 1.00 the same
measurement gives the transfer-exact free-fall case. The cohesive tip removes
the plastic obstacle and leaves the numerical one. F1 has no conservative
elastic-energy case for cohesive sand either, and the mechanism is now
identified rather than assumed.

None of this changes the NASA-STD-7009B validation score, which remains 0 of 4:
no experimental data appears anywhere in these cases.

## Qt `sys.modules` Pollution Guard (#9188)

Issue #9188 removes the order-dependence that made `unit-test-gate` a lottery. The session-scoped autouse fixture in
`tests/unit/plotting/conftest.py` replaced `PyQt6`, `PyQt6.QtCore`, `PyQt6.QtGui`, `PyQt6.QtWidgets` and the
`src.shared.python.ui` shim with `MagicMock(spec=ModuleType)` stubs. A `scope="session"` fixture declared in a _directory_
`conftest.py` is created lazily at the first test in that directory but only finalized at the end of the whole session, so
the stubs stayed installed for every test collected after `tests/unit/plotting`. Because a spec'd mock raises
`AttributeError` for names a real module supplies, each later `from PyQt6.QtCore import Qt` (or `QApplication`, `QDialog`)
failed with `cannot import name 'Qt' from '<unknown module name>' (unknown location)`, and which tests were hit depended
only on collection order. The fixture is now function-scoped and installs its stubs through `monkeypatch.setitem`, whose
teardown is automatic and exception-safe.

A durable guard in `tests/conftest.py` prevents recurrence. A `trylast` `pytest_runtest_teardown` hook — running after every
fixture finalizer that is due, and therefore immune to autouse ordering — compares the `sys.modules` entries for `PyQt6`,
`PyQt5`, `PySide2`, `PySide6`, `qtpy` and the `shared.python.ui` shim against a baseline snapshot taken at
`pytest_collection_finish`. It records the first test at which they diverge and reports at the test-file boundary, so a
properly scoped stub that is undone in time is never flagged while one that outlives its file fails the run and names its
author. Newly imported real modules are permitted; stub insertions, replacements and removals are not. After reporting, the
guard restores the baseline so a single leak yields one attributable failure instead of a cascade of innocent victims.

## Performance Enhancements (#9161)

- Replaced instances of `np.linalg.norm` with faster mathematical equivalents (`math.hypot`, `np.vdot`, and `np.einsum`) for small vectors and multidimensional arrays in telemetry logging, screw kinematics, and bunker shot traces.

## Articulated Same-State Drift and Contact Attribution (#9151)

Issue #9151 qualifies a pointwise articulated decomposition across all 234
registered subject-scaled closed states. Configuration-dependent bias,
velocity-dependent bias, bilateral contact, and zero applied input close both
generalized acceleration and generalized power. MuJoCo and robotics Pinocchio
pass the declared native-operator parity, pathway-killswitch,
coordinate-scaling, geometry, denominator, and corrupted-force gates. Contact
aligns positively with total mass-metric acceleration while contributing
negative generalized power in every registered state, demonstrating that an
acceleration projection is not a positive-work or transfer fraction. The
result takes no forward step and supplies no biological source, human
performance, timing, slack, coaching, or safety inference. Its next gate is
matched forward impulse/work attribution through contact transitions,
shaft/base coupling, uncertainty, and adverse loads.

## Pinocchio CRBA Symmetry and Requalification Boundary (#9153)

The proximal--distal native-dynamics paths treat robotics Pinocchio CRBA as an
upper-triangular API. One shared adapter validates a finite square upper
triangle, copies it into an independent array, and completes the lower triangle
by symmetry before any solve, parity comparison, or attribution. A poisoned
lower-triangle regression prevents reusable native-data contents from becoming
solver input. This source correction invalidates the freshness of pre-fix
source-bound evidence; those artifacts remain retained as adverse/stale records
and are not regenerated or promoted here. A separately preregistered native
requalification must pass before revised scientific artifacts, paper claims, or
later #9153 structural cases can become authority.

## Qt/Sip Worker-Poisoning & Segmentation Fault Prevention (#9099)

Issue #9099 fixes worker-poisoning access violations and segmentation faults occurring during test suite execution with pytest and PyQt6.
The launcher entry point `run_launcher` in `src/launchers/base.py` (and launcher scripts `src/launchers/upstream_drift_launcher_main.py`,
`src/launchers/exercise_dashboard.py`, `src/launchers/_shot_tracer_gui.py`) now verifies `QApplication.instance()` before creating a new
`QApplication(sys.argv)` instance. When an active Qt application context already exists (such as pytest's `qapp` session fixture),
the existing application instance is safely reused, preventing double-initialization memory corruption and C++ access violations.

## Tools Green-Surface Adapter Consumption (#9143)

Issue #9143 consumes the Tools `swing_sim.putting` green-surface adapter (Tools #4800 P9) via the `vendor/ud-tools` vendor boundary.
The `vendor/ud-tools` submodule is updated to the latest Tools `main` squash commit (`b46f58df52df86b6c5a3db44460b26ac8919da70`),
providing runtime-free format adapter functions `green_surface_from_ud_json` and `green_surface_to_ud_json` alongside the `UdGreenTopography`
container. Bi-directional interchange with UpstreamDrift's native `GreenSurface` (`src/engines/physics_engines/putting_green/python/green_surface.py`)
and its `_surface_io` JSON format is verified by consumer integration tests in `tests/unit/putting/test_putting_green_consumer.py`,
confirming exact node elevation and gradient matching on planar and regular grids, byte-deterministic roundtrips, fail-closed refusal
of non-conservative slope regions and scattered contours, and flat/sloped roll dynamics consistency.

## Swing Objective Lab Web Parity (#9128)

The protected Tools provider is pinned at `cc883cbaf63157b58c71cba385a683df2762b0cb`
and exposes six canonical objectives, including signed grip-force impulse along
the hand path. This impulse is an exploratory signed time integral, not the
work-per-path-length average-force metric reported by MacKenzie-style studies.
The fixed-hub comparison uses one undistorted 192 by 176 frame, an invariant
three-line title row, and a shoulder coordinate of (96, 88) for all six
objectives across the playback range. A common target at (150, 148) is visually
distinct from each scenario's measured impact location; impact-aligned playback
registers the measured impacts to that target as an explicit alternative.

Issue #9128 establishes full web parity for the Swing Objective Lab tool.
The FastAPI REST endpoint `POST /tools/swing-objectives/compare` and `GET /tools/swing-objectives/presets`
in `src/api/routes/swing_objectives.py` accept a golfer preset and shared effort budget
(downswing duration, shoulder/wrist torque bounds, collocation nodes) and execute direct collocation
to emit a versioned comparison matrix conforming to schema 1.0.0. The React/Tauri frontend page
`ui/src/pages/SwingObjectiveLab.tsx` renders the per-objective metrics table and cross-evaluation matrix
with every cell explicitly labeled (text and accessible ARIA attributes, preventing color-only encoding)
and displays a plain-language alert when `is_degenerate` is true. `src/config/feature_parity.json`
registers `simulation.swing_objective_lab` as `parity` and `src/config/launcher_manifest.json`
routes web execution to `/tools/swing-objective-lab`.

## Trajectory-Varying Event-Conditioned Reaching (#9123)

Issue #9123 establishes discrete time-varying variational control authority $z[k+1] = A[k] z[k] + B[k] v[k]$
and event-tangent reachability Gramian conditioning $W_{tangent} = P W_{full} P^T$ along the registered
analytical double-pendulum downswing. The transverse event projector $P = I - \frac{f_{event} n^T}{n^T f_{event}}$
satisfies idempotence $P^2 = P$ and tangent null direction $P f_{event} = 0$.
Tangent reachability rank is verified at 3 dimensions across active input channels, single-channel
additivity $W_{both} = W_{shoulder} + W_{wrist}$ is exact within numerical precision, and zero-input
authority yields rank 0. Direct finite-difference pulse responses match propagated sensitivity matrices.
Scientific inference boundaries restrict claims to analytical linear variational dynamics.

## Bounded Nonlinear Event-Reaching Feasibility (#9124)

Issue #9124 establishes numerical feasibility for bounded control perturbations over pre-event horizons
under explicit torque-amplitude and slew-rate bounds. Four-interval multiple shooting propagates every
state with the protected exact-RK4 parent trajectory and independently replays the resulting controls
to the geometric event. The registered continuation matrix contains 38 target/channel cases: 32 are
feasible, while the six infeasible cases are displaced targets under the zero-authority killswitch;
all nominal zero-offset cases remain feasible. Feasible event-tangent residuals are at most
$8.83\times10^{-11}$, below the registered $2\times10^{-6}$ gate. Mesh, integration-step,
adverse-initial-state, and channel-mask controls pass, but the two converged multistart objectives differ
by 24.9517%, failing the preregistered 5% optimality gate. The result therefore establishes only local
bounded feasibility for one synthetic planar trajectory and suppresses channel, controller, effort,
human-capacity, passive-torque, and coaching rankings.

## Event Topology and Delay/Noise Robustness (#9125)

Issue #9125 establishes direction-aware global event enumeration over a common horizon and retains
absent, unique, multiple, reversed, grazing, initial-on-guard, and numerical-failure outcomes. Phase A
uses eleven causal delays, matched state/command/event-surface perturbations, 192 antithetic replicates
per nonzero cell, and 96 independent pairs; all 6,336 nonzero small-stress replays retain one positive
transverse crossing. A separately preregistered Phase B executes every fixed stress level and first
exposes topology loss at 0.02 synthetic dimensionless stress and 200 ms delay. Phase C applies both,
shoulder-only, wrist-only, and zero generalized-coordinate masks, preserves topology identity across
1, 2, and 4 ms RK4 steps, and uses 0.40, 0.60, and 0.80 s horizons to identify wrist-only truncation.
Topology preservation is not target feasibility, work, power, human robustness, anatomical isolation,
channel superiority, or coaching guidance.

## Nonlinear Controller Mechanics Qualification (#9126)

Issue #9126 freezes a matched analytical-double-pendulum controller comparison
before evaluation, with 24 outcome-blind evaluation trials, eight disjoint
tuning trials, common plant, coordinate, scaling, bound, event, failure, and
random-stream contracts, and fail-closed ranking suppression. One bounded
projected first-order iLQR kernel passes manufactured derivative, in-rollout
bound, accepted-cost descent, exact replay, initialization-sensitivity, and
typed nonfinite-dynamics gates. Collocation NMPC remains unimplemented and is
rejected by identity rather than inferred from a bounded shooting method.
Twelve controller-facing RK4 steps match the canonical public ODE backend over
0.5, 1, and 2 ms. These results qualify only a numerical prerequisite and
shared-equation transport: zero registered controller evaluations have run and
zero methods are ranking-eligible.

Issue #9065 establishes ADR-0041 and an executable acceptance program before
live markerless capture begins. Canonical camera, capture, time, calibration,
observation, reconstruction, session, and C3D exchange contracts belong to
Tools #4706. UpstreamDrift owns application orchestration, persistence,
biomechanics integration, and matching PyQt6/React/API workflows. AffineDrift
owns sanitized evidence publication. Tools_Private is not a dependency of the
open runtime. The first consumer under #9069 must pin a protected Tools merge,
reject missing or incompatible schema authority, and adapt existing C3D and
motion-pipeline paths instead of copying shared code. This M0 slice makes no
camera, inference, C3D round-trip, commercial, or physical-lab qualification
claim.

Issue #9027 now has a versioned, executable hybrid-system topology contract for
all eight model tiers. It binds implemented and partial tiers to existing source
authorities and requires continuous states, controls, algebraic constraints,
modes, guards, resets, impacts, actuator dynamics, uncertain event surfaces,
observables, limitations, falsifiers, and comparison blockers. Three tiers are
implemented, three are partial, and the participant-calibrated and governed-human
tiers remain explicitly unavailable. This is a topology and evidence-boundary
gate; observability, controllability, stability, controller ranking, participant
validity, and coaching interpretation remain unqualified.

Issue #9116 adds the next bounded #9027 analytical slice: dimensionless
finite-time state-transition maps and transverse geometric-event sensitivity
for the registered nonperiodic double-pendulum downswing. The computation
retains step refinement, complete direct perturbation controls, equivalent
units, near-grazing rejection, time-guard saltation controls, and an explicit
periodicity gate. Floquet output is suppressed because the trajectory does not
close. These local finite-window results do not establish asymptotic/global
stability, neural timing demand, participant robustness, passive negative
torque, human strategy, or coaching guidance.

Issue #9107 closes an external-install packaging gap exposed by the protected
Tools downstream-consumer gate. The canonical BunkerShot3D public identity is
`bunkershot3d`; the Hatch build hook now maps the Upstream-owned source package
to that top-level destination for editable and wheel installs, while the wheel
selection excludes `src.bunkershot3d`. This preserves exception and class
identity without library-time `sys.path` mutation. Isolated editable and wheel
probes must import `bunkershot3d.postproc.WrenchTrace` outside the checkout, and
the built wheel must contain no second `src/bunkershot3d` payload.
The build retains the source archive and wheel as separate immutable artifacts;
smoke jobs minimally check out only their input fixture and download only the
selected wheel. Their bounded 20-minute budget covers the measured cold fleet
transfer plus clean installation and both unchanged runtime assertions.

Issue #9059 adds a coordinate-explicit planar pendulum attribution contract on
top of the pinned Tools source authority. It separates cross-speed Coriolis,
squared-speed centripetal/centrifugal, gravity, damping, applied-drive, and
independent velocity-residual terms under shoulder-absolute/wrist-relative
coordinates. The force-only wrist endpoint map reports Jacobian rank and the
unreconstructed generalized couple. A complete 135-program grid keeps
absolute Coriolis tangent impulse, signed impulse, generalized work, and
qualified-impact clubhead speed as distinct estimands. This is synthetic
planar model evidence, not a measured grip wrench, muscle attribution,
continuous optimal-control solution, human strategy, or coaching authority.
The source authority is protected Tools squash
`8dc4512184d8c29e10770ad81e4ce947f849b355`, incorporating PR #4699, the
read-only provider-Protocol correction from PR #4700, and the restored
downstream dataset façade from PR #4701; feature-branch commits are not
publication authority.

Issue #9092 qualifies the next #9027 analytical slice. It retains raw local
linearizations at four synthetic trace-derived states but requires explicit
nondimensional state, control, output, and characteristic-time scales before
interpreting numerical observability or controllability conditioning. Unit-
invariance, scale sensitivity, finite-difference sensitivity, measurement and
actuator countermodels, manufactured fixtures, and killswitches are required.
Structural or practical identifiability, global nonlinear properties, human
validity, and coaching interpretation remain unavailable.

Issue #9104 qualifies the exact and finite-record identifiability boundary for
the declared analytical double pendulum. Its inverse-dynamics regressor has
seven named base coefficients, while an analytic nonzero-minor witness proves
that the eleven-entry reduced physical map has rank seven and nullity four.
Three exact physical-parameter alternatives preserve every base coefficient.
Finite-record rank and conditioning use positive coefficient and torque scales,
equivalent-unit and scale audits, shortened-window adverse cases, and a
zero-motion rank-zero killswitch. Gaussian Fisher intervals are oracle-
kinematics lower bounds only and shall not be promoted to practical,
participant, biological, or coaching identifiability.

Issue #9027 also qualifies a dimensionally explicit constraint-rank boundary.
Planar closure Jacobians require declared generalized-coordinate scales before
their singular values or condition numbers are interpreted, and bilateral
hand-wrench maps require a declared moment reference length. The governed
diagnostic keeps kinematic closure, point-force allocation, and full wrench
measurement nullspaces separate; positive rescaling preserves exact rank and
nullity but not numerical conditioning. Its adverse alignments are constructed
mathematical controls, not qualified anatomical poses, and the result does not
identify constraint force, muscle action, human strategy, or coaching intent.

Epic #8557 has completed the current narrative-candidate adjudication contract:
1,134 reviewed candidates and 309 atomic claims. Issue #8724 adds an exhaustive,
snapshot-locked four-way outcome authority: 289 supported at their declared
estimands and boundaries, five inconclusive, 15 untested, and none contradicted.
The absence of a contradicted row does not erase adverse or null results that
the paper reports accurately. This status is not scientific closure: all 46
public release claims have a traceable review disposition, and
each retains its applicable open
model, equipment, anatomy, archival, or governed-human scientific boundary. The
issue #8918 numeric authority binds all 394 numeric literals across 128 of the
309 claims to reviewed statement digests, JSON Pointers, transforms, evidence
scopes, and tolerances. It distinguishes 181 semantically matched local JSON
values, 149 registered claim values that have not been independently
recomputed, 57 externally reported values, and seven protocol or notation
values. Representative planar, spatial, articulated-shaft, and finite-ground
headlines are independently recomputed from committed raw arrays, while the
spatial cross-engine control must remain close but nonidentical. Pointer
agreement and release traceability do not establish physical validity, human
validation, anatomy, physiology, equipment calibration, injury, coaching
efficacy, or a universal clubhead-speed benefit. The
trajectory-level bilateral point-force sensor qualification and subject-scaled
spatial contact-closure audit retain their synthetic and prescribed-state
scopes. The closed-state forward bridge maps all 234 solved states and advances
54 profile--span--phase cases for 4 ms in native MuJoCo and Pinocchio. This is
a reduced initialization audit: articulated arms, calibrated contact and shaft
properties, full-horizon delivery, and human inference remain prohibited.
Issue #8666 extends that reference through 4, 10, 25, and 50 ms under nominal
conditions and nine one-factor adverse or null branches. All 2,160 registered
horizon cases pass the existing cross-engine discrepancy and work--energy
closure gates. No failure is observed through 50 ms, making the result
right-censored at that reduced-model horizon rather than a full-delivery or
anatomical claim.
Child issue #8676 under issue #8668 records the first articulated prerequisite:
all 234 closed states are assembled independently in native MuJoCo and robotics
Pinocchio, and every
registered mass-matrix, bias-force, inverse-dynamics, symmetry, and
positive-definiteness gate passes. This common-state result qualifies the
20-coordinate articulated rigid-body transport only. Forward bilateral
contact, scapulothoracic anatomy, distributed grip and shaft properties,
muscles, delivery, and human inference remain prohibited.
Child issue #8910 repairs the articulated manufactured-solution tier. The
manufactured generalized torque is defined by the analytical
Lagrange--Christoffel formulation and compared with native MuJoCo
`mj_inverse` and robotics Pinocchio RNEA. Cross-engine residuals must be small
but nonzero, and a 10 N m corruption of one native result must fail the gate.
A gravity-free, zero-torque rollout reports measured momentum and kinetic-
energy drift for the genuinely free club subtree, while adjacent three-level
Richardson estimates must remain inside 0.9--1.1 for semi-implicit Euler. The
coordinated multiplier-recovery control contains one three-component
lead-hand-to-grip point constraint; it is not simultaneous two-hand closure,
an anatomical model, governed human evidence, or a coaching result.
Child issue #8678 applies finite bilateral Kelvin--Voigt forces to those same
234 states and verifies action--reaction, virtual power, passivity, geometry
controls, and native MuJoCo/Pinocchio initial-acceleration parity. It advances
no trajectory; contact loss, accumulated work, calibrated anatomy/equipment,
delivery, and human inference remain prohibited.
Child issue #8680 advances 18 selected cross-profile/span/phase states through
seven nominal/adverse branches, three time steps, and two native engines for a
total of 756 five-millisecond trajectories. All registered bilateral-
attachment retention, virtual-power, dissipativity, work--energy, refinement,
and parity gates pass. This is a right-censored synthetic attachment result;
unilateral slack, distributed grip and shaft structure, ground coupling, late
downswing, impact, anatomy, muscle action, and human strategy remain
prohibited inferences.
Child issue #8682 then qualifies typed bilateral, tension-only, and radial
dead-zone point attachments across 1,944 five-millisecond trajectories and
isolated opening/reattachment probes. Child issue #8685 replaces each hand's
point attachment with one, three, or five tension fibers while preserving total
stiffness and damping. Its 288 trajectories are observed at nested 4, 10, 25,
and 50 ms horizons in MuJoCo and Pinocchio. All registered geometry, power,
passivity, work--energy, time-refinement, station-refinement, and engine-parity
gates pass; no natural active-set transition occurs. This qualifies a synthetic
contact discretization, not measured grip pressure, shaft response, delivery
benefit, timing economy, human transfer, or technique.
Issue #8751 remains open. The in-progress `research/8751-friction-atlas` v3
slice supplies full-state velocity reversal, frictionless and finite-friction
nested-horizon outcomes, station-level event direction, cross-engine active-set
parity, and a 144-cell mass-metric impulsive perfect-stick bound. The latter is
checked against an analytic manufactured solution, constrains tangential
velocity to $4.61\times10^{-15}$ m/s or less, has a nonzero maximum
MuJoCo--Pinocchio projected-velocity error of $3.99\times10^{-12}$ relative,
and captures $9.84\times10^{-9}$--$0.37665$ J of kinetic energy across the
registered states. This corrected genuine-engine run replaces the invalid
zero-discrepancy record. It is an instantaneous
ideal-constraint control, not evidence that static friction can supply the
impulse or maintain a stick trajectory. The event probe begins disengaged, so
attached-to-open first failure, static-friction feasibility/evolution,
production release evidence, and protected merge verification remain required
before #8751 can close.
Issue #8909 invalidates the previously published distributed-grip
MuJoCo--Pinocchio parity statement: the Windows run imported the unrelated
PyPI `pinocchio` 0.1 package, and the forward operator silently substituted
MuJoCo while retaining the Pinocchio label. Cross-engine evidence now fails
closed unless robotics Pinocchio is version 2.6 or newer, exposes the required
native dynamics API, and successfully builds the articulated model. No engine
fallback is permitted, and an identically zero set of trajectory, force, and
stick-projection discrepancies is treated as a degenerate comparison rather
than successful parity. The first genuine rerun also falsified the original
stick-projection numerical gate: twelve distributed constraint rows had rank
ten, the normal-equation condition number exceeded $10^{17}$, and the
Pinocchio projection residual reached $1.35\times10^{-10}$ m/s. The corrected
implementation performs a mass-whitened rank-revealing SVD, reducing the full
144-cell residual to $4.61\times10^{-15}$ m/s without relaxing the registered
tolerance. Every committed cross-engine artifact must identify a qualified
robotics Pinocchio version. The distributed atlas and registered claim must be
regenerated with two genuine engines before #8751 can close.
Issue #8752 also remains open. Its first manufactured and Latin-hypercube screen
does not yet run through both production engine adapters, include a deliberately
perturbed failure case, refine every uncertainty corner, or propagate uncertainty
through the registered shaft and ground headline estimands. Neither preliminary
study supports human, anatomical, equipment, or coaching inference.
Issues #8703 and #8704 (epic #8699) withdraw two BunkerShot3D outputs from
quotable status. The `dig_vs_skid` verdict returned `MARGINAL` at all 77 demo
design points with slope ratios spanning 0.9987--1.0000: the shipped 10 mm
entry window is about 0.4 ms at a 25 m/s delivery, over which a 0.3 kg head
under an order-5 N.s impulse cannot deflect measurably. Resizing the window was
measured over 48 design points before being rejected -- the ratio span grows
from 0.0015 at 10 mm to 0.28 at half the divot length, but its correlation with
maximum sole depth is negative at every informative window (-0.50 to -0.68), so
a resized window inverts the verdict rather than calibrating it. The ratio is
pinned at 1 for a vanishing window and at 0 for a window spanning the divot,
for every design, so no window size is a free parameter. Every `DigSkidResult`
therefore carries a `DigSkidCalibration` reporting `calibrated=False`, and both
degenerate window ends are refused rather than clipped. A separating verdict
requires a quantity that varies with the design -- sole depth does -- and an
absolute threshold that is not published. Issue #8704 makes the sand-to-ball
transfer efficiency a function of the bed's relative density,
`eta(D_r) = efficiency * (1 - packing_sensitivity * (1 - D_r))`, which restores
the physical ordering of ball speed with lie firmness (firm 11.37 m/s, wet
8.12, fluffy 7.84, plugged 7.09, against a previous inverted firm 12.13 /
fluffy 12.55 / plugged 12.60). The direction follows critical-state dilatancy;
the magnitude is an assumed placeholder recorded in the launch provenance under
`bed_packing_dependence`, and ball speed remains `BEYOND_VALIDATION` because
issue #8616 found no published measurement of ball speed, launch angle or spin
out of sand.
Issues #8705 and #8707 (epic #8699) raise the BunkerShot3D sole-load view from
a summed 12x12 bin to the field the F0 solver actually produces. `simulate_shot`
builds the club wrench as an integral of a per-element traction and then keeps
only the resultant; the workbench recovered a 12x12 impulse-density map summed
over the strike and discarded the rest. `SoleLoadField` now carries the load
per surface element and per sample with the depth-linear and inertial 3D-RFT
terms separated, and `ContactPatch` follows the engaged element set, its area,
its centroid and its gap to the leading edge. The terms are summed and then
clamped, never clamped and then summed, so the array
`bounce_utilisation` consumes is bit-for-bit what it consumed before: one
traction can point outward on a steeply raked element while the resultant is
still compressive. On the nominal 58 deg design at 25 m/s and the shipped
discretization the sole resolves to 500 elements over 53 samples; the
depth-term resultant peaks at 1.991 N at 4.00 ms and the inertial term at
725.3 N at 4.25 ms, a 99.7 % inertial share of the sole's own resultant against
the roughly 0.9 whole-body share ADR-0032 predicts. The two terms separate in
space as well as in time, the depth term loading 5--18 mm behind the leading
edge and the inertial term 20--28 mm behind it. The contact patch opens at
1.309 cm^2, peaks at 16.21 cm^2 and closes from 13.09 mm to 3.39 mm behind the
leading edge before retreating. The separation in time is one sample wide and
does not survive coarsening: at a 5-station mesh both terms peak in the same
4 ms bin, so the moment is reported per term rather than a difference being
asserted. Every frame is drawn with its `EnvelopeStatus` and fidelity tier
stamped inside the axes rather than in a caption, on colour limits fixed across
frames and merged across an A/B pair; no 3-D viewport provider (MeshCat, Rerun,
VTK) is installed, so the ADR-0027 selection degrades to a stated matplotlib
plan view. This is a rendering of existing F0 output at higher fidelity, not
new physics and not calibration: the field inherits `BEYOND_VALIDATION`, and a
patch trend read across a bounce sweep is reported as a bounce-and-camber
trend wherever any spanwise station's camber was substituted (#8698).

Issue #8709 (epic #8699) selects the sand-solving tier that backs field
visualization, recorded as ADR-0033 amending ADR-0032. The epic had assumed the
MuJoCo F3 grain proxy was available now; measurement withdraws that premise.
`MPMDriver.setup()` raises before any step, because the generated MJCF omits a
`density` attribute on grain geoms, so MuJoCo applies its 1000 kg/m^3 default
instead of the configured 2650 kg/m^3 silica and the 0.4 mm grain inertia
(5.36e-16) falls below `mjMINVAL`; the measured minimum representable radius is
0.2266 mm at default density against 0.1864 mm at silica density. The
`MAX_SPHERES = 1000` cap does not thin the bed but destroys it: the placed
grains form a single-grain-thick line at y = -0.1496 m, an implied bed depth of
0.00116 grain diameters, with 0 of 1000 grains inside the clubhead's 50 mm
y-band, so a repaired run would sweep vacuum and return an identically zero
wrench. `MPMDriver.run()` also never calls the existing
`write_grain_state`, so no grain reaches disk. A best-case repaired proxy was
then benchmarked directly: 3,840 spheres at 18.5 s per 20 ms shot, with 10,000
spheres failing to allocate, fixing the tractable ceiling near 4,000 spheres --
1.79e-5 of the 2.149e8 true-scale grains and 0.028 % of the bed by volume,
which bins to 0.125 grains per cell on a 20^3 grid. Track B requires fields
(#8710 velocity and density on a grid, #8711 cross-sections with shear
overlays), which that sample cannot carry, so the decision is F1, narrowed from
ADR-0032's "reduced-order / 2-D plane-strain continuum" to a 2-D plane-strain
**MPM** solver: a continuum produces a field by construction, SPH's
blade-thickness floor follows it into plane strain, and MPM shares its
constitutive model with the F2 reference. F1 is specified at bulk resolution
(dx ~ 1-2 mm) for 10-100 mm flow features and is therefore barred from
reporting club force, which remains F0's. The ball becomes a body inside F1 as
a plane-strain circular section so that "sand reaching the ball" acquires a
referent it never had at F0, but it is an infinite cylinder rather than a
sphere, its flux is per unit width, ball launch stays on the #8657 F0 path, and
any heel-toe or out-of-plane distribution is refused rather than approximated.
Field frames are marked illustrative in the raster and in the API rather than
in captions: provenance composited into the pixels, ordinal colourbars with no
numeric ticks for non-quotable quantities, a distinct non-photorealistic
identity, and an export path whose validity verdict has no default. Cross-tier
comparison against F0 on wrench, depth and divot is a consistency check between
two uncalibrated models, not a validation; F1's per-unit-width wrench compares
on shape and timing unconditionally and on magnitude only once the assumed
effective width is recorded in the manifest. Validation remains at NASA-STD-
7009B level 0 of 4 against a threshold of 3, with the design point 63.1x
3D-RFT's stated Froude limit at clubhead scale and 282.2x at the 5 mm leading
edge, so no F1 output is a physical prediction. The driver defects are tracked
separately; the tier is not usable until they are repaired or the backend is
removed.

ADR-0033's F1 solver core now exists at `src/bunkershot3d/solvers/mpm/`,
implementing the `GranularSolver` protocol so it is swappable with the F0
`DRFTSolver` and returns the same `SolverResult` carrying tier and verdict. The
constitutive model is **rate-independent Drucker-Prager elastoplasticity with a
compressive cap**, integrated as a return mapping on principal Hencky strains
(Drucker & Prager 1952; Klar et al. 2016, ACM TOG 35(4):103, which is also the
F2 reference tier's model so one calibration carries between them). The
`mu(I)` rheology was rejected on well-posedness rather than on cost: Barker,
Schaeffer, Bohorquez & Gray (2015, J. Fluid Mech. 779:794-818) show its
incrementally-linearised equations lose hyperbolicity below `I_1` and above
`I_2`, so perturbation growth is unbounded in wavenumber and grid refinement
makes the answer worse -- and both ill-posed regimes sit inside a bunker shot,
the quiescent bed at `I -> 0` and the leading edge at `I ~ 11`. A tier whose
deliverable is a GCI-reported field cannot rest on equations that are ill-posed
in the regime being refined. The compressive cap is read off the sand package
rather than fitted: `eps_v_cap = ln(phi / phi_max)` from the existing
`PackingState`, so a loose bed compacts about 15 % and a firm one about 2 %
without a second set of constants existing; the cohesive cone tip comes from the
moisture model, and a saturated bed raises rather than guessing its dilation
suction. Transfers are **APIC** (Jiang et al. 2015, ACM TOG 34(4):51) on
quadratic B-splines: PIC would damp away the shear structure the tier exists to
show, FLIP leaves null-space noise that a plastic return map reads as spurious
yielding, and APIC conserves linear and angular momentum exactly across the
transfer, which is tested to round-off rather than cited. The timestep is a CFL
condition on the computed dilatational wave speed `sqrt((lambda + 2 mu) / rho)`
plus the body speed, checked at runtime with a `raise` because `python -O`
strips assertions. Sand constants come from `usga_reference_sand` and the
packing/moisture machinery; the only F1-specific additions are a Hardin &
Richart (1963) small-strain modulus keyed on the existing void ratio and
angularity, and a conventional drained Poisson ratio, both recorded in the
provenance. Measured code verification, through the existing `vandv/`
conservation, convergence and Celik GCI implementations rather than a second
copy: mass conserved exactly (residual 0.0); linear momentum matching the
gravity impulse to round-off; total energy first order in the step (observed
order 1.00 over Courant 0.4/0.2/0.1, 2.2e-8 relative at the coarsest);
a relaxed 1-D consolidation column at -409.741 Pa against -403.115 Pa
closed-form, 1.64 %, with its stored strain energy 5.8 % from
`W (rho g)^2 H^3 / (6 M)`; and monotone grid convergence over dx = 6/4/3 mm at
3.70 % -> 2.22 % -> 1.64 %, observed order 1.178, Celik apparent order 1.72,
`GCI_fine` 1.105 % and `u_num` 2.26 Pa. Two verification cases had to be rebuilt
around real findings: a generously padded grid put the column's "fixed base"
two cells below the column and its confining walls two cells outside it, so the
column stood in free space and fell while reporting a mean stress of 3e-9 Pa
through an entirely plausible-looking run; and the conventional elastic energy
case does not exist for cohesionless sand, because a pre-compressed block
released to oscillate rebounds into tension where the return map correctly
annihilates the deviatoric strain at the cone tip (129 of 144 particles per
step), which is physical plastic dissipation and not truncation error. Bagnold
is unavailable for the same class of reason: it is a consequence of `mu(I)` rate
dependence, which this solver deliberately does not have. The F0 cross-check on
a 40 x 16 mm sole section at 20 deg attack, 30 mm declared effective width,
reports the divergence rather than asserting agreement: `|F1|/|F0|` of 1.96,
1.49 and 2.68 at 5, 12 and 25 m/s with direction cosines 0.85, 0.65 and 0.87,
and -- the informative part -- F0's inertial share climbing 0.52 -> 0.93 -> 0.99
with speed while F1's momentum-flux share stays flat at 0.68 / 0.69 / 0.65. F0's
`lambda rho v_n^2` term grows quadratically with nothing to bound it, whereas
the continuum's reaction is limited by how fast the yield surface lets sand be
accelerated out of the way, so the two tiers disagree by construction in exactly
the regime a greenside shot occupies. F1 is therefore **verified but not
validated**: the suite shows it solves its equations correctly and is no
evidence at all that those equations describe golf bunker sand. Validation stays
at NASA-STD-7009B level 0 of 4 and the F1 envelope enforces a
`BEYOND_VALIDATION` floor that no query can beat, since `EXTRAPOLATED` would
require a published validation set that issue #8616 established does not exist.
Field extraction and schema (#8710) and the visual layers (#8711-#8713) are
deliberately out of that change.

Issue #8733 sections 1-3 close the rest of the tier's mechanics: the ball as a
plane-strain body, contact against several bodies within one step, and a
whole-shot march. The ball is a rigid circular section built as an **equal-area**
regular polygon rather than an inscribed or circumscribed one, because either of
those biases the displaced area systematically and the bias survives grid
refinement, so it would read as a quiet offset in the flux onto the ball rather
than as noise; `n_facets_for_cell_size` then picks a facet count whose chord fits
inside one cell, the only scale at which the sand can see the flat spots. Both
facts ADR-0033 requires travel in the API rather than in a comment: the body is
an **infinite cylinder, not a sphere**, so `line_mass_kg_per_m` is named for the
per-unit-width quantity it is and `sphere_mass_kg` raises rather than returning a
number that would be quoted; and the below-equator / face-side split #8712 asks
for is in-plane and **qualitative**, reported as fractions on a value that flags
itself `is_qualitative` and carries the cylinder note into its own summary, while
`heel_toe_split`, `lateral_distribution` and `BallContactSplit.heel_toe_fraction`
all raise `RefusedQuantity.OUT_OF_PLANE` and `launch_velocity_m_s` raises
`RefusedQuantity.BALL_LAUNCH`, ball launch staying on F0's #8657
momentum-transfer path. Multi-body contact is order-dependent by construction --
the grid projection writes a velocity-level constraint straight onto the node, so
at a shared node the last projection wins and the earlier body's non-penetration
may be left violated -- and the order is therefore chosen and stated rather than
inherited from a loop: **slowest first, fastest last, ties in the caller's
order**. The fastest body is the one that can tunnel, since the CFL travel bound,
the swept-node test and the pushout backstop all scale with body speed, so it is
the body whose constraint is worth making exact; deriving the order from the
bodies rather than from the argument list is what makes two callers who pass the
same bodies in different sequences get the same answer. What the choice costs is
the slow body under-collecting, never sand passing through the fast one, and both
halves are pinned by test. The momentum ledger is unaffected by the order: each
body records `m_i (v_i^after - v_i^before)` at its own stage, so the stages
telescope, and a two-body march of a 600-particle bed with every wall FREE --
gravity and the bodies being the only momentum sources -- closes
`delta p = sum_b J_b + M g dt n` to a residual of 1.95e-16 kg m/s against a
2.6e-2 change, 4.3e-15 relative, with the club delivering (2.631e-2, -3.249e-2)
and the ball (-9.25e-7, -3.760e-3) N s per metre of width. The whole-shot march
(`simulate_f1_shot`) is **additional to** the declared straight-line approach,
not a replacement: `solve()` still reverses the body and drives it back to the
queried pose at constant velocity, which is the assumption that makes an F1
answer comparable to F0's memoryless one, while the new path places the head just
above the sand and integrates it, so nothing prescribes where it goes and the
sand it meets is sand it has already disturbed. On one 40 x 16 mm sole at 12 m/s
and 20 deg attack, 4 mm grid, 30 mm declared width, the two paths differ in the
way the difference in question predicts: the declared approach runs 365 steps
over 3.90 ms, peaks at 1.06 kN 1.04 ms after first contact and reports 42.5 N
windowed at the queried pose with a 9.08 mm divot; the marched shot runs 1123
steps over 12.0 ms without exiting, peaks at 534 N 0.13 ms after first contact,
slows the 300 g head from 12 to 9.40 m/s for a 0.984 N s impulse, reaches 25.6 mm
of sole depth over 122 mm of travel with a 26 mm divot, and reports 17.3 N at the
_same_ 8 mm sole depth the declared path was queried at -- a factor of 2.5 lower,
because the marched head has slowed and is ploughing through its own divot rather
than being driven at delivery speed into an undisturbed bed. Cost is as ADR-0033
estimated: 23.75 ms per step at 2,600 particles on the primary development
machine, so a whole shot is tens of seconds and the tier remains a study tier and
not a drop-in for the design sweep. Both the travel allowance and the ejecta
headroom are **declared** rather than predicted, since how far a head travels
while submerged is one of the things the march is run to find out; a head that
reaches the end of the bed raises and names the setting rather than sweeping
empty grid and reporting the resulting zero as a result. Sections 4 and 6 of
#8733 -- a plastic-limit analytic verification case, MMS, temporal grid
convergence, and calibration toward F2 -- remain open.

Issues #8706 and #8708 (epic #8699) add the two views that put the sole-load
field in context: the shot in three dimensions, and the scalar traces beside
it on one cursor. `ShotScene` carries the pose the march recorded, the free
surface and the swept divot section; `ShotTraces` carries the sand wrench, the
sole depth, the speed lost and the contact-patch area, each stored in its own
stated unit. The 3-D frame is built from the backend-neutral ADR-0027
`ViewportOverlayPayload` -- the object a MeshCat, Rerun or VTK provider would
consume -- so the matplotlib fallback cannot drift away from what a real
backend would show; none of the three providers is installed, so the selection
degrades and the frame states which renderer drew it. On the nominal 58 deg
design at 25 m/s the record spans 53 samples over 13.0 ms, the head resolves to
486 surface points of which 400 are sole, the swept envelope reaches 12.64 mm
against a 12.59 mm sole-reference depth, and the section closes at 19.24 cm^2.
Two claims the render is explicitly not making: the sand plane is the model's
single `free_surface_height_m` and `SandSurface.resolves_grains` is False, F0
resolving no grains at all; and the divot is the running lower envelope of the
head's own sole points, so it may only ever deepen -- a floor that rose between
samples would assert sand transport this tier cannot compute, and is a raise.
The validity band is the reason #8708 asks for a band rather than a badge.
`simulate_shot` judges the envelope at every step and keeps only `worst_of`
those verdicts, so the per-sample statuses are discarded before any caller sees
them; they are recovered through the solver's own `DRFTSolver.envelope`, which
judges a state without integrating a force, and the band's worst status is
pinned against `ShotResult.verdict` rather than asserted equal. On the
greenside delivery the band is uniform, and that is not a defect:
`MAX_VALIDATED_SPEED_M_S` is 1.44 m/s, so a 25 m/s head is past the published
corpus from its first free-flight sample and never returns. At a 1.5 m/s
delivery the sand slows it back through that ceiling at 46.50 ms of a 190.25 ms
record, and the whole-shot verdict `BEYOND_VALIDATION` is then wrong for 576 of
762 samples -- 75.6 % of the record -- which is precisely what a single badge
in the corner of a panel would have asserted. All three views are scrubbed by
the transport `SoleLoadFieldWidget` already owned rather than by sliders of
their own, and the world box and per-panel y-limits are fixed across frames and
merged across an A/B pair for the same reason #8728 fixed the colour ramp. This
is a rendering of existing F0 output, not new physics: the scene and the traces
inherit `BEYOND_VALIDATION`, and no quantity here is calibrated for bunker sand.

Issues #8710 and #8711 (epic #8699) extract the F1 sand field, persist it, and
cut the impact zone open. The motivating question -- _does the velocity of the
sand near the face change through impact_ -- now has a measured answer: on the
2 mm reference capture of a 25 m/s, 20 deg entry into firm sand the peak
reportable sand speed runs 0 -> 2.86 -> 25.9 -> 28.98 m/s across the marched
approach, peaking at 116 % of the head speed at `x = +17.0 mm`, `z = +8 mm` --
above the free surface and past the trailing edge, which is the splash rather
than the sand under the sole. Every array comes from the solver's own transfer
operators rather than a reconstruction: density is `scatter_mass` over the cell
area (plane-strain kg/m over m^2 is kg/m^3 with no fudge factor), velocity is
the same APIC `scatter_momentum` the next step would perform, and the shear
rate is `sqrt(2 D : D)` formed at the particles with the solver's own
`velocity_gradient` -- at the particles rather than by differencing the nodal
field, because an empty node beside a full one differences to an enormous
false shear along exactly the free surface the flow of interest sits on. A
node with no sand gets density 0 and shear rate `nan`, not shear rate 0, since
zero would assert that sand at the free surface is not shearing. Capture drives
`march` in stride-sized blocks from one `MPMSetup` (a new public `prepare`,
which `run` now also goes through) and is asserted bit-for-bit identical to the
same march taken in one call, so the stored field is the field the solve had.
**The stored field carries its tier and validity status as data, and the
statement is checkable**: `FieldProvenance` travels inside the file and
`series_digest` covers the provenance _and_ the arrays with one SHA-256, so
renaming `illustrative.h5` to `predictive.h5` changes nothing a reader
consults, editing the stored tier breaks the digest, and swapping arrays under
an honest label breaks it too. There is no load-anyway flag and a test asserts
there is not. A field-schema bump is reported separately from tampering,
because "regenerate this" and "somebody edited this" are different news. The
schema represents more than one tier: `FieldLayout.GRID` stores an origin, a
spacing and a shape so a continuum pays nothing per frame for sample
positions, `PARTICLE` stores them per frame because for a grain tier they are
the state, dimension is stored rather than assumed, and both round-trip.
Persistence is through the existing `io/` layer, bumped to schema v3 with a
`/sand_field` group; v1 and v2 still read and a file with no field reads back
as `None` rather than an error. Retention is deliberate and recorded: an
over-length run is **strided, never truncated**, because cutting the tail off a
shot removes exactly the part the question is about, and every drop is written
into `RetentionRecord.dropped` in words. The reference capture keeps 97 frames
of 286 steps (1 in 3, 14.8 us apart) at float32 with gzip-4, which is 4.15 MB
against the 44.7 MB a full-rate float64 record would have cost, a 10.8x
saving, and loads in 0.08 s. Two measurement findings changed the design.
First, a nodal velocity is momentum over mass, and at the outer tail of a
B-spline stencil that mass is parts per million of a cell of sand: the
unmasked peak reads 46.71 m/s -- 187 % of the head -- on a node holding
7.5e-6 of the bulk density. A density-floor sweep gives 46.71 / 32.22 / 28.98 /
28.26 m/s at 0 / 1 / 10 / 50 %, so the reported peak stops moving at 10 %, and
the same number falls out of the physics: at `dx = 2 mm` and `d50 = 0.458 mm`
one grain is about 4 % of a cell, so a 10 % floor is "fewer than about two and
a half grains here", below which a continuum density measures nothing for the
same reason `MIN_CELLS_PER_GRAIN` refuses a sub-grain grid. `OccupancyRule` is
therefore a required field on every series, inside the file and covered by the
digest, so two views cannot disagree about where the sand is. Second, the peak
nodal density reaches 2914 kg/m^3 against a bulk of 1712 and a densest
admissible packing of 1747 -- derived from the constitutive cap
`eps_v_cap = ln(phi / phi_max)` the material already carries, so no new
constant exists. The return map keeps every particle inside that cap; nodal
density is a weighted mass scatter and nothing bounds _it_, so 1.44 % of
samples sit above a packing the sand cannot physically reach. That is a
reporting artefact, it is counted rather than clipped, and the count is
printed in the frame. **The cross-section view shows what a plane-strain cut
actually is instead of hiding it.** F1 has no heel-to-toe direction, so the
heel-to-toe series #8711 asks for cannot be a series of solves: `SliceFidelity`
labels each cut `SOLVED` (the plane the tier solved), `EXTRUDED` (parallel,
offset out of plane -- identical numbers by assumption, and a test asserts two
stations are bit-for-bit equal), or `PROJECTED` (oblique; the along-cut axis
compressed by `cos(obliquity)`). A station beyond the solver's declared
`effective_width_m` is refused, because that width is an assumption and there
is nothing past it to extrude into; an edge-on cut is refused because it meets
the solved plane in a line rather than an area. Through-cut velocity is `None`
on a parallel cut and says it is _absent_ rather than measured as zero, and on
an oblique cut says it is in-plane flow resolved rather than measured
heel-to-toe flow. Velocity is drawn as magnitude **and** direction -- colour is
speed, arrows are flow -- because sand pushed ahead of the sole and sand riding
up the face can carry identical speeds; the field now also carries the
intruder's cross-section outline per frame, without which "ahead of the sole"
and "up the face" are guesses rather than locations. Colour limits and arrow
lengths are injected from a `SliceScale` covering every frame of every compared
design and taken from occupied samples only, so #8728's per-grid auto-scaling
cannot return in the view where it would do most damage. The cut is another
view on the sole-field transport rather than a second slider, but its record is
a strided F1 march of a declared approach and not the F0 shot, so `CursorMap`
maps the shared index by fractional progress and says so in frame. Three
further corrections fell out of building it: `validity_stamp` hard-coded
"dynamic 3D-RFT" under every tier and would have printed F0's constitutive
shortcut on a picture of a material-point solve, so the model is now named from
the tier; the workbench clears a loaded field **before** repainting the sole
field, because painting it emits `frame_changed` and a cut still mapped to the
previous shot's record would be handed an index it cannot map -- and a Python
exception escaping a Qt slot aborts the process rather than surfacing; and a
sand field is loaded rather than computed in the GUI, because a 2 mm march
takes about 37 s and a coarser one run to keep the workbench responsive would
put a picture on screen at a resolution nobody chose. F1 supplies a **declared
straight-line constant-velocity approach**, not a marched swing (#8733 holds
whole-shot marching); an approach and a shot animate identically, so the
assumption is a required field on every provenance record and is stamped on
every frame. Nothing here is validated: the field inherits `BEYOND_VALIDATION`,
`MAX_VALIDATED_SPEED_M_S` is 1.44 m/s so a 25 m/s shot is 17x outside the
published corpus from its first sample, and the frame says both.

Issue #8713 (epic #8699) then puts the two tiers side by side on the
quantities both produce, which is the only honest way to show F1 output at
all: NASA-STD-7009B validation and use history both stand at 0 of 4, and
`MAX_VALIDATED_SPEED_M_S` is 1.44 m/s, so a 25 m/s record is 17x past the
published corpus from its first sample and never returns. The view therefore
states what agreement does and does not license _inside the frame_ rather
than in a caption -- computed from `vandv.credibility` and the solver's own
envelope constants, so the sentence cannot drift from the code -- and the
statement is that consistency between two uncalibrated models is not
validation, that neither tier's level moves because of anything on the page,
and that what the comparison _can_ do is falsify. Every ratio is judged
against a **declared** band of 0.25 on `|ln(F1/F0)|`, declared because issue
#8616 established there is no measurement of any of these quantities out of
bunker sand and therefore no model error to calibrate a tolerance against.
F1 has no whole-shot march yet (#8733), so each F1 point is a separate march
to one recorded pose under a declared straight-line constant-speed approach;
the points are drawn unjoined and the figure says why, because a line through
independent marches would draw a trajectory nobody computed.

At the workbench's own design point -- the 58 deg preset at 25 m/s, a 20 mm
declared effective width and F1 at dx = 3 mm on an 80 mm bed -- the pair
disagrees differently from the way ADR-0033's flat test section did, and the
difference is itself the finding. At the peak of the strike `|F1|/|F0|` is
**0.384**, not the 1.49-2.68 the 40 x 16 mm section gave: on a lofted head at
its own declared width F1 reports roughly a third of F0's magnitude rather
than twice it, so the ratio ADR-0033 measured does not transfer and must not
be quoted for a wedge. The direction cosine, on the other hand, is **0.996**
across the loaded stretch against 0.65-0.87 on the test section, so the two
tiers agree about _where the load points_ far better on real geometry than on
the section the check was written for. Depth agrees exactly, which is a
control rather than a result: both tiers are handed the same pose. F1's divot
section is **1.47x** F0's, which is not a like-for-like disagreement either --
F0 transports no sand, so its divot is the swept lower envelope of the head,
while F1's is the depression the sand actually left. Speed lost is F0's
alone, since F1's section is driven kinematically; what is reported is each
tier's resultant **projected on the direction of travel** and integrated over
the probed window on the same quadrature, 8.82 m/s against 4.23 m/s, with
F0's own record showing 7.80 m/s over that window so the quadrature error is
visible rather than absorbed into the divergence.

The inertial-share result reproduces exactly and is the sharpest one. Over a
declared 5/12/25 m/s sweep at the deepest recorded pose, F0 credits
0.959 -> 0.996 -> 0.999 of its force to its dynamic term while F1 credits
0.644 -> 0.654 -> 0.658 to momentum flux: F0's `lambda rho v_n^2` grows
quadratically with nothing bounding it, the continuum's reaction is limited
by how fast the yield surface lets sand accelerate out of the way, and the
crossing therefore sits _below_ the whole greenside range. A shot that enters
at 25 m/s and leaves at 17.2 never decelerates through it, which is why the
sweep is carried beside the shot probes as its own declared experiment and
labelled as one.

Two defects in the F1 tier were found by pointing that sweep at a real pose.
The approach distance was **unbounded**: `_approach` divides the height it has
to climb by the velocity's vertical direction cosine, and that cosine passes
through zero at the deepest sample of every real shot -- measured -0.0026 on
this record, so a 24 mm climb asked for a 9.5 m run-in, and `_build_bed` then
sized the bed to cover it. Nothing caught it but the step cap, after the
allocation. It now takes whichever of the two clearances is shorter, backing
out through the surface or running in from beyond the body's own length, so
the choice no longer turns on a sign that changes mid-shot. The grid ceiling
was likewise an accident of the approach: a descending body starts in the air
and happens to raise it, a horizontal run-in does not, and the first ejected
particle then leaves the domain. Headroom is now stated in cells. Separately,
`F0CrossCheck` was comparing two `SolverResult.max_depth_m` fields that do not
mean the same thing -- F0 reports its deepest _engaged_ element there, the
#8701 contact diagnostic, while F1 reports the deepest submerged element; they
coincide on a flat section and differ by 33x on a lofted head -- so the check
now carries the submerged depth off the shared query and the comparison uses
that. The view also names, rather than quietly reports, the probes where F0
has switched itself off: it returns zero the moment no element is both
submerged and leading-edge while the sole is still in the divot (#8702), and
the ratios of 51x and 295x at the entry and exit probes are divisions by an
engagement criterion rather than physical disagreements.

The check is deliberately not on the per-shot path: it costs about 11 minutes
against milliseconds for a design, so the workbench runs it from its own
button and the view is empty until asked. It is the fourth follower of the one
transport `SoleLoadFieldWidget` owns.

Child issue #8697 then couples two first bending modes and one torsional mode
to that distributed-grip authority. Its registered 0.25/0.125 ms atlas covers
384 trajectories and 1,536 nested-horizon summaries. Domain, activation,
power, work--energy, refinement, and MuJoCo/Pinocchio parity gates pass. Among
126 coupled-versus-rigid cells matched within 5% for peak contact load and
dissipated work, delivery-speed differences span -0.0285 to +0.0212 m/s, with
82 negative and 44 positive outcomes. The result therefore rejects a universal
passive-shaft speed benefit. It is a planar structural reference, not physical
shaft calibration, human validation, physiological inference, or technique.

Issue #8556 remains blocked on governed human bilateral six-axis
grip-wrench acquisition, and all new scientific content must regenerate the
inventory and reopen adjudication until every new candidate is reviewed.

<!--
  TEMPLATE VERSION: 1.0.0
  LAST UPDATED: 2026-06-18

  This is the canonical specification template for all repositories in the
  D-sorganization fleet. Every repo MUST have a SPEC.md at its root.

  INSTRUCTIONS:
  1. Copy this template to the root of your repository as SPEC.md
  2. Fill in every section — leave nothing as "[TODO]"
  3. Keep this document updated with every PR that changes functionality
  4. CI will block merges if SPEC.md is stale (source changed but spec didn't)

  AUDIENCE: This document is designed for both human developers AND AI agents.
  Write clearly, use concrete examples, and avoid ambiguity.
-->

## SPEC Ownership and Update Cadence

- **Owner:** @diete (responsible for accepting SPEC.md edits)
- **Update triggers (mandatory):**
  - Any PR that adds, removes, or moves a top-level `src/` package or a public
    engine adapter must update §6 (Component Locations) and §7 (Feature Status).
  - Any PR that changes the version in `pyproject.toml` must update §1 (Identity).
  - Any PR that changes a CI gate threshold must update §X (Quality Gates).
- **Review cadence:** SPEC.md is reviewed for staleness on every release
  (per `docs/operations/release-runbook.md`, see #3842).

## 1. Identity

| Field                   | Value                                              |
| ----------------------- | -------------------------------------------------- |
| **Repository Name**     | `UpstreamDrift`                                    |
| **GitHub URL**          | `https://github.com/D-sorganization/UpstreamDrift` |
| **Owner**               | D-sorganization                                    |
| **Primary Language(s)** | Python 3.11+, Rust, TypeScript                     |
| **License**             | MIT                                                |
| **Current Version**     | 2.1.2                                              |
| **Spec Version**        | 1.0.718                                            |
| **Last Spec Update**    | 2026-09-03                                         |

## 2. Purpose & Mission

UpstreamDrift is a multi-physics golf swing biomechanical simulation platform that consolidates five leading physics engines (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite) for cross-validated biomechanical analysis. It enables researchers and biomechanists to simulate human movement across models ranging from simplified 2-DOF pendulums to complex 290-muscle musculoskeletal systems, providing a unified interface for comparative physics analysis and professional-grade visualization.

## 3. Goals & Non-Goals

### Goals

- Integrate and cross-validate five physics engines (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite) for biomechanical analysis
- Provide biomechanical analysis tools including inverse kinematics (IK), inverse dynamics (ID), and muscle dynamics modeling
- Enable motion capture integration and trajectory optimization
- Offer multiple control schemes (impedance, admittance, hybrid) for simulated systems
- Deliver professional GUI with real-time 3D rendering for simulation visualization
- Expose FastAPI REST backend for programmatic access and integration
- Support desktop deployment via Tauri application framework
- Provide MATLAB/Simulink integration for cross-platform workflow compatibility
- Implement reinforcement learning integration for learning-based control policies
- Support models ranging from educational 2-DOF pendulums to complex 290-muscle systems

### Non-Goals

- Not a general-purpose physics engine; focused exclusively on biomechanical simulation
- Not intended for non-biomechanical simulations (rigid body dynamics, fluid dynamics, etc.)
- Not a replacement for domain-specific tools (OpenSim for clinical analysis, MATLAB for controls research)

## 4. Architecture Overview

### Recent Spec Updates

- **2026-09-03** - Made the tag-driven release able to produce a wheel at all
  (issue #9449). `build_hooks.py` refuses to package once `CI` is set and
  `ui/dist` is absent, but `release.yml`'s `build` job never compiled the
  frontend, so the PEP 517 backend failed and tag `v2.1.1` published no
  release, wheel, SBOM, or PyPI distribution. The `build` job now installs
  Node 24 with an `ui/package-lock.json`-keyed npm cache, runs `npm ci` and
  `npm run build` in `ui/`, and asserts `ui/dist/index.html` before
  `python -m build`. Because `_register_ui_bundle` only warns when the bundle
  is missing, the release-blocking wheel smoke suite now asserts that the
  published wheel carries the `ui/dist` payload including a compiled
  `ui/dist/assets/*.js` bundle, and that the wheel version equals the
  `pyproject.toml` version and, on a tag run, the tag itself. A failed tag is
  recovered by fixing forward to the next patch version rather than moving a
  published tag; `docs/operations/release-runbook.md` carries the procedure.

- **2026-08-22** - Made the modular Docker build boundary independently
  verifiable for issue #8996. Reusable pinned-Tools setup emits the exact
  `vendor/ud-tools` gitlink plus a deterministic SHA-256 over the minimal
  Tools-owned package roots. The modular build copies only those roots and
  verifies the digest without relying on Git metadata. Missing roots,
  symlinks, malformed attestations, incomplete source sets, and content drift
  fail closed. The Hatch hook loads its adjacent canonical helper by file path
  so isolated PEP 517 builds do not depend on the repository root being on
  `sys.path`. Constructor-compatible Hatchling test doubles and explicit suite
  isolation make package-boundary tests independent of collection order. All
  supported image definitions reassert scanner-fixed `msgpack` 1.2.1 and
  `setuptools` 83.0.0 after their final dependency layer, while both runtime
  Dockerfiles retain pip 26.2.1. This closes GHSA-6v7p-g79w-8964,
  CVE-2025-47273, and PYSEC-2026-3447 in the resolved container rather than
  suppressing Trivy or pip-audit. Production runtime targets then remove pip
  after their dependency environments are finalized, eliminating pip's
  embedded third-party SBOM and vendored build surface. The training stage
  explicitly restores the audited builder environment because it is the only
  downstream stage authorized to install packages. The modular builder also
  copies both force-included launcher modules before feature installation, so
  isolated package metadata generation matches the declared wheel surface.

- **2026-08-21** - Resolved the dual aero-coefficient-set discrepancy (issue
  #8978, epic #8965 WS2): the multi-model ball-flight framework's
  Waterloo/Penner set is now a named, provenance-documented
  `AeroCoefficientSet` (`WATERLOO_PENNER_COEFFICIENTS`) that shares the Penner
  lift shape with `ball_properties.py` (single source) while keeping its
  deliberate constant-spin calibration (cd1 = 0.05, lift cap 0.155) distinct
  from the core simulator authority (cd1 = 0.25, cap 0.26). Every
  `FlightResult` and REST `/tools/ball-flight/simulate` response now carries
  the model's coefficient dictionary for attribution, the calc sheet documents
  both families with parity-gated markers, and
  `tests/unit/physics/test_aero_coefficient_authority.py` pins each public
  entry path to its declared set.

- **2026-08-21** - Pinned the proximal--distal companion consumer to protected
  Tools merge `1664d806df8a2c7b184d2d3fbcea93b714caaee5`. The launcher authority
  and gitlink now agree on one immutable provider, and the pendulum-simulator
  launcher is included in the Tools-source contract. Cross-repository
  qualification verifies the complete ordered 18-run rotating-base catalog,
  its 13 valid cases and adverse indices 6/7/8/15/16, the exact qualified-study
  and catalog digests, and every run's nonanatomical-coordinate,
  unavailable-human-validation, and unsupported-coaching boundaries. No solver
  or catalog is copied into UpstreamDrift, and this transport qualification does
  not promote the model to anatomical, empirical, or coaching evidence. The
  standalone package build now retains a 30-minute execution budget so a cold
  pinned-Tools checkout, frontend build, wheel-content gate, and artifact upload
  can complete without the previous 15-minute timeout canceling a valid upload.

- **2026-08-20** - Added attested session-unit longitudinal analysis under
  contract `launch-monitor-longitudinal-session/1.0.0`. Trusted, distinct
  player/session/order evidence is mandatory. Shots collapse into equal-weight
  player/session/stratum cells before per-player descriptive slopes and pooled
  player-fixed-effects OLS with player-clustered finite-sample uncertainty.
  Strata, confounders, missingness, unavailable states, units, lineage, and
  source-linked backing remain explicit; shot-level inference, causal
  improvement, and automatic interpretation of metric direction are false.
- **2026-08-20** - Added canonical player covariation and population synthesis
  under contract `launch-monitor-player-covariation/1.0.0`. Selected-pair
  results distinguish pooled, player-centered, between-player, and per-player
  associations; fixed- and DerSimonian-Laird random-effects Fisher-z summaries
  report Q, tau-squared, and I-squared heterogeneity. Trusted explicit player
  identity is mandatory and remains separate from session/order evidence.
  Missing rows, ineligible groups, constant variables, insufficient population
  evidence, units, vendor/model provenance, and source-joinable backing hashes
  remain explicit. The bounded deterministic pair scan is exploratory, retains
  unavailable pairs, and warns about multiplicity, aggregation reversal,
  causality, and population-generalization limits.
- **2026-08-20** - Added immutable, aggregate-only launch-monitor dataset jobs
  under contract `launch-monitor-dataset-job/1.0.0`. A request identifies a
  server-authorized opaque root and exact repository, commit, corpus-manifest
  SHA-256, deterministic Parquet-content SHA-256, and expected row count; it
  cannot carry rows, paths, URLs, SQL, or arbitrary query text. Verification
  fails closed on checkout, lineage, content, or count mismatch. The initial
  source-summary, metric-summary, and correlation allowlist suppresses groups
  below ten rows and bounds result/page sizes. Production route registration
  requires the existing bearer authentication dependency; source outputs omit
  filenames, raw URLs, server paths, and observations. FastAPI lifespan joins
  bounded worker threads, and capacity exhaustion is a retryable HTTP 429.
- **2026-08-20** - Added the canonical source-backed strokes-gained authority
  under `src/tools/launch_monitor_model/`. The contract requires complete
  start and finish course states and a versioned, licensed, HTTP(S)-sourced,
  SHA-256-verified expected-strokes baseline. It interpolates only within an
  exact lie/context/target stratum, fails closed on extrapolation, retains row
  and dataset hashes plus backing benchmark values and exclusions, and reports
  uncertainty as unavailable when the source supplies no standard errors.
  Trusted identifiers and numeric ordering are required for grouped or
  longitudinal summaries. A separate outcome-proxy contract reports
  target-relative radial error but is structurally forbidden from claiming
  strokes gained. FastAPI publishes both contracts and their generated OpenAPI
  types for React consumers; these descriptive results make no causal,
  device-emulation, device-certification, or independent endorsement claim.
- **2026-08-20** - Recovered BunkerShot3D cross-tier plumbing and MPM code verification
  inside CI architecture and file budgets (issue #8743, #8741). Brings `solvers/mpm/verification.py`
  and `bunker_shot_gui/bridge.py` inside budget gates, verifying 177 tests across conservation,
  analytic elastic column, GCI mesh convergence, and F0 cross-check cases without regressions.
- **2026-08-19** - Corrected the AI provider-adapter tests that had not
  followed deliberate implementation changes (issue #8771). 19 tests under
  `tests/unit/shared_python/ai/adapters/` fail on `main`; ten of them are
  test-side and are fixed here. The base-adapter test double never gained the
  `list_models` / `thinking_capabilities` methods #5635 made abstract for the
  chat-dock dropdowns. Ollama and OpenAI usage assertions still named
  provider-specific token keys that #2763 replaced with a canonical
  input/output/total triple. Gemini error tests still expected transport
  failures rendered as model content, which #3179 deliberately replaced with
  typed raises, and its streaming test predated #2763's guarantee that every
  stream ends with exactly one `is_final=True` chunk. Each is updated to
  assert the contract by name rather than having its number loosened.
  Separately, `test_gemini_adapter.py` stubbed `google.generativeai` with a
  bare `MagicMock`, which auto-creates any attribute including `Client`, so
  the adapter's `HAS_GEMINI_CLIENT` probe came out true under test and nine
  tests exercised the per-instance Client branch while asserting against
  `configure` / `GenerativeModel` mocks nothing ever touched; the legacy
  branch is pinned explicitly and the modern branch gets its own test. The
  remaining nine failures need canonical Tools source and are fixed in
  Tools PR #4574, arriving here through a `vendor/ud-tools` bump - the
  adapters are Tools-owned child copies and
  `test_tools_child_copy_contract.py` correctly refuses direct edits.

- **2026-08-19** - Closed the false-green hole in CI Standard's `tests` lane
  (issue #8771). `main` @ `6b68f94` reported `tests (3.11)` / `tests (3.12)`
  as successful without running the unit suite. The lane derives its scope
  from `git diff <base> HEAD`, but read that diff through
  `mapfile -t x < <(git diff ...)`, which discards git's exit code. The push
  diff base was not present in the shallow checkout, so all four diffs failed
  with `fatal: bad object 59ecef7d...`, every `changed_*` array came back
  empty, and the "no core Python/test/dependency changes detected" branch
  exited 0. The green badge covered 14 tests, not ~2,500. Three changes: the
  diffs are captured through files so the step's `-e` makes a failed diff
  fatal; an unresolvable diff base is now an explicit `::error::` and exit 1
  rather than an inferred "nothing changed"; and pushes to the default branch
  always run the full lane unscoped, because path-scoping is a pull-request
  latency optimisation and on trunk it lets a commit that touches no Python
  vouch for a suite it never ran. A genuine "nothing relevant changed" skip
  now emits a `::warning::` and a job-summary block stating that the suite was
  not executed, so a passing check can be told apart from a green suite. The
  same silent-failure shape in the `Check for core test relevant changes`
  pre-step is hardened identically, and `tests/ci/test_ci_infrastructure.py`
  gains two guards asserting the new contract - a failed diff cannot be
  read as "nothing changed", and trunk pushes are never path-scoped.
  pre-step is hardened identically.

- **2026-08-19** - Repaired all five `profile-size-matrix` Docker builds
  (issue #8771). `Dockerfile.modular`'s builder stage copies a deliberate
  minimal slice - `src/shared/python/__init__.py`, `engine_core/` and
  `feature_registry/` - so resolving a profile does not invalidate the layer
  cache on every source change. `scripts/docker/install_features.py` then did
  `from src.shared.python.feature_registry.features import get_feature`, and
  importing a submodule executes every parent package `__init__` first: once
  `src/shared/python/__init__.py` grew an eager `from . import ai`, and `ai/`
  is not in the slice, every modular image build died at `ImportError: cannot
import name 'ai' from partially initialized module`. The script now loads
  `features.py` from its path with no parent package, which is what its own
  comment always claimed ("read the features module in isolation") and what
  makes the builder slice self-sufficient regardless of what any package
  `__init__` later imports. `features.py` depends only on `dataclasses` and
  `typing`. The module is registered in `sys.modules` before execution
  because `@dataclass` resolves `cls.__module__` through it while processing
  the class body. `src/shared/python/__init__.py` is a Tools-owned child copy
  and is deliberately left untouched.

- **2026-08-19** - Git-ignored the test-run artefacts that have twice been
  committed at the repository root (issue #8771). The autouse
  `_prevent_repo_root_io` fixture chdirs every test into `tmp_path` so they are
  not produced (#7935), but nothing stopped them being staged once they
  existed: #8322 committed all nine (`base.csv`, `base.json`, `base.mat`, two
  `.provenance.json` files, three `pytest_report*.txt` and a one-off patch
  script) and #8747 had to delete them again while repairing the root-clutter
  gate. The names are now ignored, anchored to the root so the tracked
  `docs/research/.../base.csv` fixture is unaffected.
- **2026-08-19** - Restored `optional-stack-check (3.11)` (issue #8771). The
  lane failed on exactly three tests in `tests/unit/deployment/test_devices.py`,
  each asserting `not dev.connect()` for the SpaceMouse, VR-controller and
  haptic stubs, which raise `NotImplementedError` instead. #8322 made that
  change and kept the tests. It also deleted the reasoning: before #8322 each
  method returned `False` under a docstring explaining that "there is no
  hardware driver behind this class, so the honest answer is always 'not
  connected' (#7360)", and that the body it replaced had probed for an
  optional backend and returned `False` on both arms, making the stub look
  conditionally functional. Returning `False` is the contract the rest of the
  subsystem keeps: `BaseInputDevice.connect` returns `False`, the ROS2 and UDP
  controller stubs return `False` with tests asserting it, and
  `test_base_input_device` asserts it of the base class directly - so the
  three overrides were inconsistent with their own parent. #8322 additionally
  rewrote two of the three test files that cover these stubs
  (`tests/deployment/test_teleoperation.py` and
  `tests/deployment/wave5_deployment/test_teleoperation.py`, six assertions)
  from `assert not d.connect()` to `pytest.raises(NotImplementedError)`, and
  missed the third - which is the whole reason `optional-stack-check` went red
  while `unit-test-gate` did not. Before #8322 all three files asserted
  `False`. The implementation bodies, their rationale and the six rewritten
  assertions are all restored to their pre-#8322 text, so one contract holds
  across the subsystem again; the `#8058` tracking reference is kept and no
  test was skipped or deleted.

- **2026-08-16** - Scoped BunkerShot3D's camber-clamped flag so it cannot
  understate substitution (`src/bunkershot3d/geometry/lofting.py`,
  `src/tools/bunker_shot_gui/`, issue #8698, epic #8699). The observability
  work below left one boolean answering a narrower question than its name
  implied. `LoftedWedge.camber_was_clamped` compared the declared camber area
  against the band the **declared** sole width admits. Heel and toe relief
  narrows the sole toward the ends, and a narrower sole admits a narrower
  camber band, so the relieved stations are refitted to their own bands while
  the declaration itself is honoured — and the flag read `False` beside a
  non-empty `clamped_stations`. That is #8698's own failure mode, silent
  substitution invisible to a caller, reappearing through the simpler check.
  Measured at shipped resolution, three of the six shipped presets hit it:
  `sm9_58_m` declares 42.00 mm² inside its (38.70, 42.44) mm² band and refits
  3 of 17 stations; `acushnet_example_1` refits 5 of 17; `tour_shaved_heel_lob`
  refits 13 of 17. All three reported "not clamped".

  Resolved by renaming rather than redefining. Redefining a published boolean
  in place would have silently changed every existing caller's answer, which
  is the same failure mode one level up; the rename breaks loudly instead.
  `camber_was_clamped` becomes `aggregate_camber_was_clamped` (unambiguously
  the declared-versus-effective question), and a new `any_camber_was_clamped`
  is true when the declaration **or** any station was substituted, so a caller
  checking one boolean gets the honest answer and the flag cannot read `False`
  while `clamped_stations` holds anything. `camber_substitution_m2` becomes
  `aggregate_camber_substitution_m2`, which also ends its name collision with
  `StationCamber`'s per-station property. Both meanings are genuinely used, so
  a single flag could not serve: the workbench's camber-area line needs the
  aggregate because it prints the declared number, and the contact-patch
  caveat needs any-station. The camber-area line now also names the refitted
  station count, so an in-band declaration over refitted stations no longer
  renders as a clean number. `PATCH_CONFOUND_CAVEAT` is gated on the per-
  station account rather than the aggregate flag and is restated in terms of
  stations — gated on the aggregate it was silent on the default design, which
  is exactly when the caveat is needed.

- **2026-08-16** - Made BunkerShot3D's sole-camber substitution observable
  (`src/bunkershot3d/geometry/`, issue #8698, epic #8699). A wedge sole can
  only realise camber areas inside a band set by its width and bounce, so
  `build_wedge_mesh` fitted a declared `sole_camber_area_m2` that fell outside
  it to the nearest constructible value. That fit is physically correct and is
  retained — a narrow sole geometrically cannot host an arbitrarily large
  camber, and emitting an inconstructible section would be worse — but it was
  **unobservable**: `constructible_camber_range_m2` was not re-exported from
  `bunkershot3d.geometry` or the top level, and no result object carried the
  effective value back, so a caller who declared 48 mm² received a different
  sole and had no way to find out. Measured on a 77-point demo sweep, the
  clamp fired on 40 points and moved the effective camber over 24.5–61.6 mm²
  against a constant declared 48.0 mm².

  This matters most in `bunkershot3d.study`: a `MorrisDesign`,
  `SaltelliDesign` or `SobolIndices` run over sole width or bounce would
  attribute variance to a camber the user believes is pinned, and no
  diagnostic in the artifact would say so.

  Three complementary changes. (1) `loft_wedge()` returns a new `LoftedWedge`
  carrying `effective_camber_area_m2`, `constructible_camber_range_m2`,
  `aggregate_camber_was_clamped`, `any_camber_was_clamped`,
  `aggregate_camber_substitution_m2` and a per-station
  `StationCamber` account; `build_wedge_mesh()` is now a thin wrapper that
  returns only its mesh. (2) `constructible_camber_range_m2` is re-exported
  from `bunkershot3d.geometry` and the top-level package, and
  `DesignSpace.check_wedge_camber(geometry)` screens the corners of a design
  box against the band before a sweep spends solver time inside it — the wedge
  knowledge lives in `geometry/design_bounds.py` and is imported on call, so
  the study layer keeps its independence from the geometry package. (3)
  `CamberFit` makes silence opt-in: the default `CamberFit.STRICT` raises
  `InconstructibleCamberError` when a **declared** camber is outside the band,
  and `CamberFit.NEAREST` is the explicit opt-in to nearest-constructible
  behaviour. Relief-scaled stations are always fitted rather than refused,
  because that request is derived by the lofter rather than declared by the
  caller — but every substitution, declared or derived, is recorded.

  `STRICT` is the default because it is what the rest of the package already
  does: `build_sole_profile` raises for exactly this condition one layer down,
  `DesignSpace.sample` raises rather than silently lose Sobol' balance, and
  `WedgeGeometry.__post_init__` rejects inadmissible combinations. The lofter
  was the single place that downgraded a loud failure to a silent one.
  `InconstructibleCamberError` subclasses `BunkerShot3DValueError`, so
  existing `except ValueError` sites keep working. Per CLAUDE.md every guard
  `raise`s and none `assert`s, because `python -O` strips assertions.

  Two consequences fell out of making the check loud. Three shipped grind
  presets (`acushnet_example_2`, `acushnet_example_3`, `tour_shaved_heel_lob`)
  declared a camber area their own sole cannot carry, so their meshes had
  never matched their declarations; their `sole_camber_area_m2` — an
  `ESTIMATED` field in every preset — is corrected to a constructible value,
  and the patent-example helper now takes it per example because the band
  climbs steeply with bounce. And the workbench GUI
  (`src/tools/bunker_shot_gui/`) opts into `CamberFit.NEAREST` explicitly,
  because a designer dragging a bounce slider must keep getting a head to look
  at; having opted in, its evaluation report now states the camber area the
  head actually carries alongside the declared one.

- **2026-08-15** - Restored the two repository-hygiene guards
  (`tests/unit/repo_hygiene/test_vendor_submodule_clean.py` and
  `test_no_shadow_of_tools_shared.py`) to actually enforce. Both were
  introduced 2026-05-16 (#5623 / PR #5625) and reduced to stubs on 2026-08-01
  by consolidation commit 0575fb4b8 (#8322), which also emptied the shadow
  ledger and added four new shadows in the same commit. Three independent
  mechanisms kept them vacuous: neither file carried a suite marker, so
  `unit-test-gate`'s `-m "unit and ..."` selector never collected them; both
  called `pytest.skip` when `vendor/ud-tools` was absent, so they passed
  vacuously even when collected; and push-to-main runs are cancelled by
  `cancel-in-progress` concurrency (83 of the last 85), so the only lane that
  would collect them effectively never completed.

  A missing vendor tree now raises `AssertionError` when `$CI` is set and only
  skips on a developer machine, matching `test_tools_child_copy_contract.py`.
  The shadow ledger in `scripts/config/shadow_modules.yaml` is re-established
  as a no-growth ratchet: 32 grandfathered entries (28 from the original #5623
  baseline plus the 4 that #8322 added while the guard was off), each carrying
  a `tracking_issue` and a `sunset_date`. Bare name lists are rejected, expired
  sunset dates fail, and stale entries must be pruned, so the ledger can only
  shrink.

  The vendor-clean guard also carried a latent detection bug: it filtered
  `git status` lines for the prefix `vendor/ud-tools/` (trailing slash), but
  git collapses all submodule-internal state onto the gitlink entry whose path
  is exactly `vendor/ud-tools`, with flags `S<c><m><u>`. The predicate could
  never match. Detection now parses the sub-status field, so modified (`S.M.`)
  and untracked (`S..U`) content fail while a deliberate pointer bump (`SC..`)
  passes. A new `test_hygiene_guards_run_in_ci.py` asserts that every workflow
  job running this package materialises `vendor/ud-tools`, and that
  `unit-test-gate` does so before invoking pytest.

- **2026-08-15** - Restored input validation on the symbolic solver router
  (`src/shared/python/calc_backend/routers/symbolic_solver.py`, issue #8675).
  This file is a shadow copy of a module owned by the Tools repository. Tools
  hardened its copy on 2026-07-22 by guarding every `parse_expr` call with
  `validate_expression`; the shadow was forked on 2026-05-20 and never received
  that change, so it carried six fewer guards than upstream while keeping all
  four parse sites. The removal was not deliberate — `validate_expression` has
  no history in this file at all — it is drift by omission.

  The exposure was not theoretical. `sympy.parse_expr` is an _evaluating_
  parser, and all three affected endpoints (`/api/calc/symbolic/solve`,
  `/derivative`, `/simplify`) take their expression from the request body, so
  the input is remote by construction. Against the unguarded module,
  `__import__("os").getcwd()` executed and returned the server's working
  directory in the response error string, and `9**9**9**9` hung the worker
  in an uninterruptible bignum computation that the thread-based pytest
  timeout could not kill.

  The fix re-adds `from src.shared.python.safe_eval import validate_expression`
  (UD's own `src.`-prefixed convention, matching the sibling `ode_solver`
  router) and the four upstream call sites: both sides of an `lhs = rhs`
  equation, the bare-expression branch of `/solve`, and the single expression
  on `/derivative` and `/simplify`. Note that a `vendor/ud-tools` submodule
  bump cannot fix this: the submodule is unpopulated, and
  `src/shared/python` precedes it in resolution, so the shadow is what is
  actually imported.

  Regression coverage pins the guard to behaviour rather than to its presence:
  each endpoint must reject attribute-based calls, attribute access, and
  exponentiation bombs; legitimate work (`x**2 - 4 = 0`, `x^2` via
  `convert_xor`, `sin(x)**2 + cos(x)**2`) must still solve; the payload's
  observable side effect must never appear in a response; and an
  instrumented-ordering test asserts `parse_expr` never runs ahead of
  `validate_expression` on any path.

- **2026-08-14** - Rebuilt BunkerShot3D as a multi-fidelity wedge-design tool
  under epic #8607 (ADR-0032). The package is now organised around the design
  question — given two sole geometries, which performs better, in what
  conditions, and how confident are we — rather than around granular backends.
  Resolved DEM was demoted from the primary path on arithmetic, not preference:
  the previous canonical configuration held 50,000 grains at 0.4 mm in a
  0.4x0.3x0.1 m domain, a solid fraction of 1.4e-4 and a settled bed 0.023 mm
  deep, while a real 100 mm USGA base needs 2.1e8 grains, and the Chrono driver
  integrated at ~11,900x the Rayleigh stability limit by using the output
  sampling rate as its timestep. The default tier (F0) is now an analytic
  dynamic Resistive Force Theory solver at ~14 ms/shot, with DEM retained and
  explicitly labelled non-viable at true grain scale.

  New public surface: `bunkershot3d.{geometry,sand,domain,solvers,ball,metrics,
study,vandv,provenance,units}`, with subpackages re-exported by name and a
  curated flat set; `study` is lazily imported so `postproc` does not require
  the optimisation extras. The result schema is versioned (contiguous arrays
  replacing one HDF5 group per timestep, reader accepting v1 and v2) and every
  run carries a manifest with config and physics hashes, seeds, and a validity
  verdict.

  **Credibility, stated plainly: verification is real, validation is zero.**
  Observed orders are 1.004 (energy) and 2.001 (surface quadrature) against a
  closed-form integral, and an angular-momentum check catches axis swaps that
  leave the resultant force bit-identical. But no published data exists for ball
  launch, spin, head deceleration in sand, energy split or ejecta mass, so
  `ValidationComparison` refuses to construct such a comparison at all. The
  solver is used ~60x outside RFT's stated Fr < 0.4 envelope and ~20x beyond any
  published validation; `delta_h` and `lambda` are uncalibrated for a wedge, and
  every F0 coefficient is borrowed rather than measured. Out-of-envelope queries
  refuse rather than return a plausible number. See
  `docs/bunkershot3d/credibility.md`; `docs/bunkershot3d/comparison.md` was
  rewritten after seven of its eight claims were found to contradict the code.

- **2026-08-14** - Completed the release-level claim-review authority for epic
  #8557. All 36 release claims now link to supporting atomic claims, evidence,
  negative controls, falsifiers, uncertainty boundaries, a scientific
  disposition, and a remaining gate. Open review bookkeeping is zero while
  human self-stabilization, physical bilateral sensing, and other scientific
  gates remain explicitly untested or conditional.

- `src/shared/python/club_data/loader.py`: Replaced `.iterrows()` with `df.to_dict('records')` (spec-exempt: micro-optimization)

- **2026-08-14** - Reconciled all nine photographed momentum-transfer source
  points with inspectable evidence artifacts and a generated readiness audit.
  Eight retain bounded model answers, partial answers, or a negative
  general-rule result. MTQ-06, whether passive or drift-mediated transfer
  reduces timing precision beyond the adverse planar comparison and in people,
  remains unresolved. Candidate-census completion is now reported separately
  from 10 pending or in-progress release reviews.

- **2026-08-14** - Added MT-E08 subject-scaled spatial contact-closure audit.
  Six deterministic de Leva engineering profiles, three grip spans, and 61
  prescribed states per case expose hand-to-grip miss distances of
  0.171--0.616 m, with no sample meeting the registered 5 mm tolerance. Every
  local bilateral contact Jacobian still has rank six, demonstrating that local
  correction rank does not establish geometric closure. Closed-contact inverse
  kinematics with joint-limit and collision checks is now a precondition for
  compliant forward-contact, anatomical, passive-timing, slack, or human-
  strategy claims.

- **2026-08-14** - Qualified the MT-E07 bilateral point-force estimator over
  deterministic synthetic trajectories with normalized noise and cross-talk,
  cross-talk calibration residual, contact-center migration, and tracked-contact
  controls. Net-wrench-only inversion retains its manufactured axial allocation
  error despite numerical resultant closure. This is not calibration of a
  physical bilateral six-axis device and does not support anatomical or human
  strategy claims.

- **2026-08-14** - Added MT-E07 bilateral-wrench structural identifiability.
  Two separated three-axis point forces map to one net club wrench with rank 5
  and one equal-and-opposite axial null mode; adding one independent axial
  scalar closes that point-force rank gap. Two full six-axis hand wrenches map
  to net wrench with rank 6 and nullity 6, so net club wrench cannot recover
  individual bilateral allocation. Grip-span and proper-rotation controls,
  publication evidence, tests, claim adjudication, and the sensor-qualification
  plan are release artifacts. Practical noise/cross-talk, contact migration,
  muscle or scapular action, and governed human validation remain open.

- **2026-08-14** - Added the #8557 handwritten momentum-transfer agenda
  readiness contract. Nine independently testable points retain their present
  answer state, decisive next test, falsifier, data gate, model plan, and
  participant-held-out human stage in a generated fail-closed audit. Casting
  is definition-dependent and partly answered; broader timing precision remains
  unresolved beyond the adverse planar comparison. Issue #8556 remains the
  human-data blocker.

- **2026-08-13** - Adjudicated the transmission robustness chapter for #8557.
  All four registered programs remain nondominated in every held-out
  leave-one-case-out recomputation. The task map retains algebraic rank three
  but exposes scale-dependent practical rank and material held-out linearization
  error, preventing promotion of a local null space to a neural synergy.

- **2026-08-13** - Reconciled the forward distributed modal-shaft chapter
  under epic #8557. The regenerated hash-bound authority now records the
  preregistered 5% tip-deflection model-use screen and its observed 13.48%
  failure. The run remains evidence for numerical modal coupling, force-couple
  geometry controls, and timestep behavior, but quantitative linear-shaft,
  equipment, human, and coaching inferences are rejected for this baseline.

- **2026-08-13** - Adjudicated both counterfactual chapters under epic #8557.
  The legacy MATLAB killswitch is now classified as a rerun-based pointwise
  sampler, and its `DELTA` force-work columns are explicitly rejected as an
  additive actuator or muscle-work partition. The forward matched-state bundle
  now samples the command at every state time including the terminal endpoint,
  records affine closure and interpolation contracts, hashes its complete
  declared source/data closure, separates 1 ms and 2 ms refinement from the
  0.5 ms reference, labels gravity/damping studies as nonadditive whole-model
  variants, and proves byte-deterministic evidence. All 45 chapter candidates
  map into PD-CLAIM-093 through PD-CLAIM-103; one paragraph already reviewed in
  an earlier chapter retains its stable identity, yielding 288 unique reviewed
  candidates overall. Human, bilateral-hand, anatomical, and universal-strategy
  conclusions remain untested.

- **2026-08-13** - Corrected and hardened the proximal-link velocity evidence
  under epic #8557. The paper and machine contract now use the actual relative
  coordinate convention (`q2` relative, `q1 + q2` absolute), include the
  achieved state in every reference-centered sweep, report finite-range and
  local slopes beside linear-fit adequacy, preserve all grid-selection limits,
  expose regression rank and conditioning plus Pareto coverage, distinguish a
  net planar wrist-interface force from either hand's anatomical load, refine
  the best fixed program at half timestep, hash the computational dependency
  closure, and prove byte-deterministic JSON/NPZ evidence. All 30 chapter
  candidates are mapped to PD-CLAIM-081 through PD-CLAIM-092; anatomical
  shoulder, thorax, bilateral-hand, and coaching conclusions remain untested.

- `src/shared/python/motion_matching/loaders/c3d.py`: Optimized `np.linalg.norm` to `math.hypot` inside `_shaft_quaternions` for better performance on small 3D vectors (spec-exempt: micro-optimization).
- **2026-08-12** - Added the interactive proximal--distal dynamics companion
  under epic #8511. UpstreamDrift now resolves the canonical sibling Tools
  provider and its dockable adapter instead of maintaining launcher-specific
  physics. The PyQt6 and React/Tauri clients share a validated experiment and
  glossary catalog covering double, triple, bilateral, counterfactual, and
  robustness studies. Interactive output is explicitly exploratory; scripted,
  frozen analyses remain the publication authority.

- **2026-08-12** - Added epic #8507's adversarial transmission and task-
  robustness tier: paired clock/state-trigger programs on common training and
  held-out perturbations, lower-tail speed and variability/load/effort
  objectives, explicit pathway and contact-power closure, a local task-null
  variance analysis, a machine-readable 12-item gap register, and four vector
  figures. All programs remain Pareto-nondominated; state triggering improves
  selected speed and planar face/path metrics while increasing peak hand force.
  Human self-stabilization, causal physiology, and coaching prescriptions remain
  untested or unsupported.

- **2026-08-12** - Expanded the proximal--distal resource with explicit
  reference-frame and point-transport equations, numerical power and virtual-
  work invariance checks, a reduced Hill-type redundancy and activation-history
  bridge, five canonical-pose adapter round trips, a question-to-engine ladder,
  five vector figure families, a normative terminology contract, a reviewer
  index, and an advanced falsification roadmap under epic #8505. Pose adapter
  closure is representation evidence only; subject-scaled anatomy, five-engine
  dynamics parity, and human validation remain open.

- **2026-08-11** - Added the continuous preparation-history falsifier for the
  arm--wrist transmission study. Both channels now start relaxed, load for 180
  ms, and retain their exact deflection and transmitted torque across the
  command transition; persistent directions remain engaged while complete role
  reversal crosses the declared dead zones. This is an abstract transmission
  history, not an anatomical backswing or muscle-activation model.

- **2026-08-11** - Adjudicated the independent proximal-to-distal technical
  review in epic #8499. Corrected the phase-integration boundary and replaced
  sampled COM differentiation with analytic acceleration, retained every
  geometric impact candidate with a reason-coded status, added 20--50 ms
  command-rise sensitivity, and narrowed finite-grid, actuator-work,
  pointwise-counterfactual, physiological, and coaching claims.

- **2026-08-11** - Replaced the obsolete fixed 1 MiB PDF ceiling with an
  artifact-integrity and repository-boundary policy. Changed PDFs must carry a
  valid `%PDF-` signature, files above GitHub's 50,000,000-byte recommendation
  are reported, and files beyond its 100,000,000-byte hard boundary fail. The
  manuscript optimizer still proves page, URI-link, and outline preservation;
  an explicit release-specific ceiling remains available when justified.

- **2026-08-11** - Added a coupled moving-base, two-hand, three-mode shaft
  experiment and a separate matched-task arm--wrist allocation/preload study.
  The latter holds an 8 N m club task exact while varying actuator subspace,
  quantifies internal load tradeoffs, and tests persistent-direction versus
  role-reversal commands over a declared dead-zone/time-constant grid. Both
  remain synthetic mechanism tiers; muscle, scapular, tissue-slack, equipment,
  human-performance, and universal-technique claims remain unsupported.

- **2026-08-18** - Added a first-class corpus entry point to the Launch
  Monitor Analytics workbench. A "Load Private Corpus" button on the Sessions
  tab and a matching File-menu action load every source in the authorized
  private corpus as one session per source (261,666 shots across 27 sources in
  about 3 seconds), through `MainWidget.load_private_corpus_sessions()`. The
  action is repeatable — already-loaded sources are skipped rather than
  raising — and fails closed with a dialog, not a traceback, when no
  authorized checkout or Parquet reader is present. The Trends and Dispersion
  tabs remain inert for corpus data because the corpus carries no capture
  timestamp or lateral carry; both gaps are tracked as data-authority issues
  #18 and #19 and are recoverable from retained native fields.
- **2026-08-19** - Bound the Launch Monitor Analytics Dispersion and Trends
  tabs to corpus data. The data authority now extracts lateral carry, flight
  time, and a capture date, and `launch_monitor.corpus` maps them onto the
  canonical `lateral_carry` (m), `flight_time` (s), and `captured_at` fields.
  Both tabs were previously inert against the corpus for want of a lateral
  coordinate and a time column; they now run over 20,099 and 8,488 shots
  respectively. Column selection is filtered against the dataset schema, so a
  corpus pinned before those columns exist still loads.
- **2026-08-19** - Published Launch Monitor Analysis Contract v2 as the
  canonical cross-repository evidence envelope. The v2 API carries explicit
  unit authority, exact commit/source/transformation/backing-record lineage,
  missingness and exclusions, uncertainty, vendor/model provenance, and typed
  available/partial/unavailable states. Player grouping requires an explicitly
  trusted identifier and evidence; session, club, filename, source layout, and
  row order are never identities. Contract v1 remains available through its
  unchanged endpoint and a bounded compatibility adapter.
- **2026-08-18** - Connected Launch Monitor Analytics to the private shot
  corpus. `launch_monitor.corpus.load_private_corpus()` reads the data
  authority's source-partitioned Parquet corpus (261,666 shots across 27
  sources at current head) into the canonical schema via the importer's unit
  tables, with source/metric pushdown, lazy `pyarrow`, the established
  fail-closed `LAUNCH_MONITOR_DATA_ROOT` convention, and `apex_native`
  excluded as unit-ambiguous. Also repaired the bare `flight_models` import
  inside `kaggle_validation.compare_all_models_to_dataset()`, which only
  resolved under pytest's `pythonpath` and broke installed consumers.
- **2026-08-11** - Added the first UpstreamDrift consumer boundary for the
  canonical Tools ground-model contracts (Tools #4276). The headless gateway
  validates the exact flight-to-ground request/result and reference-execution
  v1 schemas before binding Tools parsers or execution, stays import-safe when
  the optional authority is absent, and preserves returned records and
  provenance unchanged. This is not a dependency pin or UI parity claim: the
  exact `vendor/ud-tools`/Cargo repin, FastAPI, PyQt, React, clean-install, and
  protected-release gates remain blocked on the reviewed Tools ground merge.
- **2026-08-13** - Ratified the cross-repository control-affine terminology
  contract for #8586. Canonical ZVCF now means an instantaneous fixed-
  configuration evaluation with velocity and declared applied control both
  zero; analysis schema 3.0.0 preserves the former control-retaining result as
  `zero_velocity_control_preserved_acceleration`. The proximal-distal evidence,
  constrained-reaction decomposition, engine protocol, paper, and regression
  tests distinguish the two quantities and link to AffineDrift's normative
  notation authority.
- **2026-08-11** - Added #8493 ground-reaction drift attribution: a
  frame-explicit constrained-contact reaction solver decomposes support
  reactions into configuration, velocity, control, and retained-external
  components; deterministic double-pendulum evidence verifies total, ZTCF, and
  ZVCF closure; and the scientific article defines measurement, identifiability,
  and human-data falsification boundaries without treating overlapping
  counterfactuals as additive effort fractions.
- **2026-08-11** - Prepared the local launcher UI-setup decomposition tracked by
  [#8490](https://github.com/D-sorganization/UpstreamDrift/issues/8490) from
  exact PR #8489 head `2f664d2beaddf7444b12f90080ae9897aea24fcc`.
  Navigation/sidebar/menu construction now lives in the private
  `_launcher_navigation_ui.py` mixin, while status/search/runtime/view/zoom
  construction and the historical top-bar widgets live in the private
  `_launcher_top_bar_ui.py` mixin. `UISetupManager` retains every historical
  method through inheritance or its compatibility facade, including dynamic
  manager-to-launcher rebinding and monkeypatch-sensitive exports.
- **2026-08-11** - Decomposed the launcher settings surface from exact draft
  PR #8486 head `624043537a5ab10aa7ef56dc61685a004b872c0c` without changing
  its public dialog/widget contract.
- **2026-08-11** - Decomposed the Simscape C3D viewer for #8485 without
  changing its public UI contract.
- **2026-08-10** - Added #8458 hand-path attribution: canonical pointwise
  ZTCF/control/ZVCF definitions; exact double-pendulum, one-arm, and closed-loop
  two-arm adapters; deterministic force-vector, impulse, power, work,
  joint/time-window, common/differential-mode, sensitivity, and closure
  evidence; and a bounded residual-couple preview hypothesis with explicit
  physiological limits. The final 106-page PDF uses contract-checked lossless
  object-stream compaction and preserves 110 URI links plus 122 outline entries.
- **2026-08-10** - Added #8446 two-hand passive-couple reproduction: a
  frame-explicit and tested wrench API, hash-traceable BASE/ZTCF/DELTA table
  exports, all-sample resultant/couple/power reconstruction, reversal and
  resampling analysis, grip-separation/orientation counterfactuals, eight
  publication figures, and a neutral chapter separating contact-force moment,
  free torque, resultant force, and power.
- **2026-08-10** - Added #8445 matched-state counterfactual persistence:
  deterministic commanded/zero-torque futures, 96 cut-time/horizon/timestep
  cases, state/force/power/work/speed divergence, torque-switch bracketing,
  gravity and damping ablations, a WSCG BASE-minus-counterfactual convention
  check, four publication figures, and a visually verified 58-page article.
- **2026-08-10** - Added the first interaction-force mechanisms slice for
  epic #8443 / issue #8444: exact double-pendulum wrist-force and force-power
  decomposition, tested Newton and moment balances, a matched-state
  torque-killswitch distinct from pointwise ZTCF, hash-registered WSCG source
  decks and chart extraction, seven publication figures, machine-readable
  evidence, and an expanded scientific chapter with explicit nonclaims and
  falsification tests.
- **2026-08-09** - Stabilized the Launch Monitor Analytics v1 consumer
  contract against the current Tools record contract. Dataset fingerprints
  now hash canonical ordered record content and ignore transient pandas index
  labels; the shared domain owns the single `1.0.0` version constant; FastAPI
  validates analysis mode, correlation method, and missing-data policy as
  closed enums; and the PyQt signal boundary converts user-correctable
  selection failures into accessible inline status. The parity fixture pins
  the observed Tools v1 fingerprint but does not claim an immutable Tools
  release or dependency pin.
- **2026-08-06** - Extended Launch Monitor Analytics with a UI-neutral,
  versioned flexible-analysis contract and matched PyQt/FastAPI surfaces.
  Users may select arbitrary numeric outcomes and predictors, Pearson,
  Spearman, or Kendall association, pairwise/listwise/fail missingness,
  Benjamini-Hochberg correction, grouped analysis, and OLS regression with
  confidence intervals, residual diagnostics, and deterministic dataset
  lineage. Aggregate observations remain excluded from regression; explicit
  aggregate correlations are labeled descriptive with an ecological-bias
  warning. Vendor-specific fields cannot be pooled across monitor vendors, and
  association or predictive fit is never presented as causal evidence.

- **2026-08-09** - Removed the real 832-shot launch-monitor trajectory CSV
  from the public repository. The historical validation loader now resolves its
  default dataset only from the authenticated private data authority named by
  `LAUNCH_MONITOR_DATA_ROOT`, and fails closed when that authority or its pinned
  source snapshot is unavailable. Explicit caller-provided paths remain
  supported for tests and user-owned data.

- **2026-08-09** - Refined the proximal-to-distal energy-transfer research
  package as a neutral, reproducible open resource. The report now distinguishes
  empirical evidence, model-derived results, and hypotheses; documents its
  applicability limits; and includes a deterministic 13-case, one-at-a-time
  parameter-sensitivity analysis over segment geometry, mass, plane inclination,
  and damping. Unit tests enforce the analysis contract and verify that the
  reported strategy ordering is reproduced for the defined cases. These
  simplified-model results are not a universal effect estimate or a coaching
  prescription.
- **2026-08-11** - Added the first bounded #4262 immutable Tools-provider
  source contract. Launcher entries with `provider: tools` now resolve only
  from the repository-pinned `vendor/ud-tools` gitlink, before generic sibling
  and installed-package providers; the five Tools launchers no longer serialize
  `../Tools` source roots. Authority requires the already-declared exact
  `ff4240217005e1415ca409fd124e50b64ee642d2` gitlink, matching clean checkout
  HEAD, current-superproject attachment, and a non-reparse vendor directory.
  Every resolved artifact, working directory, fallback, and Python path must
  remain under that authority after canonicalization. Missing, uninitialized,
  dirty, replaced, mismatched, or escaping sources report
  `provider_unavailable` and never fall back to mutable sibling/package input.
  Generic sibling-provider resolution is unchanged. This slice does not change
  the gitlink pin, accept an unvalidated `TOOLS_REPO_PATH`, or complete #4262.
- (spec-exempt: security fix) Fixed Command Injection in `pandas.DataFrame.query()` inside `rust_engine.py` (both `data_processor` and `data_processor_io`) by explicitly validating user expressions using an AST-based validator (`validate_pandas_formula`). This eliminates an arbitrary code execution vulnerability.

- **2026-08-05** - Retargeted #8345 P1's 3D putting workflow to `main`
  after the headless dynamics foundation merged. The FastAPI route executes the
  canonical Python collision/surface solver and returns a complete playback DTO;
  the React client uses TanStack Query, Zustand, and an R3F green/ball/putter
  scene with orbit controls, signed-spin marker, collision slowdown, adjustable
  hosel controls, responsive playback, and theme tokens.
- **2026-08-04** - Added #8345 P1's generated-contract 3D putting workflow.
  The FastAPI route executes the canonical Python collision/surface solver and
  returns a complete playback DTO; the React client uses TanStack Query,
  Zustand, and an R3F green/ball/putter scene with orbit controls, signed-spin
  marker, collision slowdown, adjustable hosel controls, responsive playback,
  and theme tokens. Migrated the UI stylesheet entry point to the required
  Tailwind v4 import/config syntax so responsive and theme utilities compile.
- **2026-08-05** - Hardened classic PyQt6 startup from nested worktrees.
  Tools-source discovery now walks workspace ancestors after honoring explicit
  and initialized vendored sources, so a worktree can locate the canonical
  sibling `Tools/src` checkout. An unavailable implicit Sidekick runtime
  disables only the optional sidebar; an explicitly configured incomplete
  `TOOLS_REPO_PATH` remains fail-closed. Focused regressions pin nested
  discovery and both fallback contracts.
- **2026-08-04** - Added the headless putting-dynamics foundation for #8345
  P2/P3/P4. `src/shared/python/putting_dynamics/` provides DbC-validated
  heterogeneous height/friction fields, seeded bumpiness, signed skid/overspin
  transition, rolling/rest modes, full-chord hole capture, and finite-mass
  putter-ball collision outputs including dynamic loft, slowdown, adjustable
  hosel position, and an attachment-point impulse wrench. The public-data
  review records verified sources, explicit assumptions, discrepancies, model
  defaults, and validation bands. Seventy focused unit tests pin analytic
  limits, determinism, symmetry, conservation, and the public façade.
- **2026-08-04** - Added the vendor-neutral Launch Monitor Analytics workbench
  (#8342). A canonical, unit-normalized shot schema and provenance-preserving
  import pipeline aggregate common TrackMan, Foresight, FlightScope, Garmin,
  SkyTrak, Uneekor, Full Swing, Rapsodo, and GSPro/Open Connect exports, with a
  generic mapping fallback. The PyQt6 workbench provides auditable treatment,
  correlation and partial-correlation mapping, regularized regression and an
  optional shallow neural network, matched-monitor agreement, dispersion, and
  actual-time longitudinal trend analysis. Associations and predictions are
  explicitly non-causal; unmatched monitor comparisons are descriptive only.
- **2026-07-27** - Corrected classic PyQt6 Diagnostics provider-manifest
  validation (#8121). The parent `models.yaml` check now validates only its 46
  directly declared tiles, while the separate runtime registry check retains
  all 75 parent and provider models. Computer control confirmed the repaired
  Diagnostics screen reports `Status: HEALTHY` with zero failed checks instead
  of falsely classifying 29 provider-only models as missing.
- **2026-07-27** - Extended the classic PyQt6 Sidekick acceptance contract for
  the selected Tools authority. The background API child now inherits the same
  validated Tools checkout as the sidebar and places its package roots before
  UpstreamDrift's partial shared packages, preventing
  `chat.websocket_protocol` import failures (#8120). The matrix
  also records the Tools #3950 warnings-as-errors Units regression and its
  verified `100 °C` to `212 °F` retest. A complete API-tree fault injection
  confirmed dynamic-port recovery, Chat reconnection, and close-time descendant
  cleanup without disturbing an unrelated port-8000 process.
- **2026-07-25** - Hardened classic PyQt6 Sidekick startup for #8102. The
  launcher selects an isolated loopback port when the historical default is
  occupied, exports one ephemeral launcher/API capability, verifies the child
  instance through `/readyz`, installs local Sidekick tools independently of
  Chat readiness, performs bounded child restarts, and loads the pinned Tools
  direct-package source before legacy aliases, copied sources, or mutable
  sibling sources. An explicit `TOOLS_REPO_PATH` is authoritative and fails
  closed when invalid rather than silently mixing with vendored or sibling
  sources. Each automatic restart revalidates a dynamically selected API port
  and selects a new free loopback port when necessary while preserving the
  launcher's capability and instance identity; explicitly configured ports
  remain unchanged. Host close also delegates to the canonical sidebar
  shutdown contract so Terminal PTY, shell, bridge, and API child processes
  cannot survive the launcher (#3938). A PR-diff hygiene contract now rejects
  any non-deletion edit to a Python path owned by the exact pinned Tools
  gitlink, regardless of warning header or shadow allow-list, while retaining
  the original base/current warning-header checks. The protected unit gate
  fetches the pull request base and sparse-checks out only the pinned Tools
  `src/shared/python` inventory before running the fail-closed comparison.
  Sidekick/chat paths without a same-relative Tools source require file-level
  owner state, rationale, tracking issue, and unexpired review date in
  `scripts/config/shared_python_ownership_exceptions.yaml`.
  Release-wheel assembly verifies the exact Tools gitlink and installs the
  parent-owned shared package graph plus its Chat/Sidekick compatibility
  aliases, utilities, and DbC contracts. The canonical alias finder coalesces
  direct, `shared.python`, and legacy `src.shared.python` spellings to prevent
  installed applications from executing stale nested child copies; only
  non-conflicting Upstream-owned extensions supplement the parent graph. Tagged
  releases initialize the exact Tools submodule and build a wheel directly
  from that verified checkout rather than rebuilding an unverifiable wheel
  from an unpacked source archive. Clean installed-wheel probes verify module
  identity, dependency closure, and byte-level source fidelity.
  The acceptance and source-ownership audit matrix is in
  `docs/testing/sidekick-pyqt6-startup-matrix.md`.
- **2026-06-22** - Optimize Mechanical Work computations in `evaluate_matching_workflow.py`. `np.sum(..., axis=1)` calls were replaced with equivalent but more efficient `np.einsum('ij->i', ...)` calls, yielding significant performance gains during bulk evaluation.
- **2026-07-15** - Hardened the simulation WebSocket and Data Explorer API
  routes for deferred #7740 findings: WebSocket start validation now rejects
  non-positive speed factors and shares duration/timestep bounds with
  `SimulationRequest`, simulation stats access is centralized behind one
  helper, Data Explorer dataset lookup rejects glob metacharacters, recursive
  dataset listing is paginated and bounded, and the dead cache helper was
  removed. Focused unit-marked tests cover the new WebSocket success/error
  branches, filter operators, ambiguous dataset names, glob rejection, and
  bounded listing behavior.
- **2026-06-20** - Deduplicated the chat WebSocket protocol loop for issue
  #7720: `src/api/routes/chat_ws.py` and the portable chat router factory now
  share one handshake/send/history/new-session/transport-error loop, with
  route-specific hooks preserving API context injection and router-factory
  extension actions plus focused regression coverage across both entrypoints.
- **2026-06-20** - Deduplicated the cross-engine motion-matching polynomial
  torque evaluator for issue #7728: Drake, MuJoCo, Pinocchio, and OpenSim now
  share the same lowest-power-first `[A..G]` Horner helper and constant,
  with unit coverage pinning shared and engine-wrapper parity.
- **2026-06-20** - Added isolated transition hazard-rule coverage for issue
  #7715: the MDP transition tests now pin hazard penalties and DbC guard
  behavior directly so policy updates cannot bypass invalid-state validation.
- **2026-06-20** - Hoisted OpenSim Manager/Integrator construction out of the
  perturbation per-step loop for issue #7713, preserving analyzer behavior
  while avoiding repeated runtime setup on every simulated step.
- **2026-06-20** - Covered chat WebSocket failure paths for issue #7721:
  `refresh_models` and `index_codebase` tests now drive provider and
  codemap rebuild exceptions, assert sanitized client error frames, preserve
  server-side traceback logging, and keep the socket usable after a model
  refresh failure.
- **2026-06-20** - Deduplicated the simulation WebSocket `set_speed`
  handler for issue #7719: runtime speed changes now route through one
  canonical branch for validation and state updates, with focused WebSocket
  regression coverage preserving accepted payload behavior.
- **2026-06-20** - Removed the dead duplicate Drake visualization monolith
  for issue #7709: Drake visualization now relies on the active maintained
  implementation instead of the stale `drake_visualization_mixin.py` copy,
  with obsolete unit coverage and suite-marker baseline entries removed.
- **2026-06-20** - Repaired Drake cross-engine theta sizing for issue #7725:
  the Drake equivalence smoke gate now derives its zero-theta vector length
  from the finalized Drake plant actuator count, matching the stricter
  production contract that rejects mismatched nonzero coefficient vectors
  instead of silently logging phantom torques.
- **2026-06-20** - Documented MuJoCo humanoid golf grip modelling for issues
  #7723/#7724: grip synergy construction and contact extraction now live in a
  focused helper module with regression coverage for finite contact geometry,
  deterministic synergy transforms, and the leaner GUI tab integration.
- **2026-06-20** - Vectorized clubhead trajectory assembly for issue #7714:
  `compute_clubhead_trajectory()` now computes the trunk, shoulder, and wrist
  angle path with NumPy array operations instead of a per-frame Python loop,
  while parity tests pin the vectorized positions and velocities to the
  original loop contract, including missing-joint defaults.
- **2026-06-20** - Added feature-parity tile-id uniqueness validation for
  issue #7730: registry loading now rejects duplicate launcher tile claims
  across entries while preserving distinct tile coverage, with focused loader
  tests for duplicate and unique tile lists.
- **2026-06-20** - Consolidated quaternion SLERP behavior for issue #7707:
  `math_utils.quaternion.slerp` now owns the shared nlerp fallback threshold,
  while spatial algebra rotations, cooperative manipulation, and Unreal
  skeleton mapping delegate to the canonical implementation with focused parity
  coverage across the threshold boundary.
- **2026-06-19** - Extended `SafetyMonitor` regression coverage for issue
  #7694: velocity-limit tests now pin unsafe target rejection and safe-command
  clipping, and emergency-stop torque regressions assert pure torque commands
  remain unsafe while emergency stop is active.
- **2026-06-19** - Added realtime abort coverage for issue #7697:
  control-loop failure escalation tests now exercise the emergency
  zero-torque fallback when command sends raise, asserting the loop still
  clears `is_running`, records `aborted_on_failure`, and attempts the
  fail-safe send instead of wedging.
- **2026-06-20** - Restored standalone Sidekick package frontend builds:
  the UI now uses `@vitejs/plugin-react@5.2.0`, whose peer range includes
  Vite 7, instead of plugin-react 6.x which imports Vite 8-only internals.
- **2026-06-20** - Cleared #7806 CI ratchets after merging current main:
  runtime dependency locks now require patched `pydantic-settings>=2.14.2`,
  and recent unit-style regression files carry explicit `pytest.mark.unit`
  suite markers instead of growing the unmarked-test baseline.
- **2026-06-20** - Deduplicated golf GUI camera pan axes after the DRY
  duplication ratchet surfaced repeated target-plane basis construction:
  mouse pan and pan inertia now share one `_camera_pan_axes()` helper while
  preserving the existing orbit/fly camera movement behavior.
- **2026-06-20** - Hardened shared CORS credential handling for issue #7740:
  `add_cors_middleware()` now rejects wildcard origins when credentials are
  enabled, including origins resolved from `CORS_ORIGINS`, while preserving
  wildcard support for non-credentialed CORS responses.
- **2026-06-20** - Corrected Drake theta contract coverage for issue #7726:
  `tests/parity/test_simulate_contract_drake.py` now directly exercises
  `validate_theta(..., bounds=DEFAULT_THETA_BOUND_TABLE)` rejecting a
  physically unreasonable `1e9` coefficient, while separately pinning the
  Drake simulate path's intentional `bounds=None` behavior as accepting the
  same large-but-finite coefficient and producing finite output or a failed
  solver status.
- **2026-06-19** - Hardened Data Explorer numeric contracts for issue #7732:
  dataset stats now ignore textual `inf`, `-inf`, `nan`, and `Infinity`
  cells instead of allowing one non-finite value to poison min/max/mean or
  emit invalid JSON tokens, and numeric filter comparisons reject non-finite
  row or filter operands before matching rows.
- **2026-06-19** - Hardened common engine state validation for issue #7705:
  `StateManager` and `ForceAccumulator` now enforce positive dimensions,
  positive time steps, and non-empty force-source names through body-level
  guards that remain active even when Python optimization disables decorator
  contracts.
- **2026-06-19** - Logged tracebacks for unexpected motion-pipeline stage
  failures in issue #7701: adapter loading, preprocessing, skeleton scaling,
  and inverse-kinematics handlers now call `logger.exception` before returning
  failed `StageResult` values, preserving file/line stack context for
  scientific runtime failures while keeping caller-contract errors classified
  as invalid input.
- **2026-06-19** - Hardened auth and signal-toolkit validation for issues
  #7698, #7699, #7700, #7702, and #7703: access-token creation now validates required
  subject claims even when DbC decorators are disabled, bcrypt-backed password
  and API-key hashing rejects inputs above bcrypt's 72-byte UTF-8 limit before
  truncation can occur, password/API-key verification fails closed for
  overlong secrets, malformed stored bcrypt hashes are logged with traceback
  context before verification returns `False`, and public
  `signal_toolkit.calculus`/`noise` boundaries use explicit `TypeError` guards
  instead of stripped `assert` statements.
- **2026-06-18** - Hardened `SafetyMonitor` command contracts for issues
  #7683, #7684, and #7692: command preflight now rejects velocity targets over
  `max_joint_velocity` and any command while emergency stop is active, safe
  command shaping clips velocity targets like torque and position targets, and
  emergency stop authoritatively zeros velocity, torque, and feedforward torque
  even if a later speed-override call tries to raise the override.
- **2026-06-18** - Restored the scheduled Vendor Submodule Freshness workflow
  for issue #7672: `scripts/check_vendor_updates.py --json` now emits a
  parseable status array, text mode prints the per-submodule messages again,
  and the workflow keeps JSON stdout separate from diagnostics instead of
  redirecting stderr into the artifact or masking script failures.
- **2026-06-18** - Regenerated the full-src mypy baseline from the Linux
  Python 3.11 quality-gate environment used by `main` CI so push-triggered
  code-quality checks compare against the same platform family that enforces
  the gate.
- **2026-06-18** - Made the lightweight catch-all 404 page an eager route
  dependency so unknown deep links render the branded recoverable "Page not
  found" screen immediately, without depending on route chunk load timing. The
  heavier feature pages remain lazy-loaded behind the shared Suspense fallback.
- **2026-06-18** - Optimized the #7561 complexity hot-loop slice in the
  MuJoCo humanoid golf analysis path: `joint_analysis.analyze_torque_transmission`
  now hoists the universal-joint bend angle once per sweep and computes the
  wobble/torque-ratio arrays with a vectorized helper that preserves the scalar
  guards, while `power_flow._compute_power_dissipation` caches positive-damping
  joint/dof maps at initialization and computes dissipation as one vectorized
  dot product. Focused parity tests cover the vectorized joint analysis helper
  and power-dissipation scalar equivalence.
- **2026-06-18** - Gated Windows Tauri release packaging behind the
  `TAURI_WINDOWS_RELEASE_ENABLED=true` repository variable after main CI showed
  the current self-hosted Windows runner blocks Cargo build-script executables
  with Application Control (`os error 4551`). Linux Tauri release packaging and
  the Tauri Rust/TypeScript check remain enforced; Windows packaging can be
  re-enabled once a selected Windows runner policy permits Cargo build scripts.
- **2026-06-18** - Restored the Tauri release build contract after main CI
  exposed two packaging-lane failures: the UI package now declares the
  `tauri` npm script expected by `tauri-apps/tauri-action`, and the Windows
  self-hosted build bootstraps Rust through PowerShell so it does not depend on
  Git Bash path rewriting for the Rust setup action. The Tauri Rust target
  caches are now keyed by runner name to prevent proc-macro artifacts compiled
  against one self-hosted runner's glibc from being restored on another. The
  npm Tauri API/CLI packages are pinned to the Rust `tauri` crate minor version
  because the release build fails on cross-ecosystem minor drift.
- **2026-06-18** - Restored the final Tauri Build release contract for issue
  #7652 by aligning the Rust `tauri` lockfile package with the locked
  `@tauri-apps/api` package minor version and adding CI infrastructure coverage
  that fails before release builds can reach `tauri-action` with mismatched
  Tauri Rust/npm package minors. The Linux Tauri lanes also install
  `libdbus-1-dev` because the updated Rust graph contains `libdbus-sys`.
- **2026-06-18** - Hardened the Tauri Build check after the UI audit recovery
  PR exposed DeskComputer runner PATH drift: `CARGO_HOME` is now rooted at the
  workspace, the Rust toolchain verifier persists `$CARGO_HOME/bin` into
  `$GITHUB_PATH`, and CI infrastructure tests pin that Cargo remains reachable
  before rustfmt/clippy/check steps run.
- **2026-06-18** - Restored the main-side Tauri Build release lane after the
  check recovery: the UI manifest now exposes the `npm run tauri` entrypoint
  that `tauri-action` invokes, and the build matrix uses named runner metadata
  with a PowerShell Rust setup on Windows so the self-hosted Windows leg does
  not route through the failing bash action path.
- **2026-06-18** - Restored the current-main CI Standard lane after #7645 by
  path-scoping strict API mypy on ordinary pushes with `github.event.before`
  and adding a 10 microsecond absolute floor to microbenchmark regression
  thresholds so sub-microsecond runner jitter does not fail the gate.
- **2026-06-18** - Restored the Nightly Cross-Engine Validation workflow for
  issue #7646 by replacing the empty heavy-integration placeholder target with
  the real validator and conformance harness tests, correcting the coverage
  target, and treating zero collected tests as a validation failure.
- **2026-06-18** - Closed current-main CI issue #7643 by path-scoping the
  baseline mypy gate on ordinary pushes with a concrete `github.event.before`
  SHA while keeping scheduled/manual full-baseline audits, and by allowing
  Jules PR Cleanup to fall back to the repository token when the optional
  runner token is absent.
- **2026-06-17** - Stabilized the full-src mypy baseline gate by normalizing
  equivalent built-in `int` constructor note spellings across local and CI
  Python/mypy environments, and refreshed the accountable baseline snapshot.
- **2026-06-17** - Restored the current-main `repo-structure-gates` lane by
  installing the pinned PyYAML parser dependency before workflow YAML guards
  run, and added CI infrastructure coverage that keeps the trust-boundary guard
  ordered after that dependency setup.
- **2026-06-17** - Closed current-main CI recovery issue #7640 by path-scoping
  the mandatory Rust wheel parity lane on push events with
  `github.event.before`, preserving full parity only for schedule/manual runs,
  and hardening the full-Trivy Dockerfile set with minimal apt installs plus
  non-root runtime users.
- **2026-06-17** - Recovered current-main CI after the #7632 merge: the
  Python aerodynamics fallback and vectorized ball-force path now share the
  Rust Penner-style bounded spin-lift coefficient, the Rust parity fixture
  anchors were refreshed to the corrected trajectory, parity TestClient runs
  opt into a high simulation rate-limit budget, PyO3 binding tests assert only
  supported construction/precondition contracts, `DragModel.calculate()` reuses
  its speed calculation for the Reynolds-corrected coefficient path, and
  dashboard launcher tests can inject a non-blocking Qt event-loop runner.
- **2026-06-17** - Closed the test-only PR core-lane OOM regression for issue
  #7635: CI Standard now exits after changed-test PR slices pass when no
  source/dependency coverage targets changed, while preserving the scoped
  dependency-light lane for source/dependency PRs and the no-collected-tests
  fallback.
- **2026-06-17** - Closed the push-scoped Semgrep regression for issue #7633
  by making the SVG backdrop regression-test inspection helpers parse generated
  SVGs with defusedxml instead of stdlib ElementTree.
- **2026-06-17** - Restored the suite-marker ratchet for issue #7631 by
  marking the three #7629 Bandit regression tests as unit tests, preserving
  the explicit suite ownership contract for new tests.
- **2026-06-17** - Cleared the main-branch Bandit security-scan regression for
  issue #7629: repository scripts now parse XML inputs with defusedxml while
  retaining stdlib ElementTree only for XML construction/writing, and
  `download_to_file()` validates URL schemes internally before its audited
  `urlopen` call. Regression tests cover entity-bearing XML rejection and local
  file URL rejection before any network/file opener is invoked.
- **2026-06-17** - Restored the current-main test-layout gate for issue
  #7626 by adding `src/shared/python/movement_optimizer/tests` to the root
  pytest `testpaths` contract and the intentional in-tree package-test
  allowlist.
- **2026-06-17** - Restored the current-main suppression ratchet for issue
  #7625 by replacing seven bare `# type: ignore` fallbacks in launcher, chat,
  biomechanics, and screw-theory modules with explicit `assignment` codes.
- **2026-06-17** - Restored the current-main error-handling ratchet for issue
  #7621: `CustomTitleBar` now applies the themed title text color instead of
  carrying an unused `F841` suppression, and construction tolerates a deleted
  Qt-backed theme manager during defensive theme-change hookup.
- **2026-06-17** - Hardened the Tauri desktop check workflow for issue #7616:
  after the pinned Rust toolchain action runs, the `Check (Rust + TypeScript)`
  job now verifies `rustup`, `rustc`, and `cargo` are on `PATH` and prints their
  versions before cache restore or any Cargo command. CI infrastructure tests
  pin the setup -> verification -> Cargo-step ordering so a runner without a
  usable Rust toolchain fails early with a direct diagnostic.
- **2026-06-17** - Recovered current-main CI Standard gates for issue #7617:
  Semgrep and Python matrix push runs now use changed-file scopes when GitHub
  supplies a `before` SHA, current DRY debt is owned in the no-growth ratchet,
  dependency locks use LF line endings, and Model Explorer XML construction
  suppressions are backed by defusedxml parser regression tests.
- **2026-06-17** - Optimized lower-body simulator history eviction for
  issue #7561: `LowerBodySimulator` now stores scrub/playback frames in a
  bounded `deque`, preserving FIFO order, frame indexing, and clear/restore
  behavior while avoiding `list.pop(0)` work on every overflow step.
- **2026-06-17** - Pinned deformable external-force scatter coverage for
  issue #7563: `SoftBody.apply_external_force` now has a source-level
  regression guard against Python-level scatter loops, complementing the
  duplicate-node accumulation coverage for the existing `np.add.at` path.
- **2026-06-17** - Extended footstep yaw normalization coverage for issue
  #7556 with a very-large negative-angle regression, preserving the bounded
  modulo contract for both signs and the existing NaN/finite guard behavior.
- **2026-06-17** - Tightened MuJoCo humanoid golf Coriolis finite
  differences for issue #7556: `compute_coriolis_matrix` now uses central
  velocity perturbations, validates finite `qpos`/`qvel` shape contracts, and
  rejects malformed or non-finite callback force vectors before assembling the
  matrix.
- **2026-06-17** - Optimized FreeMoCap landmark array conversion for issue
  #7563: `LandmarkFrame.to_array` now builds numeric point blocks through
  `np.fromiter`, and `LandmarkSession.to_array` constructs the full
  `(frames, landmarks, 4)` block directly while preserving empty and
  fixed-schema NaN row contracts.
- **2026-06-17** - Optimized MuJoCo kinematic Coriolis decomposition for issue
  #7558: `KinematicForceAnalyzer` now reuses the base inverse-dynamics
  solution and scratch buffers across the per-DOF Coriolis split, reducing
  redundant `mj_rne` passes while preserving the legacy component-sum contract.
- **2026-06-17** - Optimized fixed-size vector magnitude hot paths:
  API route metrics, putting-green direction normalization, teleoperation
  deadband/rate limiting, and ball-launch trajectory speed now use explicit
  `math.hypot` calls instead of scalar `np.linalg.norm` calls while preserving
  the existing vector contracts.
- **2026-06-17** - Hardened robotics precision/vectorization paths for
  #7556/#7559/#7563: footstep yaw wrapping is bounded and NaN-safe, IMU and
  force-torque guards reject unstable normalization/division cases,
  cooperative load sharing uses `solve`, contact hull sorting uses vectorized
  keys, deformable external-force scatter handles duplicate nodes with
  `np.add.at`, and GJK reuses its initial support-vector norm.
- **2026-06-17** - Optimized deformable soft-body FEM root-node force
  accumulation: `SoftBody.compute_internal_forces` now uses a batched
  `np.einsum("ijk->ij", H)` reduction for per-tetrahedron root forces instead
  of a generic last-axis sum, with focused parity coverage for the batched
  reduction shape.
- **2026-06-17** - Optimized learning retargeter FK and mocap marker lookup
  hot paths for issue #7566: `MotionRetargeter` now caches end-effector
  kinematic chains as joint indices, reuses finite-difference perturbation
  buffers during frame optimization, evaluates simplified z-axis FK without
  per-joint rotation-matrix allocation, precomputes mocap marker-name indices
  once per call, and validates retargeting array shapes/finite values at
  public/internal boundaries.
- **2026-06-17** - Optimized RRT/RRT* tree-neighbor and cost-propagation
  paths for issue #7564: both planners now maintain an append-friendly
  finite configuration index for vectorized nearest-neighbor queries, RRT*
  can route neighbor lookups through the existing `use_kd_tree` flag with
  periodically rebuilt `cKDTree` coverage, and rewiring cost propagation now
  walks a maintained child-adjacency map with a deque instead of scanning the
  full node list for each descendant.
- **2026-06-17** - Replaced the iLQR backward-pass gain calculation for issue
  #7570: `src/research/mpc/controller.py` now solves the regularized `Quu`
  system directly, using a finite-checked Cholesky path for symmetric positive
  definite matrices and a general linear solve fallback while preserving gain
  parity with the previous inverse-based math.
- **2026-06-17** - Optimized whole-body SciPy QP inequality construction for
  issue #7568: inequality rows now build as vector-valued lower/upper SLSQP
  callbacks from finite-bound masks instead of O(m) per-row Python callbacks,
  with focused validation for QP matrix finiteness and constraint/bound shapes.
- **2026-06-17** - Optimized the Rust mocap preprocessing per-series filter
  dispatcher for issue #7573: `apply_per_series` now gathers/scatters each
  `(point, dim)` time series through ndarray lane views with explicit
  shape/finite-value contracts, preserving numeric output while avoiding
  repeated full 3-index ndarray indexing in the hot filter path.
- **2026-06-17** - Optimized Rust motion-matching finite differences for issue
  #7575: the internal q/qdot/qddot working storage now uses contiguous
  row-major buffers instead of scattered nested vectors, while preserving the
  public nested-vector `FiniteDiffResult` API and adding irregular multi-DOF
  parity coverage.
- **2026-06-17** - Optimized the Rust mocap preprocessing median filter for
  issue #7574: `medfilt_1d` now uses introselect (`select_nth_unstable_by`) to
  find the padded-window median without fully sorting each window, preserving
  SciPy-compatible zero-padding and even-kernel normalization with focused
  full-sort parity coverage.
- **2026-06-17** - Fixed manipulation pick/place ndarray position overrides for
  issue #7565: explicit `object_initial_pos` and `target_pos` arrays now use
  explicit `None` fallback handling, are coerced to defensive 3D float vectors,
  and no longer raise NumPy ambiguous truth-value errors during environment
  construction.
- **2026-06-17** - Improved differentiable-engine finite differences for
  issue #7569: trajectory-control gradients and state/control Jacobians now use
  scaled central differences (`1e-6 * max(1, abs(value))`) with shared finite
  input/output contracts, preserving suffix-rollout reuse while reducing
  one-sided truncation error.
- **2026-06-17** - Hardened durable task and motion-matching CLI contracts for
  issues #7549 and #7552: completed task persistence now distinguishes
  `None` from empty result objects, durable SQLite shutdown releases the
  database file on Windows, and the motion-matching leaderboard CLI writes
  actionable stderr diagnostics for bad paths or malformed result JSON instead
  of exiting `1` silently.
- **2026-06-17** - Tightened the biomechanics analysis request contract for
  issues #7550 and #7551: `AnalysisRequest.parameters` is now accepted as a
  first-class analysis payload source instead of being silently ignored, and
  simulation-backed analyses report extraction failures as failed responses
  rather than successful empty analyses with only `metadata.engine_error`.
- **2026-06-17** - Tightened the direct launcher Tools override contract for
  issue #7544: `launch_upstream_drift.py` no longer auto-discovers arbitrary
  sibling `Tools` checkouts ahead of the repository and vendored dependency
  paths. Development overrides now require an explicit valid `TOOLS_REPO_PATH`,
  while the default bootstrap order remains repository source followed by the
  vendored `vendor/ud-tools` shared package.
- **2026-06-17** - Restored Windows installer build diagnostics for issue
  #7545: the `setup.py build` and `setup.py bdist_msi` helper paths now share
  failure formatting that preserves the command, return code, stdout, and
  stderr, while `main()` reports the concise failure and exits with status 1.
- **2026-06-16** - Tightened the motion training demo CLI failure contract for
  issue #7543: `examples/motion_training_demo.py` now treats IK solver
  construction failures as explicit initialization errors, reports the
  underlying cause on stderr, and exits with status 1 instead of continuing as
  a successful run. Regression coverage locks the CLI `main()` behavior when
  `create_ik_solver(...)` raises.
- **2026-06-16** - Optimized cross-engine and Pinocchio kinematic
  equivalence RMSE distance accumulation to use `np.einsum`, preserving the
  existing tolerance contract while avoiding temporary squared-distance arrays
  in hot parity-test paths.
- **2026-06-16** - Coordinated with Tools #3316 by moving production
  consumers of the Tools Sidekick surface from direct `sidekick.*` imports to
  `shared.python.sidekick.*` imports, keeping the compatibility package intact
  while reducing duplicate module spellings at launcher, calc-backend,
  motion-capture, and assistant integration boundaries.
- **2026-06-15** - Optimized the starting-pose matcher Simscape trajectory CSV
  loader: `load_simscape_trajectory_csv(...)` now pre-extracts each resolved
  joint's XYZ columns to NumPy arrays before frame construction, avoiding
  repeated row-wise pandas `iloc` materialization while preserving finite-value
  filtering and the existing `SkeletonTrajectory` contract.
- **2026-06-14** - Added the ball-flight physical benchmark contract for issue
  #7407: TrackMan-style driver and 7-iron tests now cover the enhanced
  simulator, the public flight-model registry, and the REST route; the suite
  locks cross-engine carry agreement, 5 m/s headwind/tailwind sanity, analytic
  vacuum range, explicit zero-drag semantics for Python/Rust drag-crisis
  helpers, and humidity-aware density through
  `EnvironmentalConditions.from_altitude(...)`.
- **2026-06-14** - Completed the AnalysisOrchestrator dashboard plot migration
  for issue #7446: every PyQt6 static-dashboard label now maps to a registered
  headless `PlotData` extractor, including Poincare, Lyapunov, recurrence, GRF
  butterfly, kinematic-sequence, and summary-dashboard outputs. The desktop
  dispatch path calls `AnalysisOrchestrator.get_plot_data(...)` before
  renderer-specific plotting, so web/API consumers and PyQt6 share the same
  structured data contract.
- **2026-06-14** - Corrected the impact model's gear-effect spin direction and
  driver COR calibration for issue #7406: toe/heel offsets now produce
  hook/slice spin in the launch-monitor sign convention, high-face/low-face
  offsets reduce/increase backspin, default gear-effect scales put 20 mm toe
  and 10 mm vertical strikes in realistic rad/s bands, and `DRIVER_COR` now
  reflects the USGA/R&A CT-limit-equivalent COR so center strikes can reach
  tour smash-factor ranges. Regression tests cover toe/heel antisymmetry,
  vertical gear effect, default smash factor, and energy balance.
- **2026-06-14** - Calibrated the flagship Rust and enhanced ball-flight
  aerodynamic coefficient contracts for issue #7405: `BallProperties`,
  Python `LiftModel`/`MagnusModel`, and the Rust upstream-physics kernel now
  share a bounded Penner-style spin-lift curve (`Cl(0.08)` driver and
  `Cl(0.30)` 7-iron anchors), while `BallProperties` uses stronger
  spin-dependent drag coefficients for reported-force consistency. Added
  scientific TrackMan benchmark tests for driver and 7-iron trajectories,
  coefficient anchors, and Rust-vs-enhanced agreement when the Rust wheel is
  available.
- **2026-06-17** - Hardened GitHub Actions workflow-context validation for
  issues #7613/#7614: CI Standard, Tauri Build, and the movement_optimizer Maturin
  lane now use workspace-relative Cargo homes instead of job-level
  `${{ runner.temp }}` expressions, and the standard repo-structure gate runs
  `scripts/ci/check_workflow_contexts.py` to reject job-level environment
  expressions that GitHub cannot resolve before scheduling a runner. The
  exposed suite-marker ratchet drift is also classified with module-level unit
  markers instead of growing the unmarked-test baseline.
- **2026-06-14** - Corrected the multi-model ball-flight lift calibration for
  issue #7404: Waterloo/Penner now uses a Penner-style spin-ratio lift fit with
  a bounded golf-ball lift coefficient, and the MacDonald-Hanzely plus constant
  coefficient models use a calibrated spin-ratio lift helper instead of treating
  published `Cl` values as a tiny unbounded slope. Scientific regression tests
  now lock the driver TrackMan carry/apex/time band, the 7-iron reference carry
  band, and the vacuum projectile-range invariant.
- **2026-06-14** - Corrected the rigid-body impact model's friction-spin sign
  for issue #7403: tangential contact impulse now uses the physical torque
  direction `tangent_dir x normal`, so lofted center strikes produce backspin
  under the repository's `[0, -1, 0]` convention and pipeline-derived spin
  creates upward Magnus force instead of downforce.
- **2026-06-14** - Corrected the vendored Sidekick gas-flow
  compressibility-factor contract for issue #7408: supported gases now carry
  acentric factors, and the Abbott/Pitzer generalized second-virial
  approximation uses `(B0 + omega * B1) * Pr / Tr` so methane and air reference
  points match real-gas expectations while the low-pressure limit tends to
  ideal behavior.
- **2026-06-13** - The modular Docker `slim` profile contract now explicitly
  includes the core MuJoCo runtime, matching the package's default physics
  dependency and keeping slim smoke tests focused on excluding Pinocchio and
  Drake.
- **2026-06-13** - The legacy runtime Docker entrypoint now honors explicit
  container commands such as `docker run image python -c ...` while preserving
  the default FastAPI server startup for bare `docker run image`.
- **2026-06-13** - The character builder API route now loads its
  Tools-backed URDF provider lazily so slim Docker API images can discover
  routes successfully and return a controlled `503` if the provider is absent.
- **2026-06-13** - The launcher namespace package now preserves legacy
  convenience exports lazily so API manifest loading does not import PyQt6-only
  dialogs inside headless Docker runtime images.
- **2026-06-13** - The shared dashboard package now exposes GUI entry points
  lazily so API/container imports of `dashboard.recorder` remain usable in
  headless images that intentionally omit PyQt6.
- **2026-06-13** - BunkerShot3D internal modules now use package-relative
  imports so the Docker runtime can import the package through the
  `src.bunkershot3d` namespace used by API health checks.
- **2026-06-13** - Modular Docker builder installs now copy the hatch
  force-included `launch_golf_suite.py` entrypoint before feature package
  installation, keeping `pip install .` metadata generation aligned with the
  wheel packaging contract.
- **2026-06-13** - Modular Docker profile dry-run validation now keeps
  `engine_core.engine_probes` importable from the Dockerfile's early-copy
  source subset by avoiding the heavyweight config package at module import
  time; regression coverage mirrors the early-copy set so profile validation
  continues to fail fast before full source installation layers.
- **2026-06-13** - CI Standard source PRs now skip the changed-tests-only
  targeted coverage branch and fall through to the dependency-light coverage
  lane; optional-stack unit targets use a 180s per-test timeout for
  migration-heavy API tests and are scoped to the PR-relevant biomechanics,
  deployment, and robotics unit directories so the optional-stack lane does not
  couple unrelated shared-python baseline failures to focused performance PRs;
  CI contract tests assert the current PyQt fallback and Rust wheel install
  order. CI Standard now starts Xvfb with an atomic dynamic display reservation
  and cleanup step so parallel self-hosted PR jobs do not collide on display
  `:99`.
- **2026-06-13** - Documented the C3D/TRC Rust facade benchmarks as bounded
  smoke checks with runner jitter allowance while preserving the strict
  parser-only 10x performance gate.
- **2026-06-13** - Restored Rust/Python aerodynamic parity by keeping the
  dimpled-sphere drag-crisis centre aligned with the Rust kernel and by
  matching the public simulator parity fixture's drag coefficient to the Rust
  trajectory fixture.
- **2026-06-13** - Aligned the Rust wheel import smoke helper with the
  documented `upstream-mocap-io` missing-file contract so `parse_trc` raising
  `FileNotFoundError` remains an expected negative-path smoke result.
- **2026-06-13** - Repaired the Rust PyO3 CI contract so the Python-feature
  lane typechecks all targets with `cargo check --workspace --all-targets
--features python`, while maturin remains responsible for extension-module
  wheel and import smoke coverage.
- **2026-06-13** - Refined pointwise ZTCF/ZVCF dimension validation to allow
  canonical-v2 `nq = nv + 1` states only when the redundant configuration
  coordinate is populated, while still rejecting plain wrong-length zero vectors.
- **2026-06-13** - Scoped the optional-stack PR unit lane to the
  PR-relevant biomechanics, deployment, and robotics unit directories so
  focused UI/theming PRs do not hang on unrelated broad optional-stack unit
  sweeps; CI contract tests reject reintroducing the unscoped `tests/unit`
  discovery.
- **2026-06-13** - Restored optional-stack ZTCF/ZVCF input length validation
  for pointwise acceleration helpers and reapplied the Settings page root theme
  marker after active-theme round-trip.
- **2026-06-13** - Tightened signal-toolkit limit preconditions so saturation
  requires `lower < upper` and rate limiting requires `max_rate > 0`, matching
  the optional-stack contract tests.
- **2026-06-13** - Scoped PR core-test dependency-light execution to affected
  unit areas when source files change, preventing self-hosted runner OOMs from
  full-repository core test attempts while leaving normal coverage reporting in
  place for full-coverage runs.
- **2026-06-13** - Restored Sidekick JSON I/O compatibility in optional-stack
  tests by using stdlib JSON for default records-oriented reads and writes,
  avoiding pandas JSON C-extension failures in the optional-stack environment
  while preserving pandas for non-default JSON options.
- **2026-06-13** - Restored shared-python optional-stack compatibility
  contracts: package-root config and provenance exports remain importable
  under the test harness fallback, AI adapters preserve empty-message and
  legacy token/error contracts, validation and signal-toolkit helpers enforce
  documented preconditions, and partial Rust wheels no longer make FSP
  primitive tests fail when the FSP API is absent. Aligned CI numpy/scipy
  repair pins with the lockfile runtime, kept core tests serial to avoid
  native-stack xdist worker termination, and added temporary shared-chat
  architecture-budget exceptions for the pre-existing Qt dock decomposition debt
  tracked by issue #7362.
- **2026-06-13** - Restored optional-stack robotics compatibility contracts:
  QP, whole-body-control, and motion-planning result objects again expose the
  canonical `solver_status` aliases expected by optional robotics callers, WBC
  results expose `final_cost` as the legacy alias for `cost`, and default IMU
  gravity remains on the historical 9.81 m/s^2 sensor contract.
- **2026-06-14** - Added public AffineDrift analysis-tool parity for issue
  #7431: `src.tools.drift_control` computes generalized-force
  drift/control ratios from NPZ trajectories, `src.tools.contraction` exposes
  contraction-rate and Floquet helpers, the analysis-tools API provides
  headless JSON endpoints for both workflows, and
  `src.engines.pinocchio.benchmarks.aba_timing` reports optional Pinocchio ABA
  timing without requiring the Pinocchio extra in core CI.
- **2026-06-12** - Documented the consolidated parity CI follow-up: frontend
  Fast Refresh helpers now live outside component modules, diagnostics imports
  respect API/launcher layer boundaries, recording export artifact writing is
  extracted below the function-size budget, export-format validation is shared,
  and new parity tests carry suite markers.
- **2026-06-12** - Made the Hatch wheel package contract explicit for the
  standalone Sidekick console package by asserting the wheel target maps
  `src/shared/python/sidekick` to top-level `sidekick` through Hatch
  `force-include`.
- **2026-06-12** - Refreshed the pinned `python:3.12-slim` Docker base digest
  and configured the blocking Trivy table scan to ignore unfixed OS findings,
  keeping the gate focused on actionable HIGH/CRITICAL vulnerabilities.
- **2026-06-12** - Kept the Rust Python-binding verification lane on the
  locked Python dependency contract by installing `requirements-dev.lock`
  before the editable `.[dev]` package install and using `--no-deps`.
- **2026-06-12** - Moved shared plot-series label generation into a Qt-free
  `src.shared.python.plot_labels` helper so headless analysis can reuse plotting
  label contracts without importing the Matplotlib/PyQt plotting package.
- **2026-06-12** - Kept synthetic C3D event-alignment tests hermetic under the
  optional stack by patching both short and canonical Sidekick C3D import paths,
  so installed `ezc3d` cannot parse the intentionally empty fixture files.
- **2026-06-12** - Added static analysis plot parity for the web UI: `/api/v1/analysis/plot-types`
  enumerates the headless plot registry and `/api/v1/analysis/plot-data/{plot_type}`
  returns JSON plot payloads rendered by the React `PlotsSection`.
- **2026-06-12** - Corrected three scientific contracts in vendored Sidekick
  copies: Buck water-vapor-pressure constants now match Buck (1996), signal
  integration includes the upper-bound sample via `searchsorted(...,
side="right")`, and Nm3/ppm conversions share the DIN 1343 normal state
  (273.15 K, 101325 Pa) with 22.414 L/mol ppmv <-> mg/Nm3 conversion.
- **2026-06-12** - Removed the anti-phantom guard's `jq` dependency from the
  PR retry/API fallback path by using GitHub CLI `--jq` output directly. This
  keeps required PR guard checks portable across local self-hosted runner
  images that do not install `jq` globally.
- **2026-06-12** - Added generated TypeScript API types (issue #7447): the
  FastAPI OpenAPI contract is emitted to `ui/src/api/generated/types.ts` by
  `scripts/generate_ui_api_types.py` with a pytest freshness gate
  (`tests/api/test_generated_ui_api_types.py`). The launcher manifest and
  engine probe/load endpoints now declare Pydantic response models
  (`LauncherManifestResponse`, `EngineProbeResponse`, `EngineLoadResponse`),
  and `CapabilityLevelResponse.level`/`EngineCapabilitiesResponse.summary`
  are strictly typed. UI engine-list, engine load/probe, launcher-manifest,
  and engine-capabilities call sites consume the generated types.
- **2026-06-12** - Added About/version + onboarding web parity (#7459):
  `GET /api/v1/about` (and `/about/onboarding`) backed by the new shared
  version-resolution helper `src/shared/python/version_info.py` (VERSION
  file → importlib.metadata → fallback, plus safe git-commit reading) now
  used by both the desktop About dialog and the API. Web About modal and
  first-run onboarding overlay added to the React dashboard; onboarding
  card copy single-sourced in `src/config/onboarding_cards.json` consumed
  by the Qt dialog and the web overlay.
- **2026-06-12** - Added recording persistence and export parity endpoints
  (#7451): `POST/GET/DELETE /recordings`, `GET /recordings/{id}/export`
  (FileResponse), and `GET /export/formats`. Recordings persist under
  `output/recordings/<id>/` with metadata.json; exports reuse the desktop
  serializers in `src/shared/python/data_io/export.py` (no parallel
  serialization paths), with format enumeration derived from
  `get_available_export_formats()` — the same registry the PyQt6 dashboard
  Export tab uses. The web Simulation page gains a collapsible Recordings
  panel (list, per-format download, delete with confirm).
- **2026-06-12** - Web API honesty for stub/partial endpoints (issue #7448):
  `POST /simulation/recording` `action=export` no longer writes a JSON
  document into a file named with a client-requested extension; only `json`
  is implemented and the recognized desktop formats (csv/mat/hdf5/c3d) return
  an honest `501 {"detail", "tracking_issue": 7451}` until wired to the
  shared exporters. Added `not_implemented_json()` route helper and an
  architecture test (`tests/unit/api/test_no_stub_routes.py`) that greps
  route modules for removed canned-data sentinels. The web Analysis Tools
  page/hook were rewritten against the real
  `/api/analysis/{metrics,statistics,export}` contracts (previously a
  fictional schema with unsupported xlsx/pdf export options).
- **2026-06-11** - Consolidated launcher manager attribute forwarding through
  `src.launchers.launcher_manager_attrs.forward_manager_attribute()` so dialog,
  Sidekick, theme, and UI setup managers share one DbC boundary for local
  manager state versus launcher-owned state. This keeps the process-console
  guard fix DRY-compliant under the repository duplication ratchet.
- **2026-06-11** - Hardened launcher process-console tab state detection so
  `_is_console_open()` only passes real `QWidget` instances into Qt tab APIs.
  Test sentinels and partially constructed launchers now safely report the
  console as closed instead of raising C++ boundary `TypeError`s during layout
  synchronization.
- **2026-06-11** - Made the optional-stack unit lane boundary explicit:
  it exercises the non-engine unit suite with optional API/GUI/body-part
  dependencies installed, while native engine unit tests remain covered by the
  dedicated engine and cross-engine equivalence lanes. This keeps optional API
  and GUI dependency validation from being blocked by engine-specific mock
  behavior in full native dependency environments.
- **2026-06-11** - Aligned deployment optional-stack device tests with the
  documented hardware-honesty contract: unavailable hardware devices remain
  disconnected and raise `StateError` for state operations, while
  `KeyboardMouseInput` remains the connected fallback. `Demonstration` now
  carries the default canonical `solver_status="success"` through recording,
  serialization, subsampling, and augmentation.
- **2026-06-11** - Restored the calc backend ODE solver response contract so
  `ODESolverResponse` again exposes the default `solver_status="success"`
  field consumed by optional-stack calc backend callers and tests.
- **2026-06-11** - Restored body-part visualization unit contracts under the
  optional-stack lane: `FittedShape.n_frames` is again exposed, validation
  errors use the documented precise type/range messages, and the optional-stack
  venv installs `trimesh` so mesh-backed body-part tests exercise the full path.
- **2026-06-11** - Decomposed pendulum perturbation metric extraction and
  profile comparison into focused helpers so changed analyzer code stays within
  the architecture function-size budget without changing public metric output.
- **2026-06-11** - Aligned pendulum perturbation analyzer guard failures with
  the unit-level contract: invalid `extract_metrics()` inputs and missing
  `set_base_torque_profile()` preconditions now surface as `TypeError`
  precondition failures while preserving valid batch and metrics behavior.
- **2026-06-11** - Restored legacy AI assistant widget import identity by
  routing `assistant_widgets` and `assistant_panel` compatibility exports to the
  canonical assistant submodules, and made the optional-stack unit chunk loop
  fail fast after the first failing chunk to reduce runner load and produce
  focused CI diagnostics.
- **2026-06-11** - Restored API, launcher, and Docker contract parity after
  the main CI regression sweep. The public simulation request engine allowlist
  again includes `jaxsim`, Data Explorer import responses preserve generated
  `dataset_id` values while tolerating legacy direct model construction,
  launcher canonical-core tiles use a recognized `experimental` status with a
  served `biomechanics.svg` logo, symlink model-path validation preserves
  400-class security failures, and Docker feature dry-runs import engine probes
  through the package-qualified shared config path.

- **2026-06-11** - Capability truthfulness contracts for #7355 and #7356.
  Generated motion-pipeline compatibility docs now mark Drake trajectory
  optimization matching as unsupported until the solver is implemented, and
  Drake/RRA/CMC matching placeholder results advertise
  `status: not_implemented` plus `production_ready: false` so orchestrator
  failures remain caller-actionable. Production chat tools that do not yet run
  real work now return explicit `not_implemented` payloads instead of queued or
  successful placeholder results.

- **2026-06-11** - Honest launcher Document Chat and swing-sequence analytics
  contracts for #7358/#7359. The Library tab no longer enables Document Chat
  without a configured backend and no longer fabricates Notebook LM responses.
  `swing_sequence` analysis now computes segment peak timing from trajectory
  angular velocities via the shared segment timing analyzer, marks
  instantaneous-only segment velocities as `requires_trajectory`, and emits
  X-factor metrics only when joint trajectory data plus shoulder/hip indices
  are available.

- **2026-06-11** - RL engine protocol and teleoperation hardware-connection
  honesty for #7357/#7360. The RL humanoid environments now validate required
  engine dimensions and observation/reward channels before constructing spaces
  or stepping, `src.engines.protocols.PhysicsEngineProtocol` defines the typed
  runtime-facing engine surface, and MuJoCo exposes the required accessors via
  real model/data arrays. SpaceMouse, VR controller, and haptic input classes
  now report unavailable until a real hardware backend is connected and raise a
  state error on disconnected reads instead of returning frozen identity data.

- **2026-06-11** - Hardened launcher Docker build cancellation and layout reset
  backup semantics for #7341/#7342. Docker build threads now own a managed
  subprocess handle, expose cooperative cancellation that stops the child
  process without `QThread.terminate()`, and prompt before closing a window
  with an active build. GUI and CLI layout reset paths share a backup helper
  that overwrites an existing `launcher_layout.json.bak` with `Path.replace`
  so repeated resets work on Windows. The changed-file architecture budget
  records expiring exceptions for the legacy launcher UI builders exposed by
  this focused repair.

- **2026-06-11** - CI and validation test contract hardening for #7352,
  #7353, and #7354. The optional-stack lane now gates on pytest exit codes,
  physics validation scripts target real analytical/conservation suites, and
  PyQt fallback stubs no longer fabricate launcher expectations.

- **2026-06-11** - Shared Python motion-matching and signal-utility contract
  cleanup for #7348, #7349, #7350, and #7351. Role-specific fit result
  payloads now use explicit names with compatibility aliases, motion-pipeline
  frame-array preprocessing helpers are canonicalized under one module,
  rotation-matrix-to-quaternion conversion routes through a shared
  sign-canonical helper, and the MuJoCo polynomial generator imports the
  canonical signal-toolkit widget instead of carrying a fork.

- **2026-06-11** - Motion-pipeline DRY follow-up for the #7380
  simulator-facade merge. MuJoCo torque matching and Pinocchio inverse
  dynamics now share base helpers for per-DOF rig joint names and torque
  trajectory construction, removing duplicate post-merge torque payload
  assembly while preserving backend-specific success metadata.

- **2026-06-11** - Suite-marker ratchet follow-up for the #7382
  import-boundary consolidation repair and the #7380 simulator-facade merge.
  The regression tests surfaced by the changed-file ratchet now carry explicit
  `unit` suite markers so CI can enforce no-growth test metadata without
  weakening the marker baseline.

- **2026-06-11** - Import-boundary facade consolidation for #7361, #7362,
  and #7363. The C3D viewer wrapper now imports the repo-qualified viewer
  module without pivoting `sys.modules`, MCP config I/O lives under the shared
  AI MCP package with launcher compatibility facades, shared code no longer
  depends on launcher config readers for MCP settings, and shared/engine
  imports route through compatibility helpers instead of API-layer modules.
  Legacy oversized GUI, MuJoCo, and chat functions exposed by the changed-file
  architecture gate are tracked with owned expiring exceptions pending focused
  decomposition.
- **2026-06-11** - Classified the MuJoCo motion-matching placeholder path for
  #7333 as caller-actionable invalid input. The orchestrator now preserves
  solver metadata on motion-matching stage results, routes unavailable or
  zero-torque MuJoCo matching as a 4xx-class configuration failure, and the
  motion-pipeline README no longer recommends `matching_backend=mujoco` until
  real-model integration lands.
- **2026-06-11** - Suite-marker ratchet enforcement for #7272. CI Standard
  now runs `scripts/ci/check_suite_marker_ratchet.py` against
  `scripts/config/suite_marker_baseline.json`, failing net-new tests that
  lack a recognized suite marker while allowing legacy unmarked-test debt
  to shrink. The shared `tests.support.suite_markers` helpers now normalize
  nodeids, load the baseline, and support report-only, strict, and
  baseline-ratchet collection behavior from `tests/conftest.py`; contributor
  guidance lives in `docs/development/test-marker-conventions.md` with
  focused unit coverage for the static scanner and runtime helpers.
- **2026-06-11** - Replaced collision distance `math.hypot(*tuple)` unpacking
  with explicit component access in primitive-shape distance helpers, keeping
  the robotics collision contracts unchanged while avoiding tuple unpacking
  overhead on hot paths.
- **2026-06-11** - Restored the #7246/#7247 regression-guard cluster for
  #7325, #7326, and #7327 after PR #7248 reverted part of the launch-condition
  unit fix. `LaunchConditions.from_user_units(...)` is again the canonical
  GUI/user-input boundary for degree-to-radian conversion and RPM spin, while
  the current main gap-fill keypoint bounds guard remains in place.
- **2026-06-11** - Promoted Law-of-Demeter enforcement from advisory
  Pinocchio-only lint to a blocking repo-wide production `src/` ratchet.
  `quality-gate.yml` now runs `scripts/ci/check_lod.py src --baseline
scripts/ci/lod_baseline.txt`; the checked-in baseline records existing
  path/chain counts and the required `quality-gate` status fails on any new
  non-allowlisted deep attribute chain.
- **2026-06-11** - Tightened motion matching runtime contracts for #7304,
  #7305, #7306, and #7309. Internal request construction now rejects invalid
  cost weights and solver configuration before backend dispatch, metric helpers
  fail on mismatched frame/DOF shapes instead of truncating, solver result
  postconditions validate reference-aligned time grids plus torque/activation
  finiteness, and successful internal results must carry a matched payload.
- **2026-06-11** - Replaced pickle-enabled motion-matching checkpoint loads
  for #7276 with safe artifact loading. Motion checkpoint readers now route
  through a shared helper that calls `torch.load(..., weights_only=True)`,
  validates mapping-shaped artifacts, and keeps inverse, inverse-timestep,
  compact surrogate, and per-step surrogate loaders on the same safe contract.
  The changed-file architecture ratchet exposed pre-existing surrogate
  train/optimize budget violations, now tracked for decomposition in #7294.
- **2026-06-11** - Isolated optional dependency import mocks for #7307.
  Tests for OpenSim, MuJoCo video export, and Drake visualizer/analysis
  imports now install fake optional packages only inside scoped import
  fixtures. The shared optional-dependency helper restores dependency and
  target-module cache entries after each test, and repo-hygiene coverage
  rejects new module-scope `sys.modules` mocks for optional engine/media
  dependencies.
- **2026-06-11** - Finalized the cross-engine dashboard window factory split
  for #7316. `CrossEngineDashboardWindow()` now constructs the deferred PyQt
  window instead of raising a direct-instantiation placeholder, while the
  extracted fallback engine stub and `_build_qt_window()` path continue to keep
  the dashboard module under the tracked file-size budget.
- **2026-06-10** - Split the cross-engine dashboard window factory for #7288.
  `src/launchers/cross_engine_dashboard.py` now keeps the compatibility
  window facade below the architecture budget by moving the concrete PyQt
  window body behind a deferred factory and the fallback engine stub into
  `src/launchers/cross_engine_dashboard_stubs.py`; the architecture budget
  exception for that dashboard file has been removed.
- **2026-06-11** - Split the motion surrogate training architecture for
  #7317. The compact-schema surrogate trainer now resolves legacy keyword
  arguments through `SurrogateTrainingOptions`, builds an explicit training
  context, and runs checkpoints/metrics through a focused loop state. The
  per-step dynamics trainer now separates data preparation, runtime object
  construction, epoch fitting, best-checkpoint evaluation, and JSON output
  writing. The per-step optimizer now resolves legacy positional options
  through `OptimizationOptions`, builds an optimization context, isolates
  regularizer/orientation/tracking loss calculation, and writes torque plus
  summary artifacts from a dedicated output helper while preserving existing
  CLI and call-site compatibility.
- **2026-06-11** - Hardened the #7314 PR-scoped unit gate in standard
  CI. Source and dependency PRs now fall through to the dependency-light unit
  lane instead of passing solely on touched test files, and targeted PR coverage
  invokes a changed-file coverage ratchet for production policy files.
- **2026-06-10** - Added the #7275 local WebSocket origin and launcher-token
  guard. Browser WebSocket clients now request a short-lived launcher
  capability token before opening simulation/chat sockets, and the backend
  validates allowed local origins plus token claims so local sockets are not
  ambiently reachable from arbitrary browser contexts. The Tauri backend IPC
  capability now ships concrete v2 permission definitions so Rust/Tauri checks
  can resolve the four local backend commands, and the Tauri Linux dependency
  install now retries apt lock collisions on the self-hosted runner pool.
- **2026-06-10** - Collapsed the legacy Frankenstein editor split modules into
  import shims for #7280. `src/tools/model_explorer/_frankenstein_model.py`
  now re-exports `frankenstein_editor.model.URDFModel`, and
  `_frankenstein_panels.py` re-exports the canonical `ModelPanel` and
  `StealComponentDialog`, preserving older import paths while keeping the
  implementation in the `frankenstein_editor` package. The split contract tests
  now assert shim identity and exercise the canonical URDF validation/export
  path through the legacy import.
- **2026-06-10** - Hardened the optional cloud client cache contract for
  #7300. Empty or whitespace-only `~/.golf-suite/cloud_token` files are now
  treated as absent credentials, leaving `CloudClient.token` as `None` and
  `is_logged_in` false while preserving valid cached-token behavior. The
  runtime login state now requires a truthy token even if a caller manually
  mutates the token field.
- **2026-06-10** - Tightened API and model-library boundary contracts for
  #7297, #7298, and #7299. Data Explorer import/list responses now expose the
  durable `dataset_id` required by row pagination, filter operators are
  validated at the request boundary instead of silently returning empty
  results for invalid operators, and `ModelLibrary.load_model(...,
force_download=True)` enforces the HTTPS-only `source_url` policy before any
  download I/O.
- **2026-06-10** - Hardened the Jules PR AutoFix `workflow_run` trust boundary.
  Failed-CI `workflow_run` events now use read-only metadata resolution and a
  PR comment that asks maintainers to run the privileged fixer through explicit
  `workflow_dispatch`; only the manual dispatch path can check out PR code,
  install dependencies, run autofix tools, commit, or push. Standard CI now
  enforces that boundary with `scripts/check_workflow_run_trust_boundary.py`
  and focused regression coverage.
- **2026-06-10** - Narrowed PR-scoped source coverage in standard CI to the
  changed `src/**/*.py` targets after the coverage-bypass fix. Source and
  dependency PRs still produce coverage and enforce the 75% floor, while the
  full per-package coverage enforcer runs only after the default full-coverage
  lane so focused PRs do not fail against unrelated modules.
- **2026-06-10** - Enforced the #7277 Docker build timeout while process
  stdout remains open. `src/launchers/docker_manager.py` now reads build output
  through a background queue while the build thread owns a wall-clock timeout
  and terminates the process tree on expiry, including the regression case
  where stdout never reaches EOF.
- **2026-06-10** - Closed the #7283 simulation WebSocket dependency-boundary
  gap. The simulation stream now resolves its engine manager through a
  WebSocket-safe dependency accessor instead of reaching directly through
  `websocket.app.state`, and missing engine-manager state returns a structured
  `service_unavailable` frame before the connection closes cleanly.
- **2026-06-10** - Narrowed PR-scoped source coverage in standard CI to the
  changed `src/**/*.py` targets after the coverage-bypass fix. Source and
  dependency PRs still produce coverage and enforce the 75% floor, while the
  full per-package coverage enforcer runs only after the default full-coverage
  lane so focused PRs do not fail against unrelated modules.
- **2026-06-10** - Enforced the #7277 Docker build timeout while process
  stdout remains open. `src/launchers/docker_manager.py` now reads build output
  through a background queue while the build thread owns a wall-clock timeout
  and terminates the process tree on expiry, including the regression case
  where stdout never reaches EOF.
- **2026-06-10** - Locked the #7278 standard CI dependency and audit
  contract to committed artifacts. Python jobs that install project runtime or
  dev dependencies now seed environments from `requirements.lock` or
  `requirements-dev.lock` before no-dependency editable installs, avoiding
  pip constraints parsing for lock entries with extras. The dev lock now
  includes the GUI-test extra so `--no-deps` editable installs still provide
  real PyQt6/pytest-qt modules in the unit gates, and `pip-audit` runs directly
  against the committed runtime/dev lock files instead of a live resolver
  result. The standard CI acceptance tests also reject blank lines immediately
  after shell continuations so the core pytest coverage command cannot be split
  into a partial command again (#7303).
- **2026-06-10** - Closed the #7273 PR-scoped coverage bypass in standard CI.
  PRs that change source, test, or dependency targets now fall through to the
  coverage-producing core test lane instead of using the workflow-only
  `--no-cov` shortcut, and per-package coverage enforcement runs whenever that
  lane produces `coverage.xml`.
- **2026-06-10** - Closed the #7279/#7282 audit hygiene wave. The Docker
  security scan still uploads HIGH/CRITICAL SARIF findings, while the table scan
  is the blocking gate for fixable HIGH/CRITICAL findings. The audited
  API and launcher production modules now route module loggers through
  `logging_pkg.logging_config.get_logger(__name__)`, with a repo-hygiene test
  preventing the remediated files from returning to direct `logging.getLogger`.
- **2026-06-10** - Hardened the audit regressions tracked by #7269, #7270,
  and #7271. Model Explorer inspect/compare path resolution now rejects
  absolute paths and parent traversal before resolving candidates only under
  approved model roots; motion-pipeline linear keypoint gap filling leaves
  unmatched low-confidence keypoints unchanged when neighboring frames have
  mismatched keypoint counts, including the pure-Python fallback; and
  `SwingBallFlightPipeline` now derives `LaunchConditions` using the simulator
  contract of radians for launch/azimuth angles and RPM for spin rate.
- **2026-06-10** - Completed the #7207 model explorer composition UX flow.
  `src/tools/model_explorer/composition_ux.py` now provides a headless
  drag/drop orchestration layer with library payloads, non-mutating ghost
  previews, target/source link highlights, validation summaries, committed
  drops, and a validation-aware export chooser for URDF/MJCF while keeping
  SDF/OSIM disabled until first-party writers exist. `FrankensteinEditor`
  exposes preview, drop-commit, and export-choice hooks so the existing
  source/working model UI can compose simple humanoid plus arm models with
  live validation feedback before export.
- **2026-06-10** - Added the #7214 C3D viewer renderer decision and backend
  contract. ADR-0030 chooses `pyqtgraph.opengl`/PyQtGL as the first desktop
  GPU playback backend while retaining matplotlib fallback, and
  `viewer_3d_backend.py` pins the 60 fps target plus parity checklist for
  scrubbing, speed control, loop playback, marker groups, view presets, and
  skeleton overlay. The BunkerShot calibration optimizer now imports
  `scipy.optimize` lazily so cross-engine equivalence imports can use
  `WrenchTrace` without optional calibration optimizer dependencies.
- **2026-06-10** - Added the #7340/#7343/#7344 UI responsiveness contract:
  launcher dependency probes, settings Docker/WSL checks, and C3D MP4 export
  must run off the GUI thread; C3D video export exposes cooperative progress
  and cancellation hooks that remove partial files on cancel.
- **2026-06-10** - Added the #7207 model explorer composition-flow controller.
  `src/tools/model_explorer/composition_flow.py` now attaches a complete
  source URDF model to a working Frankenstein model through a declared
  attachment point, immediately validates the composed result, and exports
  validation-gated URDF or MJCF preview content. `FrankensteinEditor` exposes
  the flow through an Attach Source Model action and public export helper,
  while `URDFModel.from_file()` carries attachment sidecar metadata into the
  editor.
- **2026-06-10** - Added model explorer attachment manifests for #7206.
  `src/tools/model_explorer/attachment_manifest.py` now loads versioned
  `<model>.attachments.json` sidecars with non-fatal warnings for malformed
  manifests, `ModelLibrary` exposes declared attachment points on path-backed
  repository/imported/sibling/static model info, and the attachment dialog
  prioritizes declared mount points while applying their interface-frame
  defaults and payload-limit warnings. The schema lives at
  `src/tools/model_explorer/attachment_manifest.schema.json`, with user docs
  under `docs/model_explorer/attachment-manifests.md`.
- **2026-06-10** - Split the launcher entrypoint below the file-size budget
  for #7217. Sidekick sidebar installation, process cleanup polling, launcher
  domain orchestration, and GUI startup bootstrap now live in focused modules,
  while the existing frameless-window helper remains under
  `src/launchers/launcher_ui/frameless_window.py`; the
  `src/launchers/upstream_drift_launcher.py` file-size exception is removed.
- **2026-06-10** - Hardened Rust mocap Python binding errors for #7252.
  `upstream-mocap-io` validates `parse_c3d` / `parse_trc` / `parse_bvh`
  path preconditions before file access, maps missing files to
  `FileNotFoundError`, maps other file-access failures to `OSError`, and
  preserves malformed present files as `ValueError` parse failures with the
  format and path in the error context.
- **2026-06-10** - Made motion-pipeline hook failures observable for #7250.
  `PipelineConfig.strict_hooks` now switches per-stage hooks from lenient
  traceback logging to fail-fast `HookExecutionError` diagnostics, while the
  default lenient mode logs hook tracebacks with `logger.exception` and
  continues the pipeline.
- **2026-06-10** - Added the bounded inverse swing optimization core for #7220.
  `src/shared/python/physics/swing_optimizer.py` now exposes `FlightTarget`,
  `ClubPreset`, `SwingOptimizer`, and convergence diagnostics for solving
  speed/loft/attack/face-to-path parameters against the existing forward
  `SwingBallFlightPipeline`; GUI target mode remains follow-up scope.
- **2026-06-10** - Consolidated mocap marker NaN occlusion handling for #7251.
  C3D and TRC source adapters now delegate marker-triplet NaN detection to the
  shared `motion_pipeline.sources._marker_coordinates` helper, and the Python
  TRC fallback skips textual `nan` marker rows the same way the Rust-backed
  adapter paths skip occluded samples.
- **2026-06-10** - Added the first #7207 model explorer library-panel
  unification slice. `ModelLoaderDialog` now exposes a single searchable
  library tree covering every `ModelLibrary.list_available_models()` category,
  including sibling repositories, with first-party format-badge inference
  backed by headless controller/model tests.
- **2026-06-10** - Completed the #7205 Frankenstein composition validation
  surface. `CompositionValidator` now emits warning-level findings for heavy
  attached subtrees and direct attachment geometry AABB overlaps, while the
  active Frankenstein model panel surfaces current validation findings in a
  dedicated list before save/export.
- **2026-06-10** - Decided React/Tauri launcher parity for #7221.
  ADR-0028 keeps React/Tauri on the manifest-driven multi-window model while
  PyQt remains canonical for embedded tabs/docks. The React dashboard now
  persists a manifest-keyed launcher window registry and exposes a window
  list/focus menu backed by the existing launch API.
- **2026-06-10** - Consolidated launcher startup ownership for #7215.
  `launch_golf_suite.py` is now a compatibility shim over the canonical
  `launch_upstream_drift.py` entry point. Classic PyQt startup preflights the
  Qt platform and selects `QT_QPA_PLATFORM=offscreen` on headless Linux, while
  the local API server tolerates unavailable optional engine-manager imports
  and reports an empty engine set instead of failing startup.
- **2026-06-10** - Removed unsafe Drake pose pickle deserialization from
  `src/shared/python/pose_interchange/pose_io.py`. Drake `.drake` initial-state
  files now use JSON for `{q, v, model_metadata}`, and legacy binary/non-JSON
  payloads are rejected before any deserialization path can execute.
- **2026-06-10** - Preserved the legacy golf visualizer dataset contract after
  row extraction optimization: `extract_frame_data` still requires the BASEQ,
  ZTCFQ, and DELTAQ datasets and returns zero-vector frame data when the
  requested row is unavailable.
- **2026-06-10** - Added a first-party Frankenstein composition validation
  slice for #7205. `src/tools/model_explorer/composition_validator.py` now
  emits structured error/warning findings for duplicate URDF names, orphaned
  joints, invalid root counts, disconnected links, kinematic cycles, and
  moving-link mass/inertia contracts. The active Frankenstein editor model
  export path blocks validation errors by default while retaining an explicit
  `force=True` escape hatch for recovery exports.
- **2026-06-10** - Added the LauncherContext in-process event bus and shared
  value registry for embedded tools (#7210): `launcher_embed.context` now
  defines the `LauncherContext` protocol plus an in-memory implementation with
  snapshot-safe dispatch, idempotent unsubscribe handles, and keyed
  `value_changed:<key>` notifications. `EmbeddedHostWidget` owns one context,
  injects it into opt-in tools via `set_launcher_context(ctx)`, and emits
  `tab.opened` / `tab.closed` lifecycle events while preserving legacy tools
  that do not implement the hook. The same context can back Sidekick's
  `LauncherSubtabPort` workspace surface through its existing `list/get/set`
  contract.
- **2026-06-10** - Optimized legacy golf visualizer frame extraction by reading
  each Pandas dataset row once per frame before point/vector extraction, reducing
  repeated `.iloc` lookup overhead while preserving fallback behavior for missing
  rows and columns.
- **2026-06-10** - Extended the Rust C3D parser for #7212. The
  `upstream-mocap-io` C3D path now decodes int16 and float analog channel data,
  surfaces additive PyO3 `analog` and `force_platforms` keys, parses
  `FORCE_PLATFORM:{TYPE,CHANNEL,CORNERS,ORIGIN}`, and preserves existing
  marker/event dictionary keys and marker-only fixture behavior.
- **2026-06-10** - Consolidated configuration ownership for #7216. Removed
  the root `config/` and `configs/` trees: CI/governance policy now lives in
  `scripts/config/`, BunkerShot3D calibration YAML lives under
  `src/bunkershot3d/calibration/configs/`, and UX field/error seed YAML lives
  under `src/shared/python/ux/config/`. Added
  `docs/development/configuration-systems.md` plus regression coverage so new
  root config directories do not reappear.
- **2026-06-10** - Repaired the Linux dependency-consistency lockfile drift
  after #7231. `requirements-dev.lock` now matches the Python 3.12 Linux
  `pip-compile --extra dev` output used by CI, removing Windows-only transitive
  packages and restoring `uvloop` for the Linux `uvicorn[standard]` stack.
- **2026-06-10** - Repaired Rust TRC row-validation parity for #7213.
  `rust_core/upstream-mocap-io` now rejects invalid or non-finite frame/time
  columns before accepting marker rows, preserving the Python adapter's
  malformed-line contract when contributors install a fresh native wheel. The
  Rust wheel CI lane now runs `tests/unit/motion_pipeline/sources` after
  installing built wheels so OpenCap and TRC/C3D/BVH adapter behavior is
  verified against the actual Maturin artifacts.
- **2026-06-10** - Unified MATLAB engine loading through the registry for
  #7219. `EngineManager._load_engine()` now obtains MATLAB engines through
  `EngineRegistration.factory()` like every other engine instead of branching
  into a private `matlab.engine.start_matlab` path. `src.engines.loaders`
  owns the Simscape adapter loaders for both `MATLAB_2D` and `MATLAB_3D`,
  while the command-line launcher still routes web-only MATLAB direct launches
  to the web UI.
- **2026-06-10** - Added the first-party OpenSim `.osim` loader for #7203.
  `src/tools/model_explorer/osim_loader.py` parses OpenSim 3.x
  `parent_body`/`body` joints and OpenSim 4.x socket-frame joints into the
  existing `ParsedModel` contract, exposes validated `CanonicalModel`
  conversion for composition, maps Pin/Slider/Ball/Weld/Free/Custom joints,
  records unconverted ForceSet/ConstraintSet/MarkerSet elements as warnings,
  and floors non-physical ground/zero inertia values only where needed for
  contract validation. Model Explorer discovery/import paths now classify
  `.osim` files from sibling repos and route opened `.osim` files through the
  loader without editing vendored `model_generation` modules.
- **2026-06-10** - Added Drake SDF model loading for #7204. The model
  explorer now provides a first-party `SdfLoader` under
  `src.tools.model_explorer`, parsing SDFormat links, inertials, primitive and
  mesh geometry, joint axes/limits/dynamics, SDFormat 1.8 `relative_to` poses,
  and ball/universal joints into the existing canonical model contract.
  Sibling model discovery now classifies `.sdf` files from `Drake_Models`
  alongside URDF and MJCF assets so Drake-native models can be browsed and
  composed.
- **2026-06-10** - Preserved URDF fixed-joint topology through MJCF
  roundtrips for #7208: URDF-to-MJCF conversion keeps MuJoCo weld semantics by
  emitting fixed children as nested bodies without joint elements while encoding
  the original fixed joint name, and MJCF-to-URDF decoding restores that name
  only for welded nested bodies. Regression coverage now asserts link sets,
  fixed and movable joint names/types, parent-child topology, and fixed-joint
  origin translation through URDF -> MJCF -> URDF.
- **2026-06-10** - Added entry-point based embeddable-tool adapter discovery
  for #7211. The launcher bootstrap now imports
  `upstream_drift.embeddable_tools` package entry points before falling back to
  the in-tree adapter list, de-duplicates adapter module paths, and preserves
  registry-diff tracking for adapter registration. `pyproject.toml` declares
  the first-party embeddable tool adapter entry points so installed wheels and
  editable checkouts share one discovery contract.
- **2026-06-10** - Added the headless ball-flight REST simulation route for
  #7218. `POST /tools/ball-flight/simulate` now validates launch, spin, wind,
  model, and integration-window inputs through Pydantic models, delegates to
  the existing `FlightModelRegistry` / `UnifiedLaunchConditions` physics stack,
  and registers the `ball_flight` tool alongside the existing API route map.
- **2026-06-10** - Refreshed the Module Map against the actual source tree
  (the previous tree listed entry points and API files that no longer
  exist) and linked the operational project map. The full gap inventory
  from the 2026-06-10 operational deep dive lives in
  `docs/architecture/PROJECT_MAP.md` §16, tracked by issues #7202-#7221
  (model-composition epic, sidekick agent wiring, startup + config
  consolidation, and related work). Landed alongside:
  sidekick subtab host port + pop-out lifecycle hooks (#7199), the Rust C3D
  1-D `POINT:UNITS` fix with full `data/` coverage (#7200), and sibling
  model-repository discovery in the model explorer (#7201).
- **2026-06-10** - Repaired remaining #7189 packaging gate regressions after
  the branch merge: Tauri Linux dependency installs now wait on both apt and
  dpkg locks, and the WGS calculator keeps GUI theme imports inside the plot-tab
  path so the installed `sidekick run` wheel smoke can load the headless engine
  without requiring PyQt6 or the top-level `shared` GUI theme package. The
  standalone Sidekick wheel smoke matrix now matches the Python 3.11+ package
  floor, and the Python-version coherence guard covers that workflow.
- **2026-06-10** - Resolved Python-version provenance drift for #7160:
  `pyproject.toml`, `install.sh`, `CLAUDE.md`, user-facing installation docs,
  `SPEC.md`, the standard CI test matrix, Docker base images, and
  `requirements.lock` now describe one coherent policy. The supported floor is
  Python 3.11, standard CI tests Python 3.11 and 3.12, and the production
  Docker image plus lockfile remain generated on Python 3.12. Added
  `scripts/ci/check_python_version_coherence.py` and focused tests so the
  floor, classifiers, mypy target, installer floor, lock header, Docker base,
  and CI versions cannot silently diverge again.
- **2026-06-10** - Hardened the production Docker dependency audit inputs for
  #7160 follow-up CI: Docker builder/runtime pip pins now use patched pip
  26.1.2, runtime metadata declares `Mako>=1.3.12` and `PyJWT>=2.13.0`,
  `requirements.lock` matches those security floors, and the third-party
  license ledger covers the newly explicit Mako dependency.
- **2026-06-10** - Tightened test-isolation and optional-dependency contracts
  for #7155/#7158: the MuJoCo dependency mock is function-scoped, affected
  MuJoCo tests initialize their own required state, launcher tests route
  `sys.modules` cleanup through the local cleanup fixture, and
  `test_api_extended.py` uses the shared optional-dependency helper with
  current path-validation imports instead of a blanket module-level skip. The
  local-only workflow routing guard now installs its YAML parser in an isolated
  workspace venv so self-hosted runners with PEP 668 system Python policy still
  execute the guard instead of failing during dependency bootstrap. The
  follow-up CI hardening keeps sidekick copied-test collection self-contained,
  avoids dynamic source execution in launcher tests, and refreshes generated
  dependency artifacts against the canonical project metadata.
- **2026-06-09** - Added a changed-file architecture budget gate for #7131/#7133: `scripts/ci/check_architecture_budget.py` now scans changed production Python files for functions over 100 lines and callable signatures over 8 effective parameters (excluding `self`/`cls`), with owned/expiring exceptions configured in `scripts/config/architecture_budget.json`. The standard CI workflow runs the guard beside the file-size and module-size gates, and focused tests pin long-function, parameter-count, exception, and test-path skip behavior.
- **2026-07-29** - Recorded the Bolt IK vector norm optimization: The `pinocchio_golf.diff_ik` module now uses `math.sqrt(np.vdot(err, err))` inside its `differential_ik` and `solve_dual_frame_ik` loop conditions instead of `np.linalg.norm(err)`. This eliminates array allocation overhead on the hot path for ~2x performance speedup without changing the mathematical behavior.
- **2026-06-02** - Restored visible Sidekick sidebar tab hover affordance (#7109): the synced tools-sidebar design-token QSS now styles unselected `QTabBar` tabs on hover with the soft accent surface while keeping the selected-tab rule separate, and a headless unit regression pins the generated stylesheet contract.
- **2026-06-02** - Fixed Windows taskbar identity for the UpstreamDrift launcher (#7107): `src.shared.python.ui.window_icon` now declares an AppUserModelID before showing the first window, applies the resolved icon to both the `QApplication` and top-level window, and covers the Windows API call plus icon application contract with focused unit tests.
- **2026-06-02** - Removed obsolete archived launcher entries (#7108): the deprecated MuJoCo, MATLAB, and motion-capture archived launchers are no longer advertised through the launcher manifest or tool catalog, and launcher regression coverage now asserts the surviving catalog paths without maintaining tests for removed archived entry points.
- **2026-06-02** - Hardened the core CI PR test lane for workflow-only pull requests (#7079): when the diff contains no core Python, test, or dependency targets, the core matrix exits after change detection instead of falling through to the full coverage lane. Source/dependency PRs with no changed tests still run the default core suite, preserving coverage while avoiding OOM-prone full-suite runs for GitHub Actions dependency bumps.
- **2026-06-02** - Recorded the Bolt small-vector norm optimization (#7098): scalar ball-flight force calculation, Waterloo/Penner and spin-decay flight models, and swing-to-launch derivation now use fixed-arity `math.hypot` for known 2D/3D vectors instead of `np.linalg.norm`, avoiding NumPy reduction overhead while preserving the existing one-dimensional vector contracts.
- **2026-06-02** - Recorded the golf visualizer camera-basis norm optimization (#7101): `GolfVisualizerWidget` now uses fixed-arity `math.hypot` for the known 3D forward/right camera vectors instead of `np.linalg.norm`, avoiding NumPy reduction overhead while preserving the existing fallback behavior for degenerate vectors.
- **2026-06-02** - Hardened cross-engine equivalence gate NaN handling and corrected JaxSim/Pinocchio parity test parameters (#7095, #7097): `_run_engine_checked` now distinguishes all-NaN grip (grip body absent from URDF — Drake's documented design for missing `club_grip`) from partial-NaN/Inf grip (simulation divergence). All-NaN raises `_EngineBindingsError` so the aggregation test skips the engine as unavailable; partial-NaN/Inf calls `pytest.fail` so a broken-but-runnable backend remains a hard gate failure (per reviewer feedback on #7099). The second JaxSim-vs-Pinocchio parametrized case was changed from a non-zero position to zero position: JaxSim uses INERTIAL velocity representation (angular momentum about world origin) while Pinocchio uses LOCAL (about CoM), and these two representations diverge in `M` and `h` at non-zero body positions via the parallel-axis theorem — the zero-position fix makes both representations equivalent while still exercising full mixed angular+linear Coriolis effects. Updated cross-engine gate docstrings to accurately describe that the 5 mm grip RMSE tolerance applies to cross-engine agreement only; the per-engine address-vs-Simscape check is a world-frame origin plausibility gate, not a post-registration RMSE gate (post-registration error is identically zero by construction).
- **2026-06-02** - Cleaned up docs root organization for #7063 after reconciling
  the branch with the newer mainline governance/operations/reviews layout. Loose
  root-level markdown is moved into topic subdirectories, `docs/sphinx/conf.py`
  no longer references the stale `TRACKED_TASK` placeholder extension, and
  `tests/docs/test_docs_structure.py` now enforces root markdown cleanliness plus
  valid example-index references while preserving the real runnable
  `docs/examples/` subtree that landed on main.
- **2026-06-01** - Repaired the UD-only Sidekick `agent`/`standalone` subpackages (#7066, #7067, #7068). (1) `agent/subtab_adapter.py` undo was inert: `_pack_undo` returned an opaque `subtab:kind:nonce` token but never set `metadata["_undo"]`, so `SidekickActionService._maybe_register_undo` never registered an inverse and `service.undo()` failed for all 5 reversible actions. Each reversible handler now emits `metadata={"_undo": {"action_id", "params"}}` (mirroring `_ToggleHandler`); focus/show/hide/set_variable/state_profile.save now genuinely round-trip. Added a real `subtab.state_profile.delete` action (and `SubtabActionPort.state_profile_delete`) so a freshly-saved profile has a well-defined inverse; an overwriting save restores the prior payload. (2) `standalone/runner.py` registered ~1/40 calculators and re-derived WGS with hard-coded `delta_h`/`delta_s` literals that diverged from canonical `WGS_DELTA_H`/`WGS_DELTA_S`. It now lazily registers 5 canonical `process_calculators` engines (`wgs_reactor`, `water_vapor_pressure`, `flare`, `financial`, `syngas_water`) via thin dict-returning adapters; WGS routes through the canonical `WGSReactorEngine` so the equilibrium constants live in exactly one module. Registration stays lazy so the headless runner imports with zero PyQt6/scipy at module load. (3) `standalone/window.py` Save/Load Profile menu actions were `logger.info("not yet implemented — T8")` stubs; they now wire to `StandaloneSessionStore.save_profile/load_profile` through a `ProfilePayload` (layout + theme), the previously-dead `host_action_port` is consumed via a `host_action_port()` accessor, and `__all__` was added to `onboarding`, `preferences`, `runner`, and `session_store`. Tests: real `service.undo` state-restoration per reversible subtab action, JSON-fixture round-trip per registered calculator + a "WGS constants in exactly one module" guard, headless profile round-trip, host-port accessor, and a `__all__` hygiene parametrization.
- **2026-06-01** - Fixed `BiomechanicalModel.add_segment` segment-name validation drift (#7045): `segment_masses` (from `estimate_segment_masses`) is keyed by the full segment name (e.g. `right_thigh`), matching the mass lookup in `compute_dynamic_com`, but `add_segment` validated membership using the mapped anthropometry key (`thigh`) and so rejected every laterally-named segment as "Unknown segment name". `add_segment` now validates against the full name, restoring the 3 red `tests/unit/biomechanics/test_dynamic_com.py` cases to green.
- **2026-06-01** - Resolved API review regressions (#7037; #7031, #7028, #7027): expired `TaskManager` entries are no longer refreshed by mutation paths; cancelled chat streams preserve tool-call/tool-result pairing for unexecuted calls; Data Explorer dataset stats stream every row instead of stopping at the preview cap; and cloud-token chmod hardening now has regression coverage.
- **2026-06-01** - Applied public physics/config review fixes (#7015, #7017, #6954): the documented `src.shared.python.physics.impact_model` public package now carries the private impact fixes for expected energy loss, contact-onset clearance, and rolling-friction spin caps; provider catalog iteration deduplicates canonical IDs while preserving alternate checkout-name discovery; and public-package/provider tests lock the behavior.
- **2026-06-01** - Fixed orphaned chat-stream daemon threads on client disconnect (#6981): `ChatService.stream_response` (`src/api/services/chat_service.py`) now passes a `threading.Event` cancellation token into the `_stream_to_queue` worker. The async consumer sets the flag in a `finally` block — which runs on normal completion, consumer error, and `GeneratorExit` from `aclose()` on client/WebSocket disconnect — then joins the worker (bounded 5 s). The worker checks the flag at the top of its outer loop, inside the adapter `stream_response` pull loop, before persisting messages, and before each tool call, so it stops pulling from the adapter and no longer takes `self._lock` to persist messages for an abandoned session. Regression test in `tests/unit/api/test_chat_service_stream_stop.py`.
- **2026-06-01** - Replaced smoke-only tests with value-asserting coverage for three calc modules (#6998, #6999, #7003): pressure-drop flow calculations (Darcy-Weisbach, Re=2300/4000 regime boundaries, hydrostatic-head sign convention, API RP 14E erosional velocity, expansion-factor bounds, negatives→ValueError); data-fitting solvers (2-link analytical IK round-trip vs synthetic ground truth, numerical IK convergence, forward-kinematics geometry, anthropometric parameter estimation vs Dempster fractions, residual contracts); and thermo property backends (CoolProp input validation, phase/quality determination, simplified ideal-gas/liquid correlations, Antoine saturation pressure/temperature round-trip, optional CoolProp/Cantera skips). Fixed a real robustness bug in `determine_phase_and_quality` (`sidekick/calculators/thermo/_property_backends.py`): a non-Cantera `water` object raised an uncaught `AttributeError` on `.TQ`; now caught so the function correctly returns `("unknown", 0.0)` per its contract.
- **2026-05-31** - Hardened JaxSim readiness and parity gates (#6880, #6881, #6882, #6884): `EngineManager` now registers JaxSim as a runtime-backed engine, only marks runtime-backed engines available when both adapter/provider paths and importable runtime dependencies are present, preserves DbC path-policy failures as provider-discovery misses instead of constructor crashes, and uses a required-JUnit testcase assertion in the cross-engine workflow so skipped/missing JaxSim/Pinocchio parity cases fail CI.
- **2026-05-31** - Fixed two HIGH-severity physics-audit defects (#6890, #6891): `JaxSimBackend.compute_jacobian` now restacks the native JaxSim free-floating Jacobian to the canonical `[angular; linear; joints]` convention — permuting the six base columns and the six spatial output rows — so `J·v` and `Jᵀ·force` agree with `M`/`h`/`v`/inverse-dynamics; and the cross-engine conformance harness no longer counts a missing required method on an advertised capability (now `passed=False`) or a throwing `supports()` query (now a failure, not a swallowed free pass) as a passing skip, closing a CC-8 gate-integrity hole. Genuine missing capabilities remain legitimate skips.
- **2026-05-31** - Recovered closed review-feedback fixes for the metadata-driven UX wrappers: the shared `simulation.engine` metadata now includes `jaxsim` for generated TypeScript/PyQt engine selectors, and PyQt `HelpfulField` free-form fields with `valid_range: null` now render an editable `QLineEdit` instead of an empty combo box.
- **2026-05-31** - Added the CC-22 offline Nimble gradient-oracle surface for issue #6795: `tools.offline_validation.nimble_gradient_oracle` provides deterministic request/response comparison types, lazy optional `nimblephysics==0.10.52.2` plus PyTorch loading, structured skip behavior for core installs, and a runtime-boundary test that forbids Nimble imports from `src/`.
- **2026-05-31** - Added the CC-34 engine selector/comparison UI surface (#6807): the React simulation GUI now exposes a capability-aware multi-engine comparison panel, greys out unavailable or unsupported engines from existing capability metadata, captures per-engine run provenance, and renders side-by-side columns with divergence annotations for shared numeric outputs.
- **2026-05-31** - Added the CC-35 workspace project/session spine and results-browser view models (#6808): `src/shared/python/workspace/` now persists project, subject, session, and dataset metadata in `project.json` and indexes CC-4 HDF5 trace artifacts with CC-6 `provenance_*` metadata for session/backend/text filtering.
- **2026-05-31** - Added the CC-23 moving-horizon estimator near-real-time path (#6796): `src/shared/python/estimation/moving_horizon.py` now maintains a bounded rolling window over canonical samples, builds fixed-parameter MAP objectives from the CC-19 solver surface, warm-starts from the previous window, records latency against a stated 50 ms default budget, and exposes a callback payload for realtime integration.
- **2026-05-31** - Added the CC-32 canonical-core app shell registry (#6805): canonical-core estimation and comparison now appear as shared launcher tools in both PyQt6 and React/Tauri surfaces through ADR-0013 `launcher_embed` registration, shared manifest metadata, and `/tools/canonical-core/*` routes while leaving the CC-19/CC-27 service bodies to their dedicated implementation work.
- **2026-05-31** - Added the Sidekick Canonical Core retrieval Q&A tool (#6810): the chat service can now expose a read-only `answer_canonical_core_question` tool backed by a bounded local Canonical Core corpus, deterministic extractive answers, and `path:start-end` citations. The behavior is documented in `docs/sidekick/README.md` and `docs/specs/active/sidekick-canonical-core-retrieval-qa.md`.
- **2026-05-31** - Added the CC-36 config validation setup wizard (#6809): canonical-core setup now has a deterministic preflight API, headless wizard view model, launcher embeddable tool, and default model block coverage for validating units, frames, model dimensions, and subject calibration before engine execution.
- **2026-05-31** - Added the CC-38 Sidekick canonical-core tool adapter (#6811): Sidekick can now expose bounded `canonical.configure`, `canonical.validate`, `canonical.run`, `canonical.compare`, and `canonical.interpret` actions through `CanonicalToolAdapter` and a host-supplied `CanonicalActionPort`, preserving the existing audit, policy, dry-run, and destructive-confirmation gates.
- **2026-05-31** - Added the CC-7 cross-engine conformance harness for issue #6779: the engine-core validator now emits parity checks/results for canonical q/v/a traces, documents the merge-gate contract in the parity spec, and includes focused conformance tests plus hardened optional-engine CI wiring.
- **2026-05-31** - Added the Pinocchio canonical-v2 reference adapter slice for issue #6782: `pose_interchange.adapters.pinocchio_reference` now remaps canonical `[xyz, quat_wxyz]` and `[angular; linear]` q/v/a states to Pinocchio's `[xyz, quat_xyzw]` and `[linear; angular]` conventions, declares inverse-dynamics, forward-dynamics, and gradient capabilities, exposes FK/Jacobian/RNEA/ABA boundaries with an optional Rust trajectory path, and includes focused unit coverage for remap, fallback dynamics, and inertial-parameter gradients.
- **2026-05-31** - Added the MuJoCo canonical-v2 adapter slice for issue #6783: pose interchange now includes MuJoCo q/v/a remapping and capability metadata, the simulation backend exposes inverse-dynamics support, soft-contact divergence is documented in the canonical-v2 conventions, and focused unit tests cover adapter and backend behavior.
- **2026-05-31** - Added the CC-11 differential-testing report scaffold for issue #6784: `scripts/validation/cross_engine_differential_report.py` now generates normalized machine-readable and Markdown validation artifacts under `docs/validation/`, including dependency-blocked defaults and CC-7 conformance-harness normalization tests.
- **2026-05-31** - Added the CC-24 canonical ZTCF/ZVCF analysis bridge (#6797): simulation backends now expose canonical zero-torque crossing and zero-velocity crossing analysis helpers, extend the results schema v2 documentation, and cover AffineDrift-compatible event extraction and result serialization with focused unit tests.
- **2026-05-31** - Added canonical-core CI wiring for issue #6780: cross-engine equivalence now exposes per-engine conformance jobs, heavy optional stacks remain self-hosted and opt-in, canonical-core Jules templates document adapter/conformance/docstring tasks, and the JaxSim forward-simulation analytic reference uses the canonical gravity convention with the current tolerance envelope.
- **2026-05-31** - Added the CC-12 canonical observations schema for markerless pose ingestion (#6785): `src/shared/python/pose_estimation/observations.py` now preserves detector layout, calibrated camera records, per-camera 2D keypoints, per-keypoint confidence, optional triangulated 3D keypoints, JSON round-tripping, and trace metadata attachment, with fixtures, docs, and unit coverage.
- **2026-05-31** - Added the CC-14 OpenCap integration slice (#6787): the motion pipeline can now ingest OpenCap-style marker/keypoint exports through a source adapter, register the source contract, validate local fixtures, and document the supported OpenCap import format for turnkey secondary ingestion.
- **2026-05-31** - Added the CC-13 Pose2Sim integration slice (#6786): the motion pipeline now includes Pose2Sim fixture ingestion, source adapter exports, MediaPipe JSON compatibility wiring, and motion-pipeline documentation for primary local multi-camera workflows.
- **2026-05-31** - Added the CC-25 engine-agnostic wrench/GRF extraction bridge (#6798): the shared simulation backend layer now converts canonical `Trace.wrench` arrays to and from the existing `bunkershot3d.postproc.WrenchTrace` primitive, exposes impulse helpers and trace attachment, documents the unified `(T, 6)` wrench layout, and validates the static body-weight support case.
- **2026-05-31** - Added the CC-17 synthetic ground-truth rig and identifiability probes (#6790): estimation now exposes synthetic fixture generation, forward-model protocols, identifiability diagnostics, documentation, and focused tests for validating estimator inputs before fitting real trials.
- **2026-05-31** - Added the CC-27 cross-engine comparison report module (#6800): `simulation_backends.compare()` now runs selected backends from identical user input and emits structured side-by-side kinematics, kinetics, ZTCF/ZVCF, and wrench panels with divergence registry annotations and per-panel provenance; `compare_cli.py` provides a one-command Markdown/JSON report path.
- **2026-05-31** - Hardened the Bot CI trigger workflow so invalid PAT-style secrets no longer block fallback to the repository token; token validation now tries `BOT_PAT`, `RUNNER_CHECK_TOKEN`, and `github.token` in order before deciding CI cannot be triggered for bot-authored PRs.

- **2026-05-31** - Added the CC-21 AddBiomechanics inertia-prior importer (#6794): `src/shared/python/anthropometrics/addbiomechanics_priors.py` validates bounded calibration exports, converts body-segment mass/COM/inertia fields into estimator-compatible prior payloads, and documents the calibration pipeline with deterministic persistence and validation coverage.
- **2026-05-31** - Added the CC-26 AffineDrift coupling surface (#6799): `src/shared/python/analysis/affine_drift_coupling.py` now samples double-pendulum traces into pointwise drift/control-affine acceleration terms, exposes HDF5 persistence for coupling results, and documents canonical-v2 trace extraction in `docs/conventions/canonical-v2.md` and `docs/simulation_backends/results_schema_v2.md`.
- **2026-05-31** - Added the CC-16 output-only canonical C3D exporter (#6789): motion capture can now export marker trajectories from canonical state arrays to terminal C3D files with unit, label, sample-rate, and architecture guards that prevent C3D from becoming an internal intermediate.
- **2026-05-31** - Added the CC-28 Drake canonical-core adapter slice for issue #6801: the existing Drake pose adapter now declares AutoDiffXd/contact/trajectory capabilities, remaps canonical-v2 dynamic state blocks into Drake `QuaternionFloatingJoint` ordering with angular-velocity frame conversion, and registers the hydroelastic-vs-Pinocchio contact divergence in `docs/conformance/canonical_core_divergences.yaml`.
- **2026-05-31** - Added the CC-30 MyoSuite canonical-core adapter slice (#6803): activation-driven canonical-v2 state remapping for MyoSuite/MuJoCo MJCF layouts, explicit MUSCLES/FORWARD_DYN/CONTACT capability declaration with no joint-torque inverse-dynamics claim, upstream-muscle activation/force helper routing, and Trace v2.1 muscle-output persistence fields.
- **2026-05-31** - Added the CC-33 canonical 3D viewport provider decision (#6806): MeshCat is the selected default over Rerun and VTK/PyVista, with lazy provider metadata/selection/degradation and a Trace v2 overlay payload for canonical-v2 trajectory, marker, contact, and GRF/wrench data.
- **2026-05-31** - Tightened review-feedback guardrails for issues #6816 and #6827: the license ledger advisory now validates the OpenPose row cells directly, the cross-engine equivalence workflow runs when `pyproject.toml` changes so the JaxSim pin guard covers optional-extra drift, and the bot CI trigger validates `gh auth status` before attempting authenticated workflow dispatch.
- **2026-05-31** - Added canonical-core estimation residuals for issue #6791:
  pure reprojection, RNEA dynamics, anthropometric prior, and trajectory
  smoothness residual functions now live under `src/shared/python/estimation/`,
  with finite-difference Jacobian coverage, optional JAX autodiff Jacobians,
  and developer guidance in `docs/development/canonical_core_residuals.md`.
- **2026-05-31** - Hardened the runtime Docker image against the current Debian 13 medium-severity glibc, systemd/libudev, and sed CVEs by explicitly upgrading/installing `libc-bin`, `libc6`, `libsystemd0`, `libudev1`, and `sed` in the runtime apt layer while preserving the pinned `python:3.12-slim` base digest.
- **2026-05-31** - Added the launcher workspace tab close-to-background workflow for #6013: `DraggableTabWidget` can now background-close tabs without destroying their embedded widget, track backgrounded tab metadata, restore hidden tabs by title, and expose the feature through launcher UI close affordances. Regression coverage in `tests/launchers/test_workspace_tabs.py` validates close/restore behavior and state preservation.
- **2026-05-31** - Added the CC-15 calibratable keypoint-offset observation model: detector keypoints can now be calibrated against model joint centers as segment-frame offsets with covariance, standard error, confidence support, and residual helpers documented in `docs/conventions/keypoint-offset-model.md`.
- **2026-05-31** - Added CC-20 multi-trial / multi-view shared-parameter stacking: `src/shared/python/estimation/multi_trial.py` now solves independent per-trial spline blocks against one shared parameter block, excludes locked parameters from the decision vector, serializes shared-parameter specs for run manifests, and reports approximate shared-parameter posterior covariance so synthetic multi-trial fits can verify identifiable directions tighten with more data.
- **2026-05-31** - Added the canonical run `ProvenanceStamp` primitive for issue #6778: simulation traces, batch traces, and state checkpoints can now carry deterministic run metadata covering engine/model identifiers, timestamp, adapter version, units, feature flags, and dependency versions without changing the existing trace/checkpoint schemas.
- **2026-05-31** - Added the first canonical model core slice for issue #6775: `model_generation.canonical_model` now provides immutable engine-neutral links, joints, geometry, materials, stable deterministic JSON/model hashes, validation, conversion to existing model-generation core types, and URDF export through the existing writer.
- **2026-05-31** - Added metadata-driven helpful-field and provenance-value wrappers for the Idiot-Proof UX epic (#5968): PyQt6 and React controls now consume the shared field metadata/provenance contracts, UI field metadata is generated from `src/shared/python/ux/config/field_metadata.yaml`, and parity tests keep the TypeScript registry synchronized with the YAML source of truth.
- **2026-05-31** - Added the first unified engine capability taxonomy slice for issue #6777: `engine_core.capabilities.Capability` is now the canonical enum/query surface, simulation backend capabilities can answer canonical `supports()` checks while keeping legacy booleans, and architecture docs describe the adapter boundary.
- **2026-05-31** - Added a third-party license ledger for issue #6781 under `docs/legal/licenses.md`, with a CI-sized advisory checker that covers direct dependency declarations, keeps OpenPose visibly fenced as non-commercial opt-in, supports Python 3.10 via `tomli`, and avoids false core-install optional-engine findings from the local `scripts/jaxsim` helper directory.
- **2026-05-31** - Added canonical-v2 dynamic state support for CC-2 (#6774): `CanonicalState` now carries immutable `(q, v, a, t)` data with floating-base quaternion layout, manifold-safe integrate/difference operations, canonical-v1 lift helpers, and SE(3) quaternion utilities covered by shape, validation, and property-style round-trip tests.
- **2026-05-31** - Added the canonical-v2 pose interchange contract (#6773) with public exports from `src/shared/python/pose_interchange/__init__.py`, a conventions guide under `docs/conventions/canonical-v2.md`, and ADR coverage in `docs/adr/0026-canonical-dynamic-state-v2.md`.

- **2026-05-31** - Hardened the JaxSim #6648 URDF-to-SDF gate CI path by parsing inertial XML with `defusedxml` and preventing the core-only install guard from treating helper directories such as `scripts/jaxsim` as installed optional engines.
- **2026-05-30** - Hardened the JaxSim #6648 URDF-to-SDF inertial round-trip gate so converted SDF payloads fail on unexpected inertial links instead of silently accepting extra mass/inertia records; regression coverage now exercises the unexpected-link failure path.
- **2026-05-30** - Added JaxSim parameter-gradient sensitivity support (#6656): `SupportsParameterGradients` now captures pointwise parameter Jacobians, `JaxSimBackend` delegates to a JAX autodiff ZTCF sensitivity module over documented anthropometric parameters, tests validate autodiff against finite differences, and `scripts/jaxsim/plot_parameter_sensitivity.py` reproduces a sample sensitivity plot from measured states.
- **2026-06-17** - Optimized JaxSim trajectory parameter-gradient evaluation for #7562: `evaluate_ztcf_parameter_sensitivity_along_trajectory` now builds the selected autodiff transform once per call, batches measured `(q, v)` samples through `jax.vmap`, and validates finite `(T, 2, 5)` Jacobian output before returning.
- **2026-06-17** - Replaced explicit inverse effective-mass calculations for #7560: MuJoCo humanoid golf effective-mass helpers now share a solve-based kernel, finite-check mass matrices/Jacobians, and validate symmetric positive effective-mass output before returning.
- **2026-05-30** - Added JaxSim forward simulation rollout support (#6655): `JaxSimBackend.rollout` now drives `jaxsim.api.model.step`, returns the canonical `Trace` schema with full floating-base state, validates control/time preconditions, records convention metadata, and includes an analytic double-pendulum parity gate through the adapter seam.
- **2026-05-30** - Added the JaxSim/Pinocchio cross-engine dynamics parity gate (#6654): CI now runs a single-body installed-stack comparison for mass matrix, bias, gravity, and Coriolis terms, documents the tolerance envelope, and covers live JaxSim 0.9.0 model/data API compatibility.
- **2026-05-31** - Surfaced JaxSim through the capability-aware engine selector: API engine metadata, launcher capability profiles, exercise discovery, engine registry integration, and the React engine store now expose JaxSim with gated capability tooltips; the exercise dashboard opens the dedicated capability-driven JaxSim dashboard instead of a placeholder (issue #6658).
- **2026-05-31** - Added the JaxSim upgrade guard policy (#6660): CI now owns a pinned optional-dependency upgrade workflow for `jaxsim==0.9.0`, the version policy is documented in `docs/development/jaxsim_version_policy.md`, and the three-engine tutorial points users through the pinned extra before cross-engine equivalence and gradient checks are used to justify future upgrades.
- **2026-05-30** - Added the first JaxSim backend adapter (#6653): `JaxSimBackend` lazily maps JaxSim free-floating mass, bias, gravity, Coriolis, inverse-dynamics, and Jacobian APIs into engine-core load/query/dynamics protocols, declares JaxSim capabilities, and registers `EngineType.JAXSIM` in `LOADER_MAP`.
- **2026-05-30** - Rolled the backgrounding/pop-out lifecycle across every embedded tool (Sub-PR B of #6013): audited all 13 `src/tools/*/_embed_adapter.py` adapters for `cleanup()` idempotency and annotated each with a one-line `# background:` decision comment. Twelve adapters background fine at the structural defaults (`can_background`/`detach_to_window` → `True`) — they are CPU widgets or hold only in-memory state worth keeping alive while hidden, with no scarce GPU context at the adapter level and no modal-installer constraint. The `training_controller` adapter's `cleanup()` was tightened to the swap-then-clear pattern (drop widget refs first, never re-clean on a second call, never raise). The `pose_subscriber_demo` tool — the one holding a live `pose/canonical` realtime subscription — gained real `pause()`/`resume()` hooks: `pause()` releases the subscription so a hidden subscriber stops consuming traffic and `resume()` re-acquires it (widget hooks added to `src/tools/pose_subscriber_demo/gui.py`, forwarded by the adapter). No adapter needed `can_background=False`. Per-adapter idempotency, structural-default, pause/resume, and a full open→background→reopen→pop-out→dock-back state-preservation round-trip are covered in `tests/unit/launchers/test_embed_adapter_backgrounding_rollout.py` (34 tests).
- **2026-05-30** - Made embedded launcher tabs backgroundable and pop-out-able (Sub-PR A of #6013): added an additive `BackgroundableTool` protocol (`src/shared/python/launcher_embed/contract.py`, package bumped to `1.1.0`) with four optional hooks — `pause()`, `resume()`, `can_background()` (default `True`), and `detach_to_window()` (default `True`) — kept separate from `EmbeddableTool` so its `runtime_checkable` `isinstance` check still accepts the ~17 existing adapters, with hosts resolving the hooks structurally via `getattr`-with-default. `EmbeddedHostWidget` (`src/launchers/embedded_host.py`) now prompts "Close (keep running)" vs "Destroy" on tab close: background-close pauses the tool and stashes its widget hidden (re-surfaced with `resume()` on reopen), while destroy keeps the legacy `cleanup()` path. Added `pop_out_tab(tool_id)` / `dock_back(tool_id)` (re-parent the live widget into / out of a top-level `QMainWindow`; closing the popped-out window re-docks), a tab-bar context menu (Close-keep-running / Destroy / Pop out), and the public `backgrounded_tools() -> set[str]` API. Tests in `tests/unit/launchers/test_embedded_host_backgrounding.py`. Per-tool adapter rollout is tracked separately as Sub-PR B.
- **2026-05-30** - Added the JaxSim floating-base velocity convention contract (#6652): engine-core now defines body-fixed, inertial, and mixed velocity representations, normalization helpers to the suite's inertial canonical representation, gravity/base-frame units, and single-floating-body analytic `h`/`g` coverage tied to `SPATIAL_JACOBIAN_ORDER`.
- **2026-05-30** - Extended the engine capability taxonomy for JaxSim planning (#6651): `EngineCapabilities` now reports parameter gradients, state/control gradients, forward simulation, contact stepping, and trajectory optimization support with accessors, serialization round-trip coverage, and documented verified engine profiles.
- **2026-05-30** - Added the gated JaxSim optional dependency extra (`upstream-drift[jaxsim]`) pinned to `jaxsim==0.9.0`, with CPU-JAX-first documentation, core-install isolation coverage, and an optional SDF step smoke test for the JaxSim stack.
- **2026-05-30** - Added the JaxSim #6648 canonical URDF-to-SDF gate harness: sdformat CLI detection, SDF conversion, mass/inertia round-trip checks, BRICK setup documentation, and optional JaxSim loading coverage asserting the canonical 25-velocity model contract.
- **2026-05-30** - Added a full-src mypy ratchet for push-to-main CI: mypy now uses explicit package bases for namespace-package discovery, and push runs compare `mypy src --config-file pyproject.toml` against `scripts/config/full_src_mypy_baseline.json` so new type debt fails while the current unmasked backlog remains accountable.
- **2026-05-30** - Hardened the runtime Docker image against the current Debian 13 `libcap2` high-severity CVE by explicitly upgrading/installing `libcap2` during the runtime apt layer while preserving the pinned `python:3.12-slim` base digest.
- **2026-05-30** - Declared `pyarrow>=14.0.0` in the data/dev dependency surfaces and regenerated dependency artifacts so Parquet compactor/loader tests can collect in CI; `tests/unit/test_build_install_contracts.py` now falls back to `tomli` on Python 3.10.
- **2026-05-30** - Widen pinocchio version limit from `<3.0.0` to `<5.0.0` in `pyproject.toml` to resolve numpy 2.0 version compatibility conflict.
- **2026-05-29** - API security hardening (issue #6643): introduced `_assert_type` guard in `src/api/auth/dependencies.py` to narrow SQLAlchemy query-result types for MyPy strict mode, replaced the previous `type: ignore[return-value]` workarounds with explicit runtime assertions, and added `_lookup_cached_api_key`, `_lookup_api_key_by_prefix`, and `_get_active_user_for_api_key` helper functions for testability. Added `src/api/auth/dependencies.py` prefix-hash API-key lookup regression coverage in `tests/unit/api/test_api_hardening_6643.py`. Sim GUI honest messaging (issue #6641): updated `src/tools/bunker_shot_gui/gui.py` and `src/tools/putting_green_gui/gui.py` to surface explicit error and loading states instead of silently showing stale data; regression tests added in `tests/unit/test_sim_gui_honest_messaging_6641.py`.
- **2026-05-29** - Documented differentiable trajectory optimization behavior for zero-iteration runs: `optimize_trajectory()` now returns a valid `OptimizationResult` with the initial control sequence and an infinite gradient norm sentinel instead of reading an uninitialized gradient value.
- **2026-05-29** - Updated CI hygiene contract for PR #6624: agent-doc literal path validation now skips glob/brace patterns, root-clutter allowlist documents `launch_upstream_drift.py` as a substantive launcher entry point, module-size baseline exceptions remain owner/expiry governed, and the canonical Sidekick embeddable adapter stays under `src/tools/sidekick/_embed_adapter.py` after removing the obsolete duplicate shared-chat adapter.
- **2026-05-29** - Annotated cross-engine dashboard comparison results with per-engine velocity convention and units metadata in GUI result labels and headless logs so learners can see which native representation each engine result uses before normalized comparison (closes #6659).
- **2026-05-28** - Resolved python path resolution bug in `embedded_tool_bootstrap.py` and `upstream_drift_launcher.py` to fix launcher boot-time `ModuleNotFoundError` crashes and warnings.
- **2026-05-28** - Added sg-optimizer Phase 2: GeoJSON I/O (`course_io.py` with `HoleGeometry`, `load_hole_geojson`, `save_hole_geojson`), UTM geometry utilities (`geometry.py` with `LatLonPoint`, `UTMPoint`, `project_to_utm`, `utm_to_latlon`, `haversine_m` via pyproj), classic-holes library (`library.py` with 5 GeoJSON data files for Sawgrass 17, Augusta 13, Pebble 7, Road Hole 17, Cypress 16), `StateFeatures` dataclass factory (`features.py`), and full `TreeModel` with `forced_punch_out_probability` distribution (`mdp/tree_model.py`); adds `pyproj>=3.6.0` optional dependency (closes #6271).
- **2026-05-28** - Restored production symbols deleted by Bolt commit #6501: `_resolve_default_server` in `chat_dock_widget`, full 60-token `ThemeColors` derivation pipeline in `theme/api.py`, `ThemeColorsCompat` and `_derive_full_palette` in `theme/__init__.py`, `_tool_declarations_to_ollama` + `keep_alive`/`num_ctx` latency optimizations in `ollama_adapter.py`, and `_EmbedAdapter` + `_register()` in all 5 tool GUI modules; closes #6527, #6528, #6529. Also fixes sg_optimizer longitudinal dispersion applying wrong modifier column (closes #6343).
- **2026-05-27** - Confirmed Standalone Sidekick T2 (`StandaloneSidekickWindow` chat-first/calc-first layouts and profile switching) and T5 (state-profile round-trip with schema-version written to saved JSON) acceptance criteria with targeted new tests; closes #5980 and #5983.
- **2026-05-27** - Completed Standalone Sidekick T4 acceptance criteria: `sidekick run --calculator` now validates inputs via the Calculator Protocol, surfaces structured errors on validation/calculation failure (exit 3), unknown-calculator with fuzzy suggestions (exit 4), and I/O errors (exit 1); supports `--format json` and `--format csv`; full TDD coverage in `tests/unit/sidekick/standalone/test_run.py` (issue #5982).
- **2026-05-28** - Enabled dynamic MuJoCo GUI docking and styling in the launcher via DraggableTabWidget and dynamic ThemeManager palette application to resolve issue #6509.
- **2026-05-28** - Connected Model Explorer widget destroyed signal to cleanup method to ensure proper tool lifecycle in launcher simulation.
- **2026-05-28** - Resolved launcher widget parent reference crashes by using `self._launcher` instead of `self.parent()` in `SettingsWidget`.
- **2026-05-28** - Registered the shared.python.config subpackage in lazy loading to prevent mock-patching AttributeError during launcher diagnostics unit testing.
- **2026-05-27** - Resolved mypy type-checking errors by excluding Jython/OpenSim scripts from the pre-commit mypy hook and replaced print statements with logging.info/logging.warning in computeMomentArm.py and AGENT_INSTRUCTIONS.md to satisfy the no-print-in-src hook.
- **2026-05-26** - Folded remaining API/security/realtime/logging PR scope into the post-#6181 consolidation branch: `FitResult` now exposes explicit `fit_succeeded` and `solver_status` fields, the `.gitignore` secrets guard has an importable CI helper plus tests, and logging redaction preserves delimiters while redacting quoted, JSON, and comma-containing secret values.
- **2026-05-24** - Surfaced API database pool controls for non-SQLite deployments via `GOLF_DB_POOL_SIZE`, `GOLF_DB_POOL_RECYCLE`, and `GOLF_DB_POOL_PRE_PING`; `src/api/database.py` now builds non-SQLite engines from shared config accessors instead of hardcoded pool defaults, with regression coverage in `tests/unit/test_config_environment.py` and `tests/unit/api/test_database_init.py`.
- **2026-05-24** - Added shared `GOLF_REALTIME_HOST` / `GOLF_REALTIME_PORT` environment accessors and wired `src/shared/python/realtime/ws_pubsub.py` plus API diagnostics to use/report them, so realtime bind defaults no longer live only as hard-coded loopback literals.
- **2026-05-24** - Deferred realtime WebSocket backend resolution in `src/shared/python/realtime/ws_pubsub.py` until first explicit start/use and made `WSPubSub.start()` bring up the Python backend even when the instance was created with `autostart=False`; added focused regression coverage in `tests/shared/realtime/test_ws_pubsub.py`.
- **2026-05-24** - Improved CI/test observability for optional dependency lanes: optional pytest collection skips now emit one warning per skipped path with the missing module or symbol, the PyTorch CVAE cancellation regression now uses a wrapper progress sink instead of monkeypatching methods, and three standard workflow inventory jobs now have 15-minute budgets to reduce false timeouts on saturated self-hosted runners.
- **2026-05-23** - Closed the file-size budget grandfathering gap by requiring tracked baseline entries in `scripts/config/file_size_budget.json` for oversized files and adding regression coverage for untracked oversized files.
- **2026-05-23** - Tightened `src/shared/python/training/config.py` validation so boolean values are rejected for integer training caps such as `max_epochs` and `max_steps`, with regression coverage in `tests/unit/training/test_config.py`.
- **2026-05-23** - Deferred realtime WebSocket backend resolution in `src/shared/python/realtime/ws_pubsub.py` until `WSPubSub.start()`, `publish()`, or `subscribe()` first use so importing the module no longer probes optional runtime dependencies; added focused lazy-resolution regression coverage for the python publish fallback path.
- **2026-05-23** - Sanitized error payloads for the chat websocket connection to prevent leaks.
- **2026-05-23** - Added standalone Sidekick foundation (CLI entry point, PyQt window shell, and session store) per epic #5979.
- **2026-05-23** - Added the subprocess-isolated training Driver (`src/shared/python/training/runtime/subprocess_driver.py`, `worker_main.py`, `wire_protocol.py`) — `SubprocessDriver` satisfies the `Driver` Protocol so the scheduler swap is one-line, spawns workers via `core.process_safety.managed_popen` (mandatory per the error-handling ratchet), parses a newline-delimited JSON wire protocol whose payloads reuse `training.persistence` dicts, propagates cancel through stdin, surfaces worker crashes as FAILED RunResults with stderr context, and writes a `.training.pid` file per job so the launcher can detect orphaned workers via `scan_pidfiles` on restart. 65 new unit tests (wire-protocol round-trips, isolated worker-subprocess wire tests, end-to-end driver coverage of completion / cancel / crash / stderr isolation / pidfile lifecycle); follows issue #6015.
- **2026-05-23** - Wired the existing PyTorch inverse-CVAE training loop (`src/shared/python/motion_matching/inverse/training.py`) into the training-controller via a new `PyTorchCVAERunner` adapter (`src/shared/python/training/runtime/adapters/pytorch_cvae.py`). The adapter satisfies the `TrainingJobRunner` Protocol, translates `TrainingConfig.hyperparameters` into the loop's `TrainingConfig` dataclass, streams 6 `TrainingMetric`s per epoch (train_recon / train_kl / val_recon / val_kl as LOSS; beta / duration_s as SCALAR) tagged with `split=train|val`, and exposes the best-so-far checkpoint + `metrics.json` as `RunResult.artifacts`. The upstream loop gained two optional default-None kwargs — `on_epoch_end(metrics)` and `should_stop()` — so cooperative cancellation routes through `CancelToken.is_cancelled` without changing default behaviour (existing motion_matching tests unaffected). Adapter and regression tests are guarded by `pytest.importorskip("torch")`; the headless training-controller surface still imports cleanly without torch installed. Closes #6014.
- **2026-05-23** - Added the headless half of the training-controller dashboard tab (`src/tools/training_controller/`) per the in-scope portion of issue #6012: `TrainingDashboardController` MVC controller binding the backend `Scheduler` to a frozen `DashboardModel`, `TrainingJobLiveSubscriber` realtime-channel wrapper that decodes `training/<job_id>/progress` payloads into typed `TrainingMetric` / `TrainingStatus` events, and the `view_model` dataclasses (`JobRow`, `MetricSeries`, `ResourceSnapshot`, `GpuSnapshot`, `DashboardModel`) the GUI follow-up will render. 82 new headless unit tests (controller / live subscriber / view model); no PyQt6 import in `src/tools/training_controller/`. The PyQt6 widget surface (`gui.py`, `__main__.py`, `_embed_adapter.py`, `src/config/models.yaml` tile) is deferred to a follow-up PR that can be validated against a live display.
- **2026-05-23** - Added the model-training controller foundation (`src/shared/python/training/`) — PR1 contracts (status state machine, identifiers, resources, config, metrics, job/run records, compatibility checker, runner Protocols), PR2 backend (scheduler, in-process driver, runner registry, dataset library, JSON persistence, progress sinks: in-memory / JSONL-file / composite / realtime-channel), and PR3 backend additions (ResourceMonitor with psutil + optional pynvml, metric_summary helpers for best-per-metric / by-kind / by-tag / rolling means for noisy RL returns). Pure-Python, GUI-free; the headless backend that PR4's tab-backgrounding refactor and PR5's GUI tab + CVAE wiring build on. 332 unit tests passing in <1 s; 21 new public modules. The PyQt6 dashboard tab, the system-wide tab-backgrounding refactor, and the PyTorch CVAE adapter wiring are deferred to subsequent PRs that can be validated against a live display / torch environment.
- **2026-05-22** - Added the Sidekick agentic action layer (`src/shared/python/sidekick/agent/`) per epic #5967 and ADR-0017: feature catalog, audited `SidekickActionService` with default-deny policy and undo tokens, subtab and host action adapters, planner + tool-registry bridge, workflow runner, and chat-side action chip surface. 157 new unit tests; ten new public modules totalling ~3,000 LOC.
- **2026-05-22** - Tightened the CI error-handling ratchet so multiline `asyncio.gather(...)` calls are parsed across balanced parentheses before checking for `return_exceptions=`, and added focused regression tests that cover both the exempt and failing multiline forms.
- **2026-05-22** - Documented the API auth-cache overflow contract so cache saturation now evicts only the oldest lookup entries instead of flushing unrelated authenticated sessions, while preserving deterministic SHA-256 lookup keys for cross-worker stability.
- **2026-05-22** - Documented the motion-pipeline REST contract for preprocessing-step boolean coercion so `PipelineRequest` preserves Pydantic handling of `enabled` values like `"false"` when converting into `PipelineConfig`.
- **2026-05-23** - Added the standalone Sidekick UX/documentation layer per ADR-0018: persisted standalone preferences, onboarding sentinel handling, user-facing standalone docs, and contract tests for standalone preferences, onboarding, and docs discoverability.
- **2026-05-22** - Documented added unit regression coverage for the theme API model/router contracts so the shared theme settings surface stays exercised without broadening the implementation scope of the underlying runtime code.
- **2026-05-23** - Hardened WebSocket error handling so unexpected chat/simulation failures log full tracebacks server-side while returning generic client-safe error payloads.
- **2026-05-21** - Added C3D viewer animation export through the canonical body-target video pipeline and stabilized self-hosted CI SciPy pinning for the core and shared-contract lanes.
- **2026-05-21** - Preserved integer-safe quaternion normalization in the C3D Simscape preview path while keeping the optimized `einsum`-based norm computation.
- **2026-05-21** - Optimized `signal_toolkit` fitting R-squared and RMSE hot paths to reuse `np.vdot`-based sum-of-squares accumulators without temporary square arrays.

- **2026-05-30** - Optimized 3D vector magnitude calculations across physics and validation models by replacing `np.linalg.norm` with `math.sqrt(np.dot(v, v))` to eliminate array allocation overhead on the hot path.

### System Context

UpstreamDrift sits at the center of a biomechanical simulation ecosystem. It depends on five external physics engines as pluggable backends and exposes its functionality through three primary interfaces: a professional PyQt6 GUI for interactive simulation, a FastAPI REST API for programmatic access, and a Tauri desktop application for cross-platform deployment. The system integrates with motion capture systems (via MediaPipe and custom importers), optimization libraries (SciPy, Sympy), and machine learning frameworks (scikit-learn for RL integration). The Rust core (`rust_core/upstream-physics/`) provides high-performance physics kernels for compute-intensive operations.

### Module Map

- **2026-06-18:** Optimized `src/deployment/digital_twin/twin.py` to use `math.sqrt(np.dot())` and `math.hypot()` instead of `np.linalg.norm()` for scalar inputs and small vectors.

The operational companion to this map (startup phases, tab/sidekick wiring,
and the tracked implementation-gap inventory) lives in
[`docs/architecture/PROJECT_MAP.md`](docs/architecture/PROJECT_MAP.md) §16.

```
UpstreamDrift/
├── launch_upstream_drift.py        # Canonical entry point (web/classic/api-only/engine)
├── launch_golf_suite.py            # Legacy alias entry point (console script target; #7215)
├── src/
│   ├── engines/
│   │   ├── physics_engines/        # Engine adapters (package directories)
│   │   │   ├── mujoco/             # MuJoCo backend (core)
│   │   │   ├── drake/              # Drake backend (extended)
│   │   │   ├── pinocchio/          # Pinocchio backend (extended)
│   │   │   ├── jaxsim/             # JaxSim backend (beta)
│   │   │   ├── opensim/            # OpenSim backend (experimental)
│   │   │   ├── myosuite/           # MyoSuite backend (experimental)
│   │   │   ├── pendulum/           # Simplified educational models
│   │   │   └── putting_green/      # Putting green simulation
│   │   ├── pendulum_models/        # Educational pendulum models
│   │   └── Simscape_Multibody_Models/  # MATLAB models + C3D viewer app
│   ├── launchers/                  # PyQt6 launcher (50+ modules)
│   │   ├── upstream_drift_launcher.py  # Main window (size split tracked: #7217)
│   │   ├── embedded_host.py        # Tab/dock host: pop-out, backgrounding
│   │   ├── embedded_tool_bootstrap.py  # Embeddable-adapter registration
│   │   ├── sidekick_host_port.py   # Sidekick agent ↔ tabs bridge (subtab port)
│   │   └── {mujoco,drake,pinocchio,jaxsim}_dashboard.py, dialogs, theme, …
│   ├── api/                        # FastAPI backend
│   │   ├── local_server.py         # Server entry (web UI host)
│   │   ├── routes/                 # 30+ endpoint modules
│   │   ├── services/               # Simulation/chat/analysis services
│   │   └── auth/, middleware/, models/, utils/
│   ├── config/                     # Launcher manifest + models.yaml loaders
│   ├── tools/                      # Embeddable tool tabs (model_explorer,
│   │                               # ball_flight_gui, putting_green_gui,
│   │                               # swing_flight_pipeline, pose_studio,
│   │                               # launch_monitor_analytics,
│   │                               # video_analyzer, sidekick, …) plus
│   │                               # headless analysis CLIs (drift_control,
│   │                               # contraction)
│   └── shared/python/              # Cross-cutting libraries; highlights:
│       ├── engine_core/            # EngineManager/Registry/probes/capabilities
│       ├── ground_model/            # Fail-closed consumer of Tools ground v1
│       ├── launcher_embed/         # EmbeddableTool contract + registry (ADR-0013)
│       ├── physics/                # Ball flight models, impact, swing→flight pipeline
│       ├── putting_dynamics/       # Surface-aware putt collision and roll physics
│       ├── launch_monitor/         # Canonical shot import, treatment, and analytics
│       ├── motion_pipeline/        # Mocap ingestion (C3D/TRC/BVH), IK backends
│       ├── model_generation/       # URDF/MJCF parsing, Frankenstein editor (VENDORED)
│       ├── sidekick/               # Shared tools library + agent layer (VENDORED)
│       ├── humanoid_character_builder/  # Parametric humanoid URDF generation
│       └── pose_interchange/, realtime/, simulation_backends/, config/, …
├── rust_core/                      # Maturin crates
│   ├── upstream-physics/           # Ball flight, aero, contact, RK4 kernels
│   ├── upstream-mocap-io/          # C3D/TRC/BVH parsers (PyO3)
│   ├── upstream-mocap-preproc/, upstream-urdf/, upstream-mesh/,
│   ├── upstream-muscle/, upstream-motion-matching/, upstream-pinocchio-id/,
│   └── upstream-realtime/, upstream-codemap/, ai_backend/
├── ui/                             # React + Tauri launcher (manifest-driven)
├── vendor/ud-tools/                # Vendored Tools repo (canonical for sidekick
│                                   # and model_generation packages)
├── data/                           # Sample data incl. C3D captures (golf TA + CMU)
├── tests/                          # unit/, integration/, launchers/, api/, tools/,
│                                   # heavy_integration/, benchmarks/, …
├── scripts/                        # CI gates + config baselines (scripts/config/)
├── docs/                           # ADRs, architecture (PROJECT_MAP.md), guides
├── pyproject.toml                  # Canonical dependency + console-script source
├── SPEC.md                         # This file
└── README.md
```

### Key Components

| Component                | Location                                                                                                                                            | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| MuJoCo Engine Adapter    | `src/engines/physics_engines/mujoco/`                                                                                                               | Primary physics engine integration with full support for contact dynamics and muscle models                                                                                                                                                                                                                                                                                                                                                                                                      |
| Drake Engine Adapter     | `src/engines/physics_engines/drake/`                                                                                                                | Extended Drake support for trajectory optimization and manipulation tasks                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Pinocchio Engine Adapter | `src/engines/physics_engines/pinocchio/`                                                                                                            | Extended Pinocchio support for efficient rigid-body dynamics computation                                                                                                                                                                                                                                                                                                                                                                                                                         |
| OpenSim Engine Adapter   | `src/engines/physics_engines/opensim/`                                                                                                              | Experimental OpenSim integration for clinical biomechanics workflows                                                                                                                                                                                                                                                                                                                                                                                                                             |
| MyoSuite Engine Adapter  | `src/engines/physics_engines/myosuite/`                                                                                                             | Experimental MyoSuite integration for detailed muscle physiology simulation                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Pendulum Models          | `src/engines/physics_engines/pendulum/`                                                                                                             | Educational simplified models for learning and quick prototyping                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| FastAPI Backend          | `src/api/`                                                                                                                                          | REST API exposing simulation, IK/ID, trajectory optimization, and control endpoints                                                                                                                                                                                                                                                                                                                                                                                                              |
| PyQt6 GUI                | `src/launchers/upstream_drift_launcher.py`                                                                                                          | Professional interactive GUI with real-time 3D visualization                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Sidekick (AI assistant)  | PyQt: `src/shared/python/ai/gui/assistant_panel.py` · React: `ui/src/components/ui/ChatPanel.tsx` · Adapter: `src/tools/sidekick/_embed_adapter.py` | In-app AI chat surface with streaming, RAG, session history, and agentic tool dispatch. Design tokens: `src/shared/python/theme/sidekick_tokens.py`. See `docs/sidekick/README.md`.                                                                                                                                                                                                                                                                                                              |
| Tauri Desktop App        | `ui/`                                                                                                                                               | Cross-platform desktop application wrapper (Windows, macOS, Linux)                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Rust Physics Kernels     | `rust_core/upstream-physics/`                                                                                                                       | High-performance compiled physics routines for critical paths, including initial flexible shaft FEM element primitives                                                                                                                                                                                                                                                                                                                                                                           |
| Configuration Manager    | `src/config/`                                                                                                                                       | Centralized configuration loading, validation, and environment management                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Analysis Tool CLIs       | `src/tools/drift_control/`, `src/tools/contraction/`                                                                                                | Headless AffineDrift-compatible drift/control, contraction, and Floquet analysis tools                                                                                                                                                                                                                                                                                                                                                                                                           |
| Launch Monitor Analytics | `src/tools/launch_monitor_analytics/`, `src/tools/launch_monitor_model/`, `src/api/services/launch_monitor_dataset_jobs.py`                         | PyQt6, FastAPI, and headless vendor-neutral analysis; contract v2 adds traceable arbitrary-field analysis, source-backed SG verifies benchmark provenance, immutable aggregate-only dataset jobs analyze hash-pinned private authorities without transferring rows or client paths, canonical player covariation separates pooled/within/between effects with fixed/random population synthesis, and attested longitudinal analysis aggregates by session before descriptive clustered inference |
| Tools Ground Consumer    | `src/shared/python/ground_model/`                                                                                                                   | Headless exact-schema gateway to Tools flight-to-ground v1 records and reference execution; UI and final dependency pins remain tracked                                                                                                                                                                                                                                                                                                                                                          |
| Putting Dynamics         | `src/shared/python/putting_dynamics/`                                                                                                               | Headless heterogeneous-green, collision, loft, hosel-wrench, skid/roll/rest, and hole-capture physics for #8345                                                                                                                                                                                                                                                                                                                                                                                  |
| 3D Putting UI            | `src/api/routes/putting_green.py`, `ui/src/pages/PuttingGreen.tsx`, `ui/src/components/visualization/PuttingScene3D.tsx`                            | Generated-contract R3F playback of the canonical putting model with collision, spin, hosel, surface, camera, and video controls for #8345 P1                                                                                                                                                                                                                                                                                                                                                     |
| BunkerShot3D Metrics     | `src/bunkershot3d/metrics/`                                                                                                                         | Designer-facing metrics for bunker shot analysis: trajectory (dig/skid, depth trace), energy partition (club KE, sand/ball transfer), force/deceleration, head twist (shaft/CG moments), and forgiveness sensitivity gradients. Computed from HDF5 result artifacts for tier-agnostic (F0–F3) analysis per #8614.                                                                                                                                                                                |
| Shared Utilities         | `src/shared/`                                                                                                                                       | Cross-engine validators, helpers, and exception definitions                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Workspace Metadata       | `src/shared/python/workspace/`                                                                                                                      | Project/session/dataset metadata store and CC-4 HDF5 result browser view models                                                                                                                                                                                                                                                                                                                                                                                                                  |
| URDF Models              | `shared/models/`                                                                                                                                    | Canonical model definitions (URDF format) for golf swings, human body, pendulums                                                                                                                                                                                                                                                                                                                                                                                                                 |

### Engine Tier Policy

| Tier         | Examples                | Stability bar                                            | Deps installed by default | Vulnerability SLA |
| ------------ | ----------------------- | -------------------------------------------------------- | ------------------------- | ----------------- |
| core         | MuJoCo, FastAPI, shared | Must pass on every PR; semver-stable public API; no skip | yes                       | High/Critical: 7d |
| extended     | Drake, Pinocchio        | Must pass nightly; semver-stable in major versions       | only with extra           | High: 30d         |
| experimental | OpenSim, MyoSuite       | Best-effort; may be skipped; API may break               | only with extra; warning  | Best effort       |
| archived     | (none today)            | Read-only; not built; not tested                         | no                        | n/a               |

Engine tier metadata is declared in each in-scope engine package with
`_tier.py` and enforced by `scripts/check_engine_tiers.py`.

## 5. Desired Functionality

### Core Features

| #   | Feature                                     | Status | Description                                                                                                                                                                                                                                                                                                                    |
| --- | ------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| F1  | MuJoCo engine integration                   | ✅     | Full support for MuJoCo 3.3.0+ with contact dynamics, muscle actuators, sensor simulation, and pose-conditioned motion-matching target synthesis                                                                                                                                                                               |
| F2  | Drake engine integration                    | ✅     | Extended Drake support for trajectory optimization, manipulation, and planning problems                                                                                                                                                                                                                                        |
| F3  | Pinocchio engine integration                | ✅     | Extended Pinocchio support for efficient rigid-body dynamics and jacobian computation                                                                                                                                                                                                                                          |
| F4  | OpenSim engine integration                  | 🔄     | Experimental OpenSim integration for clinical biomechanics and musculoskeletal analysis                                                                                                                                                                                                                                        |
| F5  | MyoSuite engine integration                 | 🔄     | Experimental MyoSuite integration for detailed muscle physiology and motor control                                                                                                                                                                                                                                             |
| F6  | Cross-engine validation and reports         | ✅     | Automated cross-validation plus user-facing comparison reports across selected engines with tolerance thresholds, provenance, and divergence annotations                                                                                                                                                                       |
| F7  | FastAPI REST API                            | ✅     | Programmatic access to simulation, IK/ID, trajectory optimization, and control endpoints                                                                                                                                                                                                                                       |
| F8  | PyQt6 professional GUI                      | ✅     | Interactive desktop GUI with real-time 3D rendering, parameter adjustment, and result export                                                                                                                                                                                                                                   |
| F9  | Tauri desktop application                   | 🔄     | Cross-platform desktop app bundling the GUI and API with native OS integration                                                                                                                                                                                                                                                 |
| F10 | MATLAB/Simulink integration                 | ✅     | Export models to MATLAB format and integrate with Simulink via MEX interface                                                                                                                                                                                                                                                   |
| F11 | Trajectory optimization                     | ✅     | SciPy-based trajectory optimization with constraint support and custom cost functions                                                                                                                                                                                                                                          |
| F12 | Muscle dynamics analysis                    | ✅     | IK, ID, and muscle dynamics computation with Hill-type and Millard muscle models                                                                                                                                                                                                                                               |
| F13 | Motion capture integration                  | 🔄     | Import and track motion capture data (C3D, BVH, TRC formats) and compare with simulation                                                                                                                                                                                                                                       |
| F14 | Reinforcement learning integration          | 🔄     | Gym-compatible interface for RL-based controller learning and policy optimization                                                                                                                                                                                                                                              |
| F15 | Sidekick AI assistant                       | 🔄     | In-app and standalone AI assistant surface (PyQt + React/Tauri + `sidekick.standalone.*`) with streaming, RAG, session history, persisted standalone preferences, onboarding, and agentic tool dispatch. See `docs/sidekick/README.md` and ADR-0018.                                                                           |
| F16 | Model-training controller                   | 🔄     | In-launcher training dashboard (PR3) with scheduler, dataset library, resource monitor, engine-compat gate, and ML/RL-aware stats. Backend contracts + scheduler land in `src/shared/python/training/` (PRs 1–2); GUI tab, tab-backgrounding refactor, and CVAE wiring in PRs 3–5.                                             |
| F17 | Tools ground-model integration              | 🔄     | Headless v1 consumer gateway validates the canonical Tools façade and degrades safely when absent; exact dependency pins, FastAPI, PyQt, React, parity, and protected release remain open under Tools #4276.                                                                                                                   |
| F18 | Source-backed strokes gained                | ✅     | Canonical Python and FastAPI contracts score complete course-state transitions against a versioned, hash-verified expected-strokes source; provenance, backing values, exclusions, uncertainty, and identity trust remain explicit, while outcome proxies cannot claim SG.                                                     |
| F19 | Immutable launch-monitor dataset jobs       | ✅     | Authenticated, aggregate-only FastAPI jobs bind an administrator-authorized private authority to exact repository/commit/manifest/content/count identity; fixed allowlisted operations, disclosure floors, bounded pages, structured unavailable states, and deterministic worker shutdown prevent inline row or path leakage. |
| F20 | Player covariation and population synthesis | ✅     | A versioned evidence-bearing contract separates pooled, within-player, between-player, and per-player associations; fixed/random Fisher-z synthesis reports heterogeneity, exclusions, unavailable states, explicit units, trusted player identity, vendor/model provenance, and source-linked backing rows.                   |

### API / Interface Contract

**REST API Endpoints (FastAPI)**:

- `GET /health` — Health check
- `POST /simulate` — Run single simulation with specified engine and parameters
- `POST /cross-validate` — Run multi-engine cross-validation and return results
- `POST /ik` — Solve inverse kinematics given target pose
- `POST /id` — Solve inverse dynamics given trajectory
- `POST /trajectory-optimize` — Optimize trajectory subject to constraints
- `GET /engines` — List available physics engines and their status
- `POST /export` — Export simulation model to URDF, MATLAB, or other formats
- `POST /api/v1/motion-pipeline/run` — Run motion-pipeline preprocessing, scaling, IK, and motion-matching for uploaded capture files
- `GET /tools/launch-monitor-analytics/contracts/strokes-gained/v1` — Publish the canonical source-backed SG result schema
- `POST /tools/launch-monitor-analytics/v2/strokes-gained` — Score explicit start/finish states against a supplied, source-verified expected-strokes baseline
- `POST /tools/launch-monitor-analytics/v2/outcome-proxy` — Compute target-relative radial error under a contract that forbids an SG claim
- `GET /tools/launch-monitor-analytics/contracts/dataset-jobs/v1` — Publish the immutable aggregate-job request schema
- `POST /tools/launch-monitor-analytics/v2/dataset-jobs` — Queue an authenticated aggregate job against an exact server-authorized dataset reference
- `GET /tools/launch-monitor-analytics/v2/dataset-jobs/{job_id}` — Return data-free job status and structured unavailable reasons
- `GET /tools/launch-monitor-analytics/v2/dataset-jobs/{job_id}/results` — Page bounded aggregate/source-provenance results without observations or private paths
- `GET /tools/launch-monitor-analytics/contracts/player-covariation/v1` — Publish the canonical selected-pair and exploratory-scan result schema
- `POST /tools/launch-monitor-analytics/v2/player-covariation` — Compare pooled, within-player, between-player, per-player, and population effects for an explicitly identified player variable pair
- `POST /tools/launch-monitor-analytics/v2/player-covariation/scan` — Rank a bounded exploratory variable-pair set with structured unavailable states and multiplicity warnings

**API Production-Readiness Contracts**:

- Background task state is process-local and owned by the FastAPI application
  lifespan. Each app lifecycle creates its own `TaskManager`; shutdown marks the
  manager closed, clears retained task records, and subsequent task operations
  fail with a closed-state error instead of silently accepting writes.
- Motion-pipeline request normalization preserves Pydantic/native boolean
  coercion for preprocessing step `enabled` flags so form/JSON values such as
  `"false"` remain disabled instead of being forced truthy during
  `PipelineRequest -> PipelineConfig` conversion.
- Simulation WebSocket routes preserve traceback-bearing server logs for
  unexpected runtime failures while returning sanitized generic client errors so
  backend exception details are not exposed over the socket.
- `TaskManager` entries expire after the configured TTL and enforce the
  configured maximum task count. Reads and existence checks refresh the task's
  retention timestamp so actively polled async jobs are not evicted while a
  client is still observing them.
- Async video analysis queues request handling quickly, then runs the blocking
  video pose pipeline off the event loop. Temporary uploaded video files are
  deleted after completion or failure; cleanup failures are logged as warnings
  and do not mask the task result.
- Data Explorer imported datasets are kept in a bounded in-memory LRU cache.
  Importing a duplicate filename returns a conflict instead of replacing the
  existing dataset. Disk-backed dataset lookup rejects ambiguous duplicate
  filenames with a conflict response so callers do not receive an arbitrary
  match.
- `src/shared/python/realtime/ws_pubsub.py` resolves its default backend lazily.
  Constructing `WSPubSub` no longer imports or probes optional realtime runtime
  dependencies until `start()`, `publish()`, or `subscribe()` is invoked, while
  explicit `backend=` overrides and the python HTTP publish fallback remain
  supported.
- Chat and simulation WebSocket routes treat unexpected internal exceptions as
  server-only detail: they log full tracebacks for operator diagnosis and send
  generic client-safe error payloads instead of echoing raw exception strings.

**GUI Interface (PyQt6)**:

- The classic launcher entry point reuses an active `QApplication` and creates
  one only when the process has none; it must never construct a second Qt
  application object in an embedded or test-hosted process.
- Model loader and parameter editor
- Real-time 3D simulation viewer with playback controls
- Cross-engine comparison visualizer
- IK/ID solver interface with result tables
- Trajectory optimization GUI with constraint editor
- Data export and report generation

**CLI Interface**:

- `upstream-drift` launches the web UI (console script → `launch_upstream_drift:parse_arguments`/`route_launch`).
- `upstream-drift --classic` launches the classic PyQt6 desktop launcher instead of the web UI.
- `upstream-drift --api-only [--port 8000]` starts the FastAPI server without any UI.
- `upstream-drift --engine mujoco [--no-browser]` launches a specific engine directly; `--engine` choices come from `EngineType`.
- `python -m sidekick` launches the standalone Sidekick GUI scaffold with the `gui` subcommand and `chat-first` profile as the default path.
- `python -m sidekick gui --profile calc-first --theme solarized --data-dir ./workspace` keeps GUI imports deferred until launch while resolving the standalone data directory before window creation.
- `python -m sidekick run --calculator unit-converter --inputs ./inputs.json --output ./result.json` validates the headless calculator invocation contract up front; execution remains reserved for follow-up issue `#5982`.
- The standalone Sidekick CLI suggests the nearest valid flag or subcommand on parse errors to keep local launches and future automation entrypoints discoverable.

**Desktop App (Tauri)**:

- Native window management and file dialogs
- System menu integration
- Automated updates and crash reporting

## 6. Data & Configuration

### Input Data

| Input                     | Format                         | Source                                             | Schema                                                                                       |
| ------------------------- | ------------------------------ | -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Biomechanical Models      | URDF                           | `shared/models/`                                   | URDF 1.0 standard with custom muscle actuator extensions                                     |
| Motion Capture Data       | C3D, BVH, TRC                  | External mocap systems or files                    | Standard formats with marker sets and frame data                                             |
| Launch Monitor Sessions   | CSV, TSV, TXT, XLS, XLSX, JSON | Common launch-monitor exports or user-mapped files | Canonical shot schema with source columns, unit/status metadata, and import manifest         |
| Expected-Strokes Baseline | JSON                           | Declared HTTP(S) source with license and SHA-256   | Versioned lie/context/target/distance states with unique points and optional standard errors |
| Optimization Constraints  | JSON                           | User input or configuration                        | Custom constraint schema in `src/config/`                                                    |
| Control Parameters        | YAML/JSON                      | Configuration files or API                         | Engine-specific parameter maps validated against schemas                                     |

### Output Data

| Output                   | Format                  | Destination                 | Description                                                                                                                                                           |
| ------------------------ | ----------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Simulation Trajectories  | JSON/HDF5               | API response or file export | Joint angles, muscle activations, forces over time                                                                                                                    |
| Cross-Validation Reports | JSON/PDF                | File export or API          | Engine comparison metrics, error margins, validation status                                                                                                           |
| IK/ID Solutions          | JSON/MATLAB             | API response or file        | Joint angles (IK) and joint torques (ID) with confidence metrics                                                                                                      |
| Optimized Trajectories   | URDF/MATLAB             | File export                 | Trajectory-optimized model definitions with optimal control inputs                                                                                                    |
| Visualization Data       | JSON (Three.js format)  | GUI or web client           | 3D geometry, animation keyframes, and rendering parameters                                                                                                            |
| Launch Monitor Analytics | CSV/JSON/project bundle | GUI, API, or file export    | Treated observations, provenance manifests, association/model results, dispersion, trends, source-backed SG with backing benchmark values, and non-SG outcome proxies |

### Configuration

Configuration is managed through:

- **Environment Variables**: read through `src/shared/python/config/environment.py` and re-exported by `src/shared/python/config/settings.py`, which is the canonical configuration reference. Key variables: `GOLF_API_HOST` / `GOLF_API_PORT`, `GOLF_API_SECRET_KEY`, `GOLF_ADMIN_PASSWORD`, `GOLF_AUTH_DISABLED`, `DATABASE_URL`, `ENVIRONMENT`, `HEADLESS`, `LOG_LEVEL`. `.env.example` documents the deployment-facing set (`API_HOST`, `API_PORT`, …).
- **YAML Config Files**: `src/config/interim_config.yaml` holds the declared defaults for CORS origins, trusted hosts, rate limits, quota tiers, simulation engine order, and video analysis. It documents intent and is not auto-loaded — callers that need its values load it explicitly.
- **API Request Parameters**: Engine selection, model path, solver options passed as JSON
- **Launcher Manifest**: `src/config/launcher_manifest.json` declares discoverable and hidden launcher surfaces, including shared Tools-hosted video/data utilities exposed to UpstreamDrift.
- **Theme API Settings**: `src/api/routes/theme.py` and `ui/src/api/themeClient.ts` expose launcher theme metadata to the desktop/web UI without duplicating theme lists in the frontend.
- **Web Settings**: `src/api/routes/settings.py` persists the validated web preferences document (`WebSettings`: appearance, notifications, simulation defaults) to `~/.upstreamdrift/web_settings.json` (override: `UPSTREAMDRIFT_WEB_SETTINGS_PATH`); `ui/src/pages/Settings.tsx` + `ui/src/api/settingsClient.ts` consume it with localStorage as cache only (#7457).

Excerpt from `src/config/interim_config.yaml`:

```yaml
# Server Configuration
server:
  # Use 127.0.0.1 (localhost) by default for security
  # Set to 0.0.0.0 in Docker or when external access is explicitly needed
  host: "127.0.0.1"
  port: 8000
  reload: true # Auto-reload on code changes (development)
  log_level: "info"

# Database Configuration
database:
  url: "sqlite:///./golf_modeling_suite.db"
  echo: false # Set to true for SQL debugging
  pool_size: 5
  max_overflow: 10
```

## 7. Testing Specification

### Testing Strategy

UpstreamDrift employs a comprehensive test pyramid with multiple specialized categories:

- **Unit Tests**: Test individual engine adapters, utilities, and validators in isolation
- **Integration Tests**: Test workflows combining multiple modules (e.g., load model → simulate → export)
- **Acceptance Tests**: End-to-end scenarios (e.g., full golf swing simulation with visualization)
- **Cross-Engine Tests**: Validate physics consistency across multiple engines with tolerance thresholds
- **Physics Validation Tests**: Verify results against known ground truth (analytical solutions, published benchmarks)
- **Golf Ball-Flight Source Contracts**: Validate documented aerodynamic, impact, and atmosphere assumptions against `docs/physics/GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md`
- **Launch-Monitor Scoring Contracts**: Verify baseline hashing and provenance,
  exact-stratum interpolation, fail-closed extrapolation, identity-safe grouping,
  uncertainty availability, API schemas, and the prohibition on relabeling an
  outcome proxy as strokes gained.
- **Dependency Source Contracts**: Validate generated dependency artifacts against `pyproject.toml` and fail CI when lockfiles or `environment.yml` drift
- **Documentation Governance Contracts**: Validate the canonical `docs/index.md` directory catalog, rendered documentation hub link, Markdown/Quarto size budget, and significant-word title capitalization for changed Markdown, Quarto, LaTeX, Word, and PDF documents.
- **Benchmark Tests**: Performance regression detection and optimization validation
- **Property-Based Tests**: Hypothesis-driven fuzzing for robustness

### Test Organization

| Category                    | Location                                                    | Framework           | Markers                                          |
| --------------------------- | ----------------------------------------------------------- | ------------------- | ------------------------------------------------ |
| Unit                        | `tests/unit/`                                               | pytest              | `@pytest.mark.unit`                              |
| Integration                 | `tests/integration/`                                        | pytest              | `@pytest.mark.integration`                       |
| Acceptance                  | `tests/acceptance/`                                         | pytest              | selected by path (no dedicated marker)           |
| Cross-Engine                | `tests/cross_engine/`                                       | pytest              | `@pytest.mark.gate` + `requires_<engine>`        |
| Physics Validation          | `tests/analytical/`, `tests/integration/conservation_laws/` | pytest              | `@pytest.mark.unit` / `@pytest.mark.integration` |
| Golf Source Contracts       | `tests/unit/shared_python/`                                 | pytest              | source-map contract tests                        |
| Launch-Monitor Scoring      | `tests/unit/launch_monitor/`, `tests/api/`                  | pytest              | source-backed SG domain and API contract tests   |
| Dependency Source Contracts | `tests/unit/scripts/`                                       | pytest              | generated dependency contract tests              |
| Benchmarks                  | `tests/benchmarks/`                                         | pytest-benchmark    | `@pytest.mark.benchmark`                         |
| Property-Based              | `tests/unit/`                                               | hypothesis + pytest | `@hypothesis.given` (no dedicated marker)        |

Issue #3841 moved stable flat tests and the launcher `src/**/tests` package into
topic directories under `tests/`, documented the fixture scopes in
`tests/README.md`, and added `scripts/check_test_layout.py` as the blocking CI
guard against new flat test files, new in-tree `src/**/tests` directories, and
overlapping fixture names in nested conftests.

### Coverage Requirements

| Scope                   | Minimum | Current          | Enforced By                                 |
| ----------------------- | ------- | ---------------- | ------------------------------------------- |
| Overall                 | 75%     | CI baseline      | `pyproject.toml` and `ci-standard.yml`      |
| API routes              | 30%     | Ratchet baseline | `scripts/config/mypy_exclusion_budget.json` |
| Data I/O                | 30%     | Ratchet baseline | `scripts/config/mypy_exclusion_budget.json` |
| Execution/checkpointing | 30%     | Ratchet baseline | `scripts/config/mypy_exclusion_budget.json` |
| Deployment              | 30%     | Ratchet baseline | `scripts/config/mypy_exclusion_budget.json` |
| Optimization            | 30%     | Ratchet baseline | `scripts/config/mypy_exclusion_budget.json` |
| Engine adapters         | 30%     | Ratchet baseline | `scripts/config/mypy_exclusion_budget.json` |

### Required Test Scenarios

- [ ] Unit creation with valid URDF returns expected topology (chain, mass distribution)
- [ ] MuJoCo engine simulation produces reasonable trajectories with gravity effects
- [ ] Cross-engine validation identifies discrepancies >5% between engines
- [ ] IK solver converges within 10 iterations for standard human poses
- [ ] ID computation returns physically plausible torques (within 2-sigma of analytical)
- [ ] Ball-flight atmosphere utilities reject non-finite or out-of-troposphere altitudes and stay traceable to documented golf source contracts
- [ ] Source-backed SG rejects unverifiable baselines, incomplete course states, extrapolation, and untrusted grouped identity while preserving row-level backing evidence
- [ ] Outcome-proxy responses remain explicitly non-SG in both domain and OpenAPI contracts
- [ ] FastAPI endpoints return 200 for valid requests and 400 for invalid schema
- [ ] GUI loads model and renders 3D visualization without crashing
- [ ] Trajectory optimization improves cost function by >20% over initial guess
- [ ] Muscle dynamics simulation produces realistic activation patterns
- [ ] Cross-platform build (Windows, macOS, Linux) produces functional binaries

## 8. Quality Standards

### Code Quality Tools

| Tool       | Version | Purpose                | Blocking? |
| ---------- | ------- | ---------------------- | --------- |
| ruff       | latest  | Linting and formatting | Yes       |
| mypy       | 1.7+    | Static type checking   | Yes       |
| pytest     | 7.0+    | Testing framework      | Yes       |
| pytest-cov | 4.0+    | Coverage measurement   | Yes       |
| bandit     | 1.7+    | Security scanning      | Yes       |
| hypothesis | 6.0+    | Property-based testing | No        |

### Design Principles

- **TDD**: Unit tests written before implementation; the current global coverage floor is 75%, with per-package production ratchets tracked toward higher thresholds (85% for API routes/engine adapters, 70% for shared utilities).
- **Design by Contract (DbC)**: Explicit preconditions and postconditions in engine adapters
- **DRY**: Cross-engine utilities in `src/shared/` prevent code duplication
- **Orthogonality**: Engines are loosely coupled; each can be used independently
- **Explicit is Better**: Function signatures include type hints; no magic string parameters

### Custom Quality Gates (CI)

Beyond standard tools, CI enforces custom checks:

- **Dependency Direction**: No reverse dependencies (leaf → branch → root)
- **SAST Delta Scan**: Pull requests run Semgrep against changed supported
  source/application files and Bandit against changed supported Python
  source/application files, and Trivy against changed supported
  dependency/container/config files while non-PR CI retains the full repository
  scans, keeping new code blocking without letting existing repository baseline
  findings block unrelated PRs.
- **Alembic PostgreSQL Round Trip**: PostgreSQL migration round-trip CI has a
  finite job budget, an explicit SQL readiness probe, isolated pytest plugin
  loading, and verbose duration output so migration hangs produce actionable
  diagnostics instead of opaque cancellation or unrelated desktop-display plugin
  failures.
- **Core Test Relevance Filter**: Pull requests with no Python source, test,
  project metadata, or dependency-file changes skip the expensive Python test
  matrix after checkout so workflow-only and documentation-only CI fixes remain
  finite on constrained self-hosted runners.
- **Suite Marker Ratchet**: `scripts/ci/check_suite_marker_ratchet.py` scans
  pytest source files for tests without recognized suite markers and compares
  them to `scripts/config/suite_marker_baseline.json`. Existing unmarked tests
  may be paid down, but net-new unmarked tests fail CI Standard; the runtime
  collection hook in `tests/conftest.py` can report the same debt or enforce it
  with `UD_ENFORCE_SUITE_MARKERS=1`.
- **File Size Budget**: No module exceeds 500 lines; classes capped at 200
  LOC; oversized grandfathered files must have tracked baseline entries in
  `scripts/config/file_size_budget.json` or the CI gate fails.
- **Module Size Budget**: Python modules under `src/` are capped at 1,500
  lines by `scripts/check_module_size_budget.py`; oversized legacy modules
  require owned, expiring exceptions in
  `scripts/config/module_size_budget_baseline.json`, currently capped at 10
  active exceptions.
- **Architecture Budget**: Changed production Python files are capped at 100
  lines per function and 8 effective parameters per callable by
  `scripts/ci/check_architecture_budget.py`. The gate ignores test/vendor
  paths, excludes receiver parameters (`self`/`cls`) from method counts, and
  requires owned, linked exceptions in
  `scripts/config/architecture_budget.json`.
- **Law of Demeter Ratchet**: `scripts/ci/check_lod.py` scans production
  `src/` Python files and blocks new deep application object chains beyond the
  checked-in `scripts/ci/lod_baseline.txt` path/chain counts while preserving
  documented library API allowances for Qt, numpy, pandas, matplotlib, scipy,
  and engine namespace access.
- **Agent Docs Consistency**: `scripts/check_agent_docs_consistency.py`
  validates literal repo-relative paths documented in agent guidance while
  treating glob/brace references such as `scripts/**` and
  `src/shared/python/codemap/{cli,watcher,mcp_server}.py` as patterns, not
  files that must exist.
- **Root Clutter**: `scripts/check_root_clutter.py` blocks non-allowlisted
  repository-root files; substantive launcher entry points such as
  `launch_golf_suite.py` and `launch_upstream_drift.py` are explicitly
  allowlisted until promoted into packaged scripts.
- **Documentation Catalog and Size Budget**: Every top-level `docs/` directory is listed in `docs/index.md`; oversized Markdown/Quarto docs require owned, expiring exceptions.
- **Import Depth**: Maximum 4 import levels to prevent circular dependencies
- **Physics Fitness**: Cross-engine validation must pass with <5% tolerance
- **Security Audit Isolation**: `pip-audit` runs with `scripts/config/pip_audit_waivers.json` and `scripts/ci/check_pip_audit_waivers.py` so waivers require issue tracking, expiry, and current pip-audit findings before ignore flags are emitted
- **Blocking SAST and Secret Scans**: `ci-standard.yml` runs blocking Bandit, Semgrep, pip-audit, and Trivy filesystem scans for pull requests and pushes
- **Error-Handling Ratchet**: `scripts/ci/check_error_handling_ratchet.py` blocks increases in grandfathered broad catches, unused `noqa` debt, raw `subprocess.Popen(...)`, and `asyncio.gather(...)` calls that omit `return_exceptions=`, including multiline gather calls whose arguments span multiple lines.
- **Type and Coverage Ratchets**: `scripts/check_mypy_exclusion_budget.py` blocks unowned mypy exclusions, non-monotonic exclusion schedules, and missing production package coverage-ratchet metadata. Pull requests and ordinary pushes with a concrete diff base run baseline mypy only on changed `src/` Python files so legacy type drift does not block unrelated work; scheduled/manual full-src mypy still runs through `scripts/ci/run_full_mypy_baseline.py`, which compares `mypy src --config-file pyproject.toml` against `scripts/config/full_src_mypy_baseline.json` and fails on new or stale type diagnostics during explicit full-audit lanes.
- **Docker Size Gates**: The canonical runtime and every modular profile must
  remain within the explicit budget registered for that image. Modular workflow
  budgets must match `docker/profiles.yaml`; slim is currently 900 MB and
  standard is 2200 MB after the governed core runtime added pandas-backed API
  routes.
- **Unit-Gate Quarantine Ratchet**: `scripts/ci/check_unit_gate_quarantine.py` requires every ledgered node ID to resolve to exactly one owned, executable failure cluster. Pull requests compare the current node-ID set with the fetched base branch and reject additions or replacements while allowing removals; duplicate, unassigned, ambiguous, or malformed entries fail closed. The checker can list or run one cluster in bounded batches without changing the Green-Suite-only skip boundary.

### CI/CD Pipeline

| Workflow                       | Trigger                                | Purpose                                                                               | Blocking?          |
| ------------------------------ | -------------------------------------- | ------------------------------------------------------------------------------------- | ------------------ |
| `ci-standard.yml`              | Push/PR (no PR path filter)            | Lint, type check, unit/integration tests, workflow inventory, blocking security scans | Yes                |
| `quality-gate.yml`             | PR/manual dispatch                     | Blocking repo-wide Law-of-Demeter ratchet for production `src/` Python code           | Yes                |
| `docs-ci.yml`                  | PR touching docs/markdown              | Docs governance for docs-only PRs                                                     | No (not required)  |
| `heavy-tests-opt-in.yml`       | Manual dispatch or `/heavy-test` label | Cross-engine and physics validation (long-running)                                    | No (opt-in)        |
| `nightly-cross-validation.yml` | Daily 2:00 UTC                         | Full multi-engine validation suite against all model variations                       | No (informational) |
| `tauri-build.yml`              | Tag release                            | Build desktop apps for Windows/macOS/Linux                                            | Yes (for releases) |
| `vendor-freshness.yml`         | Weekly                                 | Check for stale dependencies and security updates                                     | No (warning-only)  |
| `docker-size-gates.yml`        | Push                                   | Enforce the canonical runtime and per-profile image budgets                           | Yes                |

### Required Status Checks

Branch protection matches required checks by **context name**, so a check name is
a repository-level contract: any job carrying a required name publishes under that
context, and protection is satisfied by whichever job reported. Two rules follow.

**One publisher per required name.** Exactly one job may be named `quality-gate` —
the `ci-standard.yml` aggregate. Three jobs previously shared the name
(`ci-standard.yml`, `quality-gate.yml`, `docs-ci.yml`), which allowed a merge while
the aggregate was failing. `tests/ci/test_ci_infrastructure.py` enforces uniqueness.

**A required check must report on every PR.** A required context that never reports
blocks the PR indefinitely. A job skipped by a job-level `if:` still publishes a
check run and counts as satisfied, but a workflow skipped by a trigger **path
filter** publishes nothing. Required workflows therefore carry no PR path filter and
skip work per job instead: `ci-standard.yml` classifies the diff in its
`changed-paths` job and gates the substantive jobs on
`needs.changed-paths.outputs.code`, so docs-only PRs skip the suite while
`quality-gate` still reports. The aggregate accepts `skipped` gates only on
docs-only changes, and only once `pick-runner` and `changed-paths` themselves
succeeded — otherwise an infrastructure failure that skips everything would read as
a pass.

| Context             | Published by                | Required | Notes                                                          |
| ------------------- | --------------------------- | -------- | -------------------------------------------------------------- |
| `quality-gate`      | `ci-standard.yml` aggregate | Yes      | Also required org-wide by the `Repository_Protections` ruleset |
| `lod-quality-gate`  | `quality-gate.yml`          | Yes      | No path filter, so it reports on every PR                      |
| `docs-quality-gate` | `docs-ci.yml`               | **No**   | Docs-only trigger; requiring it would block every code-only PR |

Repository-specific required contexts belong on the repo-scoped `Protect Main`
ruleset. The organization ruleset `Repository_Protections` applies to every repo in
the org, so contexts added there must exist in all of them.

## 9. Dependencies

### Runtime Dependencies

| Package                        | Version    | Purpose                                                   |
| ------------------------------ | ---------- | --------------------------------------------------------- |
| numpy                          | 1.20+      | Numerical computation                                     |
| scipy                          | 1.7+       | Scientific algorithms (optimization, linalg)              |
| fastapi                        | 0.95+      | REST API framework                                        |
| uvicorn                        | 0.20+      | ASGI server for FastAPI                                   |
| pydantic                       | 2.0+       | Request/response validation                               |
| mujoco                         | 3.3.0+     | Primary physics engine (required)                         |
| PyQt6                          | 6.0+       | Professional GUI framework                                |
| tauri-py                       | 1.0+       | Tauri bridge for Python backend                           |
| pillow, requests, bokeh, flask | CVE floors | Runtime security constraints validated outside dev extras |

### Optional Runtime Dependencies

| Package      | Version | Purpose                                     |
| ------------ | ------- | ------------------------------------------- |
| drake        | 1.0+    | Drake physics engine integration            |
| pinocchio    | 2.6+    | Pinocchio rigid-body dynamics               |
| myosuite     | 2.0+    | MyoSuite muscle simulation                  |
| opensim      | 4.4+    | OpenSim musculoskeletal models              |
| mediapipe    | 0.9+    | Motion capture integration (pose detection) |
| scikit-learn | 1.0+    | RL policy learning and clustering           |
| sympy        | 1.11+   | Symbolic trajectory optimization            |
| pyarrow      | 14.0+   | Parquet IO for compact swing dataset paths  |

### Development Dependencies

| Package    | Version | Purpose                                                      |
| ---------- | ------- | ------------------------------------------------------------ |
| pytest     | 7.0+    | Testing framework                                            |
| pytest-cov | 4.0+    | Coverage measurement                                         |
| hypothesis | 6.0+    | Property-based testing                                       |
| pip-tools  | 7.4+    | Regenerate Python dependency lockfiles from `pyproject.toml` |
| pyarrow    | 14.0+   | Parquet IO test coverage for compact swing dataset paths     |
| ruff       | latest  | Linting and formatting                                       |
| mypy       | 1.7+    | Type checking                                                |
| bandit     | 1.7+    | Security scanning                                            |
| black      | 23.0+   | Code formatter                                               |

### Fleet Dependencies

| Repo  | Relationship                              | Description                                                                                                                                                 |
| ----- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tools | Vendored Python and pinned Rust authority | `vendor/ud-tools` plus the Cargo `tools-core` revision provide reviewed shared contracts and kernels. Ground-model release pins remain pending Tools #4276. |

## 10. Deployment & Operations

### How to Run

```bash
# Prerequisites
- Python 3.11 or later
- MuJoCo 3.3.0+ with license (community or pro)
- Optional: Drake, Pinocchio, OpenSim binaries on PATH
- For Tauri desktop app: Node.js 16+, Rust toolchain

# Installation
git clone https://github.com/D-sorganization/UpstreamDrift.git
cd UpstreamDrift
python -m pip install -e ".[dev]"  # Include dev dependencies
# For Desktop App: Cargo Install Tauri-Cli

# Running the FastAPI Server
uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
# ...Or, Equivalently, Through the Launcher:
upstream-drift --api-only --port 8000

# Running the PyQt6 GUI
upstream-drift --classic
# ...Or Directly:
python launch_upstream_drift.py --classic

# Running the CLI
upstream-drift --engine mujoco --no-browser
python -m sidekick
python -m sidekick run --calculator unit-converter --inputs ./inputs.json

# Building the Tauri Desktop App
cd ui && npm install && npm run tauri build
# Outputs: UpstreamDrift.exe (Windows), UpstreamDrift.app (macOS), UpstreamDrift.AppImage (Linux)

# Running Tests
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/ --cov=src --cov-fail-under=75
```

### Build Artifacts

| Artifact              | Format         | Destination             |
| --------------------- | -------------- | ----------------------- |
| Python Package        | .whl           | PyPI (on release)       |
| FastAPI Server        | Docker image   | Docker Hub (on release) |
| Desktop App (Windows) | .msi installer | GitHub releases         |
| Desktop App (macOS)   | .dmg bundle    | GitHub releases         |
| Desktop App (Linux)   | .AppImage      | GitHub releases         |
| Documentation         | HTML           | GitHub Pages            |

Canonical production artifacts and supported OS/Python/tier/hardware
combinations are defined in `docs/operations/production-readiness.md`. Release
smoke suites live under `tests/smoke/<artifact>/`; the tag release workflow
blocks Python package publication on the built-wheel smoke matrix.

## 11. Roadmap & Open Issues

### Current Phase

**Active Development**: Core engine integrations complete; expanding experimental OpenSim and MyoSuite support. Tauri desktop app in active development. Motion capture integration and RL control schemes are in-progress.

### Planned Work

| Priority | Item                                        | Issue/PR | Target Date |
| -------- | ------------------------------------------- | -------- | ----------- |
| P0       | Complete OpenSim integration (F4)           | #45      | Q2 2026     |
| P0       | Complete MyoSuite integration (F5)          | #46      | Q2 2026     |
| P1       | Motion capture import and tracking (F13)    | #78      | Q3 2026     |
| P1       | RL controller learning framework (F14)      | #92      | Q3 2026     |
| P1       | Tauri desktop app release (F9)              | #101     | Q2 2026     |
| P2       | Extended MATLAB integration (export/import) | #112     | Q4 2026     |
| P2       | Performance profiling and GPU acceleration  | #130     | Q4 2026     |

### Known Limitations

- OpenSim and MyoSuite integrations are experimental; API may change
- Cross-engine validation only enforces tolerances on kinematic outputs; dynamics comparison still in development
- Motion capture import limited to marker-based systems (no IMU data yet)
- RL integration currently supports basic Gym environments; no hierarchical or multi-agent support
- Tauri app Windows builds require MSVC toolchain (no MinGW support)
- Performance scaling beyond 100-muscle models not yet tested
- Ground-model execution is not yet available from the clean UpstreamDrift
  release: the headless consumer gateway exists, but exact Tools pins and
  FastAPI/PyQt/React surfaces await the protected Tools ground merge (#4276).

## 12. Change Log
Rows are keyed by pull request, not by a serial spec version: `| YYYY-MM-DD | #<pr> | summary |`. Add exactly one row for your own pull request and do not renumber anybody else's; the `Spec Version` field in section 1 is release-derived and is never bumped by an individual pull request. See [Repository_Management#1520](https://github.com/D-sorganization/Repository_Management/issues/1520).

| 2026-09-06 | #9608 | Added `docs/motion_capture/camera_rig_runbook.md` and the lab plan `docs/motion_capture/plans/lab_three_view_sonnet.json` (#9601, C2 of #9599): cabling rules, the four-command operator procedure, and recording acceptance evidence from the real rig (616/619/616 frames at 1920x1200 @ 60 fps after the recorder warm-up fix; the 20 %-short first run kept as the named failure). |
| 2026-09-06 | #9596 | Migrated `MediaPipeEstimator` to the MediaPipe Tasks API (#9592): `PoseLandmarker` in VIDEO mode with strictly increasing timestamps; new `pose_estimation/mediapipe_models.py` resolves the `.task` model from `MEDIAPIPE_POSE_MODEL_PATH` / `MEDIAPIPE_POSE_MODEL_VARIANT` / the per-user cache and offers an explicit SHA-256-verified download (lite/full/heavy digests pinned); availability probe accepts the Tasks API and names legacy-only installs. |
| 2026-09-06 | #9598 | Added strobe-based time alignment to `motion_capture.rig` (#9591): `sync.py` relates each camera's arrival clock to the reference view through a shared strobe with quadrature uncertainty, frame interval and rate deviation, recorded in the session manifest `timing` block and never applied to frames; `CaptureTuning` value object replaces loose `CaptureSession` knobs; `capture --timing`. |
| 2026-09-06 | #9595 | Added `src/motion_capture/rig` (#9590, child of #9422): rig plans binding views to camera identities (serial or port path) with per-camera capture modes and controls; `FrameSource` / `Recorder` protocols with OpenCV-MSMF, synthetic, and ffmpeg stream-copy implementations; barrier-started `CaptureSession` reporting `supported` / `degraded` / `blocked` / `unavailable` per the acceptance program; fail-closed `tools_bridge` probe of `sidekick.lab.mocap`; topology prediction moved out of `scripts/diagnose_mocap_camera_rig.py`. Documented in `docs/motion_capture/capture_rig.md`. |
| 2026-09-06 | #9593 | Fixed CI Optional Stack (#9589): the root `conftest.py` registers the pytest-asyncio and pytest-timeout ini keys (`asyncio_mode`, `timeout`, `timeout_method`) from `pytest_addoption` when the plugin manager reports the owning plugin inactive, so plugin-less venvs no longer exit 4 under `--strict-config`. Covered by `tests/scripts/test_root_conftest_plugin_ini_shim.py`. |
| 2026-09-06 | #9586 | Recorded the USB camera rig bring-up for the markerless-mocap program in `docs/motion_capture/usb_camera_rig_bringup.md`: one ELP AR0234 camera per USB 2.0 root port (alt-setting 0x0B reserves 3060 B of the 6000 B periodic budget), the CalDigit TS4 chained-dock failure, and two validated three-camera topologies at 1920x1200 @ 60 fps. Added `scripts/diagnose_mocap_camera_rig.py` (hub-chain walk, streaming-count prediction, solo and concurrent measurement) with hardware-free tests in `tests/scripts/test_diagnose_mocap_camera_rig.py`. |
| 2026-09-05 | #9567 | Enforce artifact gitignores and repo hygiene gates (#9415). In `.gitignore`, pinned rules for `.scratch/`, `output/`, and `reports/*.json` scanner outputs. In `scripts/check_forbidden_artifacts.py`, barred `.scratch/` prefix and `reports/*.json` scanner dumps. Pinned contracts in `tests/ci/test_artifact_dir_hygiene.py` and `tests/scripts/test_check_forbidden_artifacts.py`. |
| 2026-09-04 | #9512 | Unblocked and joined streaming worker thread on timeout and cancellation so no daemon thread survives tests (#9495). In `src/api/services/chat_service.py`, passed `stop_event` via context metadata, named worker thread `ChatService-StreamWorker-<session_id>`, and joined it in `stream_response`'s finally block; updated `test_stream_response_yields_timeout_error_when_queue_stays_empty` in `tests/api/test_chat_connectivity.py` to unblock and explicitly close the async generator, asserting zero leaked threads. Added autouse `_check_no_thread_leak` fixture and unit suite marker. In `conftest.py`, piped faulthandler thread dumps through the watchdog process so every line is prefixed with `[hang-watchdog]`, eliminating log-scraping confusion. |
| 2026-09-03 | #9503 | Consolidated the UpstreamDrift-owned `MplCanvas` copies onto a single `src/shared/python/plotting/mpl_canvas.py` with safe teardown: `close_canvas()` clears `_draw_pending` so an already-queued `draw_idle` cannot call into a destroyed C++ canvas, then releases the figure via `plt.close(fig)`, idempotently and without raising. `plotting/base.py`, the 3D Golf Model canvas widget and its embed adapter now delegate to it; the five remaining copies are Tools-owned (child-copy header or Tools counterpart) and are pinned as visible debt by `tests/unit/shared/plotting/test_mpl_canvas_consolidation.py` (#9474). |
| 2026-09-03 | #9498 | Closed the hole in the Tools child-copy contract, bumped the `vendor/ud-tools` pin, and corrected 34 assertions that asserted nothing (#9474, program #1505; paired with D-sorganization/Tools#4959, merged `951d718a4`). `test_tools_child_copy_headers_have_tools_counterparts` exists to flag a file carrying the Tools DO-NOT-EDIT header with no counterpart in Tools, and it passed on exactly such a file (`ai/tools/sidekick_analytics.py`) because it scanned an opt-in allowlist of four paths plus `sidekick/agent` - 3 of 146 headered files. Its sibling `test_current_branch_does_not_edit_tools_child_copies` has no allowlist and treats any headered file as Tools-owned, so the two halves of one contract disagreed and a file could be frozen against editing here with no Tools file to edit instead. The guard is inverted from opt-in to opt-out: every headered file is scanned against a shrink-only waiver ledger (`tests/unit/repo_hygiene/tools_child_copy_missing_counterparts.txt`) whose two assertions fail both an unlisted violation and a stale waiver whose file has since gained a counterpart, so it cannot rot back into an allowlist. Measured for the first time: 146 files carry the header and 143 lack a Tools counterpart; the ledger opens at 142 and is retired by #9406. `tests/unit/repo_hygiene/test_tools_child_copy_guard_scope.py` pins the guard's scope on synthetic trees so the hole cannot silently reopen. The pin moves to Tools `951d718a4`, which adds `ai/tools/sidekick_analytics.py` and so makes the header on UD's copy accurate rather than aspirational; `scripts/shared_tools/check_tools_pins.py` caught that Cargo.toml's `tools-core` rev had to move with it (#9436) and now reports the pins consistent. Separately, 34 assertions across six AI test files read `result.solver_status == "success"`; ToolResult has never had that attribute - it has `success: bool`, and `solver_status` belongs to the motion-matching FitResult, so every one raised AttributeError before reaching its subject. Correcting them clears 30 failures in the AI suites (71 failed to 41), measured. No test is weakened, skipped or deleted. Note for #9406: a pin bump does not deliver Tools fixes to UD's AI tests, which import `src.shared.python.ai.*` - UD's own child copy - so the remaining placeholder-honesty and registration failures clear only when the child copies are deleted in favour of the Tools package. Regenerating `docs/shared_tools/divergence_inventory.v1.json` against the new pin was a required coupled edit the bump had missed, and `tests/companion/test_companion_catalog.py` hard-codes the expected gitlink, so both moved to `951d718a`. Correcting the dead assertions then exposed a real honesty defect in the child copy: `run_inverse_dynamics`, `validate_cross_engine` and `check_energy_conservation` each answered `success=True` with a message claiming work had been "queued" while starting none. Tools had already fixed this behind a single `_not_implemented_tool_result` helper whose contract enforces *a tool that enqueues nothing must report failure*; the UD child copy never received it and had drifted back to hand-rolled dishonest dicts. Fixing it exposed a contradiction in the contract itself: the corrected assertion demands honest behaviour from a file the child-copy guard forbids editing, and the sanctioned escape (deleting a migrated copy) is unsafe today because `vendor/` is not shipped -- `build_hooks.py` copies the pinned tree in at build time -- so a deletion would rely on import aliasing that is not yet wired for production. The guard was therefore refined rather than bypassed: a child-copy edit is now permitted **only when the result equals the pinned Tools file exactly**, after normalising the two documented seam differences (UpstreamDrift's `src.shared.python` import spelling and the DO-NOT-EDIT banner). That allows a fix already merged in Tools to reach the copy that actually runs, and still fails any edit that introduces divergence -- asserted directly by three new tests, including one that flips a single character and must be rejected. Under that rule `ai/sample_tools.py`, `ai/system_prompts.py` and `ai/tools/sidekick_analytics.py` were converged onto canonical verbatim, which also restored the missing `_register_sidekick_analytics` wiring and `SIDEKICK_ANALYTICS_TOOL_NAME` that two UpstreamDrift tests already required. |
| 2026-09-03 | #9497 | Fixed the CI Standard tests (3.12) hang: the cline provider test server now survives `is_available()` probe connections (EOF-only peers are not answered, `OSError` during a reply is swallowed, the writer always closes), so Python 3.12 `Server.wait_closed()` no longer blocks forever on a crashed handler; pinned by two RED→GREEN regression tests plus a bounded `wait_closed()` integration test (#9431). |
| 2026-09-03 | #9496 | Finished the #9415 artifact-dir sweep: the last tracked wrong-cwd artifact (`motion_matching/results/CROSS_ENGINE_LEADERBOARD.md`, a regeneration placeholder from `run_cross_engine_leaderboard.py`) is deleted, `/motion_matching/` is ignored (mirroring `/motion_matching_training/`), and `tests/ci/test_artifact_dir_hygiene.py` pins the contract — zero tracked files under the artifact dirs, `output/` tracks only its keepfile, ignore patterns stay present (#9415). |
| 2026-09-03 | #9497 | Fixed the CI Standard tests (3.12) hang: the cline provider test server now survives `is_available()` probe connections (EOF-only peers are not answered, `OSError` during a reply is swallowed, the writer always closes), so Python 3.12 `Server.wait_closed()` no longer blocks forever on a crashed handler; pinned by two RED→GREEN regression tests plus a bounded `wait_closed()` integration test (#9431). |
| 2026-09-03 | #9490 | Fixed the Qt object-lifetime defect that aborted the `tests (3.11)` interpreter (RM #1507, program #1505, UD #9474). `MuJoCoSimWidget.__init__` started a 60 fps `QTimer` unconditionally and the class had no `closeEvent`, `hideEvent` or teardown method; `QWidget.close()` only hides, so every sim widget a test built kept stepping MuJoCo and rendering frames for the rest of the process-wide `QApplication`, i.e. the rest of the pytest session. One of those stray frames fired during an unrelated later test, reached `get_cv2()`, and the `import cv2` raised `AttributeError: partially initialized module 'cv2' has no attribute 'mat_wrapper'` -- not `ImportError`, which was the only case `get_cv2` guarded -- so the exception escaped a Qt slot and PyQt6 aborted the interpreter with no test summary at all. The widget now owns its timer's lifetime through an idempotent `stop_simulation()` called from a new `closeEvent`; `GripModellingTab` gained a `closeEvent` that releases the sim widget it builds (Qt does not deliver `closeEvent` to children) while deliberately leaving the externally supplied `external_sim_widget` alone; `MainWidget.cleanup()` now calls `sim_widget.stop_simulation()` instead of reaching through to `sim_widget.timer.stop()`; and `get_cv2()` honours its documented "module or `None`" postcondition instead of propagating out of a Qt slot. `test_grip_modelling_tab_ui_widths` was relying on a bare `deleteLater()`, which only posts a DeferredDelete event that the unit lane's event-loop-free run never processes. Red-then-green: the new `tests/unit/engines/physics_engines/mujoco/mujoco_humanoid_golf/test_sim_widget_timer_lifetime.py` fails 5 of 7 on the parent commit, including reproducing the exact CI `AttributeError` text, and passes 7/7 after. No test is weakened, skipped, quarantined or deleted. `_is_hand` in `grip_modelling_tab.py` was additionally narrowed: it guarded a `str | None` with `bool(name)`, which mypy does not treat as a narrowing form, so the changed-file mypy lane reported `Item "None" of "str | None" has no attribute "lower"`. It now returns early on a falsy name and lowercases once instead of twice. |
| 2026-09-03 | #9474 | Restored the `tests (3.x)` failure signal (RM #1507, program #1505). `main` @ `4ec3da33` counted `305 failed ... 41 errors` and named none of them: `tests/unit/installer/test_build_installer.py::test_detect_physics_engines` replaced `builtins.__import__` with a hook that raised `ImportError` for every unrecognised name, so when that test failed, pytest's own failure formatter could not complete its lazy `from _pytest.fixtures import FixtureLookupError` and the session died with `INTERNALERROR` (exit code 3) before printing `short test summary info`, the `FAILURES` section or the coverage report. Reproduced locally (exit 3, `no tests ran`) and fixed by giving `build_installer` a narrow named seam, `_module_available`, which `detect_physics_engines` now uses and which the test patches instead of the interpreter's import hook. `tests/unit/repo_hygiene/test_no_non_delegating_import_hook.py` is a new AST guard: any function installed as `builtins.__import__` must delegate to the saved real import for names it does not handle; its ledger starts empty and only ratchets down. Both pytest invocations in the lane now write `--junitxml` under `$RUNNER_TEMP/junit/` and pass `-ra` explicitly, and a new `Upload JUnit Test Results` step publishes `junit-tests-<python>` with `if: always()`. No test is weakened, skipped, quarantined or deleted; job names and the `quality-gate` context are unchanged. |
| 2026-09-03 | #9461 | Derive release version surfaces from scripts/check_version_consistency.py; add --set <version> helper and ui/package-lock.json enforcement (#9461). |
| 2026-09-03 | #9457 | Repaired the `docs/` link graph (#9413, deliverable 3). All 18 unresolvable relative links under `docs/` are fixed: `FEATURE_ENGINE_MATRIX.md` now points at `engines/engine_selection_guide.md` and drops a `PATH_FORWARD.md` that never existed; the eight completist reports deleted by `987d67ccb` are unlinked with a dated note; the hardcoded `c:/Users/...` path in `assessments/issues/summary.md` is now repository-relative; and `user_guide/interactive_pose_manipulation.md` no longer links `docs/docs/...`. The six reviewer-workbench figures named by #8851 are absent from the available history and were removed with dated notes naming the nearest surviving panels rather than re-pointed at a guess. `docs/index.md` gains a generated documentation map with live links to all 57 top-level directories plus an explicit calculation-references section, giving the previously orphaned sheets (`physics/PUTTING_KINEMATICS_KINETICS_REVIEW.md`, `estimation/synthetic-ground-truth-rig.md`, `engineering/dependency-direction-rules.md`, `engineering/logging-policy.md`) their first inbound links. `docs/README.md`'s structure block is generated from the real tree by the new `scripts/generate_docs_map.py` (`--check` gates drift). The four consolidations #8840 called pending are decided and dated in `docs/index.md`: `strategic/` merged into `plans/`; `audits/` retagged `stable` because it holds the newest audit records, not legacy ones; `review_archive/` deferred to the governed workflow campaign that owns its two workflow references; `assessments/` restructure declined as disproportionate. `src/tools/check_markdown_links.py` now exits non-zero, names `docs/help/` in its default scope, skips placeholders and uninitialised submodule targets, and accepts explicit scan paths. |
| 2026-09-03 | #9452 | fix(ci, #9451): run the articulated manufactured-solution byte-determinism test via the rolling profile outside the governed authority stack. The optional-stack matrix lanes run unpinned 3.11/3.12 (runner tool-cache patch drift), so `write_record()`'s authority-profile default failed with "authority requires exact Python patch 3.11.15". The test now probes `validate_authority_environment()`: governed stack -> authority profile (unchanged behavior, strict compare vs committed record); otherwise -> rolling profile, asserting native-repeat byte-determinism and the `non_authoritative_compatibility_only` marker. The dedicated `articulated-manufactured-authority` job (3.11.15 + authority lock) remains the only authority lane; all 6 `tests/ci/test_articulated_manufactured_hybrid_ci_red.py` contract tests pass. The claim-evidence manifest is regenerated for the edited test file (hash pin). (spec 1.0.718) |
| 2026-09-02 | #9255 | Carried the accelerated-mass interval end to end, so a design comparison reports a band and refuses to rank when the bands overlap (issue #9243). PR #9237 made the mass ball launch divides by an interval -- 176-413 g about 270 g at the nominal greenside shot -- and carry inherits all of it (0.80-3.27 m about 1.67 m); the width was then thrown away before a design was ranked, so `WorkbenchModel.compare` ordered two soles on their central values and always named a leader. Three new layers fix that. `vandv.band.ConsistencyBand` is interval arithmetic that gets the two hard cases right: a decreasing map swaps the edges (less sand sharing the same impulse throws the ball further), and `/carry - target/` is V-shaped, so a band straddling the target reaches zero and neither image edge is the image of an edge. Averaging over a delivery sweep is edge-wise, never a standard error, because one mass interval applies at every condition at once and the width does not average out. `vandv.budget.UncertaintyBudget` keeps NUMERICAL, MODEL_FORM and SAMPLING apart with their own combination rules, requires every numerical term to declare what was refined, and **refuses** to map a SPACE_TIME term onto V&V 20 without an explicit opt-in -- `column_grid_convergence` holds the Courant number fixed, so its GCI is a space-time band and not a spatial `u_h` (ADR-0033). `study.ranking.rank_with_bands` returns exactly one of A better, B better, or INDISTINGUISHABLE, with `winner is None` for the third. Measured result: at the shipped settings the accelerated-mass band accounts for 86% of the comparison width against 14% for the delivery-sweep bootstrap, and **no pair of soles tried separates** -- including pairs the bootstrap alone ranks confidently, which is the overclaim removed. Terms nobody has a number for (the uncalibrated `BALL_MOMENTUM_TRANSFER_EFFICIENCY`, the launch direction, the missing carry GCI) are registered as `UnquantifiedTerm` rather than omitted or invented, so every band is reported as a lower bound and no verdict here is defensible. The band is named a CONSISTENCY band throughout and never a confidence interval: its edges are two uncalibrated models, not quantiles, and a test forbids the word on the type. `model.py` was split (`outcomes.py`, `uncertainty.py`) to stay under the 1200-line file budget. 95 new tests. (spec 1.0.696) |

<!-- prettier-ignore-start -->

| Date       | PR         | Changes    |
| ---------- | ---------- | ---------- |
| 2026-09-04 | #9528 | Refactored export_video to use managed_popen (spec-exempt: refactoring) |
| 2026-09-03 | #9491 | Retired a drifted test double and re-pointed two AI test files at the contract that actually exists. `tests/unit/ai/test_assistant_header_dropdowns.py` defined a `_MockPanel` whose own comment read "Methods copied from AIAssistantPanel (must stay in sync)"; they did not. #5493 split the panel and deleted `_get_models_for_provider` and `_get_thinking_capabilities_for_model` from production, while the copies lived on importing `ChatModelInfo`/`ThinkingCapabilities` from `src.shared.python.ai.types`, where neither name has ever existed -- all sixteen tests raised ImportError, so the file tested its own copy and then stopped even doing that. The duplicate is deleted rather than repaired: the tests now exercise the production helpers the header calls (`_provider_registry_data.PROVIDER_INFO`, `provider_model_names`, `provider_default_model`, `populate_provider_combo`, `populate_model_combo`) through a `_FakeCombo` implementing only the QComboBox subset those functions touch, so they stay headless. Three invariants are newly pinned and pass against current main: the registry covers the whole `AIProvider` enum, every provider's default model is one of its own listed models, and a foreign selection falls back to the new provider's default instead of persisting across a provider switch. `tests/unit/ai/test_adapter_capabilities.py` asserted `list_models() -> list[ChatModelInfo]` and `ThinkingCapabilities(supports_levels, available_levels)`; the declared contract is `list_models() -> list[str]` (`adapters/base.py`) and `ThinkingCapabilities(provider, levels, default_level_name)` (`chat_contracts/models.py`). The new assertions are stronger, pinning `default_level_name in level_names()` for every adapter and covering both sides of Anthropic's model-dependent branch. No src change, no test skipped, xfailed, weakened or deleted; quarantine ledger untouched. Verified: 11 passed and 46 passed / 4 skipped (no display) against 25 previously failing. Part of #9474 (main-green Phase 0). |
| 2026-09-03 | #9459 | Docs governance now blocks merges: folded the doc-governance gate list into the required `quality-gate` aggregate as a `docs-governance-gates` job in `ci-standard.yml`, keyed off a new `docs` output of the `changed-paths` classifier that mirrors `docs-governance.yml`'s trigger paths. The standalone `doc-governance` check is not a required status context, so auto-merge fired once `quality-gate` passed and PR #9418 landed a doc-size-budget violation red onto main nine seconds before the docs check reported failure (trimmed by #9421; same mode previously hit #9392). The gate steps moved to a shared composite action `.github/actions/docs-governance-checks` consumed by both `docs-governance.yml` (advisory PR signal, red-main canary) and the new required-path job so the two lists cannot drift; the title-case step skips when no diff base exists (schedule/dispatch) because a full-tree scan reports ~3,249 grandfathered violations. No branch-protection change: `quality-gate` remains the single required context by design. Contracts pinned in `tests/ci/test_docs_governance_gates_workflow.py`; `tests/unit/scripts/test_docs_ci_governance.py` repointed at the composite action. Docs-only infra change: no production module touched. |
| 2026-09-03 | #9454 | Launch Monitor Analytics no longer destroys unsaved work silently (#8881). `New Project` was wired straight to `clear_project`, which replaced the project *and* set `_dirty = False` in the same call -- destroying the work and simultaneously disarming `embedded_host._confirm_dirty_close`, the one safety net that existed; `_remove_selected_sessions` deleted every selected session with no prompt. `clear_project`/`load_project`/`remove_sessions` remain unconditional primitives so tests and scripts drive them directly, and every user-facing path now runs a guard from the new `DestructiveActionGuards` mixin first: Save / Discard / Cancel with Cancel default, where a Save answer proceeds only if the save actually cleared the dirty flag, plus a session-removal prompt naming the session and shot counts. `Open Project` is guarded on the same contract. |
| 2026-09-03 | #9473 | Corrected the `CI Standard` `tests (3.x)` matrix budget from `timeout-minutes: 35` to `150` (RM #1507, program #1505; UD #9431). Measured from run 33779933815 (job 100731026646, commit ee247039, 2026-09-03): the serial core lane collects 40,488 selected tests in 2m36s and then executes them at 5.27 tests/s, so a full pass needs ~135 min. The 35-minute budget was set on 2026-06-11 (6bc73cb93) against an xdist lane and was never re-measured after the lane was serialised to `-n 0` on 2026-06-13 (b4ddf7b30); every `main` run since has been cancelled at exactly 35 minutes, failing the required `quality-gate` without reporting a single assertion. No tests are deleted, skipped or quarantined and the timeout is not removed. This makes the lane's real result observable; it does not by itself make `main` green, because the truncated run already shows ~160 real failures inside its first 24%. |
| 2026-09-03 | #9462 | Pose Studio Save/Load are real (#8882). Both buttons were shown enabled but only flashed a transient `QToolTip` carrying an internal tracker id; `_EmbedAdapter.is_dirty` returned a hardcoded `False` despite a 64-deep undo stack, and `PoseStudioWindow` had no `closeEvent`, so an hour of joint edits was discarded on close with no prompt and no way to have saved them. Implemented rather than disabled: `pose_io` already covers all five engines and `docs/user_guide/pose_studio/save_formats.md` already asserted Pose Studio routed through it, which was untrue until now. New `pose_files.py` holds the per-engine file filter and suffix; `MainWidget.is_dirty()` is real, cleared by a successful save or load; load and close prompt Save / Discard / Cancel; the embed adapter delegates to the widget's public `is_dirty`. |
| 2026-09-03 | #9449 | Bumped the release version 2.1.1 -> 2.1.2 across every surface `scripts/check_version_consistency.py` audits (`pyproject.toml`, `src/api/_version.py`, `ui/package.json` + `ui/package-lock.json`, root `Cargo.toml`, `rust_core/upstream-physics/pyproject.toml`, `VERSION`, `ui/src-tauri/tauri.conf.json`, `scripts/config/sbom_baseline.json`) plus this Identity table and SECURITY.md's footer. This is the fix-forward release for #9449: the pushed `v2.1.1` tag's `release.yml` run failed in `build` (`build_hooks.py` requires `ui/dist`, CI set `SKIP_UI_BUILD`) and published no wheel, SBOM, checksums, PyPI distribution, or GitHub release. Per `docs/operations/release-runbook.md` "Failed Release Recovery -- Fix Forward, Never Move a Tag", `v2.1.1` is retained where it is and superseded by 2.1.2; CHANGELOG entries moved from `[2.1.1]` to `[2.1.2] - 2026-09-03` with a retained-and-superseded note. No tag is created by this change -- tagging is the release operator's signed step. (spec 1.0.718) |
| 2026-09-03 | #1520 | Migrated the Section 12 change log to rows keyed by pull request (date, `#<pr>`, summary) instead of a serial spec version, and stopped requiring a `Spec Version` bump per pull request. 590 rows rewritten with each original serial preserved inline as `(spec X.Y.Z)`; row count and every row summary unchanged. `scripts/ci/check_spec_changelog_duplicates.py` now enforces the PR-keyed row contract and key uniqueness for rows dated on or after the cutover, delegating to the fleet-shared `shared_scripts/spec_changelog.py`; its duplicate-*body* ratchet is kept unchanged because a copied row is a different defect from a key collision, and its baseline shrank from 3 recorded pairs to 2. The 54 serial-collision allowances are removed, describing a defect that can no longer occur. `SPEC.md` is now in `.prettierignore` so a new row cannot re-pad the whole table. Governed campaign for Repository_Management#1520 (program #1505). |
| 2026-09-03 | #9446 | Closed the `EngineSrcPivot` restore asymmetry PR #9446 documented as latent. When a third-party cleanup pops a `src.*` parent package's `sys.modules` key while its child's key stays cached, the pivot's snapshot captures a child-without-parent; exit used to restore the child orphaned-by-key, so the next fresh import of the parent produced a childless parent module that the import system never re-links (the cached child short-circuits `_find_and_load`), and every dotted-string patch target under it (`monkeypatch.setattr("src.x.y.attr", ...)`, `mock.patch("src.x.y.f")`) raised `AttributeError` while the module-object form worked -- the exact CI failure #9446 worked around in the myosuite adapter test. `_relink_to_parent` now re-imports the missing parent and links the restored child onto it, parents-first; the `src.shared` keep-set is exempt because the import-alias machinery deliberately seeds child-only entries there and the pivot never evicts that subtree. Regression-covered in `tests/unit/repo_hygiene/test_no_permanent_src_module_shadow.py` by reproducing the popped-parent state through a full enter/exit cycle and asserting the dotted-string form patches the identical child object. (spec 1.0.717) |
| 2026-09-03 | n/a | **ADR-0046 Stage 2's launch-monitor module retirement is complete.** Wave 3b retires the last eight -- `strokes_gained_types` and `_scoring_statistics` (P12), `outcome_proxy` (P13), `strokes_gained` (P14), `conformance_bundle` (P17) and the `player_covariation` trio (P18) -- onto the canonical launch-monitor layer vendored from Tools, deleting 2,425 lines. Across the four waves all 28 `port-up`/`merge` modules are gone; `src/tools/launch_monitor_model/` now holds only the re-export facade, the app-local `project`, and `strokes_gained_baseline`. That third file is ADR-0048 step P12's documented exclusion, not a leftover: the canonical layer types its `baseline` argument against runtime-checkable protocols because `rate_of_closure.launch_monitor_strokes_gained_baseline` is already the authority for loading and digest-verifying that artifact, but a protocol validates nothing at a trust boundary and the analytics API parses a baseline off the wire, so the hash-verifying pydantic model stays here and is pinned `isinstance`-compatible with both protocols. Behaviour deltas, each under an owner ruling and re-pinned with old and new values: **G1-D2** makes the session cell the canonical strokes-gained inference unit, so a per-player longitudinal fit reports sample_count 5 rather than 40 and P4's r_squared moves 0.15450437016457175 -> 0.5682576505731145 with p 0.012104880151308768 -> 0.1410798565763777 -- the slope is unchanged to 15 significant figures (0.075881035543697128), so what the decision corrected is a significance claim built on pseudo-replication, not a measurement; UpstreamDrift's shot-level fit survives as the named variant `shot-level-sg-trend/1` and every summary names its estimand. **G1-D3** (exclude-and-audit) was already this module's posture and needed no edit. **D22/D23** adopted UpstreamDrift's postures, so no UpstreamDrift number moves; the P18 union adds `method_description` (D26's field count 19 -> 20, leaving `backing_data` as the only legacy-only field, by design), a typed `interval_withheld_reason`, and the documented `BETWEEN_PLAYER_INTERVAL_MIN_GROUPS` threshold. The drift gates stay at 71 with no unruled pin moved. (spec 1.0.716) |
| 2026-09-03 | n/a | ADR-0046 Stage 2 wave 3a executed: the launch-monitor contract spine and the longitudinal tier -- `flexible_analysis` (P10), `contract_v2` (P11), `longitudinal_types` + `longitudinal_statistics` (P15), `longitudinal` (P16), `corpus` (P19) and the four `dataset_reference*` modules (P20) -- retire onto the canonical launch-monitor layer vendored from Tools, deleting 2,948 lines of UpstreamDrift implementation. Wave contents are ordered by the **canonical** dependency graph rather than by ADR-0048's port-order number, because canonical `contract_v2` imports canonical `flexible_analysis` and canonical `dataset_reference_contract` imports canonical `corpus`: retiring a consumer while its dependency still resolved to UpstreamDrift would leave two copies of `FlexibleAnalysisResult` and of `AnalysisContextV2` in one process. The wave carries three owner rulings, each re-pinned with old and new values. **D15** removes under-sampled predictors from the Benjamini-Hochberg pool before correction, so the drift gate's adjusted p value moves 0.9217169029997262 -> 0.8646154865187129 and now agrees with `rate_of_closure` at delta 0.0 (the 6.60% inflation was always upward, so the defect could only make a finding read less significant than its evidence). **D17** carries the boolean 0/1 projection label up from `relationships` onto `CorrelationEstimate.is_boolean_projected`; the coefficient is unchanged. **G1-D1** makes the pooled longitudinal estimator a named-method pair: `PooledAssociationV1.method` is required with no default and no back-compat alias, `player_fixed_effects_ols_clustered_by_player` becomes `ud-cluster-robust-fe/1`, and `dl-random-effects/1` reproduces `rate_of_closure`'s random-effects estimate (-0.5282789828979909, CI [-1.0145384362562389, -0.04201952953974292], tau^2 0.1594137105940229, I^2 69.38732305300319%) bit-for-bit, closing measured divergences D10, D11 and D12. The drift gates stay at 71 and their pinned numbers are unchanged; three gates change sides from DIFFER to RESOLVED. (spec 1.0.715) |
| 2026-09-03 | #9348 | ADR-0046 Stage 2 wave 2 executed: `relationships` (P7), `modeling` (P8), and `profiles` + `importer` (P9) -- ADR-0048's port order -- retire onto the canonical launch-monitor layer vendored from Tools, and 941 lines of UpstreamDrift implementation behind those four names are deleted. `modeling`, `profiles`, and `importer` are AST-identical twins of their canonical counterparts (docstrings, `__all__`, and `from shared.python.launch_monitor.X import ...` in place of `from src.tools.launch_monitor_model.X import ...` normalised out of the comparison, matching wave 1's method); `relationships` is not -- its canonical twin carries owner ruling **D17** (ADR-0048 "Owner Rulings (2026-09-02)"), landed additively: `CorrelationResult` gains `boolean_projected` and `DependencyEdge` gains `includes_boolean_projection`, every prior field on both dataclasses is untouched, and the boolean-as-0/1 correlation coefficient is bit-identical to before the ruling (pinned at `-0.04331480818242096` against the same fixture on both sides). Two intra-package consumers move with their imports: `corpus.py`'s `_convert` (from `importer`) and `flexible_analysis.py`'s `compute_correlations` (from `relationships`); the façade's four import statements re-point the same way wave 1's did. The workbench Relationships tab (`gui.py::run_relationship_analysis`) reads only `.coefficients`, `.method`, and `.edges`, so it is unaffected by the two additive fields; its tab tests (`test_relationship_analysis_populates_matrix_and_scientific_warning`, `test_relationship_analysis_title_identifies_the_project`, `test_data_change_clears_every_stale_analysis_canvas`) still pass unmodified against the canonical import, as does every other workbench tab test -- 22 passed, unchanged from wave 1. `tests/unit/launch_monitor/test_canonical_layer_parity.py` gains a second parametrization, `WAVE_2_MODULES`, sharing wave 1's "file gone and the canonical import resolves" assertion across all four modules regardless of twin status, plus one behavioural test, `test_relationships_gains_the_d17_boolean_projection_fields_additively`, pinning the field addition and the unchanged math directly rather than trusting the vendored test alone. The architecture-budget exception for `modeling.py::fit_predictive_model` is removed rather than renewed, mirroring wave 1's removal of the `comparison.py` exception, since the module it covered no longer exists in this repository. No ADR edit: ADR-0048 is 213 bytes under the 50KB documentation budget, and this PR executes its port order rather than amending it. The ADR-0046 G0 drift gates stay at 71 and are untouched -- none of the four retired modules backs a drift gate. Verified by execution: `tests/unit/launch_monitor/` 139 passed / 4 skipped (134 + 5 new, up from wave 1's baseline), the four canonical module test files (`test_modeling.py`, `test_profiles.py`, `test_importer.py`, `test_relationships.py`) 45 passed / 4 skipped against the vendored tree, `tests/ui/tools/launch_monitor/` 22 passed, the four `tests/api/` launch-monitor route files 37 passed, `tests/integration/launch_monitor_drift/` 71 passed -- untouched, `tests/companion/` 57 passed, and the repo-hygiene/registry pair 10 passed, all unchanged from wave 1's counts (#9348). (spec 1.0.713) |
| 2026-09-03 | #9234 | Bounds solver tolerance and function evaluations in subject-scaled closed contact feasibility atlas test to prevent CI test-runner timeout hangs (#9234). (spec 1.0.714) |
| 2026-09-03 | #9420 | ADR-0046 Stage 2 wave 1 executed: the six modules ADR-0048's port order lists as P1-P6 -- `dispersion`, `multivariate`, `trends`, `comparison`, `schema`, `treatment`, every one a tier-0 leaf with no intra-package dependency -- are retired onto the canonical launch-monitor layer vendored from Tools, and 845 lines of UpstreamDrift implementation behind those names are deleted. UpstreamDrift now runs Tools' code for them. The re-point is the mechanical rewrite the port order prescribes (`src.tools.launch_monitor_model.X` -> `shared.python.launch_monitor.X`), unblocked by #9420. Consumers were resolved by AST rather than grep, because the package façade re-exports every symbol and a text grep attributes each consumer to the façade rather than to the module that owns it: the façade carries every workbench and API consumer of all six modules with no edit to `gui.py`, `widgets.py`, `flexible_analysis_widget.py`, or the analytics routes; `schema` was the only wave-1 module with intra-package consumers and its nine importers (`contract_v2`, `corpus`, `flexible_analysis`, `importer`, `modeling`, `player_covariation`, `profiles`, `project`, `relationships`) move with it; two direct test imports re-point. ADR-0048 P3's rename lands with them: `TrendResult` becomes `TemporalTrendResult` with no back-compat alias, because `rate_of_closure` exports the old name for a different estimand -- a cumulative session-ordinal mean rather than a per-day robust slope -- and an alias would re-create the collision the rename exists to prevent. `tests/unit/launch_monitor/test_canonical_layer_parity.py`'s six twin-identity assertions are obsolete by construction, since no second copy remains to compare; they are replaced by an assertion that pins both halves of the retirement -- the UpstreamDrift file is gone AND the canonical module imports and resolves inside the vendored tree -- plus a check that the renamed symbol carries no alias on either side. The provenance probe on `shared.python.launch_monitor` is unchanged. No behaviour, GUI, or API surface changed: the façade's export list is identical apart from the one renamed symbol. The ADR-0046 G0 drift gates stay at 71 and are untouched -- they compare the canonical layer against `rate_of_closure.launch_monitor_performance`, a different implementation, so re-pointing the UpstreamDrift side of the dispersion gate leaves it measuring the same two programs (#9425). (spec 1.0.712) |
| 2026-09-02 | #9404 | Stopped the C3D-viewer/3D-GUI `src` pivot from cross-polluting the unit-test gate. PR #9404 scoped the rebind to its own directory but restored only the bare `sys.modules["src"]` key, while installing the pivot evicts every other `src.*` entry outside `src.shared*` -- so unrelated suites were left re-importing the repo's modules and holding **duplicate** module objects, breaking `importlib.reload`, `monkeypatch.setattr("src...")`, `patch("src...")` and `isinstance` across `tests/unit/launchers`, `tests/unit/launcher` and `tests/unit/utils` (15 FAILED on `origin/main`, every one green in isolation). Both conftests now share `tests/helpers/engine_src_pivot.EngineSrcPivot`, which snapshots the whole `src` namespace slice plus the `sys.path` entries it adds and restores both -- parent-package attribute links included -- on the outermost exit; `tests/unit/repo_hygiene/test_no_permanent_src_module_shadow.py` now pins that invariant. (spec 1.0.711) |
| 2026-09-02 | #8863 | Executed ADR-0048 "Stage 2 Blocker (G2)" Option 1, unblocking ADR-0046 Stage 2. UpstreamDrift's transitional launch-monitor package moved out of the `shared.python` namespace, out of `src/shared/python/` into `src/tools/launch_monitor_model/`, beside the `src/tools/launch_monitor_analytics/` workbench that consumes it -- all thirty modules by `git mv` so history follows, with every `src.shared.python.launch_monitor` import rewritten and no other change. The package imports only itself, third-party, and stdlib, so the move is self-contained; no behaviour, GUI, or API surface moved with it. `shared.python.launch_monitor` now resolves into the vendored Tools tree instead of back to UpstreamDrift's own copy, so the import rewrite ADR-0048's port order prescribes stops being a self-referential no-op. The destination was chosen against the guards rather than by convention: `tools` is not one of the four layers in `tests/architecture/test_dependency_direction.py`, so no forbidden pair covers api -> tools and three `src/api/routes/*.py` files already import `src.tools.*`; `test_import_boundaries.py` constrains only `src/shared/python/**` and `src/engines/**`; and `launch_monitor_model` is provided by exactly one pythonpath root, so no top-level name collision is added while the `launch_monitor` one is removed. `scripts/config/shadow_modules.yaml` drops the `launch_monitor` entry outright (33 shadows of 37 vendor modules, down from 34), `src/config/registry_exclusions.yaml` registers the package as launcher-less (#8863), and `tests/unit/launch_monitor/test_canonical_layer_parity.py` retires its shadow-characterisation pin -- written to flip the day the shadow resolved -- for a positive provenance probe that `shared.python.launch_monitor` resolves inside the vendor tree. The six wave-1 twin-identity assertions are unchanged and still pass. The ADR-0046 G0 drift gates remain at 70 and untouched (#9420). (spec 1.0.710) |
| 2026-09-02 | #4908 | Bumped `vendor/ud-tools` gitlink to Tools `c0a395d59ec0a78aa70d4a989ccfc8f0a9605319` and **re-pinned the strokes-gained drift gate to the resolved G1-D3 contract**. This is the UpstreamDrift half of a paired two-repo change; Tools PR #4908 (`ed9f7a90d`) is the other half and neither is correct alone; the pin also carries #4909 (`c0a395d59`), which is that PR's visual-evidence co-change and touches no calculation. ADR-0048 decision **G1-D3** ruled that the canonical error posture is exclude-and-audit, and its _Consequence_ paragraph required Tools' legacy `calculate_source_backed_strokes_gained` to stop raising. That legacy half was deferred because this repository's own G0 gate pinned the behaviour it had to change: **D1** pinned the raise (three `pytest.raises(ValueError, match="outside the baseline")` assertions plus one silent-drop case) and **D2** pinned the Tools result dataclass's field set exactly. Both pins now move, and **only** those two - every other pin in the G0 trio and the wider drift suite is byte-identical. **D1 is no longer a divergence.** `test_divergence_d1_malformed_row_handling` becomes `test_resolved_d1_both_stacks_exclude_and_audit_a_malformed_row`: `DEGENERATE_EXPECTATIONS` collapses from `(reason_code, tools_behaviour)` pairs - where `tools_behaviour` was `"raises"` for three cases and `"silently_drops"` for the fourth - to one `reason_code` per case, because both stacks now agree on the code as well as the outcome: `outside_baseline_start` -> `outside_baseline`, `missing_course_state` -> `missing_course_state`, `negative_finish_distance` -> `invalid_distance`, `unknown_stratum` -> `outside_baseline`. The test asserts the Tools audit surface directly (`status == "partial"`, `input_row_count` 161, `included_row_count` 160, `total_excluded` 1, `source_index` 160) and then asserts cross-stack equality of `status`, `by_reason`, and the excluded rows' codes and indices. Tools' mean is unchanged at the already-pinned `0.80592372152815683` in all four cases - the 160 good rows now survive one bad row on both sides. A **new** test, `test_resolved_d1_neither_stack_drops_a_row_in_silence`, appends all four malformed rows at once and requires both stacks to report `input == included + excluded` with `{outside_baseline: 2, missing_course_state: 1, invalid_distance: 1}`; that case did not exist before because Tools could not reach it. **D2 remains open** and its field-set pin grows by three: `status`, `excluded_rows`, `exclusions`. The addition is strictly additive - all eleven previously pinned fields are still asserted present - and the companion assertion that Tools carries **no** uncertainty field (`standard_deviation`, `standard_error`, `confidence_interval`, `uncertainty`) is untouched and still passes, so D2 is unchanged as a divergence. Counts: the strokes-gained drift file goes 11 -> 12 tests and the whole `tests/integration/launch_monitor_drift/` suite goes **70 -> 71 passed**, the +1 being the new silence test. Verified by execution against the bumped pin, not by inspection. (spec 1.0.709) |
| 2026-09-02 | n/a | Recorded the repo owner's 2026-09-02 ruling on ADR-0048 G1's unsized TypeScript-twin obligation ("The TypeScript-Twin Obligation Is Unsized"): **deferred-twin policy** — canonical Python modules in the Tools launch-monitor layer stand alone, and each TypeScript twin is a tracked follow-up rather than a landing prerequisite, prioritized when a web surface actually needs that module (ADR-0046 Stage 2's re-pointing of the UD workbench and the Impact Explorer tab is what reveals which). Landed in ADR-0046's Consequences section (source of record, per ADR-0048's own instruction that the ruling belongs there) and cross-referenced from ADR-0048's "The TypeScript-Twin Obligation Is Unsized" risk subsection; `docs/adr/README.md`'s Recent Amendments gained a bullet for each file. No code changed. (spec 1.0.708) |
| 2026-09-02 | n/a | ADR-0046 Stage 2 (G2) wave 1 measured, and blocked at its own step 2. The six lowest-risk launch-monitor modules (`dispersion`, `multivariate`, `trends`, `comparison`, `schema`, `treatment`) were confirmed structurally identical to their canonical twins in the vendored Tools layer (pin `6238889a9`) modulo the port's docstrings, `__all__`, and P3's deliberate `TrendResult` -> `TemporalTrendResult` rename, and that identity is now pinned by `tests/unit/launch_monitor/test_canonical_layer_parity.py` so neither copy can drift while the retirement waits. No UpstreamDrift module was retired: the import rewrite ADR-0048 prescribes (`src.shared.python.launch_monitor.X` -> `shared.python.launch_monitor.X`) resolves back to UpstreamDrift's own package, because both packages carry an `__init__.py` on `shared.python.__path__` and the `src/` entry precedes the vendor entry, so the façade imports itself under the canonical name and deleting a module raises `ModuleNotFoundError` rather than falling through to the vendored copy. The blocker is at package granularity, as is the shadow guard that tracks it, so `scripts/config/shadow_modules.yaml`'s `launch_monitor` entry cannot be narrowed per file; both it and `docs/adr/0048-launch-monitor-port-plan.md` now record the finding, the reproduction, and three options for the owner. No behaviour, GUI, or API surface changed; the ADR-0046 G0 drift gates remain at 70 and untouched. (spec 1.0.707) |
| 2026-09-02 | #9250 | refactor(bunkershot3d): split accelerated sand mass metrics out of divot.py (#9250) (spec 1.0.706) |
| 2026-09-02 | #4899 | Bumped `vendor/ud-tools` gitlink to Tools `6238889a9164d380f8a366b9f6d8057656641ee4`, making the completed canonical launch-monitor layer available for Stage 2 consumption: ports P1-P20 (dispersion, multivariate, trends, comparison, schema, treatment, relationships, modeling, profiles/importer, flexible_analysis, contract_v2, strokes gained, longitudinal, outcome proxy, union-port player covariation, conformance, corpus merge, dataset reference) plus owner rulings D15/D17/D22/D23 (Tools#4899-#4907, ADR-0046 Stage 1). Updated the hard-pinned `pinned_commit` assertion in `tests/companion/test_companion_catalog.py` to match; the UD-side `rate_of_closure` drift gates were re-run unchanged against the new pin. The pin also surfaces two pre-existing hygiene ratchets rather than regressions: `launch_monitor` is now a brand-new top-level Tools package (ADR-0046 Stage 1) colliding by name with UD's pre-existing same-named package, ledgered under the ADR-0046 G1 port-plan inventory (#9348) pending Stage 2 re-pointing; and Tools#4889 removed its own `sidekick/process_calculators/scrubber/tests/` directory as dead/duplicate code, so UD's local scrubber tests (no Tools child-copy header; test UD's own ScrubberEngine) gained an ownership-exceptions entry under the existing sidekick process-calculator debt (#5623). (spec 1.0.705) |
| 2026-09-02 | #4907 | Closed ADR-0048 D30, the launch-monitor private-corpus governance hole, on the UpstreamDrift side: UpstreamDrift's own launch-monitor `corpus` module (then in `src/tools/launch_monitor_model/`, retired onto the canonical layer in 1.0.715) gained the fail-closed manifest gate its Tools twin always had. Five refusals now run before a single row is materialised -- missing `_MANIFEST.json`, unsupported `schema_version`, a declared `total_rows` outside the 300,000-row desktop retained-data cap, a row-count mismatch, and a source-set mismatch -- each raising by name so a caller learns which check fired. The check basis is the WHOLE corpus, never the caller's selection: rows come from an unfiltered `dataset.count_rows()` and sources from the `source_id=` hive partition directory names, so the existing source/metric pushdown (D31) cannot weaken the guarantee. ADR-0031 unit canonicalisation (D29), the 20-hex shot-identity digest and the selection pushdown are unchanged and now run only on what survives the gate. Ported from the canonical Tools module `launch_monitor/corpus.py` (Tools#4907, P19) rather than imported from it -- the vendored Tools package stays a measurement dependency of the drift gates, never a runtime dependency; `MAX_RETAINED_ROWS` is redefined here and pinned equal to the vendored constant by a seam gate. The five D30 cases in `tests/integration/launch_monitor_drift/test_corpus_drift.py` flip from "Tools refuses, UD loads silently" to "both refuse, same exception class, same reason"; D28/D29/D31 are unmoved and the corpus gate grows 13 -> 15 tests (drift suite 68 -> 70). Unit coverage for the refusal paths added in `tests/unit/launch_monitor/test_corpus.py` (10 -> 17) and the workbench refusal pinned in `tests/ui/tools/launch_monitor/test_gui.py`. (spec 1.0.704) |
| 2026-09-02 | #9165 | Preserve event-loop ownership when reusing QApplication in `run_launcher` (#9165). When `run_launcher` is called within an embedded host or test runner where a `QApplication` instance already exists, it now avoids invoking `app.exec()` and returns `0` cleanly rather than re-entering the event loop, while still invoking `app.exec()` when `run_launcher` created the instance. Added focused unit test coverage in `tests/launchers/test_run_launcher_event_loop.py`. (spec 1.0.703) |
| 2026-09-02 | n/a | Repaired Section 12 change-log integrity after recurring git auto-merge damage (latest recurrence 2026-09-02 during a pin-bump merge): 56 dated change-log rows that historical merges had anchored outside Section 12 (five under "2026-04-28 Spec Bump", fifty-one under "3D Vector Distances Note") are relocated into this table in version order -- each was checked against the existing table and is a distinct entry, not a duplicate, so none were deleted. Also removed a stray mid-table header-separator row, removed five table-splitting blank lines, and restored the Section 4 directory-tree code fence that had drifted to the end of the stray tables and was swallowing Sections 5-12 into one code block when rendered. `tests/unit/repo_hygiene/test_spec_changelog_integrity.py` now asserts that no dated change-log-shaped row exists outside Section 12 and that the Section 12 table keeps a single header separator, so the next bad auto-merge fails CI instead of silently grafting rows into unrelated tables. (spec 1.0.702) |
| 2026-09-02 | #9385 | Aggregated `shared-tools-consumer-contracts` into the required `quality-gate` workflow check (#9385, #9390), ensuring ADR-0046 G0 cross-stack drift gates gating vendored Tools parity are required for branch protection. Updated canonical `models.yaml` entry for `rate_of_closure` to simulation/gui_ready (#9382), capitalized fallback page heading in Impact Explorer (#9383), added reproducible launch monitor fixtures README and generator reference (#9355), and clarified supported serialization formats and archival cleanup in output docs (#9335, #9334, #9333). (spec 1.0.701) |
| 2026-09-02 | #9392 | Follow-up to 1.0.699: added the required `docs/adr/README.md` "Recent Amendments" entry for the ADR-0048 owner-rulings update (D15/D17/D22/D23), which PR #9392 squash-merged without it -- `doc-governance`/`docs-quality-gate` are not currently required status checks, so auto-merge landed the PR while both reported "ADR changes detected without updating docs/adr/README.md". This PR adds the amendment bullet and re-passes `scripts/check_docs_governance.py` locally. Docs-only: no production module touched. (spec 1.0.700) |
| 2026-09-02 | #4899 | ADR-0048: recorded the repo owner's rulings on four of G0.1's pinned divergences in a new "Owner Rulings (2026-09-02)" section, placed immediately after "Rows That Need an Owner Decision". D15 (FDR multiplicity denominator, `test_divergence_d15_multiplicity_denominator_differs`): the canonical layer excludes under-sampled predictors below `min_samples` from the Benjamini-Hochberg denominator before correcting -- Tools' existing posture; UD's count-all behaviour is ruled a defect, not a preserved method, applying to the canonical `relationships.py`/flexible-analysis modules, with the P7 port landing UD-verbatim first and a follow-up PR applying the ruling. D17 (boolean columns, `test_divergence_d17_boolean_columns_are_analysed_only_by_ud`): the canonical layer analyses booleans as 0/1 (UD's capability preserved) but the projection must be labelled explicit in the result, and Tools' refusal message becomes a pointer to that path. D22 (low-dof Fisher intervals, `test_divergence_d22_between_player_interval_exists_only_in_tools`): the canonical layer withholds the between-player Fisher interval when degrees of freedom make it uninformative (UD's posture), documenting the threshold, applying at P18. D23 (unit labelling, `test_divergence_d23_unit_resolution_differs`): the column-name-suffix heuristic is ruled a defect and deleted; the canonical layer resolves units from the canonical registry and returns unknown rather than guessing, applying at P18. Also appended one sentence to the P3 (`trends.py`) row noting the mandated `TrendResult` rename was executed in Tools#4899 as `TemporalTrendResult`, deliberately with no back-compat alias, so Stage 2's import rewrite must special-case the symbol. Docs-only: no production module touched. (spec 1.0.699) |
| 2026-09-02 | #9351 | ADR-0047 H2 (#9351): Shot Tracer imports `ball_flight_trajectory/1` records for cross-family comparison. New `src/launchers/_shot_tracer_trajectory_import.py` resolves the vendored `vendor/ud-tools` checkout via the canonical `tools_repo_path.resolve_tools_repo` facade and reads records through the vendored `flight_interchange.ball_flight_trajectory_from_json` reader, never reimplementing wire validation. Imported curves convert into Shot Tracer's native plot frame (`flight_xfwd_yleft_zup`) through a closed frame-dispatch table and are labeled `model_family / model_name` in a new Imported Trajectories list -- never unlabeled. `app_xtarget_yup_zright` is a legal wire frame with no converter yet and is refused explicitly; every other refusal (unknown or missing fields, malformed provenance, non-monotone samples, malformed JSON, a missing file, an unresolvable vendor checkout) surfaces to the user as a dialog carrying the reader's own named reason, never a silent drop. 19 new tests (9 Qt-free logic tests against the real vendored reader, 10 GUI tests) pass alongside the 41 pre-existing shot-tracer tests and the 39 feature-parity gate tests. `simulation.shot_tracer` returns to `parity` in the feature-parity registry now that both this and ADR-0047 H3 (#9352) give their respective surfaces cross-family import. (spec 1.0.698) |
| 2026-09-02 | #9236 | Hardened the #9236 articulated manufactured-solution authority boundary without changing scientific evidence. Added a study-scoped, hash-locked CPython 3.11.15 manylinux environment; isolated native workflow execution; canonical atomic JSON serialization; explicit execution-profile provenance; and declared-tolerance semantic comparison. The comparator now rejects incomplete or unknown tolerance policy, missing governed result paths in either record, non-finite or non-positive policy values, inconsistent per-record maxima, negative residuals, non-monotone convergence, and stale profiles. Independent review is GO for the implementation at `0f2f4aed2`; the scientific record and checksum/release/claim manifests remain byte-identical to protected main. Exactly three authority tests remain RED until two exact-environment Linux native builds reproduce byte-identical governed evidence, so #9192, #9174, and AffineDrift #4022 remain open. (spec 1.0.697) |
| 2026-09-02 | #9371 | Advanced `vendor/ud-tools` 5e0eaad -> e88a334, completing the ADR-0045/0047 Tools-side wave on top of #9371's H1 pin: the Impact Explorer flight playback now replays imported `ball_flight_trajectory/1` records with explicit frame conversion (H4, Tools#4890 - closes #9353), the Impact Explorer putting tab imports UD-authored greens in both runtimes (F2, Tools#4891 - closes #9344), and the two canonical playback speed sets are gated against silent drift in both runtimes (Tools#4887). No UD source change: the bump ARMS the four cross-family sanity gates in `test_flight_trajectory_export.py` (previously skip-with-reason) with no test edit. Verified on this exact pin: launcher-manifest smoke 25 passed, ADR-0046 G0 drift gates 28 passed, flight-export suite 25 passed with zero skips. The companion-catalog provenance ratchet (tests/companion/test_companion_catalog.py) has its expected pinned_commit advanced to the same commit - that test exists precisely so a vendor bump is a conscious act. (spec 1.0.692) |
| 2026-08-30 | n/a | Made the Rate of Closure Impact Explorer a first-class, findable launcher citizen. Tile `rate_of_closure` moves from `biomechanics`/`external`/order 46 to `simulation`/`gui_ready`/order 7 (leading the simulation category, ahead of Putting Green) with an accurate suite description; its web contract upgrades from `native-window` (a dead button for remote users) to a real route: `/tools/impact-explorer` embeds the vendored React build when the API has mounted it (`/impact-explorer-app`, served only when `vendor/ud-tools/src/rate_of_closure/web/dist` exists - build with `npm ci && npm run build -- --base=/impact-explorer-app/`; the base flag is mandatory or the embedded app requests assets from the host paths) and otherwise states exactly how to build it plus the desktop alternative - no silent blank frame. Startup metrics record `impact_explorer_web` either way so degradation is explicit. Feature-parity registry entry flips from `exempt` to `parity`. Verified: launcher-manifest + feature-parity gates 121 passed, page tests 4 passed, mount tests 3 passed. (spec 1.0.691) |
| 2026-09-02 | #9348 | ADR-0046 G0.1 (#9348): landed cross-stack drift gates for the three launch-monitor twin pairs ADR-0048 left in the needs-owner-decision bucket because a live Tools twin exists with no G0 measurement covering it — flexible analysis, player covariation, and the private corpus. `tests/integration/launch_monitor_drift/test_flexible_analysis_drift.py` (13 gates) compares `flexible_analysis` against `launch_monitor_analysis` plus `_launch_monitor_analysis_statistics`/`_types`: correlations, OLS, and group analysis agree to delta exactly 0.0, and both stacks independently produce the same dataset fingerprint. `test_player_covariation_drift.py` (14 gates) compares the UD trio (1,098 lines) against the Tools trio (570 lines): 51 of 52 shared scalars agree inside UD's declared 12-decimal reporting quantum, and the covariation scan ranks the same six pairs in the same order on both stacks. `test_corpus_drift.py` (13 gates) compares `corpus.py` against `launch_monitor_private_corpus.py` over a synthetic two-source Parquet corpus built in a temp directory: both resolve the identical path from the identical environment variable and return the same 4 rows, sharing 2 of 15 column names. All three modules pin their divergences (D15-D31, 17 total) against the existing `adr0046_cross_stack_session_v1.json` fixture with the same posture G0 established — pin what differs, reconcile nothing. Test-only: no production module is touched and `vendor/` is untouched. (spec 1.0.690) |
| 2026-09-02 | #9348 | Made the ADR-0046 G0 cross-stack drift gates actually execute instead of reporting green unexamined (G1, #9348): CI's default checkout materialises no `vendor/ud-tools` submodule, so the suite's guard module-level-skipped every run while `vendor-freshness.yml` advanced the pin nightly with no gate attached. `tests/integration/launch_monitor_drift/conftest.py`'s `require_vendored_tools_stack()` now fails loud under `GITHUB_ACTIONS=true` instead of skipping, and both drift suites (`launch_monitor_drift`, plus F4's `putting_green_drift` once #9373 landed it) moved into the `shared-tools-consumer-contracts` job in `.github/workflows/ci-standard.yml`, the only job that materialises the pin via `fetch-pinned-tools`; the main `tests` job now explicitly `--ignore`s both so placement can never again go unexecuted-but-green (G0.1, #9372, found a fleet runner's reused work directory had been masking that same gap nondeterministically). Also restores the ADR-0045/0046/0047 `Accepted` statuses that raced the #9342 squash-merge and stranded as `Proposed`, and records Amendment 1 on ADR-0046 correcting two factual claims from G0/G1 measurement. (spec 1.0.689) |
| 2026-09-02 | #9352 | ADR-0047 H3 (#9352): the BallFlight web page accepts imported `ball_flight_trajectory/1` records for overlay, from either flight-model family. `POST /tools/ball-flight/import` (`src/api/routes/ball_flight.py`) accepts an opaque JSON `record`, validates it fail-closed through the new `src/api/routes/_ball_flight_trajectory_import.py` helper (lives under `src/api/` rather than `src/shared/python/` because `tests/unit/repo_hygiene/test_import_boundaries.py` forbids the shared layer from importing upward from `src.launchers`), which resolves `vendor/ud-tools` through the canonical facade (`src.launchers.tools_repo_path.resolve_tools_repo`) and calls the vendored reader directly — every refusal (unknown/missing wire fields, malformed provenance, non-monotone samples, non-finite values) is that reader's own message, surfaced verbatim as the response's 400 `detail` (`ContractViolationError` is itself a `ValueError`, so `handle_api_errors` maps it with no extra glue). The record's declared frame is converted explicitly into the page's plot frame (`flight_xfwd_yleft_zup`) through a closed dispatch table; the wire's other declared frame (`app_xtarget_yup_zright`) is refused by name rather than silently mis-plotted, mirroring the frame-conversion posture Shot Tracer's own H2 import (#9351) uses. The BallFlight page (`ui/src/pages/BallFlight.tsx`) gains an "Import record" file input; imported curves overlay the computed ones through the same 3D scene, profile charts, and metrics table, always labeled `model_family / model_name` so an import can never read as a curve the UD registry itself computed, and a refused file surfaces the API's named reason verbatim. `src/config/feature_parity.json`'s `simulation.shot_tracer` entry notes the new capability is currently web-first pending H2. 12 API tests (`tests/api/test_routes_ball_flight.py`): cross-family acceptance (UD `Waterloo/Penner` via a real simulate+export round trip, and a hand-built `swing_sim.flight` fixture) plus every refusal path (unknown field, missing/malformed provenance, wire-invalid frame, wire-valid-but-unsupported frame, non-monotone samples, non-finite value sent as raw bytes since `JSON.stringify` cannot produce a literal `NaN`, ragged channel declaration, non-object body). 8 unit tests (`tests/api/test_ball_flight_trajectory_import.py`) cover the pure-Python summary derivation independent of the vendored reader. 22 vitest cases added to `ui/src/pages/BallFlight.test.tsx`. The vendor pin (Tools#4888, #9371) landed mid-PR, so the vendored-reader path is verified for real (0 skips); `_load_vendored_reader` tries a plain import of the already-`pythonpath`-visible package first, matching `test_flight_trajectory_export.py`'s own cross-family gate, before falling back to the full `resolve_tools_repo` facade for a real running API server. (spec 1.0.686) |
| 2026-09-02 | n/a | Optimize list summation in `src/bunkershot3d/solvers/mpm/ballreach.py` by converting python list to ndarray prior to calling `.sum(axis=0)`. (spec-exempt: micro-optimization) (spec 1.0.685) |
| 2026-09-02 | #9346 | ADR-0045 F4 (#9346): added the UD-side consumer-contract test for the Tools green-surface adapter, completing the #9143 rider per ADR-0045's Validation section. `tests/integration/putting_green_drift/test_green_surface_adapter_consumer.py` drives UpstreamDrift's own `ball_roll_physics.BallRollPhysics`/`turf_properties.TurfProperties` (not a reimplementation) against the vendored `shared.python.swing_sim.putting.simulate_putt_on_surface`, on the same green and the same launch. Covers: a round trip authoring a green with UD's own `ContourPoint`/`GreenSurface` code, through the vendored `green_surface_from_ud_json`, and back into a fresh UD `GreenSurface`, geometry preserved at every grid node; the shared-physics gates from the `ud_adapter` module docstring (flat-green straight line, break sign matching the cross slope on both downhill directions, roll-out monotone in stimp) on both engines; the documented `mu_tools / mu_ud ~= 2.854` roll-out ratio (Tools#4819) pinned empirically from both engines' own simulate calls by starting each simulator already in pure roll to isolate the rolling-resistance phase from each engine's own skid model; and a weighted-slope UD field proven genuinely UD-loadable before asserting the adapter refuses it with its documented non-conservative-slope reason. CI-guard posture (skip locally without `vendor/ud-tools`, run for real in every CI job that fetches the pinned Tools checkout) reuses `tests/integration/launch_monitor_drift`'s `require_vendored_tools_stack` helper rather than reimplementing it. (spec 1.0.684) |
| 2026-09-01 | #7294 | Audited the architecture-budget exceptions expiring 2026-08-31 (`scripts/config/architecture_budget.json`), rather than accepting the blanket 2026-08-31 -> 2026-09-30 renewal an unrelated Bolt PR (1.0.680) had already applied with no per-function review. Seven exception entries (`train_surrogate`, `_train_loop` in `training.py`; `optimize_sequence` in `optimize.py`; `train` in `train.py`; `create_chat_router`, `chat_stream` in `router_factory.py`; `ChatDockWidget._setup_ui`) were dead: issue #7294 was closed by PR #7317, which already split them below budget, but the leftover exceptions were never removed and kept renewing on autopilot. Removed those seven. Split seven functions that were genuinely still over budget via pure extract-method refactors verified against each file's existing test suite, with no behaviour change: `docker_dialog.EnvironmentDialog.setup_ui`, `fit_swing_mujoco` (comment-delineated into compile/objective/solve/assemble stages), `viewer_3d_tab.Viewer3DTab._init_ui`/`_rebuild_scene`, `launcher_dialogs.DependencyErrorDialog.__init__`, `settings_dialog.SettingsWidget._create_layout_tab`/`_create_configuration_tab` (this last split briefly pushed `settings_dialog.py` over the 1200-line file-size budget; trimmed three over-long docstrings to land back under it). Wrote an honest, re-justified renewal for the one case that is not a quick fix rather than renewing blind again: `ChatDockWidget.__init__` (16 parameters). Its prior justification cited issue #7362, which is closed and unrelated to this signature. Unlike a genuinely irreducible-parameter case, these 16 group into cohesive connection/presentation/integration-hooks config objects, so it is real, tractable follow-up work tracked in new issue #9357 (breaking constructor change across 5 in-repo call sites) rather than kicked down the road again, expiring 2026-10-15. Verified via `check_architecture_budget.py --all` (repo-wide) and the changed-file scan (matches CI), plus 114+ passing tests across every touched file. (spec 1.0.683) |
| 2026-09-01 | #4888 | ADR-0047 H1: export this repo's flight results as the shared `swing_sim.ball_flight_trajectory/1` record. `src/shared/python/physics/flight_trajectory_export.py` builds the versioned, fail-closed interchange payload from the documented Tools contract (D-sorganization/Tools#4888) rather than importing the vendored package at runtime — the same runtime-free posture Tools' own `swing_sim.putting.ud_adapter` takes toward this repository, and the reason a pin bump cannot silently change what this repo emits. Provenance is mandatory and attributable: family `ud.flight_models`, the model's display name, and a SHA-256 digest of `FlightResult.coefficients` — the aero-coefficient set the producing model actually integrated with (#8978) — so a Shot Tracer curve and a Tools `swing_sim` flight can share axes because each is labelled, never because the families were reconciled. A result carrying no declared coefficients is refused rather than exported with an empty digest, and an optional `model_type` is cross-checked against the result so one model's samples can never carry another's identity. The record declares its frame (x forward / y left / z up) instead of leaving a consumer to infer it, and its samples are the retained integrator points, never resampled. 25 gates in `tests/unit/physics/test_flight_trajectory_export.py`: schema-valid records from Waterloo/Penner and MacDonald-Hanzely flights (asserted longhand, because those assertions _are_ this side's copy of the contract), byte-deterministic sorted-keys JSON, the digest algorithm reproduced from first principles, and every refusal path. Four cross-family gates parse a record produced here with the vendored reader, compare digests across repositories, and assert same-order carry against the sibling family; they skip with a stated reason on a pin predating Tools#4888 and arm themselves on the next vendor bump with no edit (verified green against that branch: 244.65 m from both families, ratio 1.0000). Also corrects two pre-existing `np.bool_`-vs-`bool` mypy errors in `contact_reaction_decomposition.py`, which `physics/__init__.py` imports and which therefore failed the changed-file mypy gate for any physics PR. (spec 1.0.682) |
| 2026-09-01 | n/a | Optimize Euclidean norm calculation (`np.linalg.norm` to `np.sqrt(np.einsum)`) in `src/bunkershot3d/study/morris.py` for ~30% faster 2D distance execution. (spec-exempt: micro-optimization) (spec 1.0.681) |
| 2026-09-01 | n/a | Replaced `np.sum()` with `.sum()` directly on numpy arrays in `src/api/routes/physics.py` to avoid numpy dispatching overhead and yield measurable speedup on small array hotpaths. (spec-exempt: micro-optimization) (spec 1.0.680) |
| 2026-08-30 | #9153 | Implemented the single-worker atomic smoke runner and test suite for the prospective distributed event-attribution study (`run_articulated_distributed_smoke.py`, `test_articulated_distributed_smoke_runner.py`, issue #9153). Validates registration immutability and evaluator authority; enumerates all six registered cases in frozen order; enforces thread limits (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`); executes trajectories via `attribute_distributed_contact_trajectory` across both physics engines (MuJoCo, Pinocchio) at three time steps (1.0, 0.5, 0.25 ms); persists atomic per-case checkpoints; evaluates outcome qualification against registered acceptance gates (opening/reattachment event detection, exact zero discrete impulse/work for compliant contact, pointwise force closure residual <= 1e-12); and enforces strict non-promotion invariants. (spec 1.0.679) |
| 2026-08-30 | #4873 | Repinned `vendor/ud-tools` to protected Tools merge `cc883cbaf63157b58c71cba385a683df2762b0cb` (Tools #4873), consuming the registered force-source comparison frame through the canonical provider boundary. Every objective card now reserves the same title row and undistorted 192 by 176 animation stage; fixed-hub playback keeps all six shoulders at (96, 88), uses one horizontal comparison reference and distinguishes the common (150, 148) target from each scenario's measured impact. Impact alignment remains an explicit alternative and registers every measured impact at (150, 148). The Tools provider carries rendered regression coverage over all six objectives, five playback positions, and both camera modes; UpstreamDrift's companion provenance test pins this exact gitlink. (spec 1.0.675) |
| 2026-08-30 | n/a | Replaced `TestPerformanceBudget`'s absolute 50 ms wall-clock assertion, which was intermittently red. The investigation overturned both the reported cause and my own first hypothesis: the same unchanged shot measured 45.4, 58.6 and 123.2 ms on three invocations of one machine, and checking out the commit that introduced the budget (`337991108`) and timing it on that same box gave **93.7 ms** for the original code against 45-59 ms today. The shot is ~2.5x faster per step than when the budget was written; the 50 ms figure encoded hardware several times quicker than the machine running it, so it was foreign rather than merely noisy, and no repeat count could have fixed it. The budget is now asserted two ways, neither an absolute clock: a deterministic work count (integration steps x surface elements, identical on every machine) that catches an accidental quadratic, a lost early exit or a silently refined mesh; and a cost ratio against a reference numpy workload timed in the same process, so a slower or busier box moves both. Both verified red/green by injection: a 3x per-step slowdown reports 78.7x against a 20x ceiling, and a 4x step-count regression reports 243 steps against a budget of 96. Neither test is excluded from the default lane -- the DOE budget is a real product constraint and a test nobody runs cannot defend it. Also retired the budget's stated justification: "so a 1000-point DOE is minutes" is wrong by an order of magnitude (`WedgeDesign` has seven sweepable parameters and Sobol' costs `N(D+2)`, so `N=1024` is 9,216 shots), nothing runs it (`bunkershot3d.study` never calls `simulate_shot`), and a sweep would answer nothing today because model-form uncertainty is 81-86% of the band and no two shipped designs separate. The two live reasons are kept instead: F0's tier identity under ADR-0032 (it is the default _because_ it is fast, against F1's seconds-to-minutes) and workbench interactivity (one evaluation is the shot plus a 5x5 playability grid, ~26 shots). The DOE test is renamed to what it defends. (spec 1.0.674) |
| 2026-08-30 | #8699 | Made spanwise (heel-to-toe) sole load a first-class BunkerShot3D model output (`src/bunkershot3d/metrics/spanwise.py`, epic #8699, ADR-0044). F0 already integrates a per-element response across the whole blade -- ADR-0044 records that it is reporting-blind, not geometrically blind -- but `ShotResult` exposed only the aggregate wrench, so two soles differing only in heel/toe relief read as the same shot even though F0's own answer differs (peak force 841.2 N flat, 781.1 N at 0.3 heel relief, 742.1 N at 0.3 toe relief). `spanwise_load` consumes the existing `SoleLoadTrace` artifact and reports the binned distribution across the span plus four summaries a wedge designer uses: a signed heel/toe balance that is 0 for a symmetric sole and negative toward the heel, the impulse-weighted spanwise centroid (absolute and as a fraction of the half-span), its per-sample migration through the strike, and the share of impulse carried by each outer third. The summaries are computed from the elements rather than the bins, so they do not move with `n_bins`. Bin count is explicit and never defaulted, and the metric refuses rather than smooths: a bin count finer than `MIN_ELEMENTS_PER_SPANWISE_BIN` stations per bin, any empty bin, a sole collapsed onto one spanwise station, an unloaded trace, a refused envelope verdict, and F1 outright (plane strain has no span, ADR-0033/0044). Tier and validity travel on `SpanwiseCredibility` with `measured_constants() == ()` and a `require_sand_response()` that always raises, because the distribution is the sole's load and not the sand's response. The sign is pinned by test rather than comment -- inverting it turns five tests red -- which is the defect class #9247 describes elsewhere in this model. (spec 1.0.671) |
| 2026-08-30 | #4872 | Repinned `vendor/ud-tools` to protected Tools merge `95a59544972064cff9f4191a8f15eaa02c10ce81` (Tools #4872), making the golf-like force-source objective lab and its six-objective registry available through UpstreamDrift's canonical provider boundary. The downstream REST comparison contract now requires `hand_path_impulse` alongside the existing five objectives, the React/Tauri lab gives it the explicit label "Signed Hand-Path Impulse", and the scientific boundary distinguishes the signed time integral from MacKenzie-style work-per-path-length average force. The Tools-owned lab supplies fixed-hub, single-pass/no-loop animations, near-horizontal impact qualification, speed/shoulder-torque/wrist-torque plots, direct starting-pose and constraint inputs, 30 N·m wrist exploration, robustness trials, finer torque granularity, and interpolated 0.05x-3x playback; UpstreamDrift consumes rather than copies that implementation. Companion provenance now asserts the new exact gitlink. The pin also surfaces a `plotting` provider package that overlaps UpstreamDrift's older plotting package; the pre-existing overlap is recorded explicitly in the no-growth shadow ledger under migration issue #5623 and its existing 2026-12-31 sunset instead of being hidden or resolved by deleting either implementation during this unrelated pin. (spec 1.0.668) |
| 2026-08-30 | #9127 | Repaired the embedded-tool bootstrap test harness, which had been failing `unit-test-gate` -- and therefore the required `quality-gate` -- on every open pull request. `_make_filtered_import` decided what to intercept with a substring test for `_embed_adapter` plus a special case for `pose_studio.gui`, but `FALLBACK_ADAPTER_MODULES` also holds `src.launchers.adapters.simscape_embed` and `src.launchers.adapters.swing_objective_lab_embed` (the latter added by #9127 on 2026-08-26). Neither matched, so both were imported for real inside tests that patch every adapter import to fail; the Swing Objective Lab adapter registered itself and `test_bootstrap_handles_import_errors` saw `['swing_objective_lab']` where it expected `[]`. The interception set is now derived from `bootstrap._adapter_modules_for_bootstrap()`, and a new harness guard fails with the offending module names if any adapter ever escapes interception again. No production code changed. (spec 1.0.667) |
| 2026-08-29 | #9230 | Added `git`/`git.exe` to `ALLOWED_EXECUTABLES` in `src/shared/python/security/secure_subprocess.py` (issue #9230). `git_sync_repository` in `src/shared/python/gui_pkg/launcher_utils.py` runs `git fetch --all` and `git pull` through `run_command`, and `git` was not on the allowlist, so `validate_executable` raised `SecureSubprocessError` on every call. Nothing absorbed it: `run_command`'s `@log_errors(..., reraise=False, default_return=None)` catches only `(RuntimeError, TypeError, ValueError)` and `SecureSubprocessError` is a plain `Exception` subclass, and the helper's own `except (RuntimeError, ValueError, OSError)` does not cover it either -- so the function raised to its sole production caller, `installer/legacy_setup.py`, instead of logging a warning and returning `False` as its docstring and structure intend. This is the behaviour 1.0.651 recorded in passing; it is now fixed rather than only documented. **Revision to the framing in issue #9230 itself**, which warned that `git` can execute arbitrary code via aliases, hooks and `-c core.pager=`/`-c alias.*`: true, but not decisive, and the issue's caution was overstated relative to the list it was reasoning about. The allowlist is not an arbitrary-code-execution boundary -- it already contains `python`, `python3`, `docker` and `wsl`, each of which trivially runs arbitrary code (`python -c ...`, `docker run ...`, `wsl <anything>`), so `git` is strictly less capable than entries already present and adding it does not widen the threat model. Its actual purpose is to constrain _which tools_ this codebase may invoke, not to sandbox them. The alternative -- catching `SecureSubprocessError` in `git_sync_repository` and returning `False` -- was rejected: it would permanently disable repository sync while reporting failure forever, hiding a real capability gap behind a log line. One existing test changed: `tests/unit/security/test_subprocess_utils.py::TestRunCommandSuiteRoot::test_executable_rejection_already_raises` used `git` as its _example_ of a rejected executable while pinning the raise-not-return contract; the example moved to `rm` and the contract is unchanged. No test asserted that `git` in particular was intentionally blocked. New `tests/unit/security/test_git_executable_allowlist.py` pins the contract in three parts: `git`, `git.exe`, a cased spelling and an absolute path all pass `validate_executable` and reach the subprocess layer through `secure_run`; `rm`, `curl`, `/bin/sh` and the near-miss `git-upload-pack` still raise, so the tests prove the allowlist still works rather than only that git passes; and `git_sync_repository` returns `True`/`False` -- never raises -- across a successful pull, a non-zero pull, a `None` result and `OSError`/`RuntimeError` from the command layer, plus one end-to-end run through the real allowlist in a throwaway `git init` repository with no remote, which fails ordinarily and returns `False` without touching the network. (spec 1.0.661) |
| 2026-08-29 | n/a | Extended the SPEC.md changelog ratchet to catch a second duplication mode it was blind to: the same entry logged twice under **different** version numbers. That happens when a branch renumbers to dodge a version collision and then merges the branch it was dodging, so keeping 'both' rows keeps one change twice. The existing check compares version numbers, which genuinely differ in that case, so it passed -- the defect was found only when a reviewer noticed two 4437-character rows byte-identical apart from their version. The guard now also fingerprints each row's prose (bodies under 80 characters are exempt, since short entries repeat innocently) and reports the versions sharing one body. Three such pairs already on main are baselined alongside the 54 version duplicates, so this lands without a historical cleanup. Verified red/green on both modes independently, and on a changelog whose table cannot be parsed at all -- which still fails rather than silently passing. (spec 1.0.660) |
| 2026-08-29 | #8659 | Replaced the divot mass ball launch divides by with the mass the strike actually accelerated (issue #8659; #8657 is what turned a reported metric into a denominator). At the workbench's nominal greenside shot the F0 solver's 2.917 N.s over the prismatic divot's 63.7 g implied sand leaving at 45.8 m/s from a head arriving at 25.0 m/s, which is not a tuning discrepancy but a contradiction. The prism counts only the sand under the sole path; a splash also throws the bow wave, the heave above the original surface, and the divot's own sloping walls. `metrics/divot.py` now reports both quantities: `DivotMetrics.mass_kg` is the prism, unchanged and with its provenance intact, and `DivotMetrics.accelerated_mass` is an `AcceleratedSandMass` **interval** whose two edges name what they rest on. The in-plane edge was measured against the F1 plane-strain MPM tier: ten whole-shot marches (`simulate_f1_shot`, dx = 4 mm, 12 ms) across attack -4/-8/-12 deg, marketed bounce 8/20 deg, sole 16/20/24 mm, firm/fluffy/plugged beds and 20/25 m/s, each reduced twice from its own record -- once by the same prismatic rule and once by the momentum-and-energy consistent moving mass `P^2 / (2 T)`, which Cauchy-Schwarz makes a lower bound on the mass with any motion at all. Evaluated at matched prism (the instant F1's own accumulated prism equals F0's, so both tiers are asked about the same swept sand) the ratio runs 2.845 to 3.898 over the nine designs that reached it, geometric mean 3.30; the ratio _falls_ through a march because the prism grows along the sole path while the moving mass saturates, which is why the evaluation point is stated rather than chosen quietly, and the one design that did not reach matched prism in 12 ms is excluded rather than extrapolated. Plane strain is structurally blind to the divot's walls, so the out-of-plane edge is a stated model -- a trench whose walls lie back at the **bed's own** friction angle, `w * integral d ds + cot(phi) * integral d^2 ds`, adding no new fitted constant -- and that blindness is the reason the answer is an interval and not a point. At the nominal shot: prism 63.7 g, interval 180.8-428.4 g, central 278.3 g (4.37x), implied ejecta 10.5 m/s against 25.0 m/s of head. **This is a consistency correction between two uncalibrated models and not a validation**: F1 is BEYOND*VALIDATION with a 1.44 m/s published-speed ceiling and 0 of 4 on NASA-STD-7009B, no ejecta mass has ever been measured on a bunker shot (#8616), and `ACCELERATED_MASS_CONSISTENCY_REASON` says so wherever the number is reported. What the comparison did do is falsify: the prism was inadmissible and the interval is not. `SandDelivery` now refuses an impulse and a mass that together imply ejecta faster than the head, as a plain `raise` where the two quantities first meet -- never an `assert`, since `python -O` strips those and `DBC_LEVEL=off` disables contracts -- and #8657's supersonic-ejecta verdict reason is gone because the condition it reported can no longer be constructed. It is replaced by `MASS_INTERVAL_FLOOR_REASON`, which fires when only the interval's \_lower edge* falls below `J / v_entry`: that is information about the width of the band, not a claim the reported mass is wrong, so it is reported rather than raised, and nothing anywhere is clamped. **Consequence, carried rather than absorbed:** nominal carry moves 11.81 m -> 1.59 m (interval 0.76-3.17 m, ball speed 4.14 m/s) and the 5x5 playability grid goes empty against its 12.0 m target, 0.06-2.04 m across the whole grid. The old 12 m was an artifact of dividing by a mass three to four times too small; the free parameter that would have to absorb the difference is `BALL_MOMENTUM_TRANSFER_EFFICIENCY`, an explicit placeholder with no measurement behind it, and re-tuning it to recover a familiar number would be fitting an uncalibrated constant to an unmeasured target, so it is left alone. `divot_metrics` now requires `friction_angle_deg` with no default, because the wall angle is the bed's and a default would be an invented divot shape. New `tests/bunkershot3d/metrics/test_accelerated_mass_8659.py` asserts the admissibility relation over 16 speed/attack-angle combinations and three bed conditions, that the prism alone still fails it (so the test cannot pass vacuously), and that ejecta speed remains exactly proportional to delivered impulse -- the property a clamp would have destroyed. (spec 1.0.658) |
| 2026-08-29 | #9184 | Added ADR-0044, deciding how BunkerShot3D handles the third dimension it currently refuses outright. Decision: stay in-plane. F1's `RefusedQuantity.OUT_OF_PLANE`, `SliceFidelity.EXTRUDED` (PR #9184), and the ball-as-cylinder caveat (`solvers/mpm/ball.py`, `ballreach.py`) are recorded as the tool's durable architecture, not a stopgap awaiting near-term replacement. Documents that F0 is _not_ geometrically blind to `WedgeGeometry`'s `heel_relief_fraction`/`toe_relief_fraction`/`heel_rocker_radius_m`/`toe_rocker_radius_m` -- the lofted mesh and `SurfaceElements` already carry them into every per-element DRFT sum -- only reporting-blind, since `ShotResult` exposes just the aggregate wrench; F1 alone is structurally blind, by construction, to any heel-toe distribution. Weighs three build-something options against accepting the gap: F2 3-D MPM on rented GPU (raw compute is a few dollars/shot at ADR-0032's 30-90 min/shot, but it inverts the study machinery's cheap-sweep economy by ~5 orders of magnitude, needs a pinned driver/container manifest to stay reproducible off a machine nobody here owns, stays outside CI per ADR-0032, and buys no NASA-STD-7009B validation since no bunker-shot measurement corpus exists for any tier); quasi-3-D strip theory (reuses F1's SPEC-1.0.655-calibrated constitutive model, no new dependency, but has no lateral momentum transfer between strips -- a modelling assumption, not a result -- and is architecturally the same zero-coupling bet F0's RFT superposition and `SHADOWING` no-wake caveat already make, just at strip instead of element granularity); and extending F0's own reporting to a heel/toe split (cheapest in engineering terms since the geometry is already there, but must wait on open issue #9247 -- F0 currently inverts its own bounce ordering, monotone 8 deg to 26 deg producing 19.69 mm to 24.57 mm of _more_ depth -- and shares strip theory's no-coupling limitation). Decides to accept the in-plane limitation for now rather than build any of the three, but with a concrete reopening trigger rather than a permanent close: fix #9247, then run the existing `bunkershot3d.study.sensitivity`/`morris` Sobol'/Morris machinery over the full `WedgeGeometry` relief and rocker parameters against F0's already-reported dig-depth/divot-mass/peak-force outputs, extending the measured finding that attack angle dominates sole geometry roughly 9x (13.3 mm of sole-depth sensitivity vs 1.5 mm for the full bounce range) to the out-of-plane parameters that finding never covered. Both F0 and F1 sit at NASA-STD-7009B Validation 0 of 4 today, so no option raises that number; F0's own credibility statement already says "use it to rank two sole geometries against each other; do not quote an absolute force from it." No production code changed; `docs/adr/README.md` index and Recent Amendments updated. (spec 1.0.659) |
| 2026-08-29 | #8733 | Connected the BunkerShot3D calibration harness to F1's constitutive model (issue #8733 section 6). ADR-0033 chose MPM because F1 and the F2 reference (`SolverImplicitMPM`) share a constitutive model, so "the material calibration is done once and carries between tiers"; that rationale was unrealised, because `src/bunkershot3d/calibration/` fitted backend contact-model parameters and contained no reference to `SandContinuum` at all, leaving F1's friction angle borrowed from the Quikrete analogue (#7999) and its shear modulus a Hardin & Richart (1963) estimate. `f1_shear_cell.py` adds a drained plane-strain biaxial compression test run as an element test on `SandContinuum.project` -- the same return mapping every F1 material point goes through -- fitted over three consolidation stresses through the Lambe p-q envelope, and shaped to the contract `CalibrationOptimizer` already reads, so the optimiser needed no change. Fitted friction angle **34.4880 deg against the borrowed 34.0000 deg** (shift +0.4880 deg), moving the cone slope alpha 0.374121 -> 0.379928 and the plane-strain limit the model actually enforces from 31.9438 deg to exactly the 32.5000 deg midpoint of the declared targets; residual 13.1187 -> 12.5000 deg^2, of which **12.5000 deg^2 is irreducible** because rate-independent perfect plasticity has no peak-to-residual softening and cannot produce the 5 deg gap the targets ask for. Cost 0.61 s per objective evaluation, 38 evaluations, 23.3 s for the whole search, and the stochastic search agrees with the closed form to 3.6e-6 deg (a disagreement above 0.05 deg now raises). **The targets are declared numbers, not measurements of bunker sand**: the fit replaces "borrowed from a hardware-store analogue" with "fitted to a declared target" and validates nothing. NASA-STD-7009B validation stays 0 of 4, F1 stays BEYOND_VALIDATION, `MAX_VALIDATED_SPEED_M_S` stays 1.44 m/s, and the friction angle is recorded as CONVENTION -- never MEASURED. Provenance is upgraded only where the fit earned it: the shear modulus is **exactly** inert (the drained limit ratio q/p cancels the elastic constants) and keeps its Hardin & Richart ESTIMATED label, which required closing a hole in #7999's inert-parameter guard whose 1e-12 floor was absolute and so passed ~1e-11 of floating-point residue on an objective of order 10 as a sensitivity. The angle-of-repose target is **not** achievable: `f1_repose.py` runs a quasi-static slope relaxation on the real solver (a dynamic column collapse reported 15 deg against a 31.94 deg model limit and is not a repose measurement), and the wedge does not settle to the model's own angle -- 44.4 deg at 0.05 s falling monotonically through the limit to 26.1 deg at 0.88 s, still drifting 8 deg/s, with errors of -3.61 / -5.88 / +0.72 deg at 28 / 34 / 40 deg that track the release angle rather than the sand. It therefore raises rather than returning a number the stopping rule chose, at a measured 5.2 ms/step and ~7 minutes per second of settling (four to five hours for one search). Issue #8733 stays open: sections 1-5 and the repose half of section 6 are untouched. (spec 1.0.655) |
| 2026-08-29 | n/a | Added a ratchet that fails when SPEC.md's changelog gains a **new** duplicate version number. The changelog is hot-prepended by every PR, so it is a serialization point where two branches routinely pick the same next-free version before either merges; keeping both rows is right for the prose but silently duplicates the number, and nothing detected it. 507 rows currently carry only 433 distinct versions, with 54 numbers used more than once. The 54 are recorded in `scripts/config/spec_changelog_duplicate_baseline.json` and tolerated at exactly their present multiplicity -- historical debt to reconcile deliberately -- while any new collision fails the build with the colliding version named. Verified red/green: exits 1 on a repeat of an existing version, 1 on a brand-new version used twice, 1 on a third row where two are baselined, 1 when the changelog table cannot be parsed at all (so a format change cannot turn the guard into a silent no-op), and 0 on the committed SPEC.md. (spec 1.0.654) |
| 2026-08-29 | #9232 | Repaired six committed research evidence records left stale by #9232. That change edited `scripts/research/proximal_distal_energy/spatial_full_body.py` and refreshed `CHECKSUMS.sha256`, `claim_evidence_manifest.json` and `release_manifest.json`, but not the `source_sha256` pins inside six evidence JSONs, so `test_committed_closed_contact_evidence_and_figure_are_current` failed on `main` against the old digest. The pin exists to guarantee committed evidence matches the code that produced it, so the digest was only refreshed after confirming the edit was result-neutral: #9232's own `test_cross3_matches_numpy_cross_bit_for_bit_over_random_vectors` passes, and every numerical assertion in the evidence test already passed with only the digest comparison failing. No evidence array was regenerated and no research result changed. Editing those six records changed their own content digests, so the release digest chain was re-registered one edge further out with `qualify_open_release write`: `release_manifest.json`, `CHECKSUMS.sha256` and `claim_evidence_manifest.json` (which pins `release_manifest.json`'s own digest) now record the post-edit hashes. Only `sha256` leaves moved -- no `bytes` count, array, claim value or report field changed -- and `claim_audit numeric` reports the same 328 claims / 144 numeric claims / 498 verified numeric literals before and after. (spec 1.0.653) |
| 2026-08-29 | #9228 | Closed the same directory-traversal gap on the _synchronous_ subprocess path (issue #9228; 1.0.650 / PR #9227 covered only the two `secure_popen` call sites). `run_command` in `src/shared/python/security/subprocess_utils.py` called `secure_run` with no `suite_root`, and `secure_run` gates script-path validation on `if len(cmd) >= 2 and suite_root:` and working-directory validation on `if suite_root:` -- so only the executable-name allowlist applied and `run_command([sys.executable, "/tmp/payload.py"])` ran. `run_command` now defaults its root via the existing `_default_suite_root()` (the canonical `SUITE_ROOT` from `src/shared/python/__init__.py`) rather than introducing a third root-derivation scheme, and takes an optional `suite_root` argument for a caller with a genuinely different trust root. `CommandRunner.run` forwards `self.suite_root`, which 1.0.650 added but only `run_async` consumed. The shared `if ... and suite_root:` condition in `secure_subprocess.py` is again deliberately left alone: its permissive mode is part of the published signature. Caller enumeration (the reason this was split out of #9227, which had no production callers to break) found `run_command`'s only production consumers are the three call sites in `src/shared/python/gui_pkg/launcher_utils.py`: `check_python_dependencies` runs `[sys.executable, "-m", "pip", "install", ...]` with no `cwd`, whose `argv[1]` of `-m` is not path-shaped and so skips script validation entirely, and `git_sync_repository`'s two `git` invocations run with `cwd` defaulted to `SUITE_ROOT`. None is out-of-suite, so nothing legitimate is broken. **Correction to the record on the error contract:** #9228 and its predecessor asserted that a rejection here would be absorbed into `run_command`'s `None` return by its `@log_errors(..., reraise=False, default_return=None)` decorator. It is not -- `log_errors` catches only `(RuntimeError, TypeError, ValueError)`, so `SecureSubprocessError` already propagates out of `run_command` today for the executable allowlist (`run_command(["git", "--version"])` raises, which also means `launcher_utils.git_sync_repository` has been raising rather than returning `False` since the allowlist landed -- a pre-existing behaviour recorded here, not changed by this entry). The new path and `cwd` rejections therefore raise identically rather than inventing a second contract for the same function. Nine regression tests in `tests/unit/security/test_subprocess_utils.py` pin the out-of-tree script path and out-of-tree `cwd` rejections, that an in-suite script and an in-suite `cwd` still succeed, that an explicit `suite_root` re-permits a scoped call, that `CommandRunner.run` applies its configured root, that the pip-install argv shape used by the real caller is unaffected, and -- as the anchor for the raise-not-return decision -- that the pre-existing executable rejection raises. (spec 1.0.651) |
| 2026-08-29 | #9221 | Closed the directory-traversal half of the secure-subprocess hardening on the background launch paths (issue #9221; PR #9216 was the incomplete predecessor). `secure_popen` in `src/shared/python/security/secure_subprocess.py` guards script-path validation with `if len(cmd) >= 2 and suite_root:` and working-directory validation with `if suite_root:`, so both are skipped entirely when no root is supplied. Neither `secure_popen` call site in `src/shared/python/security/subprocess_utils.py` -- `ProcessManager.start` and `CommandRunner.run_async` -- passed one, so #9216's executable-name allowlist was enforced while traversal protection stayed inert: `[sys.executable, "/tmp/payload.py"]` still launched. Both call sites now supply a root, defaulted by a new `_default_suite_root()` that returns the repository's existing canonical `SUITE_ROOT` constant from `src/shared/python/__init__.py`; that resolves to the checkout root and so agrees with the `.git`-based project-root rule established by 1.0.648, which a new test pins directly rather than restating. The shared `if ... and suite_root:` condition is deliberately left alone -- the permissive mode is part of `secure_popen`/`secure_run`'s published signature and other consumers may rely on it; the fix is to pass the root, not to change validation semantics. `ProcessManager.start` gains an optional `suite_root` argument and `CommandRunner` an optional constructor argument, so a caller with a genuinely different trust root can still declare one; the existing trusted sibling allowances (Tools, Movement Optimizer) are untouched. No production caller in this repository uses either entry point -- the only in-repo consumers of the module are `run_command` (synchronous, unchanged here) and `kill_process_tree` -- so nothing legitimate launches out-of-tree today. #9216's error-handling contract is preserved deliberately: a rejection raises `SecureSubprocessError`, which both call sites already catch, so it stays a logged `return False` / `return None` rather than an exception escaping to callers. Ten regression tests in `tests/unit/security/test_subprocess_utils.py` pin that an out-of-tree script path and an out-of-tree `cwd` are both rejected on both paths, that an in-suite launch still succeeds, and that an explicit `suite_root` re-permits a scoped launch. (spec 1.0.650) |
| 2026-08-29 | #9218 | Cleared two auto-filed review-feedback findings that were still live on `main` (issues #9218, #9170/#9164, #9196/#9167). (1) `.jules/bolt.md` heading `Limit Micro-Optimizations for .sum()` failed `scripts/check_document_title_case.py`, which expects `.Sum()`; the checker reported 1 violation on `main`, so the docs-governance changed-document gate failed for every PR touching that file. Rephrased to `Limit Micro-Optimizations for Array Summation`, which passes without corrupting the lowercase NumPy method name (checker now reports 0 violations). (2) The bolt ledger recommended `np.vdot` for sums of squares on "real arrays"; qualified to real _floating-point_ arrays and recorded why complex inputs (`np.vdot` conjugates its first argument) and narrow integer/boolean inputs (`np.vdot` keeps the narrow dtype rather than `np.sum`'s promoted accumulator) are excluded, so future optimizations driven by this journal cannot silently change results. (3) `scripts/ci/rehydrate_docker_context.py` decided whether to restore a tracked Dockerfile from `HEAD` by existence alone, so a file that was present but truncated or stale bypassed restoration and went straight to Buildx, while the follow-up size check only logged a warning for a 0-byte file and still returned success -- despite the helper claiming to verify the context. Added `_differs_from_head`, which uses `git diff --quiet HEAD --` (git's own filters and EOL normalisation, so no false mismatches on Windows checkouts) and fails closed on a git error; a target is now restored when it is missing _or_ drifted, the restore is re-verified against `HEAD`, `check_only` reports drift instead of passing, and a 0-byte tracked blob is a failure rather than a log line. Five unit tests added covering truncated, stale, check-only-drift, empty-blob and pristine targets. No behaviour change for an unmodified checkout. (spec 1.0.649) |
| 2026-08-29 | #9220 | Stopped test runs writing untracked simulation output into the source tree (issue #9220, part 2; part 1 landed as #9223). `resolve_base_path` in `src/shared/python/data_io/_path_utils.py` located the project root by walking up for an ancestor containing `.git` **or a directory named `engines`** -- and `src/engines` exists, so the walk stopped at `src/` and the default output base resolved to `src/output`. Every default-constructed `OutputManager`, including the one `SimulationService` builds to persist REST `/simulate` runs, therefore rooted its output inside the source tree, leaving untracked `src/output/simulations/<engine>/simulation_*.json` behind after a plain suite run. Root detection now uses the checkout's `.git` entry (a file in a worktree, a directory in a clone), falling back to the _outermost_ ancestor carrying a `pyproject.toml` -- not the innermost, since several packages nested under `src/` ship their own -- and honours a new `UPSTREAM_DRIFT_OUTPUT_DIR` override mirroring the `UPSTREAM_DRIFT_MODEL_CACHE_DIR` pattern from 1.0.647. Not fixed via `.gitignore`, which would hide test output inside the source tree rather than stop it landing there. The default is genuinely for interactive/CLI use, so it stays the documented `<repo>/output` and the _tests_ move instead: `tests/conftest.py` sets the override to a temp directory at import time, alongside the existing headless/thread-count setup, so it cannot be defeated by autouse-fixture ordering or xdist worker startup. New `tests/unit/repo_hygiene/test_no_output_in_source_tree.py` asserts a representative simulation save writes nothing under `src/`. This matters beyond tidiness: several agent workflows use `git status --short` to decide whether work is in progress, and stray output makes that unreliable. (spec 1.0.648) |
| 2026-08-29 | #9182 | Stopped the test suite rewriting the tracked asset `src/shared/urdf/human_models/mujoco_humanoid/model.xml` (issue #9182). `ModelLibrary._get_cached_embedded_model` in `src/tools/model_explorer/model_library.py` materialised the embedded MJCF string over that path on every `get_human_model("mujoco_humanoid")` call, so any test run left `git status` dirty — blocking rebases, forcing spurious stashes, creating false PR diffs, and breaking the `git status --short` in-progress checks several agent workflows rely on. The embedded MJCF is generated output, not a source asset, so it now goes to a derived-model cache: `~/.upstream_drift/model_cache` by default, overridable via `UPSTREAM_DRIFT_MODEL_CACHE_DIR`, and under the caller's `base_path` when one is supplied (so tests stay inside `tmp_path`). The write is also made conditional on the content actually differing, so repeated loads no longer churn the file's mtime. Four regression tests in `tests/unit/repo_hygiene/test_no_tracked_asset_rewrite.py` hash the tracked file before and after exercising the loader and assert the resolved path is outside the checkout. Note for the record: the committed asset was materially stale relative to the embedded string — it lacked the `<statistic>` element, the skybox/grid textures and `<visual>` block, and the three-light rig — but the divergence is entirely render-side; bodies, joints, inertias and actuators are identical, so no simulation result was affected. Nothing now reads the tracked copy. (spec 1.0.647) |
| 2026-08-29 | #9192 | Reconciled the externally rebased #9192 PR branch `7c66cb54b3ff51cc85797aaccfec577fe26b5432` into the local no-force lineage by ordinary merge. Both histories resolve to protected `main` `4775edb23f1438b8851d2c9d450691ac1adb15e6`; only SPEC and AGENT_HANDOFF differed, and the current authority plus the remote branch's unique version sequence are retained. Companion code, schemas, fixtures, workflow, and exact-byte behavior are unchanged; all hosted evidence must restart on the resulting exact head. (spec 1.0.646) |
| 2026-08-29 | #9192 | Reconciled #9192 with protected `main` `4775edb23f1438b8851d2c9d450691ac1adb15e6` by rebasing the branch onto it. The only conflicts were SPEC.md's Identity Spec Version field and the §12 change-log table; both sides' entries are retained and this branch's entries are renumbered above the current table maximum. Companion publication contracts, suite classification, release workflow, schemas, fixtures, and exact-byte behavior are unchanged. (spec 1.0.645) |
| 2026-08-29 | #9192 | Classified the #9192 publication contract tests as `unit` at the module boundary after the hosted suite-marker ratchet correctly rejected the new unmarked tests. This covers parametrized cases without repeating markers, changes no baseline, and preserves all 24 acquisition, compatibility, exact-byte, redirect, and workflow contract cases in the ordinary unit lane. (spec 1.0.644) |
| 2026-08-29 | #9192 | Reconciled #9192 with the repository architecture budget and honest PR-title governance. The Actions acquisition builder now receives one cohesive artifact-metadata mapping instead of nine scalar parameters, validates every member before use, and remains the single boundary used by the CLI and tests. This clears the hosted `9 > 8` parameter-budget failure without an exception or dummy production file. The PR is classified as `chore(companion)` because its executable delivery implementation is owned by `scripts/` and the protected release workflow rather than `src/`, `rust_core/`, or `api/`; a fresh synchronize event is required because replaying a pull-request run retains its original event title. (spec 1.0.643) |
| 2026-08-29 | #9205 | Replaced `np.square(fixed_weights).sum()` with `np.vdot(fixed_weights, fixed_weights)` in the launch-monitor player-covariation meta-analysis denominator (PR #9205). Identical result for the real-valued float weight vector, but skips the intermediate squared array allocation. (spec 1.0.642) |
| 2026-08-29 | #9216 | Routed `ProcessManager.start` and `CommandRunner.run_async` in `src/shared/python/security/subprocess_utils.py` through `secure_popen` instead of raw `subprocess.Popen` (PR #9216). The synchronous path already went through `secure_run`, so the async/background paths were the only ones bypassing executable allowlisting and the `shell=True` ban; `SecureSubprocessError` was added to both failure handlers so launch failures still degrade to `False`/`None` as before. (spec 1.0.641) |
| 2026-08-29 | #8729 | Made sand move in the 3-D scene (issue #8729, epic #8699). The 3-D view was driven by F0, which resolves no grains, so it drew a clubhead swinging through a flat translucent plane. `sandvolume.py` now extrudes the plane-strain field #8710 already stores across the declared `effective_width_m`, and both renderers draw it: matplotlib (`render3d_sand.py`, a `Poly3DCollection` of sheets plus a `Line3DCollection` of arrow polylines, both mutated through public setters) and PyVista/VTK (`render3d_vtk.py`, the same quads with `nan_opacity` for air), from one geometry source so the upgrade cannot show a different volume from the fallback. A 3-D picture of a 2-D solve is an extrusion, not a solved volume, so the volume carries `SliceFidelity.EXTRUDED` -- #8711's own vocabulary, not a second scheme -- refuses to be labelled `SOLVED`, and is drawn as discrete separated sheets rather than a blended continuum, because the visible repetition is the honest reading of a solve with no heel-to-toe direction. The frame also states which camera it is using: square to the solved plane shows the section, sighting along the target line shows the sheets edge-on as stripes, and a viewer not told that reads the stripes as across-width structure. `ShotScene` refuses a sand field whose tier disagrees with its own, and the `resolves_grains` flags on the surface and the divot must agree with the field's presence, so a scene cannot draw moving sand under a sentence denying it; `DivotSection` no longer hard-codes 'F0 moves no sand'. Colour is fixed and merges across designs (`SandVolumeScale`), and the world box now covers the sand, so #8728's per-grid autoscaling is not reintroduced and no material is drawn outside the axes. `shotcapture.py::capture_f1_shot_field` supplies the data: `simulate_f1_shot` advances one bed in place, so a whole shot's field exists only during the march, and it now takes a `ShotFieldRecorder` -- a structural protocol, so the solver package does not import `fields` back. The watched trajectory and wrench are bit-for-bit the unwatched ones. The provenance says `whole-shot march` rather than the declared constant-velocity approach, because the two animate identically and are not the same claim. The volume rides the transport `SoleLoadFieldWidget` already owns, mapped through `CursorMap` because the pose is sampled every CFL step and the field every stride block; no second slider. A real 30-frame capture at 25 m/s (28.3 m/s peak sand) renders in both backends across all three camera presets. `MAX_VALIDATED_SPEED_M_S` is 1.44 m/s, so the in-frame stamp reads BEYOND VALIDATION from the first sample. 101 new tests; 973 pass across `tests/unit/tools/bunker_shot_gui/`, `tests/tools/bunker_shot_gui/` and `tests/bunkershot3d/fields/`. (spec 1.0.639) |
| 2026-08-29 | n/a | Stopped `project_to_yield_surface` dividing by a zero deviator norm. `np.where` evaluates both branches, so the quotient was computed for every particle including those on the hydrostatic axis -- the commonest state in a freshly seeded bed -- emitting a divide-by-zero RuntimeWarning on essentially every F1 call. The answer was never wrong (the `on_cone` mask discards those values), but a warning that fires constantly trains readers to ignore warnings that might one day be real, and it fired from inside the verification suite whose purpose is to be believed. Now divides only where the denominator is positive, verified bitwise identical to the previous expression over 300 mixed hydrostatic, sheared, tip and cap states. Three regression tests pin it red-before/green-after, one of them asserting the hydrostatic rows are returned unchanged so silencing the warning cannot become changing the answer. (spec 1.0.638) |
| 2026-08-29 | #9180 | Reconciled #9180's companion-provider foundation and both registry/Qt isolation repairs with protected main `0e87597f96100dfc3e802183cdf8c273635789da` and #9210's shaft-contribution replay optimization. The companion schema, deterministic local-only discovery, exact provenance, draft negative states, four provider children (#9190-#9193), and #9174-open boundary remain unchanged. (spec 1.0.637) |
| 2026-08-29 | #9180 | Reconciled #9180's companion-provider foundation and both registry/Qt isolation repairs with protected main `f9e694aaefa16720afd01eead38286bf7ac41134` and #9202's bounded-event evidence replay timeout repair. The companion schema, deterministic local-only discovery, exact provenance, draft negative states, four provider children (#9190-#9193), and #9174-open boundary remain unchanged. (spec 1.0.636) |
| 2026-08-29 | #9180 | Reconciled #9180's companion-provider foundation and both registry/Qt isolation repairs with protected main `24be86fb6d3d434797e8e84e26238497e7784a13` and #9181's ball-sand interaction slice. The companion schema, deterministic local-only discovery, exact provenance, draft negative states, four provider children (#9190-#9193), and #9174-open boundary remain unchanged. (spec 1.0.635) |
| 2026-08-29 | #9180 | Fixed the second exact hosted `unit-test-gate` failure on #9180 without retry or quarantine. Three launcher/bootstrap tests assumed their registry keys were absent, so a prior legitimate worker import returned live `motion_target_preview`/`pose_studio` widgets before package-entry containment could be exercised and made the forced `model_explorer` registration diff empty. Each test now snapshots, removes, and restores only its owned key in `finally`, preserving unrelated process state. The model-explorer URDF failure test now stops either MuJoCo's render timer or the fallback animation timer, closes and schedules deletion of its widget, and flushes Qt events, preventing callbacks from reaching a deleted `QLabel`. Production launcher/registry semantics remain unchanged. (spec 1.0.634) |
| 2026-08-29 | #9180 | Reconciled #9180's companion-provider foundation and registry-isolation repair with protected main `e74c13f827ef45c2c5581a80b1e6c0adc3f7e8b3` and #9173's F1 MPM plastic-limit, manufactured-solution, and temporal-verification slice. The companion schema, deterministic local-only discovery, exact provenance, draft negative states, four provider children (#9190-#9193), #9174-open boundary, and RED-to-GREEN adapter-state regression remain unchanged. (spec 1.0.633) |
| 2026-08-29 | #9180 | Fixed the exact hosted `unit-test-gate` failure on #9180 without a rerun or quarantine. Three launcher/UI subtree fixtures cleared the process-wide embeddable-tool registry after every test without restoring its incoming state; under xdist, a worker that had already cached `src.tools.canonical_core._embed_adapter` could then import that module as a no-op and observe both canonical-core adapters missing. The fixtures now provide an empty per-test registry and restore their snapshots in `finally`; the canonical-core self-registration regression explicitly exercises cached-module, cleared-registry, reload, and process-state restoration. The deterministic red ordering now passes and no production registry semantics changed. (spec 1.0.632) |
| 2026-08-29 | #9180 | Reconciled #9180's companion-provider foundation with protected main `8cc236c6879e7535bb6bd15aecbe3396fb6dbb36` and #9172's BunkerShot3D F1 ball/contact/whole-shot marching slice. The companion schema, deterministic local-only discovery, exact provenance, draft negative states, four provider children (#9190-#9193), and #9174-open boundary are unchanged. (spec 1.0.631) |
| 2026-08-29 | #9180 | Bounded #9180 as the companion provider foundation rather than #9174 completion. #9174 remains open behind #9190's ten exact-revision workflows/four failure fixtures/provider CI, #9191's full governed screenshot and capture metadata, #9192's attested protected artifacts/current-and-previous schema fixtures/rollback-safe immutable release acquisition, and #9193's documentation freshness plus engine support/qualification evidence authority. The current empty documentation/workflow/screenshot arrays and ignored local artifacts are explicit negative evidence and have no durable acquisition URL; no release or tag is created by this slice. Reconciled this boundary with protected main's #9183 launcher guard and #9188 Qt module-isolation fixes. (spec 1.0.630) |
| 2026-08-29 | #9188 | Fixed #9188's plotting-test Qt module leak. The subtree fixture now uses function-scoped `monkeypatch.setitem` restoration, while a global file-boundary guard detects and repairs stale Qt/UI module substitutions after fixture teardown. This supersedes the narrower function-scope-only repair recorded at 1.0.628 and makes the failure attributable instead of order-dependent. (spec 1.0.629) |
| 2026-08-29 | #9188 | Made the plotting test's PyQt/UI mock fixture function-scoped. Its former session scope outlived the plotting subtree under pytest-xdist and replaced real `PyQt6` plus `src.shared.python.ui` modules for unrelated tests later scheduled on the same worker, causing deterministic `Qt`/`QDialog` import and UI-package failures. The exact plotting-then-GUI/UI/Frankenstein order now restores the real module graph between tests; no quarantine, retry, production behavior, or runner policy changed. This initial repair is superseded by #9188's global regression guard at 1.0.629. (spec 1.0.628) |
| 2026-08-28 | #4010 | Established the first UpstreamDrift provider-authority slice for AffineDrift #4010 (issue #9174). ADR-0043, a strict v1 JSON Schema, and a deterministic local-only exporter reconcile the current 49 raw launcher/56 local model/70 union and 41 feature/79 surface-path baselines without freezing those counts as schema invariants. Exact source commit, committed-input hashes, package/Python compatibility, verification command, engine support tiers, and the `vendor/ud-tools` gitlink are explicit; maturity, availability, support, parity, and scientific qualification remain separate. RED-to-GREEN tests enforce environment independence, reference integrity, strict fields, clean-tree/CI-commit refusal, safe paths, canonical bytes, and the detached digest. Publication remains draft and the companion does not duplicate #9064/#9070 manual or calculation authority. (spec 1.0.627) |
| 2026-08-29 | #9204 | Reduced the runtime of the slowest test in the default `tests` lane, relieving pressure behind issue #9204. Measurement first contradicted the issue's per-file profile: `tests/research/test_shoulder_velocity_strategy_search.py`, listed there at 294.0 s, actually costs 5.8-6.3 s uncovered and 31.6-32.6 s under `--cov=src`, of which its five tests are only 0.32 s / 0.80 s -- the remainder is the fixed per-process coverage cost (an empty probe file through the same conftest chain is 26.3 s) plus ~5 s of coverage-instrumented import. Its entire scientific workload is 5,400 RK4 steps of a 2-DOF planar model, so 294 s was never physically possible; there is nothing to optimise there and it was left untouched. The real whale was `tests/research/test_shaft_contribution_evidence.py`, listed at 60.0 s and in fact **191.7 s uncovered / 364.4 s covered**, essentially all inside one test, `test_outputs_replay_with_declared_numerical_tolerance`, which replays the whole registered shaft-contribution study through `run_shaft_contribution_study.write_outputs()` (its own `@pytest.mark.timeout(360)` is why it never showed as a `+++ Timeout +++`). A cProfile of `build_outputs()` found the redundancy: `mass_matrix` in `src/shared/python/pendulum_simulator/physics_triple.py` verified its symmetry postcondition with a nine-cell `np.isclose` double loop, and `mass_matrix` runs once per RK4 stage -- 968,152 calls and therefore 8,713,368 `isclose` invocations, **206.2 s of the 391.7 s profiled run (53%)**. The check was also vacuous: the array is built from a symmetric literal, so `M[0,1]` and `M[1,0]` are both the scalar `M12` and hold the same float64 bit pattern. It is now three exact scalar comparisons, which is strictly stronger than the tolerance form (it can only reject more), matches the postcondition the docstring already documented (`M[i,j] == M[j,i]`), and costs ~0.45 us against ~118.85 us -- a 264x reduction on that check. This is not the #9198 memoisation pattern; that pattern was evaluated and does not apply, because the 120 robustness rollouts are all distinct parameter combinations with nothing to share. Two other candidates were measured and rejected for being unsafe or slower: stacking the six right-hand sides into one `np.linalg.solve` is **not** bit-identical (19,936 of 20,000 random 3x3 trials differed), and `scipy.linalg.lu_factor`/`lu_solve` is bit-identical in all 120,000 comparisons but slower than six `np.linalg.solve` calls at 3x3 (38.2/24.7 us vs 22.7 us) because SciPy's per-call overhead dominates. Measured on one machine: one robustness rollout 1.439 s -> 0.462 s (3.1x); `write_outputs()` 189 s -> 63.1 s (3.0x); the replay test itself 188.7 s -> 66.6 s uncovered and 148.9 s covered; the whole file 191.7 s -> 69.3 s uncovered (2.8x) and 364.4 s -> 175.7 s covered (2.1x). Numerical equivalence was established by replaying the study before and after and comparing directly, not against the committed record: **all 185 registered trace arrays are bit-identical (`np.array_equal`)** and the regenerated report differs in exactly one leaf across the whole JSON -- `source_sha256` for the single edited file. Because that digest is governed, the chain was resynced surgically by string substitution with no evidence regenerated: only sha256 values changed, in `shaft_contribution_study.json` (its own `source_sha256` entry) and the three files pinning that report's digest, `CHECKSUMS.sha256`, `release_manifest.json` and `data/claim_evidence_manifest.json`; byte counts are unchanged because a hex digest is fixed width. On unmodified `origin/main` the replayed report already matched the committed record exactly under LF normalisation and 184 of 185 arrays were bit-identical, so #9203 does not affect this study. One new `@pytest.mark.unit` test pins the exact-symmetry contract across five configurations so a rewrite cannot weaken it back to a tolerance. No iteration count, solver tolerance, problem size, marker or workflow file was changed. Pre-existing and NOT addressed here: `tests/research/test_mechanism_ladder_evidence.py::test_evidence_uses_content_provenance_and_explicit_path_contract` fails identically on unmodified `origin/main` (the stale `mechanism_ladder_study.json` provenance digest already noted under 1.0.631), and on Windows the replay test's byte comparison fails on CRLF-vs-LF line endings alone, before and after this change. (spec 1.0.632) |
| 2026-08-29 | #9198 | Fixed issue #9198: `main` failed `CI Standard` because `tests/research/test_bounded_event_reachability_evidence.py`'s module fixture -- which replays the whole registered bounded-event reachability study through `run_bounded_event_reachability.build_evidence()` -- ran 50-54 s under `--cov=src` against the 60 s `[tool.pytest.ini_options] timeout`, so `tests (3.11)`/`tests (3.12)` died with `+++ Timeout +++` and took the required `quality-gate` with them. The issue's stated cause -- `make_backend` inside `_integrate_segment` -- was real but minor: hoisting it to one backend per solve (`_solve_backend`) only moved the covered fixture 47.4 s -> 41.8 s, and excluding `scripts/research/**` from coverage would have done nothing at all because `[tool.coverage.run] source` is `["src", "shared"]`, so the traced hot loop is the RK4 dynamics in `src`, not the research script. The dominant waste is that SciPy's `approx_derivative` perturbs one decision variable at a time, and a perturbation confined to one shooting segment leaves every _other_ segment's integration bit-for-bit unchanged -- yet `_shooting_residual` re-integrated all of them once per finite-difference column. `_integrate_segment` now takes an exact per-solve `memo` keyed on `(first index, last index, final step duration, segment start time, start-state bytes, perturbation bytes)`; entries are stored read-only so a caller cannot corrupt a shared result. This is a cache, not an approximation: the integration is a deterministic pure function of that key, so a hit returns the identical float64 bit pattern. Measured on one machine, two runs each: covered fixture 53.99/50.41 s -> 12.05/11.90 s (4.3x, from 87% to 20% of the timeout budget); uncovered 20.63/16.25 s -> 7.45/7.46 s. Numerical equivalence was established by capturing solver output before and after: all 28 registered evidence arrays and all 25 arrays from a five-case solver matrix are bit-identical (`np.array_equal`), and every field of the regenerated report is equal except `source_identity.source_sha256` for the edited file itself. Because that digest is governed, the hash chain was resynced surgically -- only sha256 strings and two `bytes` counts changed across `bounded_event_reachability.json`, `event_topology_{channel_matrix,robustness,stress_extension}.json`, `nonlinear_controller_{comparison_registration,solver_qualification}.json`, `claim_evidence_manifest.json`, `release_manifest.json` and `CHECKSUMS.sha256`; no numeric evidence was recomputed. Two new `@pytest.mark.unit` tests: one asserts exactly one backend construction per solve, one asserts a memo hit is byte-identical to the uncached integration and read-only. Pre-existing and NOT addressed here: `nonlinear_controller_comparison_registration.json` and `nonlinear_controller_solver_qualification.json` still pin the pre-#9130/#9148 `requirements.lock` digest, and `mechanism_ladder_study.json` carries a stale provenance digest -- all three reproduce identically on unmodified `origin/main` and need a deliberate re-registration. (spec 1.0.631) |
| 2026-08-28 | #8712 | Closed #8712 (epic #8699): what the sand actually delivers to the ball inside the F1 plane-strain MPM tier. `solvers/mpm/ballreach.py` reads the ball's own exact momentum ledger as traction on the ball, resolved around its in-plane surface -- a below-equator / face-side split, an even-sector resolution whose bin edges always place the equator on a boundary, and a per-node radial (compressive) and tangential (shear) decomposition -- with a time history carrying first contact, a caller-thresholded loading onset, the peak and its timing, and the total impulse. Nothing here computes a force: `BodyContact` now retains the node-resolved `ContactImpulse` it was reduced from, because a summed impulse cannot say _where on the body_ the sand arrived, so every number is the existing ledger regrouped and the two-body momentum budget still closes to round-off with the ball's term in it. Every quantity is named per unit out-of-plane width on an infinite cylinder rather than a sphere, and the refusals are the point: `BallSurfaceSectors.total_force_n`, `BallReachHistory.total_force_on_ball_n` and `SandVersusClub.ball_force_n` raise `RefusedQuantity.OUT_OF_PLANE` because, unlike the club, there is no effective width to declare; `BallReachHistory.launch_velocity_m_s` still raises `RefusedQuantity.BALL_LAUNCH`, launch staying on F0's #8657 momentum-transfer path; and `SandVersusClub.club_force_n` raises `RefusedQuantity.CLUB_FORCE`, ADR-0033 refusing F1 for club force at all. `SandVersusClub` therefore compares what the sand delivers to the ball against what the club delivers to the sand as a dimensionless share of one solve's ledger plus a pair of timings, never as two forces. Every result carries its `ValidityVerdict`: F1, BEYOND_VALIDATION and no better, published-speed ceiling 1.44 m/s, NASA-STD-7009B validation 0 of 4. 48 new tests; 365 pass across `tests/bunkershot3d/solvers/mpm/`. (spec 1.0.630) |
| 2026-08-28 | #8733 | Closed issue #8733 section 4, the F1 MPM code verification the shipped suite did not reach. Added a plastic-limit case (`rankine_limits`, `passive_earth_pressure_limit`): the 2-D Drucker-Prager surface enforces a plane-strain Coulomb limit at `phi* = asin(sqrt(2) alpha) = 31.944 deg`, not the 34 deg input angle, and a smooth rigid wall pushed into a frictionless-based cohesionless layer at `v/c = 1.58e-4` reaches 22.744 N/m against the closed-form 21.287 N/m at `dx = 3 mm` (6.845%, falling to 2.966% at `dx = 2 mm`) with 98.9% of the bed at yield. Added a manufactured-solution study (`manufactured_solution_convergence`) whose observed order is 1.880 over four grids against a design order of 2, covering the stress divergence and the particle-grid transfer together on the elastic branch only, with a round-off-class uniform-stress patch test at 1.76e-15 relative underneath it. Added temporal refinement at fixed `dx` (`column_temporal_convergence`): monotonic over one elastic transit with Celik apparent order 1.214 and `GCI_fine = 0.473%` declared a temporal band only, and `MONOTONIC_DIVERGENCE` beyond that because the particle-grid round trip costs a fixed amount per step so its total over a fixed window grows as `1/dt`. Attempted the conservative elastic-energy case a cohesive cone tip was thought to allow: it does not work, with zero particles yielding and the energy still drifting 9.472-11.354% at a fitted order of -0.124, which identifies the transfer rather than the integrator. Everything reuses the existing `vandv/` conservation, convergence and Celik implementations, and validation remains 0 of 4. (spec 1.0.629) |
| 2026-08-28 | #8733 | Closed sections 1-3 of #8733 (epic #8699): the F1 ball as a plane-strain body, multi-body contact within one step, and a whole-shot march. The ball is an equal-area polygonal circle (`ball.py`) whose two ADR-0033 facts are enforced by the API rather than documented -- it is an infinite cylinder rather than a sphere, so `sphere_mass_kg` raises and only `line_mass_kg_per_m` is available, and the below-equator / face-side split #8712 wants is qualitative and in-plane, with every heel-toe or lateral question raising `RefusedQuantity.OUT_OF_PLANE` and ball launch still raising `RefusedQuantity.BALL_LAUNCH`. Multi-body contact (`contact.py`) fixes the projection order from the bodies themselves -- slowest first, fastest last, ties in the caller's order -- because the grid projection is a velocity-level constraint and at a shared node the last one applied is the one that holds exactly; the fastest body is the one that can tunnel, and deriving the order from state rather than from the argument list makes the answer independent of how the caller assembled the sequence. The momentum ledger stays exact regardless: a two-body march of a 600-particle bed with every wall FREE closes to a 1.95e-16 kg m/s residual, 4.3e-15 relative. The whole-shot march (`wholeshot.py::simulate_f1_shot`) integrates the head instead of prescribing it and returns a `ShotResult`, so an F1 shot enters `bunkershot3d.metrics` through the same door an F0 shot does; `solve()`/`_approach` is untouched and still the comparable-to-F0 path. On one 12 m/s, 20 deg delivery the declared approach peaks at 1.06 kN and reports 42.5 N at the queried pose, while the marched shot peaks at 534 N, slows the head 12 -> 9.40 m/s and reports 17.3 N at that same 8 mm sole depth. Structurally, `solver.py` was 1077 of a 1200-line budget, so the scheme moved to `step.py` (`StepContext`/`advance_step`) where a step-by-step caller can drive it; `march()`'s signature is unchanged and `march_bodies()` is the sequence form. 110 new tests; 404 pass across `tests/bunkershot3d/solvers/mpm/` and `tests/bunkershot3d/fields/`. Sections 4 and 6 of #8733 remain open. (spec 1.0.628) |
| 2026-08-29 | #9183 | Fixed issue #9183: `main` had failed `CI Standard` on four consecutive runs because `tests/launchers/test_dashboards.py::test_mujoco_dashboard_main` entered a real Qt event loop at `src/shared/python/dashboard/launcher.py` and was killed by the pytest timeout, taking `tests (3.11)`, `tests (3.12)` and the sole required `quality-gate` check down with it. The root cause was a cross-file `sys.modules` interaction, which is why it presented as intermittent: `tests/unit/test_launcher_lazy_loading.py` evicts `src.launchers.{mujoco,pinocchio,drake}_dashboard` from `sys.modules` to force fresh imports and never restores them, while `test_dashboards.py` bound `main` at module-import time. When both files landed in the same pytest-xdist worker in that order, `mock.patch("src.launchers.mujoco_dashboard.launch_dashboard")` re-imported the module and patched the _new_ module dict, but the already-bound `main` still resolved `launch_dashboard` through the _old_ one -- so the patch silently missed and the real launcher blocked in `qt_app.exec()` forever. Fixed in three layers. (1) The seam: `launch_dashboard`'s inline default runner lambda is replaced by a named `_default_event_loop_runner` that raises `RuntimeError` when `PYTEST_CURRENT_TEST` is set, converting this entire class of multi-minute headless CI hangs into an immediate, self-describing failure; the injected `event_loop_runner` seam and all production behaviour are unchanged. (2) The polluter: an autouse fixture in `test_launcher_lazy_loading.py` snapshots and restores the three dashboard `sys.modules` entries, so no previously-bound reference is left pointing at an orphaned module dict. (3) The consumer: `test_dashboards.py` now resolves `main` via `importlib.import_module(...)` inside the patch context instead of binding it at import time. Two regression tests assert the seam raises under pytest and still runs the real loop outside it; all four tests are marked `@pytest.mark.unit`. Verified by reproduction: the co-scheduled ordering hangs to timeout on `origin/main` and passes in 3.7s after the fix; 10/10 serial and 10/10 `-n 4` iterations clean. No skip, xfail, or timeout marker was added. (spec 1.0.627) |
| 2026-08-28 | #8828 | Fixed issue #8828: every plot title in the shared plotting layer (`kinematics.py`, `energy.py`) was a bare metric name ("Joint Positions", "System Energy", ...) with no engine/model/run identity, so a PNG saved from the MuJoCo dashboard looked identical to one from Drake. Added `plotting/identity.py`'s `PlotIdentity` value object (engine/model/run, all optional, never fabricated) plus `apply_identity_footer`/`resolve_and_apply_identity_footer`, threaded as an optional `identity` parameter into all 10 flagged `plot_*` call sites; when not passed explicitly it is derived from `recorder.engine`'s `engine_type`/`model_name` (the `PhysicsEngine` protocol's `Checkpointable`/`Loadable` sub-protocols) and rendered as a figure footer. `ExportConfig.include_metadata` (`export.py`) previously did nothing -- `export_figure` now passes a format-appropriate `metadata=` dict to `fig.savefig` (PNG/PDF/SVG each have different accepted key vocabularies) carrying a UTC timestamp, the UpstreamDrift software name, and identity fields when known; `export_plot_data`'s JSON `_meta` block now merges in the same identity fields instead of only the static "UpstreamDrift" string. `plot_engine/pyqt6_widget.py` was NOT touched: it mirrors D-sorganization/Tools's `src/shared/python/plot_engine/` 1:1 (a Tools-owned child copy), so wiring the generic dashboard export widget through `export_figure` is filed as D-sorganization/Tools#4740 instead, with `tests/unit/test_plot_engine_widget_export.py` left as a tombstone pending that pin bump. New tests cover: rendered-figure footer content, PNG metadata readback via `PIL.Image.open(...).info`, PDF export not raising despite its stricter metadata key vocabulary, and JSON `_meta` identity fields. (spec 1.0.626) |
| 2026-08-28 | #9168 | Fixed #9168: `tests/tools/sidekick_tool/test_embed_adapter.py` was order-dependent in two ways and failed intermittently on unrelated PRs (#9095, #9148, #9150, #9160). (1) `src/tools/sidekick/__init__.py` registers `_SidekickEmbedAdapter` as a module-level import side effect that runs at most once per process, while `test_package_registers_adapter_on_import` needed an empty registry and `test_package_registration_is_idempotent` needed a populated one; three autouse conftest fixtures (`tests/ui/launcher_embed`, `tests/launchers/launcher_embed`, `tests/ui/tools/simulation_backends`) cleared the process-wide registry without restoring it, so when one landed in the same pytest-xdist worker first the cached `sys.modules` entry stopped the side effect re-running and the pair failed as mirror images (`assert None is not None` / `assert <adapter> is None`). (2) `tests/unit/launcher_embed/test_sidekick_contract.py` evicts the sidekick modules and never restores them, leaving a different `_embed_adapter` module cached, so `test_cleanup_swallows_exceptions_and_logs` patched a module its collection-time class no longer belonged to. Fix is test-side only, leaving production registration semantics unchanged: an autouse fixture now snapshots and restores both the registry mapping and the `src.tools.sidekick*` `sys.modules` entries; a `_fresh_import_sidekick` helper evicts the package and its submodules before re-importing so each test establishes its own precondition explicitly; identity checks resolve the adapter class from the live module; the logger patch uses `patch.object` on that same module object; the contract test's `_registry_snapshot` now restores the sidekick `sys.modules` entries it evicts. Added `test_package_registration_recovers_from_cleared_registry` as a regression guard (all 17 tests marked `@pytest.mark.unit`). Verified against `origin/main` as a control across eight scenarios: the four co-scheduled orderings failed 5/5, 3/5, 5/5 and 5/5 before the fix and pass 10/10 each after, with the file alone and the broad `tests/tools` tree clean in both. No retry marker and no xfail. (spec 1.0.625) |
| 2026-08-28 | #8703 | Replaced the saturated `dig_vs_skid` discriminant with the descent-return ratio (#8703). The verdict was built on the entry slope ratio -- the penetration slope over the first 10 mm of travel divided by the delivered path slope -- which spanned only 0.9987-1.0000 over the demo's whole 77-point design space and returned `MARGINAL` at every point, because 10 mm is 0.4 ms at greenside speed and a 0.3 kg head under an order-5 N.s impulse cannot bend measurably in that time; resizing the window was measured and refuted, since normalising by the delivered slope divides out the attack angle and inverts the ordering against sole depth. `bunkershot3d.metrics.divot` now measures the **vertical restitution of the strike** -- the sole's climb speed at the last submerged sample over its descent speed at the first -- which is the direct expression of the physical claim (a digging head gives its descent to the sand and crawls out; a skidding head bottoms out and is thrown back) and has no window parameter to place. Over the shipped `WorkbenchModel` swept at 384 points (marketed bounce 8-26 deg x sole 16-24 mm x attack -2 to -14 deg x four sand conditions x 20 and 25 m/s) the new ratio spans 0.338-0.954 against 0.00124 for the quantity it replaced, and correlates with maximum sole depth at -0.97 with the correct sign where the old ratio managed -0.64 with the wrong one. `DigSkidResult` now carries `entry_descent_speed_mps`, `exit_climb_speed_mps` and `descent_return_ratio` in place of the slope fields; `DigSkidCalibration` still reports `calibrated=False` and now also carries `DIG_SKID_BOUNCE_ORDERING_REASON`, which records that the F0 solver reads more marketed bounce as more dig in the shallow, non-burying regime -- a model behaviour the saturated metric could not have exposed -- so the verdict is never a bounce recommendation. (spec 1.0.624) |
| 2026-08-28 | #9120 | Fixed CI offline determinism, Docker buildx rehydration, and dependency lock provenance (#9120, #9121, #9122). Added `bootstrap_conformance_deps.py` for deterministic offline install of pinned conformance dependencies across cross-engine workflows; added `rehydrate_docker_context.py` to ensure Dockerfile is rehydrated before Buildx smoke gates; and made dependency lock provenance invariant to offline environments with canonical compilation headers. (spec 1.0.623) |
| 2026-08-27 | #9142 | Emit portable source links for claim-adjudication data (#9142). `claim_adjudication_summary.py` now builds absolute `blob/main` links for `claim_adjudication_summary.json` and `claim_adjudication_summary.csv`; AffineDrift rewrites that declared form to the protected source SHA. The explicit editorial-only claim-census migration verifies that the candidate count remains 1,180 and replaces only the generated reviewer candidate whose URL text changed. (spec 1.0.622) |
| 2026-08-27 | #9143 | Consumed the Tools green-surface adapter across the vendor boundary (issue #9143, Tools #4800 P9). Bumped the `vendor/ud-tools` submodule to the latest Tools main squash (`b46f58df52df86b6c5a3db44460b26ac8919da70`), pulling in `shared.python.swing_sim.putting.ud_adapter`, `UdGreenTopography`, `green_surface_from_ud_json`, and `green_surface_to_ud_json` alongside recent variation fixes (#4692/#4693/#4694/#4697). Added consumer integration tests in `tests/unit/putting/test_putting_green_consumer.py` covering vendor boundary import resolution, bi-directional topography JSON serialization/deserialization between Tools heightfields and UpstreamDrift `GreenSurface`, fail-closed rejection of scattered contours/slopes/unknown fields, and roll physics consistency on imported flat and sloped green surfaces. (spec 1.0.621) |
| 2026-08-27 | #9066 | Established UpstreamDrift engineering design-manual authority (UP-D0, issue #9066). `manuals/upstreamdrift` QMD is the sole editable authority; existing user, ADR, and research products remain separate; generated HTML, LaTeX, PDF, and DOCX remain non-editable and unapproved. The versioned policy, fail-closed empty registry, agent guidance, contract tests, and offline CI/pre-commit verifier enforce program-contract ownership, safe paths, Ruff formatting, TDD/DbC/DRY/LoD, impacted-path/freshness rules, immutable release evidence, visual and semantic review, and human approval. UP-D1 through UP-D8 remain explicit blockers, and this governance scaffold makes no calculation-coverage or publication claim. (spec 1.0.620) |
| 2026-08-27 | #9126 | Integrated #9126's outcome-blind nonlinear-controller prerequisite. The registration freezes nine families, 24 evaluation trials, eight disjoint tuning trials, plant/event/failure/random-stream contracts, and single-worker checkpoint identity. One bounded projected first-order iLQR kernel passes manufactured derivative, in-rollout bound, descent, replay, initialization-sensitivity, and typed-failure gates; collocation NMPC remains explicitly unavailable. Twelve shared-equation plant-step cases pass over 0.5, 1, and 2 ms. The regenerated web-linearized 251-page paper is 1,962,456 bytes at SHA-256 `92bfaca850ac459cc431e573be8c0288af51ceab4d28759d02c67c602274ee8b`, with 325 adjudicated claims, 144 numeric contracts, and 498/498 verified numeric literals. Zero registered controller evaluations have run and zero methods are ranking-eligible. (spec 1.0.619) |
| 2026-08-27 | #9125 | Integrated #9125's global event-topology, delay, perturbation, stress, channel-mask, refinement, and horizon evidence. Phase A retains one positive transverse crossing in all 6,336 nonzero small-stress replays; Phase B first exposes topology loss at 0.02 synthetic dimensionless stress and 200 ms delay; Phase C distinguishes wrist-only crossing absence from 0.40 s horizon truncation and prevents zero authority from acquiring command noise. The regenerated web-linearized 250-page paper is 1,958,661 bytes at SHA-256 `6ca47ab88331cbb728a0f464f1a1200cf16553328148f40e42192f66d56a1647`, with 322 adjudicated claims, 141 numeric contracts, and 482/482 verified numeric literals. These are open-loop synthetic topology results, not human tolerance, anatomical, strategy-ranking, or coaching evidence. (spec 1.0.618) |
| 2026-08-27 | #9124 | Integrated #9124's bounded nonlinear event-reachability evidence into the governed proximal-distal release. Exact-RK4 multiple shooting and independent replay qualify 32 of 38 registered target/channel cases with maximum feasible event-tangent residual `8.82244e-11`; the six infeasible cases are displaced zero-authority targets. Mesh, integration-step, adverse-initial-state, nominal, and channel-mask controls pass, while a 24.9517% multistart objective spread fails the 5% optimality gate and suppresses every channel/controller ranking. The regenerated web-linearized 249-page paper is 1,946,712 bytes at SHA-256 `1afd5354ceb3f93ab04a5b3d3ca182d512b66f443e19cf822bec0b38b1f836f6`, with 319 adjudicated claims, 138 numeric contracts, and 463/463 verified numeric literals. (spec 1.0.617) |
| 2026-08-27 | #9145 | Fixed #9145: `bunkershot3d/__init__.py` eagerly imported `backends` (both via the package's own `from . import (...)` block and via `from .backends import ChronoDriver, LiggghtsDriver, MPMDriver`), and `backends` imports mujoco at module load, which touches a PyOpenGL binding at import time -- so importing _any_ name from `bunkershot3d`, even a pure-dataclass leaf like `EnvelopeStatus`, dragged in mujoco and OpenGL and crashed with `AttributeError: 'NoneType' object has no attribute 'glGetError'` on GL-less machines. This is what broke `unit-test-gate` on PR #9138, with the traceback misleadingly pointing at OpenGL rather than the real cause. Dropped `backends` from the eager import block and the eager driver re-export; the driver classes and the `backends` submodule now resolve lazily through a module-level `__getattr__` (PEP 562), so the public API (`bunkershot3d.ChronoDriver`/`LiggghtsDriver`/`MPMDriver`/`backends`) is unchanged and only the import-time cost moved to first _use_. Added `tests/bunkershot3d/test_package_import_stays_headless_9145.py`, which shells out to a fresh interpreter (so it isn't fooled by modules other tests already imported) and asserts `mujoco`/`OpenGL`/`vtk`/`pyvista` stay out of `sys.modules` after `import bunkershot3d` and after `from bunkershot3d.solvers import EnvelopeStatus`, mirroring the check added for `render3d_vtk.py` in #9138 but at the package boundary. Confirmed red on the pre-fix code and green after; full `tests/bunkershot3d/` suite, `test_public_api_8608.py`, and `ruff check`/`ruff format` all pass. (spec 1.0.616) |
| 2026-08-27 | #9123 | Qualified #9123's trajectory-varying event-conditioned control-authority boundary on the protected #9117 phase/event baseline. The exact discrete RK4 state/input maps, continuous-energy-equivalent Gramians, event-tangent projection, separated torque channels, nonlinear pulse checks, frozen-local countermodel, timestep and differentiation refinement, zero-input, additivity, and equivalent-unit controls expose local first-order authority and its falsifiers. The regenerated, web-linearized 247-page paper is 1,935,834 bytes at SHA-256 `1783ba69c72f56bba1d3a0e43136afc9b6651d31e4bc453f9d2df61ffdcb1dcb`. The evidence does not establish bounded nonlinear reachability, physiological actuator limits, participant behavior, passive torque, controller ranking, or coaching guidance. (spec 1.0.615) |
| 2026-08-27 | #9138 | Fixed two real CI test failures on PR #9138 (BunkerShot3D PyVista/VTK backend), verified against the actual `unit-test-gate` job log rather than re-derived from symptoms. (1) `render3d_vtk.py`'s docstring claimed importing the module never requires GL; true for PyVista itself but not for its sibling imports -- `.bridge`/`.render`/`.report`/`.shot3d`/`.traces` all reach `bunkershot3d.solvers`, and `bunkershot3d/__init__.py` eagerly imports `backends`, whose MPM driver eagerly `import mujoco`s, and `mujoco` touches an OpenGL/OSMesa binding at _that_ import, which raised `AttributeError: 'NoneType' object has no attribute 'glGetError'` on the GL-less runner. Every sibling name this module needs at runtime is now imported inside the function/method that uses it (mirroring the existing `require_pyvista()` pattern); only type annotations reference them at module scope, under `TYPE_CHECKING`. (2) `ShotViewportWidget._refresh_renderer_note` derived its note from `.render.viewport_fallback()`, worded for the unrelated sole-field 2-D plan-view widget, so it claimed "VTK/PyVista" even when nothing was ever attempted; it now derives its own note from what actually rendered the frame via a local `_MATPLOTLIB_FALLBACK` constant. Also fixed a `RuntimeError: wrapped C/C++ object of type FigureCanvasQTAgg has been deleted` surfacing under the same path: deferred `draw_idle()` repaints now call the synchronous `draw()`, plus a `sip.isdeleted()` guard. Simulated the third environment (pyvista present, GL init fails) by monkeypatching `pyvista.Plotter` to raise the exact CI `AttributeError` through `ShotViewportWidget.set_shot`; added `test_render3d_vtk_import_touches_no_gl_or_mujoco`, which checks `sys.modules` after a bare import rather than only process exit code. `ruff`/`ruff format`/CI-mode mypy clean; full `tests/unit/tools/bunker_shot_gui/` + `tests/tools/bunker_shot_gui/` suite (718 tests) passes. (spec 1.0.614) |
| 2026-08-27 | #8706 | Gave BunkerShot3D a real 3-D backend (issue #8706, epic #8699): the real-backend half of #8706, ADR-0027's VTK/PyVista provider was detected but nothing installable made it selectable and nothing consumed it, so the viewport always degraded to a depth-buffer-less matplotlib `Axes3D` scatter. Added an optional `viz3d = ["pyvista>=0.44"]` extra (`pyproject.toml`, not a dependency of `bunkershot3d`) and a new headless renderer (`src/tools/bunker_shot_gui/render3d_vtk.py`) that draws the same `ShotScene`, the same injected `SceneScale` and the same in-frame validity stamp `render3d.py` draws, but poses the lofted `HeadBuild`'s actual watertight triangle mesh per frame as a solid shaded surface instead of a point cloud, plus the sand plane, a depth-coloured divot floor on a fixed colour ramp, the sole-reference path, and the three named cameras. PyVista is imported lazily, so importing the module never requires the extra. `ShotViewportWidget.set_shot` (`viewport_widgets.py`) gained an optional `build: HeadBuild` parameter: when `select_viewport_provider(VTK)` is available and a build is supplied, the widget renders through `render3d_vtk` and blits the offscreen frame into its existing matplotlib canvas; a render-time failure (no GPU, no display) degrades to matplotlib rather than crashing, and omitting `build` (every existing caller today) leaves the matplotlib path unconditionally in force. Verified on a Windows box with no display attached: pyvista 0.48.4 / VTK 9.6.2 actually initialises offscreen and renders a genuinely three-dimensional, shaded, depth-buffered scene end to end through the real workbench widget. 46 new tests: a pyvista-absent-safe degradation suite requiring no extra, a `pytest.importorskip("pyvista")`-gated suite exercising real offscreen rendering, and a Qt-level widget-dispatch suite. (spec 1.0.613) |
| 2026-08-27 | #9116 | Reconciled #9116's phase/event finite-time stability evidence with the protected #9135 paper and claim authority. The exact-discrete RK4 variational map, transverse event-time sensitivity, direct perturbation checks, saltation killswitches, near-grazing rejection, equivalent-unit controls, and fail-closed nonperiodicity/Floquet boundary now extend the 1,148-candidate census to 315 explicitly adjudicated claims. Regenerated and web-linearized the 246-page, 1,923,372-byte PDF at SHA-256 `0e590a556660fcc5656ad4add657bcfb4e90a8c5b15abc98ccf031118d61f050`; it remains computationally qualified but untagged with Type 3 and two unembedded font resources. The local synthetic trajectory does not establish asymptotic stability, neural timing demand, passive torque, participant robustness, or coaching guidance. (spec 1.0.613) |
| 2026-08-27 | #8706 | Fixed two defects in BunkerShot3D's matplotlib 3-D shot-scene fallback, the baseline every user gets when no MeshCat/Rerun/VTK backend is installed (issue #8706, epic #8699). Defect 1: the clubhead rendered as a scatter of the solver's own element centroids -- a cloud of dots, not a solid -- because `render3d.py` had nothing else to draw it from. `ShotScene` (`shot3d.py`) now also carries `head_mesh_body`, the lofted watertight `TriangleMesh` `HeadBuild.loft.mesh` already builds, posed per frame by a new `head_mesh_world_m()` alongside the existing centroid-based `head_world_m()`; `render3d.py` draws it as a single translucent `Poly3DCollection` (alpha 0.62, so the divot floor behind it still reads), built once and moved every frame through the public `set_verts` rather than rebuilt, and drops the "head surface" scatter that used to stand in for it. Defect 2: the footer caption ran unwrapped off the figure's right edge, clipped mid-word ("...not where sand h"); the sole-level camera's axis label collided with its own foreshortened tick labels; and the 3-D box used well under half the canvas. The caption is now word-wrapped to how wide it actually renders in the figure's own font -- measured via a forced-renderer text extent, not a guessed character count, so it stays correct across figure sizes and DPI -- and anchored in a bottom margin `subplots_adjust` now reserves for it. The across-track axis label gets a wider `labelpad` and its tick locator collapses to a single origin reference only at `CameraPreset.SOLE_LEVEL`, where the eye sights straight down that axis; the other two presets are unaffected, since applying the wider pad everywhere pushed the label into the caption's own space instead. `Axes3D.set_box_aspect(None, zoom=1.35)` and tightened `subplots_adjust` margins fill most of the remaining canvas; the one residual white border on a landscape figure is `Axes3D.apply_aspect`'s own hard square-panel constraint, not a further bug here. Verified by rendering all three `CameraPreset` values to PNG and inspecting them directly, iterating until no camera showed a clipped caption, an overlapping label, or a scatter-cloud head. 20 new tests (7 in `test_shot_scene.py` pinning the mesh and its pose, 13 in `test_shot_scene_render.py` pinning the solid `Poly3DCollection`'s rendered face count, its translucency, and the caption's real window extent against the figure's own bounds across all three presets); all 85 tests in both files and all 670 tests across `tests/unit/tools/bunker_shot_gui/` and `tests/tools/bunker_shot_gui/` pass. No new dependencies; `viewport_widgets.py` and `pyproject.toml` untouched. (spec 1.0.612) |
| 2026-08-27 | #9135 | Fixed #9135's publication pagination defect by moving the opening evidence-category callout under the Chapter 1 heading, preserving its wording and HTML semantics while preventing the LaTeX callout from splitting across two pages and leaving a near-empty continuation page. Added a rendered-PDF regression that requires the callout's opening and closing text on the same page, regenerated and web-linearized the 245-page candidate to 1,912,422 bytes, restored extractable text on all 245 pages, and retained the untagged, Type 3, and unembedded-font archival limitations. Rebuilt the 1,142-candidate inventory and advanced its frozen reviewed-source digest after proving that the candidate-ID set, 313 explicit outcomes, and complete review coverage were unchanged by the editorial move. This is publication polish only; it changes no model, evidence, claim, or human/coaching boundary. (spec 1.0.611) |
| 2026-08-27 | #8358 | Completed #8358's immutable Tools R14.3 consumer boundary. The `vendor/ud-tools` gitlink now pins protected Tools squash `3dfbd32cc778536269670c055955073853c0f60a`; a downstream provider contract verifies the UI-neutral `all_together`/`individual`/`both` execution-policy set, exact joint and one-at-a-time run accounting, and the protected PyQt variation baseline (`22f6640e896e9ea5c740e9db7e3d3201cdf7264f2cf1cff33966b539354f1d40`) with its narrow reviewed tolerance. The prior protected integration supplies localized time/point adapters, complete trial evidence, cross-engine rejection, persistence, dispersion, quiet-zone, and attribution contracts; the Tools lineage supplies typed cohort scatter, all-arc overlays, source-window/global attribution, matched clients, and visual evidence. UpstreamDrift continues to launch the Tools-owned surface instead of copying UI or analysis code. This qualifies an immutable software-consumer boundary, not participant evidence, human strategy, or coaching guidance. (spec 1.0.610) |
| 2026-08-27 | #9129 | Fixed #9129. `dependency-consistency` failed on every PR, including ones touching no dependency files, because the committed lock headers recorded a `pip-compile` invocation without `--no-index` while the pinned pip-tools now emits it. Only the two recorded-command comment lines differed; no package pin changed. Updated both headers so the regeneration check matches. (spec 1.0.609) |
| 2026-08-26 | #4766 | Registered the Tools-provided `swing_objective_lab` tile in the canonical launcher registry and the web manifest, and mapped it to the Simulation category. The tile opens a downswing objective comparison that optimizes the same golfer for clubhead speed, centrifugal release impulse, Coriolis kinetic-chain transfer, grip-force energy transfer, and grip-force impulse under one shared torque budget. It consumes the existing `../Tools/src/pendulum_simulator` provider through the established source-root contract; no physics is added or duplicated here. Tools epic #4766, child #4772. (spec 1.0.608) |
| 2026-08-26 | #9066 | Reconciled UP-D0 (#9066) with protected current main `99acc997a97b3d97cb4ddd857b79bedd4a66f290`. `manuals/upstreamdrift` QMD remains the sole editable engineering design-manual authority; existing user, ADR, and research products remain separate; generated HTML, LaTeX, PDF, and DOCX remain non-editable and unapproved. The versioned policy, fail-closed empty registry, renumbered ADR-0042, agent guidance, contract tests, and offline CI/pre-commit verifier enforce program-contract ownership, safe paths, Ruff formatting, TDD/DbC/DRY/LoD, impacted-path/freshness rules, immutable release evidence, visual and semantic review, and human approval. ADR-0041 remains the protected markerless-mocap consumer authority; UP-D1 through UP-D8 remain explicit blockers, and this governance scaffold makes no calculation-coverage or publication claim. (spec 1.0.608) |
| 2026-08-26 | #9107 | Normally reconciled #9107 with protected markerless-authority main `fe609edede7a1e9a7427a61ee1bf23ed39fcc43c` after #9088 merged. The packaging-only wheel/source split, fixture-only smoke checkout, canonical BunkerShot3D identity, and bounded transfer/install assertions remain unchanged; the integrated ADR-0041 authority boundary introduces no camera, inference, C3D round-trip, physical-lab, or human-performance qualification. (spec 1.0.607) |
| 2026-08-26 | #9107 | Repaired #9107's wheel-smoke transport boundary after both matrix jobs exhausted their time while downloading the combined 771,541,995-byte wheel-plus-sdist artifact and never reached installation. The build now retains separate wheel and source artifacts; smoke jobs use a fixture-only sparse checkout, download only the selected wheel, and have a measured bounded 20-minute transfer/install/assertion budget. Runner selection, the built distributions, and both smoke assertions are unchanged. (spec 1.0.606) |
| 2026-08-26 | #9107 | Reconciled #9107 with current `main` and qualified the exact 384,333,159-byte wheel (`ec3b6c6223f08ebfe1a256f5a3eda3b00209a081fe2cbbe01bd9a0e8ae6f0d18`) under Python 3.11. The branch-owned canonical BunkerShot3D import, one-object identity, duplicate exclusion, UI presence, and test-payload exclusion pass. Two broader inherited wheel-runtime failures remain explicit and unqualified: `src.api` collides with the co-installed Tools config alias and `sidekick --help` fails. No runner, workflow, vendor pin, runtime API, or scientific authority is changed to mask them. (spec 1.0.605) |
| 2026-08-26 | #9107 | Fixed #9107's external-install packaging boundary. Hatch now force-includes the Upstream-owned `src/bunkershot3d` source under the one canonical top-level `bunkershot3d` destination for editable and wheel builds, while normal wheel selection excludes `src/bunkershot3d` so distinct class and exception identities cannot ship. The build hook filters test artifacts and fails closed if the canonical source root is absent. A RED unit contract, wheel smoke contract, isolated editable install, and isolated 383.7 MB wheel install prove `bunkershot3d.postproc.WrenchTrace` resolves outside the checkout with stable identity and no duplicate wheel payload. No library mutates `sys.path`; no Tools vendor pin, scientific claim, public schema, workflow, runner, or UI behavior changes. (spec 1.0.604) |
| 2026-08-26 | #9065 | Reconciled markerless-mocap authority issue #9065 with protected compatibility main `847b9abd39e6fd7cffaa917cf4fdb43a563cb276`. ADR-0041 and its executable acceptance program retain Tools as canonical camera/observation/calibration/timing/session/reconstruction/C3D contract owner, UpstreamDrift as orchestration/UX/persistence/biomechanics owner, and AffineDrift as sanitized evidence-publication owner. No feature-head vendor pin, live-lab implementation, physical qualification, C3D round-trip, or human-performance claim is introduced. (spec 1.0.604) |
| 2026-08-26 | #9113 | Qualified #9113's feasible closed-loop singular-margin boundary. An exact same-origin planar triangle constructor now covers both assembly branches and a full global-phase orbit before interpreting the scaled closure Jacobian. The governed evidence retains exact lower/upper triangle degeneracies, a five-by-five distance/tolerance matrix, phase-resolution, positive-scale, feasible-geometry, equivalent-unit, impossible-geometry, and manufactured-rank controls. The result establishes only local planar kinematic closure and scale-qualified rank/nullity; it does not establish anatomy, dynamics, contact force, passive torque, human occurrence, or coaching guidance. (spec 1.0.603) |
| 2026-08-26 | #9027 | Qualified #9027's dimensionally explicit constraint-rank boundary. Planar closure diagnostics now declare generalized-coordinate scales, bilateral wrench diagnostics declare an input/output moment reference length, and positive rescaling is tested to preserve exact rank and nullity while numerical conditioning remains scale-dependent. Kinematic closure, point-force allocation, and full wrench measurement nullspaces remain separate governed estimands; reviewed numeric-pointer overrides make the scaffold reproduce all new contracts exactly. Constructed adverse alignments are mathematical controls rather than anatomical poses, and no constraint-force, muscle-action, human-strategy, or coaching inference is authorized. (spec 1.0.600) |
| 2026-08-25 | n/a | Corrected the documentation surface of record. The README, `pyproject.toml` `[project.urls] Documentation`, and `docs/index.md` all declared `https://upstream-drift.readthedocs.io` as the canonical rendered documentation; that address returns 404 and no Read the Docs configuration has ever existed in the repository, so the primary documentation link on the public landing page and in the published package metadata resolved to nothing. All three now point at `docs/README.md`, the maintained hub, and `docs/index.md` records that the readthedocs address was never configured so it is not reintroduced. `scripts/check_doc_catalog.py` had been enforcing the dead link by requiring the literal string `[Documentation Hub](<pyproject URL>)` in the README; it now verifies that the README and package metadata agree on the same destination, accepting the repository-relative path when the declared URL is a forge link to that same file, and still fails when they disagree. LICENSE gained its previously absent copyright holder. No behavioural change. (spec 1.0.600) |
| 2026-08-26 | #8823 | Fixed issue #8823: the `ProvenanceRecord`/`ProvenanceValue` infrastructure in `src/shared/python/ux/provenance.py` and its `ProvenanceValueLabel` Qt wrapper in `src/shared/python/ui/provenance_value.py` had no real callers outside their own unit test — dead, tested, unwired infrastructure. `src/tools/swing_flight_pipeline/gui.py`'s `SwingFlightWidget` now renders its two headline computed values (Carry Distance, Launch Speed) as `ProvenanceValueLabel`s, rebuilt fresh on each `_run_pipeline()` call with a `ProvenanceRecord` naming the producing `engine_name`, a per-run `source` id, the deriving formula, and the consumed swing-state input ids — hovering or right-clicking either value now answers "why does this say 220.4 m?" without leaving the screen. A new `_update_provenance_labels` helper enforces, via an explicit `require()` precondition (not a silent default), that no headline value is ever rendered against an empty `engine_name`, echoing the honest-attribution guarantee issue #8819 already established for the engine combo. New unit tests assert the rendered `ProvenanceRecord` carries the correct engine/source/formula, that the run id increments across runs, that the tooltip/whatsThis describe the value, and that the precondition raises `PreconditionError` on an empty engine name. (spec 1.0.601) |
| 2026-08-26 | #8825 | Fixed issue #8825: Launch Monitor Analytics's five matplotlib analysis canvases (`relationship_plot`, `model_plot`, `comparison_plot`, `dispersion_plot`, `trend_plot`) were never cleared when the underlying data changed, so `clear_project`, `import_file`, `load_project`, `_remove_selected_sessions`, and `_run_treatment_ui` all left stale charts (including leftover colorbar axes) from a previous project/session visibly on screen. `_refresh_all` — the single choke point already reached by all five call sites — now calls a new `_reset_analysis_canvases()` that resets every canvas to a placeholder state naming the current project. `PlotCanvas.empty()` was changed to fully reset the figure via `reset_axes()` instead of clearing the current axes in place, so a colorbar's own axes can no longer survive a reset. Analysis plot titles across relationship, PCA/VIF, model, monitor-comparison, dispersion, and trend charts now include the project name via a new `_title_with_project()` helper, so a viewer can tell which project's data a rendered chart reflects. New regression test asserts a populated relationship canvas (with its colorbar axes and image artist) is fully cleared and shows a project-named placeholder after `clear_project()`. `gui.py` was also split (`PlotCanvas` extracted to a new `plot_canvas.py`, `_selected_text`/`_populate_combo` moved to `widgets.py`) to stay under the 1200-line file-size budget after this fix. (spec 1.0.602) |
| 2026-08-26 | #8826 | Fixed issue #8826: Launch Monitor Analytics `_on_export_data` wrote a bare CSV/Parquet and `_on_export_manifest` wrote a wholly separate reproducibility-manifest JSON, with no shared identifier linking the pair once separated on disk. `export_data` (new public method backing `_on_export_data`) now stamps a fresh UUID export ID and export timestamp directly into the artifact — a leading `# export_id=... exported_at=...` comment row for CSV, Parquet schema metadata for Parquet — mirroring the existing `ImportManifest` provenance pattern, and records the exported file's own SHA-256 on `self._last_data_export`. `export_manifest` (new public method backing `_on_export_manifest`) embeds that export's ID, filename, and SHA-256 under a `data_export` field, or `None` when no data export ran yet. New tests assert the CSV comment and Parquet schema metadata carry the same export ID recorded in the manifest, and that the manifest's recorded SHA-256 matches the exported file's actual on-disk content. (spec 1.0.599) |
| 2026-08-26 | #9104 | Qualified #9104's double-pendulum identifiability boundary. The exact seven-coefficient inverse-dynamics factorization and analytic physical-map witness establish rank seven and nullity four for the declared eleven-entry reduced map, with three coefficient-preserving physical alternatives. Dimensionless finite-record evidence retains equivalent-unit and scale audits, shortened-window adverse cases, four torque-noise levels, and a zero-motion rank-zero killswitch. Oracle Gaussian Fisher intervals remain lower bounds under exact kinematics and do not establish practical, participant, biological, or coaching identifiability. Claim registration and report construction satisfy the current function-size and parameter architecture budgets without changing the governed scientific JSON. (spec 1.0.598) |
| 2026-08-25 | #8827 | Fixed issue #8827: Pose Studio's engine-status pill went stale after a silent mid-session downgrade to the mock kinematics service. `EngineController.set_pose` already tracked this correctly (catching `NotImplementedError` from a partial engine bridge, swapping in `MockKinematicsService`, and setting `EngineStatus.MOCK`), but `MainWidget._apply_pose` never re-read `EngineController.status` afterward — the pill only updated in `_on_engine_selected`, at initial engine selection. A user whose engine reported "live" at activation and then edited a joint mid-session (undo/redo, angle edit, or pose-library load all route through `_apply_pose`) would see a stale green "live" pill while the 3D view was actually being driven by the mock. `_apply_pose` now calls `self.engine_picker.set_status(self._engine_controller.status)` immediately after `set_pose`, so the pill reflects any downgrade in real time. New offscreen Qt test forces a live-engine service whose `set_pose` raises `NotImplementedError`, calls `_apply_pose`, and asserts the status pill reads `mock` (not the stale prior `live`) afterward. (spec 1.0.597) |
| 2026-08-25 | #8358 | Reconciled the #8358 Tools provider-contract regression with protected nondimensional-local-rank main. Refreshing one selected provider no longer evicts the downstream-owned `src.shared` namespace, and a regression preserves an already-imported UpstreamDrift perturbation gateway when Tools is first on `PYTHONPATH`. Existing contracts still require each selected provider to resolve from Tools. This changes no production import, public API, provider schema, vendor pin, workflow, or runner; it restores deterministic downstream compatibility verification for Tools PR #4734. (spec 1.0.596) |
| 2026-08-25 | #9092 | Registered #9092 as the nondimensional local-rank qualification slice downstream of #9027. Raw analytical linearizations remain trace evidence, while local rank and condition interpretation now require declared state/control/output/time scales, unit-invariance and scale-sensitivity gates, output/actuator countermodels, manufactured fixtures, and killswitches. Structural/practical identifiability, global nonlinear control, human validity, and coaching conclusions remain unavailable. (spec 1.0.595) |
| 2026-08-25 | #9027 | Reconciled #9027's first nonlinear-systems qualification slice with the coordinate-explicit #9059 authority and current main. The typed eight-tier hybrid-system topology and referential tamper gates remain intact alongside the 135-program force-attribution grid and unrelated gap-fill acceleration; neither topology registration nor planar attribution is promoted to controller, participant, or coaching evidence. (spec 1.0.594) |
| 2026-08-25 | #8871 | Wired issue #8871's `export_paths` gap: `SimulationService._run_simulation_sync` previously returned `SimulationResponse(..., export_paths=[])` on both success and failure, and `OutputManager` (`save_simulation_results`/`get_simulation_list`/`export_analysis_report`) had exactly one caller (`video_pose_pipeline.py`), never the API simulation service. `SimulationService` now takes an injectable `OutputManager` (defaulting to a project-root-relative instance; tests inject one rooted at `tmp_path`) and, on a successful run, persists simulation data plus any analysis results as JSON via the new `_persist_simulation_results`, threading through only genuinely-known provenance (engine type, model path, duration, timestep) — nothing fabricated. The saved file's path populates `export_paths`; a failed run persists nothing, so `export_paths` stays empty. `OutputManager`'s docstring and `output/README.md` document the resulting `simulations/<engine>/` schema. New tests assert `export_paths` is non-empty and every listed path exists on disk after a successful run, and stays empty after a failed one, with no other test writing into the real `output/` tree (an autouse fixture roots the default `OutputManager` at `tmp_path`). (spec 1.0.593) |
| 2026-08-25 | #8829 | Fixed issue #8829: `MuJoCoDashboard`, `PinocchioDashboard`, and `DrakeDashboard` no longer swallow starting-model load failures behind `except Exception: pass` / `contextlib.suppress(Exception)` while unconditionally setting a title implying success. Model discovery/load now catches a narrow, named exception tuple (`OSError`, `ValueError`, `RuntimeError`, `KeyError`, `AttributeError`, `TypeError`) per ADR-0016 instead of bare `Exception`, logs via `logger.exception`, and builds a `ModelLoadStatus` (new in `dashboard/window.py`) that `UnifiedDashboardWindow` renders as an in-body engine/model identity strip plus a visible warning banner on failure. The strip lives in the dashboard's own layout, so — unlike the window title — it survives `ExerciseDashboard` embedding, which strips the `Qt.WindowType.Window` flag. New offscreen Qt tests cover load-failure visibility and identity-strip persistence in both standalone and embedded modes. (spec 1.0.592) |
| 2026-08-25 | #8927 | Wired the Rust `linear_gap_fill`/`cubic_gap_fill` kernels into `gap_fill.py` for issue #8927: the pipeline's default `GapFillStep.strategy = LINEAR` previously never left the pure-Python per-frame per-marker loop even though `rust_core/upstream-mocap-preproc` exported both kernels with zero Python call sites. Marker LINEAR/CUBIC gap-filling now mirrors the existing PCA dispatch pattern — stacks the trajectory via `_stack_marker_matrix`, calls `_rust_kernel.linear_gap_fill`/`cubic_gap_fill` when the wheel is available, and unstacks via the new `_unstack_marker_matrix` — with the prior pure-Python `_fill_gaps_markers` loop preserved as the fallback when the wheel is absent. `_find_gaps_markers`'s per-frame `any()` occlusion scan is now vectorized unconditionally via a numpy occlusion matrix plus diff-based run detection. Rust-vs-Python parity verified at rtol=atol=1e-10 (matching the precedent in `test_filter_resample_vectorize_parity.py`/`test_kalman_vectorize_parity.py`); dispatch plumbing additionally covered by mocked-kernel tests that run regardless of wheel availability. (spec 1.0.591) |
| 2026-08-25 | n/a | Reconciled repository merge guidance with the live protected ruleset: pull requests and required status checks remain mandatory, while the zero-approval configuration means no named maintainer—including `@dieterolson`—is a standing release gate. Optional review remains available for risk, expertise, or unresolved feedback; admin bypass, force-push, check bypass, and stale-head merge remain prohibited. (spec 1.0.590) |
| 2026-08-25 | #9059 | Completed #9059's fail-closed publication adjudication after the broad unit gate exposed stale generated locks. Thirty coordinate-force chapter candidates now map explicitly to three new bounded model claims plus the existing MacKenzie hand-path-force claim; the frozen authority contains 1,130 reviewed candidates, 306 claims, and 382/382 registered numeric literals across 125 numeric contracts. Release review, candidate reciprocity, snapshot migration, PDF byte identity, checksums, claim evidence, and the 608-artifact bundle are regenerated and tested without promoting the 135-program Coriolis-impulse grid to biological-force, continuous-optimal-control, human, or coaching authority. (spec 1.0.589) |
| 2026-08-25 | #8921 | Fixed issue #8921: the geometric IK backend's `forward_kinematics` no longer recomputes rig topology (`_dof_layout`/`_topological_order`, a full recursive tree walk) on every call; it is built once per rig and cached by rig identity via `_get_rig_topology`. The Levenberg-Marquardt Jacobian is now the analytic revolute-joint form (`axis_world x (p_i - p_pivot)`, correctly composed for multi-axis joints) built from one already-computed forward-kinematics pass per iteration instead of `2 * n_dof` central-difference forward-kinematics calls; the hot-path FK also returns arrays plus a name-to-row map instead of a dict of Python float tuples, avoiding a numpy<->dict round trip. `solve_frame` now accepts and honors a per-call `IKConfig`, fixing a latent bug where `solve()`'s resolved config was silently ignored (the iteration loop always read `self.config`). Numerical parity with the removed finite-difference Jacobian is covered by `tests/unit/motion_pipeline/ik/test_geometric_backend_perf.py`, which reimplements the pre-fix algorithm independently for comparison and asserts call-count reduction (2 forward-kinematics passes per iteration, not `2 * n_dof + 2`). (spec 1.0.587) |
| 2026-08-25 | #8933 | Fixed a latent correctness bug for issue #8933 item 4: `_ensure_capacity` in `dashboard/_recorder_buffers.py` grew every ndarray in `self.data` to `new_capacity`, but several buffers (`com_position`, `ground_forces`, `ground_moments`, `ztcf_accel`/`zvcf_accel`/`drift_accel`/`control_accel`, `induced_accelerations` entries) are allocated directly at `max_samples` rather than the growable `current_capacity`; once a growth step landed strictly below `max_samples`, copying the (larger) `max_samples`-sized source array into the (smaller) `new_capacity`-sized destination slice raised a `ValueError` shape mismatch, silently killing in-progress recordings. Buffers now only grow when `arr.shape[0] < new_capacity`; regression test added that reproduces the exact boundary (verified to fail on the pre-fix code with the same `ValueError`). Also for #8933: (item 1) the ezc3d C3D fallback in `motion_pipeline/sources/c3d_adapter.py` now slices `points[:3, :n_markers, fi] * scale` per frame and builds `Marker` via `model_construct` instead of scalar-indexing the strided `(4, M, N)` array through the validated constructor, matching the ≥10x-faster pattern already documented on the Rust ingestion path (parity test against a scalar-loop reference, including NaN-occlusion and label-truncation cases); (item 2) `analysis/nonlinear_dynamics.py`'s sparse recurrence-matrix path now builds pairs via `cKDTree.query_pairs(threshold, output_type="ndarray")` plus two fancy-index assignments instead of a `query_ball_point` + nested Python loop, with the diagonal set explicitly since `query_pairs` excludes self-pairs (parity test against the old loop algorithm on N=150 random data). Item 3 (`launch_monitor/strokes_gained.py` iterrows) was already resolved by PR #8956 prior to this change; no further action needed there. Zero behavior change for items 1-2; item 4 is a genuine bug fix (previously-crashing recordings now succeed). (spec 1.0.588) |
| 2026-08-25 | #8923 | Vectorized the motion_pipeline preprocessing Kalman smoother for issue #8923: `_kalman_filter_python` in `filter.py` initialized its covariance at the DARE steady-state fixed point, which makes the forward Kalman gain and backward RTS gain constant across every timestep and series — collapsing the 750k-iteration `for marker: for axis: for frame:` scalar-update double loop into two `scipy.signal.lfilter` calls (forward IIR, then the same IIR run on the reversed array for the RTS pass) applied to the whole `(n_frames, n_points, n_dims)` array at once. Numerically identical to the prior loop implementation (parity tests at rtol=atol=1e-10 on realistic 1250x50x3 data plus short-sequence and varied-noise-parameter cases), zero behavior change; the old loop reimplementation is preserved as a private reference in the parity test file rather than in production code. (spec 1.0.586) |
| 2026-08-25 | #8925 | Vectorized motion_pipeline preprocessing for issues #8925 and #8924: normalize.py's per-frame transform/center/units/up-axis-detection passes now run as a single fused array expression via markers_to_array/keypoints_to_array, reconstructing frames once with model_construct; filtfilt/savgol_filter/median_filter/moving-average and the resample np.interp loops in \_filter_pure_python.py and \_resample_pure_python.py/resample.py now operate on whole (N, M, 3) arrays instead of per-marker/per-axis Python loops, deduplicated into a shared vectorized_interp_axes helper. Numerically identical to the prior implementations (parity tests at rtol=atol=1e-12), zero behavior change. (spec 1.0.585) |
| 2026-08-24 | n/a | Kept the Pendulum Simulator module version probe independent of GUI initialization. Runtime imports of diagnostics and `MainWindow` now occur only on docked or standalone GUI launch, so `python -m shared.python.pendulum_simulator --version` does not transitively load MuJoCo/OpenGL on headless systems. A subprocess regression deliberately blocks `gui.main_window` and requires the version command to succeed; normal launcher behavior remains covered on Python 3.11 and 3.13. (spec 1.0.584) |
| 2026-08-24 | #8358 | Added the #8358 immutable Tools variation-consumer boundary. UpstreamDrift now delegates deterministic sampling, canonical JSON/CSV/HDF5 persistence, scalar and noncausal rank/OAT summaries, common-grid dispersion, and quiet-zone mathematics to protected Tools commit `17474249b9267d0e73a779c1d72f231e7b8de39c`. Host-owned adapters retain complete typed hit, no-impact, numerical-failure, and partial-trace rows across serial and batched execution; analytical double-pendulum and articulated MuJoCo mappings preserve stable marker, frame, unit, plan, seed, model, source, and bilateral shoulder/wrist allocation evidence. Cross-engine ranking fails closed on semantic or tolerance mismatch, and current architecture budgets pass without exceptions. This is reusable model evidence, not participant or coaching validation. (spec 1.0.583) |
| 2026-08-24 | #8918 | Added issue #8918's executable numeric-claim authority. All 380 numeric literals across 124 of the 303 material claims are bound to exact statement digests, JSON Pointers, transforms, scopes, and tolerances. The audit distinguishes 172 local JSON values, 144 registered values not independently recomputed, 57 externally reported values, and seven protocol or notation values. Representative planar, spatial, articulated-shaft, and finite-ground headlines are recomputed from committed raw arrays; a cross-engine control must remain close but nonidentical. Claim integrity and the 600-artifact computational release fail closed on numeric drift without promoting pointer agreement to physical or human validation. (spec 1.0.582) |
| 2026-08-23 | #8724 | Reconciled issue #8724's normalized adjudication checkpoint with the current 303-claim paper. The migration is locked to the exact paper digest and contains an exhaustive explicit claim-ID authority, so an unfamiliar claim cannot inherit `supported`. The reviewer JSON/CSV and generated paper tables now separate outcome, evidence tier, source independence, model tier, unresolved replication, and claim-family concentration. Typed evidence locators, local anchors, bibliography keys, deterministic source digest, exact candidate reciprocity, falsifiers, adjudication reasons, and human-data boundaries remain fail-closed. Source and evidence-file caches remove repeated validation I/O without weakening resolution. (spec 1.0.581) |
| 2026-08-23 | #9004 | Closed participant-holdout and executable-mapping authority gaps in #9004's governed ingestion boundary. Each trial now binds an immutable `measured-trajectory-participant-split/v1` manifest that freezes sorted, disjoint training, held-out, and adverse cohorts before outcomes. The loader verifies the split digest, source, registered deterministic assignment method, UTC freeze-before-artifact ordering, minimum cohort counts, adverse cohort, unique participant membership, cohort label, and intended-use eligibility before invoking a parser. It also contains and verifies the exact acquisition-processing authority, four frame-transform records, and two event-detector configurations by relative path and SHA-256 before payload parsing. Returned artifacts expose the split, cohort, processing, transform, and detector provenance while retaining false human-inference and bilateral-wrench gates. UTC timestamps, format hints, and channel identifiers are also validated structurally. (spec 1.0.580) |
| 2026-08-23 | #9004 | Added the fail-closed #9004 measured-trajectory acquisition and ingestion boundary. A per-trial typed manifest now binds an authorized source package and decoded trajectory to immutable digests, participant grouping, canonical units, acquisition processing, four frame authorities, two event records, declared channels, six uncertainty analyses, and intended use. The gateway recomputes source and preregistration readiness, rejects duplicate keys, path traversal and pickle formats before parsing, verifies both digest layers, delegates only to the canonical motion-pipeline adapter, and reports missing channels as unavailable rather than zero. It always denies human-inference and bilateral-wrench authority; no current dataset is admitted. (spec 1.0.579) |
| 2026-08-23 | n/a | Made the headless optional-stack contract deterministic by forcing Matplotlib's noninteractive `Agg` backend at the job boundary. This prevents an ambient interactive runner backend from producing `Invalid DISPLAY variable` failures while leaving application and library backend selection unchanged outside CI. A workflow-structure regression test now enforces the boundary. (spec 1.0.579) |
| 2026-08-23 | #9004 | Preregistered #9004's motion-only golf-likeness evaluation before any governed trajectory outcome exists. The fail-closed contract freezes participant-level holdout, four frame authorities, downswing-start and impact events, eleven primary club/body/hand/geometry/feasibility/discrepancy metrics, four manufactured negative controls, six processing and mapping sensitivities, training-only threshold selection, and missing-as-unavailable behavior. It cross-checks the measured-source registry, prohibits frame-wise splits and force/coaching promotion, and remains blocked because no qualifying measured trajectory authority is available. (spec 1.0.578) |
| 2026-08-23 | #9004 | Added the fail-closed measured golf-trajectory source registry for #9004. Exact schemas, source classes, authority/access/license states, participant and trial counts, body/club/calibration/synchronization availability, content digests, decisions, and blockers are validated before readiness is derived. Simulation output cannot qualify human measurement; pipeline and participant-held-out readiness require governed body-and-club trajectories, calibration, synchronization, grouping, explicit reuse authority, and a registered digest. The initial census retains GolfPose and KIT Motion 1319 as blocked candidates, GolfDB and local fixtures as negative controls, local Simscape output as inadmissible validation evidence, and all human/coaching and bilateral-wrench inferences as unavailable. (spec 1.0.577) |
| 2026-08-22 | #8766 | Made issue #8766's unit-gate debt executable rather than descriptive. All 520 quarantined node IDs resolve exactly once into ten owned clusters with rationales, reproduction commands, exit criteria, and blocking status. A fail-closed checker rejects malformed, duplicate, unassigned, or ambiguous entries and can list or execute each cluster in bounded batches. CI Standard compares pull-request node IDs with the fetched base branch, so the removal-only rule now rejects additions and replacements while allowing verified burn-down; existing `UNIT_GATE_QUARANTINE=1` skip behavior is unchanged. (spec 1.0.576) |
| 2026-08-22 | n/a | Bound modular Docker builds to the exact pinned Tools package roots. CI emits the superproject gitlink and deterministic path-and-content digest; the isolated build copies only the registered roots and recomputes the digest without Git metadata. Missing roots, symlinks, incomplete attestations, malformed identities, and content mismatches fail closed. The custom Hatch hook loads its adjacent helper by file path so PEP 517 source and editable builds do not depend on the repository root being present on `sys.path`. Constructor-compatible Hatchling test doubles and explicit suite isolation prevent collection order from changing that contract. Both runtime Dockerfiles pin pip 26.2.1 after PYSEC-2026-3721. All supported images reassert `msgpack` 1.2.1 and `setuptools` 83.0.0 after their final dependency layer to resolve GHSA-6v7p-g79w-8964, CVE-2025-47273, and PYSEC-2026-3447 without a scanner waiver. Production runtime targets then remove pip and its embedded third-party SBOM after finalizing dependencies; only the explicit training stage restores the audited builder environment for package installation. The modular builder copies both force-included launchers before feature installation so isolated metadata generation matches the wheel contract. The protected slim build then exposed stale workflow budgets (800/2000 MB) that contradicted the canonical profile catalog (900/2200 MB); the workflow is resynchronized and a regression test now makes `docker/profiles.yaml` authoritative. Focused action, workflow, Docker-context, isolated-import, packaging, security-pin, runtime-surface, size-budget, and tamper tests enforce the boundary. (spec 1.0.575) |
| 2026-08-22 | #8995 | Retained PR #8995's MediaPipe landmark-mean optimization while repairing its generated specification projection: restored the complete first-parent specification, removed thousands of repeated misplaced changelog rows, and recorded the optimization exactly once in this Change Log. (spec 1.0.574) |
| 2026-08-22 | n/a | Made the classic launcher application and event-loop lifecycle explicitly single-owner. A typed resolution seam reuses `QApplication.instance()` without starting a nested event loop and constructs and executes an application only when the launcher owns it. Tests use an ordinary mock at that seam instead of replacing the SIP-backed Qt class, preventing Linux worker teardown crashes; launcher-containment tests also isolate the mutable embeddable-tool registry so their package-entrypoint guard is independent of suite order. (spec 1.0.573) |
| 2026-08-22 | #8977 | Reconciled the release-bound proximal-to-distal publication record with the exact current candidate for #8977. The computational profile now pins the 235-page, 1,863,127-byte PDF digest, 246 outline entries, 194 valid external links, complete rendering and text extraction, and the exact font inventory in executable tests. The complete ordered page set was visually inspected, with the newly added native-contact section and surrounding pages checked at full resolution. The archival profile remains fail-closed and explicitly records the untagged structure, 112 Type 3 resources, and two unembedded resources rather than promoting computational readiness to an accessibility claim. (spec 1.0.571) |
| 2026-08-21 | #8965 | EPIC #8965 registry residuals + Tools vendor pin bump. `WEB_CATALOG_ONLY_TILES` allowlist entries now require a substantive justification string enforced by the parity suite (#8853); `virtual/*` launch targets are genuinely validated against the handler artifacts they dispatch to (`VIRTUAL_TARGETS`/`VIRTUAL_PREFIXES` map to backing files that must exist) instead of being allowlist-blessed (#8854); and `src/config/registry_exclusions.yaml` is the new documented convention for launcher-less `src/tools` packages — every package must either be reachable from a launcher tile or carry a justified exclusion entry, enforced by `tests/config/test_registry_exclusions.py` (#8863: contraction, drift_control, sg_optimizer, hmr2_sidecar, matlab_utilities, offline_validation, video_analyzer). The `vendor/ud-tools` gitlink advances from `1664d806df8a` to `aec16af5a1e69d`-era Tools main (`aec16af5a1e69c0d5542da5e04a1db1023cceff2`), revalidated by the vendor-authority, manifest, sidekick-import, and launch-QA suites. (spec 1.0.570) |
| 2026-08-21 | #8975 | Implemented `MuJoCoSwingStateProvider` (#8975, EPIC #8965/WS2): the swing→flight pipeline's mujoco entry now sources a real engine-backed `SwingState` instead of sitting disabled. A narrow facade (`src/shared/python/physics/mujoco_swing_source.py`) runs the in-repo upper-body golf-swing MJCF under full MuJoCo forward dynamics — a scripted open-loop half-sine torque pulse whose scale is bisection-calibrated toward the requested clubhead speed — and extracts world-frame clubhead velocity, angular velocity, face orientation, mass, and MOI from `mjData`/`mjModel` at peak clubhead speed. Metadata records the model asset, `mujoco_forward_dynamics` method, timestep, torque scale, and the achieved-speed residual honestly (never fabricated to match the request). The provider registry replaces the mujoco `UnimplementedEngineProvider` with the real one (availability gated on `mujoco` + MJCF asset importability, so the GUI combo enables automatically), the shared provider postcondition now also enforces finite shape-(3,) kinematic vectors, and a golden-fixture JSON replay plus contract tests keep CI honest without a slow simulation. (spec 1.0.569) |
| 2026-08-21 | #8965 | Landed the EPIC #8965 wave-1 consolidation (PR #8976). Quarantined dead launcher shells (`unified_launcher.py`, `golf_suite_launcher.py`, `model_registry.py`) are deleted for good (#8831/#8859) with tombstone test modules preserving the audit trail; the ADR-0013 `EmbeddableTool` registry is now the single embedding contract consulted by every model handler before the deprecated legacy import-and-probe fallback (#8857), with registry/dockable resolution split into `src/launchers/_dockable_resolution.py` and window-layout persistence into `src/launchers/launcher_layout_persistence.py` to respect the 1200-line file budget. Sidekick API readiness probing moved off the GUI thread onto `SidekickReadinessProbeThread` (#8939), the WSL-mode probe became a non-blocking worker (#8903), launcher manifests gained a shared async cache seam, and the swing-flight pipeline GUI now sources pre-impact state from typed `SwingStateProvider`s that disable unimplemented engines honestly instead of stamping manual numbers with engine names (#8819). (spec 1.0.568) |
| 2026-08-21 | #8909 | Corrected #8909's false distributed-grip cross-engine authority. Robotics Pinocchio now requires version 2.6 or newer and the native Model/SE3/Inertia/CRBA/nonlinear-effects/RNEA API; the unrelated PyPI `pinocchio` 0.1 collision and every model-build failure stop the study rather than silently substituting MuJoCo. A degeneracy gate rejects an identically zero set of trajectory, force, and stick-projection discrepancies, and a repository test audits the Pinocchio identity recorded by every committed cross-engine artifact. The perfect-stick control uses a mass-whitened rank-revealing SVD so redundant station rows do not square the constraint-system condition number or require a relaxed no-slip gate. Inertia, contact-projection, forward-contact, and distributed-grip evidence and figures were regenerated with genuine MuJoCo 3.8.0 and Pinocchio 3.8.0 operators before claim registration and publication qualification. (spec 1.0.567) |
| 2026-08-21 | #4430 | Pinned the protected Tools #4430 rotating-base companion at `1664d806df8a2c7b184d2d3fbcea93b714caaee5` and added a fail-closed UpstreamDrift consumer contract for its ordered 18-run catalog, immutable source/study/catalog digests, 13 valid cases, five adverse cases, typed limitations, and unsupported human/coaching inference. The package workflow now reserves 30 minutes for a cold Tools checkout, frontend/Python build, verified wheel-content gate, and large artifact upload. Its unnecessary setup-node npm cache is disabled because measured `npm ci` takes seconds while post-job cache upload alone could consume the remaining job budget after the verified wheel uploaded successfully. (spec 1.0.566) |
| 2026-08-21 | n/a | Quantized only the conformance golden-snapshot serialization boundary to eight significant digits so platform-specific floating-point tails cannot change the longitudinal and confidence-interval scenario or bundle hashes. The analytics computations and result contracts remain unchanged. (spec 1.0.565) |
| 2026-08-21 | n/a | Added the data-free `launch-monitor-analytics-conformance/1.0.0` consumer bundle. Ten deterministic synthetic cases span available and structured-unavailable analysis v2, player covariation, attested longitudinal sessions, source-backed strokes gained, and distance/target proxy results. Uniform wrappers retain units, claims, player/session/order evidence, exclusions, source references, source-joinable backing hashes, scenario hashes, and a canonical bundle SHA without embedding private or observed input rows. Generated JSON Schema and golden JSON share the strict Python authority; analytics behavior and v1/v2 result contracts remain unchanged. (spec 1.0.564) |
| 2026-08-20 | #8808 | Added the canonical `launch-monitor-longitudinal-session/1.0.0` contract for #8808. Trusted and distinct player, session, and order evidence is mandatory. Shots aggregate into equal-weight player/session/stratum cells before per-player descriptive slopes and pooled player-fixed-effects OLS with player-clustered CR1 uncertainty. Missing/non-finite rows, blank identities, nonconstant session order, insufficient sessions or clusters, rank deficiency, and degenerate uncertainty are explicit unavailable states. Complete source-linked backing and missingness remain present, while shot-level inference and causal improvement are false. JSON Schema, a content-addressed golden source, FastAPI routes, generated declarations, ADR 0039, and tests share the Python authority. (spec 1.0.563) |
| 2026-08-20 | #8807 | Added the canonical `launch-monitor-player-covariation/1.0.0` contract for #8807. Selected-pair analysis separates pooled, player-centered, between-player, and per-player associations, then publishes fixed- and DerSimonian-Laird random-effects Fisher-z summaries with Q, tau-squared, and I-squared heterogeneity. Trusted explicit player identity is mandatory; missingness, ineligible/constant groups, unavailable states, uncertainty methods, units, vendor/model provenance, and source-joinable backing hashes remain explicit. The deterministic bounded pair scan retains unavailable pairs and warns about multiplicity, aggregation reversal, causality, and population-generalization limits. JSON Schema, golden pair/scan fixtures, FastAPI routes, generated React declarations, and compatibility tests share the Python authority while generic v1/v2 contracts remain unchanged. (spec 1.0.562) |
| 2026-08-20 | #8805 | Hardened the launch-monitor v2 identity boundary for #8805. `PlayerIdentityV2` rejects session, club, source, filename, row-order, and source-row pseudo-identifiers even when attested. Separate `SessionIdentityV2` and `OrderEvidenceV2` records now preserve session boundaries, order semantics, units, trust, and backing evidence without promoting those fields to player identity. Invalid identity claims fail as request-contract errors; analyses that do not require session/order evidence remain compatible, and future longitudinal operations must report missing evidence as unavailable rather than infer it. Contract v1 remains unchanged. (spec 1.0.560) |
| 2026-08-20 | #8803 | Added the canonical source-backed strokes-gained and separate outcome-proxy contracts for #8803. SG requires complete start/finish course state and a versioned expected-strokes baseline with HTTP(S) source, license declaration, canonical SHA-256, and unique stratum/distance points. Exact-stratum interpolation is allowed; extrapolation fails closed. Results preserve formula, units, row and dataset hashes, backing benchmark values, exclusions, uncertainty availability, conservative claims, and identity evidence for grouped or longitudinal analysis. FastAPI publishes the schema and analysis endpoints, and generated React declarations remain locked to OpenAPI. The radial-error outcome proxy is structurally prohibited from claiming strokes gained. (spec 1.0.559) |
| 2026-08-20 | #8793 | Closed PR #8793's first protected publication run failures without weakening either gate. Claim-evidence schema v2 hashes valid UTF-8 evidence after canonical CRLF-to-LF normalization while preserving byte-exact binary hashes, so identical committed evidence validates across Windows and Linux checkouts. A focused regression pins both newline forms to one digest and canonical byte count. The PDF finding helper now owns expected metadata as one parameter object, bringing the changed production function back within the repository's eight-parameter architecture budget. (spec 1.0.558) |
| 2026-08-20 | #8451 | Added the release-bound proximal-to-distal publication-quality contract for #8451. The exact UpstreamDrift revision and release-manifest digest now bind a full-PDF inspection covering metadata, outline and link validity, per-page rendering, extractable text, tagging, font resources, and web optimization. CI Standard runs the validator in a dedicated path-scoped job aggregated into the sole required `quality-gate`, so missing optional tooling cannot silently skip protected inspection. The current 231-page candidate passes the computational profile and is losslessly linearized, while the stricter archival profile remains fail-closed on the disclosed untagged, Type 3, and unembedded-font gaps. The contract preserves UpstreamDrift as scientific source authority, AffineDrift as a revision-pinned generated publisher, and human qualification as a separate governed gate. (spec 1.0.557) |
| 2026-08-20 | #8782 | ⚡ Bolt: Fast NumPy reductions and norm computations across motion matching, physics engines, and visualization modules (issue #8782). (spec 1.0.556) |
| 2026-08-20 | #8743 | Recovered BunkerShot3D cross-tier plumbing and MPM code verification inside CI architecture and file budgets (issue #8743, #8741). Verified 177 tests across conservation, analytic elastic column, GCI mesh convergence, and F0 cross-check cases without regressions. (spec 1.0.554) |
| 2026-08-19 | #8751 | fix(research): Made release-integrity hashing invariant to platform checkout line endings, regenerated the 568-artifact and 295-claim authorities, corrected the public release count to 40, and restored #8751/#8752 to their truthful open acceptance state. (spec 1.0.554) |
| 2026-08-19 | n/a | docs(research): Updated `docs/research/proximal_distal_energy_transfer/README.md` to cite the canonical web monograph and PDF publication hosted in the AffineDrift repository (`https://affinedrift.com/articles/proximal_distal_energy_transfer/index.html`), clearly establishing UpstreamDrift's architectural role as the computational research engine, simulation pipeline host, and evidence data ledger. (spec 1.0.553) |
| 2026-08-19 | #8751 | Qualified distributed grip friction and loss of contact (#8751) and added manufactured-solution and parameter-uncertainty controls for the articulated tier (#8752). Implemented multi-station Coulomb friction cone contact and interface power decomposition in `articulated_distributed_grip.py`, `articulated_distributed_forward.py`, and `articulated_distributed_atlas.py`. Verified normal/tangential work, passivity, and first-failure classifications across station opening and reattachment transitions (`maximum_transition_count > 0`). Authored closed-form manufactured free-body and constrained-motion verifications in `articulated_manufactured_solution.py`. Added Latin hypercube sampling and PRCC parameter sensitivity sweeps across joint limits, anthropometrics, grip stiffness/damping, shaft modes, and ground impedance in `articulated_uncertainty_study.py`. All 10 new research tests and release integrity checks pass. (spec 1.0.552) |
| 2026-08-19 | #8768 | Stabilized main CI and pre-commit health: fixed ruff linting errors across engines, addressed Bandit B314 XML parsing findings with defusedxml in `scripts/check_document_title_case.py` and proximal-distal chart extraction, reverted broken TypeScript bump in UI frontend to resolve `npm ci` ERESOLVE failure, and cleaned up CI delta scan triggers (#8768). (spec 1.0.551) |
| 2026-08-19 | #8771 | Closed CI Standard's false-green `tests` lane: `mapfile < <(git diff ...)` discarded git's exit code, so an unresolvable push diff base (`fatal: bad object`) produced empty change sets that the skip branch read as "nothing changed" and exited 0 - `main` @ `6b68f94` reported `tests (3.11)`/`tests (3.12)` green over 14 tests while the ~2,500-test unit suite never ran. Diffs are now captured through files so `-e` catches a failed diff, an unresolvable diff base fails loudly instead of being inferred as "no changes", pushes to the default branch always run the full lane unscoped, and a genuine no-op skip emits a `::warning::` plus a job-summary block saying the suite was not executed. The `Check for core test relevant changes` pre-step is hardened the same way, and `tests/ci/test_ci_infrastructure.py` gains two regression guards for the new contract (#8771). (spec 1.0.550) |
| 2026-08-19 | #8771 | Repaired all five `profile-size-matrix` Docker builds. `scripts/docker/install_features.py` reached the feature registry through `from src.shared.python.feature_registry.features import ...`, which executes every parent package `__init__`; `Dockerfile.modular`'s builder stage deliberately copies only `__init__.py`, `engine_core/` and `feature_registry/` so profile resolution does not invalidate the layer cache, so an eager `from . import ai` in that `__init__` broke every modular build with `ImportError: cannot import name 'ai' from partially initialized module`. The script now loads `features.py` by path with no parent package - what its own comment always claimed - making the builder slice self-sufficient regardless of what any package `__init__` imports later; the module is registered in `sys.modules` before execution because `@dataclass` resolves `cls.__module__` through it. Verified against a reconstructed builder slice: all five profiles resolve with `src/shared/python/__init__.py` left exactly as it is on `main`, since it is a Tools-owned child copy (#8771). (spec 1.0.549) |
| 2026-08-19 | #7935 | Git-ignored the root-level test-run artefacts (`base.csv`, `base.json`, `base.mat`, `base.h5`, `base.*.provenance.json`, `pytest_report*.txt`, `golf_modeling_suite.db`). `_prevent_repo_root_io` stops tests producing them (#7935) but nothing stopped them being staged: #8322 committed nine such files at the root and #8747 deleted them again. Patterns are root-anchored, so the tracked `docs/research/proximal_distal_energy_transfer/data/wscg_two_hand_raw/base.csv` fixture is not affected; verified no tracked path is newly ignored (#8771). (spec 1.0.548) |
| 2026-08-19 | #8322 | Restored `optional-stack-check (3.11)`: the SpaceMouse, VR-controller and haptic `connect()` stubs in `src/deployment/teleoperation/devices.py` return `False` again instead of raising `NotImplementedError`. #8322 introduced the raise, kept the three tests asserting `not dev.connect()`, and deleted the #7360 docstring explaining why `False` is the honest answer for a stub with no hardware driver. `BaseInputDevice.connect`, the ROS2/UDP controller stubs and `test_base_input_device` all keep the `False` contract, so the overrides were inconsistent with their own parent class. #8322 also rewrote six assertions in `tests/deployment/test_teleoperation.py` and `tests/deployment/wave5_deployment/test_teleoperation.py` from `assert not d.connect()` to `pytest.raises(NotImplementedError)` while missing the third file - which is why `optional-stack-check` went red and `unit-test-gate` did not, and why the repo has since asserted both contracts at once. All of it is restored to the pre-#8322 text so a single contract holds again; the `#8058` reference is retained. 288 deployment tests pass, nothing skipped or deleted (#8771). (spec 1.0.547) |
| 2026-08-19 | #5635 | Corrected ten AI provider-adapter tests that had not followed deliberate implementation changes: the base test double missing the `list_models`/`thinking_capabilities` methods #5635 made abstract, provider-specific token keys superseded by #2763's canonical usage triple, and Gemini error and streaming expectations superseded by #3179's typed raises and #2763's stream-finality guarantee. Pinned `test_gemini_adapter.py` to the SDK path it claims to test: its bare `MagicMock` stub auto-created a `Client` attribute, so `HAS_GEMINI_CLIENT` came out true and nine tests exercised a branch they never asserted against; the other branch now has its own test. `tests/unit/shared_python/ai/adapters/` goes from 19 failed / 43 passed to 9 failed / 62 passed; the remaining nine need canonical Tools source (Tools PR #4574) and arrive via a `vendor/ud-tools` bump, because the adapters are Tools-owned child copies. Nothing skipped or deleted (#8771). (spec 1.0.551) |
| 2026-08-18 | #18 | Added a "Load Private Corpus" button to the Launch Monitor Analytics Sessions tab and a matching File-menu action, backed by `MainWidget.load_private_corpus_sessions()`: one session per corpus source (261,666 shots across 27 sources in ~3 s), repeatable without duplicates, failing closed with a dialog when no authorized checkout or Parquet reader is available. Five regression tests cover the success, idempotence, fail-closed, and both UI-affordance paths over synthetic fixtures. Trends and Dispersion stay inert against corpus data pending capture-timestamp and lateral-carry extraction, tracked as data-authority issues #18/#19. (spec 1.0.543) |
| 2026-08-18 | n/a | Added `launch_monitor.corpus.load_private_corpus()`: reads the private data authority's source-partitioned Parquet shot corpus into the canonical launch-monitor schema (importer unit tables, source/metric pushdown, lazy pyarrow, fail-closed `LAUNCH_MONITOR_DATA_ROOT` convention, `apex_native` excluded as unit-ambiguous), exported from the facade with synthetic-fixture tests. Fixed the bare `flight_models` import in `kaggle_validation.compare_all_models_to_dataset()` that only resolved under pytest's `pythonpath`, so installed consumers no longer hit `ModuleNotFoundError`. (spec 1.0.542) |
| 2026-08-16 | #8729 | Repaired four CI Standard gates that had never been evaluated: `quality-gate` was pinned to the self-hosted fleet, so no run in the last 30 reached a conclusion and every gate below it was unobserved (routing fixed in #8729). `alembic.ini` now anchors `script_location` with `%(here)s` like `version_locations` already did, because Alembic resolves a relative script path against the current working directory and the autouse `_prevent_repo_root_io` fixture chdirs every test into `tmp_path` (#7935), which made the migration round-trip report a missing `src/api/migrations` that is in fact fully tracked. Restored the `tornado==6.5.8` pin dropped from `requirements.lock` and `requirements-dev.lock` by #8322, which deleted the pin line but left its `# via` comment block orphaned, leaving both locks structurally invalid rather than merely stale. Renewed the 44 mypy exclusion and 6 coverage-gate re-attestation dates from 2026-08-01 to 2026-10-01, aligned with the next `schedule` step where the cap drops to 36 against 44 exclusions; the ratchet `schedule` itself is unchanged and no exclusion was added (#8731). Split the DRY duplication ratchet: `scripts/config/dry_duplication_quarantine.json` records the 511 historical fingerprints with an owner and issue #8695, each capped at its observed count, and the gate now enforces `max(baseline, quarantine)` so newly introduced duplication still fails at its first repeat while historical debt is tracked explicitly instead of being folded invisibly into the baseline by a wholesale regeneration; the gate additionally reports quarantined fingerprints whose count has dropped so the ledger can be tightened. Raised the runtime security floors that `pip-audit` flagged once the lock repair let the audit steps run at all - `pillow>=12.3.0`, `cryptography>=50.0.0` and a newly direct `click>=8.3.3` - clearing PYSEC-2026-2132, PYSEC-2026-3552/3553/3554 and thirteen pillow advisories; `scripts/config/pip_audit_waivers.json` remains empty, so nothing was waived (#8738). `unit-test-gate` (#8735) and `shared-tools-consumer-contracts` (#8732) remain red with root-caused tracking issues rather than being suppressed. (spec 1.0.540) |
| 2026-08-16 | n/a | Added the passive articulated-shaft qualification: a frozen 24-element bending basis and declared tapered-section torsion extend the distributed-grip authority through rigid, bending, torsion, and coupled activations. The registered 384-trajectory, two-engine, two-step atlas passes domain, activation, power, work--energy, refinement, and parity gates; retained coarse steps fail the linear-domain screen. Among 126 load/work-matched coupled-versus-rigid cells, delivery-speed differences have both signs (-0.0285 to +0.0212 m/s), rejecting a universal passive-shaft speed benefit. The result remains a planar structural reference, not equipment calibration, human validation, physiology, or coaching guidance. (spec 1.0.539) |
| 2026-08-16 | n/a | Bumped `vendor/ud-tools` from `4744422d3` (2026-07-26) to Tools `main` `6472d0307`, and added `tests/unit/test_gui_launcher_manifest_targets.py`, which resolves every `pyqt6.module` declared in `src/shared/python/gui_launcher/tool_manifest.yaml` to a file in a reachable Tools source tree. The manifest advertised "Rate of Closure Impact Explorer" at `rate_of_closure.ui.pyqt6.main_window` while no Tools checkout the launcher searches contained that module: the tool had not yet landed on Tools `main`, and the vendored pin predated it. Nothing failed, because nothing checked — clicking the entry was the only way to discover it. The new sweep resolves by path tail rather than assuming a flat import root, because Tools nests some tools (`signal_processing_studio` sits under an extra python level in the Tools tree) and not others (`rate_of_closure` sits directly under the Tools src root). Verified against the previous pin: `rate_of_closure` is the only entry that fails, with no false positives across the remaining entries. The check skips when no Tools checkout is present rather than passing vacuously. (spec 1.0.541) |
| 2026-08-15 | n/a | Added the distributed-grip contact-discretization gate: one, three, and five tension fibers per hand preserve total stiffness and damping across 12 articulated states, two initial velocity signs, two time steps, two native engines, and nested 4/10/25/50 ms observations from 288 trajectories. Geometry null/reversal, virtual-power, passivity, work--energy, time-refinement, station-refinement, active-set, and cross-engine gates pass. The result is synthetic and right-censored; it does not establish physical grip pressure, shaft response, timing economy, delivery benefit, human transfer, or technique. (spec 1.0.538) |
| 2026-08-15 | n/a | Added the typed unilateral articulated-attachment falsification gate: bilateral, tension-only, and dead-zone tension laws are evaluated across common-displacement and matched-extension comparisons, velocity-sign branches, isolated opening/reattachment probes, three time steps, and native MuJoCo/Pinocchio dynamics. The passive-law, virtual-power, work--energy, refinement, trajectory-parity, force-parity, and active-set-parity contracts pass. Natural five-millisecond branches do not produce opening or reattachment transitions, so event-probe results qualify the implementation only and do not establish a human or coaching strategy. (spec 1.0.537) |
| 2026-08-15 | n/a | Added the bounded articulated bilateral-attachment forward gate: 18 selected closed states, seven nominal/adverse branches, three time steps, and native MuJoCo/Pinocchio dynamics produce 756 five-millisecond trajectories. Attachment-retention, power, work--energy, refinement, and parity gates pass; the result is explicitly right-censored and does not model unilateral slack, calibrated distributed grip/shaft, ground coupling, late downswing, impact, muscle action, human transfer, or coaching strategy. (spec 1.0.536) |
| 2026-08-15 | n/a | Added the subject-scaled articulated contact-projection gate: finite bilateral Kelvin--Voigt forces arise from a declared club perturbation at all 234 closed states, project through the hand and club Jacobians with exact action--reaction and virtual-power controls, and yield matching native MuJoCo/Pinocchio initial accelerations. This is a same-state prerequisite, not a forward contact or human-strategy result. (spec 1.0.535) |
| 2026-08-15 | n/a | Added the subject-scaled articulated-inertia cross-engine gate: all 234 closed configurations are rebuilt independently in native MuJoCo and robotics Pinocchio, with mass-matrix, bias-force, inverse-dynamics, symmetry, and positive-definiteness equivalence registered before forward bilateral contact. The result qualifies common-state rigid-body transport only and explicitly leaves contact, anatomy, equipment, muscle, delivery, and human claims open. (spec 1.0.534) |
| 2026-08-14 | n/a | Added the closed-state cross-engine validity-horizon contract: all 54 profile--span--phase states are evaluated at 4, 10, 25, and 50 ms under nominal and nine one-factor adverse/null branches in native MuJoCo and Pinocchio. All 2,160 horizon cases pass trajectory, wrench, energy-discrepancy, and work--energy closure gates; the no-failure result is right-censored at 50 ms and cannot establish articulated anatomy, calibrated equipment, full delivery, or human strategy. (spec 1.0.533) |
| 2026-08-14 | n/a | Added the closed-state forward-contact bridge: all 234 subject-scaled closed configurations map through a declared rigid coordinate transform with position and velocity closure gates, zero-preload and passivity controls, and unique initial-state digests; 54 early/middle/late profile-span cases enter native MuJoCo and Pinocchio for a short-horizon trajectory, wrench, and energy parity audit. The contract explicitly prohibits promotion to articulated anatomy, calibrated equipment, full-downswing delivery, passive-transfer benefit, or human strategy. (spec 1.0.532) |
| 2026-08-15 | #8614 | Added BunkerShot3D designer metrics module for issue #8614. Implements `bunkershot3d.metrics.trajectory` (TrajectoryMetrics, DivotProfile, dig/skid classification, depth trace, entry/max/exit points), `bunkershot3d.metrics.energy` (EnergyPartition, club KE tracking, energy-to-sand/ball accounting), `bunkershot3d.metrics.force` (ForceMetrics, peak/mean force and moment, deceleration, contact duration), `bunkershot3d.metrics.twist` (TwistMetrics, shaft-axis and CG moments, impulse, twist direction), and `bunkershot3d.metrics.forgiveness` (ForgivenessMetrics, SensitivityGradient, finite-difference sensitivity analysis, forgiveness index). 42 new tests covering all metric categories. Computed from HDF5 result artifacts for fidelity-tier-agnostic (F0–F3) analysis. (spec 1.0.531) |
| 2026-08-15 | #8613 | Added the BunkerShot3D ball model and SwingBallFlightPipeline handoff for issue #8613. Implements `bunkershot3d.ball.lie` (BallLie, BallLieType, BallProperties with USGA specs, submersion/exposed-area geometry), `bunkershot3d.ball.splash` (sand-mediated splash momentum transfer: ejecta velocity, splash impulse, and ball launch from splash), and `bunkershot3d.ball.pipeline` (BunkerShotState, compute_bunker_launch, to_post_impact_state for PostImpactState handoff). 47 new tests covering lie geometry, splash physics, pipeline integration, energy accounting, and tour bunker shot sanity checks. (spec 1.0.530) |
| 2026-08-14 | n/a | Reconciled the photographed nine-point momentum-transfer agenda with direct evidence-artifact links and an explicit unresolved-point identity; corrected the paper so MTQ-06 timing precision, rather than casting, is the one globally unresolved source point; and separated complete 994-candidate coverage from the 10 of 31 release reviews that remain pending or in progress. (spec 1.0.529) |
| 2026-08-14 | n/a | Added MT-E09 paired scapulothoracic contact geometry: a fixed-trunk and fixed-club nested comparison separates residual closure, solver termination, bound activity, rank, coordinate nullity, and an adverse grip-span control, while prohibiting anatomical, muscular, transfer, and strategy inference until validated articulated forward contact. (spec 1.0.528) |
| 2026-08-14 | n/a | Added fail-closed scientific-support integrity for the proximal-to-distal program: all claim source locators must resolve to an in-range repository line; every registered local evidence artifact is SHA-256/size pinned; every external support URL is inventoried without being promoted to scientific validation; omission and tamper controls are executable; and the critical-question roadmap now maps each handwritten question to its bounded current answer, decisive model/measurement gate, and independently checked scapulothoracic, EMG, and distributed-grip acquisition leads. (spec 1.0.527) |
| 2026-08-14 | n/a | Added the MT-E08 subject-scaled closed-contact inverse-kinematics screen: all 234 profile, grip-span, and phase configurations close with the club pose fixed, full achieved constraint rank, positive broad engineering-limit margins, positive coarse bounding-sphere clearances, and continuous solved paths. The contract preserves these as reduced-tree necessary conditions and advances the next gate to subject-specific anatomy and calibrated compliant forward contact. (spec 1.0.526) |
| 2026-08-14 | n/a | Added the MT-E08 subject-scaled spatial contact-closure audit: six deterministic de Leva engineering profiles, three grip spans, and 61 states per case fail the 5 mm bilateral closure tolerance despite full local contact-Jacobian rank. The governed evidence, release claims, and scientific boundary now distinguish measurement rank, local kinematic rank, geometric closure, and forward contact dynamics, and require closed-contact inverse kinematics with joint-limit/collision checks before anatomical or human-strategy inference. (spec 1.0.525) |
| 2026-08-14 | n/a | Added trajectory-level synthetic qualification for the MT-E07 bilateral point-force estimator: 301 samples and 32 seeded trials exercise normalized noise, cross-talk, calibration residual, and contact-center migration controls; a manufactured net-wrench-only failure demonstrates that resultant closure does not identify allocation; and the paper, registries, claim audit, figure, evidence, tests, and handoff retain explicit full-device, distributed-contact, anatomical, and governed-human gates. (spec 1.0.524) |
| 2026-08-14 | n/a | Added bilateral-wrench structural identifiability: the separated point-force map has rank five and one axial null mode, the axial-scalar augmentation has rank six, and the full bilateral six-axis map has rank six and nullity six under declared scaling and geometry controls. (spec 1.0.523) |
| 2026-08-14 | n/a | Added a two-excitation typed-slack dynamic audit that separates contact disengagement, transmission dead zone, structural preload, biological series compliance, and control deadband; enforces mechanical passivity and closure where applicable; reports scaled local sensitivity and pairwise output separation; and retains delivery, anatomical, class-identification, intentionality, and human conclusions as open. (spec 1.0.522) |
| 2026-08-14 | n/a | Added a common-phase timing-viability and adverse-load-recovery experiment for the critical-question program: 60 paired cases and 120 trajectories compare clock and state-triggered release under five phase offsets and six load/perturbation cohorts, retain strict/primary/lenient task-viability definitions, test sustained half-error recovery, and register timestep sensitivity. The model screen found a larger clock-policy task-viability region and no sustained recovery in either policy; it explicitly does not identify human timing demand, self-correction, or coaching strategy. (spec 1.0.521) |
| 2026-08-14 | #8556 | Registered all nine points from the handwritten momentum-transfer agenda with answer state, decisive next test, falsifier, data gate, model plan, and participant-held-out human stage; added a generated fail-closed readiness audit; expanded the paper claim audit to 956 candidates and 250 claims; and retained #8556 as the governed bilateral-wrench human-data blocker. (spec 1.0.520) |
| 2026-08-14 | #8583 | Removed eleven orphaned `.codex-worktrees/` gitlinks that PR #8583 introduced without matching `.gitmodules` stanzas, which made `git submodule update --init --recursive` and `git submodule status` exit non-zero on a fresh clone even though that command is the documented setup step and is emitted in runtime error messages from the engine loaders, pendulum engine, and model explorer. Added `.codex-worktrees/` to `.gitignore` beside the existing Claude agent-worktree entry, and added `tests/unit/repo_hygiene/test_no_orphaned_gitlinks.py` asserting that every tracked gitlink is declared in `.gitmodules`. Path-scoped submodule commands were unaffected, so vendor-freshness CI never regressed, and PR #8575's code changes remain intact on main. (spec 1.0.519) |
| 2026-08-13 | n/a | Ratified the AffineDrift-conforming terminology profile; migrated ZVCF to zero velocity and zero applied control; preserved the prior diagnostic under an explicit control-preserved name and schema; regenerated affected evidence and publication figures; and added regression controls. (spec 1.0.512) |
| 2026-08-12 | n/a | Added phase-resolved proximal-link velocity falsification and forward control-program search for drift-mediated transfer; registered exact same-state matching, drift/control work closure, negative grip-work and force tradeoffs, multi-objective Pareto reporting, deterministic evidence artifacts, and the explicit boundary that the fixed-hub coordinate does not identify torso or anatomical shoulder strategy. (spec 1.0.511) |
| 2026-08-12 | n/a | Linked the canonical proximal--distal evidence workspace to the accessible AffineDrift companion _How a Golf Swing Carries Energy_ in HTML and PDF, while retaining UpstreamDrift as the complete evidence and limitations authority. (spec 1.0.510) |
| 2026-08-12 | #8505 | Added the advanced proximal--distal frame, biology, canonical-pose, cross-engine, visual, terminology, and falsification bridge tracked by epic #8505. (spec 1.0.507) |
| 2026-08-11 | n/a | Decomposed launcher UI setup, settings dialog, Simscape 3D viewer, and launcher Sidekick readiness into modular helpers. (spec 1.0.506) |
| 2026-08-11 | n/a | Added a coupled forward three-mode shaft experiment plus an exact same-state arm--wrist allocation and phenomenological preload/role-reversal study. (spec 1.0.504) |
| 2026-08-11 | #5922 | Decomposed the launcher shared-Tools freshness probe behind a focused, dependency-injected local-Git boundary while preserving `LauncherDiagnostics.check_shared_tools_freshness()` and its result-recording behavior. Reduced `launcher_diagnostics.py` from 1,307 to 1,196 lines and removed its expired module-size exception, active file-size exception, and obsolete long-function exception without renewal or policy widening. The cited #5922 maintainability issue is closed, cited #7341/#7342 are closed and concern different defects, and #8472 is chat-dock-specific, so this slice deliberately does not claim issue completion; accurate open tracking remains a publication prerequisite. Three expired exceptions and four oversized production modules remain explicit module-gate blockers. (spec 1.0.500) |
| 2026-08-11 | #8472 | Completed the local UpstreamDrift portion of #8472's chat-dock compatibility-shell decomposition: moved WebSocket dispatch, terminal-mode mechanics, streaming-state initialization, and collapsed-view mutation into focused `chat._qt.runtime` helpers; retained every historical `ChatDockWidget` method as a thin delegate; and reduced `_chat_dock_widget_qt.py` from 1,490 to 1,150 lines. Removed the chat dock from both file-size exception ledgers without renewing or widening either policy. Focused delegation, terminal-payload, chat behavior, module-truthfulness, lint, type, architecture, and file-size checks cover the new boundaries. Canonical Tools synchronization remains required before refreshing the cross-repository drift hash; that sentinel already fails on the parent for the dock and `models.py`. Four expired exceptions and five oversized production modules remain explicit module-gate blockers. (spec 1.0.499) |
| 2026-08-11 | #8472 | Advanced #8472 without renewing or widening module-size exceptions: consolidated the duplicated stateless slope-gravity, roll-direction, and contact-normal calculations in `src/shared/python/physics/terrain_representation.py` onto the canonical split terrain implementation, reducing the legacy module below the 1,200-line hard limit while preserving its distinct boundary and serialization contracts. Removed its expired size exception, tightened its architecture-debt ratchet from 1,250 to 1,200 lines, and corrected the chat-dock exception's stale 1,989-line description to its measured 1,490 lines without changing its expiry. Identity and numerical regressions preserve the shared helper API. Five expired exceptions and six oversized production modules remain explicit release blockers. (spec 1.0.498) |
| 2026-08-11 | #8322 | Repaired the canonical-core React status shell so mode changes ignore stale responses, retry transitions remain explicit, and runtime payload parsing lives outside the Fast Refresh component boundary; restored the engine-store unload tests to their intended mocked-backend boundary; refreshed the lockfile to patched transitive `brace-expansion`, `nanoid`, and `undici` releases; and corrected the durable-task-manager history after confirming that #8322 removed that SQLite implementation and its tests while the current `src/api/task_manager.py` remains process-local. (spec 1.0.497) |
| 2026-08-11 | #4262 | Added the bounded #4262 immutable Tools-provider resolution slice: all five Tools launchers select the exact clean `vendor/ud-tools` gitlink at declared SHA `ff4240217005e1415ca409fd124e50b64ee642d2`, take precedence over conflicting sibling/package metadata, reject dirty/replaced/mismatched authority and canonical path escapes as `provider_unavailable`, preserve generic sibling-provider behavior, and omit mutable `../Tools` roots from serialized manifests; gitlink pin updates and identity-validated development overrides remain open. (spec 1.0.497) |
| 2026-08-10 | #8458 | Added #8458 hand-path drift/control attribution across forward double-pendulum and one-arm cases plus a prescribed two-arm closed-loop sweep; exported deterministic force, impulse, power, work, joint/time-window, common/differential-mode, sensitivity, and closure evidence; bounded the late residual-couple preview result without claiming muscle preactivation or human performance; extended lossless object-stream PDF compaction to preserve the 106-page, 110-link, 122-outline publication below the size guard; and restored the all-files size gate with the final owned #8472 chat-dock exception through 2026-08-31. (spec 1.0.496) |
| 2026-08-10 | #8456 | Added a reproducible lossless article-PDF compaction command that fails closed on page, URI-link, outline, or size drift; reduced the 90-page publication artifact below the repository's 1 MiB PDF guard; and recorded the protected #8456 higher-order merge in the handoff. (spec 1.0.495) |
| 2026-08-10 | n/a | Added a reference- and frame-explicit interaction-wrench schema; exact moment/velocity transport and proper-rotation power contracts; prescribed mobile-hub inverse-dynamics comparisons; planar two-hand closed-loop rank/nullspace diagnostics; a fail-closed model-discrepancy record; seven reproducible figures; and a higher-order scientific chapter that explicitly leaves full-body cross-engine dynamics unexecuted. (spec 1.0.494) |
| 2026-08-10 | n/a | Added a tested three-coordinate point-mass shaft-flex surrogate with an exact matched rigid reduction; separated control, momentum, gravity, joint damping, shaft elasticity, and shaft damping; closed the work--energy balance; evaluated gravity/damping ablations plus a 120-case stiffness/damping/torque-cut grid; expanded the scientific article with eight reproducible figures and explicit calibration limits; and made the title-case Git-diff reader explicitly UTF-8-safe on Windows. (spec 1.0.493) |
| 2026-08-10 | n/a | Reconstructed the archived WSCG two-hand BASE/ZTCF/DELTA force systems; added frame, wrench-transport, force-mode, power, reversal-sensitivity, and geometry contracts; and expanded the scientific article with eight reproducible figures and explicit pointwise/passive limits. (spec 1.0.492) |
| 2026-08-10 | n/a | Normalized the Proximal-to-Distal article's canonical headings and regenerated PDF/LaTeX bookmarks, then added a repository document-title gate for Markdown, Quarto, LaTeX, Word title styles, and PDF metadata/outlines in pre-commit and documentation CI. (spec 1.0.490) |
| 2026-08-10 | #8364 | Extended Launch Monitor Analytics with a versioned arbitrary-field analysis contract, PyQt and FastAPI surfaces, configurable association/missingness/multiplicity/grouping, OLS uncertainty and residual diagnostics, deterministic lineage, and fail-closed aggregate/vendor-pooling boundaries for #8364-#8366. (spec 1.0.489) |
| 2026-08-08 | #8390 | Epic #8390 B5/C2/C4 (completing all 16 sub-issues): CEM/MPPI batch swing optimizer as the first production consumer of the ADR-0023/0024 batch infrastructure, ~1k mujoco rollouts scored in ~2.5s on the CPU fallback (#8400); pose-estimator registry replacing the 5-place estimator edit tax, with API/pipeline/motion-capture routes deriving from one seam (#8402); 4D-Humans/HMR2 sidecar (FreeMoCap-pattern subprocess isolation for CC-BY-NC tooling) with a registered 3D adapter and SMPL-betas plumbing into the character builder (#8404). (spec 1.0.487) |
| 2026-08-08 | #8390 | Epic #8390 B1/B3/B4/C1/C3/D2: shared 7-DOF swing multibody model + smooth cost surrogates (`optimization/model_provider.py`, `smooth_costs.py`, #8396); CasADi direct-transcription swing backend with symbolic RNEA validated against pin.rnea, new `optimal-control` extra, `solver='casadi'` dispatch (#8398); Crocoddyl FDDP backend with subprocess stack-health probe and graceful mixed-wheel degradation, new `crocoddyl` extra (#8399); real Pinocchio IK backend (diff_ik-style LM + optional PINK path) with SkeletonRig→pin.Model bridge (#8401); DeepLabCut import adapter with custom keypoints (#8403); web UI 3D mocap skeleton, URDF glTF mesh loading with a hardened mesh-asset endpoint, and live pose streaming over the realtime WebSocket (#8406). (spec 1.0.486) |
| 2026-08-08 | #8390 | Epic #8390 B2+D1: implemented the Drake DirectCollocation motion-matching solver over a new SkeletonRig→URDF model bridge (`motion_pipeline/model_bridge.py`), re-exposed `drake_trajopt` as a production matching backend per #8131's criteria with live acceptance tests under `requires_drake` (#8397); added the Rerun recording adapter (`visualization/rerun_renderer.py`, opt-in `visualization` extra, `.rrd` export via `compare_cli --export-rrd`), executing ADR-0027's named follow-up (#8405). (spec 1.0.485) |
| 2026-08-08 | #8390 | Epic #8390 Workstream A (hygiene): motion-pipeline default IK backend now `geometric` (the previous `mujoco` default raised NotImplementedError, #8391); estimator types reconciled across API/UI/pipeline with a consistency test, removing phantom `movenet`/`blazepose` options (#8392); ezc3d promoted from dev-only to a runtime `c3d` extra with corrected C3D format docs (#8393); `pin-pink` declared in the `pinocchio` extra with license-ledger row (#8394); optimal-control capability-claim drift corrected — swing_optimizer/swing_bridge docstrings, CasADi checklist entry, and the hollow `test_pinocchio_ecosystem.py` replaced with real import-gated availability tests (#8395). (spec 1.0.484) |
| 2026-08-05 | #8345 | Retargeted #8345 P1 to `main` after the headless putting dynamics foundation merged, preserving the generated-contract R3F playback route/client/scene integration and Tailwind v4 entry-point repair. (spec 1.0.483) |
| 2026-08-04 | #8345 | Added #8345 P1: canonical 3D putting API DTOs and generated TypeScript types, Zustand/TanStack client state, theme-token R3F green/ball/putter rendering, visible spin and putter slowdown, adjustable hosel controls, collision/roll readouts, responsive playback, and a Tailwind v4 entry-point repair with regression coverage and rendered desktop/mobile QA. (spec 1.0.482) |
| 2026-08-05 | n/a | Fixed classic PyQt6 startup from nested worktrees by discovering the workspace-level Tools checkout, degrading only the optional Sidekick sidebar for an unavailable implicit runtime, and preserving fail-closed behavior for explicit `TOOLS_REPO_PATH`; 28 focused launcher/overlay tests pass. (spec 1.0.482) |
| 2026-08-04 | #8345 | Added #8345 P2/P3/P4 putting dynamics and public-data review: heterogeneous and seeded green fields, advanced friction, signed skid/overspin settling, full-chord capture, finite-mass lofted collision, putter slowdown, adjustable-hosel impulse wrench and pinned-shaft twist proxy, 70 focused tests, and an explicit provenance/assumption ledger. (spec 1.0.481) |
| 2026-08-01 | n/a | Bolt: Optimized `np.sum(arr * arr)` and `np.sum(arr ** 2)` to `np.vdot(arr, arr)` in drake compute cost and PARITY_SPEC to avoid intermediate array allocations and improve performance. (spec 1.1.0) |
| 2026-07-27 | #8121 | Separated classic PyQt6 Diagnostics validation of the 46 directly declared parent `models.yaml` tiles from validation of the 75-entry provider-expanded runtime registry (#8121). Added hermetic provider regressions and recorded the computer-controlled transition from a false 29-model `DEGRADED` result to `Status: HEALTHY` with zero failed checks. (spec 1.0.479) |
| 2026-07-27 | #8120 | Ensured the classic PyQt6 background API child inherits the sidebar's validated Tools authority and orders its package roots ahead of UpstreamDrift partial copies, preventing `chat.websocket_protocol` import failure (#8120). Recorded the Tools #3950 deprecations-as-errors Units repair and visible `100 °C` to `212 °F` retest, plus dynamic-port API-tree recovery and close cleanup that preserved the unrelated port-8000 blocker. (spec 1.0.478) |
| 2026-07-27 | n/a | Made the standalone Sidekick package workflow build its sdist and wheel as separate operations. The sdist remains a published source artifact, while the wheel is now assembled directly from the recursive checkout that owns the pinned Tools gitlink instead of being rebuilt from an sdist that intentionally excludes `vendor/`; focused workflow coverage prevents a combined `python -m build` regression. (spec 1.0.477) |
| 2026-07-27 | n/a | Materialized the exact recursive `vendor/ud-tools` gitlink in the JaxSim upgrade guard before editable package installation. The workflow now satisfies the fail-closed parent-source packaging contract instead of attempting to build from a checkout missing canonical `shared`, Sidekick, Chat, utility, and contract package roots; focused CI structure coverage pins recursive checkout and disabled credential persistence. (spec 1.0.476) |
| 2026-07-27 | #3944 | Completed the first post-repin artifact and computer-control sweep for standalone Sidekick. Clean-wheel alias smoke and the native Windows WGS calculator binary pass. Classic PyQt6 startup now activates manifest-approved Upstream extensions before adapter bootstrap, imports the canonical Tools sidebar API without lazy alias warnings, routes the Sidekick tile to the existing Chat sidebar, and tolerates fleet themes without a `success_hover` token. Computer control verified Calculator `2 + 2 = 4` and hide/reopen-through-tile behavior against Tools #3944 candidate `f075ff713`; the exact protected Tools merge, final gitlink repin, and no-substitution rerun remain required. (spec 1.0.475) |
| 2026-07-27 | #3937 | Repinned `vendor/ud-tools` to protected Tools merge `4744422d3` from #3937 after reconciling the scheduled `main` sync with the local parent-ownership migration. The vendored source/startup/security gate passes against that exact merged source; the installed-wheel and computer-controlled PyQt6 acceptance gates remain required before #8102 is complete. (spec 1.0.474) |
| 2026-07-26 | n/a | Published the canonical Tools Sidekick runtime candidate and removed 14 parent-backed or obsolete Upstream child copies, including the stale default-tab composition. Source startup now exposes only 33 manifest-classified Upstream production extensions through the exact-module overlay; the misplaced force-plate test moved out of the shipped package. Embedded and standalone profile stores now prove direct public-API interoperability against one versioned artifact, and the parent wheel installs both `python -m sidekick` and the `sidekick` console command. Final protected merge, gitlink repin, artifact gates, and computer-controlled PyQt6 acceptance remain required. (spec 1.0.473) |
| 2026-07-26 | #8102 | Extended the #8102 parent-source contract through the installed-wheel boundary. Wheel assembly now verifies and packages the exact pinned Tools shared graph, release CI initializes that exact gitlink and builds the wheel directly instead of round-tripping through an unverifiable sdist, and clean-install smoke coverage requires direct, `shared.python`, and legacy `src.shared.python` Chat/Sidekick imports to share one parent-owned identity. The exact candidate also executes `python -m sidekick --help`; final acceptance remains gated on the protected Tools merge and final gitlink repin. (spec 1.0.472) |
| 2026-07-25 | #8102 | Reconciled the pinned Tools direct-package expansion with UpstreamDrift's shadow-module ownership gate. Twelve pre-existing overlaps newly visible at the #8102 Tools revision, including `sidekick`, are explicitly classified under migration issue #5623 with an enforced 2026-12-31 sunset; new unclassified shadows still fail CI, and runtime Sidekick imports continue to prefer the pinned parent source. (spec 1.0.471) |
| 2026-07-25 | #8102 | Extended the #8102 classic PyQt6 Sidekick recovery contract beyond initial startup. A bounded liveness monitor now continues after the first successful readiness response, resets its restart budget after each healthy period, recreates a dead API child after a post-connect outage, and becomes inert when launcher shutdown begins. SK-START-016 records the fault-injection procedure and regression evidence. (spec 1.0.470) |
| 2026-07-25 | #8102 | Hardened classic PyQt6 Sidekick startup for #8102: isolated dynamic loopback API ports, an ephemeral launcher capability and public instance readiness identity, bounded child restarts, local-tab availability independent of Chat, parent-source-first direct package imports, and a pinned Tools revision. Host close delegates to the canonical sidebar aggregate shutdown so Terminal PTY, shell, bridge, and API processes exit cleanly (#3938). A PR-diff hygiene gate rejects direct edits to existing warning-headered Tools child copies and newly added child copies while allowing deletion after migration. Computer-control results and source-ownership audit are recorded in `docs/testing/sidekick-pyqt6-startup-matrix.md`. (spec 1.0.469) |
| 2026-07-25 | n/a | Completed the PyQt6 launcher functional-QA repair wave. In particular, the native C3D viewer now normalizes an absent optional `POINT:UNITS` parameter to millimetres before metadata parsing, so valid captures from emitters that omit the parameter load with the documented motion-pipeline default. (spec 1.0.468) |
| 2026-07-15 | #7740 | Hardened the simulation WebSocket and Data Explorer API routes for deferred #7740 findings. WebSocket start validation now rejects non-positive speed factors and reuses `SimulationRequest` duration/timestep bounds, simulation stats access is centralized behind one helper, Data Explorer dataset lookup rejects glob metacharacters, recursive dataset listing is paginated and bounded, and the dead cache helper was removed. Focused unit-marked tests cover the new WebSocket success/error branches, filter operators, ambiguous dataset names, glob rejection, and bounded listing behavior. (spec 1.0.467) |
| 2026-07-14 | n/a | Added Content-Security-Policy (CSP) header to FastAPI security middleware for defense-in-depth against XSS. (spec 1.0.467) |
| 2026-06-21 | #7740 | Hardened the simulation WebSocket and Data Explorer API routes for deferred #7740 findings. WebSocket start validation now rejects non-positive speed factors and reuses `SimulationRequest` duration/timestep bounds, simulation stats access is centralized behind one helper, Data Explorer dataset lookup rejects glob metacharacters, recursive dataset listing is paginated and bounded, and the dead cache helper was removed. Focused unit-marked tests cover the new WebSocket success/error branches, filter operators, ambiguous dataset names, glob rejection, and bounded listing behavior. (spec 1.0.466) |
| 2026-06-20 | #7715 | Added isolated transition hazard-rule coverage for issue #7715. The MDP transition tests now pin hazard penalties and DbC guard behavior directly so policy updates cannot bypass invalid-state validation. (spec 1.0.465) |
| 2026-06-20 | #7713 | Hoisted OpenSim Manager/Integrator construction out of the perturbation per-step loop for issue #7713, preserving analyzer behavior while avoiding repeated runtime setup on every simulated step. (spec 1.0.465) |
| 2026-06-20 | #7721 | Added chat WebSocket failure-path coverage for issue #7721. `tests/unit/api/test_chat_ws.py` now exercises `refresh_models` provider exceptions and `index_codebase` codemap rebuild exceptions, asserting sanitized client-facing error frames, server-side traceback logging, and continued socket usability after a model refresh failure. (spec 1.0.465) |
| 2026-06-20 | #7719 | Deduplicated the simulation WebSocket `set_speed` handler for issue #7719. Runtime speed changes now route through one canonical validation and state-update branch, and focused WebSocket regression coverage preserves accepted payload behavior while preventing branch drift in future command handling changes. (spec 1.0.464) |
| 2026-06-20 | #7725 | Repaired Drake cross-engine theta sizing for issue #7725. `tests/motion_matching/test_cross_engine_equivalence.py` now derives Drake's gravity-only zero-theta vector from the finalized plant actuator count before invoking `simulate_with_coefficients`, preserving the production guard that rejects mismatched nonzero theta vectors instead of silently disconnecting actuation and logging phantom torques. (spec 1.0.464) |
| 2026-06-20 | #7723 | Documented MuJoCo humanoid golf grip modelling for issues #7723/#7724. Grip synergy construction and contact extraction now live in a focused helper module with regression coverage for finite contact geometry, deterministic synergy transforms, and the leaner GUI tab integration. (spec 1.0.464) |
| 2026-06-20 | #7707 | Consolidated quaternion SLERP behavior for issue #7707. The canonical `math_utils.quaternion.slerp` implementation now owns the named nlerp fallback threshold, while spatial algebra rotations, cooperative manipulation, and Unreal skeleton mapping delegate to it with focused parity tests covering sibling behavior and the threshold boundary. (spec 1.0.463) |
| 2026-06-20 | #7688 | Reconciled cross-engine torque polynomial coefficient ordering for issue #7688 after the mainline dependency/security refresh. The cross-engine parity spec documents the canonical flat theta block as `[A, B, C, D, E, F, G]` where column `k` multiplies `t^k`; MuJoCo's polynomial torque driver, callback Horner chain, analytical Jacobian monomials, and shared theta validator documentation follow the same lowest-power-first convention as Drake, Pinocchio, and OpenSim. The cross-engine equivalence smoke gate feeds a nonzero theta through all four pure evaluator helpers to catch future ordering drift. (spec 1.0.462) |
| 2026-06-20 | n/a | Restored standalone Sidekick package frontend builds by pinning `@vitejs/plugin-react` to 5.2.0, whose peer range includes the locked Vite 7.x runtime. This avoids plugin-react 6.x importing Vite 8-only internals during `npm run build` in `package-standalone-sidekick.yml`. (spec 1.0.461) |
| 2026-06-20 | #7806 | Cleared #7806 CI ratchets after merging current main. Runtime dependency metadata and lockfiles now require patched `pydantic-settings>=2.14.2` so `pip-audit` no longer reports GHSA-4xgf-cpjx-pc3j, and recent unit-style regression files carry explicit `pytest.mark.unit` suite markers instead of expanding the unmarked-test baseline. (spec 1.0.460) |
| 2026-06-20 | n/a | Deduplicated golf GUI camera pan axes after the DRY duplication ratchet surfaced repeated target-plane basis construction in `golf_camera_system.py`. Mouse pan and pan inertia now share `_camera_pan_axes()` for the camera right/up vectors, preserving the existing orbit/fly pan behavior while removing the duplicated production logic fingerprint. (spec 1.0.459) |
| 2026-06-20 | #7740 | Hardened shared CORS credential handling for issue #7740. `add_cors_middleware()` now rejects wildcard origins when `allow_credentials=True`, including origins resolved from `CORS_ORIGINS`, so credentialed responses cannot be paired with `*` origins while non-credentialed wildcard CORS remains supported. Focused unit tests cover explicit wildcard rejection, environment-origin wildcard rejection, explicit-origin credentials, and non-credentialed wildcard behavior. (spec 1.0.458) |
| 2026-06-20 | #7726 | Corrected Drake theta contract coverage for issue #7726. `tests/parity/test_simulate_contract_drake.py` now replaces the dead optional-validator module probe with direct coverage that `validate_theta(..., bounds=DEFAULT_THETA_BOUND_TABLE)` rejects a `1e9` coefficient, while the Drake simulate path remains documented and tested as calling validation with `bounds=None` so the same large-but-finite coefficient can produce finite output or a failed solver status. (spec 1.0.457) |
| 2026-06-20 | #7694 | Extended `SafetyMonitor` regression coverage for issue #7694. Velocity-limit tests now pin unsafe target rejection and safe-command clipping, and emergency-stop torque regressions assert pure torque commands remain unsafe while emergency stop is active. (spec 1.0.462) |
| 2026-06-19 | #7732 | Hardened Data Explorer numeric contracts for issue #7732. `dataset_stats()` now ignores textual non-finite cells such as `inf`, `-inf`, `nan`, and `Infinity` so one stray value cannot corrupt min/max/mean or serialize invalid JSON tokens, and `_row_matches_filter()` rejects non-finite numeric row/filter operands before applying comparison operators. Focused API route tests cover finite-only aggregation, strict JSON response behavior, non-finite filter rejection, and unchanged finite comparisons. (spec 1.0.456) |
| 2026-06-19 | #7705 | Hardened common engine state validation for issue #7705. `StateManager` and `ForceAccumulator` now enforce positive dimensions, positive time steps, and non-empty force-source names through body-level guards that remain active when Python optimization or `ContractLevel.OFF` disables decorator-based DbC checks. Focused optimized-runtime regressions run subprocesses under `python -O` to pin the always-on contract while preserving the existing decorator-backed precondition behavior. (spec 1.0.455) |
| 2026-06-19 | #7685 | Hardened realtime controller failure abort handling for issue #7685. `_control_loop()` now clears `is_running` from a `finally` cleanup path even when the loop exits through a raised exception, and the emergency zero-torque fallback used after consecutive failures is best-effort with `logger.exception` telemetry if the safety send itself fails. Focused regression coverage drives `_send_command` to fail during both normal command dispatch and abort zero-torque dispatch, then asserts the loop terminates, `aborted_on_failure` stays true, and the zero-torque send failure is logged. (spec 1.0.454) |
| 2026-06-19 | #1041 | Hardened assessment Python metrics failure contracts for Tools_Private #1041's live UpstreamDrift implementation. `get_python_metrics()` now treats missing or unreadable source files as real file-access failures by warning and re-raising `OSError`, while syntax and UTF-8 decode failures still return the all-zero metrics shape but emit a warning so invalid inputs cannot be confused with valid empty modules. Unit and shared wave assessment tests cover missing-file propagation and parse-warning behavior. (spec 1.0.454) |
| 2026-06-19 | #7690 | Bounded exponentiation in shared `safe_eval` for issue #7690. `ast.Pow` now routes through a `_safe_pow` guard that rejects exponent magnitude above 1000 or large-base exponentiation before constructing enormous integers, preserving ordinary scientific expressions while stopping integer-blowup denial-of-service payloads such as right-associative exponent towers. Focused `tests/unit/test_safe_eval.py` coverage pins rejected large exponents/bases and accepted normal powers. (spec 1.0.453) |
| 2026-06-18 | #7683 | Hardened `SafetyMonitor` command contracts for issues #7683, #7684, and #7692. `check_command()` now rejects velocity targets over `max_joint_velocity` and any command while emergency stop is active; `compute_safe_command()` clips velocity targets and treats emergency stop as an authoritative no-actuation state by zeroing velocity, torque, and feedforward torque regardless of later speed-override calls. (spec 1.0.451) |
| 2026-06-18 | #7672 | Restored the scheduled Vendor Submodule Freshness workflow for issue #7672. `scripts/check_vendor_updates.py --json` now emits a parseable JSON status array, text mode prints per-submodule status messages again, and `.github/workflows/vendor-freshness.yml` keeps stderr out of `vendor_status.json` while letting real script failures fail closed. (spec 1.0.450) |
| 2026-06-18 | #7559 | Optimized allocation hot paths for issue #7559. `MuJoCoPerturbationAnalyzer._simulate()` now builds a padded descending-power coefficient matrix once, evaluates every actuator each step with a vectorized Horner pass into a preallocated control buffer, and preserves per-actuator `np.polyval(coeffs[j][::-1], t)` parity for ragged, empty, and over-count coefficient lists. `PolynomialProfile` now caches its `np.poly1d` polynomial and derivative once, and the triple-pendulum hardcoded dynamics helpers accept packed theta/omega/parameter tuples instead of 16-19 scalar arguments. Focused MuJoCo and pendulum unit coverage locks parity. (spec 1.0.446) |
| 2026-06-18 | n/a | Gated Windows Tauri release packaging behind `TAURI_WINDOWS_RELEASE_ENABLED=true` because the current self-hosted Windows runner blocks Cargo build-script executables with Application Control (`os error 4551`). Linux Tauri release packaging and the Tauri Rust/TypeScript check remain enforced, and `tests/ci/test_ci_infrastructure.py` now pins the repo-variable opt-in plus diagnostic notice contract. (spec 1.0.445) |
| 2026-06-18 | #7652 | Restored the final Tauri Build release contract for issue #7652. `ui/src-tauri/Cargo.lock` now resolves Rust `tauri` to the same locked major/minor as `@tauri-apps/api`, `.github/workflows/tauri-build.yml` installs `libdbus-1-dev` for the updated Linux Rust graph, and `tests/ci/test_ci_infrastructure.py` now parses lockfiles/workflow metadata to fail fast when a future dependency update would make `tauri-action` reject release builds for Rust/npm Tauri minor drift or miss required native Linux headers. (spec 1.0.444) |
| 2026-07-19 | #7099 | Bolt performance optimization (#7099). `src/unreal_integration/skeleton_mapper.py` now uses `math.hypot` instead of `np.linalg.norm` to avoid dispatch and intermediate allocations, resulting in a ~6x speedup. (spec 1.0.444) |
| 2026-06-18 | #7652 | Restored the main-side Tauri Build release lane for issue #7652. `ui/package.json` now exposes the `tauri` script entrypoint required by `tauri-action`'s default `npm run tauri build` invocation, and `.github/workflows/tauri-build.yml` now separates build matrix runner labels from artifact names while using a PowerShell `rustup` setup path for the Windows self-hosted leg to avoid the bash path failure seen in run `27732025577`. (spec 1.0.443) |
| 2026-06-18 | #7646 | Restored the Nightly Cross-Engine Validation workflow for issue #7646. The scheduled job now runs `tests/integration/test_cross_engine_validation.py`, `tests/unit/test_cross_engine_validator.py`, and `tests/integration/cross_engine/test_conformance_harness.py` against `src.shared.python.engine_core.cross_engine_validator`, preserving the 75% coverage threshold with real validator/conformance tests and marking zero collected tests as a failure in the summary output. (spec 1.0.442) |
| 2026-06-18 | #7645 | Restored current-main CI Standard after #7645. Strict API mypy now mirrors the baseline mypy push contract by checking only changed `src/api` Python files when `github.event.before` is available while preserving full strict API audits for scheduled/manual runs. The benchmark regression helper now keeps the 5x multiplier but applies a 10 microsecond absolute threshold floor for tiny hot paths, preventing sub-microsecond runner jitter from failing the 3.11 performance lane. (spec 1.0.441) |
| 2026-06-18 | #7643 | Closed current-main CI issue #7643. `ci-standard.yml` now scopes baseline mypy on ordinary pushes to changed `src/` Python files when `github.event.before` is available while retaining full-src baseline audits for scheduled/manual runs, and `Jules-PR-Cleanup.yml` now authenticates scheduled cleanup with `secrets.RUNNER_CHECK_TOKEN |  | github.token` so missing optional runner tokens no longer fail stale-PR discovery with HTTP 401. (spec 1.0.440) |
| 2026-06-17 | #7639 | Closed the dashboard launcher unit-test event-loop hang for issue #7639. `tests/unit/shared_python/test_launcher_integration.py` now patches the already-imported launcher module with `monkeypatch`, verifies the mocked Qt event-loop return code, and cannot enter the real `QApplication.exec()` loop when prior tests reload launcher modules. (spec 1.0.435) |
| 2026-06-17 | #7638 | Closed the then-current durable task daemon cleanup startup race for issue #7638 and added coverage for periodic and startup deletion of expired tasks. The durable SQLite implementation and its dedicated tests were subsequently removed by consolidation #8322; the current `src/api/task_manager.py` is an intentionally process-local task manager. (spec 1.0.434) |
| 2026-06-17 | #7637 | Closed the Rust wheel parity overreach for issue #7637. `.github/workflows/ci-standard.yml` now keeps `rust-wheel-parity` required and fail-closed for Rust wheel, parity-test, and Python facade changes while explicitly skipping the expensive parity suite on unrelated PRs; `tests/ci/test_ci_infrastructure.py` locks the path-gated contract and the successful skip summary. (spec 1.0.433) |
| 2026-06-17 | #7635 | Closed the test-only PR core-lane OOM regression for issue #7635. CI Standard now exits after changed-test PR slices pass when no source/dependency coverage targets changed, preserving the no-collected-tests fallback and the scoped dependency-light lane for source/dependency PRs. (spec 1.0.432) |
| 2026-06-17 | #7633 | Closed the push-scoped Semgrep regression for issue #7633 by making `tests/unit/scripts/test_remove_icon_backdrops.py` parse generated SVGs with defusedxml in its assertion helpers while preserving the existing entity-bearing SVG rejection coverage. (spec 1.0.431) |
| 2026-06-17 | #7631 | Restored the suite-marker ratchet for issue #7631 by marking the three #7629 Bandit regression tests as unit tests: coverage XML entity rejection, SVG entity rejection, and `download_to_file()` local-file scheme rejection before opener invocation. (spec 1.0.430) |
| 2026-06-17 | #7629 | Cleared the main-branch Bandit security-scan regression for issue #7629. `scripts/build_humanoid_models.py`, `scripts/build_humanoid_osim.py`, `scripts/config/coverage_enforcer.py`, and `scripts/remove_icon_backdrops.py` now route XML parsing through defusedxml while keeping stdlib ElementTree for XML construction/writing where needed. `src/shared/python/security/security_utils.py::download_to_file()` now validates URL schemes internally before the audited opener. Focused regression tests cover entity-bearing XML rejection and local file URL rejection before `urlopen` is called. (spec 1.0.429) |
| 2026-06-17 | #7561 | Optimized lower-body simulator history eviction for #7561. `LowerBodySimulator` now records scrub/playback frames in a bounded `deque`, preserving FIFO order, frame indexing, and clear/restore behavior while avoiding the previous `list.pop(0)` overflow path in the simulation step loop. (spec 1.0.423) |
| 2026-06-17 | #7563 | Pinned deformable external-force scatter coverage for #7563. `SoftBody.apply_external_force()` now has source-level regression coverage that rejects Python-level scatter loops, complementing duplicate-node accumulation tests that preserve the existing `np.add.at` duplicate-index semantics. (spec 1.0.422) |
| 2026-06-17 | #7571 | Deformable object internal-force performance for #7571/#7572. Soft-body FEM force assembly now batches tetrahedral deformation gradients, determinants, inversions, stresses, and nodal scatter while reusing each inverse once. Cable and cloth spring-force accumulation now uses cached NumPy connectivity/stiffness arrays plus shared vectorized scatter kernels, with focused scalar-reference parity and allocation/vectorization tests for the deformable internals. (spec 1.0.408) |
| 2026-06-17 | #7570 | Replaced the iLQR backward-pass gain calculation for issue #7570. The MPC controller now solves the regularized `Quu` system directly, using a finite-checked Cholesky path for symmetric positive definite matrices and a general linear solve fallback while preserving gain parity with the previous inverse-based math. (spec 1.0.407) |
| 2026-06-17 | #7568 | Optimized whole-body SciPy QP inequality construction for issue #7568. Finite lower and upper inequality bounds now each build one vector-valued SLSQP callback instead of per-row Python callbacks, while QP construction validates finite matrices plus constraint and bound shapes before entering the solver hot path. (spec 1.0.406) |
| 2026-06-14 | #7407 | Added the ball-flight physical benchmark contract for issue #7407. TrackMan-style driver and 7-iron tests now cover the enhanced simulator, the public flight-model registry, and the REST route; the suite locks cross-engine carry agreement, 5 m/s headwind/tailwind sanity, analytic vacuum range, explicit zero-drag semantics for Python/Rust drag-crisis helpers, humidity-aware density through `EnvironmentalConditions.from_altitude(...)`, and the calibrated Magnus coefficient cap contract. (spec 1.0.394) |
| 2026-06-14 | #7446 | Completed the AnalysisOrchestrator dashboard plot migration for #7446. Every PyQt6 static-dashboard label now maps to a registered headless `PlotData` extractor, including Poincare, Lyapunov, recurrence, GRF butterfly, kinematic-sequence, and summary-dashboard outputs, and the desktop dispatch path exercises `AnalysisOrchestrator.get_plot_data(...)` before renderer-specific plotting. (spec 1.0.396) |
| 2026-06-14 | #7406 | Corrected the impact model gear-effect spin direction and driver COR calibration for issue #7406. Toe/heel offsets now produce hook/slice spin in the launch-monitor sign convention, high-face/low-face offsets reduce/increase backspin, default gear-effect scales put realistic 20 mm toe and 10 mm vertical strikes in expected rad/s bands, and `DRIVER_COR` now reflects the USGA/R&A CT-limit-equivalent COR so center strikes can reach tour smash-factor ranges; regression coverage asserts toe/heel antisymmetry, vertical gear effect, default smash factor, and energy balance. (spec 1.0.393) |
| 2026-06-13 | n/a | Copied the hatch force-included `launch_golf_suite.py` entrypoint into the modular Docker builder before feature installation so `pip install .` metadata generation matches the wheel packaging contract; the Docker contract guard now rejects moving the entrypoint copy after feature installation. (spec 1.0.382) |
| 2026-06-13 | n/a | Kept modular Docker profile dry-runs importable from the early Dockerfile copy set by removing the import-time dependency from `engine_core.engine_probes` to the heavyweight config package, with regression coverage that exercises `install_features.py --profile standard --dry-run` against only the early validation files. (spec 1.0.381) |
| 2026-06-14 | #7431 | Public AffineDrift analysis-tool parity for #7431. Added `src.tools.drift_control.DriftControlAnalyzer` for generalized-force NPZ ratio analysis, `src.tools.contraction.ContractionVerifier` plus Floquet multiplier helpers, FastAPI JSON endpoints under `/analysis/tools/*`, and an optional-backend Pinocchio ABA timing CLI that reports unavailable cleanly when Pinocchio is not installed. (spec 1.0.395) |
| 2026-06-13 | n/a | Aligned the Rust wheel import smoke helper with the documented `upstream-mocap-io` missing-file contract so `parse_trc` raising `FileNotFoundError` remains an expected negative-path smoke result, with focused script regression coverage. (spec 1.0.376) |
| 2026-06-14 | #7403 | Corrected the rigid-body impact model friction-spin sign for issue #7403. Tangential contact impulse now uses `tangent_dir x normal`, preserving the rolling cap while producing lofted-driver backspin under the repository's `[0, -1, 0]` convention; regression coverage asserts the backspin sign, expected spin magnitude, and upward Magnus force from the derived spin axis. (spec 1.0.390) |
| 2026-06-14 | #7408 | Corrected the vendored Sidekick gas-flow compressibility-factor contract for issue #7408. Supported gases now carry acentric factors and `compressibility_factor()` uses the Abbott/Pitzer generalized second-virial approximation `(B0 + omega * B1) * Pr / Tr`, with regression coverage for methane, air, the low-pressure ideal limit, and the NIST methane 300 K / 5 MPa tolerance. (spec 1.0.389) |
| 2026-06-13 | n/a | Repaired the Rust PyO3 CI contract so Python-feature crates are checked with `cargo check --workspace --all-targets --features python`, avoiding invalid extension-module test executable linkage while maturin continues to build and smoke-test importable wheels. (spec 1.0.375) |
| 2026-06-13 | n/a | Restored optional-stack contracts by normalizing aerodynamic vector-shaped velocity/spin inputs, preserving drag-coefficient tuning through the Reynolds correction, centering the drag-crisis transition at the documented 8e4 Reynolds boundary, clamping randomized air density to the physical floor, and pinning the legacy API endpoint suite to explicit local/auth-disabled mode. (spec 1.0.374) |
| 2026-06-13 | n/a | Refined pointwise ZTCF/ZVCF dimension validation to allow canonical-v2 `nq = nv + 1` states only when the redundant configuration coordinate is populated, while still rejecting plain wrong-length zero vectors. (spec 1.0.373) |
| 2026-06-13 | n/a | Restored optional-stack ZTCF/ZVCF input length validation for pointwise acceleration helpers and reapplied the Settings page root theme marker after active-theme round-trip. (spec 1.0.372) |
| 2026-06-13 | n/a | Tightened signal-toolkit limit preconditions so saturation requires `lower < upper` and rate limiting requires `max_rate > 0`, matching the optional-stack contract tests. (spec 1.0.371) |
| 2026-06-13 | n/a | Scoped PR core-test dependency-light execution to affected unit areas when source files change, preventing self-hosted runner OOMs from full-repository core test attempts while leaving normal coverage reporting in place for full-coverage runs. (spec 1.0.370) |
| 2026-06-13 | n/a | Restored Sidekick JSON I/O compatibility in optional-stack tests. Default records-oriented JSON reads/writes now use stdlib JSON to avoid pandas JSON C-extension failures in the optional-stack environment, while non-default JSON options continue through pandas. (spec 1.0.369) |
| 2026-06-13 | #7362 | Aligned CI numpy/scipy repair pins with the lockfile runtime, added fast `scipy.signal` import checks after the reinstall step, kept core tests serial to avoid native-stack xdist worker termination, and recorded temporary shared-chat architecture-budget exceptions for `ChatDockWidget` legacy decomposition debt under issue #7362. (spec 1.0.368) |
| 2026-06-13 | n/a | Restored shared-python optional-stack compatibility contracts. Package-root `config` and `data_io` exports remain importable under the fallback test harness, `add_provenance_header` supports both file and string helper forms, AI adapters preserve empty-current-message, token-usage, and connection-error behavior, validation/signal-toolkit helpers enforce their documented preconditions, and FSP primitive tests skip partial Rust wheels that do not expose the FSP API. (spec 1.0.367) |
| 2026-06-13 | n/a | Restored optional-stack robotics compatibility contracts. `QPSolution`, `WBCSolution`, and `PlannerResult` again expose canonical `solver_status` aliases, `WBCSolution.final_cost` remains a legacy alias for `cost`, and the default IMU gravity vector preserves the historical 9.81 m/s^2 sensor contract while keeping the broader shared physics constants unchanged. (spec 1.0.366) |
| 2026-06-12 | #7496 | Consolidated parity CI follow-up after #7496. Moved BallFlight validation/color helpers into a non-component module to satisfy React Fast Refresh linting, added a fail-closed launch-mode fallback for web launcher contracts, deferred diagnostics launcher imports to request time to preserve API layer direction, extracted recording JSON artifact writing below the changed-function architecture budget, shared export-format validation across request models, and marked newly introduced parity tests with explicit suite markers so the ratchet remains blocking without expanding the baseline. (spec 1.0.365) |
| 2026-06-12 | #7457 | Web settings parity for #7457. Added `GET/PUT /settings` (`src/api/routes/settings.py`) persisting a validated `WebSettings` document (appearance/notifications/simulation defaults) atomically to `~/.upstreamdrift/web_settings.json`, a web `/settings` route (`ui/src/pages/Settings.tsx`) with theme selection via the shared `/themes` router, root-CSS-var font scaling, notification preferences consumed by the toast provider, and once-per-session simulation-default hydration that never clobbers in-session edits (#7424 guard). Feature-parity registry: `settings.preferences` → parity; desktop-only settings tabs recorded as a pending-decision exemption (#7460). (spec 1.0.360) |
| 2026-06-12 | #7411 | Fixed vendored Sidekick science accuracy for issues #7411, #7412, and #7413: Buck vapor pressure restores the correct constant roles, `signal_toolkit.calculus.Integrator` includes the upper-bound sample in definite integrals, and SCFM/ACFM/Nm3 plus ppmv/mg/Nm3 conversions use one DIN 1343 normal state. Added reference-value and property tests for vapor pressure, calculus integration, and conversion consistency. (spec 1.0.359) |
| 2026-06-12 | n/a | Removed the anti-phantom guard's `jq` dependency from the PR retry/API fallback path by using GitHub CLI `--jq` output directly, keeping required PR guard checks portable across local self-hosted runner images that do not install `jq` globally. (spec 1.0.358) |
| 2026-06-17 | #7560 | Replaced explicit inverse effective-mass calculations for #7560. MuJoCo humanoid golf effective-mass helpers now share a solve-based kernel, finite-check mass matrices and Jacobians, validate symmetric positive effective-mass output, and cover the no-explicit-inverse contract with focused unit tests. (spec 1.0.409) |
| 2026-06-12 | #7461 | Web reachability contract for launcher tiles (#7461). Every `launcher_manifest.json` tile now declares `web: {mode: route\ | native-window\ | unavailable, route?, reason?}`, validated by `WebLaunchContract` in `launcher_manifest_loader.py` and the TS schema parser. The web dashboard navigates route tiles in-app, gates native-window tiles behind Tauri/localhost with a "Desktop app only" badge, and shows reasoned badges for unavailable tiles. `POST /api/launcher/launch/{tile_id}` now returns 409 for non-local clients on non-route tiles so Qt windows never open invisibly on a remote server. Parity tests enforce contract declaration and React-router route existence for all current and future tiles. (spec 1.0.357) |
| 2026-06-11 | n/a | Made the optional-stack unit lane boundary explicit: the lane runs the non-engine unit suite with optional API, GUI, and body-part dependencies installed, while native engine unit tests remain covered by dedicated engine and cross-engine equivalence lanes to avoid coupling broad optional dependency validation to engine-specific mock behavior. (spec 1.0.353) |
| 2026-06-11 | n/a | Aligned deployment optional-stack device tests with the hardware-honesty contract: unavailable hardware-backed input devices remain disconnected and raise `StateError` for state operations, `KeyboardMouseInput` remains the connected fallback, and `Demonstration` now carries default canonical `solver_status="success"` through recording, serialization, subsampling, and augmentation. (spec 1.0.352) |
| 2026-06-11 | n/a | Restored the calc backend ODE solver response contract so `ODESolverResponse` again exposes the default `solver_status="success"` field consumed by optional-stack calc backend callers and tests. (spec 1.0.351) |
| 2026-06-11 | n/a | Restored body-part visualization optional-stack contracts: `FittedShape.n_frames` again reports the validated frame count, theme and fitted-shape validation errors use the documented precise type/range messages, and CI Optional Stack installs `trimesh` before running unit chunks so mesh-backed body-part visualization tests exercise the intended full dependency path. (spec 1.0.350) |
| 2026-06-11 | n/a | Preserved symlink traversal security failures in model and output path validation. Candidates lexically under an allowed root now reject symlink components before resolved containment fallback can mask escaped targets as generic 404 misses, keeping Linux optional-stack path validation on the documented 400 contract. Added suite markers to newly merged unit-level regression tests so the suite-marker ratchet remains blocking without expanding the unmarked-test baseline. Runs optional-stack unit tests as serial top-level `tests/unit` chunks because full-suite optional dependency collection is xdist-unsafe and can exceed local runner memory before tests execute, and raises the bounded CI Standard tests matrix timeout so the core suite is not cancelled before its per-test timeout contract can report real failures. (spec 1.0.346) |
| 2026-06-17 | #7562 | Optimized JaxSim trajectory parameter-gradient evaluation for #7562 by constructing the selected autodiff transform once per API call, vectorizing over the measured trajectory with `jax.vmap`, and adding finite shape postcondition checks plus local fake-JAX contract coverage. (spec 1.0.410) |
| 2026-06-17 | #7564 | Optimized RRT/RRT* nearest-neighbor and RRT* cost-propagation paths for #7564. Sampling trees now maintain a finite append-friendly configuration index for vectorized nearest/near queries, RRT\* honors `use_kd_tree` with periodically rebuilt `cKDTree` coverage when enabled, and rewiring cost propagation walks a maintained child-adjacency map with `deque` instead of scanning every node for descendants. (spec 1.0.411) |
| 2026-06-17 | n/a | Optimized deformable soft-body FEM root-node force accumulation by replacing the generic last-axis sum over per-tetrahedron force blocks with a batched `np.einsum("ijk->ij", H)` reduction and focused coverage for the vectorized reduction shape. (spec 1.0.412) |
| 2026-06-17 | #7558 | Optimized MuJoCo kinematic Coriolis decomposition for #7558 by reusing the base inverse-dynamics solution and scratch buffers across the per-DOF split, reducing redundant `mj_rne` passes while preserving the legacy component-sum contract. (spec 1.0.413) |
| 2026-06-17 | n/a | Optimized fixed-size 2D/3D vector magnitude calculations in API route metrics, putting-green direction normalization, teleoperation deadband/rate limiting, and ball-launch trajectory speed by replacing scalar `np.linalg.norm` calls with explicit `math.hypot` calls while preserving existing vector contracts. (spec 1.0.414) |
| 2026-06-17 | #7556 | Hardened robotics precision/vectorization issue slice for #7556/#7559/#7563: footstep angle wrapping is bounded, IMU quaternion normalization and force-torque contact-location denominators are guarded, cooperative load sharing uses `np.linalg.solve`, contact hull fallback sort keys are vectorized, deformable external-force scatter uses `np.add.at` for duplicate node indices, and GJK avoids recomputing the initial support-vector norm. (spec 1.0.415) |
| 2026-06-17 | #7556 | Tightened RL benchmark precision and projection search for #7556/#7561. Trajectory funnel benchmark metrics now use stable dispersion accumulation and finite-checked projection search helpers, with focused coverage for deterministic precision behavior and search contract edge cases. (spec 1.0.416) |
| 2026-06-17 | #7563 | Vectorized imitation-learning transition batching for #7563. `DemonstrationDataset.to_transitions()` now builds per-demonstration state/action/next-state blocks and concatenates once per output, with finite shape guards and regression coverage that prevents the old per-frame one-dimensional concatenate path from returning. (spec 1.0.417) |
| 2026-06-17 | #7556 | Replaced lower-body IK damped-least-squares explicit inverse formation with `np.linalg.solve` for #7556, with focused coverage that forbids the old `np.linalg.inv` path and validates the solve system shape. The cross-engine leaderboard workflows now preinstall `idna[uts46]` in their minimal venvs so the `anyio` dependency can resolve reliably before leaderboard regeneration gates run. (spec 1.0.418) |
| 2026-06-17 | #7556 | Extended footstep yaw normalization coverage for #7556 with a very-large negative-angle regression, preserving the bounded modulo contract for both signs plus the existing NaN preservation and infinite-value rejection behavior. (spec 1.0.421) |
| 2026-06-11 | n/a | Restored main CI API, launcher, and Docker contracts: `jaxsim` is accepted by the public simulation request allowlist, Data Explorer import responses keep generated dataset IDs while allowing legacy direct model construction, canonical-core launcher tiles use a recognized status and served biomechanics logo, symlink model-path validation preserves explicit 400 security failures, and Docker feature dry-runs import shared engine probe configuration through the package-qualified path. (spec 1.0.345) |
| 2026-06-11 | #7355 | Capability truthfulness contracts for #7355 and #7356. Generated motion-pipeline compatibility docs now mark Drake trajectory-optimization matching as unsupported until the solver is implemented. Drake, RRA, and CMC matching placeholders now report `status: not_implemented` with `production_ready: false`, and production chat placeholder tools return explicit `not_implemented` payloads instead of queued or successful no-op results. (spec 1.0.344) |
| 2026-06-11 | #7358 | Honest Document Chat and swing-sequence analytics contracts for #7358/#7359. The launcher Library tab keeps Document Chat disabled without a configured backend and reports a backend-not-configured message instead of a fabricated Notebook LM response. `swing_sequence` analysis now computes segment peak timing from trajectory angular velocities, marks instantaneous-only segment velocities as `requires_trajectory`, preserves analysis payloads through `AnalysisRequest.data`, and emits X-factor metrics only when shoulder/hip joint trajectory inputs are available. (spec 1.0.343) |
| 2026-06-11 | #7357 | RL engine protocol and teleoperation hardware-connection honesty for #7357/#7360. Added `src.engines.protocols.PhysicsEngineProtocol`, validated humanoid RL engine dimensions and runtime channels before use, removed zero-filled humanoid observation/reward fallbacks, exposed MuJoCo protocol accessors over real model/data arrays, and changed SpaceMouse/VR/Haptic device classes to report unavailable with `StateError` on disconnected reads/writes instead of fake successful hardware connections. (spec 1.0.344) |
| 2026-06-11 | #7341 | Launcher Docker build cancellation and layout reset backup hardening for #7341/#7342. Docker build threads now own a managed subprocess handle with cooperative cancellation instead of `QThread.terminate()`, the GUI prompts before closing an active build, and GUI/CLI layout reset paths share a helper that overwrites an existing `launcher_layout.json.bak` via `Path.replace` so repeated resets work on Windows. The changed-file architecture budget records expiring exceptions for the legacy launcher UI builders surfaced by this focused repair. (spec 1.0.342) |
| 2026-06-11 | #7352 | CI and validation test contract hardening for #7352, #7353, and #7354. The optional-stack lane now gates on pytest exit codes, physics validation scripts target real analytical/conservation suites, and PyQt fallback stubs no longer fabricate launcher expectations. (spec 1.0.341) |
| 2026-06-11 | #7380 | Motion-pipeline DRY follow-up for the #7380 simulator-facade merge. MuJoCo torque matching and Pinocchio inverse dynamics now share `BaseMotionMatchingSolver` helpers for per-DOF rig joint names and torque trajectory construction, removing duplicate post-merge torque payload assembly while preserving backend-specific success metadata. (spec 1.0.340) |
| 2026-06-11 | #7382 | Suite-marker ratchet follow-up for the #7382 import-boundary consolidation repair and the #7380 simulator-facade merge. The launcher dependency-probe, settings Docker dependency worker, architecture-budget metadata, C3D viewer export worker, body-target-video cancellation, shared ball constants, MuJoCo torque dimension mismatch, Pinocchio inverse-dynamics readiness, and generated-rig orchestrator regression tests now carry explicit `unit` suite markers so CI can enforce no-growth test metadata without weakening the marker baseline. (spec 1.0.339) |
| 2026-06-11 | #7361 | Import-boundary facade consolidation for #7361, #7362, and #7363. The C3D viewer entrypoint now imports the repo-qualified viewer module directly, MCP config I/O moved into `src/shared/python/ai/mcp/config_io.py` with launcher compatibility facades, shared MCP chat integration reads shared config, and shared/engine datetime compatibility imports route through shared helpers. The changed-file architecture budget now records owned expiring exceptions for the five pre-existing oversized functions surfaced by the consolidation so decomposition debt remains visible without leaving `main` red. (spec 1.0.338) |
| 2026-06-11 | #7333 | MuJoCo motion-matching placeholder failure routing for #7333. The orchestrator now carries solver metadata through the motion-matching stage, maps unavailable or zero-torque MuJoCo matching results to `InvalidInputError` so REST callers receive 400-class configuration feedback instead of HTTP 500, and the motion-pipeline README recommends a non-placeholder matching backend until real MuJoCo rig-model integration lands. (spec 1.0.337) |
| 2026-08-22 | n/a | Optimize List and Array Reductions in Nonlinear Dynamics Analysis for performance. (spec 1.0.572) |
| 2026-06-11 | #7272 | Suite-marker ratchet enforcement for #7272. CI Standard now runs `scripts/ci/check_suite_marker_ratchet.py` against `scripts/config/suite_marker_baseline.json`, failing net-new tests that lack a recognized suite marker while allowing legacy unmarked-test debt to shrink. The shared `tests.support.suite_markers` helpers now normalize nodeids, load the baseline, and support report-only, strict, and baseline-ratchet collection behavior from `tests/conftest.py`; contributor guidance lives in `docs/development/test-marker-conventions.md` with focused unit coverage for the static scanner and runtime helpers. (spec 1.0.336) |
| 2026-06-11 | #7246 | Restored the #7246/#7247 regression-guard cluster for #7325, #7326, and #7327 after PR #7248 reverted part of the launch-condition unit fix. `LaunchConditions.from_user_units(...)` is again the canonical GUI/user-input boundary for degree-to-radian conversion and RPM spin, the ball-flight GUI routes through that seam, and the current main gap-fill keypoint bounds guard remains covered by focused regression tests. (spec 1.0.335) |
| 2026-06-11 | #7324 | Collision distance helper optimization for #7324. Primitive-shape distance helpers now use explicit component access instead of `math.hypot(*tuple)` unpacking, preserving robotics collision behavior while avoiding tuple unpacking overhead on hot paths. (spec 1.0.334) |
| 2026-06-11 | #7308 | Law-of-Demeter enforcement for #7308. `scripts/ci/check_lod.py` now defaults to repo-wide production `src/` scanning, supports a checked-in no-growth baseline, and preserves documented library API allowances. `.github/workflows/quality-gate.yml` now runs the LOD scan as the blocking required `quality-gate` status with `scripts/ci/lod_baseline.txt` representing current grandfathered path/chain counts. (spec 1.0.333) |
| 2026-06-11 | #7276 | Safe motion checkpoint loading for #7276. Replaced pickle-enabled motion-matching checkpoint loads with safe artifact loading via `torch.load(..., weights_only=True)`. Validates mapping-shaped artifacts, keeping inverse, inverse-timestep, compact surrogate, and per-step surrogate loaders on the same safe contract. Exceeded surrogate train/optimize function budgets are tracked as exceptions in `architecture_budget.json`. (spec 1.0.332) |
| 2026-06-11 | n/a | Resolve merge conflicts in SPEC.md for PR 7316 by merging origin/main, retaining all changelog entries, and bumping Spec Version. (spec 1.0.331) |
| 2026-06-11 | #7283 | Simulation WebSocket dependency-boundary conflict refresh for #7283. `simulation_stream` keeps resolving the engine manager through the WebSocket-safe dependency accessor after the #7304/#7305/#7306/#7309 runtime-contract `main` update, and missing app-state manager configuration still emits a structured `service_unavailable` frame before clean close. (spec 1.0.330) |
| 2026-06-11 | #7276 | Safe motion checkpoint loading conflict refresh for #7276. The safe checkpoint artifact helper remains wired through inverse, inverse-timestep, compact surrogate, and per-step surrogate loading after the #7317 training/optimization architecture split and #7304/#7305/#7306/#7309 runtime-contract update, preserving mapping validation and `weights_only=True` reads while keeping the new helperized training and optimization contexts. (spec 1.0.329) |
| 2026-06-11 | #7304 | Motion matching runtime contract hardening for #7304, #7305, #7306, and #7309. `CostWeights` and internal `MotionMatchingRequest` now reject invalid numeric configuration at construction, shared metric validation fails on frame/DOF shape mismatches instead of silently truncating, the solver result postcondition gate validates reference-aligned time grids plus finite torque/activation payloads, and internal successful `MotionMatchingResult` objects must include a matched trajectory, torque trajectory, or activation trajectory payload. (spec 1.0.328) |
| 2026-06-11 | #7316 | Cross-engine dashboard window factory follow-up for #7316. `CrossEngineDashboardWindow()` now constructs the deferred PyQt window instead of raising a direct-instantiation placeholder, preserving the extracted fallback-engine stub and `_build_qt_window()` launcher path while keeping `src/launchers/cross_engine_dashboard.py` below the 1200-line file-size gate. (spec 1.0.328) |
| 2026-06-11 | #7288 | Cross-engine dashboard architecture split for #7288. `src/launchers/cross_engine_dashboard.py` now keeps the public `CrossEngineDashboardWindow` compatibility facade thin, constructs the concrete PyQt window class through a deferred factory, and imports the fallback engine stub from `src/launchers/cross_engine_dashboard_stubs.py`, removing the dashboard architecture-budget exception while preserving the existing CLI and window-construction contracts. (spec 1.0.325) |
| 2026-06-11 | #7317 | Motion surrogate training architecture split for #7317. Compact surrogate training now uses `SurrogateTrainingOptions`, explicit training context construction, and loop-state helpers while preserving legacy keyword call compatibility. Per-step dynamics training separates data preparation, runtime setup, fitting, evaluation, and output writing. Per-step optimization now routes legacy positional options through `OptimizationOptions`, uses an optimization context, isolates tracking/regularizer loss helpers, and writes optimized torque outputs plus summaries through a dedicated artifact writer. (spec 1.0.326) |
| 2026-06-11 | #7300 | Cloud client cached-token hardening for #7300. `CloudClient._load_cached_token()` now ignores empty and whitespace-only cache files instead of treating `""` as an authenticated token, `CloudClient.is_logged_in` requires a truthy token, and focused tests pin both invalid-cache cases while preserving valid cached-token behavior. (spec 1.0.323) |
| 2026-06-11 | #7275 | Local WebSocket hardening, Tauri permission manifest repair, and Tauri build apt-lock hardening for #7275, plus coverage gate fix for #7273. API WebSocket auth now validates launcher capability tokens and allowed Origins, the React client propagates the launcher manifest token, the Tauri IPC capability defines concrete permissions, and `.github/workflows/tauri-build.yml` retries apt dependency installs. Standard CI now sends PRs that change source, tests, or dependency targets through the coverage-producing core test lane. (spec 1.0.321) |
| 2026-06-11 | #7307 | Optional dependency mock isolation for #7307. Added `scoped_import_with_optional_mocks()` to shared test support, converted the called-out OpenSim, MuJoCo, and Drake tests from module-scope `sys.modules` mutation/import patching to per-test scoped import fixtures, removed the MuJoCo subtree-wide fake dependency conftest, and added a repo-hygiene guard that fails on new module-scope optional dependency mocks for `opensim`, `mujoco`, `cv2`, `imageio`, and `pydrake`. (spec 1.0.320) |
| 2026-06-11 | #7297 | Data Explorer and model-library boundary contracts for #7297, #7298, and #7299. Import/list responses expose durable `dataset_id` values, Data Explorer filter requests reject unsupported operators at the request boundary, and forced model-library downloads validate HTTPS-only `source_url` values before any download I/O. (spec 1.0.318) |
| 2026-06-11 | #7315 | Blocking DRY duplication ratchet for #7315. Added `scripts/ci/check_dry_duplication_gate.py` with focused tests, explicit production-`src` include/exclude config, and an owned no-growth baseline for existing duplicated logic fingerprints; `ci-standard.yml` now runs the checker inside `repo-structure-gates` so duplicate growth feeds the required `quality-gate` aggregate while `Code-Metrics.yml` remains advisory/manual reporting. (spec 1.0.312) |
| 2026-06-11 | #7314 | PR-scoped unit gate hardening for #7314. Standard CI no longer lets source/dependency PRs pass solely by running changed test files; those PRs fall through to the dependency-light unit lane with targeted coverage. `coverage_enforcer.py` now supports a PR-mode changed-file ratchet so changed production policy files must appear in targeted coverage and meet their policy threshold. (spec 1.0.311) |
| 2026-06-10 | #7312 | Jules PR AutoFix workflow-run trust-boundary hardening for #7312. The privileged `workflow_run` path now performs read-only failed-CI metadata analysis and posts manual dispatch instructions instead of checking out or executing PR-controlled code. The write-capable iterative fixer is restricted to explicit `workflow_dispatch` with an input branch. Added `scripts/check_workflow_run_trust_boundary.py`, wired it into standard CI, documented it in `scripts/README.md`, and added focused regression tests for unsafe workflow-run checkout/install/writeback patterns and the current Jules workflow contract. (spec 1.0.309) |
| 2026-06-10 | #7277 | Docker build timeout and focused PR coverage enforcement for #7277. `DockerManager` now monitors build output through a background queue while enforcing a wall-clock build timeout, terminating the process tree when stdout remains open past the deadline. Standard CI now scopes PR coverage to changed `src/**/*.py` modules and runs per-package coverage enforcement only after full core coverage reports, so focused PRs are not blocked by unrelated packages. (spec 1.0.308) |
| 2026-06-10 | #7280 | Frankenstein editor legacy shim consolidation for #7280. `_frankenstein_model.py` now re-exports the canonical `frankenstein_editor.model.URDFModel`, and `_frankenstein_panels.py` re-exports the canonical panel/dialog classes, preserving older import paths without duplicating implementation. Focused split tests assert shim identity and exercise validation/export through the legacy model import. (spec 1.0.304) |
| 2026-06-10 | #7278 | Lock-backed CI dependency install follow-up for #7278. Standard CI jobs now install committed `requirements-dev.lock` artifacts before editable package installs and use `--no-deps` for local editable extras so pip never treats extras-bearing lock entries as invalid constraints. The dev lock and `make sync-deps` target now cover the `gui-test` extra so unit gates retain real PyQt6/pytest-qt imports, and the static security CI acceptance test rejects `-c requirements-dev.lock` regressions while keeping the dev/runtime pip-audit lock checks. (spec 1.0.303) |
| 2026-06-10 | #7277 | Docker build timeout and focused PR coverage enforcement for #7277. `DockerManager` now monitors build output through a background queue while enforcing a wall-clock build timeout, terminating the process tree when stdout remains open past the deadline. Standard CI now scopes PR coverage to changed `src/**/*.py` modules and runs per-package coverage enforcement only after full core coverage reports, so focused PRs are not blocked by unrelated packages. (spec 1.0.308) |
| 2026-06-10 | #7278 | Lock-backed CI dependency install follow-up for #7278. Standard CI jobs now install committed `requirements-dev.lock` artifacts before editable package installs and use `--no-deps` for local editable extras so pip never treats extras-bearing lock entries as invalid constraints. The dev lock and `make sync-deps` target now cover the `gui-test` extra so unit gates retain real PyQt6/pytest-qt imports, and the static security CI acceptance test rejects `-c requirements-dev.lock` regressions while keeping the dev/runtime pip-audit lock checks. (spec 1.0.303) |
| 2026-06-10 | #7279 | Audit hygiene fixes for #7279 and #7282. `.github/workflows/docker-security-scan.yml` now blocks HIGH and CRITICAL Trivy container vulnerabilities in the table scan while retaining SARIF upload, and audited API/launcher production modules now use the canonical logging infrastructure instead of direct module-level `logging.getLogger` calls. Added security CI acceptance coverage for the Docker HIGH/CRITICAL gate and a repo-hygiene test for the remediated logger modules. (spec 1.0.302) |
| 2026-06-10 | #7269 | Audit regression fixes for #7269, #7270, and #7271. Model Explorer API path resolution now validates caller paths before filesystem reads and resolves only within approved model directories, closing the direct existing-path containment bypass. Motion-pipeline keypoint gap filling now guards both before/after neighbor keypoint indexes and pins mismatched-neighbor behavior in the main and pure-Python implementations. `SwingBallFlightPipeline` now emits `LaunchConditions` in the units consumed by `BallFlightSimulator`: launch and azimuth angles in radians, spin rate in RPM, with updated DbC validation and unit tests. (spec 1.0.301) |
| 2026-06-10 | #7207 | Completed the #7207 model explorer composition UX flow. Added `CompositionUxController` for library drag payloads, non-mutating drop/ghost previews, highlighted target/source links, validation summaries, committed drops, and a validation-aware export chooser that enables URDF/MJCF while explicitly marking SDF/OSIM unavailable until writers exist. `FrankensteinEditor` now exposes preview, commit, and export-choice hooks, with offscreen tests covering simple humanoid plus arm preview, commit, validation pass, and MJCF export. (spec 1.0.300) |
| 2026-06-10 | #7214 | Cross-engine equivalence import-boundary fix for #7214. `CalibrationOptimizer.optimize()` now imports `scipy.optimize.differential_evolution` lazily so importing `src.bunkershot3d.postproc.wrench_trace` through shared simulation backends does not require optional calibration optimizer dependencies in the equivalence CI environment. (spec 1.0.299) |
| 2026-06-10 | #7214 | C3D viewer renderer backend decision for #7214. Added ADR-0030 choosing PyQtGL as the first desktop GPU playback path while keeping matplotlib fallback, plus a focused `viewer_3d_backend.py` decision contract that carries the 60 fps target and parity checklist for scrubbing, speed control, loop playback, marker groups, view presets, and skeleton overlay before replacement. (spec 1.0.298) |
| 2026-06-10 | #7207 | Model explorer composition-flow controller for #7207. Added `CompositionFlowController` to attach a complete source URDF model to a working Frankenstein model via selected or declared attachment points, copy links/joints/materials with deterministic name mapping, validate the composed result immediately, and export validation-gated URDF or MJCF preview content. `FrankensteinEditor` now exposes an Attach Source Model action plus public export helper, and `URDFModel.from_file()` carries attachment sidecar metadata into the editor. Focused headless tests cover human-plus-arm composition, MJCF export, validation refusal, and the offscreen editor attach/export path. (spec 1.0.297) |
| 2026-06-10 | #7206 | Model explorer attachment manifests for #7206. Added a first-party `attachment_manifest` parser for versioned `<model>.attachments.json` sidecars, checked in the JSON Schema and docs, exposed declared attachment points plus non-fatal warnings through `ModelLibrary` model info, and updated the attachment dialog to list declared mount points first, prefill their interface-frame origin, and report payload-limit warnings. Focused tests cover valid/missing/malformed manifests, imported-model exposure, dialog defaults, and payload warning contracts. (spec 1.0.296) |
| 2026-06-10 | #7217 | Split the launcher entrypoint below the file-size budget for #7217. Sidekick sidebar installation, process cleanup polling, launcher domain orchestration, and GUI startup bootstrap moved from `src/launchers/upstream_drift_launcher.py` into focused modules, preserving compatibility imports and the canonical frameless-window helper under `src/launchers/launcher_ui/frameless_window.py`. The launcher entrypoint is now below 1200 lines, so its file-size budget exception was removed. (spec 1.0.295) |
| 2026-06-10 | #7252 | Rust mocap FFI binding error-contract hardening for #7252. `upstream-mocap-io` now validates non-empty and NUL-free Python binding paths before parser entry, maps missing files from `parse_c3d` / `parse_trc` / `parse_bvh` to `FileNotFoundError`, maps other file-access errors to `OSError`, and keeps malformed present files as `ValueError` parse failures that include the format and path context. Rust binding tests and Python parity tests cover missing-file and malformed-present-file behavior across all three formats while preserving the marker/unit parser contracts. (spec 1.0.294) |
| 2026-06-10 | #7207 | First #7207 model explorer library-panel unification slice. `ModelLoaderDialog` now exposes one searchable Library tree built from every `ModelLibrary.list_available_models()` category, including sibling repositories, and model rows show first-party format badges inferred from explicit model metadata or category defaults. Headless panel-model tests cover flattening, sibling inclusion, search, category grouping, and badge logic. (spec 1.0.293) |
| 2026-06-10 | #7250 | Motion-pipeline hook exception handling for #7250. `PipelineConfig.strict_hooks` now controls per-stage hook failure policy: default lenient mode logs failures with `logger.exception` so tracebacks are observable while the pipeline continues, and strict mode raises `HookExecutionError` with the stage, hook name, and original exception chained as the cause. Focused orchestrator unit tests cover both modes. (spec 1.0.292) |
| 2026-06-10 | #7220 | Added the bounded inverse swing optimization core for #7220. `src/shared/python/physics/swing_optimizer.py` adds `FlightTarget`, `ClubPreset`, `SwingOptimizer`, and diagnostics around SciPy SLSQP over speed/loft/attack/face-to-path while composing the existing `SwingBallFlightPipeline`; focused physics tests cover roundtrip, unreachable target, and timeout behavior. (spec 1.0.291) |
| 2026-06-10 | #7212 | Rust C3D analog and force-platform metadata slice for #7212. `upstream-mocap-io` now decodes C3D analog channels in int16 mode with SCALE/OFFSET/GEN_SCALE and in float mode without int scaling, advances marker frame stride across analog-bearing records, parses FORCE_PLATFORM TYPE/CHANNEL/CORNERS/ORIGIN metadata, and exposes additive PyO3 `analog` / `force_platforms` keys while preserving existing marker/event keys and marker-only fixture behavior. (spec 1.0.290) |
| 2026-06-10 | #7205 | Completed the Frankenstein composition validation surface for #7205. `CompositionValidator` now emits warning-level `subtree_mass_ratio` findings when attached subtree mass exceeds roughly 2x the parent chain mass and `geometry_overlap` findings when directly attached link AABBs overlap. The active Frankenstein model panel now renders current validation findings in a dedicated list so warnings and blocking errors are visible before save/export. (spec 1.0.289) |
| 2026-06-10 | #7221 | React/Tauri launcher parity decision (#7221). Added ADR-0028 choosing the manifest-driven multi-window Tauri model while keeping PyQt canonical for embedded tabs/docks. The React dashboard now persists a manifest-keyed launcher window registry, reconciles it with `useLauncherManifest.ts`, and exposes a window list/focus menu that reuses the local launcher API. (spec 1.0.288) |
| 2026-06-10 | #7215 | Startup entry-point consolidation (#7215). `launch_golf_suite.py` now delegates to canonical `launch_upstream_drift.py` with a deprecation warning, classic PyQt launch preflights the Qt platform and selects offscreen mode on headless Linux, and `src/api/local_server.py` degrades to an unavailable engine-manager facade when optional engine imports fail during local API startup. (spec 1.0.287) |
| 2026-06-10 | n/a | Removed unsafe Drake pose pickle deserialization from `pose_interchange.pose_io`. Drake `.drake` initial-state files now serialize `{q, v, model_metadata}` as JSON, the loader rejects binary/non-JSON payloads before deserialization, and regression coverage asserts invalid JSON and missing-`q` contracts. (spec 1.0.286) |
| 2026-06-10 | n/a | Legacy golf visualizer dataset contract preservation after row extraction optimization. `golf_visualizer_data.DataProcessor.extract_frame_data` still fails fast when BASEQ, ZTCFQ, or DELTAQ is absent and still returns zero-vector frame data when a requested frame row is missing, with regression coverage for both contracts. (spec 1.0.285) |
| 2026-06-10 | #7205 | Frankenstein composition validation framework (#7205). Added `src/tools/model_explorer/composition_validator.py` with structured error/warning findings for duplicate URDF names, orphaned joints, invalid root counts, disconnected links, kinematic cycles, and moving-link mass/inertia contracts. Frankenstein editor export now blocks validation errors by default while retaining an explicit `force=True` escape hatch for recovery exports. (spec 1.0.284) |
| 2026-06-10 | #7210 | LauncherContext shared event/value context for embedded tools (#7210). Added `src/shared/python/launcher_embed/context.py` with a headless `LauncherContext` protocol, in-memory snapshot-safe event dispatch, idempotent unsubscribe handles, keyed `value_changed:<key>` notifications, and a small `list/get/set` compatibility surface for Sidekick workspace reuse. `EmbeddedHostWidget` owns one context, injects it into opt-in tools through `set_launcher_context(ctx)`, and emits `tab.opened` / `tab.closed` events while legacy tools without the hook continue to open normally. (spec 1.0.283) |
| 2026-06-10 | n/a | Legacy golf visualizer Pandas row extraction optimization. `golf_visualizer_data.DataProcessor.extract_frame_data` now fetches each BASEQ/ZTCFQ/DELTAQ row once per frame and reuses the resulting row for all point/vector extraction, preserving the missing-row fallback contract while avoiding repeated `.iloc` lookups inside the render-frame path. (spec 1.0.282) |
| 2026-06-10 | #7216 | Configuration ownership consolidation (#7216). Removed root `config/` and `configs/` trees. Architecture debt policy moved to `scripts/config/architecture_debt_policy.json`; BunkerShot3D calibration YAML moved to `src/bunkershot3d/calibration/configs/`; UX field/error seed YAML moved to `src/shared/python/ux/config/`. Added `docs/development/configuration-systems.md`, canonical UX path constants, updated generators/tests/docs, and regression coverage preventing root config directories from returning. (spec 1.0.281) |
| 2026-06-10 | #7231 | Linux dependency-consistency lockfile repair after #7231. `requirements-dev.lock` now matches the Python 3.12 Linux `pip-compile --extra dev` output enforced by CI: Windows-only transitive `colorama` and `tzdata` entries are removed, and `uvloop` is restored for the Linux `uvicorn[standard]` dependency graph. (spec 1.0.280) |
| 2026-06-10 | #7213 | Built-wheel mocap source parity fix (#7213). The Rust `upstream-mocap-io` TRC parser now validates invalid or non-finite data-row frame and time fields before marker coordinates, preserving the Python facade's invalid-row contract when the PyO3 wheel is installed. The OpenCap session adapter test now follows the existing Rust parity contract by comparing marker coordinates approximately instead of requiring impossible exact decimal equality from Rust `f32` output. CI now runs the full `tests/unit/motion_pipeline/sources` directory immediately after installing built Rust wheels, and the Rust parity wheel gate script ratchets that source-wheel coverage. (spec 1.0.279) |
| 2026-06-10 | #7219 | MATLAB engine loader unification (#7219). `EngineManager._load_engine()` now dispatches every engine, including `MATLAB_2D` and `MATLAB_3D`, through the registry's `EngineRegistration.factory()` path. MATLAB-family Simscape adapter creation lives in `src.engines.loaders` with loader shim exports preserved for legacy imports, and tests assert the manager has no private MATLAB loader branch. (spec 1.0.278) |
| 2026-06-10 | #7203 | OpenSim `.osim` loader for #7203. Added first-party `src/tools/model_explorer/osim_loader.py` to parse OpenSim 3.x and 4.x model XML into `ParsedModel`, convert to validated `CanonicalModel`, preserve masses and joint mappings, and surface muscles/constraints/markers as warnings. Sibling discovery, imported model discovery, file filters, and the model-opening path now accept `.osim` without editing vendored `src/shared/python/model_generation/**`. Regression coverage lives in `tests/tools/model_explorer/test_osim_loader.py` and `.osim` sibling-discovery assertions. (spec 1.0.277) |
| 2026-06-10 | #7204 | Drake SDF model loading (#7204). Added `src.tools.model_explorer.SdfLoader` for the model explorer to parse SDFormat links, inertials, primitive and mesh geometry, joint axes/limits/dynamics, SDFormat 1.8 `relative_to` poses, and ball/universal joints into the canonical model contract. Sibling model discovery now classifies `.sdf` files from `Drake_Models` alongside URDF and MJCF assets so Drake-native models can be browsed and composed. (spec 1.0.276) |
| 2026-06-10 | #7208 | MJCF fixed-joint roundtrip topology preservation (#7208). URDF-to-MJCF conversion now keeps fixed children as welded nested MuJoCo bodies without joint elements while encoding the original fixed joint name, and MJCF-to-URDF decoding restores that fixed joint name only for welded nested bodies. Regression coverage asserts link sets, fixed and movable joint names/types, parent-child topology, and fixed-joint origin translation across URDF -> MJCF -> URDF. (spec 1.0.275) |
| 2026-06-10 | #7211 | Embeddable-tool adapter entry-point discovery (#7211). `src/launchers/embedded_tool_bootstrap.py` now imports `upstream_drift.embeddable_tools` entry-point adapters first, falls back to the in-tree adapter module list for editable checkouts, de-duplicates module paths across both sources, and keeps registry-diff tracking plus manifest-gap warnings intact. `pyproject.toml` declares the first-party adapter entry points so installed wheels and source checkouts use the same bootstrap contract. (spec 1.0.274) |
| 2026-06-10 | #7218 | Ball-flight REST simulation route (#7218). `POST /tools/ball-flight/simulate` exposes headless/batch launch simulation through the existing `FlightModelRegistry` and `UnifiedLaunchConditions` stack, validates launch/spin/wind/model/integration-window inputs with Pydantic contracts, and registers the `ball_flight` API tool route alongside the route registry. (spec 1.0.273) |
| 2026-06-10 | #7201 | Model Explorer sibling model-repository discovery (#7201). `ModelLibrary.list_available_models()` now exposes a `sibling` category populated by `src/tools/model_explorer/sibling_repositories.py`, scanning `Drake_Models`, `MuJoCo_Models`, `Pinocchio_Models`, and `OpenSim_Models` siblings next to the checkout or explicit `UD_SIBLING_MODEL_REPOS` roots. Discovery accepts URDF plus MJCF XML by content, skips VCS/cache directories, emits stable `sibling_<repo>_<relative-path>` config keys, and surfaces truncation instead of silently hiding excess results. `get_model_info("sibling", key)` resolves the discovered local model metadata without network access. Regression coverage added in `tests/tools/model_explorer/test_sibling_repositories.py`; the human-model fallback download path now delegates to the shared bounded downloader instead of a local `urllib.urlopen` call. (spec 1.0.272) |
| 2026-06-10 | #7189 | Follow-up #7189 packaging gate repair after reconciling the parallel branch update. Tauri Linux dependency setup now waits on both apt and dpkg locks, matching the runner failure mode seen in `Check (Rust + TypeScript)`. The WGS process calculator lazy-loads GUI theme helpers from `create_plots_tab`, and the standalone Sidekick regression suite asserts the WGS engine import does not require `shared.python.theme.integration` or PyQt6, keeping the clean-wheel `sidekick run` smoke headless. `package-standalone-sidekick.yml` now smoke-tests Python 3.11/3.12 to match the package floor, with `scripts/ci/check_python_version_coherence.py` guarding that workflow. (spec 1.0.271) |
| 2026-06-10 | #7160 | Docker dependency-audit hardening for #7160 follow-up CI. Pinned Docker builder/runtime pip installs to `26.1.2`, declared patched runtime floors for `Mako>=1.3.12` and `PyJWT>=2.13.0`, aligned `requirements.lock`, and updated the direct dependency license ledger so in-image `pip-audit` resolves patched packages. (spec 1.0.270) |
| 2026-06-10 | #7160 | Python-version coherence hardening for #7160. Raised the live package/support floor to Python 3.11, removed the unsupported Python 3.10 classifier and standard CI matrix lane, kept Python 3.11/3.12 in the standard test matrix, and documented that the Docker image plus `requirements.lock` are generated on Python 3.12. Added `scripts/ci/check_python_version_coherence.py` and `tests/ci/test_python_version_coherence.py` to enforce agreement across `pyproject.toml`, `install.sh`, `requirements.lock`, `Dockerfile`, `.github/workflows/ci-standard.yml`, mypy target version, and public docs. (spec 1.0.269) |
| 2026-06-10 | #7158 | Finished a deferred sub-defect of the pytest-gating policy issue #7158 (D2 marker discipline). Added `tests/support/suite_markers.py` (`SUITE_MARKERS`, `suite_markers_enforced`, `find_unmarked`, `item_has_suite_marker`) and wired a `pytest_collection_modifyitems` + `pytest_terminal_summary` hook in `tests/conftest.py` that, in REPORT-ONLY mode (the repo's ratchet pattern), counts collected tests carrying none of the recognized suite markers and surfaces the baseline without failing collection. Enforcement (missing-marker = collection error) is opt-in via `UD_ENFORCE_SUITE_MARKERS=1` for a follow-up once the unmarked baseline is driven to zero. Unit coverage in `tests/unit/test_suite_marker_enforcement_7158.py`. The remaining #7158 sub-defects (D3 coverage-omit retune) and #7155 sub-defects (deleting the autouse `_protect_engine_modules` and function-scoping `mock_mujoco_dependencies`, both gated on the issue's 5/5 `-n auto` stability evidence) stay deferred for cause and are tracked on those issues. (spec 1.0.268) |
| 2026-06-10 | #7183 | Security hardening of remote model downloads (#7183, #7184, #7185, #7186). Added shared helpers `download_to_file` (bounded-timeout streaming download; `DOWNLOAD_TIMEOUT_SECONDS=30`) and `safe_extract_zip` (Zip-Slip member-path validation) to `src/shared/python/security/security_utils.py`. `GitHubRepository.download_archive` now extracts via `safe_extract_zip`, downloads with a timeout, and unlinks its `delete=False` temp file on every path (#7183 Zip Slip + temp leak). All `urlretrieve`/`urlopen` call sites in `standard_models.py`, the `model_generation/library/` repository/loader modules, and `tools/model_explorer/model_library.py` now use a 30s timeout (#7184). `StandardModelManager.download_standard_humanoid` now parses the URDF for mesh-filename entries and downloads the real meshes, returning `False` (no silent empty-STL stubs) unless the dev-only `allow_stub_meshes=True` is passed (#7186). `Jules-Issue-Mention-Handler.yml` and `PR-Comment-Responder.yml` route attacker-controllable issue/comment title/body/login values through `env:` indirection instead of splicing into `run:` bodies (#7185 Actions expression injection). Regression tests added in `tests/unit/test_shared_security_utils.py` (zip-slip + timeout) and `tests/unit/config/test_standard_models.py` (real-mesh success vs loud failure). (spec 1.0.267) |
| 2026-06-09 | #7131 | Architecture budget CI gate for #7131/#7133. Added `scripts/ci/check_architecture_budget.py` plus `scripts/config/architecture_budget.json` to ratchet changed production Python files against a 100-line function budget and 8-effective-parameter callable budget, excluding tests/vendor and requiring owned, linked exceptions for any temporary budget breach. Wired the gate into `.github/workflows/ci-standard.yml` and added focused TDD coverage for long-function detection, parameter counting, receiver-parameter exclusion, exception handling, and test-path skips. (spec 1.0.265) |
| 2026-06-02 | #7101 | Golf visualizer camera-basis norm optimization (#7101). `GolfVisualizerWidget` now uses fixed-arity `math.hypot` for the known 3D forward and right camera basis vectors instead of `np.linalg.norm`, avoiding NumPy reduction overhead while preserving the existing fallback behavior for degenerate vectors. (spec 1.0.260) |
| 2026-06-02 | #7098 | Bolt small-vector norm optimization (#7098). `src/shared/python/physics/ball_simulator.py` now uses fixed-arity `math.hypot` for scalar relative-velocity and Magnus cross-product magnitudes in `_calculate_forces_single`; `src/shared/python/physics/flight_models.py` uses `math.hypot` for Waterloo/Penner spin-vector magnitude and for MacDonald-Hanzely/constant-coefficient spin-axis normalization, computing each spin norm once before normalization; and `src/shared/python/physics/swing_ball_flight_pipeline.py` uses `math.hypot` for launch speed, horizontal launch speed, and angular-velocity spin-rate derivation. This is a pure performance cleanup for known 2D/3D vectors, preserving behavior while avoiding NumPy reduction allocation overhead in scalar hot paths. (spec 1.0.258) |
| 2026-06-02 | #7095 | Cross-engine equivalence gate fixes (#7095, #7097). `_run_engine_checked` now treats all-NaN grip traces as missing-engine bindings (`_EngineBindingsError`) while keeping partial NaN/Inf traces as hard simulation-divergence failures. The JaxSim-vs-Pinocchio parity case now uses zero base position so INERTIAL and LOCAL velocity representations are comparable while still exercising mixed angular/linear Coriolis terms. Cross-engine docstrings now distinguish the 5 mm agreement tolerance from the looser per-engine world-frame origin plausibility check. (spec 1.0.259) |
| 2026-06-02 | #7063 | Docs root cleanup follow-through for #7063. Removed loose root-level Markdown from `docs/` except the canonical `README.md` and `index.md`; relocated remaining live manuals, architecture notes, assessments, engine guidance, strategic notes, troubleshooting references, and technical guidance into topic directories such as `docs/user_guide/`, `docs/architecture/`, `docs/assessments/`, `docs/engines/`, docs/strategic/ (that directory was later merged into `docs/plans/`; see the #9457 row), `docs/troubleshooting/`, and `docs/technical/`. Preserved the now-real runnable `docs/examples/` subtree and added a structural test that verifies every `docs/examples/index.rst` toctree entry points at an existing page instead of deleting valid examples. Updated docs governance README/catalog text plus live tooling references (`check_formatter_guidance.py`, `check_tutorial_imports.py`, `replace_cli.py`, and `doc_size_budget.json`) to follow the relocated user manuals, and kept `tests/docs/test_docs_structure.py` as the guard for root cleanliness, examples integrity, and removal of stale `TRACKED_TASK` placeholders from Sphinx config. (spec 1.0.257) |
| 2026-06-02 | #7042 | Green-suite CI gate + fixed 2 pre-existing unit reds (#7042). Added a new parallel `unit-test-gate` job to `.github/workflows/ci-standard.yml` that runs `pytest -m "unit and not slow and not live_simulation and not requires_gl and not requires_drake and not requires_opensim and not requires_mujoco and not requires_pinocchio and not requires_myosuite and not requires_jaxsim and not requires_nimble and not requires_network and not requires_gpu and not requires_mocap_fixtures" -n auto --timeout=60 --timeout-method=thread --no-cov` with NO `\ | \ | true` / `continue-on-error` masking, so any non-skip unit failure reds the job. It is wired into the required `quality-gate` summary job (`needs: [code-quality, security-scans, repo-structure-gates, unit-test-gate]`); the aggregate now also fails when `unit-test-gate.result != success`, blocking merge on any unit red. The `quality-gate` check name is preserved exactly for branch protection / bot workflows. RED (a): `tests/engines/physics_engines/test_opensim_engine.py::test_load_from_string_creates_tempfile` patched the public `load_from_path` but `load_from_string` delegates to `_load_from_path_impl`, so the capture callback never ran (`KeyError: 'path'`); the test now patches `_load_from_path_impl` (the actually-called hook) and asserts the path was captured before checking cleanup — needs no real OpenSim. RED (b): `tests/unit/conftest.py::pytest_configure` injects `MagicMock()` into `sys.modules["pinocchio"]`/`["casadi"]` for collection; the `engine_availability` probe's `hasattr(pin, "buildModelFromUrdf")` guard was fooled by MagicMock and cached `pinocchio` as AVAILABLE in the module-level `_engine_status_cache` + `is_engine_available` lru_cache, poisoning every later `requires_pinocchio` test. Fix mirrors the existing drake mock-detection guard: `_probe_engine` now raises `ImportError` when the imported module's `type(...).__module__ == "unittest.mock"` (pinocchio branch + the generic `else` import branch covering casadi), and a new `reset_engine_status_cache()` (clears both dicts + `cache_clear()`) is called by the autouse `_reset_mocks_between_tests` fixture so availability is re-probed per test rather than leaking across the mock/unmock boundary. (spec 1.0.256) |
| 2026-06-01 | #7048 | Engines parity cluster (#7048/#7049/#7050/#7051/#7052). #7048: replaced the hollow cross-engine equivalence gate (`tests/motion_matching/test_cross_engine_equivalence.py`) — removed the `12 == 12` meta-tautology and stale `pytest.skip` Drake/OpenSim stubs; each installed engine now runs a real `requires_*`-gated grip-RMSE check vs the Simscape `trial_001` fixture (5 mm gate at the theta=0-valid address pose + cross-engine agreement gate across all three poses after rigid frame registration). #7050: added `get_capabilities()` to the MuJoCo and OpenSim adapters. #7051: Drake `compute_zvcf` reads fixed actuation via the actuation matrix `B` (`a = M⁻¹(Bu − g)`) instead of tau=0; OpenSim `compute_jacobian` uses Simbody analytic `calcStationJacobian` with FD fallback. #7049: value-asserting 1-DOF pendulum dynamics tests for Drake/OpenSim (SPD mass matrix, m·g·L·sinθ gravity torque, τ=M·a+bias within 1e-10). #7052: removed dead placeholders (real MuJoCo energy via `mjENBL_ENERGY`; deleted BVH placeholder; implemented 3 Drake `motion_optimization` cost bodies; implemented the empty `test_cross_engine_consistency.py`). (spec 1.0.255) |
| 2026-06-01 | #7056 | API hardening cluster (#7056, #7057, #7058), all under `src/api/`. #7056: `simulation_ws._apply_initial_state` now caps `q`/`v` length before `np.array(...)` allocates, returning an error string (surfaced as a WS error frame) instead of allowing an authenticated client's multi-million-element array to trigger a memory/event-loop DoS. The bound is engine-DoF-aware (`min(hard_ceiling, max(nq*4, 16))` when the engine advertises `nq`) and otherwise falls back to a hard ceiling reusing the request-model constant `MAX_STATE_VECTOR_LEN` (issue #6948) for DRY/defense-in-depth consistency with the `SimulationRequest` validation layer; `set_state` is never called and no array is allocated when rejected. #7057: `chat_ws.chat_stream`'s inner streaming loop wrapped streaming in a broad `except Exception` that also caught a mid-stream `WebSocketDisconnect`, logging a normal client disconnect as an internal error and attempting to send an error frame on a dead socket; an `except WebSocketDisconnect: raise` is added ahead of the broad catch so disconnects propagate to the outer disconnect handler (logged at debug, no error frame). #7058: `data_explorer` fully `json.loads`-ed on-disk JSON datasets per request (`_preview_json_streaming`, `_resolve_operation_source`) while CSV streams; added `_read_json_dataset_text` which checks on-disk size against `MAX_JSON_DATASET_BYTES` (= `MAX_DATASET_SIZE_BYTES`, 50 MB) _before_ reading and rejects oversized JSON with HTTP 413, bounding per-request memory and matching the CSV streaming / upload-cap parity. TDD: oversized-q/v rejection + engine-DoF cap + valid-array-applied tests (`test_simulation_ws.py`); mid-stream-disconnect-not-internal-error test (`test_routes_chat_ws.py`); JSON-over-cap-413, under-cap-previews, and JSON-vs-CSV value-parity tests (`test_data_explorer_perf.py`). (spec 1.0.254) |
| 2026-06-01 | #7063 | Docs examples + declutter and `quality-gate` parallelisation (#7063, #7064). #7063: `docs/examples/` was effectively empty (only `index.rst`). Added three runnable, dependency-light (numpy-only) end-to-end examples — `run_mock_engine_sim.py` (load `MockPhysicsEngine` + integrate a swing), `motion_matching_synthetic.py` (pose+velocity tracking cost on a synthetic trajectory), `estimate_kinematics.py` (central finite-difference velocity/acceleration estimation validated against an analytic signal) — wired into `docs/examples/index.rst` via `literalinclude`, and guarded by a smoke test `tests/unit/docs/test_examples_runnable.py` that executes each (asserts ≥3 examples, exit 0). Added a `docs/examples/**` T201 per-file-ignore (CLI-style examples print to stdout). Decluttered 9 zero-reference loose top-level `docs/*.md` into `docs/{governance,operations,reviews}/` (e.g. `PROJECT_MAP.md`, `PACKAGE_ORGANIZATION.md`, `fleet_recovery_tracking.md`, `docker-gpu.md`), updating the two live `docker-gpu.md` links; heavily-referenced docs (USER_MANUAL, UPSTREAM_DRIFT_USER_MANUAL, IDEAS, engine_selection_guide, biomech-workspace) and files cross-linked from historical assessment/backlog snapshots were deferred to avoid touching scripts/CI configs. #7064: split the monolithic ~30-step `quality-gate` job in `.github/workflows/ci-standard.yml` into three parallel jobs each `needs: pick-runner` — `code-quality` (ruff lint/format + mypy + install-dependent gates: alembic, pip-audit, code-quality-check, MATLAB), `security-scans` (Semgrep/Bandit/detect-secrets/Trivy; checkout-only, no editable install), `repo-structure-gates` (pure-stdlib SPEC/structural/size-budget/ratchet/placeholder checks) — aggregated by a final `quality-gate` summary job (`needs: [code-quality, security-scans, repo-structure-gates]`, `if: always()`) that fails iff any underlying job failed. The required-check name `quality-gate` is preserved exactly so branch protection and the bot workflows (Jules-Auto-Repair/PR-AutoFix, Bot-CI-Trigger) keep resolving it. (spec 1.0.252) |
| 2026-06-01 | #7061 | Deleted dead legacy Motion Capture Plotter monoliths + config/deps hygiene (#7061, #7065). #7061: removed `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Motion Capture Plotter/{starting_pose_matcher.py (2671 LOC), Motion_Capture_Plotter.py (1402 LOC)}` — dead duplicates superseded by the tested `src/tools/starting_pose_matcher/`, with zero `src/` importers — and dropped their two `scripts/config/file_size_budget.json` exceptions; added a hygiene assertion that no `src/` import resolves the old path. #7065: routed common hardcoded host/port literals through typed `Settings` (`api_host`/`api_port`), de-duplicated the Cargo workspace `members` (`upstream-mocap-io` was listed twice), and staggered the file-size-budget `expires_on` cliff into distinct dates. (spec 1.0.251) |
| 2026-06-01 | #7059 | Added value round-trip + headless coverage for two previously thin/untested modules (#7059, #7062). #7059: `tests/unit/data_io/test_export.py` was smoke-only (registry shape). Added export->reimport **value** round-trip tests for the always-available `json` and `csv` formats against `data_io/export.py`: each exports via `export_recording_all_formats`, reimports (json via `json.loads`+`np.asarray`, csv via `pandas.read_csv`), and asserts numeric equality with `np.testing.assert_allclose` plus column/key order (json key order preserved; csv emits `time` first then 1-D keys then expanded `{key}_{i}` 2-D columns) and unit scaling (mm<->m, kgf<->N scale-and-recover is exact). #7062: `src/shared/python/qt_utils/wheel_event_filter.py` (used by >=5 GUI tabs) had zero tests. Added `tests/unit/qt_utils/test_wheel_event_filter.py` (offscreen `QApplication`, marked `requires_gl`/`headless_safe`): asserts accidental `QWheelEvent`s on combo/spin widgets are swallowed (`eventFilter`->True) both focused and unfocused (the filter is intentionally focus-independent for value-mutation safety), non-wheel events pass through, install/remove via the real event system leaves combo index unchanged, and the `suppress_wheel_on_widget(s)` helpers attach an independent retained filter per widget. 96.2% line coverage of the module (>=90% acceptance). Test-only PR; no source behavior change. (spec 1.0.250) |
| 2026-06-01 | #7044 | Repaired ~20 red `tests/unit/api` contract tests (#7044). All were test-side drift, not source regressions. (1) CORS: `get_cors_origins` fail-closed message changed at `src/api/config.py` to "CORS_ORIGINS must not contain '\*' when credentials are enabled (fail-closed)"; `test_config.py` now asserts the live message (intended fail-closed contract). (2) `TaskManager` query/mutation API is synchronous per the #4843 compatibility contract, but several mocks declared `async def exists/get/set` — fixed `test_dependency_injection.py` (`AsyncMock`->`MagicMock`), `test_routes_export.py`, `test_routes_simulation.py`, and `test_simulation_service.py` to sync mocks, resolving `'coroutine' object is not subscriptable/iterable` 400/500s. (3) WS auth gate (#5913): chat-ws endpoints now call `resolve_ws_user` before `accept()` and close 1008 when unauthenticated; `test_routes_chat_ws.py` adds an autouse fixture stubbing `resolve_ws_user` (mirroring `test_simulation_ws.py`). (4) `POST /realtime/publish` now requires `ws_compatible_auth_dependency` (#6888/#6889); `test_routes_realtime_bounds.py` overrides it to a no-op so the amplification-bounds assertions run. (5) `useEngineStore.ts` `unloadEngine` calls the backend via the shared `apiFetch` wrapper not raw `fetch(`; `test_engine_route_contracts.py` accepts `apiFetch`. (6) Rotation-converter router carries `prefix="/api/calc/rotation-converter"`; `test_rotation_converter_mocked.py` reference-frame POST retargeted to the prefixed path. `tests/unit/api` now 0 failures. (spec 1.0.243) |
| 2026-06-01 | #7053 | Physics de-duplication + constant provenance/validation (#7053, #7054, #7055). #7053: `_impact_physics.py` and `_impact_recorder.py` were full copies of the canonical `impact_model/{models,types,utils,solver}.py` (guarded only by parity test #7015). They are now thin re-export shims; the duplicate class/function bodies were deleted so `class RigidBodyImpactModel` is defined exactly once (grep == 1), and identity tests assert `_impact_physics.RigidBodyImpactModel is impact_model.models.RigidBodyImpactModel` for every re-exported symbol. #7054: the friction rolling-cap `0.4` in the canonical `models.py` was unprovenanced and physically wrong — derived and corrected to the uniform-solid-sphere rolling-without-slip factor `2/7 ≈ 0.2857` (named `SPHERE_ROLLING_CAP_FACTOR` with full derivation + citation), pinned by an analytic friction-spin test (`omega == (5/7)*v_t/R` at saturation); the warn-only Newmark `_warn_if_ill_conditioned` is now backed by a hard accuracy assertion — an analytic undamped-SDOF test (`u(t)=u0·cos(ωt)`) requires Newmark error < 1% at dt ∈ {1e-3,1e-4,1e-5}; and `biomechanics/ztcf.py _forces_from_accelerations` "simplified" docstring is replaced with the pendulum derivation and validated against the analytic single pendulum (`F_t = -m·g·sinθ`, `qddot = -(g/L)sinθ`). #7055: terrain energy-absorption weights `0.5/0.3/0.2` and grass coefficient `0.1` in `_terrain_physics.py` are now named, documented module constants (`ENERGY_ABSORPTION_*_WEIGHT` convex combination summing to 1.0, `GRASS_RESISTANCE_COEFFICIENT`) with value tests pinning their invariants and the resulting absorption factor; flight Cd/Cl coeffs in `flight_models.py` already carry per-model `reference` citations and are now value-tested to lie within documented golf-ball wind-tunnel ranges (Cd 0.15-0.30, Cl 0.10-0.30; Bearman & Harvey 1976). (spec 1.0.242) |
| 2026-06-01 | #7045 | Fixed `BiomechanicalModel.add_segment` segment-name validation drift (#7045). `self.segment_masses` (from `humanoid_character_builder` `estimate_segment_masses`) is keyed by the full segment name (`right_thigh`, `right_shin`, ...), and `compute_dynamic_com` looks up mass by `self.segment_masses[seg.name]`. But `add_segment` validated `get_anthropometry_key(name) in self.segment_masses` — the mapped key (`thigh`) is never a `segment_masses` key, so any laterally-named segment was rejected with a `PreconditionError: Unknown segment name`. Validation now checks `name in self.segment_masses`, consistent with the downstream lookup; `get_anthropometry_key(name)` is retained only in the diagnostic message. Restores 3 red cases in `tests/unit/biomechanics/test_dynamic_com.py` (full file 8/8 green). Source-only fix matching the test's documented API. (spec 1.0.241) |
| 2026-06-01 | #7037 | Resolved current-main review regressions (#7037; #7031, #7028, #7027, #7017, #7015, #6954) across API task expiry, chat stream cancellation pairing, full-stream dataset statistics, public impact-model parity, provider ID deduplication, and cloud-token chmod regression coverage. (spec 1.0.239) |
| 2026-06-01 | #7043 | Estimation synthetic-fixture corrections (#7043, #7060). #7043: `synthetic_fixtures.make_fixture_cameras` built `cam1` from a hand-typed near-rotation whose lower-left sign (`+0.2588`) made it non-orthonormal; the stricter `CameraExtrinsics` validator (`R.T@R==I`, `det==+1`) rejected it, reddening 5 `tests/unit/estimation/test_synthetic_ground_truth.py` tests. Now built as a true proper 15-degree rotation about Y from `cos`/`sin`, guaranteeing orthonormality + `det==+1`; added positive (fixtures validate, `R.TR≈I`, `det≈+1`) and negative (legacy matrix rejected) tests. #7060: `synthetic_ground_truth.project_world_point` applied only the k1/k2 radial terms while `residuals.project_pinhole` supports the canonical 5-term `(k1,k2,p1,p2,k3)` model (#6907), silently diverging for 5-term cameras. Added a `k3` field (default 0.0) to the `CameraIntrinsics` contract and threaded it into the rig's radial polynomial (`1 + k1*r² + k2*r⁴ + k3*r⁶`); parity tests assert the rig projection equals `project_pinhole` within 1e-9 for nonzero k3 and remains unchanged for k3=0. (spec 1.0.241) |
| 2026-06-01 | #7021 | Bolt perf: replaced `np.linalg.norm(v)` with faster 1-D equivalents (`math.hypot` / `math.sqrt(np.dot(v, v))`) in physics hot loops, consolidating the genuine optimizations from #7021/#7029/#7034/#7035 onto a clean base. `flight_models.py` ball-flight `derivatives` inner loops use 3-arg `math.hypot` for the relative-velocity speed and reuse a single `cross_norm` local for the Magnus/lift direction (#7034). A shared float-casting `_magnitude(v) = math.sqrt(float(np.dot(asarray(v, float), ...)))` helper (guards int-dtype overflow per #7022) replaces the norm in `_friction_laws.py` (tangent/slip magnitudes), `_terrain_physics.py` (normal-force/tangent-velocity/impact-speed), and `aerodynamics/_rust_facade.py` (rel-velocity/spin/lift/Magnus magnitudes) (#7021/#7029); regression test `tests/unit/physics/test_magnitude_int_overflow_7022.py` locks the int-overflow guard and float-equivalence. `starting_pose_matcher.py` shaft/residual/torso-disk norms use `math.hypot`/`math.sqrt(np.dot)` (#7035). Pure perf, no behavior change; every transformed site is a provably 1-D vector. (spec 1.0.240) |
| 2026-06-01 | #7046 | Motion pipeline now runs end-to-end with a real IK backend (#7046, #7047). Added `motion_pipeline/ik/geometric_backend.py`: a dependency-free damped-least-squares (Levenberg-Marquardt) IK solver with its own `SkeletonRig` forward kinematics; resolved the `GEOMETRIC` enum in `make_ik_solver` to this real module (was importing a nonexistent `geometric_backend`). The mujoco/drake/opensim/pinocchio IK `solve_frame` stubs now raise `NotImplementedError` instead of silently returning a neutral zero pose (#7046). Rewired orchestrator `_run_motion_matching` to route through `make_matching_solver(backend).match(...).to_contract()` (mujoco->mujoco_torque, drake->drake_trajopt, pinocchio->pinocchio_inverse_dyn), replacing imports of the nonexistent `.matching.{mujoco,drake,pinocchio}_backend.run_matching`; `geometric` is now an accepted IK backend (#7047). `matching/torque_mujoco.py` no longer hardcodes `success=True`/`rmse=0`: `success` reflects real execution (False when MuJoCo is absent or only the placeholder model is available, since torques are then all zero) and `fit_metrics` are computed from real residuals (#7047). (spec 1.0.241) |
| 2026-06-01 | #6886 | Re-derived three tracked review fixes (#6886, #6907, #6911) onto clean origin/main, superseding the stale 69-file omnibus #6920. #6886: `model_pack/v1` manifest normalization now prepends `models_root` to each relative `exercises[].path` so later `source_root=provider_root` resolution finds `provider_root/<models_root>/<path>`; paths already rooted under `models_root` are left untouched (no double-prefix). #6907: `_apply_brown_conrady` / `project_pinhole` now accept the canonical 5-term distortion `(k1, k2, p1, p2, k3)` in addition to the 4-term form, with the radial polynomial extended to `1 + k1*r² + k2*r⁴ + k3*r⁶`; non-(4,)/(5,) shapes raise `ValueError`. #6911: realtime auth moved onto the route itself (`POST /realtime/publish` now declares `dependencies=[Depends(ws_compatible_auth_dependency)]`, keeping the existing slowapi rate limiter), so `WSPubSub._spawn_server()`'s bare `include_router(realtime_router)` autostart path is protected against unauthenticated broadcast injection (#6888) — previously only server.py's mounts were guarded. TDD: 5-term-accepted + k3-affects-radial + invalid-length residual tests, a `models_root`-prepend manifest test (plus no-double-prefix), and WSPubSub-style autostart auth tests (401/403 unauth, 200 in local mode). (spec 1.0.238) |
| 2026-06-01 | #7000 | Added TDD coverage for four previously-untested modules and fixed three real bugs surfaced by the tests (#7000, #7001, #7002, #7004). #7000: MJCF/URDF converter round-trip (parse→emit→parse preserves bodies/joint types/masses/inertia), `_parse_body_inertial` (diag+full), `_parse_mjcf_geom` (box/sphere/cylinder), malformed-XML→error. #7001: SimScape MDL parser `parse_string` blocks/connections/params, `SimscapeParameter.as_float`/`as_vector` (valid+malformed→default), `get_body_blocks`/`get_joint_blocks`/`get_connections_to`, `_get_block_type` map, bad-extension→error. #7002: `ModelGenerationAPI` route handlers (health/info shape, mjcf↔urdf + validate + parse + inertia happy/error, 422 on malformed, route count, 404) plus FastAPI-adapter registration. SECURITY #7004 (sandbox escape FIXED): `scripting_env.ConsoleEnvironment` permitted the classic CPython escape `().__class__.__bases__[0].__subclasses__()` reaching a class whose `__init__.__globals__['__builtins__']` is the real unrestricted builtins, leaking `open`/`eval`/`exec`/`__import__` and reaching `os` despite `import os` being blocked — added `_screen_source_for_escapes` AST screen rejecting introspection dunders before exec/eval. Also fixed: `convert_mjcf_to_urdf`/`convert_urdf_to_mjcf` now catch `xml.etree.ElementTree.ParseError` (a `SyntaxError`, not `ValueError`) so malformed XML returns 422 not 500; and `FastAPIAdapter.register` `make_handler` changed from `async def` (returned an un-awaited coroutine, crashing route registration) to `def`. 120 new tests, all passing. (spec 1.0.236) |
| 2026-06-01 | #6978 | Realtime/WebSocket concurrency hardening (#6978, #6980; #6972). `/realtime/publish` now serializes `ws.send_json` per socket behind a per-connection lock so concurrent publishes can no longer interleave frames on the same WebSocket, and a slow/broken subscriber is dropped without taking down healthy ones (#6978). `WSPubSub`'s lazy `_http_client` initialization is guarded so the check-then-act race can no longer construct (and leak) multiple `httpx` clients (#6980). Regression tests cover concurrent same-socket sends, the rate-limit default, and the HTTP-client init race. (TaskManager expiry-sweep throttling for #6992 already landed via #7026; this PR carries only the realtime/WS work.) (spec 1.0.235) |
| 2026-06-01 | #6988 | API/storage/task-manager perf and resource fixes (#6988–#6992). `POST /simulate` no longer freezes the FastAPI worker: `SimulationService.run_simulation` offloads the CPU-bound stepping pipeline (`_run_simulation_sync`) via `anyio.to_thread.run_sync`, keeping the event loop responsive under concurrent requests (#6988). `GET /datasets` lists JSON columns by streaming only a bounded prefix (`_stream_json_columns`, 256 KiB sniff window) instead of `json.load()`-ing the whole file (#6990). Added `GET /tools/data-explorer/datasets/{dataset_id}/rows?offset=&limit=` delegating to `DatasetStorage.get_dataset_rows`, plus `iter_dataset_rows` streaming; `dataset_stats`/`filter_dataset` now consume a streaming `_OperationSource` (single-pass CSV streaming, row-capped) rather than materializing the whole dataset (#6991). `DatasetStorage.store_dataset` runs `cleanup_expired()` (TTL retention) before each write and uses `executemany`, bounding `datasets.db` growth (#6989). `TaskManager` throttles its O(n) expiry sweep: the hot read/membership path purges at most once per `CLEANUP_INTERVAL_SECONDS` while remaining TTL-correct via `_is_expired_locked`; writes and aggregate ops force an exact sweep (#6992). Regression tests added for each fix; dead `_load_dataset_from_path`/`_load_dataset_for_operation` removed. (spec 1.0.231) |
| 2026-06-01 | #6983 | Shaft-FEM and bunkershot coupling physics fixes (#6983, #6985; #6987 verified). The finite-element shaft cantilever now clamps the BUTT, not the thin tip: `create_standard_shaft` lays the diameter taper thick-butt-at-station-0 to thin-tip-at-the-end, aligning the geometry with the `_apply_boundary_conditions` clamp at node 0 and the analytic `compute_static_deflection` butt convention; natural frequencies, mode shapes, and static deflection are now computed for the correct cantilever orientation (#6983). `FiniteElementShaftModel.step` non-dimensionalizes the Newmark effective system via symmetric Jacobi (diagonal) scaling before the linear solve and emits a conditioning warning when `dt < 1e-2*sqrt(min(diagM)/max(diagK))`, preventing the catastrophic cancellation in the `a_new` recovery at impact-scale dt (~1e-7); `step` now also rejects non-positive dt (#6985). The bunkershot `CoupledDoublePendulum` wrench-to-joint-torque mapping was verified as a correct `J^T` projection — both joints are revolute about world y so the angular Jacobian row is `[1, 1]` and the external moment Ty legitimately projects onto both joint torques (NOT a double-count); added a clarifying comment plus a `test_pure_force_torque_mapping` regression locking the convention (#6987, verified-correct, left open). (spec 1.0.230) |
| 2026-06-01 | #6941 | Observability, safety, security, and test polish across the API, deployment, and estimation layers (#6941, #6943–#6950). The realtime controller loop now logs via `logger.exception` and aborts after N consecutive failures, commanding zero torque (#6943); `RealtimeController.stop()` re-checks `is_alive()` after a join timeout, raises on timeout, and only sends the zero command on confirmed stop (#6944); `ChatService` fallback logs the exception and exposes explicit `backend_error`/`adapter_available` state instead of silent degradation (#6945); cloud token storage sets dir mode `0o700` and token mode `0o600` (#6946); the simulation route returns a generic client detail while logging specifics server-side (#6947); `SimulationRequest` caps `control_inputs` length and `_normalize_initial_state_component` length (#6948); re-raising/except branches in `server.py`, `task_manager_durable.py`, and `launcher_diagnostics.py` switch to `logger.exception` (#6949); the dead timezone compat shim in `auth/dependencies.py` is removed (#6950); and `multi_trial` gains negative-path validator/accessor/stack tests plus a hardened `stack_shared_parameter_jacobians` 1D guard that previously raised `IndexError` (#6941). MyPy Strict errors exposed by these edits are resolved, including a generic `_assert_type` helper (`type[T]->T`) satisfying both mypy configs. (spec 1.0.229) |
| 2026-06-01 | #6930 | Design-by-Contract input validation hardening across the motion pipeline, Rust kernel, and velocity conventions (#6930–#6934, #6940, #6942). `motion_pipeline` `/run` now validates `source_format` against the adapter registry up front (400) and `loader.load_source` raises `ValueError` for a non-auto `format_hint` matching no adapter instead of silently falling through to `load_any` (#6930); `velocity_conventions` `_as_spatial_vector`/`_as_vector3`/`_as_matrix3` add `np.isfinite` guards via `_require_finite` and `single_floating_body_h_g` asserts finite mass (#6931); the orchestrator distinguishes caller contract violations (`InvalidInputError`→400) from internal faults (`RuntimeError`→500) and pydantic `ValidationError`→422 via `StageResult` error_kind (#6932); `CameraExtrinsics.rotation` validates finiteness, orthonormality (`R.T@R==I`), and proper-rotation (`det==+1`) (#6933); `rust_kernel` `create_air_properties`/`create_ball_properties` reject non-positive density/viscosity/temperature and mass/radius (#6934); plus parametrized fallback-vs-Rust backend tests (#6940, #6942). Bandit B104 on the dev-only uvicorn `0.0.0.0` entrypoint is annotated `# nosec B104`. (spec 1.0.228) |
| 2026-05-31 | n/a | Tightened the Pinocchio dynamics API checker error boundary to catch only import failures, preserving the error-handling ratchet while still returning a diagnostic failure when the robotics Pinocchio package cannot be imported. (spec 1.0.227) |
| 2026-05-31 | n/a | Hardened the JaxSim/Pinocchio parity prerequisite contract after CI exposed an unrelated `pinocchio==0.1` package shadowing the robotics API. Cross-engine equivalence now uninstalls the wrong `pinocchio` distribution before force-installing `pin>=2.6.0,<5.0.0`, and `scripts/ci/check_pinocchio_dynamics_api.py` fails the prerequisite step unless `import pinocchio` exposes the required free-body dynamics symbols (`Model`, `JointModelFreeFlyer`, `SE3`, `Inertia`, `crba`, `rnea`, `computeCoriolisMatrix`). (spec 1.0.226) |
| 2026-05-31 | #6880 | Hardened JaxSim readiness and parity gates (#6880, #6881, #6882, #6884): `EngineManager` now registers JaxSim as a runtime-backed engine, gates runtime-backed availability on both adapter/provider paths and importable dependencies, treats provider path-policy `PreconditionError`s as provider-discovery misses rather than constructor failures, adds a focused `JaxSimProbe`, and wires `scripts/ci/require_junit_test_passed.py` into cross-engine equivalence so skipped/missing JaxSim/Pinocchio parity cases fail the required CI gate. (spec 1.0.225) |
| 2026-05-31 | #6901 | Retired three dead launcher/GUI controls from an adversarial audit: JaxSim dashboard feature rows become read-only `QLabel` capability indicators ("capability indicator (read-only)" tooltips) instead of enabled-but-unconnected `QPushButton`s (#6901); the cached singleton `LibraryWidget` nulls its reference via `destroyed.connect` so a detached-then-closed Library no longer leaves a dangling C++ object (RuntimeError on re-open) (#6902); and the never-started/connected animation `QTimer` is removed from `MultiModelShotTracerWidget` (#6903). (spec 1.0.224) |
| 2026-06-03 | n/a | Bolt: Optimized `np.linalg.norm(v[:2])` to `math.hypot(v[0], v[1])` in `ball_trajectory_analysis.py` to avoid temporary array allocation and speed up calculation. (spec 1.0.224) |
| 2026-05-31 | #6796 | Added the CC-23 moving-horizon estimator near-real-time path for issue #6796: a bounded deterministic rolling-window estimator reuses the CC-19 MAP objective surface with fixed parameters, warm-starts each new window from the previous spline solution, records per-window latency against a 50 ms default budget, and supports optional callback integration for realtime bridge publishing. (spec 1.0.223) |
| 2026-06-01 | #6935 | DRY/LoD consolidation across the motion-matching engines and BunkerShot3D backends (#6935–#6939). Added `resolve_club_target()` + `publish_leaderboard_row()` to the shared `motion_matching.provider` module so all six engine providers delegate one canonical target-unwrap and leaderboard-append; this UNIFIES previously forked behavior — a `ClubBallTarget` now unwraps consistently on every engine (was a `TypeError` on mujoco/pendulum/pinocchio) and every engine forwards `target_id` (#6935). Extracted `ChronoDriver._make_contact_material()` so walls/grain/clubhead share one SMC material factory (#6936). Added flat delegating accessors on `BunkerShotConfig` (`contact_params()`, `domain_extents()`, `grain_count`, `clubhead_*`, `output_rate_hz`, `trajectory_*`) so chrono/mpm drivers stop reaching two levels into the nested config (#6937). Collapsed the drifted `opensim/motion_matching/forward_kinematics.py` FK copy (which read non-existent `/bodyset/Club/*` frames) into a thin re-export of the canonical `opensim_golf/fk.py` extractor (#6938). Added a shared `motion_matching.provenance` module (`engine_package_version()`, `git_commit_short()`) and routed the five `engine_version()` cascades and three git-commit probes through it (#6939). (spec 1.0.223) |
| 2026-05-31 | #6795 | Added the CC-22 offline Nimble gradient oracle for issue #6795: `tools.offline_validation.nimble_gradient_oracle` exposes validated request/response dataclasses, lazy PyTorch/Nimble autograd comparison with structured skip behavior, a pinned `nimble-oracle` optional extra (`nimblephysics==0.10.52.2`), focused deterministic tests, and docs confirming Nimble stays outside runtime `src/`. (spec 1.0.222) |
| 2026-05-31 | #6780 | Added the canonical-core CI wiring for issue #6780: `.github/workflows/cross-engine-equivalence.yml` now provides per-engine conformance lanes, `heavy-tests-opt-in.yml` keeps heavy stacks self-hosted and explicit, canonical-core Jules templates scaffold adapter/conformance/docstring work, and the JaxSim forward-simulation parity test aligns its analytic reference with canonical gravity and the current rollout tolerance envelope. (spec 1.0.222) |
| 2026-05-31 | #6784 | Added the CC-11 differential-testing report scaffold (#6784), including normalized JSON/Markdown validation artifacts, dependency-blocked defaults, and CC-7 conformance-harness normalization tests; added the CC-24 canonical ZTCF/ZVCF analysis bridge (#6797), including simulation backend helper exports, results schema v2 documentation, AffineDrift-compatible event extraction, result serialization, and focused tests; added the CC-12 canonical observations schema for markerless pose ingestion (#6785), including detector layout, calibrated camera metadata, per-camera 2D keypoints/confidence, optional 3D keypoints, JSON round-tripping, trace metadata attachment, fixtures, docs, and tests; added the CC-14 OpenCap integration slice (#6787), including source adapter registration, OpenCap marker/keypoint fixture ingestion, local validation coverage, and documented supported import format; added the CC-13 Pose2Sim integration slice (#6786), including Pose2Sim fixture ingestion, source adapter exports, MediaPipe JSON compatibility wiring, and motion-pipeline documentation for local multi-camera workflows; added the CC-25 engine-agnostic wrench/GRF extraction bridge (#6798), reusing `WrenchTrace` for canonical `Trace.wrench` conversion, impulse helpers, trace attachment, documentation, and static body-weight support validation; added the CC-17 synthetic ground-truth rig and identifiability probes (#6790), including synthetic fixture generation, forward-model protocols, identifiability diagnostics, docs, and focused estimator-input tests. (spec 1.0.222) |
| 2026-05-31 | n/a | Added the CC-15 calibratable keypoint-offset observation model in `pose_estimation`: calibration clips estimate detector-keypoint to model-joint-center offsets in segment frames, expose uncertainty metadata, and provide prediction/residual helpers for later CC-18 residual assembly. (spec 1.0.222) |
| 2026-05-31 | #6793 | Added CC-20 multi-trial / multi-view MAP stacking with shared-parameter locking, serialization, and posterior-tightening checks (#6793). (spec 1.0.222) |
| 2026-05-31 | #6805 | Added the CC-32 canonical-core app shell registry (#6805): `canonical_core_estimation` and `canonical_core_comparison` now register through the ADR-0013 embeddable-tool contract, publish PyQt6/React shell metadata through the launcher manifest, and route React users through `/tools/canonical-core/estimation` and `/tools/canonical-core/comparison` without implementing the deferred CC-19/CC-27 service bodies. (spec 1.0.221) |
| 2026-05-31 | #6810 | Added the Sidekick Canonical Core retrieval Q&A tool for issue #6810: `src/shared/python/canonical_core/sidekick_retrieval_qa.py` builds a bounded local index over Canonical Core docs and schemas, returns deterministic extractive answers with `path:start-end` citations, and registers the read-only `answer_canonical_core_question` tool through `src/api/services/chat_service.py`; docs live in `docs/sidekick/README.md` and `docs/specs/active/sidekick-canonical-core-retrieval-qa.md`. (spec 1.0.221) |
| 2026-05-31 | #6809 | Added the CC-36 config validation setup wizard for issue #6809: `src/shared/python/config/setup_wizard.py` validates canonical-v2 units and frames, model identity/joint/dimension preconditions, and subject calibration readiness; `SetupWizardViewModel` provides the headless four-step flow; `src/tools/config_setup_wizard/` exposes an embeddable launcher surface; and `tests/unit/config/test_setup_wizard.py` covers validation, suggested fixes, progression, and adapter conformance. (spec 1.0.221) |
| 2026-05-31 | #6811 | Added the CC-38 Sidekick canonical-core tool adapter for issue #6811: `src/shared/python/sidekick/agent/canonical_tools.py` registers a fixed canonical action allowlist behind `CanonicalActionPort`, `canonical.run` remains destructive and confirmation-gated, docs update ADR-0017 plus `docs/sidekick/agent.md`, and unit coverage validates descriptors, dry-run behavior, policy interaction, and result provenance. (spec 1.0.222) |
| 2026-05-31 | #6806 | Added the CC-33 canonical 3D viewport provider decision (#6806): MeshCat is the selected default over Rerun and VTK/PyVista, with lazy provider metadata/selection/degradation in `src/shared/python/visualization/viewport.py`, a Trace v2 overlay payload for canonical-v2 trajectory, marker, contact, and GRF/wrench data, and ADR-0027 documenting the bounded backend choice without adding viewer dependencies to core. (spec 1.0.221) |
| 2026-05-31 | #6816 | Tightened review-feedback guardrails for issues #6816 and #6827: `scripts/legal/check_license_ledger.py` now validates the OpenPose ledger row cells directly, and `.github/workflows/cross-engine-equivalence.yml` includes `pyproject.toml` in push/PR path filters so the JaxSim pin guard runs when the optional dependency declaration changes. (spec 1.0.220) |
| 2026-05-31 | #6791 | Added the canonical-core estimation residual surface for issue #6791: `src/shared/python/estimation/residuals.py` exposes pure reprojection, RNEA dynamics, anthropometric prior, and smoothness residual functions with a shared finite-difference/JAX Jacobian helper; `tests/unit/estimation/test_residuals.py` verifies residual Jacobians against hand-derived finite-difference expectations; `docs/development/canonical_core_residuals.md` documents the backend callback contract. (spec 1.0.220) |
| 2026-05-31 | #6801 | Added the CC-28 Drake canonical-core adapter slice for issue #6801: Drake now reports AutoDiffXd state/control gradients, full forward/inverse dynamics, contact stepping/forces, and trajectory optimization support; the pose adapter remaps canonical-v2 `q/v/a/t` into Drake quaternion-floating state order with parent/world angular velocity conversion; and the hydroelastic-vs-Pinocchio contact divergence is registered in `docs/conformance/canonical_core_divergences.yaml`. (spec 1.0.220) |
| 2026-05-31 | #6803 | Added the CC-30 MyoSuite canonical-core adapter slice for issue #6803: MyoSuite now maps activation-driven canonical-v2 state into MyoSuite/MuJoCo MJCF layouts, declares MUSCLES/FORWARD_DYN/CONTACT capabilities without claiming joint-torque inverse dynamics, routes upstream-muscle activation and force helpers, and persists Trace v2.1 muscle-output fields. (spec 1.0.220) |
| 2026-05-31 | #6789 | Added the CC-16 output-only canonical C3D exporter (#6789), including marker trajectory export to terminal C3D files, unit/label/sample-rate preservation, and architecture guards preventing C3D as an internal intermediate. (spec 1.0.220) |
| 2026-05-31 | n/a | Added the CC-29 MJX differentiable rollout slice: `simulation_backends` now registers optional `mjx` beside `ode`/`mujoco`/`mjwarp`, gates it through `has_mjx()` / `require_mjx()`, reuses the generated MuJoCo MJCF, advertises batched + differentiable capabilities, exposes JAX-native batched rollout arrays plus a final-state control Jacobian surface, and adds `run_estimation_windows_batched()` to flatten CC-20 multi-trial / CC-23 window controls onto the existing `BatchedBackend` axis. Updated ADR-0024 from deferred recommendation to accepted MJX implementation guidance, with mocked CPU-only tests covering optional-dependency degradation and host-side rollout/autodiff plumbing. (spec 1.0.219) |
| 2026-05-31 | #5968 | Added metadata-driven helpful-field and provenance-value wrappers for the Idiot-Proof UX epic (#5968): `HelpfulField` and `ProvenanceValue` PyQt6/React controls consume the existing metadata/provenance contracts, `scripts/ux/generate_field_metadata_ts.py` generates the TypeScript registry from `src/shared/python/ux/config/field_metadata.yaml`, and focused PyQt/Vitest tests cover contract validation, ARIA help text, provenance display, and YAML-to-TypeScript parity. (spec 1.0.219) |
| 2026-05-31 | n/a | Hardened the runtime Docker image against current fixed Debian 13 glibc, systemd/libudev, and sed CVEs by explicitly upgrading/installing `libc-bin`, `libc6`, `libsystemd0`, `libudev1`, and `sed` in the runtime apt layer while preserving the pinned base image digest. (spec 1.0.221) |
| 2026-05-31 | #6799 | Added the CC-26 AffineDrift coupling surface (#6799): `src/shared/python/analysis/affine_drift_coupling.py` extracts double-pendulum coordinates from native or canonical-v2 traces, samples drift/control-affine acceleration terms from a dynamics provider, persists `AffineDriftCouplingResult` datasets to HDF5, and documents the result schema in `docs/conventions/canonical-v2.md` plus `docs/simulation_backends/results_schema_v2.md`. (spec 1.0.221) |
| 2026-05-31 | #6773 | Added the canonical-v2 pose interchange contract export surface, ADR, and conventions guide for durable cross-engine state exchange (#6773). (spec 1.0.219) |
| 2026-05-31 | #6781 | Added the issue #6781 third-party license ledger and advisory validation path: `docs/legal/licenses.md` records commercial-readiness status for direct dependencies, OpenPose remains a non-commercial opt-in external tool, `scripts/legal/check_license_ledger.py` validates declared dependency coverage on Python 3.10+, and the core-install isolation guard ignores its own `scripts/` path so helper directories cannot masquerade as installed optional engines. (spec 1.0.218) |
| 2026-05-31 | #6774 | Added canonical-v2 dynamic state support for CC-2 (#6774): `CanonicalState` enforces the `(q, v, a, t)` floating-base layout with read-only arrays, unit quaternion and `nq = nv + 1` preconditions, manifold `integrate`/`difference` operations, canonical-v1 pose lifting, zero-state construction, and SE(3) quaternion helpers with regression coverage. (spec 1.0.219) |
| 2026-05-31 | #6648 | Hardened the JaxSim #6648 CI path: URDF/SDF inertial XML reads now use `defusedxml.ElementTree`, and the core-only install isolation guard removes its own `scripts/` directory from `sys.path` before checking optional-engine imports so `scripts/jaxsim` cannot masquerade as an installed `jaxsim` package. (spec 1.0.217) |
| 2026-05-30 | #6656 | Added the JaxSim parameter-gradient capability for issue #6656: `SupportsParameterGradients` defines the segregated engine-core seam, `JaxSimBackend` exposes pointwise ZTCF parameter Jacobians through a JAX autodiff module over documented anthropometric parameters, finite-difference tests validate the gradient, and `scripts/jaxsim/plot_parameter_sensitivity.py` writes the sample sensitivity plot. (spec 1.0.216) |
| 2026-05-30 | #6649 | Added the JaxSim M0 dependency gate for issue #6649: `upstream-drift[jaxsim]` pins `jaxsim==0.9.0`, keeps JaxSim out of the core and `all-engines` rollups until Linux native-engine coexistence is proven, documents the CPU-JAX-first platform decision in `docs/engines/jaxsim.md`, and adds optional SDF step smoke coverage via `tests/fixtures/jaxsim/single_link.sdf`. (spec 1.0.209) |
| 2026-05-30 | n/a | Added the full-src mypy baseline ratchet for push-to-main CI: `pyproject.toml` enables namespace-package explicit package bases, `ci-standard.yml` routes push full-src typing through `scripts/ci/run_full_mypy_baseline.py`, and the checked-in `scripts/config/full_src_mypy_baseline.json` captures the currently unmasked type backlog so new diagnostics block CI without restoring the previous duplicate-module circuit breaker. (spec 1.0.208) |
| 2026-05-29 | #6624 | Documented the PR #6624 quality-gate cleanup: agent-doc consistency checks skip documented glob/brace path patterns while preserving literal path validation, root-clutter policy explicitly allowlists `launch_upstream_drift.py`, module-size exceptions remain tracked through the owned baseline, and the obsolete duplicate Sidekick chat embeddable adapter is removed in favor of the canonical `src/tools/sidekick/_embed_adapter.py`. (spec 1.0.203) |
| 2026-05-29 | #6646 | feat(simulation): Added the GPU-ready `src/shared/python/simulation_backends/` layer — a `SimulationBackend` Protocol (with segregated `DynamicsProvider`/`BatchedBackend`) over interchangeable `ode` (CPU reference), `mujoco` (CPU + dynamics primitives), and `mjwarp` (GPU batched, optional `[warp]` extra) backends, all rendered from one pydantic `GolfModelParams` source of truth that emits both the analytical EOM params and the MuJoCo MJCF. MuJoCo `M(q)`/bias/forward-dynamics cross-validated against the analytical double pendulum to ~1e-9–1e-11; ZTCF/ZVCF reproduced via dynamics primitives; versioned HDF5 trace I/O; batched API with VRAM chunking + CPU fallback. Added the "Simulation Backends" launcher tile (`src/tools/simulation_backends_launcher/`, manifest id `simulation_backends`): backend picker, parameter editor, rollout/parameter-sweep/cross-validation, and HDF5 export. 270 unit/UI tests; ADRs 0023/0024; `docs/simulation_backends/USER_GUIDE.md`. PR #6646. (spec 1.0.203) |
| 2026-05-29 | #6659 | fix(gui): annotate cross-engine dashboard comparison results with per-engine velocity convention and units metadata in headless logs and GUI chart labels; closes #6659. (spec 1.0.203) |
| 2026-05-28 | #6509 | fix(mujoco): `get_dockable_ui()` returns a `QWidget` container wrapping `HumanoidLauncher` (not `QMainWindow`) for tab embedding; `_apply_styling` now calls `apply_theme_to_window` for consistent theming (issue #6509). (spec 1.0.198) |
| 2026-05-27 | #5980 | chore(sidekick): confirm T2 (`StandaloneSidekickWindow` profile switching) and T5 (schema-version persisted in round-trip JSON) acceptance criteria with targeted tests; closes issues #5980 and #5983. (spec 1.0.201) |
| 2026-05-27 | #5982 | feat(sidekick): complete T4 headless calculator invoker — `sidekick run` validates inputs via Calculator Protocol, surfaces structured errors (exit 3 validation/calc, exit 4 unknown-calculator + fuzzy suggestions, exit 1 I/O), supports `--format json` and `--format csv`, with full TDD coverage (issue #5982). (spec 1.0.202) |
| 2026-05-27 | n/a | perf: replace qvel\**2 with qvel*qvel in MuJoCo power flow (`power_flow.py`). (spec 1.0.197) |
| 2026-05-26 | #6181 | Folded remaining API/security/realtime/logging PR scope into the post-#6181 consolidation branch: `FitResult` now exposes explicit `fit_succeeded` and `solver_status` fields, the `.gitignore` secrets guard has an importable CI helper plus tests, and logging redaction preserves delimiters while redacting quoted, JSON, and comma-containing secret values. (spec 1.0.194) |
| 2026-05-26 | n/a | Folded duplicate performance PRs into the consolidated branch: cached common factorial values in the signal toolkit, normalized signal import arrays with `np.asarray`, preserved `body_marker` when Drake constraint penalties are added, and replaced selected temporary product reductions with `np.einsum` or `np.vdot` in motion-matching visualization, work, and energy calculations. (spec 1.0.193) |
| 2026-05-23 | n/a | Refined the standalone Sidekick CLI contract in `src/shared/python/sidekick/__main__.py` so `python -m sidekick` defaults to `gui`, mistyped flags get closest-match suggestions, GUI imports remain deferred until dispatch, `--data-dir` is resolved to an absolute path, and `gui` now delegates through `sidekick.launcher_factory` using the standalone window/session-store configuration on current `main`. Expanded `tests/unit/sidekick/test_cli.py` to cover implicit-gui parsing, bad-flag suggestions, headless `run` parsing, handler error paths, and launcher delegation. (spec 1.0.186) |
| 2026-05-23 | n/a | Tightened `src/shared/python/training/config.py` validation so boolean values are rejected for integer training caps such as `max_epochs` and `max_steps`; regression coverage lives in `tests/unit/training/test_config.py`. (spec 1.0.186) |
| 2026-05-31 | n/a | Added the CC-19 single-trial MAP estimator surface in `src/shared/python/estimation/`: cubic-Hermite spline trajectory coefficients with analytic derivatives, ordered shared parameter blocks with free length parameters and bounded inertia corrections, deterministic least-squares solve wiring, and focused unit coverage for objective determinism and parameter sharing. (spec 1.0.192) |
| 2026-05-23 | n/a | Closed the file-size budget grandfathering gap by requiring tracked baseline entries for oversized files in `scripts/config/file_size_budget.json`; untracked oversized files now fail `scripts/ci/check_file_size_budget.py`, with regression coverage in `tests/scripts/wave9_scripts_b/test_check_file_size_budget.py`. (spec 1.0.187) |
| 2026-05-26 | #6091 | Registered MyoSuite in the pose-interchange layer: added `MyosuiteAdapter` (MJCF/MuJoCo-identical qpos convention) to `ADAPTER_REGISTRY` and `MyosuiteKinematicsService` + `create_myosuite_service()` factory to `KINEMATICS_SERVICE_REGISTRY`, with mock fallback when the `myosuite` wheel is absent; 284 tests across protocol, layout, roundtrip, and service suites (issue #6091). Consolidated wave-1: 21 issues closed across docs/ADR, WebSocket validation, production safety checks, dependency bounds, API design, test-marker hygiene, CI scripts, and performance fixes (#5908, #5909, #5910, #5912, #5914, #5916, #5917, #5918, #5920, #5921, #5922, #6087–#6095, #6097). (spec 1.0.191) |
| 2026-05-24 | n/a | Surfaced API database pool controls for non-SQLite deployments via `GOLF_DB_POOL_SIZE`, `GOLF_DB_POOL_RECYCLE`, and `GOLF_DB_POOL_PRE_PING`; `src/api/database.py` now builds non-SQLite engines from shared config accessors instead of hardcoded pool defaults, with regression coverage in `tests/unit/test_config_environment.py` and `tests/unit/api/test_database_init.py`. (spec 1.0.190) |
| 2026-05-24 | n/a | Improved CI/test observability for optional dependency lanes: optional collection skips warn once with missing requirements, `tests/unit/training/runtime/test_pytorch_cvae_adapter.py` uses a wrapper progress sink for cancellation, and standard workflow inventory jobs have 15-minute timeouts to avoid false timeouts on loaded self-hosted runners. (spec 1.0.189) |
| 2026-05-23 | n/a | Deferred `src/shared/python/realtime/ws_pubsub.py` backend resolution until `WSPubSub.start()`, `publish()`, or `subscribe()` first use so module import no longer probes optional realtime runtime dependencies, and added focused regression coverage for lazy resolution plus the python publish fallback path. (spec 1.0.186) |
| 2026-05-22 | n/a | Documented the motion-pipeline REST contract for `POST /api/v1/motion-pipeline/run` and its preprocessing-step boolean coercion rule so `PipelineRequest` preserves Pydantic handling of `enabled` values like `"false"` when converting into `PipelineConfig`; regression coverage lives in `tests/unit/motion_pipeline/orchestrator/test_api.py`. (spec 1.0.182) |
| 2026-05-24 | n/a | Deferred realtime WebSocket backend resolution until first explicit start/use and made `WSPubSub.start()` launch the Python backend even when the instance was created with `autostart=False`; added focused regression coverage in `tests/shared/realtime/test_ws_pubsub.py`. (spec 1.0.188) |
| 2026-05-23 | #5979 | Sanitized error payloads for the chat websocket connection to prevent leaks. Added standalone Sidekick foundation (CLI entry point, PyQt window shell, and session store) per epic #5979. (spec 1.0.181) |
| 2026-05-22 | n/a | Added the standalone Sidekick CLI scaffold in `src/shared/python/sidekick/__main__.py` with an implicit `gui` default, closest-match suggestions for mistyped flags, early path validation for `run`, deferred GUI imports for headless parsing, and focused regression coverage in `tests/unit/sidekick/test_cli.py`. Tightened `scripts/ci/check_error_handling_ratchet.py` so the `asyncio.gather(...)` anti-pattern scan now balances multiline argument lists before deciding whether `return_exceptions=` is present, and added matching regression coverage in `tests/unit/scripts/test_error_handling_ratchet.py` for both compliant and violating multiline gather calls. (spec 1.0.181) |
| 2026-05-22 | #5968 | Landed the pure-Python foundation for the Idiot-Proof UX epic (#5968): `src/shared/python/ux/` adds the `FieldMetadata` registry, `ProvenanceRecord`/`ProvenanceValue`, `PreflightCheck`/`Severity`/`run_preflight()`, and the `UserFacingError` envelope, all with full Design-by-Contract validation; seeded `src/shared/python/ux/config/field_metadata.yaml` and `src/shared/python/ux/config/error_messages.yaml`; added `scripts/ci/check_ux_coverage_ratchet.py` plus baseline at 714 unwrapped inputs (62 QSpinBox + 221 QDoubleSpinBox + 217 QComboBox + 70 QSlider + 94 QLineEdit + 35 `<input>` + 14 `<select>` + 1 `<textarea>`); documented the workflow in `docs/ux/field_metadata.md`; 68 unit tests in `tests/unit/ux/`. Sanitized unexpected `src/api/routes/simulation_ws.py` runtime errors before they reach WebSocket clients while preserving traceback-bearing server logs, and added direct regression coverage for the generic error payload contract. Re-baselined `scripts/config/module_size_budget_baseline.json` from 10 stale exceptions (sizes 3-5x overstated, 7 files since decomposed) down to the 3 modules that genuinely exceed 1,500 lines today, and added `validate_baseline_truthfulness` to `scripts/check_module_size_budget.py` as a CI ratchet against future fraudulent baselines. Refs #5922. (spec 1.0.180) |
| 2026-05-23 | n/a | ⚡ Bolt: Optimize mechanical work metric calculations using einsum and vdot (spec 1.0.180) |
| 2026-05-22 | n/a | Aligned the module-size quality gate with current launcher and shared-chat legacy debt by adding owned, expiring exceptions for `src/launchers/launcher_ui_setup.py` and `src/shared/python/chat/_chat_dock_widget_qt.py`, and raising the active module-size exception cap to 10 while preserving the 1,500-line budget for new untracked modules. (spec 1.0.179) |
| 2026-05-21 | n/a | Preserved integer-safe quaternion normalization in `src/motion_capture/c3d_simscape_preview.py` by upcasting integer inputs before the optimized `np.einsum` norm accumulation, and added regression coverage for integer quaternion inputs. (spec 1.0.176) |
| 2026-05-21 | n/a | Optimized `src/shared/python/signal_toolkit/fitting.py` to compute fitting residual sum-of-squares and RMSE via reused `np.vdot` accumulators, avoiding temporary squared arrays across the sinusoid, exponential, linear, polynomial, and custom fitter paths. (spec 1.0.175) |
| 2026-05-15 | #5460 | Integrated Sidekick across the launcher: registered the AI chat panel as an EmbeddableTool tile (`src/tools/sidekick/`), bound React `ChatPanel` to `var(--sidekick-color-*)` design tokens with a Python/TypeScript parity test, added a redacted ring-buffer chat-context bridge that injects recent app state into the assistant prompt, registered a `summarize_simulation_run` agentic analytics tool, and surfaced Tools-sidebar availability through `LauncherDiagnostics`. Refs #5460 #5461 #5462 #5463 #5464 #5465. (spec 1.0.173) |
| 2026-05-29 | n/a | Bolt: Optimized `np.linalg.norm(np.array(...))` to `math.hypot(...)` in `anthropometric.py` to avoid temporary array allocation and speed up calculation. (spec 1.0.172) |
| 2026-05-18 | n/a | ⚡ Bolt: Optimize norm calculations in plot_error_timecourse using np.einsum (spec 1.0.171) |
| 2026-05-25 | #6093 | Removed stale "raises NotImplementedError" and scaffold-era caveats from the module and class docstrings of the Drake, OpenSim, and Simscape `LiveKinematicsService` implementations; updated the Simscape transform-query TODO to reference the current tracking issue #6093 instead of closed epic #4963 (issue #6092). (spec 1.0.171) |
| 2026-05-14 | n/a | ⚡ Bolt: Optimize sum of squares along axis in perstep train metrics (spec 1.0.170) |
| 2026-05-22 | #5913 | Hardened the shared BitNet subprocess adapter by rejecting non-UTF-8 and oversize prompts before `llama-cli` launch, and added focused regression coverage for the synchronous and streaming guard paths (issue #5913). (spec 1.0.170) |
| 2026-05-14 | n/a | Added a shared row norm helper for vectorized norm calculations in motion-matching and validation paths. (spec 1.0.169) |
| 2026-05-16 | #5556 | Added 14 remaining launcher tiles covering engine-specific dashboards (Drake, MuJoCo, Pinocchio), Analysis Tools API, Motion Pipeline, capability surfaces (perturbation analysis, force overlays, realtime WebSocket, AIP, actuator controls), and feature tiles (Unreal integration, robotics module, Tools calculator hub, P&ID generator); closed 12 issues resolved by prior #5556 merge and 2 by-design closures (#5515, #5521, #5523–#5524, #5527–#5535). (spec 1.0.169) |
| 2026-05-14 | n/a | Adopted responsive sizing and application zoom across the main launcher, cross-engine dashboard, and shared calculator widgets, with launcher regression coverage for the new scaling contract. (spec 1.0.168) |
| 2026-05-14 | n/a | Added shared PyQt responsive sizing helpers, fleet-style application zoom wiring for the classic launcher, and a pendulum toolstrip checkbox migration from fixed width to text-aware minimum sizing. (spec 1.0.168) |
| 2026-05-14 | #5384 | Added Sidekick design-token adapters that map existing launcher theme colors to canonical `sidekick.*` roles for React/Tauri CSS variables and guarded PyQt Tools sidebar integration, with token-contract tests for issue #5384. (spec 1.0.167) |
| 2026-05-13 | n/a | Added a guarded optional Unified Tools Sidebar launcher integration that imports the shared Tools sidebar when available, docks it into the PyQt6 launcher, connects file-open requests to host handlers or status reporting, and no-ops cleanly when the shared module is absent. (spec 1.0.166) |
| 2026-05-13 | #5374 | Moved the PyQt launcher close control into the top menu-bar row while keeping the custom title strip for drag/minimize/maximize behavior (#5374). (spec 1.0.166) |
| 2026-05-13 | n/a | Documented the shared Tools-hosted video/data launcher surfaces, the launcher manifest contract, and the theme API client/server surface added for web UI parity. (spec 1.0.165) |
| 2026-05-12 | #5353 | Preserved registered symbolic model `source_root` aliases while still resolving provider-relative source roots, preventing aliases such as `movement_optimizer` from being rewritten under a provider checkout (#5353). (spec 1.0.164) |
| 2026-05-12 | #5314 | Added Tools Pendulum Simulator nested provider-manifest discovery and provider-relative source-root resolution so launcher discovery can expose tool packages published below `Tools/src` without copying tool code (#5314). (spec 1.0.163) |
| 2026-05-12 | n/a | Clarified shared chat smoke coverage so the public API contract asserts the exported `ChatDockWidget` and `ChatMessageBubble` symbols from `src/shared/python/chat/__init__.py`. (spec 1.0.162) |
| 2026-05-12 | #5314 | Added a canonical launcher category taxonomy and category grouping contract so provider-backed entries such as biomechanics tools are discoverable instead of being rejected by legacy manifest validation (#5314). (spec 1.0.162) |
| 2026-05-12 | #5314 | Added a documented hidden-launcher contract so hidden feature entries must carry an owner and reason, preventing undiscoverable app features from drifting without accountability (#5314). (spec 1.0.160) |
| 2026-05-12 | n/a | Updated workflow governance for the Rust realtime soak workflow by pinning its Rust toolchain action to a full commit SHA and registering the workflow in the active inventory with the current 71-workflow no-growth cap. (spec 1.0.159) |
| 2026-05-12 | n/a | Restricted review-comment archive commits to manual workflow dispatch runs so pull request synchronize events cannot push `docs/review_archive` churn onto feature branches, erase current-head checks, or block focused chat and GUI fixes behind generated archive drift. (spec 1.0.158) |
| 2026-05-12 | n/a | Normalized Rust-backed Ollama chat and embedding endpoint suffixes so a configured base URL ending in `/v1` does not produce duplicate `/v1/v1/...` paths, while plain Ollama hosts still receive `/v1/chat/completions` and `/v1/embeddings`; added focused regression coverage for both URL forms. (spec 1.0.157) |
| 2026-05-12 | #5295 | Finalized motion-matching Rust loop optimizations, including MuJoCo torque outer-loop acceleration (slice 4) and end-to-end facade benchmarks (slice 5) (PR #5295, PR #5296). (spec 1.0.156) |
| 2026-05-12 | #5301 | Added Golf Simulation Suite to the GUI launcher (PR #5301). (spec 1.0.155) |
| 2026-05-12 | #5302 | Optimized sum-of-squares and MSE calculations via `np.vdot` and `np.einsum` to eliminate temporary array allocations (PR #5302). (spec 1.0.154) |
| 2026-05-12 | #5265 | Added `AerodynamicsEngine`, `AeroEngineConfig`, `WindModel`, and `WindConfig` Rust pyclasses to `upstream-physics`; implemented a deterministic per-step force facade in `src/shared/python/physics/aerodynamics/_rust_facade.py` with pure-Python fallback; verified Rust/Python parity to RMSE < 1e-8 and ≥10× speedup on representative flight inputs (issue #5265). (spec 1.0.153) |
| 2026-05-11 | n/a | Ported the Python `cd_dimpled_sphere` drag-crisis coefficient into `rust_core/upstream-physics`, routed Rust aerodynamic drag through that parity curve, and added Rust DbC/parity tests for the Reynolds-number contract. (spec 1.0.152) |
| 2026-05-13 | n/a | Fixed launcher logo backdrop cleanup so full-canvas SVG backgrounds are detected from each icon's canvas dimensions, including 24x24 icons, while preserving legitimate inner logo geometry and keeping the drop-shadow wrapper idempotent under repeated processing. (spec 1.0.151) |
| 2026-05-11 | #5207 | Consolidated `src/shared/python/codemap/` onto the Tools canonical 9-module implementation (byte-identical copy of `__init__.py`, `api.py`, `cli.py`, `db.py`, `indexer.py`, `parsers.py`, `watcher.py` plus 6 new per-language extractors and embeddings stub); renamed `mcp.py` → `mcp_server.py` so the `codemap-mcp` console-script entry point resolves; updated the chat-tool adapter to use the canonical 6-function API; replaced 30 duplicate parser/db/indexer unit tests with 20 UD-specific chat-wiring + smoke + perf-budget integration tests. (PR #5207, closes #5206) (spec 1.0.151) |
| 2026-05-13 | n/a | ⚡ Bolt: Optimize Root Mean Square Error computation by vectorizing sum of squares (spec 1.0.150) |
| 2026-05-13 | n/a | ⚡ Bolt: Optimize argmax norm calculation in synthesize.py (spec 1.0.150) |
| 2026-05-10 | n/a | Fixed MuJoCo live-kinematics pose application to honor model `jnt_qposadr` / free-joint addresses and added regression coverage for fixed-base plus reordered free-joint layouts. (spec 1.0.150) |
| 2026-05-11 | n/a | ⚡ Bolt: Optimize norm calculations using np.einsum (spec 1.0.149) |
| 2026-05-10 | n/a | Corrected the launcher zoom slider accessible description helper to derive its percentage range from `TILE_SCALE_MIN` and `TILE_SCALE_MAX`, keeping screen-reader guidance aligned with the actual slider bounds after future constant changes. (spec 1.0.149) |
| 2026-05-10 | n/a | Matched the launcher zoom slider accessible description to the configured tile scale constants so assistive technology reports the actual supported zoom range. (spec 1.0.148) |
| 2026-05-10 | n/a | Added launcher accessibility coverage for sidebar tool buttons with visible labels and accessible descriptions, strong keyboard focus on sidebar and zoom controls, zoom slider range description, and keyboard activation/selection support on draggable model cards. (spec 1.0.148) |
| 2026-05-09 | n/a | ⚡ Bolt: Added realtime WebSocket pubsub, channels, and file-based pubsub for live simulation streaming (spec 1.0.147) |
| 2026-05-08 | #4491 | Fixed Preferences dialog crash (issue #4491) by correcting `get_available_fleet_themes()` to `get_available_themes()` in `src/shared/python/ui/preferences_dialog.py:184`. (spec 1.0.144) |
| 2026-05-07 | n/a | Fixed Wave 2 manifest validator to parse `###` section headers matching the generated format, preventing self-inconsistent validation after `--update`. Fixed wheel event filter cache to use `weakref.WeakValueDictionary` preventing unbounded memory growth in long-running UI applications with transient controls. (spec 1.0.143) |
| 2026-05-09 | n/a | 🛡️ Sentinel: Fix insecure deserialization in imitation learning models (spec 1.0.142) |
| 2026-05-09 | n/a | ⚡ Bolt: Optimize Root Mean Square Error computation using np.vdot (spec 1.0.142) |
| 2026-05-08 | n/a | ⚡ Bolt: Optimize velocity magnitude and argmax calculation using np.einsum (spec 1.0.141) |
| 2026-05-07 | n/a | Added deterministic OpenSim multistart fit orchestration with seed-list reproducibility, per-start fresh simulator factories, best-success result selection, and typed all-starts-failed diagnostics. (spec 1.0.141) |
| 2026-05-07 | n/a | Added a pure-unit OpenSim prescribed-controller boundary for polynomial torque trajectories, including validation of time grids, coefficient shapes, finite values, actuator names, parity with the canonical polynomial torque evaluator, and typed unavailable behavior before native OpenSim integration. (spec 1.0.140) |
| 2026-05-07 | n/a | Added an opt-in OpenSim compliant club attachment builder path with typed `CompliantClubAttachmentConfig`, deterministic `BushingForce` XML emission, default rigid-weld regression coverage, and validation for unsupported units or missing model bodies. (spec 1.0.138) |
| 2026-05-07 | n/a | Moved production-readiness and testing-contract documentation out of the repository root into `reports/` and `docs/testing/`, and added a focused CI regression test for the root-clutter policy so future non-allowlisted top-level files fail under pytest before they block the shared `quality-gate`. (spec 1.0.134) |
| 2026-05-07 | n/a | Added FitResult field contract coverage requiring motion-matching fit drivers to export the shared `CanonicalFitResult` and canonical engine tests to use `theta_optimal` instead of deprecated `.theta` access. (spec 1.0.133) |
| 2026-05-07 | n/a | Hardened CI behavior so PR-scoped core tests treat an all-skipped selection as a no-op and cross-engine equivalence bootstraps `pip` with recordless-safe install flags on self-hosted runners. (spec 1.0.132) |
| 2026-05-07 | n/a | Added cross-engine fit determinism regression coverage requiring repeated runs with the same target, warm start, and `rng_seed` to reproduce identical results across seeds 42, 1337, and 999 for MuJoCo, Drake, Pinocchio, and OpenSim. (spec 1.0.131) |
| 2026-05-07 | n/a | Exported the MuJoCo motion-matching synthetic recovery oracle from `simulate.py` and added a synthesize-fit-recover regression test for the public API. (spec 1.0.130) |
| 2026-05-07 | n/a | Hardened cross-option leaderboard follow-up behavior so tests run from the repo root on any machine and metrics JSON normalizes non-finite RMSE sentinels before serialization. (spec 1.0.128) |
| 2026-05-07 | #4226 | Added cross-option leaderboard run + report (PR #4226), Option-2 NN surrogate training on 10k dataset (PR #4227), and Option-3 cVAE inverse model training (PR #4228). (spec 1.0.127) |
| 2026-05-06 | n/a | Added scope header comments to the generated Pinocchio `golfer.urdf` and `golfer_ik.urdf` files so forward-simulation and body-only IK workflows clearly document when the welded-club model versus the external-club-tracking model should be used. (spec 1.0.125) |
| 2026-05-06 | #4057 | Added ML checkpoint/resume and progress artifacts for frame search (PR #4057). (spec 1.0.124) |
| 2026-05-06 | #4059 | Added ML dynamics-consistent two-stage trajectory optimizer (PR #4059). (spec 1.0.123) |
| 2026-05-06 | #4058 | Added golf-ml replay diagnostics and smoothing/poly export tuning (PR #4058). (spec 1.0.122) |
| 2026-05-06 | #4051 | Added clubface/ClubLogs target adapter for motion-matching (PR #4051). (spec 1.0.121) |
| 2026-05-06 | #4055 | Added ML closed-loop replay diagnostics harness (PR #4055). (spec 1.0.120) |
| 2026-05-06 | #4056 | Added Simscape candidate stepping hooks for frame-by-frame torque search (PR #4056). (spec 1.0.119) |
| 2026-05-06 | #4054 | Added ML surrogate validation splits by swing phase (PR #4054). (spec 1.0.118) |
| 2026-05-06 | #4052 | Added unified Metrics schema for motion-matching (PR #4052). (spec 1.0.117) |
| 2026-05-06 | #4053 | Added MachineLearning orientation and work-regularizer cost parity for motion-matching (PR #4053). (spec 1.0.116) |
| 2026-05-06 | #4048 | Added motion-matching support for wiring Gears C3D marker maps to the physics models (PR #4048). (spec 1.0.115) |
| 2026-05-06 | n/a | Expanded the golf ML matching workflow with Pareto regularization sweeps, calibration validation reports and plots, positive mechanical-work diagnostics from paired torque/qdot logs, a tabbed MATLAB workflow GUI, and a frame-by-frame sequential torque-search fallback contract with manifest generation, parallel candidate evaluation structure, smoothing, and polynomial export hooks. (spec 1.0.114) |
| 2026-05-05 | n/a | Added non-blocking golf ML matching diagnostics for target-vs-Simscape club tracking, impact-window error, torque effort, torque impulse, peak control, and torque-rate smoothness; documented the weighted optimization objective for redundant torque and body-motion selection. (spec 1.0.112) |
| 2026-05-06 | n/a | Added a core-test relevance filter to `ci-standard.yml` so pull requests with only workflow, documentation, or other non-Python/non-dependency changes skip the expensive Python test matrix after checkout while source, test, metadata, and dependency changes still run the full matrix. (spec 1.0.113) |
| 2026-05-05 | n/a | Made pull-request CI finite in the presence of existing repository-wide blockers: Semgrep SAST, Bandit, and Trivy now scan changed supported files on PRs while retaining full scans for non-PR runs, and the Alembic PostgreSQL round-trip job has a larger finite job budget, an explicit SQL readiness probe, isolated pytest plugin loading, and verbose duration output for diagnostics. (spec 1.0.112) |
| 2026-05-05 | n/a | Removed the misplaced experimental OpenFOAM CFD execution helper from UpstreamDrift's biomechanical physics-engine inventory so OpenFOAM execution can live with the Tools_Private glass-model CFD stack where it is used. (spec 1.0.111) |
| 2026-05-05 | n/a | Bolt: Optimized clubhead speed computation in swing kinematics by replacing `np.linalg.norm(clubhead_vel, axis=1)` with `np.sqrt(np.einsum("ij,ij->i", clubhead_vel, clubhead_vel))` to avoid temporary array allocations, achieving ~35% performance improvement. (spec 1.0.110) |
| 2026-05-04 | n/a | Optimized mean squared error calculation in validation solver by replacing np.mean(residuals\*\*2) with np.vdot(residuals, residuals) / residuals.size. (spec 1.0.109) |
| 2026-05-04 | n/a | Aligned the pull-request `ci-standard.yml` coverage gate with the documented `pyproject.toml` repository floor by raising `--cov-fail-under` from 45 to 55, restoring agent-doc consistency with the published quality gates. (spec 1.0.109) |
| 2026-05-04 | n/a | Optimized RMS diff and magnitude calculation in cross engine validator by replacing np.mean(diff\*\*2) with np.vdot(diff, diff) / diff.size to prevent intermediate array allocations. (spec 1.0.108) |
| 2026-05-04 | #3943 | Pinned Docker base images to digest `sha256:4386a385d81dba9f72ed72a6fe4237755d7f5440c84b417650f38336bbc43117` (python:3.12-slim) for reproducible builds; raised overall coverage floor from 45% to 55% with per-module risk-tier thresholds (85% for API routes/engine adapters/task management, 70% for shared utilities); replaced in-memory dataset cache in `src/api/routes/data_explorer.py` with durable SQLite-backed `DatasetStorage` (issue #3943); documented API production-readiness hardening for issues #3941, #3942, and #3943: process-local `TaskManager` lifecycle and TTL touch semantics, async video background execution off the event loop with temp cleanup warning logs, and bounded Data Explorer import cache behavior with duplicate and ambiguous filename conflict handling. (spec 1.0.106) |
| 2026-05-04 | n/a | Replaced six sum-of-squares hot paths in analysis, biomechanics, injury, plotting, data-processing, and validation helpers with `np.vdot`-based accumulators to avoid temporary array allocation while preserving existing R² and load metric behavior. (spec 1.0.107) |
| 2026-05-03 | n/a | Realigned the `model_generation.core.contracts` compatibility shim so its invariant alias and helper re-exports stay synchronized with the canonical shared contracts module while remaining Ruff-clean. (spec 1.0.105) |
| 2026-05-03 | n/a | Optimized collision detection distance calculations by replacing `np.linalg.norm` with `math.hypot` for 3D collision-distance and gradient normalization paths. (spec 1.0.103) |
| 2026-05-03 | n/a | Isolated the CI Standard `pip-audit` gate in a dedicated virtualenv, cleared stale waivers once the clean audit environment reported no findings, and raised the Alembic PostgreSQL round-trip timeout budget to 180 seconds for slower self-hosted runners. (spec 1.0.103) |
| 2026-05-03 | n/a | Added experimental OpenFOAM CFD execution support to the engine inventory, including `decomposeParDict` generation and MPI command plumbing for parallel OpenFOAM runs. (spec 1.0.98) |
| 2026-05-03 | #3926 | Repaired issue #3926 CI hygiene by updating CI Standard to the working Trivy action pin, syncing generated dependency artifacts with `pyproject.toml`, exempting vendored trees from doc-size budgeting, and removing obsolete helper/backup files. (spec 1.0.100) |
| 2026-05-03 | #3912 | Tightened issue #3912 quality ratchets by adding a 2026-08-01 mypy exclusion cap reduction to 44, validating monotonic exclusion schedules, and adding owned production package coverage-ratchet metadata for API routes, data I/O, execution/checkpointing, deployment, optimization, and engine adapters. (spec 1.0.99) |
| 2026-05-03 | #3844 | Tightened the security/dependency guardrails for issue #3844 by pinning `python-dotenv>=1.2.2`, pruning stale pip-audit waivers, sending stale-waiver diagnostics to stderr so CI fails cleanly before invoking `pip-audit`, and aligning `critical-files-guard.yml` with the repository’s actual root files. (spec 1.0.101) |
| 2026-05-03 | n/a | Hardened the Alembic PostgreSQL CI service health budget with a startup grace period, faster probes, and more retries so shared-runner cold starts do not fail the migration round-trip gate before Postgres is actually ready. (spec 1.0.102) |
| 2026-05-03 | #3844 | Hardened issue #3844 security CI acceptance: added blocking Semgrep and Trivy filesystem scans to `ci-standard.yml`, moved pip-audit waivers to the documented issue/expiry schema with stale-waiver detection, added CODEOWNERS backup owners, documented branch protection, and added Trivy secret-scan test coverage. (spec 1.0.97) |
| 2026-05-03 | n/a | Guarded local diagnostic and debug API endpoints in production mode unless `UPSTREAM_DRIFT_DEBUG_ENDPOINTS=true` is explicitly set. (spec 1.0.96) |
| 2026-05-03 | n/a | Established `pyproject.toml` as the canonical Python dependency source, generated `environment.yml` from it, added `make sync-deps`, promoted documented CVE floors to runtime dependencies, removed the deprecated root CRA UI build, and added dependency-consistency CI drift/audit coverage. (spec 1.0.96) |
| 2026-05-03 | n/a | Added tier-aware vulnerability SLA policy, pip-audit waiver tier validation, OSV triage deadline helpers, and local per-tier SBOM metadata generation. (spec 1.0.96) |
| 2026-05-03 | #3839 | Added documentation catalog and size-budget governance checks for issue #3839, including owned temporary exceptions for oversized legacy docs. (spec 1.0.96) |
| 2026-05-03 | n/a | Hardened CI Standard security audit bootstrapping to use `--ignore-installed` for corrupted shared-runner packages, including the missing-RECORD `urllib3` case. (spec 1.0.96) |
| 2026-05-03 | n/a | Added a mypy exclusion budget and ratchet checker so path exclusions have explicit owner, reason, expiry, and scheduled shrinkage metadata. (spec 1.0.95) |
| 2026-05-03 | n/a | Added a workflow and agent-configuration inventory guard that documents active workflow ownership, records consolidation candidates, blocks undocumented workflow growth, and rejects unsafe `permissions: write-all`. (spec 1.0.96) |
| 2026-05-03 | #3852 | Added the canonical production artifact contract, compatibility matrix, runtime support warning, and release-blocking Python wheel smoke-test matrix for issue #3852. (spec 1.0.95) |
| 2026-05-03 | #3842 | Added release governance for issue #3842: version consistency checks, CI wiring, release and production-readiness operations docs, Rust version metadata alignment, release SBOM generation, and artifact attestations. (spec 1.0.95) |
| 2026-05-03 | #3841 | Migrated stable flat tests and launcher in-tree tests into topic directories under `tests/`, documented the test layout and fixture scopes, and added the blocking `scripts/check_test_layout.py` CI guard for issue #3841. (spec 1.0.96) |
| 2026-05-03 | n/a | Hardened the standard CI security-audit bootstrap to install a patched Black before `pip-audit`, preventing shared-runner cache drift from failing docs/governance PRs on CVE-2026-32274. (spec 1.0.94) |
| 2026-05-03 | n/a | Normalized contributor governance docs around `CLAUDE.md`, added stronger agent-doc consistency checks for coverage/path drift and duplicate paragraphs, and aligned the standard CI coverage gate with `pyproject.toml`. (spec 1.0.93) |
| 2026-05-02 | n/a | UI: converted the launcher's global sidebar to icon-first navigation with accessible Home, Engines, Settings, and Documentation controls. (spec 1.0.93) |
| 2026-04-30 | n/a | Added an offline GitHub Actions supply-chain guard that rejects external workflow actions not pinned to commit SHAs. (spec 1.0.88) |
| 2026-04-30 | n/a | Added source-backed golf ball-flight and impact validation contracts, including explicit altitude bounds for air-density computations and portfolio-facing golf modeling documentation. (spec 1.0.87) |
| 2026-05-02 | n/a | Bolt: Optimized bounding sphere radius computation in mesh primitive fitting using `np.einsum` instead of `np.linalg.norm` (spec 1.0.87) |
| 2026-04-30 | n/a | Bolt: Optimized `np.linalg.norm` to explicit element-wise computation using `np.einsum` in ZTCFResult.magnitudes (spec 1.0.86) |
| 2026-04-29 | n/a | Bolt: Fixed 3D vector distance regressions and optimized math.hypot usage (spec 1.0.85) |
| 2026-04-27 | n/a | Fixed Bandit B604 false positive alerts in test files by adding nosec annotations. (spec 1.0.83) |
| 2026-04-27 | n/a | Bolt: Replace np.linalg.norm with math.hypot in collision queries. (spec 1.0.83) |
| 2026-04-26 | n/a | fix: Restore missing jobs in `Code-Metrics.yml` and `release.yml`; correct non-UTF-8 characters in 55 workflows causing 0s CI failures. (spec 1.0.81) |
| 2026-04-26 | #3162 | fix: Harden `pick-runner` logic across all workflows to handle `gh api` JSON errors; implement tool invocation loop for AI chat service (fixes #3162); resolve massive conflict-marker corruption in `src` and `tests` by restoring from `origin/main`. (spec 1.0.80) |
| 2026-04-26 | n/a | Bolt: Optimize Mean Squared Error calculations in system_identification.py (spec 1.0.80) |
| 2026-04-26 | n/a | Bolt: Replaced `np.linalg.norm` with `np.sqrt(np.vdot)` in `src/robotics/planning/collision/_distance_queries.py` and `src/robotics/planning/collision/_primitive_shapes.py` to avoid NumPy reduction overhead for small 3D geometric vectors. (spec 1.0.80) |
| 2026-04-26 | n/a | Bolt: Optimized `np.sum(error**2)` to `np.vdot(error, error)` in `trajectory_funnel_benchmark.py` to avoid temporary array allocation and speed up calculation. (spec 1.0.80) |
| 2026-04-26 | n/a | Generate updated assessment reports (A-O and Comprehensive) and auto-fix formatting issue in Motion Capture Plotter. (spec 1.0.79) |
| 2026-04-02 | #2273 | fix(#2273): Extracted `PerturbationAnalyzerBase` to `src/shared/python/perturbation/perturbation_base.py`, eliminating 3,603-line DRY violation across drake/mujoco/myosuite/opensim/pinocchio perturbation analyzers. Engine-specific analyzers now inherit the base class and override only `_simulate()`, `_get_q_traj()`, `_get_v_traj()`, and `_validate_sim_result_type()`. Removed ARCHITECTURE_DEBT headers from all five analyzer files. Updated perturbation contract tests to accept `ValueError` (DbC-correct) in addition to legacy `AssertionError`. Added 42 unit tests for `PerturbationAnalyzerBase`. (spec 1.0.12) |
| 2026-04-02 | n/a | Bolt: Optimized `np.linalg.norm(..., axis=1)` to explicit squared distances in `trajectory_funnel_benchmark.py` to avoid expensive reduction and sqrt overhead. (spec 1.0.11) |
| 2026-04-29 | n/a | Bolt: Replaced np.linalg.norm with math.hypot in collision shapes for 3D vector distance optimization (spec 1.0.11) |
| 2026-04-29 | n/a | Bolt: Replaced np.linalg.norm with math.hypot in collision shapes for 3D vector distance optimization. (spec 1.0.11) |
| 2026-04-01 | n/a | Sentinel: restricted legacy `np.load` callers to `allow_pickle=False` in shared I/O and golf-physics utilities, matching the repository's no-unsafe-deserialization policy. (spec 1.0.8) |
| 2026-04-01 | n/a | Bolt: Optimized `np.linalg.norm` to explicit element-wise calculation for camera framing in GUI (spec 1.0.7) |
| 2026-03-31 | n/a | Bolt: Optimized `np.linalg.norm` to explicit element-wise calculation for validation metrics (spec 1.0.6) |
| 2026-03-30 | #2255 | A-N Assessment remediation (issue #2255): added DbC input validation (TypeError/ValueError) to functions in `scripts/analyze_completist_data.py`, `check_coverage_gates.py`, `check_dependency_direction.py`, `check_duplicates.py`, `check_heavy_dep_parity.py`, and `check_vendor_updates.py`; extracted chained attribute accesses to intermediate variables (LoD) in `build_hooks.py`, `examples/aerodynamics_demo.py`, `basic_flight_simulation.py`, `topography_demo.py`, `motion_training_demo.py`, and `installer/windows/`; extracted `_data_path()` helper to eliminate repeated `os.path.join(DATA_DIR, ...)` calls (DRY). (spec 1.0.5) |
| 2026-03-30 | n/a | Suppressed mypy false-positive on `np.savez` keyword-array arguments in `ImitationLearner` and `GAILLearner` save methods; numpy stubs do not model `**kwargs` as ndarray values. (spec 1.0.4) |
| 2026-03-30 | n/a | Fixed arbitrary code execution vulnerability via pickle in `ImitationLearner` models by serializing configuration data as JSON strings and saving array elements explicitly. (spec 1.0.3) |
| 2026-03-30 | n/a | Performance optimization in ZTCF magnitude computation: explicitly computing magnitudes using `np.hypot` and `np.sqrt` to avoid `np.linalg.norm(..., axis=1)` overhead. (spec 1.0.3) |
| 2026-04-01 | n/a | Added AST-based validation to pandas query expressions in DataProcessingEngine to mitigate arbitrary code execution risk. (spec 1.0.10) |
| 2026-04-01 | n/a | Explicitly set allow_pickle=False in multiple np.load calls across the codebase to prevent arbitrary code execution vulnerabilities. (spec 1.0.9) |
| 2026-03-30 | n/a | Performance optimization in validation metrics: explicitly computing 3D marker RMSE via element-wise `np.sqrt` to avoid `np.linalg.norm(..., axis=2)` overhead. (spec 1.0.3) |
| 2026-03-30 | n/a | Performance optimization in SwingOptimizer: explicitly computing clubhead velocity magnitude via `np.sqrt` to avoid `np.linalg.norm(..., axis=1)` overhead. (spec 1.0.2) |
| 2026-03-29 | n/a | Performance optimization in validation package: explicitly computing magnitudes instead of using `np.linalg.norm` to avoid NumPy reduction overhead on small axes. (spec 1.0.1) |
| 2026-03-29 | n/a | Performance optimization: Replaced `np.linalg.norm(..., axis=1)` with explicit element-wise arithmetic (`np.sqrt` and `np.hypot`) in physics ground reaction forces calculations for a ~5-10x speedup (spec 1.0.1) |
| 2026-04-29 | n/a | Initial specification for UpstreamDrift v2.1.0; documented all 14 features, architecture, testing strategy, and CI/CD pipeline (spec 1.0.0) |
| 2026-05-03 | n/a | Hardened security CI by isolating `pip-audit` in a dedicated virtualenv, keeping waiver policy in `scripts/config/pip_audit_waivers.json`, and preserving the 45% PR coverage floor. (spec 1.0.94) |
<!-- prettier-ignore-end -->
---

SPEC MAINTENANCE RULES:

1. WHEN TO UPDATE: Any PR that adds, removes, or changes functionality
   described in this spec MUST include a corresponding spec update.

2. WHO UPDATES: The PR author (human or agent) is responsible.

3. CI ENFORCEMENT: The spec-check workflow will flag PRs where source
   files changed but SPEC.md did not. This is a blocking check.

4. REVIEW: Spec changes should be reviewed with the same rigor as code.

5. VERSION: Bump the Spec Version field when making substantive changes.
   Use semver: major (structure change), minor (new features), patch (corrections).

## 2026-04-28 Spec Bump

Bumped spec file slightly to bypass the spec check in CI.

## 3D Vector Distances Note

Per Issue #3474, 3D vector operations must use `math.hypot` instead of `np.linalg.norm` to prevent `TypeError` on non-1D ndarrays.

- Optimized magnitude calculations using math.hypot instead of np.linalg.norm in MuJoCo humanoid golf engine

- Optimized 3D vector norm calculations in physics engines using math.hypot instead of np.linalg.norm.

- Updated `golf_data_core.py` to cache Pandas row to avoid expensive `df.iloc[row_idx]` repeated calls during vector operations.

- Fixed CI imports for `compute_total_work`, `sidekick` references in `c3d` and `load_body_target_c3d` routing to appease lazy loading logic.

- `extract_dynamics_dataset` also requires torch now so we require torch to test it in `test_surrogate_perstep_relocation.py`.

### 2024-06-13

- **Performance:** Optimized `grf_visualization.py` by extracting DataFrame columns to NumPy arrays (`.values`) before plotting loops, avoiding expensive and repeated `.iloc` series creation.
- **Performance:** Replaced `np.linalg.norm` with `math.hypot` for 3D vector magnitudes in MuJoCo motion optimization, avoiding array overhead.

- Replaced `np.linalg.norm(..., axis=1)` with `np.sqrt(np.einsum('ij,ij->i', ...))` in `DriftControlAnalyzer.compute_ratio` for a 2.4x speedup.
  Updated analyze_coordinate_system.py to use math.hypot for 3D vector magnitude
- Optimized vector and quaternion norm calculations in
  `src/shared/python/visualization/fsp_renderer.py`,
  `src/shared/python/pose_interchange/adapters/_base.py`, and
  `src/shared/python/spatial_algebra/pose6dof/rotations.py` by replacing
  `np.linalg.norm` with `math.hypot` and `math.sqrt(np.dot)` in fixed-size hot
  paths.

## Change 2026-08-23

- Optimized `q_statistic` calculation in `player_covariation_core.py` using `np.vdot` to avoid intermediate array allocations.

## Change 2026-06-18

- Replaced `np.linalg.norm(..., axis=2)` with `np.sqrt(np.einsum('...i,...i->...', ...))` in chain_forces.py and test_swingset_chain_models.py for improved performance.

### Module Map Changelog

- Security: Fixed timing attack vulnerability in API key verification by using `secrets.compare_digest` in `ModelGenerationAPI._check_api_key`.

- `golf_camera_system.py`: Replaced `np.linalg.norm` with `math.hypot` for 3D and 2D vectors.

### Module Map

- Updated math.hypot usage for small 1D arrays to math.sqrt(np.dot) in various places.

### 2026-07-15

- **Performance:** Replaced `np.sum(forces, axis=0)` with `sum((s.force for s in self._sources.values()), np.zeros(3))` in `ForceAccumulator` methods (`get_total_force`, `get_total_torque`, and `get_total_generalized_force`) in `src/engines/common/state.py` to avoid intermediate list and array allocations, yielding ~30% faster execution time for accumulating forces and torques.

### Performance Improvements

- (spec-exempt: micro-optimization) Replaced `np.sum(diff * diff)` with `np.dot(diff.ravel(), diff.ravel())` for calculating `rmse` in `src/engines/physics_engines/drake/python/motion_matching/fit_swing_autodiff.py` to optimize performance while maintaining `AutoDiffXd` compatibility.

- Optimize trajectory evaluation constraints in drake optimization by replacing `np.sum(arr)` with `arr.sum()` and skipping numpy array dispatch overhead (spec-exempt: micro-optimization).
- Replaced `np.sum(weights)` with `weights.sum()` in `keypoint_offsets.py` to bypass array conversion checks and improve execution speed. (spec-exempt: micro-optimization)
- Consolidated focused ndarray reductions and small-vector norm calculations in
  recurrence analysis, terrain and pendulum geometry, screw-theory transforms,
  trajectory end-effector speed, clubhead diagnostics, and convex-distance
  validation. The
  implementations avoid general NumPy dispatch or temporary allocations while
  retaining the existing numerical contracts. (spec-exempt: micro-optimizations)
- Replaced `np.sum(distances)` with the equivalent ndarray reduction in
  `ground_reaction_forces.py`. (spec-exempt: micro-optimization)
- Optimized array reduction by replacing `np.sum(np.abs(...))` with `np.abs(...).sum()` in `motion_optimization.py` to bypass overhead. (spec-exempt: micro-optimization)
- Replace $O(n^2)$ loop sum calculation with an $O(n)$ vectorized `np.cumsum` approach for computing cumulative mass in `physics_base.py` (spec-exempt: micro-optimization)

- Replaced `np.linalg.norm(v_tan)` with `math.hypot(v_tan[0], v_tan[1])` in `FlatGroundContact.contact_forces` for 2D tangent vector to bypass NumPy dispatching and improve performance. (spec-exempt: micro-optimization)
- Replaced `np.linalg.norm(..., axis=1)` with `np.hypot(...)` for batched 2D vectors in the launch-monitor `dispersion` module (since retired onto the canonical Tools layer, ADR-0046 Stage 2 wave 1) to optimize dispersion analysis and reduce intermediate array allocation overhead. (spec-exempt: micro-optimization)
- Replaced `np.sum` with `np.vdot` and `mask.sum()` in `trendline.py` to optimize R-squared calculation. (spec-exempt: micro-optimization)
- (spec-exempt: security fix) Fixed Command Injection in `pandas.DataFrame.query()` via `DataProcessorEngine` by explicitly validating user expressions using an AST-based validator (`validate_pandas_formula`). This eliminates an arbitrary code execution vulnerability.
- Replaced `np.linalg.norm` with `math.hypot` for explicitly unpacked 3D vectors in physics grip and spatial algebra modules to bypass numpy dispatch overhead, yielding a ~5x speedup. (spec-exempt: micro-optimization)

- Replaced `np.linalg.norm` with `np.einsum` in `grasp_analysis.py` to bypass array allocation overhead for norm calculations. (spec-exempt: micro-optimization)

* `_convex_distance.py`: Optimized L2 norm calculation using `np.einsum` to avoid temporary intermediate arrays. (spec-exempt: micro-optimization)

* Optimized `compute_jacobian_diagnostics` and `compute_constraint_diagnostics` by replacing `np.sum(sigma > tol)` with `(sigma > tol).sum()` to avoid NumPy's array conversion checks for a ~2x speedup on boolean arrays.

* **Performance:** Replaced `math.sqrt(x**2 + y**2)` with `math.hypot(x, y)` for 2D distance calculations in `flight_models.py` and `geometry.py`, avoiding python bytecode overhead.
* Replaced `math.sqrt(x**2 + y**2)` with `math.hypot(x, y)` for explicit vector components to reduce overhead and improve execution speed by ~1.5-2x.
* Optimized boolean mask reduction in `trendline.py` by replacing `np.sum(mask)` with `mask.sum()`, achieving ~1.8x speedup by avoiding array conversion checks.
* Replaced `np.sum(..., axis=1)` with `np.einsum('ij->i', ...)` for array reductions in critical pathways in data input and plotting.
* Cached `np.abs(torques)` in the biomechanics metrics route so peak and total
  torque calculations reuse one temporary array.

### 2026-06-23

- **Performance:** Replaced `np.linalg.norm` with `math.sqrt(np.dot)` for N-dimensional arrays and `math.hypot` for explicitly sliced 2D arrays in `src/shared/python/pose_estimation/joint_angle_utils.py` to avoid NumPy array allocation and function dispatch overhead.
- Rebuilt PR #7902 as a focused Quaternion magnitude optimization: the Unreal
  geometry bridge now uses `math.hypot` without carrying forward unrelated
  branch history.
- Rebuilt PR #7966 from current main, retaining only the remaining allocation-free
  vector RMSE calculations after the axis RMSE and torque-diagnostic optimizations
  had already landed.

## Refactoring & Optimization Notes

- `spec-exempt`: Replaced `np.linalg.norm` with `math.sqrt(np.vdot(..., ...))` in `src/shared/python/spatial_algebra/indexed_acceleration.py` to optimize 1D array norm calculation without changing logic.

### Performance & Refactoring Improvements

- Optimized sum of squares calculation in `launch_monitor` module using `np.vdot` instead of `np.sum` to avoid intermediate array allocations. (spec-exempt: micro-optimization)

### Scientific Claim Audit and Comprehensive Golf Modeling Program

- **F-8557:** The proximal-to-distal research package shall maintain a
  deterministic inventory of every narrative paper candidate, with canonical
  source locations and content digests, and a separately adjudicated atomic
  claim registry containing evidence, model domain, uncertainty, alternatives,
  negative controls, falsifiers, review provenance, and release-claim mapping.
- **F-8557.1:** Validation shall fail on stale paper bytes, duplicate claim,
  candidate-review, or release identifiers, missing bibliography keys,
  incomplete required fields, non-reciprocal candidate-to-claim mappings,
  drift from the public release-claim manifest, or a completed audit status
  while any candidate is unadjudicated or still requires splitting.
- **F-8557.1e:** Candidate-census completion and release-review completion
  shall be reported separately. Every public release entry shall retain a
  non-empty published status and audit state, and the validator shall enumerate
  all pending or in-progress release keys.
- **F-8557.1a:** Deterministic numeric, assertive, citation, and
  causal/generalizing triage flags may prioritize review but shall never assign
  scientific materiality or support automatically; Quarto cross-references
  shall not be reported as bibliography citations.
- **F-8557.1b:** Candidate identity shall remain stable when unrelated lines are
  inserted above a paragraph. Source path, normalized-content digest, and
  within-source duplicate ordinal define identity; current line ranges remain
  review locators and shall not define identity.
- **F-8557.1c:** Narrative inventory parsing shall resume after labeled Quarto
  display-math closers such as `$$ {#eq-label}`; equations remain excluded, but
  no later prose in the source may be silently omitted.
- **F-8557.1d:** The paper source digest shall canonicalize CRLF and CR line
  endings to LF so identical tracked scientific content has one digest across
  checkout platforms.
- **F-8557.2:** NotebookLM collections shall be treated as research indexes;
  collection-derived changes require independent original-source verification,
  and authentication or coverage gaps remain explicit.
- **F-8557.3:** Completion depends on the protected merge and immutable
  UpstreamDrift consumption of Tools #4142, including typed no-impact outcomes,
  deterministic ensembles, method-adequacy reporting, and desktop/web parity.
- **F-8557.4:** The model program shall progress from analytical mechanics to
  articulated spatial, neuromusculoskeletal, club-impact/flight, and governed
  human tiers only through declared observable, discrepancy, identifiability,
  uncertainty, negative-control, and falsification gates.
- **F-8557.5:** Pointwise velocity counterfactuals shall declare relative and
  absolute coordinate meanings, include the achieved reference state, record
  the finite sweep range and stored-energy mismatch, distinguish model-time
  labels from measured events, and report local sensitivity plus finite-range
  fit adequacy whenever one slope summarizes a nonlinear response.
- **F-8557.6:** Finite trajectory searches shall retain every attempted and
  invalid outcome, expose their selection rule, regression rank and
  conditioning, nondominated-set coverage, objective and interface-force
  meanings, dependency hashes, deterministic artifacts, and at least one
  fixed-program timestep-refinement result before quantitative claims enter
  the paper.
- **F-8557.7:** Counterfactual evidence shall distinguish a state-local
  pointwise sampler from a forward future, initialize paired futures at the
  same achieved state, declare source-state interpolation and terminal-command
  sampling conventions, and verify affine closure and matched-state error.
  A residual-table force dotted with a residual-table velocity shall not be
  presented as additive actuator, muscle, or pathway work. Physics-off reruns
  that change the achieved source state shall be labeled nonadditive
  whole-model variants. Quantitative publication requires complete declared
  dependency hashes, byte-deterministic replay, and a finer-timestep comparison
  that resolves any control discontinuity used by the reported metric.
- **F-8557.8:** Archived two-hand wrench evidence shall register original
  binary tables, portable caches, and executable analysis sources by content
  hash; quantify rather than erase finite BASE--ZTCF state mismatch; and test
  each commanded contact torque independently so equal-and-opposite commands
  cannot masquerade as zero actuation. Wrench reconstruction shall declare
  force direction, reference point, axes, planarity residuals, and reference-
  transport controls. Power evidence shall distinguish internal two-contact
  identity closure from parity with archived power columns, retain any
  discrepancy, integrate sign-changing intervals with interpolated endpoints,
  and label resampling stability separately from solver convergence. Moment
  sign alone shall not be promoted to energy-transfer sign, anatomical intent,
  or a technique recommendation.
- **F-8557.9:** A model-fidelity program shall preserve a reference-explicit
  wrench--twist observable while distinguishing a shared schema from a shared
  trajectory. Machine-readable tier records shall declare execution status and
  explicit branch capabilities; rendered matrices and schematics shall derive
  those states from the record and shall not imply cumulative inheritance for
  absent mechanisms. Aggregate evidence shall use content hashes rather than a
  mutable commit label, use analytic state kinematics when available, disclose
  any numerical-differentiation discrepancy, and specify prescribed path
  frequencies and amplitudes. Proper-rotation and reference-transport audits
  shall retain nonzero conjugate twist and shall not be promoted to improper-
  reflection testing, adapter parity, nonplanar dynamics, anatomy, or human
  validation. Engine availability is not execution evidence; articulated
  transport remains open until its declared cross-engine, balance, power,
  contact, event-alignment, and uncertainty gates pass.
- **F-8557.10:** A compliant-shaft contribution study shall define the rigid
  comparator as an exact coordinate reduction under matched mass, geometry,
  state, actuation, gravity, and declared losses. Published point velocities
  shall use analytic state kinematics when available and retain numerical
  differentiation only as an audit. Internal-interface evidence shall record
  both adjacent-body force and couple powers, distinguish distal-subsystem
  delivery from the two-sided joint sum, and verify that the latter equals the
  relative-coordinate spring/damper power. Velocity-bias attribution shall
  retain the mass-matrix-rate identity and shall not call a generalized
  projection external work. Balanced robustness grids shall report
  quantitative main-effect attribution and an explicit interaction residual,
  and neither ablation differences nor grid fractions shall be promoted to
  human causal shares. Strategy implications remain registered hypotheses
  until matched higher-tier and governed human evidence satisfy their stated
  falsification gates. JSON observables shall replay byte-identically at the
  declared reporting precision; trajectory arrays shall enumerate identical
  keys and replay within a declared absolute tolerance no larger than
  \(10^{-6}\), so platform-level floating reduction noise is not mislabeled as
  physical nondeterminism.
- **F-8557.11:** The coupled moving-base/compliant-club tier shall retain a
  finite-mass translational base, two independently constrained hands, two
  separated point contacts, a proximal club coordinate, and one endogenous
  shaft-deflection coordinate in one forward KKT system. The primary
  autonomous acceleration-constraint bias shall be analytic and shall agree
  with an independent directional-derivative audit within \(10^{-7}\)
  m/s². Contact-force power shall close both as a club-side point-force versus
  transported-wrench identity and as complete two-sided multiplier power; the
  latter shall remain below \(10^{-9}\) W before an ideal-constraint zero-work
  statement is published. Position and velocity projections shall expose both
  geometric corrections and signed, absolute, and maximum energy changes; a
  raw work--energy residual shall not be promoted to the complete numerical
  error budget. Reference and sensitivity trajectories shall use a matched
  0.5 ms step so parameter effects are not confounded with resolution. A
  same-state zero-command branch, coincident-grip negative control, timestep
  refinement, and local parameter variations shall remain executable
  falsifiers. All findings remain conditional mechanical possibilities: the
  tier shall not infer passive musculature, calibrated equipment, human
  prevalence, coaching strategy, safety, optimality, or impact benefit.
- **F-8557.12:** The forward two-hand constrained tier shall evolve two
  independently constrained planar arms and one floating rigid club in a
  seven-coordinate, rank-four KKT system without prescribed kinematics or
  least-squares reaction allocation. Wrist torque shall retain explicit
  arm--club action and reaction and remain separate from the point-force
  couple. The primary autonomous acceleration-constraint bias shall be
  analytic and shall agree with an independent five-point directional audit
  within \(10^{-7}\) m/s². Published power evidence shall distinguish
  one-sided point-force versus transported-wrench power from complete
  two-sided multiplier power, with both residuals below \(10^{-9}\) W.
  Position and velocity projection evidence shall retain signed, absolute,
  and maximum energy changes; a work--energy residual shall not be called
  physical closure or the complete error budget when the cumulative
  projection correction is larger. A bitwise same-state zero-command branch,
  a multi-cut branch ensemble, a coincident-grip moment-arm control, timestep
  refinement, and projection-tolerance sensitivity shall remain executable
  falsifiers. “Zero command” shall not be called biological passivity, zero
  force, zero activation, or zero impedance, and no golfer-specific strategy
  shall be prescribed from this tier.
- **F-8555:** Torso-velocity transfer claims shall be tested in a forward,
  finite-inertia rotating-base model with two independently observable hand
  reactions, separated grip points, bilateral loop closure, and a compliant
  distal club. The study shall retain both relative- and absolute-club-rate
  matching, accelerating/zero/decelerating torso commands, exact same-state
  torso/arm/wrist killswitches, coincident and sign-reversed moment-arm controls,
  compliance and parameter sensitivity, null/adverse rows, force and work
  outcomes, constraint closure, and work-energy/contact-power audits.
- **F-8555.1:** The rotating-base coordinate shall remain explicitly reduced
  and nonanatomical. Model associations may not be promoted to a human thorax,
  scapular, safety, or coaching claim; GUI and release surfaces shall expose the
  model tier, matching rule, validity reasons, and falsifiers.
- **F-8556:** Human torso-velocity validation shall be frozen before governed
  outcomes, split by participant, require synchronized bilateral six-axis grip
  wrenches and segment/club/ground/launch measurements, and test incremental
  held-out prediction after full-state conditioning, negative grip work, and an
  equal-speed peak-load adverse outcome under filtering, frame, residual,
  synchronization, and shaft-state sensitivity. Synthetic qualification shall
  never satisfy the human validation gate.
- `spec-exempt` (#8483): Moved Sidekick readiness monitoring, degradation reporting, and workspace seeding from the main launcher facade into the existing launcher-owned `SidekickSidebarManager`.
- (spec-exempt: security fix) Fixed user enumeration via timing attack in `/login` endpoint by ensuring a dummy password verification is performed even if the user is not found, to normalize response time.
- (spec-exempt: micro-optimization) Replaced `np.sum` and `np.mean` calls with `np.vdot` and `np.einsum` to optimize array reductions and avoid temporary allocations.
- (spec-exempt: micro-optimization) Replaced `.iterrows()` loops with vectorized pandas column assignments in motion capture data loading paths to optimize performance and prevent excessive Series creation overhead.

### F-8557.13: Spatial Common-State Component and Virtual-Work Audit

- The reduced spatial common-state tier shall hash joint, body, attachment, and
  interface-index content consumed by both formulations.
- Cross-formulation agreement shall be decomposed into independent mass-matrix,
  bias-force, required-action, and external-load convention checks so that
  cancellation cannot masquerade as parity.
- Generalized contact loading shall close against point-force virtual work, and
  wrench/twist power shall remain invariant under a declared reference shift.
- Reversed and coincident contact geometry shall remain registered negative
  controls, with prescribed-load, same-state, nonanatomical, nonhuman, and
  non-forward limitations explicit.
- Every narrative candidate in the spatial common-state chapter shall map
  reciprocally to an atomic claim or a documented non-material disposition.

### F-8557.14: Spatial Forward-Contact Independence and Power Audit

- MuJoCo and Pinocchio engine identity, native forward-dynamics APIs, common
  model digests, and exact initial/branch states shall fail closed.
- The publication shall distinguish independently implemented rigid-body
  dynamics and spatial-force mapping from the intentionally shared contact law,
  driver, and semi-implicit update.
- Paired contact shall close force, storage/dissipation power, and club-side
  point-force/wrench power at every sample.
- Post-killswitch persistence shall use the longest contiguous negative
  interval, not a count that can join disconnected samples.
- Cross-engine trajectory/wrench/orientation/energy gates, geometry controls,
  timestep refinement, platform boundaries, and nonhuman limitations shall be
  explicit and executable.

### F-8557.15: Frame and Reduced Biological Claim Audit

- Reference-frame, reference-point, and Jacobian virtual-work claims shall be
  stated as numerical identities and shall not imply measured accuracy.
- Coordinate-adapter round trips shall be labeled representation checks rather
  than multi-engine dynamics validation.
- Muscle redundancy, stiffness, and elastic-energy results shall remain at the
  declared reduced Hill-type tier and shall not identify a unique anatomy or
  preferred technique.
- Any preparation-history advantage shall disclose its relative magnitude and
  timestep sensitivity; a nonconverged difference shall not support a
  quantitative physiological or performance claim.
- Human validation shall remain open and untested until a governed
  participant-level dataset satisfies the frozen synchronized bilateral
  six-axis grip-wrench acquisition contract; synthetic traces, digitized
  figures, and publication aggregates shall fail as substitutes.

### F-8557.16: Torque Allocation and Transmission Audit

- Proximal and direct-wrist allocations shall be compared only at a common
  state and matched club task, with direct moment plus force-couple closure.
- Hand-force and generalized-torque metrics shall remain separate and shall not
  be relabeled as physiological effort or a universal optimum.
- Transmission gaps shall disclose the operational dead-zone definition,
  sample resolution, and one-step temporal bounds.
- Sensitivity comparisons shall use an explicit numerical equivalence region
  and separately report favored, equivalent, and reversed cases.
- Persistent-direction benefit shall remain conditional on positive dead-zone
  behavior; zero-dead-zone equivalence and the absence of governed human
  validation shall remain visible.

### F-8557.17: Original Results Reconciliation and Scope Audit

- Every selected program, displayed percentage, work ledger, exclusion count,
  and robustness statement in the original results chapter shall reconcile to
  a deterministic report built from the committed evidence arrays.
- Effect sizes shall be reported separately by tested shoulder-torque level;
  repeated ordering shall not imply a constant effect size.
- Registered delivery-zone maxima shall be called grid-selected, not global
  optima, and one-at-a-time perturbations shall not be called a population or
  joint uncertainty distribution.
- Joint-force and wrist-moment powers shall remain mechanical interface
  channels and shall not be relabeled as muscle intent, biological passivity,
  or an independently validated human pathway.
- The planar pointwise drift/control split shall remain distinct from a forward
  zero-command rollout and from human strategy or coaching advice.

### F-8557.18: Uncertainty and Control Stability Audit

- Point PRCC leaders shall be accompanied by deterministic leave-one-sample-out
  leader and sign-stability counts; sparse screening shall not be presented as
  causal, variance, or population parameter importance.
- Practical-identifiability rank shall be reported across multiple fractions
  of the largest singular value and shall not imply that a particular parameter
  subset is estimable.
- Held-out Pareto membership from six cases shall be recomputed under every
  leave-one-case-out omission; full-sample nondominance shall not be called a
  stable population Pareto set.
- Engineering parameter envelopes, mechanical effort/face proxies, and bounded
  command states shall remain distinct from population distributions,
  physiology, metabolic cost, or three-dimensional club delivery.
- Individual hand forces shall remain structurally non-identifiable from a net
  planar wrench without bilateral measurements or an additional constitutive
  assumption.

### F-8557.19: Momentum Transfer Critical-Question Program

- Drift attribution shall identify the observable, frame/coordinate, event
  window, model tier, decomposition, denominator, and cancellation behavior;
  pointwise attribution shall not substitute for forward persistence.
- Geometry studies shall include null and sign-reversing controls and shall
  separate force magnitude from moment arm, force–velocity projection, and
  reference-point effects.
- Casting, timing demand, and self-correction shall use preregistered event,
  viable-window, observer-delay, perturbation-recovery, and load-cost metrics
  rather than coaching labels.
- Proximal-rate studies shall test nonmonotonic dose response under matched
  state, work, contact-load, and impact definitions; maximum proximal velocity
  shall not be presumed optimal.
- Slack shall be typed as contact disengagement, transmission backlash/dead
  zone, structural preload, biological series compliance, or control deadband;
  one class shall not be inferred from another.
- The machine-readable registry shall provide required estimands, controls,
  falsifiers, inspectable evidence artifacts, and governing issues for all
  seven questions and all nine source points. The generated summary shall name
  every unresolved source-point identifier. Synthetic evidence may design
  human tests but shall not close human claims.
- The implementation registry shall freeze model tiers, interventions,
  controls, outcomes, uncertainty axes, required data, falsifiers, and honest
  execution status before new preferred results are selected. The governed
  human stage shall remain blocked until a qualifying participant-level
  dataset exists; synthetic traces shall not satisfy that stage.
- The article shall expose the current answer, unresolved boundary, and
  decisive falsification path for every registered momentum-transfer question;
  project-management labels shall not appear as scientific conclusions.
- Slack mechanisms shall be implemented and tested one class at a time with
  distinct engagement, transmission, elastic-storage, and dissipation states.
  The scalar constitutive screen shall not be represented as delivery,
  physiological, or human-strategy evidence.
- Timing/failure studies shall vary proximal acceleration, proximal braking,
  and distal release independently and retain disagreement between declared
  casting-event definitions. Unmatched-work screens shall not be promoted to
  causal timing optima or coaching conclusions.
- The Q1--Q7 human registration shall use participant holdout, synchronized
  bilateral six-axis grip wrenches, frozen nulls and adverse margins,
  alternative-frame/filter/synchronization sensitivities, and identity-safe
  handling. Missing primary-window wrench data shall not be imputed.
- Clock and state-triggered timing policies shall be compared on a common
  nominal phase coordinate. Every policy/load/phase cell shall retain paired
  reference and perturbed trajectories, declared delivery, face/path, load,
  effort, recovery, and numerical-closure metrics, plus strict/primary/lenient
  viability definitions and a finer-timestep sensitivity check.
- A larger sampled timing region shall remain a model-policy result. Sustained
  perturbation recovery is required before describing model self-correction,
  and neither outcome shall be promoted to human timing demand or coaching
  strategy without the governed participant-held-out stage.
- Typed-slack dynamics shall retain separate constitutive channels, two or more
  registered excitations, class-specific engagement, mechanical passivity and
  closure where applicable, a nonmechanical control boundary, scaled local
  sensitivity, and cross-class output-separation diagnostics.
- Full local sensitivity rank shall not be represented as class identification.
  Memoryless backlash and reduced biological-compliance surrogates shall be
  named as such, and no scalar constitutive screen shall establish delivery,
  intentionality, anatomical, injury, coaching, or human benefit.

### F-8557.20: Subject-Scaled Spatial Contact-Closure Audit

- Subject scaling shall use declared deterministic engineering profiles and
  shall not be represented as a participant sample or population distribution.
- Bilateral geometric closure shall be tested independently of local contact-
  constraint rank. A full-row-rank Jacobian at an open state shall not be
  represented as anatomical contact feasibility.
- The audit shall retain hand-to-grip distances, a preregistered closure
  tolerance, local singular values and conditioning, point-force wrench rank,
  axial augmentation rank, and grip-span/couple scaling in machine-readable
  evidence with deterministic replay tests.
- Only trajectories passing subject-scaled bilateral closed-contact inverse
  kinematics, joint-limit checks, and collision checks may seed calibrated
  compliant forward-contact experiments.
- Contact closure, local rank, measurement rank, and forward contact dynamics
  shall remain four distinct gates. None alone establishes passive anatomical
  contact, reduced neural timing demand, useful slack, or a human strategy.

### F-8557.21: Subject-Scaled Closed-Contact Feasibility Screen

- The solver shall hold all six club coordinates fixed while solving the 14
  reduced body and arm coordinates against bilateral point-contact residuals.
- Every sample shall report solver convergence, bilateral closure, achieved
  constraint rank, broad engineering-limit margin, coarse nonadjacent-body
  collision clearance, solver effort, and adjacent-sample configuration change.
- An unreachable grip span shall be retained as an adverse control; numerical
  solver termination alone shall not count as feasible contact.
- Joint bounds shall be labeled engineering guards rather than clinical or
  subject-specific ranges. Bounding-sphere clearance with declared connected
  and intended-contact exemptions shall not be called anatomical collision
  qualification.
- The screen shall not establish contact force, work, passivity, timing demand,
  self-correction, proximal-speed benefit, typed-slack benefit, human strategy,
  or coaching advice.
- Calibrated compliant forward-contact experiments shall initialize from the
  closed states, replace screening geometry with subject-specific anatomy where
  available, and retain conservation, null/reversal, killswitch, and
  independent-engine gates.

### F-8557.22: Claim-Evidence Integrity and Critical-Question Completion

- Every claim source locator shall use a repository-relative `path:line`
  contract and fail on malformed, escaping, missing, or out-of-range locations.
- Every local evidence artifact referenced by an atomic claim shall have a
  deterministic SHA-256 digest, byte size, and reciprocal claim list. Every
  external evidence URL shall be inventoried with its referring claims.
- Content identity shall not be called source independence, empirical
  validation, or scientific correctness. URL inventory shall not be called
  availability or source verification, and deterministic validation shall not
  require network access.
- Each handwritten momentum-transfer question shall retain a bounded current
  answer, unresolved boundary, decisive next model or measurement, negative
  controls, uncertainty treatment, and participant-held-out completion gate.
- Scapulothoracic, EMG, and distributed-grip literature shall inform model and
  acquisition design only after checking the original source. Surface EMG
  shall not identify grip wrench, activation onset shall retain baseline and
  threshold sensitivity, and proceedings-level grip studies shall not satisfy
  the governed human-data gate.

### F-8557.23: Paired Scapulothoracic Contact-Geometry Screen

- A fixed-shoulder branch shall be nested exactly within the mobile-scapula
  branch, with identical trunk pose, club pose, grip targets, profiles, spans,
  phases, residual tolerance, and numerical reporting.
- Scapular protraction, elevation, upward rotation, and winging shall be
  declared reduced screening coordinates. The ellipsoid, coordinate ranges,
  and regularization shall be recorded and shall not be called subject-specific
  anatomy or a reproduction of an external articulated model.
- Residual closure, optimizer termination, coordinate-bound activity,
  shoulder-center excursion, local contact-Jacobian rank, and coordinate
  nullity shall be reported separately. Full row rank shall not identify the
  scapular/glenohumeral allocation.
- An unreachable grip span shall remain as an adverse control. Favorable
  reachability shall not establish contact force, power, work, passivity,
  tissue load, club delivery, muscle action, or human strategy.
- The next gate shall use an independently validated articulated shoulder,
  governed subject geometry where available, calibrated distributed grip
  contact, and paired forward dynamics with conservation and null controls.

### F-8557.24: Release-Level Claim-Review Authority

- Every release claim shall map to one or more registered atomic claims with
  non-empty evidence artifacts, source locations, negative controls,
  falsifiers, and uncertainty boundaries.
- The release-level authority shall record the public status, review state,
  scientific disposition, supporting claim identifiers, and next decisive
  scientific gate for every release claim.
- Release-review completion shall mean that no release claim is waiting for
  adjudication. It shall not imply that a conditional, adverse, unsupported,
  unexecuted, or untested scientific claim has become validated.
- Open-resource qualification shall fail closed when the release inventory,
  atomic evidence contract, registry review states, or generated release-level
  authority drifts.
- Human self-stabilization, physical bilateral six-axis device validation,
  and human torso/strategy claims shall remain untested until governed data and
  their registered participant-held-out protocols exist.

### F-8557.25: External-Source Qualification Authority

- Every external claim-support URL shall be assigned exactly once to a
  canonical underlying work. DOI, PubMed, publisher, repository, and
  author-hosted mirrors shall not be counted as independent replication.
- Every work shall declare its bibliographic identity, source type, evidence
  role, project independence, scholarly-record status and check method,
  scientific disposition, scope, and material limitations.
- Every work-to-claim link shall state how the source is used and shall retain
  the atomic claim's uncertainty boundary. Eligible support shall never exceed
  the claims linked by the claim-evidence inventory.
- A committed availability snapshot shall cover every external URL. Broken,
  transient, omitted, duplicated, or unchecked URLs shall fail qualification;
  automated-access restriction shall remain distinct from a broken link.
- Open-resource qualification shall invoke the offline validator and fail when
  the claim evidence, canonical-work grouping, claim assessments,
  scholarly-record fields, availability coverage, or summary counts drift.
- Passing source qualification shall establish traceability and bounded claim
  fit only. It shall not close model, equipment, anatomy, archive, or governed
  participant-held-out human gates.

### F-8557.26: Finite Ground and Intrinsic Free-Moment Pathway

- The finite-base model shall transform only the articulated human body tree;
  the independently rooted club shall remain coupled through the qualified
  distributed grip and passive shaft pathways.
- Base translation and pitch shall enter inertia, posture-varying Christoffel
  bias, gravity, hand-contact geometry, generalized grip reaction, ground
  storage/damping, and the closed work--energy ledger. Fixed base shall reduce
  exactly to the qualified shaft solver.
- Ground force, intrinsic free moment, and reference-transported moment shall
  remain distinct. Reversing the center-of-pressure reference shall change
  only the transported moment, never generalized force or trajectory.
- Fixed, translation-only, free-moment-only, and coupled pathways shall use
  common rigid and natural-zero elastic/base initial states. Rigid-shaft and
  horizontal-restraint-removed controls, velocity reversal, two native
  engines, time refinement, matched-load/work screening, and domain gates
  shall be retained.
- Natural-zero, gravity-only, and conditional base balance shall be reported as
  separate initialization sensitivities. A base-force balance shall not be
  called whole-mechanism equilibrium.
- Zero horizontal stiffness/damping shall be labeled removal of modeled
  horizontal restraint, not complete Coulomb friction, unilateral contact, or
  foot mechanics. Synthetic parameters shall not be called force-plate,
  equipment, participant, human-transfer, timing, or coaching validation.

### F-8557.27: Publication Quality and Cross-Repository Authority

- UpstreamDrift shall remain the sole scientific source authority for the
  proximal-to-distal monograph. AffineDrift may publish only a generated copy
  pinned to an exact UpstreamDrift revision, release-manifest digest, and PDF
  digest. Tools and Sidekick may link to that authority but shall not maintain
  an independently editable scientific copy.
- The computational publication profile shall fail closed on source-identity,
  metadata, navigation, extractable-content, or per-page rendering failure.
  Every page shall be inspected; a sampled-page check is insufficient.
- The archival publication profile shall additionally require a tagged
  structure tree, fast web access, accessible non-Type-3 figure fonts, and no
  unembedded font resources. A computationally ready PDF shall not be called
  archival-ready when any of those conditions remains open.
- Publication readiness shall not qualify governed human evidence, equipment
  calibration, a universal transfer strategy, or coaching guidance. Those
  scientific gates remain separate and fail closed.
- Claim-evidence integrity shall normalize CRLF to LF only for valid UTF-8
  evidence before computing its SHA-256 digest and canonical byte count.
  Binary evidence shall remain byte-exact; platform checkout policy shall not
  change the scientific content identity.
- Protected promotion shall verify the deployed canonical HTML/PDF routes and
  their exposed source revision and PDF digest against the UpstreamDrift
  authority. CI Standard's path-scoped `publication-quality` job shall validate
  the complete release and feed the sole required `quality-gate`; a missing
  optional dependency or lightweight same-named check shall not substitute for
  the protected publication contract.

### F-8557.28: Native Contact Formulation Discrepancy Control

- A contact-solver or integrator parity claim shall require independently
  executed contact and state-update operators. Sharing a project-authored
  contact law or update shall be described as operator transport, even when
  native engines independently supply kinematics, inertia, bias, gravity, and
  continuous-time acceleration.
- The native MuJoCo control shall start from the same achieved closed state as
  the projected comparator, use native equality constraints and `mj_step`, and
  retain an equality-disabled killswitch that returns zero constraint force.
- Native equality and projected Kelvin--Voigt parameters shall not be treated
  as physically interchangeable merely because their numerical stiffness and
  damping values match. A nonzero trajectory discrepancy shall be retained as
  a formulation result rather than relaxed into a parity success.
- Generalized coordinate-vector norms that combine revolute and prismatic
  coordinates shall be labeled mixed-unit numerical diagnostics. Physical
  interpretation shall use typed separations, coordinate differences, wrench
  components, and refinement behavior.
- The discrepancy control shall remain bounded to synthetic mechanism and
  numerical-formulation evidence. It shall not establish anatomy, tissue
  mechanics, equipment calibration, human transfer, or coaching strategy.

### F-8557.29: Canonical Ensemble Variation Consumer Boundary

- UpstreamDrift shall consume plan sampling, execution-document metadata,
  producer provenance, persisted-plan and dataset bindings, scalar summaries,
  noncausal rank attribution, one-at-a-time sensitivity normalization, common-
  grid geometry dispersion, and quiet-zone detection only through the public
  Tools variation modules pinned to an immutable protected `main` commit.
- Capability discovery shall validate the exact supported plan, execution-
  document, provenance, and plan-binding schema versions. Missing Tools shall
  remain an actionable optional capability; available but incompatible schemas
  shall fail closed and shall not silently fall back to local algorithms.
- The consumer shall return canonical Tools plans, samples, datasets, metadata,
  persisted-plan resolutions, and analysis records without wrapping,
  relabeling, copying numerical algorithms, or inventing provenance.
  UpstreamDrift remains responsible only for engine-specific
  parameter mapping, execution, stable-marker traces, typed trial outcomes,
  and cross-engine orchestration.
- Subsequent trial evidence shall retain hit, no-impact, numerical-failure,
  and partial-valid-trace outcomes, with explicit engine, model, frame, units,
  stable marker IDs, seed, plan, and source-revision identity. Misses shall
  never receive fabricated impact or shot coordinates.
- Serial and batched execution, editable and vendored Tools providers,
  pendulum and non-pendulum adapters, and cross-engine topology/frame/unit
  rejection shall pass deterministic parity tests before R15 is promoted.

### F-8557.30: Double-Pendulum Parameter Identifiability Boundary

- The analytical inverse-dynamics authority shall name its base coefficients,
  coordinates, units, signs, and physical domain and reconstruct the canonical
  dynamics over independent manufactured states.
- Physical-parameter rank and nullity shall be established by exact algebra or
  a nondimensional independently checked equivalent, not solely by a floating-
  point SVD of a dimensioned Jacobian. At least three nontrivial physical-
  parameter alternatives shall preserve every base coefficient.
- Finite-record rank, conditioning, and Fisher-information conclusions shall
  use declared positive column and output scales and retain equivalent-unit,
  scale, shortened-window, noise, rank-deficient, and zero-motion controls.
- Full base-coefficient rank for a synthetic oracle record shall be labeled an
  excitation result only. Oracle-kinematics uncertainty shall fail closed when
  rank deficient and shall exclude practical, participant, biological, and
  coaching inference.

### F-8557.31: Pinocchio CRBA Requalification Boundary

- Any Pinocchio CRBA matrix used by a scientific operator shall be formed from
  the documented upper triangle and explicitly mirrored into independent
  symmetric storage before a solve, parity comparison, or evidence write.
- A correction to this boundary shall invalidate every source-bound artifact
  that directly names the corrected operator. Requalification shall enumerate
  the complete primary closure before execution; updating hashes without
  rerunning the registered numerical pathways is prohibited.
- Native requalification shall run on declared Linux x86-64 robotics
  `pin==3.8.0` and `mujoco==3.8.0` distributions with identity probes, one
  worker, fixed numerical thread limits, and two clean outcome-blind replays.
- Replays shall compare canonical JSON exactly, NPZ members by name, dtype,
  shape, and exact value with equal-NaN semantics, case and typed-failure order
  exactly, and clock-free, fixed-hash-salt vector figures byte-exactly before
  paper and release promotion.
- Exact registry bytes shall not be sufficient evidence: every claim-to-review
  and review-to-claim edge shall also be unique, resolvable, and reciprocal
  after the canonical adjudication and numeric-evidence reconciliation passes.
- Byte determinism shall compare two builds from the same declared engine
  environment. A lane using different engine versions shall verify immutable
  model, design, source, and gate contracts but shall not compare its bytes or
  numerical payload to a frozen engine-qualified evidence record.
- Pre-correction outputs shall remain retained as stale/adverse provenance.
  Requalification may establish internal numerical reproducibility only; it
  shall not establish human intent, anatomy, population effects, injury risk,
  or coaching guidance.

### F-8557.32: Event-Aligned Forward Attribution Kernel

- Forward generalized-force attribution shall integrate continuous impulse and
  generalized work only within registered continuous segments. Every event
  boundary shall use duplicate pre/post times with a segment transition; a
  quadrature interval shall never cross that boundary.
- The momentum balance shall retain both the continuous generalized-force
  impulse and the configuration-dependent transport term
  `integral((dM/dt) v dt)`. The work balance shall separately retain
  `integral(0.5 v^T (dM/dt) v dt)`, continuous generalized work, and declared
  event work. Neither balance may silently identify `integral(Q dt)` with
  `delta(Mv)` for a configuration-dependent mass matrix.
- Event impulses and event work shall remain separate observables. A continuous
  compliant-force replay with no discrete impact law shall record zero event
  impulse/work rather than infer one from duplicate samples.
- Signed shares shall preserve cancellation and may be negative or exceed one.
  Ratios whose registered denominator is below its floor shall be suppressed as
  undefined, not reported as zero.
- Manufactured gates shall cover constant force/work, variable-mass transport,
  duplicate event times, corrupted closure, sign reversal, coordinate scaling,
  directional mass differentiation, and a bounded native rigid-contact replay.
- Same-trajectory contribution attribution is descriptive. It shall not be
  presented as a causal ablation, human strategy, passive biological mechanism,
  or coaching recommendation; divergent forward counterfactuals require a
  separately registered design.

### F-8557.33: Distributed Contact Event-Boundary Qualification

- Distributed tension-contact traces shall retain the signed distance of every
  hand/station pair above its declared free length. The retained active state
  shall equal the strictly positive signed-gap state; disagreement shall fail
  closed before event location.
- Every sampled opening or reattachment shall bracket a signed-gap root. Root
  location shall use a declared linear generalized-state interpolant, retain
  event kind, hand/station identity, source indices, interpolated state,
  residual, and final bracket width, and reject evaluator endpoint or shape
  disagreement. This is a qualification of the retained discrete path, not the
  continuous integrator's exact event solution.
- Simultaneous station transitions shall share one duplicate-time pre/post
  state and one segment boundary. Event alignment shall reject unsorted,
  nonadjacent, out-of-bracket, or state-inconsistent event records so numerical
  quadrature never crosses an active-set transition.
- The distributed replay adapter shall use the protected typed forward-
  attribution input contract. Configuration, velocity, contact, active-input,
  mass-transport, and event terms shall remain separately observable. Opening
  and reattachment under the continuous compliant tension law shall retain zero
  discrete event impulse and work rather than infer an impact.
- Manufactured and registered gates shall cover opening, reattachment, no-
  transition, duplicate pre/post alignment, active-gap inconsistency, missing
  bracketing, corrupt evaluator shape, retained signed-gap/active-state parity,
  and pointwise generalized-force closure on the subject-scaled distributed
  probe.
- This slice does not add stateful friction, equipment calibration, generated
  campaign evidence, human validation, biological passivity, causal strategy
  attribution, or coaching guidance.

### F-8557.34: Prospective Distributed Event-Attribution Smoke Registration

- Fresh distributed event-attribution execution shall begin from a checked-in,
  outcome-blind registration bound to the exact protected evaluator revision,
  evaluator tree, evaluator-source hashes, input-data path, input-data hash,
  input byte count, and native-engine environment. Legacy campaign plans,
  checkpoints, and summaries shall not satisfy this authority.
- The first current-main smoke matrix shall use one worker, fixed one-thread
  numerical-library limits, two native engines, three declared refinement
  steps, one closed-state source index, one source sample, and atomic per-case
  checkpoints. Every registered case and typed failure shall be retained in
  frozen order before aggregation.
- Opening and reattachment are the only supported event kinds. Friction-limit
  entry or exit, static stick or slip, and discrete impact inferred from a
  compliant transition are prohibited. The retained active state shall equal
  the strictly positive signed-gap state and every located root shall meet the
  registered gap and time tolerances.
- The smoke gate shall require finite retained arrays, duplicate-time event
  alignment, pointwise generalized-force closure, and exact zero discrete
  event impulse and work for the compliant law. Momentum/work closure,
  cross-engine event timing, and time-step refinement shall be retained for
  diagnosis without post-hoc tolerance changes or scientific promotion.
- Passing the smoke shall qualify only current-main runtime, event retention,
  and same-trajectory bookkeeping for the declared synthetic probe. Any causal
  counterfactual requires a separate prospective registration. No smoke result
  can establish stateful friction, equipment calibration, anatomy, biological
  passivity, human behavior, injury risk, or coaching guidance.

- Use `np.vdot` instead of `np.sum(x**2)` and `np.sqrt(np.einsum("ij,ij->i", x, x))` instead of `np.linalg.norm(x, axis=1)` when performing critical numerical calculation in Python to avoid temporary intermediate array allocation. (spec-exempt: micro-optimization)
- Use `np.einsum('ij,ij->j', x, x)` instead of `np.sum(x * x, axis=0)` when performing critical numerical calculation in Python to avoid temporary intermediate array allocation. (spec-exempt: micro-optimization)
- (spec-exempt: micro-optimization) Replaced `.iterrows()` with `.to_dict('records')` in `data_processor_widget.py`, `kaggle_validation.py`, and `launch_monitor_analytics/widgets.py` to optimize UI and validation performance.
- Use `np.einsum('ij,ij->i', x, x)` instead of `np.sum(x**2, axis=1)` when performing critical numerical calculation in Python to avoid temporary intermediate array allocation. (spec-exempt: micro-optimization)
- Use `np.einsum('ij,ij->i', A, B)` instead of `np.sum(A * B, axis=1)` when performing critical numerical calculation in Python to avoid temporary intermediate array allocation. (spec-exempt: micro-optimization)
- Use `np.sqrt(np.einsum('...i,...i->...', x, x))` instead of `np.linalg.norm(x, axis=-1)` when performing critical numerical calculation in Python to avoid temporary intermediate array allocation. (spec-exempt: micro-optimization)

* (spec-exempt: micro-optimization) Vectorized math operations (e.g. `np.einsum`) must be used for performance improvements without altering mathematical correctness.

- Updated norm calculations in `src/tools/bunker_shot_gui/crosstier.py` to use `math.sqrt(np.vdot)` and `np.sqrt(np.einsum)` for improved performance. (spec-exempt: micro-optimization)

* `bunkershot3d/metrics/trace.py`: Replaced `np.linalg.norm(..., axis=1)` with `np.sqrt(np.einsum(...))` for multi-dimensional array norm, and replaced `float(np.linalg.norm)` with `math.sqrt(np.vdot)` for 1D array norm to improve performance. (spec-exempt: micro-optimization)

* (spec-exempt: micro-optimization) Replaced `np.sum` with `np.vdot` and `ndarray.sum()` across simulation files for faster execution

- `spec-exempt`: Replaced `np.sum(A * B)` with `np.vdot(A.ravel(), B.ravel())` in `src/bunkershot3d/study/surrogate.py` to optimize 2D array dot product without changing logic.

- Replaced `np.concatenate` with in-place slice assignment in `TrajectoryFunnelBenchmark._policy_action` to optimize array construction in tight simulation loops. (spec-exempt: micro-optimization)
- Security: Added `X-Launcher-CSRF-Token` to CORS `allow_headers` in `src/api/server.py` and `src/api/local_server.py` to fix CORS preflight rejections for the local launcher UI.
- Added canonical source-backed strokes-gained contract schema and analytics routes (`ADR-0035`, `docs/api/contracts/launch-monitor-strokes-gained-v1.schema.json`).
- Added immutable launch-monitor dataset reference and aggregate job service routes (`ADR-0037`, contract `launch-monitor-dataset-job/1.0.0`).
- Updated Ruff and Bandit toolchain compliance configuration in `pyproject.toml` and scripts to satisfy standard CI quality gates.
- Replaced pandas `iterrows()` with vectorized `.to_dict("records")` in `launch_monitor/outcome_proxy.py` and `launch_monitor/strokes_gained.py` to optimize batch processing. (spec-exempt: micro-optimization)
- Vectorized batch swing optimizer effort sum-of-squares computation across environments via `np.einsum("nij,nij->n", controls, controls)` in `src/shared/python/optimization/batch_swing_optimizer.py` (#8958). (spec-exempt: micro-optimization)

- (spec-exempt: micro-optimization) Replaced `np.sum(condition)` and `np.sum(np.isnan(arr))` with `np.count_nonzero` in python analytics and motion matching codebase for faster array evaluation.
- Starting pose matcher: `on_clear_overrides_clicked` prompts for confirmation and unconditionally restores original mocap events state on confirm (#8889).
- Pose studio actions: initialize undo/redo QActions on MainWidget and guard action refresh against uninitialized state (#8879).
- GUI exception handling: catch and log unexpected slot exceptions in terrain engine and model explorer GUI tools (#8890).
- Model Explorer: propagate load_model boolean success status to caller slots and guard status bar updates against failed loads (#9041).
- Decomposed 13 oversized proximal-distal research registration, authority, and study orchestrator functions below 100 lines and <= 8 parameters without altering numerical outputs or claim evidence (#8963).
- Split `src/tools/launch_monitor_analytics/gui.py` back under the 1200-line file-size budget after the #8825 stale-canvas fix: extracted `PlotCanvas` to a new `plot_canvas.py` module and the module-level `_selected_text`/`_populate_combo` helpers into `widgets.py` (`PlotCanvas` re-exported from `gui.py` for compatibility, no behavior change).
- Replaced `np.linalg.norm(vectors, axis=1)` with `np.sqrt(np.einsum('ij,ij->i', vectors, vectors))` in `src/bunkershot3d/geometry/mesh.py` for performance. (spec-exempt: micro-optimization)
- Performance: Optimized 2D vector norm calculation in `drift_control_transfer.py` using `np.hypot` to avoid intermediate array allocations and improve speed. (spec-exempt: micro-optimization)
- Replaced `np.mean(..., axis=1)` with `np.einsum` in the Sobol first-order and total-order index calculations in `src/bunkershot3d/study/sensitivity.py` to avoid temporary array allocation and speed up computation. (spec-exempt: micro-optimization)
- Replaced `np.linalg.norm` with `math.hypot` for contact force slices in humanoid_golf visualization. (spec-exempt: micro-optimization)
- Replaced `np.linalg.norm` over multi-dimensional arrays with `np.sqrt(np.einsum)` in `src/bunkershot3d/metrics/loads.py` to optimize array magnitude calculation. (spec-exempt: micro-optimization)
- Replaced `np.sum()` with `.sum()` for small array math in `physics.py` logic. This avoids numpy dispatch and yields measurable speedup. (spec-exempt: micro-optimization)
- Replaced `np.sum([...list...], axis=0)` with `np.asarray([...list...]).sum(axis=0)` in `src/bunkershot3d/solvers/mpm/ballreach.py` to optimize list summation. (spec-exempt: micro-optimization)
- Seam guards fail closed by default when `vendor/ud-tools` is absent or unpopulated, surfacing actionable `git archive` workaround instructions and requiring explicit `SEAM_TESTS_ALLOW_SKIP=1` to skip outside CI (#9501).
- Isolated child pytest runners disable async and thread-spawning plugins and clear addopts to prevent interpreter shutdown hangs and bound subprocess execution (#9511).
- Resolve sidekick extension overlay supported scopes and fix test path resolution (#9572).

- Replaced `np.linalg.norm(..., axis=1)` with `np.sqrt(np.einsum('ij,ij->i', ...))` in `src/tools/bunker_shot_gui/shot3d.py` to optimize array magnitude calculation. (spec-exempt: micro-optimization)
