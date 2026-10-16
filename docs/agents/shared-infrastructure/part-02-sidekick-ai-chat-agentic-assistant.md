# Shared Infrastructure Directory (Part 2/2)

Part of the [Shared Infrastructure Directory](../shared-infrastructure.md) index.

### Sidekick (AI Chat / Agentic Assistant)

Sidekick is the cross-shell AI chat surface. Two host implementations
consume one shared design-token contract:

- **React/Tauri shell:** [`ui/src/pages/Chat.tsx`](../../../ui/src/pages/Chat.tsx)
  routes at `/chat`; the panel itself is
  [`ui/src/components/ui/ChatPanel.tsx`](../../../ui/src/components/ui/ChatPanel.tsx)
  and binds its surface palette to `var(--sidekick-color-*)` /
  `var(--sidekick-space-*)` CSS variables (declared in
  [`ui/src/index.css`](../../../ui/src/index.css)).
- **PyQt launcher panel:**
  [`src/shared/python/ai/gui/assistant_panel.py`](../../../src/shared/python/ai/gui/assistant_panel.py)
  (`AIAssistantPanel`, window title "Sidekick"). The launcher
  embeds it both as a splitter pane
  (`src/launchers/launcher_ui_setup.py`) and as a registered
  [embeddable tool](#launcher-embedding--cross-tool-ipc) so users can
  open it via right-click → "Launch in Tab" / "Launch in Dock".

Shared infrastructure:

- **Design tokens:**
  [`src/shared/python/theme/sidekick_tokens.py`](../../../src/shared/python/theme/sidekick_tokens.py)
  maps active launcher theme colors onto canonical `sidekick.color.*` /
  `sidekick.space.*` / `sidekick.radius.*` / `sidekick.font.*` keys. The
  TypeScript mirror is
  [`ui/src/api/themeClient.ts`](../../../ui/src/api/themeClient.ts); both maps are
  pinned in lock-step by
  [`tests/unit/theme/test_sidekick_parity.py`](../../../tests/unit/theme/test_sidekick_parity.py).
- **Embeddable adapter:**
  [`src/tools/sidekick/_embed_adapter.py`](../../../src/tools/sidekick/_embed_adapter.py).
  Self-registers via
  [`src/launchers/embedded_tool_bootstrap.py`](../../../src/launchers/embedded_tool_bootstrap.py)
  and exposes the tile through
  [`src/config/models.yaml`](../../../src/config/models.yaml).
- **Chat context bridge:**
  [`src/shared/python/ai/chat_context.py`](../../../src/shared/python/ai/chat_context.py)
  — thread-safe ring buffer (`record_event`, `get_chat_context`) with
  redaction of `password` / `token` / `secret` / `api_key` / `/home/` /
  `C:\` patterns and a 4 KB dump cap. The WebSocket handler in
  [`src/api/routes/chat_ws.py`](../../../src/api/routes/chat_ws.py) injects the
  payload as a `system` message when populated; gate with
  `UPSTREAMDRIFT_SIDEKICK_CONTEXT=0` to disable.
- **Agentic tools:** new analytical surfaces register through the existing
  AI tool registry. The current cross-engine example is
  [`src/shared/python/ai/tools/sidekick_analytics.py`](../../../src/shared/python/ai/tools/sidekick_analytics.py)
  (`summarize_simulation_run`). The system prompt in
  [`src/shared/python/ai/system_prompts.py`](../../../src/shared/python/ai/system_prompts.py)
  advertises registered tools to the assistant.
- **Agent action layer (epic [#5967](https://github.com/D-sorganization/UpstreamDrift/issues/5967), ADR-0017):**
  [`src/shared/python/sidekick/agent/`](../../../src/shared/python/sidekick/agent)
  — the single audited choke-point for every agentic action. New
  Sidekick actions register through `SidekickActionService`; the
  planner translates LLM tool calls into validated `PlannedStep`s; an
  access policy gates writes and destructive actions; an audit sink
  records every call. Host integrations (launcher, Pose Studio, ...)
  implement `HostActionPort` — sidekick never imports them. See
  [`docs/sidekick/agent.md`](../../sidekick/agent.md) for the worked
  example of adding a new action.

When extending Sidekick — adding a tool the assistant can call,
extending the chat context bridge, or restyling the panel — reuse these
surfaces rather than forking new color/spacing constants or new event
buses. For anything the assistant should _do_ (not just compute),
register a new `ActionDescriptor` via `SidekickActionService` rather
than wiring a one-off `ai.tools.*` module.

**If you need a standalone Sidekick app, use `sidekick.standalone.*`; do not
write a new shell.** The standalone package
([`vendor/ud-tools/src/shared/python/sidekick/standalone/`](../../../vendor/ud-tools/src/shared/python/sidekick/standalone))
provides:

- `sidekick.standalone.runner` — headless `sidekick run --calculator <name>`
  dispatcher (no GUI or display required).
- `sidekick.standalone.preferences` — typed preference surface backed by an
  injectable `SessionStore`.
- `sidekick.standalone.onboarding` — first-run sentinel + 3-step state machine.
- `sidekick.standalone.session_store` — `InMemorySessionStore` (tests) and
  `FileSessionStore` (production).

These modules are canonically owned by Tools. Never recreate or edit them under
`src/shared/python/sidekick/`; make changes in Tools, merge them, and repin the
exact `vendor/ud-tools` gitlink. UpstreamDrift-only Sidekick extensions must be
classified in
[`scripts/config/shared_python_ownership_exceptions.yaml`](../../../scripts/config/shared_python_ownership_exceptions.yaml);
startup exposes only those exact module names and fails closed on unclassified,
unresolved, or newly conflicting files.

See [`docs/sidekick/standalone.md`](../../sidekick/standalone.md) for the user-
facing guide and [ADR-0018](../../adr/0018-standalone-sidekick.md) for design
rationale (issues #5984, #5985, #5986, #5987).

### Rust Kernels

- `rust_core/upstream-physics/` — RK4 integrator, aerodynamics, contact,
  swing-plane fit. Built via `maturin develop`; consumed by Python
  via `src/shared/python/physics/rust_kernel.py` (which falls back to
  pure Python if the Rust wheel isn't installed). See section F for
  when to write Rust.

## Shared Model and Video Analysis

- `motion_capture.coaching.reference_distances` uses registered sampling for
  metric point/plane readouts; missing motion remains unavailable.
- `motion_capture.coaching.native_geometry.NativeGeometryRenderer` submits
  scene-bound reference meshes through the existing `fsp_renderer.Viewport`
  protocol. The world frame is explicit. Pose Studio supplies the qualified
  native adapter; use shared geometry rather than another plane generator.
- `tools.capture_rig.geometry_controls.GeometryControls` edits the same
  immutable document in comparison, model analysis and Pose Studio.
- `motion_capture.reference.trace_import` reads bounded Trace v2 marker motion
  into the existing reference importer. See the Trace metadata contract in
  `docs/motion_capture/reference_model_fitting.md`; do not infer FK from q-only
  traces or duplicate the shared Trace reader.

The [Shared Analysis Contracts](../../architecture/SHARED_ANALYSIS_CONTRACTS.md)
record units, frames, ownership, failure behavior and tests across these providers
and consumers. Use the existing capability registry as the product map authority.

## Native Simscape Marker Frames

The MATLAB `motion_matching/shared/build_golf_kinematics.m` factory resolves the loaded golf model's coordinate identities and body frames. `intrinsic_xyz_to_rotm.m` converts its intrinsic XYZ rotation outputs into world-from-body matrices; `project_body_markers.m` applies fixed body-local marker offsets in metres. Use these with native solved states and the current geometry snapshot. Kinematic solves do not establish forward-dynamics agreement. `simulate_with_coefficients.m` remains the sole production forward-simulation entry point. The frame schema deliberately leaves Hip's direct log source empty because `HipGlobalPosition` measures the fixed joint base.

`configure_capture_velocity_targets.m` returns a cleanup guard that holds the native independent velocity priorities for a fit session. Keep it alive across all forward replays, then clear it to release Fast Restart and restore target settings. Native tests cover repeated cold/warm replays and exception cleanup. Per-call SimulationInput mask overrides are not a substitute: they failed restoration during Fast Restart. No source model is saved, and simulate_with_coefficients remains the sole simulation entry point.

`golf_marker_velocity_map.m` maps the schema's independent native rates to all joint rates and marker-major XYZ velocities for fixed body offsets. It uses explicit SI units and the native [frame-velocity conventions](https://www.mathworks.com/help/sm/ref/simscape.multibody.kinematicssolver.addframevariables.html): linear in base/world axes, angular in follower/body axes. Its angular column is tested against native pose differencing. Position and velocity seeding share `dependent_coordinates` in the native schema. Final forward initialization must separately satisfy the requested rates.

`fit_golf_pose_seed.m` fits native frame proxies or optional fixed body-local marker offsets with fixed geometry and a weak pose prior. Repeated frame names support multiple attachments; it reuses the native rotation conversion and marker projector. Its local optimum is not a global kinematic lower bound. It resets the supplied KinematicsSolver roles and requires final native targets and constraints to hold. Its explicit Euclidean proxy RMS avoids confusion with per-coordinate RMS. This is pose seeding only; geometry identification and final forward dynamics require separate evidence.

`simulate_golf_markers.m` connects the sole native forward wrapper to fixed marker attachments. It maps original logged joint states through the qualified kinematic snapshot before interpolating Cartesian positions; canonical interpolated joint angles must not be used as closed-chain poses. The shared Python `prefix_fit.fit_prefixes` accepts an optional relative finite-difference step for solver-resolved sensitivity calculations.

`src/engines/Simscape_Multibody_Models/python/tour_fit_state.py::qualified_fit_identity` validates and copies a qualified native SI q/qd state, inch geometry and fixed attachments for a capture and prefix horizon. The checkpoint reader requires exact declared identity equality. The runner verifies the state again in native forward dynamics before fitting.

`tour_fit_state.py::transfer_prefix_candidate` reuses a completed torque candidate for a fresh longer-prefix objective after verifying fixed native state/attachments and effort units. The shared `bernstein_to_simscape` converter accepts 1–7 control values per joint and retains a seven-coefficient native output, allowing constant/linear profiles before unlocking higher degrees. Polynomial transfer supports degree zero through six: De Casteljau interval transformation followed by degree elevation preserves the existing physical-time curve, with extrapolated control bounds checked before simulation. Native higher-degree fitting still needs qualification. `prefix_fit.py::bernstein_effort_range` computes numerical endpoint/interior-stationary extrema for reporting actual profile ranges separately from Bernstein control bounds. `motion_matching/rigidity.py::rigid_attachment_residuals` reuses the existing Kabsch helper to quantify a conditional fixed-attachment residual floor with independent body poses; it must never replace native forward replay. `enable_revolute_actuator_log.m` adds the native revolute effort sensor without replacing existing signals; the instrumented reference supports all 27 actuator channels. `audit_saved_golf_candidate` requires 27 by default, with explicit 22-channel legacy mode.

`audit_golf_actuator_torques.m` compares native polynomial commands against the 16 explicit scalar ActuatorTorque logs on their own timestamps. It reports the remaining coordinates as unlogged, rejects missing logs, and never substitutes reaction forces/torques. Use it on retained raw output to qualify the observable actuator paths.

The root Bushing Joint also exposes six actuator-sensing channels under HipLogs.TranslationForce[XYZ]Input and HipTorque[XYZ]Input. Despite the legacy suffix, these are sensor outputs. Translation polynomial commands are world-resolved; sensed forces are in the fixed hip base and require transpose(Rx(PlaneTilt)) before comparison. `audit_golf_actuator_torques` schema 2 requires this explicit rotation and separates N from Nm. The source trace is in the tour-matching evidence directory; the [Bushing Joint sensing reference](https://www.mathworks.com/help/sm/ref/bushingjoint.html) distinguishes primitive actuator outputs from composite reactions.

`select_capture_marker_frame.m` selects validity-flagged finite observations in requested label order from the tour capture payload and returns attachment indices. It distinguishes a valid origin from missing zero placeholders. `attached_pose_probe.m` uses it for connected-pose diagnostics.

`pose_estimation.estimate_keypoint_offset` is reused by the Simscape `calibrate_pose_attachments.py` reproduction command for confidence-weighted fixed local marker attachments from verified native body frames. Preserve its proper-rotation checks, covariance/scatter reporting and valid-observation selection; fitted attachment candidates require fresh forward-state qualification.

`tour_fit_state.verify_native_initial_state` requires declared q/qd and native kinematic marker projection to agree with a forward replay, while reporting nonzero capture residual independently. This allows calibrated geometry/attachments without falsely requiring perfect initial marker fit or relaxing physical-state verification.

## Historical Research Fit Jobs

The optional `motion_matching.pipeline.MarkerLinearizationPlant` capability
creates a `MarkerLinearizer` returning immutable `MarkerLinearization` world
points and native-coordinate derivatives. It does not extend the required
`MatchingPlant` protocol. Historical fitting combines this owner-provided
derivative with `CameraProjection.project_jacobian`; unsupported providers
retain finite differences and malformed advertised providers fail closed.
See [Analytic Marker Derivatives](../../development/necromatcher-analytic-marker-derivatives.md).

The workspace facade exports `NativeRefitOptions` and `start_native_refit` for
immutable source-bound historical refits. Reuse `motion_matching.jobs` for owned
job execution, cancellation, manifests and recovery. `MatchingWorkOutcome`
retains explicit rejection after computational success and serializes publication
against cancellation. The native worker uses canonical image evidence and Hermite
MAP fitting with immutable warm-start samples; source PTS and unqualified camera
assumptions cannot certify physical derivatives or joint efforts. See the
[Necromatcher Native Fitting Procedure](../../development/necromatcher-native-fit.md).

The canonical estimation facade exports `initialize_authored_hermite` and its immutable receipt DTOs. Historical fitting exports `initialize_image_trajectory` for explicit seed authoring without optimization; `fit_image_trajectory` shares preparation/result assembly while preserving original priors and pixels. Reuse `NativeFitBinding.authored_coordinate_bounds()` for captured definition/XML/unit-bound authored radian ranges; compiled joint limits remain disabled. See [Authored Initialization](../../development/necromatcher-authored-initialization.md) and [Authored Ranges](../../development/necromatcher-authored-ranges.md).

`historical_fit.ImageSplineStart` and the optional exact start on `fit_image_trajectory` preserve physical Hermite coefficients, native model/order identities and the original knot clock. Reuse `workspace.necromatcher_spline.preserved_fit_spline` for SDK-free plan/worker recall, and `refit_plan.baseline_config` for complete canonical recipes across both hosts. See [Saved-Spline Restart](../../development/necromatcher-saved-spline-restart.md).

### Repository Python Worker Environments

`core.repo_python_environment(repo_root, base=None)` is the public, pure
repository-worker environment builder shared by Capture Rig and Necromatcher.
Run module commands with cwd set to repo_root. It prioritizes repo_root/src for
bare package imports, preserves SDK/runtime settings and does not mutate the
parent mapping. Regression tests cover absent PYTHONPATH, duplicate source
roots, idempotence and real native workers launched by a fresh Qt parent.

### Historical Research Refit Controls

workspace.NativeRefitSession owns bounded live handles over MatchingJobService
and reopens canonical run manifests for history. submit/view/cancel/close are
shared by the native modeless dialog and local-client HTTP routes. refit_plan
returns verified coordinate/sample choices and recorded priors. It does not
invent missing scales. project_store.validate_workspace_id exposes the existing
identity contract for early admission before native work. UI request types are
generated from the strict RefitRequest schema.

### Historical Research Video Captions

`workspace.CaptionOverlayOptions` is the SDK-free, versioned display recipe
owned by `workspace/necromatcher_caption.py`. Pass its optional `caption_overlay`
keyword to `workspace.export_fit_video` or `workspace.NativeVideoSession.submit`
to reuse measured compact research captions in the existing exporter/worker.
Omitted options preserve legacy captions; enabled manifests authenticate source
dimensions, rational PTS, qualification and complete semantic/layout records.
See [Compact Research Captions](../../development/necromatcher-compact-captions.md).

### Historical Authored Generalized Efforts

workspace.AuthoredEffortProfile and NecromatcherLibrary.load_effort_profile
preserve exact model/fit bindings and ordered N/N\*m channels for declared m/rad
coordinates. Import and export reuse canonical PiecewisePolynomialTorque with
strict JSON validation and bounded finite evaluation. Native units and source
physical timing remain separate qualification requirements. See the
[Authored Effort Procedure](../../development/necromatcher-effort-profiles.md).

### Compiled Historical Model Resources

workspace.load_native_fit_binding returns one NativeFitBinding over the canonical
MatchingPlant factory. MuJoCo exposes compiled coordinate_units through the
optional ScalarCoordinateUnits protocol, with slide m and hinge rad units in
named order. Projection and native refitting reuse this resource; efforts checks
exact model/fit hashes, ordered units and authored timing before native command
mapping. Scientific acceptance remains independent. See the
[Native Resource Procedure](../../development/necromatcher-native-resources.md).

### Bounded Hermite Coordinate Domain

`src.shared.python.estimation.HermiteBoundsDomain` represents a conservative C1 whole-segment bound through Bernstein controls. `MapEstimatorOptions.trajectory_domain` preserves physical callback/saved coefficient contracts while the existing single-trial MAP solver optimizes bounded internal positions/slopes. `ImageFitConfig.coordinate_bounds` binds immutable named native limits; unsupported names or infeasible seeds fail before solving. Reuse this domain instead of clipping stored trajectories or introducing another fitting scheduler. Active branch ties use an explicitly documented generalized derivative; bounds do not qualify nonlinear grip/contact or dynamics. See [Hermite Bounds Procedure](../../development/necromatcher-hermite-bounds.md).

### Authored Source Contact Phases

Reuse `historical_fit.ContactPinSchedule` / `ScheduledConstraintOptions` for exact source-clock phase-dependent normal-height pins. Legacy native options, stable residual rows and unilateral nonpenetration remain canonical. `workspace.necromatcher_contacts.contact_schedule_binding` validates capture/hash, exact original boundary PTS and reviewed decoded-frame hashes before native optimization. Phase assumptions do not infer no-slip contact or physical forces. See [Contact Phase Procedure](../../development/necromatcher-contact-phases.md).

### Explicit Fitting Coordinates and Stored Overlay Recall

Reuse public `historical_fit.expand_image_spline_coordinates` and `SplineCoordinateExpansion` to expand a checked Hermite decision selection without changing geometry or old coefficients. Added coordinates remain constant before optimization, with a detached reference-pose receipt and changed decision identity. Reuse `workspace.build_native_fit_payload` for canonical worker/derived-seed serialization; expansion-only records cannot inherit optimized/converged metadata. Strict initializer and native authored-range extraction remain authoritative. See [Coordinate Expansion Procedure](../../development/necromatcher-coordinate-expansion.md).

`NativeVideoSession.stored_runs(fit_id)` and `view_for_fit(fit_id, run_id)` reuse canonical status and current parent binding checks for durable native recall. The dialog selector submits no new job. Guarded downloads/transfer remain shared; successful research exports remain rejected. Unreadable unpublished run directories are logged and omitted, while selected binding corruption is rejected. No full video hashing is added to polling.

### Raw Shaft Fragment Derivative Diagnostics

Reuse the curated `motion_matching.historical_fit.FragmentDerivativeOptions`,
`FragmentDerivativeCheck`, `FragmentDerivativeAssessment`,
`assess_fragment_derivatives` and `optional_fragment_derivatives`. The pure core
uses canonical `CameraProjection` and public `MarkerLinearizer` data; no native
state, FK, solver or renderer is duplicated. Results are immutable raw px/rad
checks with strict order/units and explicit unavailable/bound-limited nulls.
Source/model/clock authentication remains in canonical workspace admission.
See [Procedure](../../development/necromatcher-shaft-fragment-diagnostics.md).

### Authenticated Replay Impact and Local Research Golf

Reuse public `workspace.NativeImpactSession` for bounded authored replay-impact
jobs, verified saved-run recall and checked four-file exports. The public
`workspace.load_replay_impact_receipt` authenticates companion state against
exact retained trajectory bytes; portable receipt loading does not freshly
authenticate Library parents. `workspace.load_research_impact_shot` performs
Library admission and returns an immutable `ResearchImpactShot` with exact saved
velocity/spin, parent/receipt hashes, assumptions and authored-clock limits.
Declared proper aim rotations use the existing spatial transformation owner.

`GolfSessionService.prepare_research_shot` explicitly admits unverified
`MODEL_CONTACT` only to the actual connected `LocalReferenceAdapter`. Reuse its
ordinary arm, cancel and submit lifecycle and `get_local_trajectory_record` for
detached confirmed local samples. Ordinary qualified-contact admission and
external delivery gates remain unchanged. This creates a new local-default
flight, separate from the original retained impact flight and environment;
local session results are not claimed as restart-persistent. Native avatar,
course feedback, physical source time and historical qualification remain open.

Both hosts reuse these owners; future players should supply persistent Library
records and explicit declarations rather than another physics or lifecycle
implementation. See [Replay Impact Procedure](../../development/necromatcher-replay-impact.md)
and [Local Research Golf Procedure](../../development/necromatcher-local-research-golf.md).

### Golf View Presets and Bundle Overlay Provider

`golf_view_presets` holds the four golf camera presets (face-on, down-the-line,
overhead, oblique) in the Z-up spec frame (golfer faces -X, target line -Y) with
adapters for MuJoCo, Drake MeshCat, meshcat-python (Rx(-90 deg) scene transform)
and the simbody visualizer. `force_overlay.bundle_provider` builds
ForceTorqueFrame series from same-input bundles (efforts to joint-torque arcs,
the shared contact law to per-foot GRF at the CoP, weight at the CoM); the
MuJoCo `overlay_source` supplies its contact and kinematics. Epic #11673.

### Lift Pack Parity Audit

`src/shared/python/lifting/pack_audit/` loads the OpenSim, MuJoCo, Drake and Pinocchio lift model packs behind one `EngineAdapter` (canonical frame, FK, CoM, closure), and reduces the receipt to tables and gap rules (`analysis.py`, `gaps.py`, `report.py`). Reuse it for same-input lift parity (LIFT-2 onward) instead of writing another per-engine loader. Entry point: `scripts/lifting/run_pack_parity_baseline.py`; record: `docs/development/lifting/PACK_PARITY_BASELINE.md`.

`src/shared/python/lifting/barbell_bushing.py` (`BarbellBushingPlant`, `Barbell`) wraps any same-input `VectorPlant` with a rigid barbell held at both hands by the OSV-7 grip bushing (shared law, stiffness and modal damping design), coupled two-way through each engine's own kinematics; `BushingSample.grip_analysis()` gives the GCV-7 per-hand reduction. `barbell_runs.py` holds a pose, builds a loaded bundle and measures cross-engine parity. Record: `docs/development/lifting/BARBELL_BUSHING.md` (LIFT-4, #11744).

## Native OpenSim Marker Geometry

`tour_matching/native_marker_geometry.py` supplies explicit native body-attached
frame poses and station positions for `OpensimMatchingPlant` and existing shared
calibration/trajectory IK. It validates actual native assembly and all source
coordinate ranges; it does not supply dynamics or grip closure. Reuse the existing
`full_body_ik.solve_full_body_ik_trajectory` optional finite `coordinate_bounds`
for bounded TRF. See canonical chapter26 and #11903; do not reintroduce metadata
placeholders or revive the retired OpenSim IK backend.

## Segmental Spine

- `src/shared/python/spine/` — segmental spine kinematics and loads: per-level ISB ZXY transforms, chain composition, coupling laws and intervertebral load contract. Conventions per [ADR-0055](../../adr/0055-segmental-spine-kinematics-and-loads.md); research reference `docs/research/segmental_spine/segmental_spine.tex`. Couplers prescribe kinematics only.
