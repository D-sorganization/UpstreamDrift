# Shared Infrastructure Directory (Part 1/2)

Part of the [Shared Infrastructure Directory](../shared-infrastructure.md) index.

### Native Camera and Model Hypothesis Admission

Reuse `workspace.NativeHypothesisRequest`, `bind_native_hypothesis` and
`author_native_hypothesis` for explicit, source-authenticated unoptimized seed
rebinding. `NativeModelBinding` verifies registered XML, native coordinate order
and compiled scalar units without a candidate fit. Capture identities preserve
separate encoded PNG/decoded BGR domains and the existing complete-clock digest.
The fixed two-operation clean worker reuses refit cancellation/cleanup; candidate
initialization and stored lineage never qualify camera, anatomy or physical time.
See [Native Hypothesis Procedure](../../development/necromatcher-native-hypothesis-admission.md).

Grouped by concern, with one-line descriptions and the public symbols
worth knowing. Not exhaustive — when in doubt, grep — but these are
the modules most often missed.

### Engine Abstraction

`src/shared/python/engine_core/`

- `interfaces.PhysicsEngine` — runtime-checkable Protocol.
- `sub_protocols` — focused mixins (`Loadable`, `Steppable`, `Queryable`,
  `DynamicsComputable`, `Recordable`).
- `engine_registry`, `engine_manager`, `engine_loaders` — plugin
  discovery, lifecycle, deferred imports.
- `capabilities` — feature-capability declarations per engine.
- `mock_engine` — fallback for headless / CI tests.

### Simulation Backends (GPU-Ready, Backend-Agnostic)

`src/shared/python/simulation_backends/` — interchangeable physics backends
for the golf double-pendulum behind one Protocol. See
[ADR 0023](../../adr/0023-mujoco-warp-backend.md) and the package README.

- `protocol.SimulationBackend` — runtime-checkable Protocol every backend
  satisfies. `EquationsOfMotionProvider` (mass matrix / bias forces) and
  `BatchedBackend` (parallel `rollout_batch`) are **segregated** optional
  Protocols — `isinstance`-check the exact capability you need.
- `protocol.{SimState, Trace, BatchTrace, BackendCapabilities}` — the shared
  state/output/capability types. `Trace`/`BatchTrace` are the **one** rollout
  schema across all backends (HDF5 I/O in `trace_io`).
- `model_params.GolfModelParams` — **the single source of truth** (pydantic).
  **One model, many renderers:** it renders _both_ the analytical EOM params
  (`to_double_pendulum_parameters()`) _and_ the MuJoCo MJCF
  (`mjcf.params_to_mjcf`). **Never hand-edit a derived representation** — change
  `GolfModelParams` and let both renderers follow. A regression test
  (`test_foundation.py`, task M2.3) fails if they drift apart.
- `factory.make_backend(name, params, **kw)` — the only constructor. Names:
  `"ode"` (CPU reference + dynamics), `"mujoco"` (CPU + dynamics primitives),
  `"mjwarp"` (GPU, batched, optional `[warp]` extra). Backends are imported
  **lazily**, so importing the package never needs a GPU.
- `capabilities.{has_warp, has_mujoco, require_warp, require_mujoco}` — guarded
  optional-dependency checks; the suite runs fully on CPU with zero GPU.
- `validation` — cross-backend gate (mass-matrix, bias, trajectory, energy).
  **All cross-checks are tolerance-based** (`np.allclose`), never `==`: GPU and
  CPU never bit-match (FMA, non-associative reductions, float32).
- `ztcf_zvcf` — ZTCF/ZVCF reproduced via MuJoCo/analytical dynamics primitives.
  These are **pointwise/instantaneous** decompositions sampled along the
  measured trajectory — _not_ forward-integrated. Do not "fix" them into a time
  integration (see the `# AGENT-NOTE:` in that module).

### Shot Pattern Experiments

`src/tools/shot_pattern_analysis/` exposes `AnalysisConfig`, `run_analysis`, and
`export_analysis` for paired straight/draw/fade Monte Carlo studies. Reuse its
`ShotPhysics` adapter to the existing impact and Rust flight solvers; do not
copy flight equations. See the [Experiment Reference](../../research/shot_pattern_analysis/README.md)
for frames, aiming, uncertainty and the simplified-impact limitation.

The matched-delivery extension exposes `MatchedConfig`, `ExperimentConfig`,
`run_matched_analysis`, and `execute_and_export` from its named modules.
Reuse `club_geometry.club_delivery`, `noise.NoiseSpec`, and the shared
`physics.incremental_contact.solve_incremental_contact` rather than copying
rotation, covariance, or impulse logic. The [Matched Study Design](../../research/shot_pattern_analysis/phase2/DESIGN.md)
documents raw-target matching, energetic restitution, conditional populations,
and empirical limitations. The full-tensor Newton impulse solver remains an
explicit comparator.

### Tools Ground-Model Consumer Boundary

`src/shared/python/ground_model/` is UpstreamDrift's thin, headless gateway to
the canonical `shared.python.swing_sim.ground` façade owned by Tools. It checks
the exact request, result, and reference-execution schema versions before
binding parsers or execution. Keep physics and wire records in Tools; build
UpstreamDrift API, PyQt, and React presenters on this gateway without copying
the solver or relabeling provenance.

### Data Processing Provider Boundary

`src/shared/python/sidekick_data_processing_bridge.py` binds legacy Sidekick
core and exception imports to the pinned Tools provider. Use the public
`sidekick.data_processing` exports; the [boundary reference](../../development/data_processing_provider_boundary.md)
records class identity, calculation behavior, and reproduction checks.

### Signal Provider Boundary

Use the pinned Tools `signal_toolkit` public exports through the canonical
shared namespace. UpstreamDrift's split seam retains its biomechanical
`signal_processing.py` extension; the [boundary reference](../../development/signal_provider_boundary.md)
records import identity, fitting compatibility and reproduction checks.

### Motion Matching (The Big One)

`src/shared/python/motion_matching/`

- `target.py`, `club_target.py`, `load_club_target.py`,
  `loaders/` — mocap target structures and loaders (xlsx / C3D / JSON).
- `BodyTarget`, `ClubBallTarget`, `MultiSourceTarget` —
  `src/shared/python/motion_matching/`. Frozen dataclasses + an
  aggregator covering club, ball-aware, and full-body capture
  targets. Cost-function code dispatches on `has_club()`,
  `has_ball()`, `has_body()`. See
  [ADR 0006](../../adr/0006-multi-source-motion-targets.md).
- `load_body_target`, `load_club_target` — format-agnostic dispatcher
  loaders in `src/shared/python/motion_matching/`. Route on file
  extension to the per-format loader under `loaders/`.
- `default_body_segments` — helper returning the canonical full-body
  segment label set; use it instead of hard-coding segment names in
  cost terms or visualisations.
- `align_to_simulation_grid.py` — re-time mocap onto a sim grid.
- `cost.py`, `final_cost.py` — cost terms and aggregators.
- `validators.py`, `validate_theta.py` — DbC checks for inputs.
- `metrics.py` — RMSE, peak-velocity match, etc.
- `plot_trajectory_overlay.py`, `plot_error_timecourse.py`,
  `plot_fit_quality_card.py` — canonical fit visualisations.
- **`diagnostics/`** ← this subpackage is the most often missed.
  - `forward_kinematics.forward_kinematics(angles)` — minimal Python FK
    (pelvis → spine → torso → shoulders → elbows → wrists → hands →
    butt → clubhead). Takes a Simulink-Parameter-style angle dict
    (degrees) and returns a `SkeletonPose` (Cartesian metres).
  - `reference_pose.reference_golfer_setup()` — canonical Address-pose
    joint angles. Single source of truth.
  - `reference_pose.compare_to_reference(angles)` — flag joint angles
    outside the plausible Address range.
  - `_skeleton_render.{draw_segments, draw_delta_arrows, equalize_3d_axes}`
    — matplotlib helpers for 3D skeleton overlays. **Use these instead
    of hand-rolling matplotlib boilerplate.**
  - `clubhead_trace.{compare_clubhead_traces, plot_3d_overlay,
plot_setup_pose_skeletons}` — canonical clubhead-trace comparison.
  - `initial_state_diff.{plot_skeleton_overlay, plot_per_joint_delta_bars,
plot_cartesian_delta_summary, summarize_for_pr_comment}` —
    input-MAT requested-vs-resolved diagnostics.

### Cross-Engine Pose Interchange

`src/shared/python/pose_interchange/`

- `CanonicalPose` — frozen dataclass; pelvis SE(3) + joint angles in
  the canonical convention (intrinsic XYZ Euler in degrees, joint names
  matching `reference_golfer_setup`). See ADR 0012. **For full dynamic
  state (q, v, a + quaternion floating base), see the `canonical-v2`
  contract — [ADR-0026](../../adr/0026-canonical-dynamic-state-v2.md) /
  [`docs/conventions/canonical-v2.md`](../../conventions/canonical-v2.md)
  (Canonical Core EPIC #6772).**
- `PoseConventionAdapter` — runtime-checkable Protocol; one
  implementation per engine in `adapters/`.
- `LiveKinematicsService` — Protocol; one implementation per engine in
  `services/`. Falls back to `MockKinematicsService` when the engine
  wheel is absent.
- `pose_io` — save/load to engine-native initial-state files and to
  `BodyTarget` motion-matching JSON.
- User guides:
  [`docs/user_guide/pose_studio/quickstart.md`](../../user_guide/pose_studio/quickstart.md),
  [`docs/user_guide/pose_studio/cross_engine_conventions.md`](../../user_guide/pose_studio/cross_engine_conventions.md),
  [`docs/user_guide/pose_studio/save_formats.md`](../../user_guide/pose_studio/save_formats.md).

### Body-Part Visualisation Toolkit

`src/shared/python/body_part_viz/`

- `body_part_viz` package with shapes / fitters / renderers / asset
  library — the **canonical shape stack** for any tool that draws body
  segments. See [ADR 0008](../../adr/0008-body-part-viz-toolkit.md).
- `BodyPartShape`, `ShapeFitter`, `ShapeRenderer` — runtime-checkable
  Protocols. Implementations live under `shapes/`, `fitters/`,
  `renderers/`.
- `MatplotlibRenderer` — **canonical 3D renderer for any new tool that
  needs marker / mesh rendering.** A `PyQtGLRenderer` ships alongside
  for tools that need GPU-rate redraws; both implement the same
  `ShapeRenderer` Protocol.
- `default_body_segments` (in `motion_matching/`) — canonical full-body
  segment label set; pair with this toolkit to drive segment lists in
  the C3D Viewer, the matcher, and the URDF generator.
- `SegmentVizSet` / `SegmentVizSpec` — JSON v2 persistence with
  auto-migration from the legacy v1 `SegmentSet`.
- `ShapeLibrary` — bundled mesh resolver under
  `assets/body_part_shapes/default/`; named shapes (head, torso,
  upper_arm, …) are available from a fresh install.
- `urdf_bridge.shape_to_urdf_visual` — re-use the same shape vocabulary
  as URDF visual elements; a custom mesh imported in the C3D Viewer is
  re-usable as a URDF visual link without re-modelling.
- `renderers.projective_renderer`: `SurfaceMesh`, `SurfaceLayer`, `render_surface_layer` and `composite_surface` project shared shape triangles
  through an explicitly supplied saved camera, with near-plane clipping,
  perspective-correct nearest depth and checked opacity. Use this for
  source-footage registration rather than a free-view camera screenshot.
- `ShapeOverlayOptions` validates finite opacity in [0,1]; `shapes.BoxShape`
  joins the existing capsule/ellipsoid primitives. `visual_skeleton` remains
  the shared native-definition visual provider. Derived surfaces are model
  proxies where authored visual solids are absent, not measured anatomy.
- See `docs/user_guide/body_part_viz/` for end-user workflow guides
  and `docs/api/body_part_viz.md` for the full API surface.

### Force and Torque Overlay Pipeline

`src/shared/python/force_overlay/` — engine-agnostic physical vector and moment contracts (ADR-0052, #11285).

- `WrenchKind`, `OverlayWrench`, `ForceTorqueFrame` — immutable data layer with optional halves (unavailable is `None`, never zero) and Z-up world frame.
- `ForceTorqueSeries` — strictly increasing time series with gap-bounded linear interpolation and pickle-free NPZ serialization.
- `ForceTorqueProvider`, `read_force_torque_frame` — runtime-checkable Protocol and validated accessor for engine providers.
- Schema: `schemas/force-torque-frame-v1.json` and shared fixtures in `schemas/force-torque-frame-examples.json`.
- User Guide: `docs/user_guide/force_overlay.md` for wrench palettes, moment arcs, web and video overlays.
- `biomechanics/grip_wrench.py` — shared hand-on-club grip wrench (`analyze_grip`, `GripAnalysis`, `GripSeries`): per-hand wrench, net force at the grip midpoint, equivalent couple split into contact-force moment and free torque, per-hand MOF, club-local components and `split_method`. Transport reuses `force_overlay.conversions.move_wrench_point`; do not write another (GCV-7, #11713).
- `biomechanics/grip_extraction.py` — per-engine helpers that build a `GripAnalysis` from engine data (`holding_hand_wrench`, `closure_and_club_analysis`, `allocation_grip_analysis`, `net_only_analysis`, `unavailable_analysis`) (GCV-8, #11714).
- `force_overlay/grip_frame.py` — `grip_frame`, `grip_wrenches_and_metadata`, `GRIP_LABELS`: the one `GripAnalysis` to `ForceTorqueFrame` conversion, with `grip_split_method` and `grip_unavailable_labels` metadata. `force_overlay/palette.label_variant_hex` shades the per-hand grip labels (GCV-10, #11716).
- `biomechanics/grip_plot_model.py` — `GripPlotSeries`, `build_grip_plot_series`, JSON round trip for the Grip Wrench Plots tile and API (GCV-10).
- `grip_contact/pad_layout.py`, `pad_contact.py`, `contact_run.py`, `static_balance.py` — contact grip (OSV-7 phase 3, #11739): pad layout with pad stiffness matched to the bushing (`matched_pad_parameters`), the shared engine-agnostic pad law (`pad_wrench`, reusing `physics.contact_law`), `ContactRun` slip bookkeeping, and the static force and moment balance residual `static_balance`. Engines: `<engine>/python/grip_contact_sim.py` (MuJoCo, Drake, Pinocchio); the bushing variants run in MyoSuite through `myosuite/python/grip_bushing.py`. Design reference: `docs/development/full_body_models/GRIP_PARITY_DECISIONS.md` section 19.
- `biomechanics/plot_traces.py` — `vector_trace`, `none_if_nan`: NaN to JSON `null` for every plot payload; reuse it instead of writing another serializer (GCV-5, #11711).

### Ground-Reaction Analysis Core

`src/shared/python/biomechanics/ground_reaction.py` — the single implementation of per-foot and net GRF, CoP (with ground height `z_g` and the named `COP_MIN_FZ_N` = 10 N threshold), free moment, and moments about the whole-body CoM (GCV-1, #11707, epic #11706).

- `foot_reaction(label, forces, points, torques=None, *, ground_height_m, cop_min_fz_n)` and `analyze_ground_reaction(contacts_by_foot, com_m)` return `FootReaction` / `GroundReactionBreakdown`; unavailable CoP quantities are `None`, never zero.
- `to_overlay_wrenches` emits the `contact:grf_*`, `contact:free_moment_*`, `contact:moment_com_*` labels (ADR-0052); `to_contact_reaction` populates `motion_matching.force_torque.ContactReaction`; `GroundReactionSeries` stacks frames for plots and export.
- `center_of_pressure(force, moment_about_origin, ground_height_m, min_fz)` is the one CoP implementation (GCV-6, #11712). `physics.ground_reaction_forces.compute_cop_from_grf`, `motion_matching.force_torque.compute_center_of_pressure` and `physics._grip_forces.compute_center_of_pressure` are deprecation shims (the grip one is now `compute_grip_pressure_centre`, a pressure-weighted centroid, not a ground CoP). Do not add another CoP helper.
- `biomechanics/ground_reaction_wrenches.py` (`foot_contact_sets`, `ground_reaction_overlay`) and `biomechanics/foot_membership.py` map contact bodies to feet; `mujoco/python/ground_contacts.py` extracts MuJoCo/MyoSuite contacts. Every engine's force-torque source and `force_overlay/bundle_provider.py` route through them (GCV-2, #11708); parity test `tests/integration/cross_engine/test_ground_reaction_parity.py`.

### Impact Parameters

`src/shared/python/impact_parameters/` — the target-line club delivery at impact for every engine (GCV-15, #11721).

- `TargetFrame` (`target_frame.py`) — the target line, up and handedness; the UpstreamDrift default is Z-up with the target along -Y (ADR-0041).
- `ClubheadSeries` and `adapters/` — clubhead pose and twist series from MuJoCo, MyoSuite, Drake, Pinocchio, OpenSim and Simscape; `select_impact_index` picks the impact sample. Simscape has no clubhead orientation, so face quantities are unavailable.
- `extract_impact_parameters` returns `ImpactParameters` at the last pre-contact state (speed, attack angle, path, face, face-to-path, dynamic and spin loft, swing direction and plane, low point, toe/high, smash factor). The optional `BallObservation` carries the impact-model ball contact point, speed and calibration status that the smash factor needs. An unavailable quantity is `None` with its reason in `unavailable`, never zero.
- `tools_gateway` maps the result onto the Tools D-plane (`load_tools_delivery_gateway`); keep the Tools runtime behind this gateway.
- `panel_model.build_impact_card` feeds the impact card in the PyQt and web panels.

### Video Timing

`src/shared/python/video_timing/` — time-based frame selection for every video export (GCV-14, #11720).

- `FrameSchedule` — frame `j` shows the swing at `t0 + j * speed / fps`; coarse source steps are interpolated (linear joints, `slerp` for free and ball quaternions found by `quaternion_groups_from_model`).
- `speed_suffix` (`_1x`, `_0p5x`, `_0p25x`) and `DEFAULT_FPS` = 60, `MAX_SPEED` = 4. `stride_for_speed` exists only for the deprecated `--stride` alias.

### Bounded One-Step Muscle Allocation

`src/shared/python/biomechanics/muscle_allocation.py` exposes
`MuscleRecruitmentPath`, `MuscleAllocationResult`, and
`allocate_muscle_excitations`: a pure-Python bounded single-step allocator
that consumes desired generalized effort and returns, but does not apply, a
raw excitation candidate. It reuses `ActivationDynamics`, `HillMuscleModel`,
and `EquilibriumSolver`; callers provide named moment arms, effort units,
tolerances, bounds, rate limits, and immutable prior state. Effective
activation-kernel floor values are reported separately from raw commands.
This analytic quasistatic prototype has only a two-muscle/three-coordinate
fixture; it is not an efficient or qualified full-body controller. See
chapter39 and `docs/development/feedback_controls/F08_MUSCLE_RECRUITMENT_TURNOVER.md`.

### Shared Quasistatic Muscle Force Evaluation

`src/shared/python/biomechanics/muscle_force_evaluation.py` exposes
`evaluate_quasistatic_muscle_force` and the immutable
`QuasistaticMuscleForceEvaluation` snapshot. It composes the existing analytic
Hill-model and fiber-equilibrium calculation at zero fiber velocity; callers
supply activation and muscle-tendon length. It has no dependency on the
allocator/controller and accepts no native-engine state. The module reports
force-equilibrium residuals but does not decide tolerance acceptance. The
aggregate `biomechanics` initializer still imports the allocator. See chapter
39 and the F08 recruitment turnover; serialized artifact validation is a
separate future consumer.

### MyoFullBody Muscle-Driven Swing

`src/shared/python/myofullbody/` — maps the 44-coordinate spec swing onto the
MyoSuite MyoFullBody model (416 muscles, Apache-2.0, MoBL-derived arms are
non-commercial) and resolves muscle redundancy. Assets are never committed:
`assets.fetch` downloads a pinned `myo_sim` commit into a cache and verifies
every file by sha256 (`scripts/fetch_myofullbody.py`).

- `assets` — pinned fetch, manifest, tamper rejection, licence record, model receipt.
- `couplings.JointCoupling` — exact `mjEQ_JOINT` polynomial substitution.
- `anatomy`, `ik`, `mapping.MyoMapper` — landmark segment frames and orientation
  IK (`map_pose`) plus the rate map `velocity_map` (`Phi`).
- `mapping_report` — key frames, ROM exceedance, hinge cross-check, muscle lengths.
- `redundancy` — `MuscleBasis`, fast `solve_frame`, group reserves, fail-closed
  `qualification`; `swing_pipeline.run_swing` builds the receipt.
- `render` — blue-to-red activation colours, frame selection, four camera views
  (`scripts/render_myofullbody_swing.py`).
- Reference: `docs/research/myofullbody_swing/myofullbody_swing.tex`.

### MOSAIC Model-Aware Matching Estimator

`src/shared/python/estimation/mosaic/` — model-aware multi-trial
forward-dynamics matching. Design reference:
`docs/research/model_aware_matching/model_aware_matching.tex`; epic plan in
`docs/plans/EPIC_MOSAIC_MODEL_AWARE_MATCHING.md`.

- `inertial`, `planar_chain` — vectorized planar inertial regressor fixture.
- `kinematic_basis` — B-spline kinematic basis.
- `inner_solve` — variable-projection inner solve; `outer_solve` — LM outer
  solve with continuation.
- `identifiability_subspace`, `ik_init` — inertial observability and IK start.
- `local_policy` — TVLQR local policy and replay gate.
- `torque_template`, `subject_fit` — torque template and subject pipeline.
- Scope: planar fixture only; no engine regressor provider or human capture
  qualification yet.

### Anthropometrics

`src/shared/python/anthropometrics/`

- `SegmentProperties`, `SubjectAnthropometrics` — frozen, DbC-validated
  canonical records (mass, length, CoM, 3 × 3 inertia in SI units).
- `Estimator`, `Reader`, `Writer`, `EngineAdapter` —
  `@runtime_checkable` Protocols in `contracts.py`.
- `estimators.from_de_leva.DeLevaEstimator` (default),
  `from_dempster.DempsterEstimator`, `from_zatsiorsky.ZatsiorskyEstimator`
  — three regression estimators implementing the `Estimator` Protocol.
- `pipeline.run_pipeline()` — single public entry point: C3D →
  `SubjectAnthropometrics` → URDF / MJCF / `.osim` exports +
  `subject.json` + deterministic `report.html`.
- `engine_adapters.ADAPTER_REGISTRY` — map of `engine_name` to the
  paired export/import adapter (`drake`, `pinocchio`, `myosuite`,
  `opensim`, `simscape`).
- `ui.calibration_dialog.SubjectCalibrationDialog` and
  `ui.segment_properties_panel.SegmentPropertiesPanel` — Qt UI
  surface; thin wrappers over `run_pipeline()`.
- See [ADR 0009](../../adr/0009-anthropometrics-pipeline.md) (canonical
  record + Protocols) and
  [ADR 0010](../../adr/0010-anthropometrics-pipeline.md) (pipeline
  orchestrator + cross-engine bridge).
- User guides:
  [`docs/user_guide/anthropometrics/quickstart.md`](../../user_guide/anthropometrics/quickstart.md),
  [`docs/user_guide/anthropometrics/cross_engine.md`](../../user_guide/anthropometrics/cross_engine.md),
  and the consolidated
  [`docs/user_guide/anthropometrics.md`](../../user_guide/anthropometrics.md).

### Plot Style Toolkit

`src/shared/python/plot_style/`

- Canonical marker-styling stack for every tool that draws markers
  (C3D Viewer, starting-pose matcher, cross-engine dashboard). See
  [ADR 0011](../../adr/0011-plot-style-toolkit.md).
- `MarkerStyle`, `MarkerShape`, `CustomMeshSpec` — frozen dataclasses
  describing every visual property of a marker except its position.
- `StaticColor`, `PaletteColor`, `DataDrivenColor` — three
  `ColorScale` variants. `MarkerStyle.fill_color` accepts any of them;
  data-driven colouring (by clubhead speed, force magnitude, per-frame
  error, ...) is a first-class feature.
- `MarkerRenderer`, `MarkerShapeRenderer`, `ColorResolver` — three
  runtime-checkable Protocols. Implementations live under
  `renderers/`, `shapes/`, `resolvers/`.
- `MatplotlibMarkerRenderer` — **canonical 2D / 3D marker renderer for
  any new tool that needs marker rendering.** A `PyQtGLMarkerRenderer`
  ships alongside for tools that need GPU-rate redraws; both implement
  the same `MarkerRenderer` Protocol.
- `COLORMAP_REGISTRY` (via `get_colormap` / `register_custom_colormap`),
  `SHAPE_REGISTRY`, `RESOLVER_REGISTRY` — dispatch tables that go from
  enum / dataclass to renderer-ready object without isinstance ladders.
- `PresetLibrary.default()` — four curated themes (`default`,
  `scientific_violet`, `monochrome`, `high_contrast`) in
  `BUILTIN_PRESET_NAMES`. JSON v1 round-trip via `PlotStyleSet.save` /
  `PlotStyleSet.load`.
- `MarkerStylePicker`, `ColorPicker`, `ColormapPicker`,
  `DataChannelEditor` — PyQt6 widget surface (lazy import — headless
  consumers can still `import plot_style`).
- See `docs/user_guide/plot_style/` for end-user workflow guides:
  - [`quickstart.md`](../../user_guide/plot_style/quickstart.md) —
    pick a marker shape + color, load a preset, apply to a renderer.
  - [`data_driven_coloring.md`](../../user_guide/plot_style/data_driven_coloring.md) —
    color markers by clubhead speed / force / error; bulk path for
    animation playback.
  - [`colormap_author_guide.md`](../../user_guide/plot_style/colormap_author_guide.md) —
    register custom colormaps and palettes, naming conventions,
    perceptually-uniform recommendations.

### Pose Editor (Interactive Joint-Angle UI)

`src/shared/python/pose_editor/`

- `core.{JointType, JointInfo, PoseEditorState}` — joint metadata
  dataclasses; engine-agnostic.
- `widgets` — slider/spinbox composites for editing joints.
- `library` — preset poses.
- Used by per-engine GUIs (MuJoCo, Drake) for live joint editing.
  **Not** the same as the _starting-pose matcher_ (which solves a
  rigid-body transform across an entire skeleton, not per-joint).

### Model Appearance (Engine-Agnostic Visuals)

`src/shared/python/model_appearance/` — purely visual, never read by physics.

- `schema` — `appearance-v1` document (`document_from_dict`, `load_appearance`,
  `appearance_path_for`), JSON Schema `appearance_v1.schema.json`, and
  `physics_spec_sha256` (spec hash that ignores visual-only keys).
- `library` — named PBR materials, skin tones, clothing presets, club finishes,
  `classify_body` (body name to anatomical part).
- `geometry` — `lofted_segment` / `ellipsoid_mesh` smooth meshes.
- `grip_pose` — the one two-hand grip definition (`DEFAULT_GRIP_POSE`, lead above trail, positions mirroring the spec `GripInterface`, spacing, V angles, overlap/interlock/ten-finger, `HAND_GRIP_ANCHOR_IN_HAND_M`). Engines import it; never keep per-engine hand offsets. Residual series: `grip_contact.closure_series` (`GripClosureSeries`, unavailable is `None`, never zero); report: `python3 -m scripts.grip_closure_report`.
- `grip_hand_mesh` / `grip_hands` — curled-finger hand meshes wrapped on the grip
  (`build_grip_hand_mesh`, hand-grip frame) and their placement on a `ClubAssembly`
  grip in the club-body frame (`grip_hand_frame`, `grip_hand_meshes`,
  `assembly_meshes_with_hands`). Draw the hands on the club body with the club;
  MuJoCo (`appearance_layer`), MyoSuite and the Drake/Pinocchio MeshCat
  backends (`native_viewer_export.backends._club`) already do.
- `head` — parametric visible head (skull, eyes, brows, nose, mouth, ears, neck,
  optional hair or cap), `build_head_parts`, `resolve_head_anchor` (the spec
  `Head` body on the neck at the cervicale; head length from de Leva), and
  `head_override_rotation`. Licence: procedural geometry, CC0-1.0, no third-party mesh.
- Document fields `head` (`enabled`, `headwear` none/hair/cap, `scale`,
  `forward_axis`/`up_axis`, `orientation_override`) and `body_model`
  (`ellipsoid`; `meshes` is rejected until CMB-6 #11657 lands).
- Gaze channel: `head.orientation_override` (`frame` parent/world, `yaw_rad`,
  `pitch_rad`, `roll_rad`, `channel` name) detaches the visual head onto a massless
  mocap body; `head_visual.drive_visual_head(model, data, yaw, pitch, roll, frame)`
  places it per frame. Axes: x forward, y left, z up; yaw left-positive, pitch
  up-positive, roll toward the right shoulder. OSV-3 #11729 drives it for engines
  without a neck.
- MuJoCo translation: `src/engines/physics_engines/mujoco/python/appearance_layer.py`
  via `attach_visual_layer(..., appearance=doc)`; the head is `head_visual.py`
  (`with_head=True` adds only the head to the legacy capsule layer, used by MyoSuite).
  Drake, Pinocchio and OpenSim native viewers use
  `src/tools/native_viewer_export/backends/_head.py` (OBJ per part).

### Theme / Typography

`src/shared/python/theme/`

- `style_constants.Styles` — QSS class names + literals.
- `typography.{get_qfont, get_display_font, Weights}` — font factory.
- `matplotlib_style` — dark-theme matplotlib defaults.
- `colors`, `stylesheets`, `theme_manager` — palette + QSS dispatch.

### Mocap Data Loading

`src/shared/python/club_data/`

- `catalog.py` — optional attributed club properties/build identity; `catalog_io.py`
  provides validated exchange and `catalog_legacy.py` marks historical defaults unverified.
  See [club specification guide](../../motion_capture/club_catalog.md).
- `targets.py` — engine-agnostic loaders for C3D, CSV, JSON, xlsx mocap.
- **Wiffle xlsx values are in CENTIMETRES** despite the workbook's
  "Definitions" tab claiming inches. The MATLAB loader
  (`load_club_target_excel.m` line 53: `CM_TO_METRES = 0.01`) is the
  source of truth — see `MATLAB_GOLF_MODEL_GUIDE.md`.

### Launch Monitor Analytics

`src/tools/launch_monitor_model/` is the canonical vendor-neutral shot-data
stack. Use it for TrackMan, Foresight, FlightScope, Garmin, SkyTrak, Uneekor,
Full Swing, Rapsodo, GSPro/Open Connect, or generic tabular imports instead of
adding a one-off CSV reader.

- `import_session` and `detect_profile` — format/header discovery, unit
  normalization, raw-column retention, and SHA-256 provenance.
- `LaunchMonitorProject` — multi-session aggregation, portable persistence,
  and durable treatment audit records.
- `apply_treatment` — structured filters, missing/duplicate/outlier flags, and
  explicitly labeled identity-derived metrics.
- `compute_correlations`, `compute_pca`, `compute_vif`, and
  `fit_predictive_model` — interdependency and predictive analysis with
  leakage warnings and reproducible splits.
- `compare_monitors`, `analyze_dispersion`, and `analyze_trend` — matched or
  descriptive monitor comparison, shot-pattern ellipses, and longitudinal
  change analysis.
- `src/tools/launch_monitor_analytics/` — the PyQt6 workbench; keep new
  analysis logic in the headless shared package.

See ADR 0031 and `docs/user_guide/launch_monitor_analytics.md`. Correlation and
predictive performance do not establish causality; unmatched monitor
comparisons are not calibration evidence.

### Logging / Config

- `src/shared/python/logging_pkg/logging_config.get_logger(__name__)`
  — canonical logger factory. Don't `import logging; logging.getLogger(...)`
  directly — the shared factory pulls in JSON config, log rotation,
  and fleet-aware filters.

### Launcher Base

- `src/launchers/base.BaseLauncher` — `QMainWindow` subclass for
  **grid-of-tiles launcher windows** (the main UpstreamDriftLauncher, sub-
  launchers showing a card grid). **Not** for single-purpose tool
  windows; those should be standalone `QMainWindow` subclasses
  registered as tiles in `src/config/models.yaml`.

### Launcher Embedding + Cross-Tool IPC

`src/shared/python/launcher_embed/`

- `EmbedCapabilities`, `EmbeddableTool` Protocol, registry.
- See [ADR-0013](../../adr/0013-launcher-composability.md) for design
  rationale and
  [`docs/development/embedding_a_tool.md`](../../development/embedding_a_tool.md)
  for the tool-author guide.

`src/shared/python/realtime/`

- File + WebSocket pub-sub behind one `subscribe`/`publish` facade.
- Channel registry in `channels.py` chooses transport per channel.
- See [`docs/development/realtime_ipc.md`](../../development/realtime_ipc.md).

`src/shared/python/upstream_drift_tools/ui/tools_sidebar/`

- **Sidekick (UnifiedToolsSidebar)**: The right-hand collapsible dock that provides Chat Assistant, Reporting/Summarization, and context-aware utilities.
- Exposes `create_tools_sidebar()` to act as the universal in-repo fallback.

`src/launchers/embedded_host.py`

- `EmbeddedHostWidget` — central QTabWidget + QDockWidget area for
  in-launcher tool hosting.

`src/launchers/launch_routing.py`

- `LaunchMode` enum + `resolve_launch_mode()` for per-tile routing
  (AUTO / NEW_WINDOW / TAB / DOCK / EXTERNAL).

### Tools Sidebar (Optional)

`src/shared/python/gui_launcher/tools_sidebar_integration.py` is the
host-side adapter for the **Unified Tools Sidebar**, a PyQt dock widget
that ships from the sibling [`D-sorganization/Tools`](https://github.com/D-sorganization/Tools)
repository. The widget itself lives in Tools; UpstreamDrift only owns
the optional install path and the Sidekick design-token passthrough.

- **Setup:** run `scripts/setup_tools_workspace.sh` to wire an editable
  sibling checkout, or pass `--tools-mode editable` to pytest (see the
  `--tools-mode` fixture in `tests/conftest.py` and the "Cross-Repo
  Dependencies" section of `CLAUDE.md`).
- **Detection:** `gui_launcher.is_tools_sidebar_available()` returns
  whether the shared module imports. `LauncherDiagnostics.check_tools_sidebar()`
  exposes the same probe in the diagnostic report.
- **Fallback:** when the sibling repo isn't installed (the default),
  `install_tools_sidebar()` no-ops and the launcher continues to run.
  The Sidekick design tokens still apply to the React/Tauri shell; only
  the optional PyQt sidebar is skipped.
