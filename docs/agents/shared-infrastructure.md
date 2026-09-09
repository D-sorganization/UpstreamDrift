# Shared Infrastructure Directory

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
[ADR 0023](../adr/0023-mujoco-warp-backend.md) and the package README.

- `protocol.SimulationBackend` — runtime-checkable Protocol every backend
  satisfies. `DynamicsProvider` (mass matrix / bias forces) and
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

### Tools Ground-Model Consumer Boundary

`src/shared/python/ground_model/` is UpstreamDrift's thin, headless gateway to
the canonical `shared.python.swing_sim.ground` façade owned by Tools. It checks
the exact request, result, and reference-execution schema versions before
binding parsers or execution. Keep physics and wire records in Tools; build
UpstreamDrift API, PyQt, and React presenters on this gateway without copying
the solver or relabeling provenance.

### Motion Matching (The Big One)

`src/shared/python/motion_matching/`

- `target.py`, `club_target.py`, `load_club_target.py`,
  `loaders/` — mocap target structures and loaders (xlsx / C3D / JSON).
- `BodyTarget`, `ClubBallTarget`, `MultiSourceTarget` —
  `src/shared/python/motion_matching/`. Frozen dataclasses + an
  aggregator covering club, ball-aware, and full-body capture
  targets. Cost-function code dispatches on `has_club()`,
  `has_ball()`, `has_body()`. See
  [ADR 0006](../adr/0006-multi-source-motion-targets.md).
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
  contract — [ADR-0026](../adr/0026-canonical-dynamic-state-v2.md) /
  [`docs/conventions/canonical-v2.md`](../conventions/canonical-v2.md)
  (Canonical Core EPIC #6772).**
- `PoseConventionAdapter` — runtime-checkable Protocol; one
  implementation per engine in `adapters/`.
- `LiveKinematicsService` — Protocol; one implementation per engine in
  `services/`. Falls back to `MockKinematicsService` when the engine
  wheel is absent.
- `pose_io` — save/load to engine-native initial-state files and to
  `BodyTarget` motion-matching JSON.
- User guides:
  [`docs/user_guide/pose_studio/quickstart.md`](../user_guide/pose_studio/quickstart.md),
  [`docs/user_guide/pose_studio/cross_engine_conventions.md`](../user_guide/pose_studio/cross_engine_conventions.md),
  [`docs/user_guide/pose_studio/save_formats.md`](../user_guide/pose_studio/save_formats.md).

### Body-Part Visualisation Toolkit

`src/shared/python/body_part_viz/`

- `body_part_viz` package with shapes / fitters / renderers / asset
  library — the **canonical shape stack** for any tool that draws body
  segments. See [ADR 0008](../adr/0008-body-part-viz-toolkit.md).
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
- See `docs/user_guide/body_part_viz/` for end-user workflow guides
  and `docs/api/body_part_viz.md` for the full API surface.

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
- See [ADR 0009](../adr/0009-anthropometrics-pipeline.md) (canonical
  record + Protocols) and
  [ADR 0010](../adr/0010-anthropometrics-pipeline.md) (pipeline
  orchestrator + cross-engine bridge).
- User guides:
  [`docs/user_guide/anthropometrics/quickstart.md`](../user_guide/anthropometrics/quickstart.md),
  [`docs/user_guide/anthropometrics/cross_engine.md`](../user_guide/anthropometrics/cross_engine.md),
  and the consolidated
  [`docs/user_guide/anthropometrics.md`](../user_guide/anthropometrics.md).

### Plot Style Toolkit

`src/shared/python/plot_style/`

- Canonical marker-styling stack for every tool that draws markers
  (C3D Viewer, starting-pose matcher, cross-engine dashboard). See
  [ADR 0011](../adr/0011-plot-style-toolkit.md).
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
  - [`quickstart.md`](../user_guide/plot_style/quickstart.md) —
    pick a marker shape + color, load a preset, apply to a renderer.
  - [`data_driven_coloring.md`](../user_guide/plot_style/data_driven_coloring.md) —
    color markers by clubhead speed / force / error; bulk path for
    animation playback.
  - [`colormap_author_guide.md`](../user_guide/plot_style/colormap_author_guide.md) —
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
  See [club specification guide](../motion_capture/club_catalog.md).
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
- See [ADR-0013](../adr/0013-launcher-composability.md) for design
  rationale and
  [`docs/development/embedding_a_tool.md`](../development/embedding_a_tool.md)
  for the tool-author guide.

`src/shared/python/realtime/`

- File + WebSocket pub-sub behind one `subscribe`/`publish` facade.
- Channel registry in `channels.py` chooses transport per channel.
- See [`docs/development/realtime_ipc.md`](../development/realtime_ipc.md).

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

### Sidekick (AI Chat / Agentic Assistant)

Sidekick is the cross-shell AI chat surface. Two host implementations
consume one shared design-token contract:

- **React/Tauri shell:** [`ui/src/pages/Chat.tsx`](../../ui/src/pages/Chat.tsx)
  routes at `/chat`; the panel itself is
  [`ui/src/components/ui/ChatPanel.tsx`](../../ui/src/components/ui/ChatPanel.tsx)
  and binds its surface palette to `var(--sidekick-color-*)` /
  `var(--sidekick-space-*)` CSS variables (declared in
  [`ui/src/index.css`](../../ui/src/index.css)).
- **PyQt launcher panel:**
  [`src/shared/python/ai/gui/assistant_panel.py`](../../src/shared/python/ai/gui/assistant_panel.py)
  (`AIAssistantPanel`, window title "Sidekick"). The launcher
  embeds it both as a splitter pane
  (`src/launchers/launcher_ui_setup.py`) and as a registered
  [embeddable tool](#launcher-embedding--cross-tool-ipc) so users can
  open it via right-click → "Launch in Tab" / "Launch in Dock".

Shared infrastructure:

- **Design tokens:**
  [`src/shared/python/theme/sidekick_tokens.py`](../../src/shared/python/theme/sidekick_tokens.py)
  maps active launcher theme colors onto canonical `sidekick.color.*` /
  `sidekick.space.*` / `sidekick.radius.*` / `sidekick.font.*` keys. The
  TypeScript mirror is
  [`ui/src/api/themeClient.ts`](../../ui/src/api/themeClient.ts); both maps are
  pinned in lock-step by
  [`tests/unit/theme/test_sidekick_parity.py`](../../tests/unit/theme/test_sidekick_parity.py).
- **Embeddable adapter:**
  [`src/tools/sidekick/_embed_adapter.py`](../../src/tools/sidekick/_embed_adapter.py).
  Self-registers via
  [`src/launchers/embedded_tool_bootstrap.py`](../../src/launchers/embedded_tool_bootstrap.py)
  and exposes the tile through
  [`src/config/models.yaml`](../../src/config/models.yaml).
- **Chat context bridge:**
  [`src/shared/python/ai/chat_context.py`](../../src/shared/python/ai/chat_context.py)
  — thread-safe ring buffer (`record_event`, `get_chat_context`) with
  redaction of `password` / `token` / `secret` / `api_key` / `/home/` /
  `C:\` patterns and a 4 KB dump cap. The WebSocket handler in
  [`src/api/routes/chat_ws.py`](../../src/api/routes/chat_ws.py) injects the
  payload as a `system` message when populated; gate with
  `UPSTREAMDRIFT_SIDEKICK_CONTEXT=0` to disable.
- **Agentic tools:** new analytical surfaces register through the existing
  AI tool registry. The current cross-engine example is
  [`src/shared/python/ai/tools/sidekick_analytics.py`](../../src/shared/python/ai/tools/sidekick_analytics.py)
  (`summarize_simulation_run`). The system prompt in
  [`src/shared/python/ai/system_prompts.py`](../../src/shared/python/ai/system_prompts.py)
  advertises registered tools to the assistant.
- **Agent action layer (epic [#5967](https://github.com/D-sorganization/UpstreamDrift/issues/5967), ADR-0017):**
  [`src/shared/python/sidekick/agent/`](../../src/shared/python/sidekick/agent)
  — the single audited choke-point for every agentic action. New
  Sidekick actions register through `SidekickActionService`; the
  planner translates LLM tool calls into validated `PlannedStep`s; an
  access policy gates writes and destructive actions; an audit sink
  records every call. Host integrations (launcher, Pose Studio, ...)
  implement `HostActionPort` — sidekick never imports them. See
  [`docs/sidekick/agent.md`](../sidekick/agent.md) for the worked
  example of adding a new action.

When extending Sidekick — adding a tool the assistant can call,
extending the chat context bridge, or restyling the panel — reuse these
surfaces rather than forking new color/spacing constants or new event
buses. For anything the assistant should _do_ (not just compute),
register a new `ActionDescriptor` via `SidekickActionService` rather
than wiring a one-off `ai.tools.*` module.

**If you need a standalone Sidekick app, use `sidekick.standalone.*`; do not
write a new shell.** The standalone package
([`vendor/ud-tools/src/shared/python/sidekick/standalone/`](../../vendor/ud-tools/src/shared/python/sidekick/standalone))
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
[`scripts/config/shared_python_ownership_exceptions.yaml`](../../scripts/config/shared_python_ownership_exceptions.yaml);
startup exposes only those exact module names and fails closed on unclassified,
unresolved, or newly conflicting files.

See [`docs/sidekick/standalone.md`](../sidekick/standalone.md) for the user-
facing guide and [ADR-0018](../adr/0018-standalone-sidekick.md) for design
rationale (issues #5984, #5985, #5986, #5987).

### Rust Kernels

- `rust_core/upstream-physics/` — RK4 integrator, aerodynamics, contact,
  swing-plane fit. Built via `maturin develop`; consumed by Python
  via `src/shared/python/physics/rust_kernel.py` (which falls back to
  pure Python if the Rust wheel isn't installed). See section F for
  when to write Rust.
