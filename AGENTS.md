# AGENTS.md — Discovery Workflow & Shared-Infrastructure Directory

> **Read this first.** This file exists because we kept reinventing
> infrastructure that already lived in the repo (FK solvers, skeleton
> renderers, reference golfer poses, mocap loaders, theme constants).
> Tracked as issue #4377; updated as we discover new shared modules.

For binding repo policy (ruff, file-size budgets, branch naming, PR
targets, etc.) see [`CLAUDE.md`](CLAUDE.md). For background and
historical agent-workflow notes see
[`docs/development/agents.md`](docs/development/agents.md).

This file is narrower: a **discovery workflow** for agents and a
**directory of shared infrastructure** so we stop duplicating work.

---

## A. Before you write new code — discovery workflow

When you're about to add functionality, run these five steps **in
order** before writing a line:

1. **Grep `src/shared/python/` for the concept.** Most cross-engine
   primitives live there (FK, reference poses, mocap loaders, theme
   constants, club-target structures, optimisation drivers).
2. **Grep `src/tools/` and `src/launchers/` for similar tools.** If a
   tile already does ≥60 % of what you want, extend it instead of
   forking.
3. **Read the relevant `__init__.py`** (public API surface) before
   importing internals. The `motion_matching/diagnostics/__init__.py`
   in particular is a curated façade that hides private helpers.
4. **Read the docs.** `docs/` has design notes for the bigger systems
   and `MATLAB_GOLF_MODEL_GUIDE.md` documents pitfalls (e.g. the
   cm-vs-inches Wiffle xlsx gotcha).
5. **Only then** propose new code. If you find yourself
   reimplementing something that lives in `src/shared/`, stop and use
   the shared version — even if it forces minor API massaging on your
   side.

> **The "I'll just write it inline, it's only 20 lines" trap is real.**
> Twenty lines today becomes a divergent re-implementation tomorrow
> when the shared module changes its sign convention.

---

## B. Shared-infrastructure directory

Grouped by concern, with one-line descriptions and the public symbols
worth knowing. Not exhaustive — when in doubt, grep — but these are
the modules most often missed.

### Engine abstraction

`src/shared/python/engine_core/`

- `interfaces.PhysicsEngine` — runtime-checkable Protocol.
- `sub_protocols` — focused mixins (`Loadable`, `Steppable`, `Queryable`,
  `DynamicsComputable`, `Recordable`).
- `engine_registry`, `engine_manager`, `engine_loaders` — plugin
  discovery, lifecycle, deferred imports.
- `capabilities` — feature-capability declarations per engine.
- `mock_engine` — fallback for headless / CI tests.

### Simulation backends (GPU-ready, backend-agnostic)

`src/shared/python/simulation_backends/` — interchangeable physics backends
for the golf double-pendulum behind one Protocol. See
[ADR 0023](docs/adr/0023-mujoco-warp-backend.md) and the package README.

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

### Motion matching (the big one)

`src/shared/python/motion_matching/`

- `target.py`, `club_target.py`, `load_club_target.py`,
  `loaders/` — mocap target structures and loaders (xlsx / C3D / JSON).
- `BodyTarget`, `ClubBallTarget`, `MultiSourceTarget` —
  `src/shared/python/motion_matching/`. Frozen dataclasses + an
  aggregator covering club, ball-aware, and full-body capture
  targets. Cost-function code dispatches on `has_club()`,
  `has_ball()`, `has_body()`. See
  [ADR 0006](docs/adr/0006-multi-source-motion-targets.md).
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

### Cross-engine pose interchange

`src/shared/python/pose_interchange/`

- `CanonicalPose` — frozen dataclass; pelvis SE(3) + joint angles in
  the canonical convention (intrinsic XYZ Euler in degrees, joint names
  matching `reference_golfer_setup`). See ADR 0012. **For full dynamic
  state (q, v, a + quaternion floating base), see the `canonical-v2`
  contract — [ADR-0026](docs/adr/0026-canonical-dynamic-state-v2.md) /
  [`docs/conventions/canonical-v2.md`](docs/conventions/canonical-v2.md)
  (Canonical Core EPIC #6772).**
- `PoseConventionAdapter` — runtime-checkable Protocol; one
  implementation per engine in `adapters/`.
- `LiveKinematicsService` — Protocol; one implementation per engine in
  `services/`. Falls back to `MockKinematicsService` when the engine
  wheel is absent.
- `pose_io` — save/load to engine-native initial-state files and to
  `BodyTarget` motion-matching JSON.
- User guides:
  [`docs/user_guide/pose_studio/quickstart.md`](docs/user_guide/pose_studio/quickstart.md),
  [`docs/user_guide/pose_studio/cross_engine_conventions.md`](docs/user_guide/pose_studio/cross_engine_conventions.md),
  [`docs/user_guide/pose_studio/save_formats.md`](docs/user_guide/pose_studio/save_formats.md).

### Body-part visualisation toolkit

`src/shared/python/body_part_viz/`

- `body_part_viz` package with shapes / fitters / renderers / asset
  library — the **canonical shape stack** for any tool that draws body
  segments. See [ADR 0008](docs/adr/0008-body-part-viz-toolkit.md).
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
- See [ADR 0009](docs/adr/0009-anthropometrics-pipeline.md) (canonical
  record + Protocols) and
  [ADR 0010](docs/adr/0010-anthropometrics-pipeline.md) (pipeline
  orchestrator + cross-engine bridge).
- User guides:
  [`docs/user_guide/anthropometrics/quickstart.md`](docs/user_guide/anthropometrics/quickstart.md),
  [`docs/user_guide/anthropometrics/cross_engine.md`](docs/user_guide/anthropometrics/cross_engine.md),
  and the consolidated
  [`docs/user_guide/anthropometrics.md`](docs/user_guide/anthropometrics.md).

### Plot Style Toolkit

`src/shared/python/plot_style/`

- Canonical marker-styling stack for every tool that draws markers
  (C3D Viewer, starting-pose matcher, cross-engine dashboard). See
  [ADR 0011](docs/adr/0011-plot-style-toolkit.md).
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
  - [`quickstart.md`](docs/user_guide/plot_style/quickstart.md) —
    pick a marker shape + color, load a preset, apply to a renderer.
  - [`data_driven_coloring.md`](docs/user_guide/plot_style/data_driven_coloring.md) —
    color markers by clubhead speed / force / error; bulk path for
    animation playback.
  - [`colormap_author_guide.md`](docs/user_guide/plot_style/colormap_author_guide.md) —
    register custom colormaps and palettes, naming conventions,
    perceptually-uniform recommendations.

### Pose editor (interactive joint-angle UI)

`src/shared/python/pose_editor/`

- `core.{JointType, JointInfo, PoseEditorState}` — joint metadata
  dataclasses; engine-agnostic.
- `widgets` — slider/spinbox composites for editing joints.
- `library` — preset poses.
- Used by per-engine GUIs (MuJoCo, Drake) for live joint editing.
  **Not** the same as the _starting-pose matcher_ (which solves a
  rigid-body transform across an entire skeleton, not per-joint).

### Theme / typography

`src/shared/python/theme/`

- `style_constants.Styles` — QSS class names + literals.
- `typography.{get_qfont, get_display_font, Weights}` — font factory.
- `matplotlib_style` — dark-theme matplotlib defaults.
- `colors`, `stylesheets`, `theme_manager` — palette + QSS dispatch.

### Mocap data loading

`src/shared/python/club_data/`

- `targets.py` — engine-agnostic loaders for C3D, CSV, JSON, xlsx mocap.
- **Wiffle xlsx values are in CENTIMETRES** despite the workbook's
  "Definitions" tab claiming inches. The MATLAB loader
  (`load_club_target_excel.m` line 53: `CM_TO_METRES = 0.01`) is the
  source of truth — see `MATLAB_GOLF_MODEL_GUIDE.md`.

### Logging / config

- `src/shared/python/logging_pkg/logging_config.get_logger(__name__)`
  — canonical logger factory. Don't `import logging; logging.getLogger(...)`
  directly — the shared factory pulls in JSON config, log rotation,
  and fleet-aware filters.

### Launcher base

- `src/launchers/base.BaseLauncher` — `QMainWindow` subclass for
  **grid-of-tiles launcher windows** (the main UpstreamDriftLauncher, sub-
  launchers showing a card grid). **Not** for single-purpose tool
  windows; those should be standalone `QMainWindow` subclasses
  registered as tiles in `src/config/models.yaml`.

### Launcher embedding + cross-tool IPC

`src/shared/python/launcher_embed/`

- `EmbedCapabilities`, `EmbeddableTool` Protocol, registry.
- See [ADR-0013](docs/adr/0013-launcher-composability.md) for design
  rationale and
  [`docs/development/embedding_a_tool.md`](docs/development/embedding_a_tool.md)
  for the tool-author guide.

`src/shared/python/realtime/`

- File + WebSocket pub-sub behind one `subscribe`/`publish` facade.
- Channel registry in `channels.py` chooses transport per channel.
- See [`docs/development/realtime_ipc.md`](docs/development/realtime_ipc.md).

`src/shared/python/upstream_drift_tools/ui/tools_sidebar/`

- **Sidekick (UnifiedToolsSidebar)**: The right-hand collapsible dock that provides Chat Assistant, Reporting/Summarization, and context-aware utilities.
- Exposes `create_tools_sidebar()` to act as the universal in-repo fallback.

`src/launchers/embedded_host.py`

- `EmbeddedHostWidget` — central QTabWidget + QDockWidget area for
  in-launcher tool hosting.

`src/launchers/launch_routing.py`

- `LaunchMode` enum + `resolve_launch_mode()` for per-tile routing
  (AUTO / NEW_WINDOW / TAB / DOCK / EXTERNAL).

### Tools sidebar (optional)

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

### Sidekick (AI chat / agentic assistant)

Sidekick is the cross-shell AI chat surface. Two host implementations
consume one shared design-token contract:

- **React/Tauri shell:** [`ui/src/pages/Chat.tsx`](ui/src/pages/Chat.tsx)
  routes at `/chat`; the panel itself is
  [`ui/src/components/ui/ChatPanel.tsx`](ui/src/components/ui/ChatPanel.tsx)
  and binds its surface palette to `var(--sidekick-color-*)` /
  `var(--sidekick-space-*)` CSS variables (declared in
  [`ui/src/index.css`](ui/src/index.css)).
- **PyQt launcher panel:**
  [`src/shared/python/ai/gui/assistant_panel.py`](src/shared/python/ai/gui/assistant_panel.py)
  (`AIAssistantPanel`, window title "Sidekick"). The launcher
  embeds it both as a splitter pane
  (`src/launchers/launcher_ui_setup.py`) and as a registered
  [embeddable tool](#launcher-embedding--cross-tool-ipc) so users can
  open it via right-click → "Launch in Tab" / "Launch in Dock".

Shared infrastructure:

- **Design tokens:**
  [`src/shared/python/theme/sidekick_tokens.py`](src/shared/python/theme/sidekick_tokens.py)
  maps active launcher theme colors onto canonical `sidekick.color.*` /
  `sidekick.space.*` / `sidekick.radius.*` / `sidekick.font.*` keys. The
  TypeScript mirror is
  [`ui/src/api/themeClient.ts`](ui/src/api/themeClient.ts); both maps are
  pinned in lock-step by
  [`tests/unit/theme/test_sidekick_parity.py`](tests/unit/theme/test_sidekick_parity.py).
- **Embeddable adapter:**
  [`src/tools/sidekick/_embed_adapter.py`](src/tools/sidekick/_embed_adapter.py).
  Self-registers via
  [`src/launchers/embedded_tool_bootstrap.py`](src/launchers/embedded_tool_bootstrap.py)
  and exposes the tile through
  [`src/config/models.yaml`](src/config/models.yaml).
- **Chat context bridge:**
  [`src/shared/python/ai/chat_context.py`](src/shared/python/ai/chat_context.py)
  — thread-safe ring buffer (`record_event`, `get_chat_context`) with
  redaction of `password` / `token` / `secret` / `api_key` / `/home/` /
  `C:\` patterns and a 4 KB dump cap. The WebSocket handler in
  [`src/api/routes/chat_ws.py`](src/api/routes/chat_ws.py) injects the
  payload as a `system` message when populated; gate with
  `UPSTREAMDRIFT_SIDEKICK_CONTEXT=0` to disable.
- **Agentic tools:** new analytical surfaces register through the existing
  AI tool registry. The current cross-engine example is
  [`src/shared/python/ai/tools/sidekick_analytics.py`](src/shared/python/ai/tools/sidekick_analytics.py)
  (`summarize_simulation_run`). The system prompt in
  [`src/shared/python/ai/system_prompts.py`](src/shared/python/ai/system_prompts.py)
  advertises registered tools to the assistant.
- **Agent action layer (epic [#5967](https://github.com/D-sorganization/UpstreamDrift/issues/5967), ADR-0017):**
  [`src/shared/python/sidekick/agent/`](src/shared/python/sidekick/agent/)
  — the single audited choke-point for every agentic action. New
  Sidekick actions register through `SidekickActionService`; the
  planner translates LLM tool calls into validated `PlannedStep`s; an
  access policy gates writes and destructive actions; an audit sink
  records every call. Host integrations (launcher, Pose Studio, ...)
  implement `HostActionPort` — sidekick never imports them. See
  [`docs/sidekick/agent.md`](docs/sidekick/agent.md) for the worked
  example of adding a new action.

When extending Sidekick — adding a tool the assistant can call,
extending the chat context bridge, or restyling the panel — reuse these
surfaces rather than forking new color/spacing constants or new event
buses. For anything the assistant should _do_ (not just compute),
register a new `ActionDescriptor` via `SidekickActionService` rather
than wiring a one-off `ai.tools.*` module.

**If you need a standalone Sidekick app, use `sidekick.standalone.*`; do not
write a new shell.** The standalone package
([`src/shared/python/sidekick/standalone/`](src/shared/python/sidekick/standalone/))
provides:

- `sidekick.standalone.runner` — headless `sidekick run --calculator <name>`
  dispatcher (no GUI or display required).
- `sidekick.standalone.preferences` — typed preference surface backed by an
  injectable `SessionStore`.
- `sidekick.standalone.onboarding` — first-run sentinel + 3-step state machine.
- `sidekick.standalone.session_store` — `InMemorySessionStore` (tests) and
  `FileSessionStore` (production).

See [`docs/sidekick/standalone.md`](docs/sidekick/standalone.md) for the user-
facing guide and [ADR-0018](docs/adr/0018-standalone-sidekick.md) for design
rationale (issues #5984, #5985, #5986, #5987).

### Rust kernels

- `rust_core/upstream-physics/` — RK4 integrator, aerodynamics, contact,
  swing-plane fit. Built via `maturin develop`; consumed by Python
  via `src/shared/python/physics/rust_kernel.py` (which falls back to
  pure Python if the Rust wheel isn't installed). See section F for
  when to write Rust.

---

## C. Where new code goes — decision tree

```
Used by exactly one engine?
    → src/engines/<engine>/python/<module>.py
Used by 2+ engines?  (e.g. mocap loader, FK, cost terms)
    → src/shared/python/<topic>/
Standalone PyQt6 desktop tool?
    → src/tools/<tool_name>/  (PACKAGE, not a single file)
        - __init__.py, __main__.py, gui.py, core.py, README.md
        - register a tile in src/config/models.yaml
Engine-specific MATLAB code (Simscape model dynamics, helpers)?
    → src/engines/<engine>/matlab/
Inner-loop math kernel reused by Python AND WASM?
    → rust_core/<crate>/
        - PyO3 bindings via maturin
        - parity-tested against pure-Python fallback
Backend service / API?
    → src/api/{routes,services}/
```

When relocating something, leave a thin shim at the old path with a
`DeprecationWarning` and a one-line re-export so any external pinned
references keep working through one release cycle. Example: see the
shims at
`src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Motion Capture Plotter/starting_pose_*.py`
which redirect to the new `src/tools/starting_pose_matcher/` package.

---

## D. Tile-launcher contract

The main UpstreamDriftLauncher reads `src/config/models.yaml` and renders one
tile per entry. A tile entry needs:

```yaml
- id: "starting_pose_matcher"
  name: "Starting-Pose Matcher" # display name
  description: "..." # one-line tooltip
  type: "special_app" # other types: model, custom_humanoid, drake, ...
  path: "src/tools/starting_pose_matcher/__main__.py" # relative to REPO_ROOT
  launcher:
    category: "tool" # tool | physics_engine | document
    logo: "assets/<icon>.png" # optional; falls back to default tile art
    status: "ready" # ready | beta | broken
```

Tiles are launched as `subprocess.Popen` with the system Python. Any
tool listed here should be runnable as `python -m <package>` from the
repo root — the launcher resolves the path, prepends the repo root to
`sys.path`, and invokes the module directly.

If your tool has runtime dependencies beyond core (PyQt6, matplotlib,
pandas, openpyxl), declare them in a named `[project.optional-dependencies]`
extra in `pyproject.toml` and reference the install command in the
tile's `description`. Example: the `gui-tools` extra installs PyQt6 +
matplotlib + pandas + openpyxl for the desktop tools.

---

## E. Tests

Tests go in `tests/<area>/<subarea>/test_*.py` — mirror the source
layout under `src/`. See `tests/README.md` and `tests/conftest.py` for
the full set of markers and fixtures.

### Markers (from `pyproject.toml`)

- `unit` — fast, deterministic, no engines.
- `integration` — exercises 2+ modules together.
- `slow` — > 5 s runtime; skipped from default CI lane.
- `live_simulation` — actually runs MuJoCo / Drake / Simscape; **always
  skipped** in default `pytest` runs. Opt in with `-m live_simulation`.
- `requires_gl`, `requires_mocap_fixtures` — skipped on CI fleet that
  lacks the resource.
- Engine-specific: `requires_mujoco`, `requires_drake`, `requires_pinocchio`,
  `requires_opensim`, `requires_matlab`.

### Optional GUI deps

PyQt6 / matplotlib are optional. Tests that need them MUST skip
cleanly when the import fails — `pytest`'s user-site Python on Windows
sometimes has a broken PyQt6 DLL search path even when the regular
interpreter loads it fine. Pattern:

```python
def _load_module():
    try:
        import PyQt6.QtCore  # noqa: F401
        import matplotlib    # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6/matplotlib not loadable: {exc}")
    ...
```

### Pure-data layer

For PyQt6 desktop tools, separate the pure-data math (`core.py`) from
the GUI (`gui.py`) so the data layer can be tested in any environment.
The matcher (`src/tools/starting_pose_matcher/`) is the canonical
example of this split.

---

## F. When to use Rust

The repo has exactly two Rust crates (as of this writing):

- `rust_core/upstream-physics/` — RK4 integrator, aerodynamics, contact,
  swing-plane. Inner-loop numerics; used identically by the Python
  backend (`src/shared/python/physics/rust_kernel.py`) and the WASM
  browser frontend.
- `ui/src-tauri/golf-modeling-suite` — Tauri desktop shell. Spawns the
  Python API server; **no physics or GUI logic in Rust**.

The criteria that produce a Rust crate in this project:

1. **Profiling shows a sub-millisecond inner loop** that gets called
   thousands of times per simulation, AND
2. The same kernel needs to run identically in Python AND another
   runtime (WASM, embedded C++, etc.).

Notably **NOT** Rust-shaped:

- GUI tools. PyQt6 is faster to iterate, has a far richer ecosystem
  for desktop scientific UIs, and binds the matplotlib stack we already
  use everywhere else.
- Data pipelines. pandas / numpy / scipy already cover everything we
  need at speeds that don't bottleneck.
- Config, orchestration, tests. These should stay in Python.

If you're tempted to rewrite something in Rust, profile first and
verify that the workload genuinely lives in inner loops. Most of the
time you'll find the bottleneck is matplotlib's 3D renderer or pandas'
xlsx parser, neither of which Rust can help with.

---

## G. Lessons learned (this section grows)

- **Wiffle xlsx is in CM, not inches.** The "Definitions" tab of the
  workbook is wrong; trust `load_club_target_excel.m` (line 53:
  `CM_TO_METRES = 0.01`).
- **The Simscape body chain has a `torso` joint** between spine and hub
  that hosts the revolute Z (twist) joint. Don't collapse to
  `hip → spine → hub` directly — you'll lose the visible torso coil.
  See
  `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/mdl_reference/GolfSwing3D_Kinetic.mdl`
  block "Torso Kinetically Driven" (SID 8331).
- **Don't subclass `BaseLauncher` for single-purpose tools.**
  `BaseLauncher` is for grid-of-tiles launcher windows. Standalone
  tools should be plain `QMainWindow` subclasses registered as tiles.
- **Test-environment pip vs. interpreter mismatch.** If pytest under
  `C:\Python314\python.exe -m pytest` fails to load PyQt6 with
  `DLL load failed`, it's because pytest is being resolved from
  `AppData\Roaming\Python\Python314` (user-site). Tests should skip
  gracefully — never assume PyQt6 imports.
- **The shared FK has known asymmetry** in hand height with the
  reference Address pose. When deriving fallback skeletons, use FK
  for "the body shape" (torso, shoulders) but be cautious about
  hand heights — they may need hand-tuning.
- **Touching ANY file under `src/shared/python/motion_matching/` may
  surface pre-existing mypy/bandit debt.** PR #4924 cleared the worst
  of it (engine_init_profiler md5/sha1 → usedforsecurity=False;
  precondition lambdas wrapped in bool(); excel loader pandas-cell
  type ignores). If a hook fails on a file you didn't author, follow
  the same sidecar pattern.
- **Two-tool live coupling.** When a tool publishes high-frequency
  state (e.g. Pose Studio's canonical pose at 30 Hz), use the
  WebSocket transport — file pub-sub adds 100+ ms of disk latency.
  The `realtime` facade picks automatically via the channel
  registry; don't hard-code. See
  [`docs/development/realtime_ipc.md`](docs/development/realtime_ipc.md).
- **Embed adapters keep PyQt6 imports lazy.** The
  `EmbeddableTool` Protocol module deliberately doesn't import
  PyQt6 (widget types are spelled `typing.Any`) so headless CI and
  the docs builder can introspect adapters without the GUI extras.
  Your `_embed_adapter.py` should follow the same pattern: import
  PyQt6 inside `create_main_widget`, not at module top.
- **`cleanup()` must be idempotent.** The host calls it on tab
  close, on parent shutdown, and on `closeEvent` — sometimes more
  than once during teardown. Drop the widget reference first, then
  tear down resources, and guard with `if widget is None: return`
  at the top.

---

## H. How to update this file

Found a shared module you wished you'd known about? Add it to section
B with a one-liner. Found a new trap? Add it to section G. Keep
the file focused — if it's growing past 400 lines, split sub-pages
into `docs/agents/`.

After adding to section B, also link the change in `CLAUDE.md`'s
"Shared Code Layout" section if one exists, and tag the file in your
PR description so reviewers can verify discoverability.

---

<!-- BEGIN FLEET-MANAGED: reasoning-engagement -->

## 🧠 Reasoning & Engagement

> This section is managed centrally by Repository_Management and synced fleet-wide.
> Do NOT edit it directly in individual repositories — edit the source in Repository_Management/AGENTS.md.

These rules govern _how_ you engage with a task before and during implementation. They exist because LLM agents tend to pick an interpretation silently, overcomplicate the solution, and edit code they were not asked to touch. Each rule directly counteracts one of those failure modes.

- **Surface ambiguity. Do not guess silently.** If the request has more than one plausible interpretation, list the options and ask before implementing. Picking one and running with it is the single most common cause of rework in this fleet.
- **Push back on overcomplication.** If a simpler approach would satisfy the request, say so before you build the complicated one. Do not implement bloated 1000-line constructions when 100 would do. The senior-engineer test: would they call this overcomplicated? If yes, simplify.
- **Stay surgical.** Every changed line must trace directly to the user's request. Do not "improve" adjacent code, comments, formatting, or imports. Do not refactor things that are not broken. Match existing style even if you would do it differently.
- **Spotted ≠ fix.** If you notice unrelated dead code, latent bugs, or stylistic problems while working, _mention them in the PR body or as a follow-up issue_ — do not fix them in the same PR. (The `mcp__ccd_session__spawn_task` tool is the right channel when working interactively.)
- **Clean up only your own orphans.** If your changes leave imports, variables, or functions newly unused, remove them. Do not delete pre-existing dead code unless the task asked for it.
- **State a verifiable success criterion before coding.** For a bug fix, that's a failing test that reproduces it (RED → GREEN, see TDD section below). For a feature, the explicit check that says "done." "Make it work" is not a success criterion.

**The diff test:** every line in your final diff should answer "this is here because the user asked for X." If you cannot answer that for a given line, remove it.

<!-- END FLEET-MANAGED: reasoning-engagement -->

---

<!-- BEGIN FLEET-MANAGED: network-api-hygiene -->

## 🛑 NETWORK & API HYGIENE (CRITICAL)

> This section is managed centrally by Repository_Management and synced fleet-wide.
> Do NOT edit it directly in individual repositories — edit the source in Repository_Management/AGENTS.md.

### GitHub API Quotas

| API Type                  | Quota        | Consumed By                                                        |
| ------------------------- | ------------ | ------------------------------------------------------------------ |
| REST (`gh api repos/...`) | 5,000 req/hr | Safe for polling                                                   |
| GraphQL                   | 5,000 req/hr | `gh pr list --json`, `gh pr checks`, `gh pr create`, `gh pr merge` |

GraphQL and REST have **separate** quotas. Exhausting GraphQL blocks PR creation and merging fleet-wide for an entire hour.

### Mandatory Rules

- **NO MASS POLLING**: Agents MUST NEVER use `gh pr list`, `gh issue list`, or arbitrary REST/GraphQL loops in a bulk manner to "scan" or "sweep" the repository fleet. Single, scoped repository lookups are allowed when needed (e.g., checking if a specific PR exists).
- **LOCAL FIRST**: Rely on local `.md` files, previously generated `issues.json` artifacts, or user assistance to find task context — do not query GitHub to discover what to work on.
- **NO PARALLELIZED GITHUB CLI**: Never write or execute scripts that loop over multiple repositories performing `gh` operations (automated PR merge scripts, fleet-wide status sweeps, etc.).
- **NO TIGHT POLLING LOOPS**: Never implement `while true; do gh pr checks $PR; sleep 30; done` patterns. Each iteration of such a loop costs 1–3 GraphQL calls; at 30-second intervals that drains the 5,000/hr quota in under 3 hours.
  - ❌ `while true; do gh pr checks; sleep 30; done`
  - ✅ `gh run watch <run-id>` — streams CI events without polling
  - ✅ Check status once at natural work breakpoints (after completing other tasks)
- **BATCHING**: If remote information is absolutely necessary, use a single focused query — not a loop of queries.
- **REST OVER GRAPHQL FOR CI STATUS**: Use REST endpoints for CI polling; they don't consume the GraphQL quota.
  - ❌ `gh pr checks <N>` (GraphQL)
  - ✅ `gh api repos/OWNER/REPO/actions/runs` (REST)
  - ✅ `gh api repos/OWNER/REPO/actions/jobs/<id>/logs` (REST)
- **STOP MONITORS IMMEDIATELY**: When using background monitor tasks, call `TaskStop <id>` the moment the monitored condition is satisfied. Do not leave monitors running "just in case."
- **LONG POLLING INTERVALS**: Background monitors must use ≥270-second intervals (keeps the prompt cache warm). Default to 1200–1800 s for idle monitoring. Never chain short sleeps to work around the 60-second minimum.
- **SILENT FAILURES**: If an API rate limit is hit, HALT NETWORK ACTIVITY IMMEDIATELY. Do not write retry-loops that further exhaust the quota. Alert the user and pivot to local work.

### Checking Rate Limit Status

```bash
gh api rate_limit | python3 -c "
import json, sys, datetime
d = json.load(sys.stdin)['resources']
for k in ['core', 'graphql']:
    r = d[k]
    reset = datetime.datetime.fromtimestamp(r['reset']).strftime('%H:%M:%S')
    print(f'{k}: {r["remaining"]}/{r["limit"]} remaining — resets {reset}')
"
```

<!-- END FLEET-MANAGED: network-api-hygiene -->

## Closing issues — non-negotiable rule

NEVER close a feature or bug issue without one of:

1. A merged PR that demonstrably implements the acceptance criteria (use `Closes #N` in the PR description), OR
2. An explicit `wontfix`, `roadmap`, `duplicate`, or `invalid` label.

The Verify-Issue-Closure workflow will automatically reopen any issue closed without evidence. Do not work around it.

When implementing an issue:

- Write or update tests FIRST (TDD: red → green → refactor)
- Add Design-by-Contract preconditions/postconditions where it clarifies invariants
- Respect Law of Demeter — don't reach through three layers of objects
- Don't duplicate code (DRY)
- Run the tests locally before pushing; don't rely on CI to find basic breakage
- If you can't fully implement, leave the issue open and post a status comment instead of closing

---

<!-- BEGIN FLEET-MANAGED: repo-context-codemap -->

## 🧭 Repo Context & Codemap Freshness

> This section is managed centrally by Repository_Management and synced fleet-wide.
> Do NOT edit it directly in individual repositories — edit the source in Repository_Management/AGENTS.md.

Use repo-local context before broad exploration:

- Read `AGENTS.md` first, then check `docs/codemap.md` or `docs/operations/codemap_freshness_runbook.md` when present.
- If `.codemap/` exists, treat it as a generated local cache for navigation; verify important claims against source files before editing.
- If `.codemap/` is missing or stale, use source search (`rg`), focused file reads, and tests as the fallback. Report the missing/stale index as a rollout gap instead of blocking unrelated work.
- Do not commit `.codemap/` or `.codemap/index.db`. Codemap indexes are cache/artifact data and must stay ignored.
- To audit local fleet posture, run `python -m scripts.codemap_context_inventory --root .. --format markdown` from `Repository_Management`. This is a local, network-free inventory; it is not a substitute for repo-specific validation.

<!-- END FLEET-MANAGED: repo-context-codemap -->
