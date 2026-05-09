# AGENTS.md — Discovery Workflow & Shared-Infrastructure Directory

> **Read this first.**  This file exists because we kept reinventing
> infrastructure that already lived in the repo (FK solvers, skeleton
> renderers, reference golfer poses, mocap loaders, theme constants).
> Tracked as issue #4377; updated as we discover new shared modules.

For binding repo policy (ruff, file-size budgets, branch naming, PR
targets, etc.) see [`CLAUDE.md`](CLAUDE.md).  For background and
historical agent-workflow notes see
[`docs/development/agents.md`](docs/development/agents.md).

This file is narrower: a **discovery workflow** for agents and a
**directory of shared infrastructure** so we stop duplicating work.

---

## A. Before you write new code — discovery workflow

When you're about to add functionality, run these five steps **in
order** before writing a line:

1. **Grep `src/shared/python/` for the concept.**  Most cross-engine
   primitives live there (FK, reference poses, mocap loaders, theme
   constants, club-target structures, optimisation drivers).
2. **Grep `src/tools/` and `src/launchers/` for similar tools.**  If a
   tile already does ≥60 % of what you want, extend it instead of
   forking.
3. **Read the relevant `__init__.py`** (public API surface) before
   importing internals.  The `motion_matching/diagnostics/__init__.py`
   in particular is a curated façade that hides private helpers.
4. **Read the docs.**  `docs/` has design notes for the bigger systems
   and `MATLAB_GOLF_MODEL_GUIDE.md` documents pitfalls (e.g. the
   cm-vs-inches Wiffle xlsx gotcha).
5. **Only then** propose new code.  If you find yourself
   reimplementing something that lives in `src/shared/`, stop and use
   the shared version — even if it forces minor API massaging on your
   side.

> **The "I'll just write it inline, it's only 20 lines" trap is real.**
> Twenty lines today becomes a divergent re-implementation tomorrow
> when the shared module changes its sign convention.

---

## B. Shared-infrastructure directory

Grouped by concern, with one-line descriptions and the public symbols
worth knowing.  Not exhaustive — when in doubt, grep — but these are
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

### Motion matching (the big one)
`src/shared/python/motion_matching/`
- `target.py`, `club_target.py`, `load_club_target.py`,
  `loaders/` — mocap target structures and loaders (xlsx / C3D / JSON).
- `BodyTarget`, `ClubBallTarget`, `MultiSourceTarget` —
  `src/shared/python/motion_matching/`.  Frozen dataclasses + an
  aggregator covering club, ball-aware, and full-body capture
  targets.  Cost-function code dispatches on `has_club()`,
  `has_ball()`, `has_body()`.  See
  [ADR 0006](docs/adr/0006-multi-source-motion-targets.md).
- `load_body_target`, `load_club_target` — format-agnostic dispatcher
  loaders in `src/shared/python/motion_matching/`.  Route on file
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
    butt → clubhead).  Takes a Simulink-Parameter-style angle dict
    (degrees) and returns a `SkeletonPose` (Cartesian metres).
  - `reference_pose.reference_golfer_setup()` — canonical Address-pose
    joint angles.  Single source of truth.
  - `reference_pose.compare_to_reference(angles)` — flag joint angles
    outside the plausible Address range.
  - `_skeleton_render.{draw_segments, draw_delta_arrows, equalize_3d_axes}`
    — matplotlib helpers for 3D skeleton overlays.  **Use these instead
    of hand-rolling matplotlib boilerplate.**
  - `clubhead_trace.{compare_clubhead_traces, plot_3d_overlay,
    plot_setup_pose_skeletons}` — canonical clubhead-trace comparison.
  - `initial_state_diff.{plot_skeleton_overlay, plot_per_joint_delta_bars,
    plot_cartesian_delta_summary, summarize_for_pr_comment}` —
    input-MAT requested-vs-resolved diagnostics.

### Body-part visualisation toolkit
`src/shared/python/body_part_viz/`
- `body_part_viz` package with shapes / fitters / renderers / asset
  library — the **canonical shape stack** for any tool that draws body
  segments.  See [ADR 0008](docs/adr/0008-body-part-viz-toolkit.md).
- `BodyPartShape`, `ShapeFitter`, `ShapeRenderer` — runtime-checkable
  Protocols.  Implementations live under `shapes/`, `fitters/`,
  `renderers/`.
- `MatplotlibRenderer` — **canonical 3D renderer for any new tool that
  needs marker / mesh rendering.**  A `PyQtGLRenderer` ships alongside
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

### Pose editor (interactive joint-angle UI)
`src/shared/python/pose_editor/`
- `core.{JointType, JointInfo, PoseEditorState}` — joint metadata
  dataclasses; engine-agnostic.
- `widgets` — slider/spinbox composites for editing joints.
- `library` — preset poses.
- Used by per-engine GUIs (MuJoCo, Drake) for live joint editing.
  **Not** the same as the *starting-pose matcher* (which solves a
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
  "Definitions" tab claiming inches.  The MATLAB loader
  (`load_club_target_excel.m` line 53: `CM_TO_METRES = 0.01`) is the
  source of truth — see `MATLAB_GOLF_MODEL_GUIDE.md`.

### Logging / config
- `src/shared/python/logging_pkg/logging_config.get_logger(__name__)`
  — canonical logger factory.  Don't `import logging; logging.getLogger(...)`
  directly — the shared factory pulls in JSON config, log rotation,
  and fleet-aware filters.

### Launcher base
- `src/launchers/base.BaseLauncher` — `QMainWindow` subclass for
  **grid-of-tiles launcher windows** (the main GolfLauncher, sub-
  launchers showing a card grid).  **Not** for single-purpose tool
  windows; those should be standalone `QMainWindow` subclasses
  registered as tiles in `src/config/models.yaml`.

### Rust kernels
- `rust_core/upstream-physics/` — RK4 integrator, aerodynamics, contact,
  swing-plane fit.  Built via `maturin develop`; consumed by Python
  via `src/shared/python/physics/rust_kernel.py` (which falls back to
  pure Python if the Rust wheel isn't installed).  See section F for
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
references keep working through one release cycle.  Example: see the
shims at
`src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Motion Capture Plotter/starting_pose_*.py`
which redirect to the new `src/tools/starting_pose_matcher/` package.

---

## D. Tile-launcher contract

The main GolfLauncher reads `src/config/models.yaml` and renders one
tile per entry.  A tile entry needs:

```yaml
- id: "starting_pose_matcher"
  name: "Starting-Pose Matcher"          # display name
  description: "..."                      # one-line tooltip
  type: "special_app"                     # other types: model, custom_humanoid, drake, ...
  path: "src/tools/starting_pose_matcher/__main__.py"  # relative to REPO_ROOT
  launcher:
    category: "tool"                      # tool | physics_engine | document
    logo: "assets/<icon>.png"             # optional; falls back to default tile art
    status: "ready"                       # ready | beta | broken
```

Tiles are launched as `subprocess.Popen` with the system Python.  Any
tool listed here should be runnable as `python -m <package>` from the
repo root — the launcher resolves the path, prepends the repo root to
`sys.path`, and invokes the module directly.

If your tool has runtime dependencies beyond core (PyQt6, matplotlib,
pandas, openpyxl), declare them in a named `[project.optional-dependencies]`
extra in `pyproject.toml` and reference the install command in the
tile's `description`.  Example: the `gui-tools` extra installs PyQt6 +
matplotlib + pandas + openpyxl for the desktop tools.

---

## E. Tests

Tests go in `tests/<area>/<subarea>/test_*.py` — mirror the source
layout under `src/`.  See `tests/README.md` and `tests/conftest.py` for
the full set of markers and fixtures.

### Markers (from `pyproject.toml`)
- `unit` — fast, deterministic, no engines.
- `integration` — exercises 2+ modules together.
- `slow` — > 5 s runtime; skipped from default CI lane.
- `live_simulation` — actually runs MuJoCo / Drake / Simscape; **always
  skipped** in default `pytest` runs.  Opt in with `-m live_simulation`.
- `requires_gl`, `requires_mocap_fixtures` — skipped on CI fleet that
  lacks the resource.
- Engine-specific: `requires_mujoco`, `requires_drake`, `requires_pinocchio`,
  `requires_opensim`, `requires_matlab`.

### Optional GUI deps
PyQt6 / matplotlib are optional.  Tests that need them MUST skip
cleanly when the import fails — `pytest`'s user-site Python on Windows
sometimes has a broken PyQt6 DLL search path even when the regular
interpreter loads it fine.  Pattern:

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
  swing-plane.  Inner-loop numerics; used identically by the Python
  backend (`src/shared/python/physics/rust_kernel.py`) and the WASM
  browser frontend.
- `ui/src-tauri/golf-modeling-suite` — Tauri desktop shell.  Spawns the
  Python API server; **no physics or GUI logic in Rust**.

The criteria that produce a Rust crate in this project:

1. **Profiling shows a sub-millisecond inner loop** that gets called
   thousands of times per simulation, AND
2. The same kernel needs to run identically in Python AND another
   runtime (WASM, embedded C++, etc.).

Notably **NOT** Rust-shaped:

- GUI tools.  PyQt6 is faster to iterate, has a far richer ecosystem
  for desktop scientific UIs, and binds the matplotlib stack we already
  use everywhere else.
- Data pipelines.  pandas / numpy / scipy already cover everything we
  need at speeds that don't bottleneck.
- Config, orchestration, tests.  These should stay in Python.

If you're tempted to rewrite something in Rust, profile first and
verify that the workload genuinely lives in inner loops.  Most of the
time you'll find the bottleneck is matplotlib's 3D renderer or pandas'
xlsx parser, neither of which Rust can help with.

---

## G. Lessons learned (this section grows)

- **Wiffle xlsx is in CM, not inches.**  The "Definitions" tab of the
  workbook is wrong; trust `load_club_target_excel.m` (line 53:
  `CM_TO_METRES = 0.01`).
- **The Simscape body chain has a `torso` joint** between spine and hub
  that hosts the revolute Z (twist) joint.  Don't collapse to
  `hip → spine → hub` directly — you'll lose the visible torso coil.
  See
  `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/mdl_reference/GolfSwing3D_Kinetic.mdl`
  block "Torso Kinetically Driven" (SID 8331).
- **Don't subclass `BaseLauncher` for single-purpose tools.**
  `BaseLauncher` is for grid-of-tiles launcher windows.  Standalone
  tools should be plain `QMainWindow` subclasses registered as tiles.
- **Test-environment pip vs. interpreter mismatch.**  If pytest under
  `C:\Python314\python.exe -m pytest` fails to load PyQt6 with
  `DLL load failed`, it's because pytest is being resolved from
  `AppData\Roaming\Python\Python314` (user-site).  Tests should skip
  gracefully — never assume PyQt6 imports.
- **The shared FK has known asymmetry** in hand height with the
  reference Address pose.  When deriving fallback skeletons, use FK
  for "the body shape" (torso, shoulders) but be cautious about
  hand heights — they may need hand-tuning.

---

## H. How to update this file

Found a shared module you wished you'd known about?  Add it to section
B with a one-liner.  Found a new trap?  Add it to section G.  Keep
the file focused — if it's growing past 400 lines, split sub-pages
into `docs/agents/`.

After adding to section B, also link the change in `CLAUDE.md`'s
"Shared Code Layout" section if one exists, and tag the file in your
PR description so reviewers can verify discoverability.
