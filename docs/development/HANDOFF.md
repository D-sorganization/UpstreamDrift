# Matplotlib 3D and QPainter 2D Glyph Renderers Delivery — #11285 / #11292 (FTO-7)

- Repository: `D-sorganization/UpstreamDrift`; branch `feat/fto-11292-matplotlib-qpainter-glyphs`; commit SELF; PR: #11348 (`Closes #11292`, `Refs #11285`)
- Governing issue: #11292 (parent epic #11285, design authority ADR-0052 §5 and `force_torque_overlay_epic.md`)
- Objective: [FTO-7] Matplotlib 3D and QPainter 2D glyph renderers with dark halos, 12-facet cone heads, and migrated legacy vector overlays.
- Completed:
  - `src/shared/python/force_overlay/renderers/matplotlib_glyphs.py`:
    - `draw_glyphs_3d(ax, glyphs, *, linewidth_pt=2.0, halo=True) -> list[Artist]`: draws 3D force arrows (with 12-facet cone heads via `Poly3DCollection`) and 3D torque arcs (polyline + cone head) onto a Matplotlib 3D axes, optionally underlaid with a dark halo. Returns list of created artists supporting `.remove()`.
    - `draw_legend(ax, glyphs, *, loc='upper right', fontsize=9.0) -> Artist`: renders deterministic legend showing active force/torque kinds.
  - `src/shared/python/force_overlay/renderers/qpainter_glyphs.py`:
    - `draw_glyphs_2d(painter, project, glyphs, *, px_width=2.0, halo=True) -> None`: draws 2D projected force arrows and torque arcs using QPainter with anti-aliasing and optional dark halos.
  - Migrated legacy vector renderers:
    - `src/shared/python/plotting/renderers/force_vectors.py`: delegates to `draw_glyphs_3d`.
    - `src/shared/python/movement_optimizer/gui/vector_overlay.py`: delegates to `draw_glyphs_2d`.
  - Tests:
    - Unit tests in `tests/unit/force_overlay/test_matplotlib_glyphs.py` and `tests/unit/force_overlay/test_qpainter_glyphs.py`.
- Validation:
  - Ruff check and format clean.
  - Pytest passed.
- Next steps: Merge PR #11348; unblocks remaining renderers.

# MuJoCo MjvScene Glyph Renderer Delivery — #11285 / #11291

- Repository: `D-sorganization/UpstreamDrift`; branch: `feat/fto-11291-mujoco-glyphs`; commit: a9515fb04e; PR: #11355 (merged; `Closes #11291`, `Refs #11285`)
- Governing issue: #11291 (parent epic #11285, design authority ADR-0052 §5 and `force_torque_overlay_epic.md`)
- Completed:
  - `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/force_glyphs.py`:
    - `SceneGlyphReceipt(added: int, dropped: int)`: frozen dataclass reporting geoms added and dropped.
    - `segment_geom_count(glyphs: GlyphSet) -> int`: pure function returning exact geom capacity required for arrows and torque arc capsules/heads.
    - `add_glyphs_to_scene(scene: mujoco.MjvScene, glyphs: GlyphSet, *, arc_width_m: float = 0.006) -> SceneGlyphReceipt`:
      - Appends `mjGEOM_ARROW` connectors for `ArrowGlyph` using 2·shaft_radius_m.
      - Appends `mjGEOM_CAPSULE` connectors along the polyline of `TorqueArcGlyph` plus a final `mjGEOM_ARROW` connector from `head_base_m` to `head_tip_m`.
      - Detects and adapts both modern `mujoco.mjv_connector` and legacy `mujoco.mjv_makeConnector`.
      - Enforces strict buffer overflow protection (`scene.ngeom < scene.maxgeom`) and records dropped geoms without exceptions or out-of-bounds writes.
  - `tests/unit/engines/mujoco/test_force_glyphs.py`:
    - 5 tests covering 1-arrow geom and endpoint matching within 1e-9, 32-segment arc (32 capsules + 1 arrow), overflow recording with capacity bounds, `segment_geom_count`, and offscreen pixel rendering with `mujoco.Renderer` verifying arrow color detection.
  - Updated `SPEC.md` §12 changelog table row.
- Validation:
  - Ruff check and format clean.
  - Pytest 5/5 passed.


# MuJoCo 3.14 Axial-Load Axis Discovery — #11349

- Repository: `D-sorganization/UpstreamDrift`; worktree: `/home/user/ud-wt/mj314`; branch `fix/mujoco-314-axial-load-axis`; commit: SELF; PR: see the PR for this branch (`Closes #11349`).
- Completed: `MujocoAxialLoadSource._discover_axes` compares geom and joint types via `int()` (numpy int vs pybind enum `in`/`==` is direction-dependent and False on mujoco 3.14, so no rods were found and `sample()` returned None). Regression tests for capsule/cylinder discovery and free/non-rod exclusion in `tests/unit/body_part_viz/test_mujoco_axial_loads.py`.
- Validation: `pytest tests/unit/body_part_viz/test_mujoco_axial_loads.py` 9 passed on mujoco 3.14.0. Failures outside scope here: tests needing PyQt6 (not installed in this venv).
- Next steps: none for this fix; other mujoco 3.14 drift is listed in the PR body.

# OpenSim Engine State and Control Setters Under OpenSim 4 — #11344

- Repository: `D-sorganization/UpstreamDrift`; worktree: `/home/user/ud-wt/11344`
- Branch: `fix/issue-11344-opensim-set-state`; commit: SELF; PR: #11352 (merged; `Closes #11344`, `Refs #11285`); DL entry `DL-#11285`
- Completed: `OpenSimPhysicsEngine.set_state` builds vectors with `opensim.Vector(list)` (4.x has no `Vector(int)`), keeps time, realizes Velocity and raises `ValueError` on length mismatch. `set_control` goes through `Model.setControls` (a bare `updControls` + `markControlsAsValid` does not invalidate an already-realized Dynamics stage, so actuation stayed 0); the values are retained in `self._controls` and re-applied by `set_state` because changing q/u drops realized controls. ZTCF/ZVCF snapshot and restore that retained value. `tests/unit/engines/opensim/test_opensim_set_state_control.py` (10 live tests, including a nonzero actuator torque in the force/torque frame) and `tests/integration/cross_engine/test_opensim_engine_state_control_parity.py` (engine set_state/set_control gives the same force/torque frame as a model with state and PrescribedController baked in).
- Known limits: `step()` integrates through the Manager, which recomputes controls from the model controllers, so a `set_control` value does not persist across steps without a controller. Not fixed here: `reset()` calls `Manager.setSessionTime` (absent in 4.x), `compute_inverse_dynamics` uses `Vector(n_u)` (so `compute_gravity_forces`/`compute_bias_forces` return empty arrays on 4.x). The parity builders in `tests/integration/cross_engine/test_force_overlay_parity.py` (PR #11345) can switch from baked-in state/controller to `set_state`/`set_control`.
- Validation: `python3 -m pytest tests/unit/engines/opensim` passes with the new file; the wider opensim/analytical/audit set shows the same 50 failures before and after (pre-existing in this environment).
- Next steps: fix `reset()` and the inverse-dynamics `Vector` call in a follow-up; update the parity builders after #11345 merges.

# Colour Utilities DRY — #11289 (FTO-4)

- Repository: `D-sorganization/UpstreamDrift`; worktree: `/home/dieterolson/Repositories/UpstreamDrift-worktrees/antigravity-11289`
- Branch: `feat/fto-11289-colour-utils-dry`; commit: SELF; PR: #11338
- Governing issue: #11289 (parent epic #11285, design authority ADR-0052 §1 and `force_torque_overlay_epic.md`)
- Objective: [FTO-4] Colour utilities DRY: one hex/RGBA helper and a registered tension/compression colormap.
- Completed:
  - `src/shared/python/plot_style/color_utils.py`:
    - `hex_to_rgba`: parses `#RGB`, `#RGBA`, `#RRGGBB`, `#RRGGBBAA` formats with strict length validation, case-insensitivity, and optional override alpha `[0.0, 1.0]`. Returns normalized 4-tuple float RGBA in `[0.0, 1.0]`.
    - `rgba_to_hex`: formats normalized RGBA or RGB sequences into 7-character `#rrggbb` hex strings.
    - Re-exported both in `src/shared/python/plot_style/__init__.py`.
  - Registered `ColormapId.TENSION_COMPRESSION = "tension_compression"` in `src/shared/python/plot_style/colormaps.py` with `TENSION_COMPRESSION_STOPS = ((0.0, "#2166AC"), (0.5, "#F7F7F7"), (1.0, "#B2182B"))`.
  - Registered colormap in `src/shared/python/plot_style/registry.py` with `LinearSegmentedColormap(name, ..., N=257)` to guarantee exact neutral center sampling at position 0.5.
  - Replaced ad-hoc parsers in `meshcat_force_colors.py`, `mujoco_force_colors.py`, `pyqtgl_renderer.py`, `_viewer_3d_segments.py`, and `meshcat_adapter.py`.
  - Replaced hardcoded default hex colors in `src/shared/python/body_part_viz/force_colors.py` with registered constants `DEFAULT_TENSION_COLOR`, `DEFAULT_COMPRESSION_COLOR`, and `DEFAULT_NEUTRAL_COLOR`.
  - Documented signed quantity convention in `kinetics.py` and added `TENSION_COMPRESSION` section to `docs/user_guide/plot_style/colormap_author_guide.md`.
  - 20 unit tests in `tests/unit/plot_style/test_color_utils.py` and `tests/unit/plot_style/test_colormaps.py`.
- Next steps: Wave B child issues: FTO-5 (#11290) PySide/PyQtGL overlay renderer and FTO-24 (#11309) video camera projection.

# Glyph Builder and Force Kind Palette — #11288 (FTO-3)

- Repository: `D-sorganization/UpstreamDrift`; worktree: `/home/dieterolson/Repositories/UpstreamDrift-worktrees/antigravity-11288`
- Branch: `feat/fto-11288-glyph-builder`; commit: SELF; PR: #11288
- Governing issue: #11288 (parent epic #11285, design authority ADR-0052 §2-§4 and `force_torque_overlay_epic.md`)
- Objective: [FTO-3] Glyph builder: ForceGlyphStyle, build_glyphs, scale_for_view and FORCE_KIND_PALETTE.
- Completed:
  - `FORCE_KIND_PALETTE` registered in `src/shared/python/plot_style/colors.py` with 7 categorical hex colors compliant with ADR-0052 (strictly avoiding pure blue `#0000ff` and pure red `#ff0000` reserved for axial tension/compression). Exported from `src/shared/python/plot_style/__init__.py`.
  - Palette documented in `docs/user_guide/plot_style/colormap_author_guide.md` under `### Force and Torque Overlay Palette (ADR-0052)` with hex swatch table and rationale.
  - Implemented `ForceGlyphStyle`, `ArrowGlyph`, `TorqueArcGlyph`, `LegendSpec`, `GlyphSet`, `build_glyphs`, and `scale_for_view` in `src/shared/python/force_overlay/glyphs.py` (387 lines, within 400-line budget, LoD <= 2, DbC validation on inputs and postconditions).
  - Wired into `src/shared/python/force_overlay/__init__.py` with headless import guards and clean `__all__`.
  - Wire schema `schemas/glyph-set-v1.json` (Draft 2020-12) and generator `scripts/generate_glyph_set_examples.py` producing 4 synthetic fixture cases in `schemas/glyph-set-examples.json`.
  - 14 comprehensive unit tests in `tests/unit/force_overlay/test_glyphs.py` and `tests/unit/force_overlay/test_glyph_serialization.py` covering styling, clamping, right-hand torque arcs, view scaling, schema validation, and headless import purity.
- Validation:
  - `python3 -m ruff check src/shared/python/force_overlay/ tests/unit/force_overlay/ src/shared/python/plot_style/`: 0 violations.
  - `python3 -m ruff format --check src/shared/python/force_overlay/ tests/unit/force_overlay/ src/shared/python/plot_style/`: 0 diffs.
  - `python3 -m pytest tests/unit/force_overlay -n auto --timeout=60`: 39 passed.
  - `python3 scripts/ci/check_file_size_budget.py`: OK.
  - `python3 scripts/ci/check_architecture_budget.py`: OK.
  - `python3 scripts/ci/check_error_handling_ratchet.py`: OK.
  - `python3 scripts/ci/check_lod.py src --baseline scripts/ci/lod_baseline.txt`: OK (clean no-growth scan, 0 new violations).
- Next steps: Wave C renderers (FTO-5 MeshCat, FTO-6 MjvScene, FTO-7 Matplotlib/QPainter, FTO-8 OpenCV Video) consuming serialized `GlyphSet`.

