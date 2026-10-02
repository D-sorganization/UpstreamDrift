# Force and Torque Overlay Epic — #11285

Planning record for the epic **Force and Torque Overlays for Every Engine,
Tension/Compression Shading and Source-Video Overlays**. The architecture is
fixed by [ADR-0052](../adr/0052-force-torque-overlay-contract.md). This document
records the 2026-10-02 assessment, the child issues (FTO-1 … FTO-30), their
dependency order, and the rules every implementing agent follows.

## 1. Goal

For **Simscape, MuJoCo, Drake, Pinocchio and OpenSim**, we want to:

- see the joint torques, joint reaction forces, contact/ground reaction forces,
  grip forces and (OpenSim) muscle forces as sharp 3D arrows and torque arcs;
- shade every segment by **tension (blue) / compression (red)**;
- draw the same arrows, the model, and the shading **over source footage**.

## 2. Assessment — How Close Are We?

| Area                           | State                                                                                                                                                                      | Evidence                                                                                                                                                                                                                              |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tension/compression policy     | **Shipped**: shared policy, colour scale, Qt/Matplotlib/PyQtGraph/React adapters                                                                                           | #9833 / PR #9840; `src/shared/python/body_part_viz/force_colors.py`, `force_display.py`, `axial_loads.py`; `docs/user_guide/body_part_viz/force_colors.md`                                                                            |
| Tension/compression producers  | MuJoCo (`MujocoAxialLoadSource`) and pendulums only                                                                                                                        | Drake/Pinocchio GUIs install the menu but never `bind()`/`set_frame()`; Simscape `set_segment_axial_loads` has no production caller; OpenSim has no producer                                                                          |
| Shared force/torque data model | **Missing**: seven parallel vector types, none filled by engines                                                                                                           | `ForceVector3D`, `engines/common.ForceOverlay`, `unreal_integration.ForceVector`, `movement_optimizer.ForceArrow`, `ViewportOverlayPayload.wrench`, two TS shapes. `motion_matching/force_torque.py::SpatialWrench` is the right base |
| Arrow rendering                | At least 10 duplicated builders; shared matplotlib ones have no callers; MeshCat "arrows" have no heads                                                                    | `plotting/renderers/force_vectors.py`, `vectors.py`, `sim_rendering_mixin.py`, `meshcat_adapter.py`, two Pinocchio mixins, `drake_gui_viz.py`, pendulum widgets                                                                       |
| MuJoCo data                    | Richest native data, but overlay code has bugs                                                                                                                             | `xaxis[3*j:3*j+3]` slicing (5 sites); `cfrc_*` read without `mj_rnePostConstraint`; torque from `ctrl`; contact points dropped                                                                                                        |
| Drake data                     | Inverse dynamics only; GUI "forces" are gravity                                                                                                                            | reaction-force, net-actuation and hydroelastic outputs unused                                                                                                                                                                         |
| Pinocchio data                 | `data.f` computed in GUI only; adapter contact = zeros                                                                                                                     | one mixin draws local-frame wrenches as world                                                                                                                                                                                         |
| OpenSim data                   | Inverse dynamics + scalar muscle forces; no reactions or contact read                                                                                                      | `HuntCrossleyForce` built but never read; grip forces placeholder `{}`                                                                                                                                                                |
| Simscape data                  | **Rich force logs exist** (per-joint constraint/total F/T, hand-on-club, base-on-hip) in the `.slx` sensing and the exported CSV datasets; Python adapter reads only `tau` | `3D_Golf_Model/golf_swing_dataset_20250907_bk/*.csv`; `src/engines/simscape/_simscape_io.py`. Per-joint "Local" columns come with `<J>Logs_Rotation_Transform_I11..I33` (global = R·local, as in `calculateForceMoments.m`)           |
| Web                            | Three.js arrows exist, fed by a route that **fabricates geometry**; round trip drops `force_type`                                                                          | `src/api/routes/force_overlays.py`, `ui/src/pages/Simulation.tsx`, `Scene3D.tsx`                                                                                                                                                      |
| Video                          | Calibrated cameras, time maps and footage compositors exist; **no vector projection**; trace import drops torques/wrench; no calibrated engine render                      | `motion_capture/reconstruct/cameras.py`, `reference/registration.py`, `reference/synchronization.py`, `tools/capture_rig/reference_rendering.py`, `overlay_render.py`, `workspace/necromatcher_video.py`                              |

**Summary:** the colour policy and the video camera stack are ready. The work is:

- one shared physics contract;
- one geometry builder;
- thin renderer adapters;
- a real provider in each engine;
- the video glue.

## 3. Architecture (ADR-0052)

```
engine adapter ──► ForceTorqueFrame ──► GlyphSet ──► renderer adapter
 (native arrays)    (physics, world)    (geometry)    (MeshCat / MjvScene / matplotlib /
                                                       QPainter / OpenCV-on-video / three.js)
```

- Package: `src/shared/python/force_overlay/`. It contains:
  - `contracts.py`;
  - `conversions.py`;
  - `glyphs.py`;
  - `renderers/`;
  - `schemas/force-torque-frame-v1.json` plus shared fixtures.
- Engine SDK imports stay in `src/engines/**`.
- Tension/compression for non-MuJoCo engines comes from **one** converter,
  `axial_loads_from_reactions`, applied to `JOINT_REACTION` wrenches.
- Video reuses `PinholeCamera` / `project_reference_to_camera`, `TimeMapping`
  and the existing compositors. No second camera or clock is introduced.

## 4. Graphics Quality Bar ("Sharp Graphics")

Every renderer child must meet this bar. Its golden images prove it.

1. **Real 3D arrows:** a shaft plus a cone head (MeshCat / MjvScene / three.js).
   Torque is a 270° arc with an arrowhead, using the right-hand rule. Line-only
   "arrows" are not acceptable.
2. **Colour:** categorical colours come from `FORCE_KIND_PALETTE` (ADR-0052).
   Pure blue and pure red are reserved for tension and compression.
3. **Legend:** every view shows a reference arrow (for example "500 N") and a
   reference arc (for example "50 N·m"), with units, the kind swatches, and the
   engine/source label.
4. **Long vectors:** these are clamped to `max_length_m` and drawn with a
   dashed tail marker, so a clamped vector never misleads.
5. **Video:**
   - anti-aliased (`cv2.LINE_AA`);
   - 1–2 px dark halo;
   - sizes scale with frame height (reference 1080 px);
   - legend in a semi-transparent corner box;
   - glyphs behind the camera are omitted and counted in the receipt.
6. **Unavailable data** shows as "unavailable" in the legend. It is never drawn
   as zero.

## 5. Child Issues and Dependency Order

| FTO | Issue  | Slice                                                                              | Tier   | Blocked by                 |
| --- | ------ | ---------------------------------------------------------------------------------- | ------ | -------------------------- |
| 1   | #11286 | Contract module, wire schema and shared fixtures                                   | cli    | —                          |
| 2   | #11287 | Shared conversions (joint torque → moment, local → world, reactions → axial loads) | cli    | 1                          |
| 3   | #11288 | Glyph builder, style and `FORCE_KIND_PALETTE`                                      | cli    | 1                          |
| 4   | #11289 | Colour utilities DRY and tension/compression colormap registration                 | cli    | —                          |
| 5   | #11290 | MeshCat glyph renderer                                                             | cli    | 3                          |
| 6   | #11291 | MuJoCo `MjvScene` glyph renderer                                                   | cli    | 3                          |
| 7   | #11292 | Matplotlib 3D and QPainter 2D glyph renderers; retire duplicate arrow code         | cli    | 3                          |
| 8   | #11293 | OpenCV video glyph renderer (calibrated projection)                                | cli    | 3                          |
| 9   | #11294 | MuJoCo force/torque provider and overlay bug fixes                                 | cli    | 1, 2                       |
| 10  | #11295 | MuJoCo GUI rewiring (native + MeshCat)                                             | cli    | 5, 6, 9                    |
| 11  | #11296 | Drake force/torque provider and axial loads                                        | cli    | 1, 2                       |
| 12  | #11297 | Drake GUI rewiring and segment force colours                                       | cli    | 5, 11                      |
| 13  | #11298 | Pinocchio force/torque provider and axial loads                                    | cli    | 1, 2                       |
| 14  | #11299 | Pinocchio GUI consolidation and segment force colours                              | cli    | 5, 13                      |
| 15  | #11300 | OpenSim torques, joint reactions, contacts and axial loads                         | cli    | 1, 2                       |
| 16  | #11301 | OpenSim muscle lines of action                                                     | cli    | 15                         |
| 17  | #11302 | OpenSim animated playback with arrows and segment shading                          | cli    | 7, 15                      |
| 18  | #11303 | Simscape Python force loader (CSV/.mat → `ForceTorqueSeries`)                      | cli    | 1, 2                       |
| 19  | #11304 | Simscape simulation-output force channels (`SimscapeOutput`, R2025b host)          | cli    | 18                         |
| 20  | #11305 | Simscape 3D viewer arrows and segment shading                                      | cli    | 7, 18                      |
| 21  | #11306 | Cross-engine physical parity suite for forces                                      | cli    | 9, 11, 13, 15, 18          |
| 22  | #11307 | API and WebSocket: real provider frames replace fabricated geometry                | cli    | 1, 3, 9                    |
| 23  | #11308 | Web three.js overlay rendering the serialized `GlyphSet`                           | cli    | 22                         |
| 24  | #11309 | Force series through trace import, time maps and video frames                      | cli    | 1                          |
| 25  | #11310 | Arrow layer in the reference-comparison and capture-rig video compositors          | cli    | 2, 8, 24                   |
| 26  | #11311 | Engine-agnostic model-on-footage layer with tension/compression fill               | cli    | 8, 24                      |
| 27  | #11312 | Calibrated MuJoCo mesh render composited on footage                                | strong | 8, 24                      |
| 28  | #11313 | Necromatcher force layer (after #11246)                                            | cli    | 8, 9, 24, #11246           |
| 29  | #11314 | Web video overlay: VideoAnalyzer fixes and SVG glyph layer                         | cli    | 8, 22                      |
| 30  | #11315 | Gallery, golden images, user guide and parity ledger                               | cli    | 10, 12, 14, 17, 20, 23, 25 |

**Dispatch waves.** Within a wave, issues run in parallel.

- **Wave A:** FTO-1, FTO-4.
- **Wave B:** FTO-2, FTO-3, FTO-24.
- **Wave C:**
  - Renderers: 5, 6, 7, 8.
  - Providers: 9, 11, 13, 15, 18.
- **Wave D:**
  - GUIs: 10, 12, 14, 17, 20.
  - 16, 19, 22.
  - Video: 25, 26, 27.
- **Wave E:** 21, 23, 28, 29.
- **Wave F:** 30.

## 6. Relationship to Existing Work (Do Not Duplicate)

- **#9833** (closed) owns the tension/compression _policy_ and colour adapters.
  This epic adds _producers_ and _wiring_ only. Never add a second colour
  formula.
- **#10482 MV-06** (closed) owns `SpatialWrench`, `transform_wrench` and
  `ContactReaction`. This epic reuses them.
- **#10286** (open, reaction-wrench / ZTCF epic): its CF-7 force/couple arrows
  must consume this epic's `GlyphSet` and renderers instead of building new
  ones.
- **#11246** (Necromatcher source-video overlays, Codex claim): FTO-28 waits
  until it merges and only adds a layer through its existing renderer.
- **#11268 / COV-10 #11278**: the video comparison path stays theirs. FTO-25
  adds a vector layer to the shared compositor that they can switch on.
- **#7452** (`simulation.controls_wiring` parity gap): FTO-22, FTO-23 and FTO-30
  supply the evidence to close the force-overlay part.

## 7. Rules for Every Child

- **Read first:**
  - `CLAUDE.md`, `AGENTS.md`, `docs/agents/shared-infrastructure.md`;
  - ADR-0052 and this document;
  - your issue.
- **Lease and worktree:**
  - Check the lease: `python -m scripts.check_agent_claim --repo UpstreamDrift --issue <N>`.
  - Post your lease.
  - Work in a worktree:
    `git worktree add ../UpstreamDrift-worktrees/<agent>-<N> -b feat/fto-<N>-<slug> origin/main`.
- **TDD:**
  - Write the issue's red tests first and paste the failing output in the PR.
  - Synthetic fixtures are named `synthetic_*`.
  - Engine-dependent tests use the existing skip markers/`importorskip`, never
    module-level `sys.modules` mocks.
- **DbC:**
  - Validate shapes, finiteness, frames, units and labels at every public
    function.
  - Raise `ValueError`/`TypeError`.
  - Document postconditions.
  - "Unavailable" is `None`, never zero.
- **LoD:** no chains deeper than `a.b.c()`. UIs call a service or provider;
  renderers see only `GlyphSet`.
- **DRY:**
  - Reuse `SpatialWrench` (its vector validation and transforms),
    `AxialLoadFrame`, `ForceColorScale`, `PinholeCamera`, `TimeMapping` and the
    compositors.
  - When you replace a duplicate, delete private helpers and their now-dead
    tests in the same PR. A **public, importable** module or function that
    moves keeps a thin shim at the old path: a one-line re-export plus a
    `DeprecationWarning`, for one release cycle (AGENTS.md "relocating
    something").
  - Remote clients render the serialized `GlyphSet` (`glyph-set-v1`). They
    never reimplement the glyph algorithm.
- **Headless:** `QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg`,
  `MUJOCO_GL=egl` (or `osmesa`). MATLAB runs only with `-batch` on an R2025b
  host by explicit path.
- **Gates:** never loosen a tolerance. Golden-image updates need a rendered
  before/after in the PR body.
- **Validate before pushing:**
  - `python3 -m ruff check .`
  - `python3 -m ruff format --check .`
  - focused `python3 -m pytest <tests> -n auto --timeout=60`
  - `python3 scripts/ci/check_file_size_budget.py`
  - `python3 scripts/ci/check_error_handling_ratchet.py`
  - for `ui/`: `npm run lint`, `npm run typecheck` (or the repo's equivalents)
    and `npx vitest run <tests>`
- **Records:**
  - Every commit updates `docs/development/HANDOFF.md` and the `DL-#11285`
    development-log entry.
  - Add one SPEC.md §12 row keyed by your PR.
  - User-facing changes update `src/config/feature_parity.json` and regenerate
    the matrix.
  - Open a **ready-for-review** PR, never a draft (AGENTS.md,
    Repository_Management#1390), with `Refs #11285`. Use `Closes #<N>` only
    when every acceptance box passes. A frontier agent or the owner reviews it
    before merge.
