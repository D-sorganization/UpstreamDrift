# ADR-0052: Engine-Agnostic Force and Torque Overlay Contract

- Status: Proposed
- Date: 2026-10-02
- Decision Makers: repository owner (acceptance pending); proposed in the #11285 planning session
- Related Issues/PRs: epic #11285 and children #11286–#11315 (FTO-1 … FTO-30); ADR-0008, ADR-0011, ADR-0026, ADR-0027, ADR-0041; epics #9833 (segment tension/compression colours, closed), #10286 (reaction-wrench analysis), #11246 (source video overlays); MV-06 #10482 (`SpatialWrench`)

## Context

Users want to see the forces and torques acting on every model (MuJoCo, Drake,
Pinocchio, OpenSim, Simscape), shade segments by tension and compression, and
draw the same vectors over source video together with the model. A 2026-10-02
audit found:

- **Seven parallel vector types** (`ForceVector3D`, `engines/common.ForceOverlay`,
  `unreal_integration.ForceVector`, `movement_optimizer.ForceArrow`,
  `ViewportOverlayPayload.wrench`, two TypeScript shapes). No engine fills a
  shared one; each GUI reads native arrays.
- **At least ten arrow builders.** Two shared matplotlib renderers have no
  callers. The MuJoCo/Pinocchio/Drake code is copy-pasted, and its MeshCat
  "arrows" have no heads.
- **Correctness bugs:**
  - MuJoCo `xaxis[3*j:3*j+3]` slicing, and `cfrc_*` read without
    `mj_rnePostConstraint`.
  - Torque arrows drawn from `ctrl`.
  - Pinocchio local-frame wrenches drawn as world.
  - Drake "forces" show gravity only.
  - The `/simulation/forces` route fabricates geometry.
  - The web round trip drops `force_type`.
- **Tension/compression is half wired.** The policy and adapters from #9833 are
  shared, but only MuJoCo and the pendulums produce loads. Drake, Pinocchio and
  the Simscape viewer show controls that nothing feeds. OpenSim and Simscape
  have no producer.
- **Video has no arrows.** Calibrated cameras (`PinholeCamera`,
  `project_reference_to_camera`), time maps (`TimeMapping`) and footage
  compositors exist, but nothing projects vectors. The trace import drops
  `torques` and `wrench`, and no engine model is rendered through a calibrated
  camera.

`motion_matching/force_torque.py::SpatialWrench` already carries frame, point,
sign convention and SI units. `body_part_viz.AxialLoadFrame` already carries
signed axial loads (N, tension positive).

## Decision

Create one engine-agnostic pipeline in four layers. Each layer depends only on
the layer above it.

```
engine adapter ──► ForceTorqueFrame ──► GlyphSet ──► renderer adapter
 (native arrays)    (physics, world)    (geometry)    (MeshCat / MjvScene /
                                                       matplotlib / QPainter /
                                                       OpenCV-on-video / three.js)
```

### 1. Data Layer — `src/shared/python/force_overlay/contracts.py`

- `WrenchKind(str, Enum)` has these values: `JOINT_ACTUATOR`,
  `JOINT_REACTION`, `CONTACT`, `GRIP`, `EXTERNAL`, `GRAVITY`, `MUSCLE`.
- `OverlayWrench` is a frozen dataclass with these fields:
  - `kind: WrenchKind`.
  - `label: str`: a stable id such as `"joint:left_elbow"` or
    `"contact:right_foot:0"`.
  - `body: str`: the body the wrench acts **on**.
  - `wrench: SpatialWrench`: must use `application_frame == "world"`,
    `direction_convention == "applied_to_body"`, and a point at the physical
    application point (joint anchor, contact point, muscle insertion).
  - `force_available: bool` and `torque_available: bool`. An unavailable half
    is never drawn, and is never reported as zero.
  - `source: str`: the engine and method, e.g. `"mujoco:cfrc_int"`.
- `ForceTorqueFrame` is a frozen dataclass with these fields:
  - `time_s: float`.
  - `engine: str`.
  - `wrenches: tuple[OverlayWrench, ...]`.
  - `axial_loads: AxialLoadFrame | None`: reuses #9833 and is not copied.
  - `world_frame: str = "world_Zup"`, matching ADR-0026.
  - `units = {"force": "N", "torque": "N*m", "length": "m"}`.
- `ForceTorqueSeries` holds an immutable time-indexed series of frames:
  - Times are strictly increasing.
  - `frame_at(t, max_gap_s)` linearly interpolates matching labels and
    returns `None` across gaps.
  - It has `to_npz` / `from_npz` and `to_dict` / `from_dict`.
- `@runtime_checkable class ForceTorqueProvider(Protocol)` declares
  `get_force_torque_frame(self) -> ForceTorqueFrame | None`. It mirrors the
  existing `AxialLoadProvider`.
- Wire schema: `schemas/force-torque-frame-v1.json` (JSON Schema draft 2020-12).
  Python `to_dict` and TypeScript both validate against shared fixtures in
  `schemas/force-torque-frame-examples.json`, which is the same pattern as
  `force-color-examples.json`.
- Shared converters live in `force_overlay/conversions.py`. Every engine uses
  them, so no engine rewrites this math (DRY):
  - `joint_torque_wrench(label, body, tau_nm, axis_world, anchor_world, source)`
    builds a moment vector `tau·axis` at the joint anchor. It is used for
    scalar per-DOF torques; multi-DOF joints sum per-axis moments at one anchor.
  - `world_wrench_from_local(label, body, kind, force_local, torque_local,
rotation_world_from_local, point_world, source)` wraps `transform_wrench`.
  - `axial_loads_from_reactions(frame, segment_axes)` returns an
    `AxialLoadFrame`. It calls `axial_force_from_proximal_reaction` on each
    `JOINT_REACTION` wrench, using a `SegmentAxis(proximal_m, distal_m)` map.
    This is the single path to tension and compression for every engine except
    MuJoCo, which keeps its validated `MujocoAxialLoadSource`.

### 2. Geometry Layer — `src/shared/python/force_overlay/glyphs.py`

- `ForceGlyphStyle` is a frozen dataclass with these fields and defaults:
  - `force_scale_m_per_n` (default 1/1000: 1 kN draws 1 m).
  - `torque_scale_m_per_nm` (default 1/200).
  - `min_length_m`, `max_length_m`, `shaft_radius_m`, `head_length_ratio`,
    `head_radius_ratio`.
  - `torque_style: Literal["arc", "axis_double_head"]`.
  - `kinds: frozenset[WrenchKind]`, `show_labels`, `magnitude_floor_n`.
  - `palette: Mapping[WrenchKind, str]`, defaulting to `FORCE_KIND_PALETTE`.
- `build_glyphs(frame, style) -> GlyphSet` is a **pure function**:
  - No rendering imports.
  - Deterministic.
  - Clamps lengths and records the clamping in `Glyph.clamped`.
  - Skips unavailable halves.
- `GlyphSet` contains the following:
  - `arrows: tuple[ArrowGlyph, ...]`. Each `ArrowGlyph` has `tail_m`, `tip_m`,
    `head_base_m`, `radius_m`, `rgba`, `kind`, `label`, `magnitude` and
    `units`.
  - `torque_arcs: tuple[TorqueArcGlyph, ...]`. Each has `center_m`, `axis`,
    `polyline_m (n,3)`, `head`, `rgba`, `kind`, `label`, `magnitude` and
    `units`.
  - `legend: LegendSpec`: a reference arrow (e.g. 500 N, 50 N·m), the scale
    text and the kind swatches.
- Palette: `FORCE_KIND_PALETTE` is registered once in `plot_style` as a
  categorical palette. It uses colour-blind-safe Okabe–Ito hues and **never**
  pure `#0000ff` or `#ff0000`, which stay reserved for tension and compression
  fills:
  - `JOINT_ACTUATOR` `#E69F00`
  - `JOINT_REACTION` `#CC79A7`
  - `CONTACT` `#009E73`
  - `GRIP` `#56B4E9`
  - `EXTERNAL` `#000000` (light theme) / `#FFFFFF` (dark theme)
  - `GRAVITY` `#999999`
  - `MUSCLE` `#D55E00`

### 3. Renderer Adapters — `src/shared/python/force_overlay/renderers/`

Each adapter consumes only `GlyphSet` and draws no physics. Each has the
signature `render(glyphs, target) -> None` or a stateful `update(glyphs)` that
deletes stale objects.

- `meshcat_glyphs.py`: real cylinder and cone meshes behind a small
  `MeshcatSink` protocol, so one renderer serves meshcat-python (Pinocchio,
  MuJoCo) and `pydrake.geometry.Meshcat` (Drake).
- MuJoCo `MjvScene` adapter: `mjv_connector(mjGEOM_ARROW…)` into `MjvScene`
  user geoms, for the native viewer and offscreen renders. It imports
  `mujoco`, so it lives in
  `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/force_glyphs.py`
  (FTO-6), not in the shared package. The Drake MeshCat sink follows the same
  rule (FTO-5).
- `matplotlib_glyphs.py`: 3D for reports, golden images and OpenSim offline
  playback.
- `qpainter_glyphs.py`: a 2D projector-based adapter that replaces the
  duplicate pendulum and movement-optimizer arrowheads.
- `opencv_glyphs.py`: draws on video frames:
  - Projects through `PinholeCamera` / `project_reference_to_camera(...,
clip_image=False)` and clips with `cv2.clipLine`.
  - Anti-aliased (`cv2.LINE_AA`), with a 1–2 px dark halo for contrast on
    footage.
  - Handles glyphs behind the camera.
  - Draws the legend box.
- Web: `ui/src/components/visualization/ForceOverlay.tsx` renders the
  `force-torque-frame-v1` wire schema directly. Torque arcs are drawn from
  `TorqueArcGlyph` or recomputed client-side by the same algorithm, validated
  against the shared fixtures.

### 4. Engine Adapters

Each engine adapter implements `ForceTorqueProvider`. Where it can, it also
implements `AxialLoadProvider` through `axial_loads_from_reactions`. It sets
`EngineCapabilities.force_visualization` honestly. Engine SDK imports stay in
the engine adapter packages (`src/engines/...`), never in
`src/shared/python/force_overlay/`. New MuJoCo-specific code goes beside the
engine, not into `body_part_viz`.

### Semantics That Renderers and Reviewers Rely On

- **World frame:** Z-up, SI units, and the point is the physical application
  point. Video adapters convert to the ADR-0041 y-up frame through the existing
  `canonical_z_up_to_adr0041_world` and `ReferenceRegistration`, which:
  - rotate vectors;
  - transform points;
  - scale glyph lengths with the registration scale;
  - never change magnitudes.
- **Sign:** every wrench is the wrench **applied to `body`**. A
  `JOINT_REACTION` is the parent-on-child reaction at the joint anchor, which
  is consistent with `axial_force_from_proximal_reaction`.
- **Unavailable is not zero.** The contract never stores fabricated zeros. A
  test asserts that every provider returns `None` (or omits the label) when the
  engine cannot compute a quantity.
- **Generalized vs Cartesian:** joint torques become moment vectors only
  through `joint_torque_wrench` with the engine's world joint axis. A
  generalized torque on a free or ball joint without axes is reported
  unavailable, never guessed.
- **Qualification:** overlays are visualization. Necromatcher, single-view
  video or synthetic sources carry the existing qualification labels. No
  overlay may claim a validated physical measurement.

## Alternatives Considered

1. **Extend `ViewportOverlayPayload.wrench` (T, 6).** Rejected because it has
   one wrench per frame, no application points, no kinds and no availability.
   It stays the canonical Trace export. FTO-24 adds a converter from
   `ForceTorqueSeries`.
2. **Extend the API `ForceVector3D`.** Rejected as the core type because it is
   pydantic, transport-only, drops the torque/force pairing, and has an
   un-validated `color`. It becomes a serializer view of `OverlayWrench`.
3. **Per-engine native visualizers** (MuJoCo `mjVIS_CONTACTFORCE`, Drake
   `ContactVisualizer`). Rejected because they cover contact only, are
   inconsistent across engines, and cannot be drawn on video. They may stay as
   optional debug toggles.

## Consequences

- **Positive:**
  - One physics contract, one geometry builder, one palette, and thin
    renderer adapters.
  - Cheap agents can implement engines and renderers in parallel against fixed
    fixtures.
  - Tension and compression reach every engine through one converter.
  - Video overlays reuse the existing camera and time-map authorities.
- **Negative:**
  - Existing engine GUI drawing code must be removed and rewired: MuJoCo
    `sim_rendering_mixin`, `meshcat_adapter`, the two Pinocchio mixins, and
    `drake_gui_viz`.
  - The unused `plotting/renderers/force_vectors.py` and `vectors.py` must be
    reduced to thin wrappers or removed with their tests.
- **Follow-ups:**
  - MyoSuite muscle provider, JaxSim, and a Unreal adapter.
  - #10286 CF-7 consumes this layer for ZTCF/ZVCF force/couple arrows instead
    of building its own.

## Validation

- Contract tests: validation, immutability, wire round trip and shared JSON
  fixtures (Python and TypeScript).
- Pure-geometry tests for `build_glyphs`: length clamping, the arc
  right-hand rule, and deterministic output.
- Golden-image tests per renderer (matplotlib, OpenCV, MuJoCo offscreen).
- A cross-engine physical parity suite. For a static hanging pendulum and a
  standing two-foot contact fixture:
  - the joint reaction equals the supported weight;
  - the summed ground reaction force equals body weight within a stated
    tolerance;
  - the tension/compression sign agrees across engines.
- No engine or renderer may be marked complete in `src/config/feature_parity.json`
  without its parity row.
