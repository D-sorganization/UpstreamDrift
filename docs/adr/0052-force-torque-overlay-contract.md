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
  - `point_m: tuple[float, float, float]`: the physical application point
    (joint anchor, contact point, muscle insertion) in the world frame.
  - `force_n: tuple[float, float, float] | None` and
    `torque_nm: tuple[float, float, float] | None`: each half is **optional**,
    and at least one must be present. An unavailable half is stored as `None`
    and serialized as `null`. It is never fabricated, never drawn and never
    reported as zero.
  - `source: str`: the engine and method, e.g. `"mujoco:cfrc_int"`.
  - The frame is always `"world"` and the direction convention is always
    `"applied_to_body"`. Neither is a field, so neither can be set wrongly.
  - Why not embed `SpatialWrench`: it requires both halves to be finite
    3-vectors, so a torque-only or force-only channel would have to invent the
    missing half. `OverlayWrench` reuses `SpatialWrench`'s vector validation
    (`_validate_vec3`) and transforms (see conversions). `to_spatial_wrench()`
    returns a `SpatialWrench` only when both halves are present, and raises
    `ValueError` otherwise.
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
rotation_world_from_local, point_world, source)` wraps `transform_wrench`
    when both halves are present. With one half `None`, it rotates the present
    half only. Moving a wrench to a **different** point changes the torque by
    `r × F`, so `move_wrench_point(...)` returns `torque_nm=None` when the force
    half is unknown, rather than guessing.
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
- `GlyphSet` is serializable: `to_dict()` / `from_dict()` against
  `schemas/glyph-set-v1.json`. **This serialized `GlyphSet` is the only
  rendering wire format.** Every remote or non-Python client (web three.js,
  web SVG-on-video) renders it as received. No client reimplements scaling,
  clamping, availability or arc geometry, so the web cannot drift from the
  native and video renderers. The physics wire format
  (`force-torque-frame-v1`) stays available for analysis and export, not for
  drawing.
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

### Grip Label Table (GCV-7, #11713)

`grip_wrench.to_overlay_wrenches` emits these `GRIP` labels. Every wrench is the
loading **exerted by the hand on the club**, in the world frame. Unavailable
quantities are omitted (never zero). `grip_wrench.GripAnalysis` carries
`split_method`, because the left/right split of two rigid welds is set by the
solver.

| Label                 | Application point | Force half            | Torque half                          |
| :-------------------- | :---------------- | :-------------------- | :----------------------------------- |
| `grip:hand_left`      | left grip point   | `F_L`                 | `tau_L` (omitted if not supplied)    |
| `grip:hand_right`     | right grip point  | `F_R`                 | `tau_R` (omitted if not supplied)    |
| `grip:net_midpoint`   | grip midpoint     | `R = F_L + F_R`       | none                                 |
| `grip:couple_midpoint`| grip midpoint     | none                  | `M_M` = contact moment + free torque |
| `grip:mof_left`       | grip midpoint     | none                  | `(r_L - r_M) x F_L`                  |
| `grip:mof_right`      | grip midpoint     | none                  | `(r_R - r_M) x F_R`                  |

The Simscape channel labels `grip:total_hand`, `grip:lh_mof`, `grip:rh_mof` and
`grip:midpoint_couple` in `src/engines/simscape/force_channels.py` remain the
logged-signal forms; their reference points must be confirmed from the model
before asserting equality with the shared definitions (see GCV-9, #11715).

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
  serialized `glyph-set-v1` payload produced server-side by `build_glyphs`. It
  turns each `ArrowGlyph` into a cylinder and cone, and each
  `TorqueArcGlyph.polyline_m` into a tube and cone. It has no client-side glyph
  algorithm; scale changes are a new request to the server.

### 4. Engine Adapters

Each engine adapter implements `ForceTorqueProvider`. Where it can, it also
implements `AxialLoadProvider` through `axial_loads_from_reactions`. It sets
`EngineCapabilities.force_visualization` honestly. Engine SDK imports stay in
the engine adapter packages (`src/engines/...`), never in
`src/shared/python/force_overlay/`. New MuJoCo-specific code goes beside the
engine, not into `body_part_viz`.

### Contact Label Table (GCV-1, #11707)

Ground-reaction overlays come from one pure-numpy core,
`src/shared/python/biomechanics/ground_reaction.py`
(`to_overlay_wrenches`). All are `WrenchKind.CONTACT`, world frame, force
exerted **by the ground on the foot**. `<foot>` is the sanitised foot label.

| Label                        | Halves | Application point | Meaning                                                  |
| :--------------------------- | :----- | :---------------- | :------------------------------------------------------- |
| `contact:grf_<foot>`         | force  | foot CoP          | Resultant foot GRF (contact centroid when CoP is absent) |
| `contact:grf_net`            | force  | net CoP           | Both-feet resultant GRF (`body="system"`)                |
| `contact:free_moment_<foot>` | torque | foot CoP          | Vertical free moment `T_z` of the foot about its CoP     |
| `contact:free_moment_net`    | torque | net CoP           | Free moment of the net wrench about the net CoP          |
| `contact:moment_com_<foot>`  | torque | whole-body CoM    | `M_O,f - c x F_f`, the foot wrench moment about the CoM  |
| `contact:moment_com_net`     | torque | whole-body CoM    | Sum of the foot CoM moments (exactly)                    |

The CoP exists only when `F_z >= 10 N` (`COP_MIN_FZ_N`); below that the free
moment labels are omitted (unavailable, never zero) and the GRF arrow is
anchored at the loaded-contact centroid. The net free moment is computed from
the net wrench and is not the sum of the foot free moments.

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

## Addendum: Calibrated MuJoCo Mesh Render on Source Footage (FTO-27, #11312)

- **Date:** 2026-10-03
- **Context:** Rendering native MuJoCo model meshes and 3D force glyphs onto source footage requires aligning the MuJoCo offscreen camera with the calibrated pinhole camera's intrinsics ($K$), extrinsics ($R, t$), and lens distortion.

### Decision 2: Principal Point and Aspect Ratio Handling

MuJoCo cameras and OpenGL perspective frustums assume centered principal points and square pixels. To support calibrated cameras with off-centre principal points $(c_x, c_y) \neq (W/2, H/2)$:

- **Chosen Approach:** (a) Render an enlarged frame and crop.
- **Mechanism:**
  - For image dimensions $(W, H)$ and principal point $(c_x, c_y)$, render an enlarged offscreen buffer of size $W_{\text{render}} = 2 \cdot \max(c_x, W - c_x)$ and $H_{\text{render}} = 2 \cdot \max(c_y, H - c_y)$.
  - Calculate vertical field of view for the enlarged buffer: $\text{fovy} = 2 \cdot \text{atan}(H_{\text{render}} / (2 \cdot f_y))$.
  - The crop box $(x_0, y_0, x_0 + W, y_0 + H)$ with $x_0 = \text{round}(W_{\text{render}} / 2 - c_x)$ and $y_0 = \text{round}(H_{\text{render}} / 2 - c_y)$ precisely places the optical center at $(c_x, c_y)$ in the cropped frame.
  - Works with any MuJoCo model without modifying model XML or requiring pre-allocated camera tags.
  - Proved in `tests/unit/engines/mujoco/test_footage_composite.py` with 3D model body points reprojecting within $\le 1.0\text{ px}$ of `PinholeCamera.project`.

### Decision 3: Lens Distortion Policy

Calibrated cameras exhibit radial/tangential distortion ($k_1, k_2, \dots$), while MuJoCo offscreen rendering is rectilinear.

- **Chosen Policy:** Undistort the source footage frame before compositing (`cv2.undistort` with the camera's coefficients).
- **Rationale:**
  - Avoids non-linear resampling blur and artifacting on rendered 3D meshes and alpha edges.
  - The arrow layer (FTO-8 `opencv_glyphs.py`) and MuJoCo scene glyphs (`add_glyphs_to_scene`) share the identical rectilinear pinhole geometry.
  - Verified by unit tests demonstrating that 3D arrows and mesh attachment points agree within $\le 1.5\text{ px}$.

## Addendum: Arrow Scale Modes, Clamping and Group Toggles (GCV-4, #11710)

The fixed 1 mm per N scale clamped at 0.6 m drew every ground reaction force
above about 600 N at the same length, so peak loads looked small and identical.

- **Scale modes.** `ForceGlyphStyle.scale_mode` is `fixed` (default,
  `force_scale_m_per_n`), `body_weight` or `peak`. The last two map
  `reference_force_n` to `reference_length_m` (default 0.5 m): the caller passes
  body weight $m g$ for `body_weight` and the series peak for `peak`. Both
  require `reference_force_n`; an unknown mode raises `ValueError`.
  The effective scale is $L_\mathrm{ref} / F_\mathrm{ref}$ metres per newton.
  `kind_scale` multiplies that scale for one `WrenchKind`.
- **Clamping is surfaced.** `ArrowGlyph.clamped` is true iff the raw length
  exceeds `max_length_m`. Raising a tiny arrow to `min_length_m` is a floor, not
  a clamp. `LegendSpec` gains `scale_mode` and `clamped_labels` (optional keys, so
  older payloads still load). Renderers draw a clamped arrow with a distinct tip:
  a double chevron in OpenCV and Three.js, a white marker arrow past the tip in
  the MuJoCo scene; the legend reports the count.
- **Group toggles.** `ForceGlyphStyle.groups` selects overlay groups by label
  prefix only, so providers never import this module: `per_foot`
  (`contact:grf_<foot>`), `net` (`contact:grf_net`), `free_moment`
  (`contact:free_moment_*`), `moment_about_com` (`contact:moment_com_*`),
  `contact_points` (any other `contact:` label), and the grip groups
  `grip_per_hand` (`grip:hand_*`), `grip_net` (`grip:net_midpoint`),
  `grip_couple` (`grip:couple_midpoint`) and `grip_mof` (`grip:mof_*`).
  Ungrouped labels are never filtered. `contact_points` and `moment_about_com`
  are off by default; raw contacts are still shown when the frame carries no
  aggregated GRF label, so engines that predate the aggregated labels keep their
  arrows.
- **Surfaces.** The PyQt Visualization tab, the web `ForceOverlayPanel`, the
  `/simulation/forces` API and the WebSocket style object expose the same
  controls. Native export defaults to `body_weight` with the model mass, so one
  body weight is 0.5 m and a 3 m ceiling avoids silent clamping in a swing.
- The default shaft radius is now 12 mm, and the OpenCV shaft is 4 px at 1080p.

