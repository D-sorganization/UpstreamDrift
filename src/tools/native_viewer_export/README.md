# Native Viewer Export

Renders a same-input swing in each physics engine's own viewer and writes mp4
clips: one per camera view (face-on, down-the-line, overhead, oblique) plus a
labelled 2x2 clip, with force and torque overlays.

| Engine    | Native viewer      | Capture                            | Overlay                             |
| --------- | ------------------ | ---------------------------------- | ----------------------------------- |
| Drake     | MeshCat            | Headless Chromium (SwiftShader)    | 3D glyphs via `DrakeMeshcatSink`    |
| Pinocchio | MeshcatVisualizer  | Headless Chromium (SwiftShader)    | 3D glyphs via `MeshcatPythonSink`   |
| OpenSim   | simbody-visualizer | `xvfb-run` private display + `xwd` | 2D projected glyphs                 |
| MyoSuite  | MJRenderer (EGL)   | `mujoco.Renderer`                  | 3D glyphs via `add_glyphs_to_scene` |
| MuJoCo    | `mujoco.Renderer`  | Offscreen (`MUJOCO_GL` egl/osmesa) | 3D glyphs via `add_glyphs_to_scene` |

The MuJoCo backend (`backends/mujoco_native.py`, worker `mujoco_worker.py`)
exports the full-body MJCF with the default appearance document
(`export_full_body_mjcf(..., appearance=...)`): smooth body segments, garments,
head, club shaft/grip/head meshes, ground, sky and lights, so the clip shows
the body and the club rather than a stick figure. The pose is the bundle's `q`
mapped through its coordinate order; cameras, look-at tracking and the HUD are
the shared view presets and compositor. Set `MUJOCO_GL=osmesa` where there is
no GPU (containers); the default is `egl`.

The OpenSim visualizer cannot draw dynamic 3D decorations, so its overlay is
projected with the same pinhole camera and drawn on the captured frame.

## Usage

```bash
python3 -m src.tools.native_viewer_export \
  --bundle driver.npz --swing driver --club Driver --out out/ \
  --receipt-drake driver_closed_drake.json \
  --receipt-pinocchio driver_closed_pinocchio.json
```

- `--engines` selects a subset; unavailable backends are skipped with a reason.
- `--receipt-<engine>` plays that engine's closed-loop rollout (the receipt's
  specification hash must match the bundle); without it the bundle reference
  trajectory is shown.
- `--no-overlay` disables every overlay; `--force-vectors/--no-force-vectors`
  and `--torque-vectors/--no-torque-vectors` switch the straight force arrows and
  the curling torque arcs independently (both on by default). See
  [Segment Shading](#segment-shading).
- `--no-grid`, `--views`, `--fps` (default 60), `--size` and
  `--preset hq|preview` control the output. `hq` is 1280x720, libx264,
  `yuv420p`, CRF 18; `preview` is 640x544.
- `--grip` adds the grip overlay (per-hand force, net force and couple at the
  grip midpoint, labelled with the split method; unavailable is shown, never
  zero) and writes `<swing>_<engine>_grip_wrench.json`. Drake uses its own KKT
  multiplier; the other viewers show the MuJoCo plant at that engine's pose.
  `--views hands_closeup --no-grid` follows the kinematic grip midpoint per frame (club pose, so `--no-overlay` and no `--grip` track the hands too).
  `--no-hud` writes clean frames. `--speeds= --impact-window 0.6
--impact-speed 0.25` writes only the 0.25x impact clip. Impact time is
  `--impact-time`, else the bundle provenance `impact_time_s`, else
  `model_appearance.club_face.ball_passage` on the `Clubhead` frame (the
  sub-sample instant of closest approach to address, with the height and
  ball-radius checks), else the last sample.
- The decorative golf ball (GCV-13, #11719; `--no-ball` disables it) is drawn at
  address and hidden after impact, visual only (never in a dynamics model). One
  placement per swing (`ball.py`, `BallPlacement`) is shared by every engine:
  the bundle's recorded `ball_impact().ball_centre_m` (source
  `bundle:ball_impact`, the ball every replay strikes), else the address
  geometry (MuJoCo FK of the first state, face centre and normal through
  `model_appearance.ball.ball_position_at_address`, source `address_geometry`).
  With the `strike` placement of `model_appearance/ball_strike_settings.json`
  (#15150, the default) that ball is then moved to where the face centre passes
  nearest it around impact (source `strike_geometry`, record key `strike`). It
  is a display placement: no joint angle or club dimension changes, and the
  face is not squared. A swing that misses by more than `max_shift_m` keeps the
  address ball and records why.
  A ball that rests more than `min_gap_m` above the ground (a driver ball) gets a
  tee (#15150): a peg from the ground to a cup under the ball, drawn from
  `model_appearance/ball_tee.py` and `ball_tee_settings.json` (`enabled`,
  `min_gap_m`, `shaft_radius_m`, `cup_radius_m`, `cup_height_m`, `rgba`). The
  tee is render geometry only (no collision, no mass), is not hidden at impact
  and stays standing when the ball leaves; a ball on the turf gets none. MuJoCo
  and MyoSuite draw it in the render scene; Drake, Pinocchio and OpenSim draw
  no tee yet.
  When neither exists (no club, ungrounded first state, no MuJoCo) no ball is
  drawn and the reason is logged and returned in `ExportResult.ball`
  (`{"unavailable": reason}`, printed by the CLI); a placement is recorded with
  its position, source and `hide_after_s`. MuJoCo and MyoSuite append a render
  scene sphere per visible frame; Drake and Pinocchio MeshCat hold one `/ball`
  sphere and toggle `visible`. OpenSim's simbody visualizer caches fixed
  geometry at `initSystem` (toggling Appearance visibility does not redraw), so
  its worker hangs the sphere on a tiny render-only body on a slider joint from
  ground and slides it 1 km out of view after impact (`ball_slide_m`); body
  poses are redrawn from the state every frame. Post-impact flight is a later
  slice.
- Force arrows use the `body_weight` scale mode when the model mass is known:
  one body weight is 0.5 m and the ceiling is 3 m (`default_glyph_style`).
- `--speeds 1,0.5,0.25` (default: full, half, quarter) writes one clip set per playback speed, named
  `_1x`, `_0p5x`, `_0p25x`. Frames are chosen by the time-based
  `video_timing.FrameSchedule` and interpolated when the source step is coarse.
- `--impact-window 0.1` (with optional `--impact-time`) adds a `_impact_0p1x`
  clip of that many swing seconds centred on impact.
- `--stride` is deprecated (alias for one fixed index step; warns).
- The HUD shows playback speed and milliseconds from impact.
- A HUD line wider than the frame wraps onto further rows at its ` | `
  separators (`compositor.wrap_hud_line`), so the per-point ellipsoid lines are
  never cut at the frame edge.
- Mobility and capability ellipsoids are evaluated at the pose of the frame they
  are drawn on (`OverlayFeed.following`), not at a source state: a clip frame is
  a time sample, so frame `k` is not source state `k`. Their display scale is
  fixed once from the source states and shared by every clip. A feed whose
  ellipsoids cannot be evaluated at the clip frames is refused (`ValueError`).
- A bundle whose every effort is zero is a kinematic replay
  (`kinematic_replay.py`): its contact wrenches are not applied forces, so force
  and torque vectors are hidden unless `--force-vectors` or `--torque-vectors` is
  passed, with one warning and the HUD note "Kinematic Replay: Forces Not Shown".
  The efforts decide, not a provenance label (the writers share none).
- The ellipsoid values are a compact side panel (`ellipsoid_panel.py`, drawn by
  the impact factors panel's renderer): a row per point with its largest
  mobility and effective-mass radii and condition number, a units row and, with
  `--ellipsoid-color-by kind`, a legend. It sits opposite the impact factors
  panel, stacks below it on a shared edge and stays above the bottom HUD rows;
  layout `text` in its settings file restores them.
- Ellipsoid visibility (`ellipsoid_visibility.py`, `ellipsoid_visibility_settings.json`,
  `--ellipsoid-visibility-settings JSON`): `scale_grouping` `per_kind` (one display
  scale per kind, the clubhead sets it; sizes comparable between points) or
  `per_point` (each point's largest radius maps to `target_radius_m`, the former
  literal 0.09 m; sizes are **not comparable between points**, so the panel adds a
  "Scaled Per Point" footer). `outer_alpha` is the alpha of the larger ellipsoid
  when two kinds share a centre, so the inner one shows (1 = off).
  `min_semi_axis_m` floors a displayed semi-axis (0 = off); a floored axis is **not
  to scale**, the glyph is a visibility aid along it. The packaged file ships
  `per_point`, 0.09 m, alpha 0.35 and a 0.03 m floor; the code defaults of
  `EllipsoidDisplayScale` are unchanged.

Environment: `NATIVE_VIEWER_CHROMIUM` (Chromium executable; a full Playwright
build renders much faster than the headless shell), and
`NATIVE_VIEWER_MYOSUITE_PYTHON` (interpreter that has `myosuite`).

The OpenSim backend only ever runs under `xvfb-run` and refuses to start when
`NATIVE_VIEWER_XVFB` is missing or `DISPLAY` is a real display.

## Contact Frame and Sun Light

- **Contact frame (#15150).** With a known impact time and a face that reaches
  the ball (`impact_state.series_at_contact`) the exporter renders a frame at
  exactly the contact instant (`FrameSchedule(extra_times_s=...)`); the HUD's
  "ms from impact", the impact panel and the tracer count from it, and the ball
  stays on its tee through that frame. `--contact-hold-frames N` repeats it N
  more frames (default: 24 in an impact-window clip, 0 in full-speed clips).
- **Sun light (MuJoCo).** `backends/sun_light_settings.json` adds a directional
  light (direction, colour, intensity, shadow; `enabled: false` removes it) so the
  400 m ground is lit to the horizon. Default look change: the near turf is
  brighter (about 1.9x mean level) and the golfer's shadow fainter.
- **Showpiece lighting (MuJoCo, opt-in).** `--lighting showpiece` (or
  `--lighting-settings PATH`) replaces the scene lights and the sun with the
  directional lights, headlight, shadow map and ground tone of
  `backends/showpiece_lighting_settings.json`. Display only; default `studio`.

## Appearance Flags

The golfer's look can be chosen per export (HRG-B2, #13167); without any of
these flags every engine renders the contract default golfer
(`src/config/appearance_defaults.json`).

```bash
python3 -m src.tools.native_viewer_export --bundle driver.npz --swing driver \
  --out out/ --engines mujoco \
  --appearance driver.appearance.json \
  --skin-tone skin_5 --hair '#2a5db0' \
  --clothing golf_polo_shorts --top-color '#c0392b' --bottom-color '#f4d03f'
```

- `--appearance PATH` is an `appearance-v1` document or a `<spec>.appearance.json`
  sidecar. `--skin-tone` and `--hair` take a palette id (`skin_1`..`skin_8`,
  `hair_brown`, ...) or a `#rrggbb` colour; `--clothing` takes a preset id;
  `--top-color` and `--bottom-color` take `#rrggbb` and recolour the preset's
  shirt and trousers/shorts. `--glove {none,lead,trail,both}` (HRG-G1, #13186)
  puts the white golf glove on the named grip hand by role; the lead hand is the
  left one for a right-handed golfer (document field `handedness`, default
  `right`). The default `none` draws bare hands in the skin colour. Custom colours become document `materials` entries
  (see `docs/help/character_builder.md`).
- **Precedence:** an explicit flag beats the `--appearance` document, which beats
  the contract defaults (`model_appearance.appearance_selection.resolve_appearance`).
  GUI settings (QSettings) are never read.
- The resolved document travels in `ExportSettings.appearance` and
  `WorkerJob.appearance` and is applied by the MuJoCo, MyoSuite, OpenSim, Drake
  and Pinocchio workers. Unknown ids, a malformed colour, a garment colour on
  the `none` preset, or a hair colour with a cap are errors (exit 2).

## Segment Shading

`--segment-shading` (off by default; engine support below) paints each segment
that has a defined axis translucently by its axial load: tension positive,
compression negative, from inverse dynamics (`mj_inverse` with the provider's
own `q`, the bundle reference velocity and accelerations by an unfiltered
central finite difference of that velocity, then the `cfrc_int` reaction at the
proximal section projected on the segment axis). The torque arcs and the shading
come from one evaluation per frame (`ForceTorqueFrame.axial_loads`).

A skinned body (`subject.body_visual_backend: skinned_cc0`) is drawn from
MuJoCo skins, which are not scene geoms. With shading on, the MuJoCo worker
cuts each skin into one piece per segment and colours the pieces per frame
(`backends/_skin_shading.py`, #15150); the pelvis pieces take the lumbar load
and hair, eyes, brows and lips keep their colour (`skin_load_settings.json`).
With shading off, or zero opacity, the skins are not cut.

| Option                                       | Meaning                                                                                                        |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `--shading-opacity 0..1`                     | strength (default 0.6; 0 is identical to off)                                                                  |
| `--shading-mode tint\|replace`               | `tint` keeps unloaded segments' own colour (default); `replace` blends every loaded segment to the load colour |
| `--tension-color`, `--compression-color`     | `#RRGGBB` (defaults blue, red)                                                                                 |
| `--shading-limit-n N`                        | load that saturates the colour, both signs (default 1000)                                                      |
| `--tension-limit-n`, `--compression-limit-n` | separate saturation loads                                                                                      |

One `ForceColorScale` validates all of these (the CLI exits with status 2 and an
`error:` line on a bad value). The HUD gains a legend line with both colours, the
saturation load, opacity and mode (`--no-hud` hides it). Segments without a
defined axis are never shaded (the club shaft shades with its own load).

Every engine is shaded from that one MuJoCo evaluation. The axial loads are
keyed by the specification body name, which is also the body name of every
engine's model and of the exporter's visual items, so no per-engine name
translation exists: a segment without a load, or without geometry in a viewer,
simply stays unshaded. Each run logs one debug line per engine with the number
of shaded over total segments (`shading.log_coverage`).

| Engine    | Segment shading | How                                                                                                                                                                                                                        |
| --------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MuJoCo    | yes             | `apply_mujoco_scene_colors` on the render scene (`backends/_scene_overlays.py`)                                                                                                                                            |
| MyoSuite  | yes             | the same helper as MuJoCo (same `mujoco.Renderer` scene)                                                                                                                                                                   |
| Drake     | yes             | MeshCat `color` property per illustration geometry (`drake_color_bindings`), base colour from the geometry, alpha kept                                                                                                     |
| Pinocchio | yes             | MeshCat `color` property per visual (`pinocchio_color_bindings`), base colour from the visual item, alpha kept                                                                                                             |
| OpenSim   | no              | the simbody visualizer caches geometry and its colour at `initSystem` and exposes no per-frame recolouring from Python (the ball needs a sliding body for the same reason); the export logs a warning and renders unshaded |

The supported set is `backends.registry.SHADING_ENGINES`; the runner warns only
for engines outside it. All blending is `ForceColorScale.shade_rgb`. Example:

```bash
python3 -m src.tools.native_viewer_export --bundle driver.npz --swing driver \
  --engines mujoco --views face_on --speeds 1 --no-grid --segment-shading \
  --shading-opacity 0.5 --shading-limit-n 800 --no-force-vectors --out out/
```

## Palm-Thumb Force

`--palm-thumb-force` (off by default) draws the allocated trail-palm-on-lead-thumb
normal force at the contact patch as a sky-blue arrow, independent of
`--force-vectors` and `--torque-vectors`. It is an equivalent of the grip closure
force (`method="allocation"`), not a measured contact; unloaded frames draw
nothing and the legend says "allocated (not measured)". MuJoCo overlay source only;
the compiled model is unchanged (HRG-H1, #13189).

## Appearance Flags

The golfer's look can be chosen per export (HRG-B2, #13167); without any of
these flags every engine renders the contract default golfer
(`src/config/appearance_defaults.json`).

```bash
python3 -m src.tools.native_viewer_export --bundle driver.npz --swing driver \
  --out out/ --engines mujoco \
  --appearance driver.appearance.json \
  --skin-tone skin_5 --hair '#2a5db0' \
  --clothing golf_polo_shorts --top-color '#c0392b' --bottom-color '#f4d03f'
```

- `--appearance PATH` is an `appearance-v1` document or a `<spec>.appearance.json`
  sidecar. `--skin-tone` and `--hair` take a palette id (`skin_1`..`skin_8`,
  `hair_brown`, ...) or a `#rrggbb` colour; `--clothing` takes a preset id;
  `--top-color` and `--bottom-color` take `#rrggbb` and recolour the preset's
  shirt and trousers/shorts. `--glove {none,lead,trail,both}` (HRG-G1, #13186)
  puts the white golf glove on the named grip hand by role; the lead hand is the
  left one for a right-handed golfer (document field `handedness`, default
  `right`). The default `none` draws bare hands in the skin colour. Custom colours become document `materials` entries
  (see `docs/help/character_builder.md`).
- **Precedence:** an explicit flag beats the `--appearance` document, which beats
  the contract defaults (`model_appearance.appearance_selection.resolve_appearance`).
  GUI settings (QSettings) are never read.
- The resolved document travels in `ExportSettings.appearance` and
  `WorkerJob.appearance` and is applied by the MuJoCo, MyoSuite, OpenSim, Drake
  and Pinocchio workers. Unknown ids, a malformed colour, a garment colour on
  the `none` preset, or a hair colour with a cap are errors (exit 2).

## Layout

- `core.py`: settings, clip plans (speeds, impact window), the backend protocol,
  `export_swing`; frame timing lives in `src/shared/python/video_timing`.
- `compositor.py`: labelled 2x2 grid, HUD text and the impact card panel. Text
  lines are spaced by their outlined height and the panels start below the view
  label, so neither can overlap (`hud_layout_settings.json`).
- `impact_card.py`: one impact card per swing from the displayed pose.
- `flight_tracer.py`: the ball flight of a swing in swing time and world
  coordinates (#15150), launched by the repository impact and ball-flight
  model from the clubhead state the impact card reads: the state where the
  face reaches the ball (`impact_state.py`), and the tracer starts at that
  instant, not at the sample before it. It is a model
  prediction from kinematic impact conditions, not a measurement; a flight
  the model flags as outside its validity envelope is not shown unless
  `draw_invalid_flight` is set. Options: `flight_tracer_settings.json`.
- `impact_factors.py`: the optional impact factors panel (#15150,
  `ExportSettings.impact_factors`, default off): the card's rows in a panel of
  one fixed size on every frame, blank before impact and frozen at their impact
  values from the impact frame to the end of the clip; it replaces the card.
  Options and their meaning are in `impact_factors_settings.json`.
  `--impact-factors` turns it on and `--impact-factors-settings PATH` reads
  another settings file (`pre_impact: live` shows the extractor's value at the
  frame on screen until impact). A face that passes the ball without touching
  it is labelled "closest approach, no contact" with the miss distance; with
  `closest_approach_fallback` a swing with no detected impact uses its closest
  approach as the reference instant and says so.
  `--ball-flight` (or `--ball-flight-settings PATH`; default off) flies the
  ball from the impact frame in playback time with a persistent shot tracer,
  adds the flight numbers to the HUD and holds the clip on the last pose until
  the ball has landed. Ball and tracer are a 2D overlay projected through the
  view preset camera (MuJoCo's 45 degree lens; views that follow the grip are
  left alone). `--views down_the_line_flight --no-grid` is the wide
  down-the-line view (`golf_view_presets/flight_view.json`, 4.6 m from the
  golfer) that keeps the flight in the picture. The MuJoCo ground patch is
  `ground_half_extent_m` wide each way (`backends/scene_extent_settings.json`),
  so the ball lands on visible turf.
  The MuJoCo worker also renders scene depth from the impact on, so the golfer
  hides the tracer and ball where nearer to the camera (`occlude_by_scene`, on
  by default; without depth a warning is logged). `landing_marker` draws a
  ground ring at the landing point; its optional aids are a `ground_track`
  (the flight path on the ground, revealed with the tracer), a `drop_line` from
  the ball to the ground while in flight and a `carry_label` (the flight
  model's carry, unit and format from the settings) once landed.
  `ExportResult.ball` reports the contact
  instant as `hide_after_s`.
- `flight_inset.py`: opt-in 2D flight inset (#15150), a broadcast-style panel
  that makes a long carry readable from behind the golfer: a side profile
  (height against down-range distance, the vertical scale exaggerated by
  `vertical_exaggeration`, 1 for true aspect) with the apex and carry labelled,
  and a plan view (lateral offset against down-range distance, target line,
  landing offset such as "15 m R"). Both are revealed with the tracer, with a
  dot at the ball; once landed the full curve and labels stay. The numbers
  are the flight model's. It starts below the view label, stacks below the impact
  factors panel (or ellipsoid panel) on its edge, stays above the HUD rows and
  shrinks, or is not drawn, when no room is left. Off unless
  `flight_inset_settings.json` sets `enabled` (position, size, margins, colours,
  line widths, label formats, units and font scale are all settings there).
- `overlay.py`: overlay feed built from `force_overlay.bundle_provider`.
- `overlay2d.py`: pinhole projection for the OpenSim 2D glyphs.
- `backends/`: one module per engine plus shared helpers.

Camera presets live in `src/shared/python/golf_view_presets`; the glyph
pipeline is `src/shared/python/force_overlay` (ADR-0052). The OpenSim and
MeshCat viewers use the shared vertical field of view `VIEWER_FOV_Y_RAD`
(0.7 rad). Before NV-9 (#11697), MeshCat kept the 75 deg three.js default,
which left the golfer at about a quarter of the frame height.
`golf_view_presets.framing` measures the projected extent of a set of points
(`projected_extent`). It also gives the camera distance that frames them with
a 15 % margin (`fit_distance_m`).

## Owner Showcase Driver

`python3 -m scripts.render_forces_showcase --capture capture-A --swing driver=driver.npz`
renders the `clean/`, `club_forces/` and `grf/` clip set (1280x720, 60 fps,
CRF 18) into `~/Videos/Parity Audit/forces_and_impact/` through this tool and
writes `manifest_<capture>.json`. A missing bundle or unavailable engine stops
the run with a message (`--skip-unavailable` records it instead). The `grf/` and
`club_forces/` clips carry the impact card and an inset plot strip (see below);
`clean/` carries neither.

## Inset Plot Strips

`ExportSettings.inset_strip` (default off; the showcase enables it for `grf/` and
`club_forces/`) draws a translucent plot strip along the bottom of every frame
(above the HUD), with a vertical time cursor at the frame's swing time
(`inset_strip.py`, GCV-19, #11725). The strip kind follows `ExportSettings.grip`:

- GRF strip (`grip=False`): left panel, vertical ground-reaction force of the
  left foot, right foot and their net in body weights (BW; `Fz / reference_force_n`
  of the glyph style, i.e. the golfer's `mass * g`); right panel, the net centre of
  pressure world X and Y in metres (taken from the `contact:free_moment_net`
  wrench, which sits at the net CoP).
- Grip strip (`grip=True`): net grip force magnitude |F| in N and couple
  magnitude |M| in N m from `OverlayFeed.grip_analyses`.

Both show a dashed vertical line at the impact time and a legend. The curves are
read once per swing from the SOURCE samples (at most 600, stride-thinned), not
from the resampled clip frames, and rendered once per frame size with matplotlib
Agg; per frame only the cursor changes, so every speed clip and the impact window
share one strip. Unavailable samples (a foot without loaded contact, a CoP below
the force threshold, a failed grip solve) are gaps in the line, never zero. When
nothing can be plotted (no overlay feed, no body-weight scale, no loaded contact,
grip overlay off, no grip wrench) no strip is drawn and `ExportResult.inset_strip`
(and the showcase manifest) holds the reason; otherwise it holds the panels and
per-curve sample counts. The values are model outputs of the shared contact law and
grip extraction, not measurements.

## Impact Parameter Card

`ExportSettings.impact_card` (default on, needs the HUD) draws the GCV-17
`ImpactCard` as a semi-transparent right-hand panel from the impact frame to the
end of each clip (`compositor.draw_impact_card`). `impact_card.impact_card_for`
computes it once per swing, on the unresampled source states (identical for
every speed clip), from the displayed pose `q`: `club_pose.ClubPoseModel` (shared
with the decorative ball) places the spec club body through MuJoCo and gives the
true face centre, normal, toe and grip axes; `np.gradient` of the face-centre
positions gives the velocity; `extract_impact_parameters` gives the rows. With a
ball the impact state is the club where its face reaches the ball
(`impact_state.series_at_contact`, settings in `impact_state_settings.json`): a
fit of the samples before contact carried to the contact instant, so no row is
read a sample early or across the strike (`extras.impact_state`). A bundle's
recorded impact sample already carries the strike, so the fit ends before it
(`ModelClubSeries.strike_time_s`): a ball moved forward is met by the swing as
it was before the strike, not by the struck head; without a ball
it is the last sample before the impact time. The specification has no target
line, so the target direction is the horizontal part of the address (state 0)
face normal of the address-square head (as `plan_gaze` does; it does not turn
with the head roll that is drawn), right-handed, Z-up;
the card records `extras.target_source = "address_face_normal"`, so face angle
and path are relative to that assumed line. Unavailable rows show "unavailable" and a
reason, never zero. Impact Location (toe, high) uses the decorative ball
(`SwingInput.ball`, passed explicitly by `export_swing`): the contact point is the
ball surface point facing the club, `ball_centre - radius * n_face` at the impact
sample, and toe/high are its offsets from the face centre along the face toe axis
(positive toward the toe) and `toe x normal` (positive toward the crown). The toe axis is the head's own heel-to-toe line (head frame +z, rolled with the drawn head) projected into the face plane, never derived from the shaft, which leans about 34 deg from it on the driver; the card records
`extras.ball_source` (the placement source, e.g. `address_geometry`). With the
address-geometry ball this is the offset from the address face line, not a
measured contact. A contact more than 50 mm (`MAX_FACE_OFFSET_M`) from the face
centre lies off every club face. That means the model face passed the ball
without striking it, so the row is "unavailable" and its reason carries the
measured miss. Without a ball the row stays unavailable ("needs the ball
contact point (no ball placement supplied)"). When no card can be computed (no MuJoCo, no club, speed
below 1 m/s) nothing is drawn, `ExportResult.impact_card` holds the reason, and
the showcase manifest records it. The panel shows model estimates, not a
measurement.

## Shot Setup

Two opt-in display settings, both off by default (zero changes nothing). The
body motion is replayed unchanged: no joint angle changes, and this is a
kinematic replay, not a refit. Limits live in `shot_setup_settings.json`.

- `--club-roll-in-hands DEG` turns the club rigidly about its shaft relative
  to the hands (`|DEG| <= 20`). The shaft bend is ignored. Positive opens the
  face: on the matched right-handed driver swing +1 degree of roll turns the
  face about 0.73 degrees to the right and adds about 0.67 degrees of loft.
  The rolled spec reaches the drawn head, impact card, flight and ball.
- `--ball-position-shift M` moves the computed address ball along the
  horizontal target line (`|M| <= 0.15`, positive toward the target) before
  the strike placement, so the ball is still struck at the face centre. A
  ball already on the swing or recorded by the bundle is not moved. Forward of
  the captured ball the measured club was already slowed by the real ball, so
  forward positions are valid only for the first few centimetres; the ball
  record carries `ball_position_shift_m` and a `position_note`.
- `--shot-shape {straight,draw,fade}` solves the club roll instead of taking a
  number. `straight` is the roll at which the flight model lands the ball on
  the target line (|side| <= `straight_tolerance_m`, found by a bracketed
  secant search within +-`solve_span_deg` of zero, at most `max_solve_steps`
  flight evaluations, no rendering); `draw` / `fade` are that roll minus / plus
  `shape_roll_step_deg`. The names are the ball curve of a right-handed golfer:
  a draw curves left (more closed, more negative roll). It cannot be combined
  with a non-zero `--club-roll-in-hands` but combines with
  `--ball-position-shift`. A flight that is unavailable or a bracket without a
  sign change is an error, never a silent fallback. The solved roll is logged
  and recorded in the ball record under `shot_setup`; a non-zero roll or shift
  adds a `Setup: ...` line to the clip HUD.

## Replay Mode Label

Every clip HUD carries a `Replay: ...` line and every engine row of the export
manifest carries `"replay_mode"`, so a clip never passes recorded motion off as
simulated. The mode is determined from the bundle, never guessed
(`replay_mode.py`; labels in `replay_mode_settings.json`):

- `kinematic_replay`: every effort is zero (recorded motion, no dynamics).
- `feedback_tracked_dynamics`: non-zero efforts from the closed-loop tracker
  (the bundle provenance records its `controller`).
- `open_loop_dynamics`: efforts replayed from an initial state; the bundle
  must declare it with provenance `replay_mode: open_loop_dynamics`, as no
  writer does yet.
- `inverse_dynamics`: the recorded motion with efforts from inverse dynamics
  and allocation (the kinetic match, `motion_matching/kinetic_match.py`, which
  declares it). The motion is not simulated, and the efforts are one solution
  in the null space of the closed arm loop.
- `unknown_replay_mode` (shown as "Unknown Replay Mode (provenance not recorded)"): anything else.

`--no-hud` removes the line; the manifest field stays.

## Aim Line

- `--aim-line {finish,address}` picks the line the plan view, side figure and
  wide-view ground line are read against. `finish` runs from the ball to where
  it finishes (the rest point with a roll), so a draw or fade that ends on
  target reads as on target; `address` keeps the address target line. The
  default is `finish` with `--shot-shape`, else the picture is unchanged. The
  HUD and plan title name the line; flight numbers never change. Colour, width
  and halo are in `aim_line_settings.json`.
