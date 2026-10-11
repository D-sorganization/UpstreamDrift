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
- `--no-overlay`, `--no-grid`, `--views`, `--fps` (default 60), `--size` and
  `--preset hq|preview` control the output. `hq` is 1280x720, libx264,
  `yuv420p`, CRF 18; `preview` is 640x544.
- `--grip` adds the grip overlay (per-hand force, net force and couple at the
  grip midpoint, labelled with the split method; unavailable is shown, never
  zero) and writes `<swing>_<engine>_grip_wrench.json`. Drake uses its own KKT
  multiplier; the other viewers show the MuJoCo plant at that engine's pose.
  `--views hands_closeup --no-grid` follows the grip midpoint per frame.
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
  When neither exists (no club, ungrounded first state, no MuJoCo) no ball is
  drawn and the reason is logged and returned in `ExportResult.ball`
  (`{"unavailable": reason}`, printed by the CLI); a placement is recorded with
  its position, source and `hide_after_s`. MuJoCo and MyoSuite append a render
  scene sphere per visible frame; Drake and Pinocchio MeshCat hold one `/ball`
  sphere and toggle `visible`. **OpenSim limitation:** the simbody visualizer
  caches fixed geometry, and toggling Appearance visibility or opacity after
  `initSystem` does not redraw (verified by a smoke render), so the OpenSim ball
  stays visible after impact; this is recorded as `limitation` in the result.
  Post-impact flight is a later slice.
- Force arrows use the `body_weight` scale mode when the model mass is known:
  one body weight is 0.5 m and the ceiling is 3 m (`default_glyph_style`).
- `--speeds 1,0.5,0.25` (default: full, half, quarter) writes one clip set per playback speed, named
  `_1x`, `_0p5x`, `_0p25x`. Frames are chosen by the time-based
  `video_timing.FrameSchedule` and interpolated when the source step is coarse.
- `--impact-window 0.1` (with optional `--impact-time`) adds a `_impact_0p1x`
  clip of that many swing seconds centred on impact.
- `--stride` is deprecated (alias for one fixed index step; warns).
- The HUD shows playback speed and milliseconds from impact.

Environment: `NATIVE_VIEWER_CHROMIUM` (Chromium executable; a full Playwright
build renders much faster than the headless shell), and
`NATIVE_VIEWER_MYOSUITE_PYTHON` (interpreter that has `myosuite`).

The OpenSim backend only ever runs under `xvfb-run` and refuses to start when
`NATIVE_VIEWER_XVFB` is missing or `DISPLAY` is a real display.

## Layout

- `core.py`: settings, clip plans (speeds, impact window), the backend protocol,
  `export_swing`; frame timing lives in `src/shared/python/video_timing`.
- `compositor.py`: labelled 2x2 grid, HUD text and the impact card panel.
- `impact_card.py`: one impact card per swing from the displayed pose.
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
the run with a message (`--skip-unavailable` records it instead). The inset
plot strips are not rendered yet (GCV-19, #11725).

## Impact Parameter Card

`ExportSettings.impact_card` (default on, needs the HUD) draws the GCV-17
`ImpactCard` as a semi-transparent right-hand panel from the impact frame to the
end of each clip (`compositor.draw_impact_card`). `impact_card.impact_card_for`
computes it once per swing from the displayed pose `q`: the MuJoCo plant of the
specification export places the `Clubface Vector` body, central differences of
`q` give the rates, `clubhead_series_from_mujoco` and
`extract_impact_parameters` (impact state = last sample before the impact time,
default `TargetFrame`) give the rows. Unavailable rows show "unavailable" and a
reason, never zero. When no card can be computed (no MuJoCo, no club, speed
below 1 m/s) nothing is drawn, `ExportResult.impact_card` holds the reason, and
the showcase manifest records it. The panel shows model estimates in the default
target frame (ADR-0041), not a measurement.
