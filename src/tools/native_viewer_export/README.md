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

### Decorative Address Ball (GCV-13, #11719)

Every backend draws a regulation-radius ball at the swing's resolved address
position (`ball.resolve_address_ball`, from the bundle's true `q[0]` frame
composed with the club assembly -- never the exporter's own static reference
pose). It is on by default (`ExportSettings.ball = True`, `--no-ball`
disables it) and is visual only: massless and non-colliding, and it never
moves. When the address frame cannot be resolved (no club body, or the
address clubhead is not plausibly grounded), the reason is logged and no
ball is drawn -- never a guessed position.

- Drake and Pinocchio (`backends/drake_meshcat.py`,
  `backends/pinocchio_meshcat.py`): a `visual_ball` sphere registered
  directly on the MeshCat scene.
- MuJoCo and MyoSuite (`backends/mujoco_worker.py`,
  `backends/myosuite_worker.py`): a `visual_ball` MJCF geom
  (`contype="0" conaffinity="0" mass="0"`) attached by the shared
  `backends/_ball.py` helper, which also strips whatever the appearance/visual
  layer's own static-reference-pose fallback may have drawn.
- OpenSim (`backends/opensim_worker.py`): a sphere attached once to the
  model's ground frame when the model is built -- a plain geometry
  component, not a body, so (like the per-frame force/torque overlays, which
  stay 2D-projected because the real simbody-visualizer takes no dynamic 3D
  decorations from Python) it adds no mass or DOF.

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
- `ball.py`: `resolve_address_ball` (GCV-13, #11719), the decorative address
  ball resolved from the swing bundle's address frame.
- `compositor.py`: labelled 2x2 grid and HUD text.
- `overlay.py`: overlay feed built from `force_overlay.bundle_provider`.
- `overlay2d.py`: pinhole projection for the OpenSim 2D glyphs.
- `backends/`: one module per engine plus shared helpers, including
  `backends/_ball.py` (the MJCF `visual_ball` geom shared by the MuJoCo and
  MyoSuite workers).

Camera presets live in `src/shared/python/golf_view_presets`; the glyph
pipeline is `src/shared/python/force_overlay` (ADR-0052). The OpenSim and
MeshCat viewers use the shared vertical field of view `VIEWER_FOV_Y_RAD`
(0.7 rad). Before NV-9 (#11697), MeshCat kept the 75 deg three.js default,
which left the golfer at about a quarter of the frame height.
`golf_view_presets.framing` measures the projected extent of a set of points
(`projected_extent`). It also gives the camera distance that frames them with
a 15 % margin (`fit_distance_m`).
