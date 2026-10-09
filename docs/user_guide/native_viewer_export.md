# Native Viewer Export

The native viewer export tool renders a same-input swing in the native viewer of
each physics engine and writes mp4 clips with force and torque overlays. It
complements the shared renderers: each clip shows what the engine's own viewer
displays for the engine's own rollout.

## Quick Start

```bash
python3 -m src.tools.native_viewer_export \
  --bundle docs/.../driver.npz --swing driver --club Driver --out renders/ \
  --receipt-drake driver_closed_drake.json
```

Clips are named `<swing>_<engine>_<view>_<speed>.mp4` and
`<swing>_<engine>_2x2_<speed>.mp4`, for example `driver_drake_face_on_1x.mp4`
and `driver_drake_face_on_0p5x.mp4`.

## Playback Speed and Frame Rate

Playback is time-based. Video frame `j` shows the swing at
`t0 + j * speed / fps`, so a 1x clip really runs at real time and a 0.5x clip
at half speed, independent of the source time step. When the source step is
coarser than that, the pose is interpolated (linear for joint coordinates,
spherical for free and ball joint quaternions) instead of repeated. Force and
torque glyphs come from the nearest source sample.

- `--fps` (default 60) sets the frame rate.
- `--speeds 1,0.5` (default) writes one clip set per speed, each in `(0, 4]`;
  suffixes are `_1x`, `_0p5x`, `_0p25x`.
- `--impact-window 0.1` adds a clip of that many swing seconds centred on
  impact at 0.1x speed (suffix `_impact_0p1x`). Impact is `--impact-time`, else
  the bundle provenance `impact_time_s`, else the detected impact (see
  [Ball](#ball)), else the last sample.
- `--impact-speed 0.25` changes the speed of the impact clip. With an empty
  `--speeds=` only the impact clip is written, for example a clean 0.25x clip
  around impact.
- `--stride` is deprecated: it still works as an alias (one clip at the speed
  that stride implies) and logs a warning.
- The heads-up display shows the playback speed and the real swing time in
  milliseconds from impact. `--no-hud` writes clean frames without the view
  label, HUD or legend text.

## Quality Presets

`--preset hq` (default) renders 1280x720 tiles encoded with libx264,
`yuv420p`, CRF 18. `--preset preview` keeps the small 640x544 tiles at CRF 23.
`--size WIDTHxHEIGHT` overrides the preset size.

## Camera Views

All engines use the same four presets (golfer faces -X, target line -Y for a
right-handed golfer): face-on, down-the-line, overhead and oblique. The
look-at point defaults to the golfer's centre of mass at address, at 0.9 m
height.

## Overlays

Ground reaction forces act at the centre of pressure of each foot, joint torques
are drawn as arcs at the joint anchors and body weight acts at the centre of
mass. Wrenches come from one MuJoCo evaluation of the shared specification, so
every engine shows identical glyph definitions while the poses come from that
engine's rollout. This is a display of the shared contact law, not a
measurement.

### Arrow Scale

When the model mass is known, force arrows use the `body_weight` scale mode
(ADR-0052): one body weight (mass times 9.80665 m/s²) draws 0.5 m, and the
length ceiling is 3 m (6 body weights), so a driver swing is not silently
clamped. Without a mass the fixed scale of 0.7 m per kN with a 0.9 m ceiling is
kept. An arrow that reaches the ceiling is drawn with a distinct tip and is
counted in the legend.

### Grip Overlay

`--grip` adds the grip wrench of the hands on the club (GCV-10): the force of
each hand at its grip point, the net force and the equivalent couple at the grip
midpoint, and the per-hand moments of force. The labels are listed in
[Force Overlay](force_overlay.md#grip-wrench-per-hand-loading-midpoint-net-force-and-couple).
The legend names the left/right split method. A quantity that cannot be
computed is shown as unavailable, never as a zero arrow. Each engine also writes
`<swing>_<engine>_grip_wrench.json` for the grip plots. Drake uses its own KKT
multiplier; the other viewers show the MuJoCo plant at that engine's pose. Use
`--views hands_closeup --no-grid` for a camera that follows the grip midpoint.

### Ball

No ball is drawn. The ball enters only the impact detection, which runs when
neither `--impact-time` nor the bundle provenance gives an impact time. It
places the `Clubhead` frame by MuJoCo forward kinematics at every state and
picks the closest approach to the address position, subject to its height and
ball-radius checks (`model_appearance.club_face.impact_frame`). Without MuJoCo,
or when no frame is a valid impact, the export logs a warning and uses the last
sample.

## Requirements

Each backend reports why it cannot run (missing package, no Chromium, no
`xvfb-run`) and is skipped. See the tool README for environment variables.
