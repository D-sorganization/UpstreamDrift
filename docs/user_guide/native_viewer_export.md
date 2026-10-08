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
  the bundle provenance `impact_time_s`, else the last sample.
- `--stride` is deprecated: it still works as an alias (one clip at the speed
  that stride implies) and logs a warning.
- The heads-up display shows the playback speed and the real swing time in
  milliseconds from impact.

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

## Requirements

Each backend reports why it cannot run (missing package, no Chromium, no
`xvfb-run`) and is skipped. See the tool README for environment variables.
