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

Clips are named `<swing>_<engine>_<view>.mp4` and `<swing>_<engine>_2x2.mp4`.

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
