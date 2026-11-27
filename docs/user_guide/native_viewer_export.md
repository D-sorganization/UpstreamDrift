# Native Viewer Export Quick Start

The native viewer export renders a swing bundle (a `.npz` file holding the
golfer's joint angles over time, plus any recorded joint efforts) into video
clips. It runs headless: no window opens. Clips are written as mp4 files, one
per camera view and one 2x2 grid, at several playback speeds.

For the developer reference see the tool README in
`src/tools/native_viewer_export/`. To learn what each drawn arrow, arc, shape
or panel means, read [What Am I Looking At](../help/native_viewer_export.md).

## Which Kind of Motion Is in Your Bundle

A bundle can hold three kinds of motion. The exporter treats them the same way
when it draws the body, but the force pictures mean different things.

| Kind                      | What it is                                                                | What the exporter shows                                                                                                                                                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Kinematic replay          | The fitted joint angles played back, with no forces applied.              | Every recorded effort is zero. By default the clips carry the note "Kinematic Replay: Forces Not Shown" and leave out the force and torque arrows; `--force-vectors` or `--torque-vectors` draws them anyway, which does not make them measured or applied forces. |
| Feedback-tracked dynamics | A controller's torques drove the model to follow the fit.                 | Efforts are non-zero. The torques are what the tracking controller applied.                                                                                                                                                                                        |
| Open-loop dynamics        | The torques are replayed with no feedback, so the model moves on its own. | Efforts are non-zero. Not accepted for a full swing yet: treat such clips as experiments.                                                                                                                                                                          |

The exporter only checks whether the step and hold effort tables are all zero.
This is a display rule, not proof of how the motion was produced: it does not
read the bundle's provenance, and non-zero efforts do not tell feedback-tracked
dynamics from open-loop dynamics. Use the run's receipt to identify the
dynamics. Ellipsoids, impact factors, ball flight, lighting and the ball work
for every kind.

## Minimal Command

```bash
python3 -m src.tools.native_viewer_export \
  --bundle bundle.npz --swing driver --club Driver --out renders/
```

Clips are named `<swing>_<engine>_<view>_<speed>.mp4` and
`<swing>_<engine>_2x2_<speed>.mp4`, for example `driver_mujoco_face_on_1x.mp4`.
Use `--engines mujoco` for the MuJoCo renderer. Without a graphics card set
`MUJOCO_GL=osmesa`. Each backend that cannot run is skipped with a reason.
Run `--help` for the full flag list.

## Recipes

### Views, Speeds, and the Impact Window

```bash
python3 -m src.tools.native_viewer_export --bundle bundle.npz --swing driver \
  --out renders/ --views face_on,down_the_line --speeds 1,0.25 \
  --impact-window 0.1 --impact-speed 0.1
```

Views are `face_on`, `down_the_line`, `overhead` and `oblique` (the default
four), plus `down_the_line_flight` and `hands_closeup`. `--speeds` takes values
in (0, 4] and writes one set per speed (default 1, 0.5 and 0.25). The impact
window adds a clip of that many swing seconds centred on impact. Limit: the
impact time comes from `--impact-time`, the bundle, or detection; if none is
found the last sample is used. `--preset preview` and `--size WxH` make smaller
clips, `--no-hud` removes text, `--no-grid` skips the 2x2.

### Impact Factors Panel

```bash
... --impact-factors
```

Draws club speed, path, face, face to path, attack angle, loft and strike
point. Values are blank before impact and freeze at impact. Limit: kinematic
values at the ball position the tool places; see the help page.

### Camera Framing

All engines use the same four presets (golfer faces -X, target line -Y for a
right-handed golfer): face-on, down-the-line, overhead and oblique. The
look-at point defaults to the golfer's centre of mass at address, at 0.9 m
height (`camera_framing_settings.json`).

For a club other than a teed driver, set the framing: `--camera-height 0.6`
lowers only the look-at height, `--camera-lookat X,Y,Z` (metres, world frame)
replaces the whole point and wins over the automatic one, and
`--camera-distance 3.5` sets the camera distance. Without these flags the
framing is unchanged.

### Ball Flight Tracer and Inset

```bash
... --ball-flight --views down_the_line_flight,face_on
```

After impact the ball flies along the model-predicted carry path with a
tracer, a small flight inset, and the clip holds until landing. Limit: carry
only, ending at first ground contact; no bounce or roll. Use `--no-ball` to
hide the ball before impact.

### Showpiece Lighting and Colours

```bash
... --lighting showpiece --skin-tone skin_4 --top-color "#1f4e79" \
  --bottom-color "#333333" --glove lead
```

Showpiece lighting adds lights, one soft shadow and a ground tone (MuJoCo
export only). Other appearance flags: `--hair`, `--clothing`, `--appearance`.
Limit: appearance never changes the motion or any computed number.

### Club Roll and Ball Position

```bash
... --club-roll-in-hands 3 --ball-position-shift 0.02
```

`--club-roll-in-hands DEG` turns the club about its shaft relative to the
hands (positive opens the face), which changes the face angle and loft seen at
impact. `--ball-position-shift M` moves the ball along the target line
(positive toward the target). The shift is applied only when the ball is
placed at the address position (`SOURCE_ADDRESS`); a ball already on the swing
or taken from the bundle's recorded ball (`bundle:ball_impact`) is not moved,
so the option has no effect there. To confirm it was applied, open the ball
record in the output: it contains `ball_position_shift_m` only when the shift
was applied. If that key is missing, the requested position was not rendered.
Limits: the same body motion is replayed, the
shaft is treated as rigid, and forward of the captured ball position only
about the first few centimetres are valid because the real ball slowed the
measured club. Both are bounded by `shot_setup_settings.json`.

`--shot-shape straight|draw|fade` solves the roll instead: `straight` is the
roll at which the flight model lands the ball on the target line, and `draw` /
`fade` close / open it by a set step (right-handed ball curve). It cannot be
combined with a non-zero `--club-roll-in-hands`, needs a flight the model can
compute, and stops with an error rather than falling back. The solved roll is
written to the ball record and the clip HUD.

### Force and Torque Vectors, Shading, and Grip Forces

```bash
... --grip --views hands_closeup --palm-thumb-force --segment-shading
```

Force arrows and torque arcs are on by default and are hidden on a kinematic
replay unless you pass `--force-vectors` or `--torque-vectors`. `--grip` adds
the hand forces on the club and a `<swing>_<engine>_grip_wrench.json` file.
Limits: these need a bundle with non-zero efforts to mean anything. They are
model displays, not measurements. See the Known Issues below.

### Ellipsoids

```bash
... --interface-ellipsoid mobility --interface-ellipsoid effective-mass \
  --mobility-points hands,clubhead --ellipsoid-color-by kind
```

Kinds are `mobility`, `effective-mass`, `acceleration-capacity` and
`force-capacity`. `--hand-mobility` is a shortcut for the hands. Limit: with
the default per-point scaling, sizes cannot be compared between points.

## Known Issues

As of this writing (epic #15150), fixes are under way on other branches:

- Segment shading and the palm-thumb arrow can still be drawn on a kinematic
  replay, where they carry no physical meaning.
- A capacity ellipsoid requested without torque limits can fall back to
  mobility sizes while keeping the capacity label.
- `--drift-arrow` is accepted but draws nothing.

## Requirements

Python 3.11 or newer with the project dependencies. Each backend reports why
it cannot run (missing package, no `xvfb-run`) and is skipped. The
[Force Overlay](force_overlay.md) page lists the arrow and arc palette.
