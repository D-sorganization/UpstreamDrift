# Golf Grip Controls

This guide explains the grip presets and sliders of the Character Builder (desktop
and web) and the matching command-line flags (HRG-GR4, #13205).

## Golf Terms

- **Lead hand and trail hand.** The lead hand is the left hand of a right-handed
  golfer, the trail hand the right.
- **Grip strength (rotation about the shaft).** How far a hand is turned about
  the shaft. A strong grip turns the hands away from the target, so more lead
  knuckles show at address; a weak grip turns them toward it. Positive degrees
  are stronger. The presets are weak (-15 degrees, about one knuckle), neutral (0,
  about two) and strong (+15, about three). They are product presets, not
  anatomical thresholds, and the knuckle count varies with hand size.
- **Palm diagonal.** The angle at which the shaft crosses the palm. A smaller
  angle puts the shaft more in the fingers, a larger one more in the palm. The
  neutral value is 40 degrees.
- **Set angle.** The angle at the wrist, about the flexion axis.
- **Hand spacing and depth.** The gap between the two hands on the grip and how
  far the lead hand sits below the butt of the club, both in metres.
- **Grip style.** How the little finger of the trail hand sits: overlap,
  interlock or ten finger. This changes only the picture.

## Physical and Picture-Only Choices

Strength, palm diagonal, set angle, depth and spacing are **physical**: they move
the hands in the model. Choosing one makes a **new model revision** (the spec
records its parent and the grip configuration), so the revision **needs a refit**
before its results are compared with a swing captured before the change.

Style and the lead thumb are **picture-only**. They are written to the
`.appearance.json` sidecar (`grip` block) and never change `physics_spec_sha256`.
A physical value is refused in a sidecar with a message naming the offending key.

Precedence, highest first: your pick, then the document (the model's own grip or
the sidecar), then the defaults. A preset sets both hands to that strength at the
neutral palm diagonal and set angle; an explicit slider then wins over the preset.

## Desktop and Web

Open the Character Builder. The Grip group has Grip Strength, the five sliders,
Grip Style and Lead Thumb Offset. The summary updates as you pick and says when a
change makes a new revision. On the web, Build Grip Revision shows the new
revision and physics hashes; the appearance export carries only the picture-only
picks.

## Command Line

The physical flags belong to the spec builder, which writes the new revision:

```bash
python3 -m src.shared.python.motion_matching.execution.spec_builder \
  --native N --osim O --native-candidate C --stature 1.80 --mass 84 \
  --output out --grip-preset strong --hand-spacing 0.09 --grip-style interlock
```

- `--grip-preset {weak,neutral,strong}`
- `--lead-shaft-rotation`, `--trail-shaft-rotation` (degrees, positive stronger)
- `--lead-palm-diagonal`, `--trail-palm-diagonal` (degrees)
- `--hand-spacing` (metres)
- `--grip-style {overlap,interlock,ten_finger}` (picture only; also written to
  `<spec>.appearance.json`)

`native_viewer_export` accepts `--grip-style`. A physical flag there is refused:
build the revision with the spec builder, refit it, then export from its receipt.
