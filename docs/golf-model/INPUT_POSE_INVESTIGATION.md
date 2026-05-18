# Input pose investigation — `3DModelInputs_Impact.mat`

## TL;DR

`3DModelInputs_Impact.mat` does not store an impact pose; the values
embedded in the model workspace look like a **TOP-OF-BACKSWING** pose
with the **forward spine tilt zeroed out**. That combination is what
the user is seeing as "doesn't look right" — the figure stands fully
upright, the trail elbow is flexed ~100°, and both wrists are hinged
~80–98°. No constraint settling is required to produce the reported
artefact: the input data itself is wrong.

The file's storage format prevents a Python-only patch (it contains
`Simulink.Parameter` objects in the MAT v5 MCOS subsystem). A MATLAB
fix script is provided at `scripts/fix_impact_pose.m`.

## Field inventory

`3DModelInputs_Impact.mat` is MAT v5 (header `MATLAB 5.0 MAT-file …
Wed Dec 25 23:48:12 2024`). The only top-level variable is an MCOS
opaque blob (`Simulink.Parameter` array) plus a 944 KB
`__function_workspace__` substream. `scipy.io` cannot decode
`Simulink.Parameter.Value` scalars from this format.

The 50+ field names were recovered by reading
`SCRIPT_TransferStartPositionVelocityIntoModelFromMATFile.m`:

| Group              | Fields (X / Y / Z as applicable)                                      |
| ------------------ | --------------------------------------------------------------------- |
| Pelvis rotation    | `HipStartPosition`, `HipStartVelocity`                                |
| Pelvis translation | `TranslationStartPosition`, `TranslationStartVelocity`                |
| Spine              | `SpineStartPosition` (X, Y), `SpineStartVelocity` (X, Y)              |
| Torso (axial)      | `TorsoStartPosition`, `TorsoStartVelocity`                            |
| Scapulae           | `LScapStartPosition` (X, Y), `RScapStartPosition` (X, Y) + velocities |
| Shoulders          | `LSStartPosition` (X, Y, Z), `RSStartPosition` (X, Y, Z) + velocities |
| Elbows             | `LEStartPosition`, `REStartPosition` + velocities                     |
| Forearms           | `LFStartPosition`, `RFStartPosition` + velocities                     |
| Wrists             | `LWStartPosition` (X, Y), `RWStartPosition` (X, Y) + velocities       |

Numeric values are stored as scalar `double` inside each
`Simulink.Parameter`, in **degrees** for angular DOFs.

## How the numeric values were recovered

The Dataset Generator CSV pipeline writes the model workspace
parameters into every trial CSV under `model_<Field>` columns
(verified in
`matlab/Scripts/Dataset Generator/golf_swing_dataset_20251030/trial_001_*.csv`).
Every dated CSV in the repo carries the same signature, so it
reflects the values currently loaded from `3DModelInputs_Impact.mat`.

| Field                 | Stored value (deg) |
| --------------------- | -----------------: |
| `HipStartPositionZ`   |             -45.00 |
| `SpineStartPositionX` |           **0.00** |
| `SpineStartPositionY` |           **0.00** |
| `TorsoStartPosition`  |             -45.00 |
| `LScapStartPositionX` |              34.16 |
| `LSStartPositionX`    |             -50.37 |
| `LSStartPositionY`    |             -24.61 |
| `LSStartPositionZ`    |        **-135.72** |
| `RSStartPositionZ`    |         **+96.03** |
| `LEStartPosition`     |               5.78 |
| `REStartPosition`     |        **+100.70** |
| `LWStartPositionX`    |         **-97.84** |
| `RWStartPositionX`    |         **-80.02** |

Bold rows fail the address-pose plausibility check (see
`compare_to_reference` in
`src/shared/python/motion_matching/diagnostics/reference_pose.py`).

## Why it looks wrong

Two things compound:

1. **Spine X = 0 and Y = 0.** A real golfer at any point in the swing
   has 25–40° of forward spine tilt. With the spine vertical, every
   downstream segment ends up in space at angles that look
   anatomically impossible because the torso isn't where the eye
   expects it.
2. **Distal segments (shoulders, trail elbow, both wrists) carry
   top-of-backswing magnitudes.** `REStartPosition = 100.7°` and
   `LWStartPositionX = -97.8°` are the giveaway — those are the
   values you'd see at the top of the backswing, not at impact.
   Combined with the squared-up pelvis and zero spine tilt, the
   resulting visual is a figure standing erect with the club wrapped
   around the head.

## Data vs constraint settling

The flagged values are 50–75° outside their plausible address
ranges. No amount of loop-closure constraint settling can transform
a 100° trail-elbow flexion into a credible impact pose; the
constraint Jacobian only nudges things by a few degrees during
initialization. **The issue is in the data**, not in Simscape's
constraint solver.

The companion initial-state-diff diagnostic (sibling agent) is still
useful for catching residual drift after a future fix, but it is not
the cause of this specific report.

## Proposed fix

Two options:

- **Quick win:** treat `3DModelInputs_TopofBackswing.mat` as the
  authoritative top-of-backswing pose, and _replace_
  `3DModelInputs_Impact.mat` with values that come either from the
  impact frame of an existing motion-capture trial or from a
  hand-authored impact pose with proper forward tilt. A starter set
  is encoded in `scripts/fix_impact_pose.m`.
- **Long-term:** make the file regeneration explicit — a MATLAB
  script that takes a chosen frame from a measured swing, transfers
  the joint angles into the model workspace via
  `SCRIPT_TransferStartPositionVelocityIntoModelFromMATFile.m`, then
  saves the workspace back to a fresh `.mat`. This removes the
  hand-edited-MAT failure mode permanently.

The Python diagnostic
(`scripts/diagnose_input_pose.py --input <csv>`) flags every joint
that drifts outside the address-pose plausibility ranges and writes
both `report.md` and `skeleton.png`. Re-running it against a fixed
CSV confirms whether the pose is now credible without touching
Simscape.
