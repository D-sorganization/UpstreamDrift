# Data Audit for the Full-Body Model

Issue [#10985](https://github.com/D-sorganization/UpstreamDrift/issues/10985),
epic [#10950](https://github.com/D-sorganization/UpstreamDrift/issues/10950).
MATLAB R2025b, Python ezc3d via `pyenv`.

This audit asks one question: do we have every parameter a full-body model
with ground contact needs, and where does each one come from? The short
answer is no. The geometry of the stance is measured, but four things are
missing or in conflict:

- No ground reaction force (GRF) data exists.
- The upper-body masses double-count the legs.
- The foot and thigh proxies disagree with the anthropometric tables.
- The pelvis path in the existing drive cannot be walked by planted feet.

## Capture Inventory

`gs3dx_capture_stance` reads the capture and never writes it. The tests in
`tests/test_gs3dx_capture.m` pin these facts.

| File                                    | SHA-256 (prefix) | Frames | Rate   | Force plates | Analog channels |
| --------------------------------------- | ---------------- | ------ | ------ | ------------ | --------------- |
| `data/C3D_TA_Driver.c3d` (tour average) | `545405cc…`      | 654    | 360 Hz | 0 used       | 0               |
| `C3DExport Tour average.c3d`            | `545405cc…`      | —      | —      | —            | —               |
| Iron capture (`C3DExport … iron`)       | `395deb1f…`      | 657    | 359 Hz | 0 used       | 0               |

- The two tour-average files are byte-identical.
- **None of the captures has any GRF or analog data.** Ground contact cannot
  be validated against measured forces. It can only be checked for physical
  consistency (Newton's second law, see [GROUND_CONTACT.md](GROUND_CONTACT.md)).
- Leg and foot markers are complete in the driver capture.
  - `WaistRight` is missing 5 frames.
  - `RShoulderTop` is missing 80–85% of frames in every file, so it is unusable.
- Axes: the capture is Y-up. It is converted to Z-up as (x, −z, y), and the
  golfer faces −X.
- Impact is frame 476 (1-based): the peak speed of the club-head marker
  cluster. The top of the backswing is frame 378, found at the pelvis-yaw
  extremum.

## Stance at Address

The target frame is [facing, toward target, up] with its origin at the waist
centre. Markers are skin and shoe markers, so every value below is a proxy,
not a joint centre.

| Quantity                            | Left           | Right           | Used in the model                          |
| ----------------------------------- | -------------- | --------------- | ------------------------------------------ |
| Ankle (facing, lateral) from waist  | (0.021, 0.319) | (0.021, −0.332) | `leg.stance.ankle_L/R`                     |
| Ankle drop below waist centre       | 0.941 m        | 0.920 m         | `leg.stance.drop` = 0.931 (mean)           |
| Foot yaw (toe out, + toward target) | −2.15°         | +8.15°          | `leg.stance.foot_yaw_L/R`                  |
| Toe marker width                    | 0.107 m        | 0.117 m         | contact width 0.11; brick `FootWidth` 0.10 |
| Knee angle proxy (180 = straight)   | 166.7°         | 164.3°          | not used (the IK sets the knee)            |
| Ankle stance width                  | 0.650 m        |                 | FullBody weld spacing was 0.18 m           |

The malleolus marker sits about 3.5 cm lateral to the ankle centre, so the
model moves each ankle 0.035 m toward the midline (`leg.stance.inset`).

## Segment Parameters

| Parameter                      | Source in the model                   | Capture proxy   | Status                                                       |
| ------------------------------ | ------------------------------------- | --------------- | ------------------------------------------------------------ |
| Shank length                   | de Leva, 1.80 m                       | 0.423 m (−4.6%) | consistent                                                   |
| Thigh length                   | de Leva, 1.80 m                       | 0.515 m (+18%)  | proxy overstates it (the waist marker is not the hip centre) |
| Foot length (ankle to toe)     | de Leva 0.2736 m                      | 0.169 m (−38%)  | **conflict**: the marker sits on the forefoot, not the tip   |
| Ankle height                   | de Leva 0.070 m                       | 0.094 m         | marker on the malleolus, not the sole                        |
| Thigh mass                     | de Leva 11.3 kg                       | —               | **conflict** with canonical spec 7.05 kg                     |
| Upper-body masses              | literal values in the original solids | —               | **conflict**, see below                                      |
| Joint stiffness and damping    | none                                  | —               | **missing**: no data                                         |
| Leg joint torques              | `LegTorqueCommand` = 0                | —               | **missing**: servo holds the stance                          |
| Foot–ground stiffness/friction | `leg.contact` (assumed)               | —               | **missing**: no GRF to fit                                   |

### Mass Double-Count

The original upper body was built to be carried by a driven pelvis, and its
solids carry the mass of the whole trunk.

- Inertia sensor: the upper body is 77.6 kg.
- The trunk alone (LowerTorso 20, UpperTorso 4 + 16, HubtoLS/HubtoRS 10 each)
  is 60 kg, against about 35 kg in de Leva.
- The literal masses mix units: g (grip), lbm (head 11 lbm = 4.99 kg, arms)
  and kg.
- Adding de Leva legs makes GS3DX_FullBody 109.4 kg, well above the 80 kg the
  legs are scaled for.

**Decision needed from the owner.** Either rescale the trunk so that
trunk + legs is about 80 kg, or re-derive the leg masses from the golfer.
The exploratory models leave the upper body unchanged, so every existing
comparison against the original still holds.

## Drive Versus Planted Feet

The impact drive starts mid-downswing (clubhead 19.5 m/s at t = 0). Over
0.3 s, Quat's pelvis moves:

- x from −0.171 to 0.080 m
- y from −0.042 to 0.084 m
- z from −0.058 to 0.121 m

With the feet at the captured stance, following that path would need 0.62–0.70
m of ankle drop and 88–102° of knee flexion. That is out of reach or
unrealistic. The pelvis path in the drive is therefore **not** compatible
with planted feet. It was never meant to be: it came from fitting the upper
body alone.

This is why `GS3DX_FullBodyContact` turns the pelvis drive off and lets the
legs carry the pelvis. Refitting the swing on the full body
([#10979](https://github.com/D-sorganization/UpstreamDrift/issues/10979))
needs a pelvis path taken from the capture, not from the drive.

## What Is Needed Before a Validated Full-Body Swing

1. Force-plate data (GRF per foot), or a published tour-average GRF
   profile, to fit contact stiffness and friction and to validate.
2. An owner decision on the trunk mass (above).
3. A pelvis trajectory from the capture, in the model frame, for the #10979
   refit.
4. Leg joint torques (inverse dynamics once 1–3 exist); the servo is a
   placeholder that holds the address stance.
