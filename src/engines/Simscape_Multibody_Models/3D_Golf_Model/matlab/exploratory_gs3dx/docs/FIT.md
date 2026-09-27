# Fitting the Model to the Golfer's Capture

Issue [#10979](https://github.com/D-sorganization/UpstreamDrift/issues/10979),
epic [#10950](https://github.com/D-sorganization/UpstreamDrift/issues/10950).
MATLAB R2025b.

**Owner direction (2026-09-27).** The golfer's anthropometry is unknown, so
it must be matched to the capture data while the swing is fitted.

Two things can be identified from marker kinematics alone:

- segment lengths;
- where the markers sit on each segment.

Body mass cannot be identified: it scales every force equally. It stays the
assumed 80 kg of [ANTHROPOMETRY.md](ANTHROPOMETRY.md), and forces are
reported in body weights.

## 1. Joint Centres From the Markers

`gs3dx_capture_joint_centres` turns the skin and shoe markers into per-frame
joint-centre estimates. They are expressed in the address target frame
[facing, toward target, up], with the origin at the address waist centre.

| Point                  | From                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------- |
| Pelvis origin and axes | the four waist markers                                                                      |
| Hips                   | pelvis + [0, ±0.09, −0.10] m in the pelvis axes (the GS3DX leg-mount convention)            |
| Knees, ankles          | `KneeOut` / `AnkleOut`, moved 5 cm / 3.5 cm toward the midline                              |
| Shoulders              | acromion markers lowered 4 cm; `RShoulderTop` rebuilt from `RShoulderBack` in 80% of frames |
| Elbows, wrists         | `ElbowOut`, `WristTop`                                                                      |
| Club head, grip        | the two club-marker cluster centroids                                                       |

The offsets are typical anatomy, not measured on this golfer. Rigid segments
must keep their length, so the spread over the 654 frames measures how well
each proxy tracks:

| Segment                 | Median (m)    | SD (m)        |
| ----------------------- | ------------- | ------------- |
| Thigh L / R             | 0.461 / 0.459 | 0.013 / 0.027 |
| Shank L / R             | 0.425 / 0.423 | 0.003 / 0.003 |
| Upper arm L / R         | 0.305 / 0.364 | 0.009 / 0.018 |
| Forearm L / R           | 0.278 / 0.282 | 0.006 / 0.006 |
| Shoulder width (GH–GH)  | 0.321         | 0.029         |
| Pelvis to shoulder line | 0.488         | 0.015         |

The trail upper arm is not used. Its shoulder point is rebuilt in most
frames, and it reads 6 cm longer than the lead arm.

## 2. `GS3DX_Fit`: Segment Lengths From the Data

`gs3dx_fit_lengths` maps those lengths onto the model variables.
`gs3dx_build_fit` copies `GS3DX_Golfer` to `GS3DX_Fit` and applies them.

| Variable                                     | Before     | After (from the capture)                 |
| -------------------------------------------- | ---------- | ---------------------------------------- |
| `FitHubtoSLength` (each shoulder hub)        | 10 in      | 6.31 in (half GH–GH)                     |
| `FitUpperArmLength`                          | 12 in      | 11.99 in (lead arm)                      |
| `FitLowerArmLength`                          | 14 in      | 11.02 in (mean forearm)                  |
| `FitLowerTorsoLength`, `FitUpperTorsoLength` | 12 + 12 in | 9.60 + 9.60 in (pelvis to shoulder line) |
| `ThighLength`                                | 0.4365 m   | 0.460 m                                  |
| `ShankLength`                                | 0.4437 m   | 0.424 m                                  |

The regression drive file sets `LowerTorsoLength` and `UpperTorsoLength`.
So the eleven upper-body cylinders are re-pointed at new `Fit*` variables,
which no drive file sets.

- The frames follow the solids' geometry, so the joints move with the
  lengths.
- The masses stay the `GS3DX_Golfer` table.
- The edit is parameter-only: the block count is unchanged, and the model
  still compiles to 967 blocks.

The model's own forward kinematics (KinematicsSolver) confirms the result:

| Distance              | GS3DX_Golfer | GS3DX_Fit | Data  |
| --------------------- | ------------ | --------- | ----- |
| Lead GH to elbow      | 0.305        | 0.305     | 0.305 |
| Lead elbow to wrist   | 0.356        | 0.280     | 0.278 |
| Pelvis to scapula hub | 0.612        | 0.490     | 0.488 |

## 3. Whole-Body IK: `gs3dx_whole_body_ik`

The IK fits the model's joint positions, frame by frame, to 14 points:

- pelvis, hips, knees and ankles;
- shoulders, elbows and wrists;
- club head.

**How it works.**

- **Forward kinematics.** It comes from Simscape's `KinematicsSolver` on the
  model itself (5 ms per evaluation), so the fit sees the real geometry.
- **Least squares outside the solver.** The solver refuses more targets than
  degrees of freedom (status −3), so the least squares is done outside it:
  `lsqnonlin`, Levenberg–Marquardt, over the 33 independent coordinates.
- **The grip loop.** Both hands are welded to the club. The right shoulder,
  elbow and wrist are therefore left to the solver, which closes that loop
  in every evaluation.
- **Rotation conventions.** Spherical joints use rotation vectors. The
  solver's frame `Rotation` outputs are intrinsic X-Y-Z angles, a convention
  verified against `axang2rotm` to 3e-16.
- **Marker offsets.** Each marker is modelled as a constant offset in the
  frame of the body that carries it. The offsets are calibrated by
  alternating tracking with the closed-form update, which is OpenSim's
  marker-registration step.
- **Tracking.** Frames are tracked forward, then backward, keeping the lower
  cost per frame. The backward pass repairs cold-start local minima: the
  first frame of a window went from 193 mm to 27 mm.

**Frame and ground.** The capture frame is used as the model World (both
Z-up, gravity −Z); the free pelvis joint absorbs the placement.

### Results

A short window calibrated on itself fits to 1 mm RMS. That is overfitting:
the 42 offset parameters absorb the error. The real test is out of sample:
calibrate the offsets on the **backswing only**, then fit the last 0.25 s
before impact with them held fixed.

| Target (downswing, out of sample) | Median | Max   |
| --------------------------------- | ------ | ----- |
| Pelvis, hips                      | 5–7 mm | 22 mm |
| Knees, ankles                     | 2–9 mm | 28 mm |
| Shoulders                         | 2–8 mm | 15 mm |
| Lead elbow                        | 6 mm   | 26 mm |
| Trail elbow                       | 30 mm  | 33 mm |
| Club head                         | 40 mm  | 47 mm |
| Trail wrist                       | 52 mm  | 56 mm |
| Lead wrist                        | 88 mm  | 92 mm |
| **All 14 (RMS per frame)**        | 31 mm  | 32 mm |

The **whole trial** (654 frames) takes 63 minutes: the offsets are
calibrated on every third frame, then every frame is tracked. It fits within
**7.1 mm RMS median, 12.7 mm p95 and 20.3 mm max per frame**. No frame
exceeds 30 mm.

| Window         | Frames  | RMS median | RMS max |
| -------------- | ------- | ---------- | ------- |
| Address        | 1–150   | 4–7 mm     | 7 mm    |
| Backswing      | 151–400 | 2–7 mm     | 8 mm    |
| Downswing      | 386–476 | 8 mm       | 15 mm   |
| Impact         | 451–500 | 14 mm      | 20 mm   |
| Follow-through | 501–654 | 9–12 mm    | 14 mm   |

![Whole-body IK residual over the swing](screenshots/whole_body_ik_residual.png)

The calibrated offsets are anatomical:

- joint markers within 1–2 cm, except the trail acromion (5 cm; its marker
  is rebuilt) and the trail epicondyle (4 cm);
- wrist markers 6–7 cm from the wrist joints;
- club-head cluster 3 cm from the model club-head frame.

**Gap-filled samples are not fitted** (`jc.gap`). The club-head cluster is
missing at frames 445, 519–551 and 653–654. The first full run fitted the
linearly filled points and was pulled off track: 0.5 m errors through the
whole follow-through. With the gaps dropped from the residual, it tracks.

In sample, the whole-swing offsets absorb the grip mismatch (wrists 11 mm
median, 34 mm max). Out of sample they do not, which is the table above.

The whole-trial run above used the original grip. Section 4 re-runs it with
the fitted grip.

## 4. Hand-on-Grip Geometry: `gs3dx_fit_grip`

With the original grip, the pelvis, legs and shoulders generalized to 1–3 cm,
but the hands and club did not. What was wrong is **where the model's hands
sit on the grip**. `GS3DX_Golfer` puts both wrist joint centres 1 in to the
same side of the grip axis, 3 in apart along it. Those are literal lengths
in the original model.

`gs3dx_fit_grip(cap, ik)` measures the golfer's grip from the capture:

1. **Club pose.** The two three-marker club clusters are rigid (pairwise
   distances within 4 mm). Each frame's club pose is the least-squares
   rigid fit of the six markers.
2. **Wrist centres.** Both hands are welded to the club in the model, so
   each wrist joint centre is a fixed point in the club frame. The
   `WristTop` marker (forearm side of the joint) moves on a sphere about it.
   A sphere fit of each `WristTop` trajectory in the club frame gives the
   functional wrist centre (radius 34 / 50 mm, fit RMS 11 / 6 mm).
3. **Shaft axis.** The model's two hand-sphere centres lie on the grip axis.
   From the IK, they are carried into the club-marker frame and averaged
   (spread 0.5–2°). Only the axis is used, not the model's roll of the club
   about it.
4. **Mapping.** The wrist centres' positions along the axis give the lead
   hand position and the hand spacing.

**Only the separation across the shaft is identifiable.** Moving the shaft
sideways relative to both wrists moves the club head in the club frame, and
the IK's club-head marker offset absorbs that. No marker sits on the shaft
axis. Two estimates, from two model grips, agreed on the separation (72 and
74 mm) but not on its split between the hands (53/19 mm on opposite sides,
then 87/21 mm on the same side).

The separation is therefore split equally: one standoff on each side, as
the palms face each other across the grip.

| Variable (in)           | GS3DX_Golfer   | GS3DX_Fit (from the capture) |
| ----------------------- | -------------- | ---------------------------- |
| `FitButtToLeadHand`     | 2.5            | 3.08                         |
| `FitHandSpacing`        | 3.0            | 3.03 (77 mm)                 |
| `FitGripToShaft`        | 5.0            | 4.39 (butt to shaft 10.5 in) |
| `FitLeftWristStandoff`  | 1.0, same side | 1.46, **flipped**            |
| `FitRightWristStandoff` | 1.0            | 1.46                         |

`gs3dx_build_fit` re-points the six grip solids at these variables. It also
swaps the end features of the lead standoff's two frames: the wrist frame
moves to the bottom curve and the grip frame to the top. That reverses the
standoff direction without changing either frame's axes. The edit is
parameter-only: the block count is unchanged and the model still compiles
to 967 blocks.

**Out of sample** (the same backswing calibration and downswing tracking as
section 3; the fitted grip tracked every 6th frame):

| Target                | Original grip | Fitted grip |
| --------------------- | ------------- | ----------- |
| Lead wrist (median)   | 88 mm         | 18 mm       |
| Trail wrist (median)  | 52 mm         | 15 mm       |
| Trail elbow (median)  | 30 mm         | 16 mm       |
| Club head (median)    | 40 mm         | 29 mm       |
| RMS per frame (max)   | 32 mm         | 17 mm       |
| Largest marker offset | 68 mm         | 46 mm       |

The per-target medians come from the first-pass trial (a 53/19 mm
opposite-sides split). The equal split gives the same fit, as the
identifiability argument predicts: RMS max 17.1 mm, wrists 26.7 mm max,
club head 33.2 mm max.

**Whole trial, fitted grip** (654 frames, 43 min): 7.0 mm RMS median,
16.1 mm p95, 19.1 mm max per frame; no frame exceeds 30 mm.

| In sample, whole trial | Original grip | Fitted grip |
| ---------------------- | ------------- | ----------- |
| Wrists (max)           | 34 / 37 mm    | 25 / 32 mm  |
| Club head (median)     | 7 mm          | 12 mm       |
| Follow-through RMS     | 11.4 mm       | 14.1 mm     |
| Largest marker offset  | 70 mm         | 47 mm       |

In sample, the original grip's oversized offsets absorbed part of the club
error. With anatomical offsets that error shows, so the club head and
follow-through read slightly worse. Out of sample, the fitted grip is
better everywhere.

- The wrist marker offsets fell from 6–7 cm to 2.8 cm, the expected depth of
  a dorsal wrist marker.
- **Fixed point.** Re-estimating the grip from the rebuilt model's IK
  returns its own values within 0.07 in.

`test_gs3dx_fit` pins all of this:

- wrists under 35 mm and the club head under 40 mm out of sample;
- RMS under 25 mm (tightened from 35 mm);
- marker offsets under 6 cm (tightened from 12 cm);
- the grip fixed point within 0.25 in.

## 5. Leg Servo References: `gs3dx_leg_reference`, `GS3DX_FitLegs`

`gs3dx_leg_reference(ik, jc, cap)` turns the whole-trial IK on the fitted
grip into the 12 angles the leg servo tracks
(`[L hip X Y Z, knee, ankle X Y, then R]`, in degrees) and their rates, on
the capture's time base.

- **Pelvis path.** The model's own pelvis frame (the follower of the pelvis
  joint, `Lower Torso`) at every IK frame. It is also returned in
  pelvis-joint coordinates for the start state.
- **Foot path.** The feet are not planted. The trail heel is already 36 mm
  up at impact, and a planted trail foot went out of reach 8 frames later.
  Each foot follows the ankle joint centre. Its orientation is the address
  foot frame, carried by the rotation of the (ankle, ToeIn, ToeOut) triad
  since address.
- **Floor.** The right ankle-centre proxy sat 18.5 mm above the left at
  address, which left the right foot hanging above the ground. Each foot
  path is shifted by a constant (±9.3 mm) so both address soles sit on
  their mean height.
- **Filtering.** Positions and quaternions are filtered with a 4th-order
  zero-phase Butterworth at 10 Hz.
- **Torsion.** The model ankle is a universal joint, with no rotation about
  the shank. A foot pose therefore fixes the knee's swivel about the
  hip-ankle line. Two alternatives were tried and rejected:

  - prescribing the measured foot yaw put the knees 4–5 cm from the capture
    (p95 8–9 cm);
  - a free yaw matched the knees but spun the lead foot by up to 99° once
    the knee straightened.

  So one constant yaw offset per foot is fitted over the still address
  frames, weighing the knee centre, ankle position and sole normal. The
  offsets are +24.0° (lead) and −17.9° (trail).

- **Leg angles.** `gs3dx_leg_ik` solves exactly (six angles for six
  foot-pose numbers) for every filtered pelvis pose.
- **Reach.** Where the hip-to-ankle distance exceeds 0.999 of the leg
  length, the ankle target moves toward the hip onto it. This is at most
  1.5 mm, only after impact (from frame 506).

| Whole trial (654 frames)         | Lead         | Trail        |
| -------------------------------- | ------------ | ------------ |
| Knee vs capture, to impact       | 19 mm median | 20 mm median |
| Knee vs capture, to impact (p95) | 56 mm        | 49 mm        |
| Knee vs capture at impact        | 48 mm        | 55 mm        |
| Address torsion fit, knee        | 14 mm        | 13 mm        |
| Peak knee rate to impact         | 634 deg/s    | 480 deg/s    |

The knees drift from the capture through the downswing because the real
shank rotates over the foot, which the model's ankle cannot do. A
three-axis ankle would remove this. It is a model change, left for a later
variant. The pelvis path and the foot path are met exactly.

`gs3dx_build_fit_legs(info, ref)` builds `GS3DX_FitLegs` from `GS3DX_Fit`:

- **Servo.** The `Lower Body/Leg Torque Commands` Constant becomes a
  From Workspace block of the same name. It plays
  `LegTorqueCommand + Kp .* LegReferenceAngle + Kd .* LegReferenceRate` on
  `LegReferenceTime`, so the servo torque is
  `LegTorqueCommand + Kp (q_ref − q) + Kd (qd_ref − qd)`. It is one block for
  one (773 nonvirtual, 967 compiled).
- **Start state.** The legs and the pelvis start on the reference. The
  start variables are also saved as the struct `LegReferenceStart`. The
  drive file overrides two of them, so the caller passes the struct after
  the drive:
  - the pelvis start;
  - `PlaneTilt`, the tilt of the pelvis joint base about World X. The drive
    file sets 22.5°; the saved model, the IK and the reference use 30°.
    This mismatch first tilted the start pelvis by 7.5° and drove the feet
    13 BW into the ground.
- **Ground.** The capture frame is the model World, so `GroundRotation` is
  the identity and `GroundOffset` lies under the start feet.

Standing from rest at address (the `GS3DX_Golfer` standing test, same
bounds):

| Model                  | Slip L / R   | Lift L / R   | Support (BW) | Pelvis travel |
| ---------------------- | ------------ | ------------ | ------------ | ------------- |
| `GS3DX_Golfer`         | 1.3 / 1.4 mm | 0 / 0 mm     | 0 – 1.14     | 29 mm         |
| `GS3DX_Fit` (stale q0) | 1.5 / 1.5 mm | 2.8 / 3.0 mm | 0.44 – 3.07  | 33 mm         |
| `GS3DX_FitLegs`        | 1.8 / 1.5 mm | 0 / 0 mm     | 0.24 – 1.06  | 38 mm         |

`GS3DX_Fit` still holds the stance angles computed for the old leg lengths,
and it lifts its feet. The pelvis travel in every row comes from the
upper-body drive. The impact drive is passive (`ModelingMode` 0: every
upper-body joint torque is zero), and it starts mid-downswing, so it is not
synchronized with the capture.

`test_gs3dx_fit_legs` checks the following on an IK of every 18th frame to
impact:

- the reference against the foot path (1e-6);
- the clamp, levelling, torsion and knee bounds;
- the servo block and the start variables;
- the standing test.

## Next

1. Upper-body drive: inverse dynamics of the whole-trial IK motion as
   feedforward torques, plus PD tracking of the IK joint angles, in place
   of the passive impact drive.
2. Compare the summed contact GRF with `gs3dx_kinematic_grf` (1.24–1.33 BW
   peak about 60 ms before impact) and read the pelvis residual.
