# GS3DX Joint Range of Motion

Issue [#11158](https://github.com/D-sorganization/UpstreamDrift/issues/11158),
epic [#10950](https://github.com/D-sorganization/UpstreamDrift/issues/10950).
MATLAB R2025b, measured 2026-09-29.

The whole-body IK fits joint centres, not joint angles. Several anatomically
impossible poses put the joint centres in the same places, and without a bound
the fit takes some of them. `tools/gs3dx_joint_rom.m` is the single table of
normal adult ranges for every revolute and universal joint axis of
`GS3DX_Human`. The IK penalty, the joint limits and every range-of-motion
check read that table.

## The Table

Each row is one joint primitive. It gives the anatomical motion that is
positive, a sign, the primitive's value at the anatomical neutral, the normal
range and a source. The anatomical angle is
`A = sign * wrap180(q - neutral)`.

| Joint           | Positive motion          | Sign (L/R) | Neutral (deg) | Range (deg)       | Source                    |
| --------------- | ------------------------ | ---------- | ------------- | ----------------- | ------------------------- |
| Neck Rx / Ry    | flexion / bend to trail  | +1         | 24.01 / 7.67  | ±45 / ±45         | AAOS                      |
| Spine Rx / Ry   | flexion / bend to trail  | +1         | 0             | −25..80 / ±35     | AAOS thoracolumbar        |
| Torso Rz        | rotation toward target   | +1         | −26.81        | ±60               | Cheetham 2001, Joyce 2010 |
| Elbow Rz        | flexion                  | +1 / −1    | 0             | −5..150           | AAOS                      |
| Forearm Rz      | pronation                | +1 / −1    | 90            | ±80               | AAOS                      |
| Scapula Rx / Ry | protraction / elevation  | +1 / −1    | 0             | ±25 / −10..40     | Ludewig 2009              |
| Wrist Rx / Ry   | radial dev. / flexion    | +1, +1/−1  | none          | arc 50 / 150      | AAOS                      |
| Knee Rz         | flexion                  | −1         | 0             | −5..135           | AAOS                      |
| Ankle Rx / Ry   | inversion / dorsiflexion | −1/+1, −1  | 0             | −15..35 / −50..20 | AAOS                      |
| Midfoot Rz      | toe extension            | −1         | 0             | −45..70           | AAOS first MTP            |

## How the Signs and Neutrals Were Measured

The probes live in the session scratchpad and are summarized here:

- **Signs.** Each primitive is turned at the address pose. The velocity of a
  distal point, `w × (p − c)`, is projected on the anatomical positive
  direction, for example the chest's forward axis for trunk flexion.
- **Knees.** The knees were settled by a separate probe. It sets each knee to
  −30, 0 and +30° on `GS3DX_Fit` and `GS3DX_Human` and measures on which side
  of the thigh line the ankle lands. The Human leg servo reference, which is
  fitted to the foot pose as well as the joint centres, holds the knees at −5
  to −57° all swing. The unregularized IK on `GS3DX_Fit` holds them at +10 to
  +53°. That is the mirror branch: the hip spins the leg about its long axis
  and the knee hyperextends, which puts the hip, knee and ankle centres in the
  same places. So flexion is −Rz.
- **Neutrals.** The neutrals are the primitive values that align the child
  segment's long axis with the parent's (spine, neck). For the torso, the
  neutral is where the chest's lateral axis is closest to the pelvis's. For
  the forearm, it is where the wrist's dorsal–palmar axis lies along the
  elbow axis. Elbows and knees are straight at zero. The scapulae, ankles and
  midfoot use the model's zero.
- **Wrists.** The drawn hands sit across the shaft, 66° off the forearm axis,
  so their long axis is no anatomical reference until the grip is fixed
  (#11157). Their rows bound the swing's arc only.

## What the Reference Motion Does Today

`human_references_stay_in_the_human_range` in `tests/test_gs3dx_joint_rom.m`
checks the references that drive `GS3DX_Human` (`<J>TrackAngle`,
`LegReferenceAngle`, `NeckReference`). It is red until the pipeline is
rebuilt from a range-limited IK. These are the excesses beyond the normal
range:

| Joint   | Motion                         | Reached (deg)           | Excess (deg) |
| ------- | ------------------------------ | ----------------------- | ------------ |
| Spine   | trunk flexion                  | −33 .. 38               | 8            |
| Spine   | lateral bend to trail          | −30 .. 81               | 46           |
| LF / RF | forearm pronation              | −22 .. 124 / −103 .. 26 | 44 / 23      |
| LScap   | lead protraction               | −27 .. 77               | 54           |
| RScap   | trail elevation                | −27 .. 49               | 26           |
| LW / RW | wrist arcs (deviation)         | 94 / 251                | 44 / 201     |
| RW      | trail wrist flexion arc        | 177                     | 27           |
| LA / RA | ankle dorsiflexion / inversion | 22 / 36                 | 2 / 1        |

## The IK Penalty

`gs3dx_whole_body_ik(..., rom_weight=W)` adds, for every bounded row, `W`
(m/rad) times how far the anatomical angle lies beyond its range to the
least-squares residual. The closed right-arm loop is included through the
solver outputs.

The penalty is applied by **continuation**. Every frame is first fitted
without it, and that chain supplies the warm start of the next frame. Each
frame is then polished from its own chain pose with the penalty on, and the
smoothing term holds the polish near that pose. Applied from the first frame
instead, a penalized pose that left the markers' basin seeded its neighbours
and the fit ran away.

Every 10th frame from address to impact (49 frames), regularized, measured
2026-09-30:

| Penalty                  | Mean RMS | Worst frame | Largest excess               |
| ------------------------ | -------- | ----------- | ---------------------------- |
| None                     | 23.3 mm  | 37 mm       | 90° (lead forearm pronation) |
| Weight 1, from the start | 38.6 mm  | 427 mm      | 28.3°                        |
| Weight 3, from the start | 99.9 mm  | 541 mm      | 5.4°                         |
| Weight 1, continuation   | 37.6 mm  | 122 mm      | 2.85°                        |
| Weight 3, continuation   | 37.7 mm  | 104 mm      | 0.71°                        |
| Weight 10, continuation  | 38.4 mm  | 137 mm      | 0.08°                        |

Holding the human range costs about 14 mm of RMS. `the_rom_penalty_keeps_the_ik_in_the_human_range`
runs the weight-3 case. It holds every range within 1° and requires the
penalty to cost no more than 15 mm of RMS.

Three isolated frames (address, top and impact) cannot be fitted this way:
without neighbours to warm-start them they stall at 105 mm RMS before any
penalty.

## Open Work

- The spherical shoulders and hips have no rows yet. Their range is a cone
  plus a twist, not a box.
- Rebuild the pipeline from the range-limited IK, then turn the reference
  check green.
- Add joint limits in the built model (the Simscape joint limit blocks count
  against the 975-block budget).
