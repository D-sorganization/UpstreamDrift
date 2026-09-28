# GS3DX_Human: A Human Body Shape, a Square Face and Jointed Feet

Issue [#10979](https://github.com/D-sorganization/UpstreamDrift/issues/10979),
epic [#10950](https://github.com/D-sorganization/UpstreamDrift/issues/10950).
MATLAB R2025b, measured 2026-09-28.

`GS3DX_Human` is `GS3DX_Neck` changed where the address pose looked wrong. It
is built by `gs3dx_build_human(info)`. It compiles to 942 blocks (cap 975),
and its total mass is `GS3DX_Neck`'s.

## What Looked Wrong, and Why

| Complaint                           | Cause found                                                                                                                                                                      |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Torso too large                     | The trunk was three 6 in radius cylinders: 30 cm deep and round. A torso is about 22 cm deep and wider than deep.                                                                |
| Hips about 30° open                 | `ZeroMassHipReference`, a massless 12 in bar, sits 33° off the hip line and off the hips. The hip joint centres are 4.3° open, the capture 3.7°, and the widths match (0.180 m). |
| Hip attachments look asymmetric     | The hip joints are placed from the capture; their midpoint is 3.8 cm forward of the lower-trunk cylinder's axis. Nothing centred on them was drawn.                              |
| Clubface right of target at address | The grip roll: the face normal was 20.7° right of target (the purple rod drawn in the face plane, not along its normal).                                                         |
| Feet backwards in renders           | Renders used the whole-body IK, which has no toe target, so each foot could spin about its shank. The simulation's leg reference sets the feet from the toe markers.             |
| Head too far forward, long neck     | The neck is zero at address, so it continues the upper trunk's axis (59° from vertical at address): the head centre sat 126 mm from the capture's head markers.                  |

## Changes

- **Body.** The trunk, neck, shoulder bars, upper arms, forearms and hand
  standoffs are hidden (`GraphicType 'None'`); their mass and inertia are
  unchanged. A massless ellipsoid is drawn on each one's reference frame.
  The pelvis ellipsoid is centred 4 cm above the hip joint midpoint and lies
  along the hip line. The abdomen lies along the same line, and the chest along
  the shoulder line at address. Ellipsoidal Solid rejects every frame on its
  surface, so the cylinders that carry the joint frames remain as the mass
  carriers.
- **Reference frames.** A cylinder with custom frames shows only those
  (end-centre frames, the arms' and shoulder bars' with turned axes). The
  builder exposes each parent's reference frame (`DoExposeReferenceFrame`)
  and attaches the visual there.
- **Head.** The radii go from 88.9 × 88.9 × 114.3 mm to 90 × 90 × 105 mm
  (graphics only: its inertia is Custom). `NECK_ADDRESS` (−24.2°, −7.0°) turns the
  neck at address to aim it at the capture's head-marker centroid, and is
  composed into `NeckReference`.
- **Club.** `FaceSquareRoll` (−20.73°, model workspace) is added to
  GripStrength's roll about the shaft. The head is a sphere centred on the
  shaft, so no mass moves. The sphere and the face-plane rod are hidden. A
  driver-shaped head (120 × 110 × 60 mm) with 10.5° loft and a red
  face-normal pointer are drawn instead.
- **Feet.** A revolute joint crosses each foot at the ball of the foot, 73% of
  the foot length from the heel and 25 mm above the sole. Its axis runs along
  the foot's left axis. It has a spring (`MidfootStiffness`, 100 N·m/rad)
  and a damper (`MidfootDamping`, 0.5 N·m·s/rad). A positive angle bends the
  toes down; the heel rises over a negative angle. `ForefootMass`
  (0.25 kg) moves to the forefoot. The rearfoot keeps the rest, with its
  centre of mass moved back so the foot's is unchanged at zero angle. The toe
  contact spheres ride on the forefoot at the same points. Both parts are
  drawn as ellipsoids.
- **Blocks.** The "Inertia Sensor" subsystem is removed. It held twelve
  sensors on physical ports that nothing reads, and cost 75 compiled blocks
  (975 → 900). The visuals and the feet bring the total to 942.

## Results

Face normal from the face-normal pointer:

| Frame   | Right of target | Loft  |
| ------- | --------------- | ----- |
| Address | 0.0°            | 10.5° |
| Impact  | 14.2°           | 5.1°  |

Before the roll, the face was 20.7° open at address and about 35° open at impact.
The remaining open face at impact comes from the posed hands, not from the grip.

Head centre against the capture's head-marker centroid, to impact, every
third frame plus the top (`gs3dx_render` on `gs3dx_reference_pose`, neck
angles from each model's `NeckReference`):

| Model         | At address (mm) | 3D RMS / max (mm) | Vertical RMS (mm) | Vertical travel error RMS / max (mm) |
| ------------- | --------------- | ----------------- | ----------------- | ------------------------------------ |
| `GS3DX_Neck`  | 126             | 99.8 / 165.2      | 90.5              | 37.0 / 62.2                          |
| `GS3DX_Human` | 40              | 83.9 / 125.4      | 39.7              | 19.1 / 42.1                          |

![GS3DX_Human face-on at address](screenshots/GS3DX_Human_fo_addr.png)
![GS3DX_Human down the line at address](screenshots/GS3DX_Human_dtl_addr.png)

## Rendering in the Simulated Pose

`gs3dx_reference_pose(ik, ref, neck=NeckReference)` returns the IK with the
pelvis, hips, knees and ankles taken from the leg reference the servos track,
and the neck from the model's reference. At address both feet then point at
the ball, 22° flared on the lead side and 9° on the trail. Use it for renders
of any variant: rows are named by `gs3dx_joint_keys`, and joints it lacks,
such as the midfoot joints, are drawn at zero.

## Tests

`tests/test_gs3dx_human.m` (5 tests):

- every visible solid off the club is an ellipsoid, except the 1 cm contact
  spheres;
- the total mass is unchanged, and each forefoot takes `ForefootMass` from its foot;
- the sensor subsystem is gone and the model compiles within the reserve;
- the midfoot joints are sprung revolutes carrying the toe contact points;
- at the same joint angles, every solid both models draw has its
  `GS3DX_Neck` position, and the face is square with loft showing.
