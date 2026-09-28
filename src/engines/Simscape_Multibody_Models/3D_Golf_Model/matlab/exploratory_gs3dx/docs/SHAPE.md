# GS3DX_Shape: De Leva Segment Inertia and Ellipsoid Limbs

Issue [#10979](https://github.com/D-sorganization/UpstreamDrift/issues/10979),
epic [#10950](https://github.com/D-sorganization/UpstreamDrift/issues/10950).
MATLAB R2025b, measured 2026-09-28.

`GS3DX_Shape` is `GS3DX_FitBalance` with the segment inertia of de Leva
(1996) on the limbs and head, and the thighs, shanks, hands and head drawn as
ellipsoids. It is built by `gs3dx_build_shape(info, ref)`, and
`docs/INERTIA.md` gives the cylinder model it replaces.

## Where the Joints Sit

A geometry probe of `GS3DX_FitBalance` measured every joint frame in the
reference frame of the solids beside it, at address, the top and impact
(`KinematicsSolver` translations from World, rotated into the solid frame).
Every value was constant to 7e-16 m (the right hand to 2e-11 m):

| Solid                  | Proximal joint (z, m)   | Distal joint (z, m) |
| ---------------------- | ----------------------- | ------------------- |
| thigh                  | hip +0.2301             | knee −0.2301        |
| shank                  | knee +0.2121            | ankle −0.2121       |
| upper arm              | shoulder +0.1523        | elbow −0.1523       |
| upper forearm half     | elbow +0.0700           | wrist −0.2099       |
| lower forearm half     | elbow +0.2099           | wrist −0.0700       |
| left hand / right hand | wrist −0.0372 / +0.0372 |                     |

Each limb solid is centred on its segment, with both joints on its z axis
and the proximal one at +z. A centre of mass at de Leva's fraction `c` of
length `L` from the proximal joint is therefore at `z = L/2 − c L`, and the
thighs, shanks and hands carry no frame but R, so a different solid shape
leaves the kinematics unchanged.

## What Changes

| Solid             | Mass   | Centre of mass (z, m) | Moments (kg·m²)            | Drawn as  |
| ----------------- | ------ | --------------------- | -------------------------- | --------- |
| thigh             | 11.328 | +0.0416               | 0.2596 0.2596 0.0532       | ellipsoid |
| shank             | 3.464  | +0.0230               | 0.0396 0.0396 0.00661      | ellipsoid |
| upper arm         | 2.168  | −0.0235               | 0.0155 0.0155 0.00502      | cylinder  |
| forearm half (×2) | 0.648  | +0.0119               | 0.000543 0.000543 0.000743 | cylinder  |
| hand              | 0.488  | 0 (kept)              | 0.00119 0.00119 0.000583   | ellipsoid |
| head              | 4.719  | 0 (kept)              | 0.0266 0.0266 0.0190       | ellipsoid |

- **Moments.** De Leva's radii of gyration at the model's segment lengths,
  with the sagittal and transverse moments averaged so each segment is
  axisymmetric about z. The two forearm halves each carry half the forearm,
  with centres of mass a half forearm apart; they lump to de Leva's forearm.
  `gs3dx_inertia_audit` gives 1.00 for every limb and the head, transverse
  and longitudinal (the cylinders were 0.42 to 1.60).
- **Kept.** Masses (80.3929236 kg in total), all frames, the feet and the
  trunk. A hand's de Leva centre of mass lies about 3 cm further from the
  wrist, which moves the whole-body centre of mass by 0.2 mm, so the hands
  and head keep theirs.
- **Ellipsoids.** A thigh or shank ellipsoid has the cylinder's radius and
  reaches 5 % past each joint. A hand is 0.8 × 0.65 × 1.25 of its 2 in
  sphere, and the head 0.7 × 0.7 × 0.9 of its 5 in sphere, long along the
  neck. The swap keeps the block's name, position, colour and every port on
  its net: `Grip/RHand` has eight, which only `PortConnectivity` lists.
- **Blocks.** 973 compiled, as `GS3DX_FitBalance`. An extra visual-only solid
  compiles to 7 blocks (973 → 980), and `GS3DX_CONTACT_CHECK` needs 10 of
  the 27 left, so frame-bearing solids (upper arms, forearm halves, trunk)
  keep their cylinders.

![GS3DX_Shape down the line at the top](screenshots/GS3DX_Shape_dtl_top.png)

Stills at address, the top and impact, face-on and down the line, are in
`docs/screenshots/GS3DX_Shape_*.png` and, for comparison,
`GS3DX_FitBalance_*.png` (`docs/RENDERING.md`).

## Balance

The centre of mass moves with the de Leva inertia, so `GS3DX_Shape` needs its
own centre-of-mass offset. It was measured the same way as for
`GS3DX_FitBalance`: a balance-off run to impact, then
`gs3dx_balance_com_offset`. Runs of `gs3dx_contact_check` to impact
(1.319 s), about 18 min each:

| Reference                                | Support (BW) | Peak (BW) at | Pelvis RMS / end (mm) | COM horizontal RMS / max (mm) | Slip L/R (mm) |
| ---------------------------------------- | ------------ | ------------ | --------------------- | ----------------------------- | ------------- |
| balance off                              | 0.28 – 1.47  | 1.47, 1.319  | 182 / 430             | —                             | 67 / 98       |
| Shape offset                             | 0.008 – 2.64 | 2.64, 1.315  | 25 / 66               | 16.8 / 23.4                   | 54 / 78       |
| capture centre of mass                   | 0.36 – 1.83  | 1.44, 1.319  | 42 / 69               | 13.2 / 34.5                   | 44 / 53       |
| `GS3DX_FitBalance` offset (FIT.md, §7)   | 0.056 – 2.63 | 2.63, 1.312  | 27 / 73               | 17.9 / 25.7                   | 58 / 88       |
| `GS3DX_FitBalance`, capture (FIT.md, §7) | 0.36 – 1.81  | 1.44, 1.319  | 42 / 72               | 14.1 / 36.1                   | 44 / 53       |

The references themselves, before impact:

| Reference          | Vertical range (mm) | Against the capture: RMS / max (mm) |
| ------------------ | ------------------- | ----------------------------------- |
| `GS3DX_FitBalance` | 53.4                | 12.9 / 23.7                         |
| `GS3DX_Shape`      | 51.8                | 12.4 / 23.6                         |
| capture            | 49.3                | —                                   |

**Segment inertia does not explain the impact spike.** De Leva's centres of
mass move the model's reference by less than 1 mm RMS against the capture, and
both balance runs repeat `GS3DX_FitBalance` to within 0.01 BW. The 12 mm RMS
that remains, and the 2.6 BW peak it drives, come from the tracked pose
itself (segment lengths, the trunk's mass split, the IK), not from how mass is
distributed along the limbs. The saved model carries the Shape offset
reference.

## Tests

`tests/test_gs3dx_shape.m` (6 tests):

- a centre-of-mass offset and path are not both accepted;
- masses are unchanged, 80.3929236 kg in total;
- every limb and the head audit at 1.00 against de Leva to 1e-9, and the
  feet and trunk are unchanged;
- the thigh and shank centres of mass sit at de Leva's fractions, and the
  forearm halves lump to de Leva's forearm centre, to 1e-12 m;
- the thighs, shanks, hands and head are connected Ellipsoidal Solids with
  custom inertia;
- every solid has the pose it has in `GS3DX_FitBalance` at address and
  impact to 1e-12, each drawn ellipsoid is its radii to 1e-9 m, and the head's
  long axis (z, shared with the upper trunk) runs along the neck to the head.
