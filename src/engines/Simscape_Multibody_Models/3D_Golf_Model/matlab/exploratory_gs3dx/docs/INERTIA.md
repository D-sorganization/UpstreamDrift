# Segment Inertia: The Cylinder Model Against de Leva

Issue [#10979](https://github.com/D-sorganization/UpstreamDrift/issues/10979),
epic [#10950](https://github.com/D-sorganization/UpstreamDrift/issues/10950).
MATLAB R2025b, measured 2026-09-28 on `GS3DX_FitBalance`.

## What Is Compared

Every body segment of the GS3DX family is one uniform Solid (cylinder,
sphere or brick) with `InertiaType = CalculateFromGeometry` and the de Leva
(1996) male segment mass (`docs/ANTHROPOMETRY.md`). The masses are
anatomical; the moments of inertia and the centres of mass (at the middle
of each solid) are those of the primitive.

- `gs3dx_anthropometry` now carries de Leva's Table 4 radii of gyration
  (`.gyration`, [sagittal transverse longitudinal] as fractions of segment
  length) and reference lengths (`.length`, m).
- `gs3dx_segment_inertia(mass, len, com_fraction, gyration)` gives a
  segment's mass, centre of mass from the proximal end, and principal
  moments `mass * (gyration * len).^2`.
- `gs3dx_inertia_audit(mdl)` evaluates every Solid block of a loaded model
  (43 solids, 80.3929 kg with the club and hand standoffs) and compares each
  body segment with de Leva at the model's own segment length.

Conventions:

- **Longitudinal ratio**: the model moment about the solid's long axis over
  de Leva's longitudinal moment. A cylinder's long axis is its z. The foot
  brick's long axis is its largest dimension, x.
- **Transverse ratio**: the mean of the two other model moments over the
  mean of de Leva's sagittal and transverse moments.
- **Split segments**: forearm halves, and `UpperTorsoBase` + `UpperTorsoTop`,
  are lumped by the parallel-axis theorem. The lumped moments equal a single
  cylinder's to 1e-12.
- **Whole trunk**: lumps the lower and upper trunk along the torso axis,
  although they are joined by the spine joints.
- **Not compared**: the neck (de Leva's head includes it) and the shoulder
  hubs (part of de Leva's upper trunk).

## Result

| Segment     | Mass (kg) | Length (m) | Model I (kg·m²)         | de Leva I (kg·m²)       | Transverse | Longitudinal |
| ----------- | --------- | ---------- | ----------------------- | ----------------------- | ---------- | ------------ |
| head        | 4.719     | 0.2033     | 0.0305 0.0305 0.0305    | 0.0256 0.0276 0.0190    | 1.15       | 1.60         |
| upper arm   | 2.168     | 0.3047     | 0.0178 0.0178 0.00214   | 0.0163 0.0146 0.00502   | 1.15       | 0.43         |
| forearm     | 1.296     | 0.2799     | 0.00910 0.00910 0.00128 | 0.00774 0.00713 0.00149 | 1.22       | 0.86         |
| hand        | 0.488     | 0.0862     | 0.00050 0.00050 0.00050 | 0.00143 0.00095 0.00058 | 0.42       | 0.86         |
| thigh       | 11.328    | 0.4601     | 0.214 0.214 0.0278      | 0.260 0.260 0.0533      | 0.82       | 0.52         |
| shank       | 3.464     | 0.4242     | 0.0541 0.0541 0.00433   | 0.0405 0.0387 0.00661   | 1.37       | 0.65         |
| foot        | 1.096     | 0.2736     | 0.00136 0.00729 0.00775 | 0.00542 0.00493 0.00126 | 1.45       | 1.08         |
| lower trunk | 15.468    | 0.2438     | 0.166 0.166 0.180       | 0.348 0.279 0.317       | 0.53       | 0.57         |
| upper trunk | 15.440    | 0.2438     | 0.166 0.166 0.179       | 0.234 0.094 0.198       | 1.01       | 0.90         |
| whole trunk | 30.908    | 0.4875     | 0.792 0.792 0.359       | 1.017 0.885 0.268       | 0.83       | 1.34         |

Left and right are identical.

## Findings

1. **Limbs spin too easily.** The cylinder radii put too little mass away
   from the long axis:
   - upper arm 0.43 of de Leva;
   - thigh 0.52;
   - shank 0.65.
     Arm and leg rotation about the segment axis (forearm roll, hip internal
     rotation) is therefore under-resisted.
2. **Slender segments bend too hard.** Transverse moments are too large:

   - shank 1.37;
   - forearm 1.22;
   - upper arm 1.15.

   A uniform cylinder keeps its mass out to the distal joint, where a limb
   tapers.

3. **The lower-trunk ratios (0.53 / 0.57) are not a finding.** They apply
   de Leva's lower-trunk radii of gyration, fractions of his pelvis segment
   (about 0.145 m), to the model's 0.244 m `LowerTorso`, which also carries
   half the middle trunk. Setting `LowerTorso` to that reference on
   `GS3DX_Shape` took the whole-trunk longitudinal ratio from 1.34 to 1.85,
   so the trunk is compared as a whole only, and left as it is.
4. **Head and hand** are spheres: the head is 1.6 times too inert about its
   long axis, and the hand has 0.42 of its bending inertia.
5. **The foot is close** (1.08 longitudinal, 1.45 transverse): a brick is a
   fair foot.

The moments are not the only error. The centres of mass also sit at the
middle of each solid rather than at de Leva's fractions (for example, the
thigh at 0.41 of its length from the hip). That shifts the whole-body centre
of mass along the tracked pose. `GS3DX_Shape` (`docs/SHAPE.md`) gives the
limbs and head de Leva moments and centres of mass (every limb ratio 1.00) and
draws the thighs, shanks, hands and head as ellipsoids. Its centre-of-mass
reference is within 1 mm RMS of the cylinder model's (12.4 against 12.9 mm
from the capture's), so the balance reference's error is not a segment
inertia error.

## Tests

`tests/test_gs3dx_inertia.m` (7 tests, 2026-09-28: all pass):

- the gyration and length table;
- `gs3dx_segment_inertia` on hand-computed cases and its input validators;
- the audit reproduces the analytic cylinder for the thigh to 1e-12;
- total mass 80.3929236 kg;
- parallel-axis lumping equals a continuous cylinder;
- every compared segment is present with finite ratios;
- the foot uses its brick's long axis, and the neck and shoulder hubs are
  not compared.
