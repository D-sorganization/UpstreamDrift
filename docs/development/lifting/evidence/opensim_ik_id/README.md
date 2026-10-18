# OpenSim IK and Inverse Dynamics Receipts (OSL-3 #13041, OSL-12 #13048)

`receipts.json` is written by `scripts/measure_opensim_lift_ik_id.py` with
real OpenSim 4.6 and the Rajagopal model. For each barbell lift it holds:

- `static_hold`: the contact-derived ground reaction of the balanced hold
  pose at static equilibrium (all five lifts). It records the per-sphere
  penetration against the 8 mm limit, the centre of mass, each support's
  centre of pressure, and the pelvis residual left at equilibrium.
- `ik_id`: the IK and inverse-dynamics receipt against the LIFT-5 same-input
  bundle. This includes marker and segment-angle errors, coordinate-range
  violations, and the virtual ASIS/PSIS pelvis landmark receipt. It also has
  the pelvis residuals with the Hicks et al. (2015) gate, peak joint moments
  and the bundle SHA-256.

Only the back squat has a bundle. The other four lifts are recorded
`unavailable` with the reason, never as zero.

OSL-12 changes the following:

- The contact stiffness is now 1e7 Pa, from OpenSim_Models PR #476.
- Standing holds are balanced by an ankle lean.
- The pelvis is scaled to the bundle hip width.
- Coordinate clamping is on, except for the arms.

Every hold now sits within 8 mm of penetration, with zero pelvis residual.
Two squat results remain below acceptance:

- The squat IK still fails the marker best-practice gate, because the shoulder
  and forefoot geometry of the golf body does not match Rajagopal.
- The squat ID still fails the residual gate.

OSL-14 (#13057) moves the bundles onto the AN-1 lift body. That body has a
lumbar joint, de Leva hips and cited-range IK bounds. OSL-14 also scales the
OpenSim torso to the bundle shoulder width.

The committed squat receipt is measured on the LIFT-4 barbell re-track
(`back_squat_barbell_bundle.npz`). The re-track was regenerated with the
#13072 rigid-bar controller. The arms now stay on the bar, so no coordinate
leaves its range: `range_violations` is empty, where it used to list six arm
coordinates.

The OSL-12 bar-reach fix (#13048) seeds the hold arms on the bundle's grips.
The IK first runs on the hold-pose model with the bar unweighted. A bounded
solve then puts each hand centre on its bundle grip, and the model is rebuilt
with those arms (`msk_lift_arm_reach.reaching_lift_build`). The IK receipt
records this in `arm_reach`. The bar marker is tracked at weight 1 again.

- Marker RMS falls from 6.81 cm to 1.18 cm and the maximum from 27.56 cm to
  2.32 cm, which meets the 2 cm / 4 cm gate. The bar falls from 27.56 cm to
  0.29 mm.
- Every coordinate stays in range, and the pelvis landmarks are tracked.
- The Hicks residual gate still fails. The peak pelvis force is 214.2 N
  against a 167.2 N limit, and the peak pelvis moment is 153.8 N m (tilt;
  546.4 N m before) against a 21.9 N m limit. Both peaks fall at the first
  state of the window, where the bundle is still settling onto its loaded
  foot contact. Over states 1-40 the peaks are 24.4 N and 20.7 N m.

Before the OSL-17 rack, on the whole-row lift body (#13037) the squat
receipt read: marker RMS 1.74 cm and maximum 3.70 cm (right forefoot), inside
the 2 cm / 4 cm gate; segment-angle RMS 1.47 deg. The reach solve ended at the
150 deg `elbow_flex` bound on both sides, 8.3 mm (left) and 21.5 mm (right)
from the grips (bar marker 12.56 mm off), with `elbow_flex` 1.6e-4 rad and
`pro_sup` -1.9e-4 rad past their ranges. The bar target was then the parity
standard's start pose scaled by height (1.618 m), and the longer whole-row
trunk raised the shoulders 7.1 cm, so the shoulder-to-grip distance fell from
0.253/0.235 m to 0.190/0.175 m.

The committed receipt is the OSL-17 high-bar rack on the whole-row body
(#13041): marker RMS 2.45 cm and maximum 3.81 cm, so the 2 cm RMS gate now
fails (it was 1.94 cm on the earlier body). The foot markers carry the
increase (forefoot 3.8 cm): the 10 deg tibial torsion (#12814) turns the
bundle feet out from 8.8 to 17.9 deg, which the torsion-free Rajagopal foot
cannot follow. The gate is not loosened. The Hicks gate passes on all 41
states (146.6 N, 18.5 N m). The reach misses the high-bar grips by 3.81/5.69
cm (4.63/6.03 cm on the earlier body) and the arm coordinates go up to 8.0e-4
rad past their ranges (strict xfails). Both reach misses have one cause: the
folded Rajagopal arm cannot bring the hand within about 0.20 m of its
shoulder landmark in these directions; shoulder-to-grip distance plus reach
miss is 0.198-0.213 m in all four measured cases, while the grips sit
0.146-0.190 m from the shoulders. The elbow and arm ranges and the reach
tolerance are not widened.

A diagnostic fit of the bodyweight squat bundle tracked the pelvis to 3 mm.
Excluding the bar, its markers came to 1.78 cm RMS and 4.15 cm maximum. See
`lift_models.tex`, OSL-14 subsection.

These are software receipts, not physical acceptance. See
`docs/research/lift_models/lift_models.tex` (OSL-3, OSL-12, OSL-14 and "Hands That Reach the Bundle Bar"
subsections)
for the method, the numbers and the limitations.

Reproduce from the repository root:

```bash
MUJOCO_GL=egl python3 scripts/measure_opensim_lift_ik_id.py
```
