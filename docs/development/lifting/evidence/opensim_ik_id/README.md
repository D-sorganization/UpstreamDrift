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

Under OSL-3 (#13041), all five barbell lifts have registered LIFT-4 barbell
bundles in `SAME_INPUT_LIFT_BUNDLES` and complete OpenSim IK and ID receipts.
No lift is recorded as `unavailable`.

## All Five Barbell Lifts Complete (OSL-3, #13041)

The table below summarizes the measured IK and ID metrics across all five
barbell lifts from `receipts.json` (41 frames, stride 5):

| Lift | Bundle SHA-256 | Marker RMS / Max (m) | Best Practice | Reach Miss L / R | Max Range Overshoot (rad) | Pelvis Residual Force (N, vs Hicks) | Pelvis Residual Moment (Nm, vs Hicks) | Hicks Gate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Squat** | `f2fab146...` | 0.0193 / 0.0355 | Pass | 3.81 / 5.69 cm | 8.0e-4 (`arm_flex/rot`, `elbow`, `pro_sup`) | 25.21 (vs 146.59) | 17.43 (vs 18.49) | **Pass** |
| **Deadlift** | `a8483a37...` | 0.0112 / 0.0390 | Pass | 7.11 / 6.91 µm | 6.5e-4 (`pro_sup_l/r`) | 288.23 (vs 110.92) | 147.88 (vs 18.28) | Fail (xfail) |
| **Bench Press** | `f700f098...` | 0.0581 / 0.2164 | Fail | 0.418 / 0.433 m | 4.4e-4 (`elbow_flex_l/r`) | 4598.27 (vs 32.49) | 1575.93 (vs 3.30) | Fail (xfail) |
| **Snatch** | `24c7e8e4...` | 0.0089 / 0.0579 | Fail (bar max) | 7.68 / 7.57 µm | 3.8e-4 (`pro_sup_l/r`) | 463.57 (vs 155.12) | 202.60 (vs 27.10) | Fail (xfail) |
| **Clean & Jerk** | `0b5221ff...` | 0.0198 / 0.1612 | Fail (bar max) | 6.43 / 6.02 µm | 2.3e-3 (`pro_sup_l/r`) | 458.52 (vs 223.10) | 259.03 (vs 38.77) | Fail (xfail) |

### Key Findings and Gate Status

1. **Static-Hold Equilibrium:** Across all five lifts, the contact ground
   reaction equals $(m_{\mathrm{body}} + m_{\mathrm{bar}} + m_{\mathrm{bench}}) \times g$
   to relative precision $< 3\times 10^{-13}$ (80 kg lifter, 120 kg bar;
   plus bench 846.8 N for bench press).
2. **Kinematic Marker Tracking:** Deadlift and back squat pass the best-practice
   marker tracking gate (RMS $\le 0.02$\,m, max $\le 0.04$\,m). Snatch has low RMS
   (0.0089\,m) but exceeds the max gate on the bar marker (0.0579\,m). Clean and
   jerk similarly has low RMS (0.0198\,m) with bar marker max 0.1612\,m.
   Bench press fails marker tracking (RMS 0.0581\,m, max 0.2164\,m) due to the
   supine posture and wide grip.
3. **Arm Reach to the Bar:** Deadlift, snatch, and clean-and-jerk hands reach the
   bundle bar within micrometres ($< 10$\,µm), well within the 1.0\,mm threshold.
   Back squat exhibits the known OSL-19 posterior shoulder workspace bottleneck
   (3.81 / 5.69\,cm). Bench press hands miss by $> 0.4$\,m due to the fixed supine
   trunk and wide grip.
4. **Hicks Residual Gate:** The back squat satisfies the Hicks criteria (forces
   and moments within thresholds on all 41 states). The other four lifts exceed
   the thresholds. These failures are strictly xfailed with measured numbers;
   thresholds are not loosened.

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
tolerance are not widened. Follow-ups: the foot-torsion marker gate is OSL-18
(#13157), the arm reach OSL-19 (#13158), and the open-loop wrist mode that
blocks full-window parity OSL-20 (#13159).

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
