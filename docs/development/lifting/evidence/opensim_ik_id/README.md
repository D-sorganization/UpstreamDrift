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
| **Squat** | `0179c5a2...` | 0.0164 / 0.0304 | Pass | 2.53 µm / 1.94 mm | none (0 rad) | 23.82 (vs 146.93) | 11.75 (vs 18.64) | **Pass** |
| **Deadlift** | `a8483a37...` | 0.0112 / 0.0391 | Pass | 7.10 µm / 6.90 µm | 6.1e-04 (`pro_sup_l`, `pro_sup_r`) | 288.29 (vs 110.92) | 148.02 (vs 18.28) | Fail (xfail) |
| **Bench Press** | `f700f098...` | 0.0160 / 0.0556 | Fail (bar max) | 12.49 µm / 6.67 mm | 1.4e-03 (`pro_sup_l`, `pro_sup_r`) | 511.58 (vs 221.85) | 318.32 (vs 23.54) | Fail (xfail) |
| **Snatch** | `24c7e8e4...` | 0.0089 / 0.0579 | Fail (bar max) | 7.68 µm / 7.57 µm | 3.8e-04 (`pro_sup_l`, `pro_sup_r`) | 463.21 (vs 155.12) | 202.62 (vs 27.10) | Fail (xfail) |
| **Clean & Jerk** | `0b5221ff...` | 0.0198 / 0.1612 | Fail (bar max) | 6.43 µm / 6.01 µm | 2.2e-03 (`pro_sup_l`, `pro_sup_r`) | 458.56 (vs 223.10) | 259.00 (vs 38.77) | Fail (xfail) |

### Key Findings and Gate Status

1. **Static-Hold Equilibrium:** Across all five lifts, the contact ground
   reaction equals $(m_{\mathrm{body}} + m_{\mathrm{bar}} + m_{\mathrm{bench}}) \times g$
   to relative precision $< 3\times 10^{-13}$ (80 kg lifter, 120 kg bar;
   plus bench 846.8 N for bench press).
2. **Kinematic Marker Tracking:** Deadlift and back squat pass the best-practice
   marker tracking gate (RMS $\le 0.02$\,m, max $\le 0.04$\,m; back squat achieves
   0.0164\,m RMS and 0.0304\,m max). Snatch has low RMS (0.0089\,m) but exceeds the max
   gate on the bar marker (0.0579\,m). Clean and jerk similarly has low RMS
   (0.0198\,m) with bar marker max 0.1612\,m. Bench press achieves low marker RMS
   (0.0160\,m) below the 0.02\,m threshold, exceeding the max gate on the bar marker
   (0.0556\,m vs 0.04\,m).
3. **Arm Reach to the Bar:** Deadlift, snatch, and clean-and-jerk hands reach the
   bundle bar within micrometres ($< 10$\,µm), well within the 1.0\,mm threshold.
   Bench press reaches within 12.49\,µm (left) and 6.67\,mm (right) via flexed
   elbow arm seeds. Back squat hands reach within 2.53\,µm (left) and 1.94\,mm (right).
4. **Hicks Residual Gate:** The back squat satisfies the Hicks criteria (forces
   and moments within thresholds on all 41 states: peak force 23.82\,N vs 146.93\,N limit,
   peak moment 11.75\,N$\cdot$m vs 18.64\,N$\cdot$m limit). The other four lifts exceed
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

### Squat Coordinate Range Closeout (OSL-12, #13048)

Under OSL-12 (#13048), the back squat achieves full satisfaction of all four acceptance criteria:
1. **Marker Best Practice:** RMS 0.0164 m $\le$ 0.02 m and max 0.0304 m $\le$ 0.04 m (`within_best_practice: True`).
2. **Hicks Residual Gate:** Peak pelvis residual force 23.82 N $\le$ 146.93 N limit (`forces_within: True`); peak pelvis residual moment 11.75 N$\cdot$m $\le$ 18.64 N$\cdot$m limit (`moments_within: True`).
3. **Contact Penetration:** Max sphere penetration 7.32 mm $\le$ 8.00 mm limit (`within_limit: True`).
4. **Coordinate Ranges:** Every coordinate remains strictly within its model range (`range_violations == {}`, zero rad overshoot).

The root cause of coordinate overshoot on main was twofold:
1. Arm coordinates were left unclamped during IK (`_UNCLAMPED_PREFIXES` in `msk_lift_ik_id.py`) because clamping them directly causes OpenSim's `AssemblySolver::assemble()` to fail with Ipopt infeasibility (`status 2`) under the two-hand welded barbell constraint (attempt 2a, logged).
2. The bounded least-squares reach solve placed the arm seed directly on the model range limits (`arm_flex_r = 1.5707963 rad`), allowing weak coordinate tasks in unconstrained IK to drift just past the bound by ~4.2e-5 rad.

The resolution (attempt 2b) applies an interior bound margin:
- `ARM_REACH_BOUND_MARGIN_RAD = 1e-3` rad in `msk_lift_arm_reach.py` restricts the reach solve bounds to $[q_{\min} + 10^{-3}, q_{\max} - 10^{-3}]$, preventing seeds from sitting on range limits.
- `ARM_TASK_BOUND_MARGIN_RAD = 1e-3` rad in `msk_lift_ik_id.py` clips coordinate task target values inside the same interior bounds.

With this margin, all 41 states of the back squat stay strictly inside their model ranges (`range_violations == {}`), with no adverse impact on reach quality (left 2.53 µm, right 1.94 mm) or solver convergence.

These are software receipts, not physical acceptance. See
`docs/research/lift_models/lift_models.tex` (OSL-3, OSL-12, OSL-14 and "Hands That Reach the Bundle Bar"
subsections)
for the method, the numbers and the limitations.

Reproduce from the repository root:

```bash
MUJOCO_GL=egl python3 scripts/measure_opensim_lift_ik_id.py
```
