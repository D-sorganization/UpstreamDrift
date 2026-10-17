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
(`back_squat_barbell_bundle.npz`), which diverges from its source in the golf
body and the lift body alike. That receipt therefore fails the acceptance
gates: marker RMS 5.04 cm, maximum 29.7 cm, and the pelvis is not tracked.

A diagnostic fit of the bodyweight squat bundle tracked the pelvis to 3 mm.
Excluding the bar, its markers came to 1.78 cm RMS and 4.15 cm maximum. See
`lift_models.tex`, OSL-14 subsection.

These are software receipts, not physical acceptance. See
`docs/research/lift_models/lift_models.tex` (OSL-3, OSL-12 and OSL-14 subsections)
for the method, the numbers and the limitations.

Reproduce from the repository root:

```bash
MUJOCO_GL=egl python3 scripts/measure_opensim_lift_ik_id.py
```
