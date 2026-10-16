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

These are software receipts, not physical acceptance. See
`docs/research/lift_models/lift_models.tex` (OSL-3 and OSL-12 subsections)
for the method, the numbers and the limitations.

Reproduce from the repository root:

```bash
MUJOCO_GL=egl python3 scripts/measure_opensim_lift_ik_id.py
```
