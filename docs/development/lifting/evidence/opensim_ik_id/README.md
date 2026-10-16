# OpenSim IK and Inverse Dynamics Receipts (OSL-3, #13041)

`receipts.json` is written by `scripts/measure_opensim_lift_ik_id.py` with
real OpenSim 4.6 and the Rajagopal model. For each barbell lift it holds:

- `static_hold`: the contact-derived ground reaction at the OSL-2 hold pose
  (all five lifts).
- `ik_id`: the IK and inverse-dynamics receipt against the LIFT-5 same-input
  bundle. This includes marker and segment-angle errors, pelvis residuals with
  the Hicks et al. (2015) gate, peak joint moments and the bundle SHA-256.

Only the back squat has a bundle. The other four lifts are recorded
`unavailable` with the reason, never as zero.

The squat IK meets the segment-angle threshold but fails the marker
best-practice gate. The squat ID fails the residual gate, because pelvis
orientation is weakly observed. These are software receipts, not physical
acceptance. See `docs/research/lift_models/lift_models.tex` (OSL-3 subsection)
for the method, the numbers and the limitations.

Reproduce from the repository root:

```bash
MUJOCO_GL=egl python3 scripts/measure_opensim_lift_ik_id.py
```
