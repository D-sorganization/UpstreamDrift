# Foot Contact Geometry and Torsional Friction

Issue #11671, epic #11667. Sources: `contact_law.torsional_friction_moment`, `MuJoCo adapter generalized_forces` (`src/engines/physics_engines/mujoco/python/full_body_model.py`), `pipeline/lane.py` (`add_foot_width_spheres`, `add_torsional_friction`, `expand_stance_for_width`). Opt-in CLI flags `--foot-half-width-m` and `--torsional-patch-m`.

## Problem

Each foot is three centre-line spheres (heel, forefoot, toe): a line contact. The contact law is a point law, so a sphere carries no moment about the ground normal, and the only torsional capacity of a foot is the lever of tangential forces at points on one line. Under the finish the feet pivot and slide (driver 502 mm, pelvis yaw error 34.5 degrees).

## Models

**Torsional friction.** At every loaded contact sphere the adapter adds a moment about the ground normal $n$

$$M = -\mu_d\, f_n\, r\, \tanh(\omega_n / \omega_0)\; n,\qquad \omega_n = n\cdot\omega_{site}$$

with $\mu_d$ the dynamic friction coefficient of the contact law (0.8), $f_n$ the sphere normal force, $r$ the patch radius (`contact.torsion.patch_radius_m`) and $\omega_0 = 0.5$ rad/s. The moment is dissipative (it opposes the spin), bounded by $\mu_d f_n r$, and is zero without load. It enters `xfrc_applied` and the generalized contact force through the site rotational Jacobian. Only the MuJoCo adapter applies it; the Drake and Pinocchio adapters ignore the `torsion` key. It changes no geometry, so the fitted pre-impact trajectory is unchanged (receipts below: identical to 0.1 mm).

**Foot width.** `add_foot_width_spheres` adds a lateral pair at $\pm w$ along the calcaneus z axis for the heel and forefoot (named `<stem>_zp_<side>`, `<stem>_zn_<side>`). `expand_stance_for_width` pins them in the IK with their centre-line sphere.

Coordinates and units: SI, OpenSim calcaneus frame (x forward, y up, z lateral), ground normal z.

## Calibration and Validation

The torsion patch radius was tuned on the driver only and validated on the 7-iron. Replays of the saved baseline reference (same IK, `--static-seeds`, iron with `--zmp-filter`), simulation from address. Marker RMS is against the capture; finish is $t \ge 1.0$ s.

| Club | Patch | Marker RMS all / finish | Pre-impact (0-1 s) | Foot slide | Pelvis yaw error max / final | ZMP inside 1.0-1.5 s / finish |
| ---- | ----- | ----------------------- | ------------------ | ---------- | ---------------------------- | ----------------------------- |
| Driver | none | 83.3 / 121.0 mm | 25.6 mm | 502 mm | 34.5 / 34.5 deg | 1.00 / 0.88 |
| Driver | 0.03 m | 71.0 / 102.2 mm | 25.6 mm | 300 mm | 11.6 / 11.6 deg | 1.00 / 0.95 |
| Driver | 0.04 m | 68.7 / 98.5 mm | 25.6 mm | 250 mm | 8.7 / 6.1 deg | 1.00 / 0.93 |
| Driver | **0.05 m** | **67.3 / 96.3 mm** | 25.6 mm | 210 mm | 9.0 / 1.9 deg | 0.99 / 0.93 |
| Driver | 0.06 m | 66.6 / 95.3 mm | 25.6 mm | 177 mm | 9.3 / -1.6 deg | 0.99 / 0.95 |
| Driver | 0.08 m | 66.9 / 95.8 mm | 25.6 mm | 125 mm | 11.4 / -7.1 deg | 0.99 / 0.92 |
| Driver | 0.12 m | 70.1 / 100.9 mm | 25.6 mm | 62 mm | 17.5 / -13.6 deg | 0.98 / 0.93 |
| 7-iron | none | 57.9 / 81.7 mm | 23.7 mm | 340 mm | 19.4 / 19.4 deg | 1.00 / 0.99 |
| 7-iron | **0.05 m** (validation) | **54.3 / 76.2 mm** | 23.7 mm | 93 mm | 14.5 / -9.9 deg | 1.00 / 0.99 |
| 7-iron | 0.03 m (not selected) | 52.8 / 73.8 mm | 23.7 mm | 181 mm | 13.4 / 0.3 deg | 1.00 / 0.99 |

The chosen value 0.05 m is the middle of the driver's flat optimum (0.04 to 0.08 m) and was fixed before the iron was evaluated. The full pipeline with `--torsional-patch-m 0.05` reproduces these two rows (driver 67.25 / 96.33 mm, iron 54.34 / 76.15 mm; pre-impact root error 19.9 and 21.7 mm, baseline 20.0 and 21.8 mm).

## Foot Width Did Not Help

Re-running the full pipeline (IK, calibration, scaling, replay) with lateral spheres:

| Run | Marker RMS all | Pre-impact root error | Slide | Yaw error max |
| --- | -------------- | --------------------- | ----- | ------------- |
| Driver, 0.03 m, spheres not pinned | 90.7 mm | 15.1 mm | 630 mm | 41.3 deg |
| Driver, 0.02 m, pinned | 106.6 mm | 14.2 mm | 359 mm | 25.6 deg |
| Driver, 0.03 m, pinned | 84.0 mm | 20.1 mm | 497 mm | 22.8 deg |
| 7-iron, 0.03 m, spheres not pinned | 187.1 mm | 123.5 mm | 527 mm | 58.0 deg |
| 7-iron, 0.02 m, pinned | 94.5 mm | 24.1 mm | 271 mm | 11.2 deg |
| 7-iron, 0.03 m, pinned | 133.1 mm | 123.8 mm | 280 mm | 28.2 deg |

No width is better than the baseline on both clubs; the iron at 0.03 m loses the pre-impact root fit (124 mm). Width therefore stays an opt-in experiment. An earlier replay-only trial (spheres added without re-IK) broke the 0 to 1 s root fit by 34 mm, which is why the IK pins the lateral spheres.

## Limitations and Status

- The pelvis yaw error is below 8 degrees on neither club at its maximum (driver 9.0, iron 14.5 degrees), so the epic's yaw target is not met; marker RMS, foot slide and the ZMP-inside fraction improve on both.
- The iron's peak error is earlier than its final error, and the final value overshoots to the other sign; the patch radius trades the two.
- The torsion patch is a calibrated lumped parameter, not a measured shoe-ground property. It is validated on one iron and one driver capture of one subject.
- Nothing here is scientific qualification; it is software correctness plus a simulation-level fit.

## Reproduction

```bash
python3 -m src.shared.python.motion_matching.pipeline.cli \
  --spec docs/development/full_body_models/full_body_spec_anthro_driver.json \
  --skip-hip-calibration --static-seeds --capture driver --torsional-patch-m 0.05 --out <dir>
python3 -m src.shared.python.motion_matching.pipeline.cli \
  --spec docs/development/full_body_models/full_body_spec_anthro_iron7.json \
  --skip-hip-calibration --static-seeds --zmp-filter --capture iron \
  --torsional-patch-m 0.05 --out <dir>
```
