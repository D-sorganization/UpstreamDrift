# Hip Rewrite Pelvis Alignment: Address Leg-Chain Yaw Budgets (#12109)

MuJoCo calibrated address, `--static-seeds --foot-progression capture`, both
tour captures (capture-A driver, capture-B 7-iron). Each `leg_yaw_budget.json`
is written by `scripts/diagnose_address_leg_yaw.py` and run on brick.

- `before_*`: commit `bee92b09ff`. The hip rewrite takes its alignment from
  `build_receipt_v2.json`.
- `after_*`: commit `6496647cde`. The hip rewrite uses
  `hip_calibration.pelvis_alignment_from_spec` on the spec being rewritten.

| Quantity (right / left)                                  | Driver before | Driver after | 7-iron before | 7-iron after |
| -------------------------------------------------------- | ------------- | ------------ | ------------- | ------------ |
| Femur forward vs pelvis forward, hip coordinates 0 (deg) | 23.7 / 64.4   | -5.7 / 25.5  | 29.1 / 65.5   | -1.5 / 26.4  |
| `hip_rotation` at address (deg)                          | -40.0 / 35.7  | -17.2 / 9.9  | -40.0 / 39.2  | -23.8 / 17.6 |
| `hip_adduction` at address (deg)                         | -39.6 / 18.6  | -6.5 / 0.8   | -31.6 / 22.7  | -4.0 / 7.7   |
| Toe-out error, trail / lead (deg)                        | -2.56 / 0.00  | -0.11 / 0.22 | -4.45 / 0.00  | -0.05 / 0.19 |
| Address marker RMS (mm)                                  | 7.38          | 7.13         | 7.29          | 4.85         |

All four runs are from the same host. The same pre-fix driver command on
ControlTower (`ud-sim`) gave -2.07 deg and 7.45 mm, which matches the #12057
evidence; the difference is host-to-host solver variation.

After the fix, the femur yaw at zero equals the measured hip zero-twist
(`hip_zero_twist_deg`), so no rotation other than the zero-twist correction
remains. Both feet are within the 2 deg OSV-6 target on both captures, and no
hip coordinate is on its +-40 deg range limit. The range was not widened and
no tolerance was changed.

The capture's own toe-out estimators (`toe_out_estimators_deg`) disagree on
the trail foot by 5.1 deg (driver) and 4.0 deg (7-iron) between the
malleolus-corrected ankle-to-toe axis and the forefoot normal. The 2 deg
criterion is therefore tighter than the definitional spread of the marker
estimate. The binding target stays the corrected axis.

This is the calibrated address only (inverse kinematics). It is not a
dynamics or open-loop replay result; see the regenerated ground-support
receipts for the full pipeline.

Reproduce:

```bash
python3 -m scripts.diagnose_address_leg_yaw --capture driver \
  --spec docs/development/full_body_models/full_body_spec_anthro_driver.json \
  --out runs/legyaw_driver
python3 -m scripts.diagnose_address_leg_yaw --capture iron \
  --spec docs/development/full_body_models/full_body_spec_anthro_iron7.json \
  --out runs/legyaw_iron
```
