# Neutral Hip Zero-Twist: Pelvis Turn at Impact (#12042 Slice 3)

MuJoCo calibrated runs with `--hip-zero-twist neutral` on both tour captures
(capture-A driver, capture-B 7-iron). They ran on ControlTower (`ud-sim`) at
`7eb87c7c5c`. Each directory holds the run's `receipt.json` and its final
scaled specification. The canonical receipts in
`../ground_support/anthro_driver_seeds/` and
`../ground_support/anthro_iron_seeds_zmp/` keep the default `capture` mode and
are unchanged. Same-commit runs in that mode reproduce them exactly.

Method, mechanism and limitations: `docs/research/hip_axis_mirroring/hip_axis_mirroring.tex`,
section "Neutral Hip Zero-Twist and the Pelvis at Impact (#12042)".

## Results

Turn errors are model minus capture markers, in degrees (top / impact). FD is
the whole-swing marker RMS, with address-to-impact / after-impact in brackets.

| Run | Pelvis | Upper Trunk | Shoulder Girdle | Lead `hip_rotation` at Impact | Address / IK (mm) | FD (mm) |
| --- | --- | --- | --- | --- | --- | --- |
| Driver, `capture` (canonical) | -0.5 / +9.7 | +18.3 / +4.0 | +14.6 / -5.5 | 40.0 (bound) | 6.3 / 29.5 | 80.8 (34.2 / 145.0) |
| Driver, `neutral` | +3.2 / +2.5 | +17.7 / +5.6 | +12.0 / -9.0 | 27.0 | 5.3 / 25.6 | 51.2 (31.6 / 83.8) |
| 7-iron, `capture` (canonical) | +2.7 / +7.2 | +13.6 / +6.1 | +10.8 / +3.3 | 40.0 (bound) | 6.6 / 26.8 | 54.5 (30.1 / 92.1) |
| 7-iron, `neutral` | +1.4 / +2.8 | +13.9 / +5.2 | +11.0 / +8.3 | 27.7 | 2.1 / 22.9 | 91.2 (34.0 / 165.3) |

- **Pelvis:** with `neutral` it is within 3.2 deg of the markers at top and
  impact on both captures, inside the 5 deg #12042 bound.
- **Lead hip:** it leaves its +40 deg bound at impact.
- **Lead-leg IK residual:** 36 to 15 mm (driver) and 40 to 16 mm (7-iron).
- **Range of motion:** neither run has a range-of-motion flag.

## Limitations

- **Mixed dynamics:** the 7-iron after-impact FD gets worse (92 to 165 mm),
  the follow-through rollout sensitivity documented on #12042. Its
  address-to-impact FD rises 3.9 mm, near the about 3 mm noise floor.
- **Trail hip:** `hip_rotation_r` now reaches its -40 deg bound late in the
  downswing.
- **Upper trunk and shoulder girdle:** the top-of-backswing over-twist is
  unchanged (#12042 slice 7).
- **Kinematics only:** this is inverse kinematics plus feedback-assisted
  tracking. It is not an open-loop dynamics result.

Switching the default to `neutral` would regenerate the canonical receipts
and the fixtures built on them. That decision is the owner's.

## Reproduce

```bash
python3 -m src.shared.python.motion_matching.pipeline.cli \
  --spec docs/development/full_body_models/full_body_spec_anthro_driver.json \
  --capture driver --static-seeds --hip-zero-twist neutral --out runs/hzn_driver
python3 -m src.shared.python.motion_matching.pipeline.cli \
  --spec docs/development/full_body_models/full_body_spec_anthro_iron7.json \
  --capture iron --static-seeds --zmp-filter --hip-zero-twist neutral \
  --out runs/hzn_iron
```
