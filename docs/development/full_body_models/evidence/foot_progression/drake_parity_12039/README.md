# Drake and MuJoCo Address Parity (#12039)

Address-only solves (`--address-only --static-seeds --foot-progression capture`) on ControlTower `ud-sim` for capture-A (reference driver) and capture-B (reference iron).

- `before_*_drake`: `main` @ `afd3340171`.
- `after_*`: the fix @ `37793f8105`.

The MuJoCo reports are byte-identical at both commits, so only `after_*_mujoco` is kept.

**Root cause.** Both plants have identical forward kinematics, and their marker Jacobians match finite differences. The dual-grip weld rotation row did not match:

- **MuJoCo** uses the principal log `log(R_a R_b^T)` with its exact Jacobian `J_l^{-1}(phi) (J_a - R_a R_b^T J_b)`.
- **Drake** used the small-angle `vee(R_a R_b^T - R_b R_a^T) / 2` with the Jacobian `J_a - J_b`. That Jacobian is exact only at zero weld error. At random poses it is up to 0.84 off finite differences, while MuJoCo's is 1e-10 off.

The closure rotation weight defaults to 100 in every pose solve, so each Drake address fit followed a different Levenberg-Marquardt path. Both adapters now share `full_body_ik._closure_rotation_row`.

Address marker RMS, Drake minus MuJoCo:

| Stage              | Driver before | Driver after | 7-iron before | 7-iron after |
| ------------------ | ------------- | ------------ | ------------- | ------------ |
| Static trial       | -0.27 mm      | 0.00 mm      | +0.13 mm      | 0.00 mm      |
| Seed offsets       | -0.84 mm      | 0.00 mm      | -0.87 mm      | 0.00 mm      |
| Calibrated address | +1.64 mm      | 0.00 mm      | -0.54 mm      | +0.02 mm     |

Calibrated address RMS after the fix:

- Driver: MuJoCo 7.13 mm, Drake 7.13 mm.
- 7-iron: MuJoCo 4.78 mm, Drake 4.80 mm.

Drake toe-out error, lead / trail:

- Driver: +0.22 / -0.11 deg (before: +0.24 / -0.34).
- 7-iron: +0.19 / -0.05 deg (before: +0.16 / 0.00).

These now equal MuJoCo's. The hip coordinates agree within 0.1 deg.

This is the calibrated address (inverse kinematics), not a dynamics result. No bound, weight, prior or tolerance changed.

Reproduce:

```bash
python3 -m pytest tests/unit/motion_matching/test_drake_mujoco_ik_parity.py
for E in mujoco drake; do
  python3 -m src.shared.python.motion_matching.pipeline.cli --address-only \
    --spec docs/development/full_body_models/full_body_spec_anthro_driver.json \
    --static-seeds --capture driver --foot-progression capture \
    --engine $E --out runs/driver_$E
done
```
