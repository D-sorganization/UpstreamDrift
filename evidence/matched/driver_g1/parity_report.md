# Unified Cross-Engine Parity Report

- **Candidate:** `candidate`
- **Candidate SHA:** `1da2a4776534181b`
- **Reference Engine:** `pinocchio`
- **Generated At:** `2026-09-20T17:11:45Z`
- **Overall Verdict:** `PARTIAL`

## 1. Same-Model Numerical Parity

Direct pointwise trajectory and torque comparisons against the reference model.
Predeclared targets: **<= 1.0 mm** marker RMS, **<= 2.0 %** torque error (with 1.0 N·m floor).

| Engine        | Status        | Marker RMS Diff (mm) | Max Marker Diff (mm) | Torque Diff (%) | Total Work (J) | Wall Clock (s) | Reason                                                                                             |
| :------------ | :------------ | :------------------- | :------------------- | :-------------- | :------------- | :------------- | :------------------------------------------------------------------------------------------------- |
| **mujoco**    | `unverified`  | -                    | -                    | -               | -              | -              | Kinematic evaluation error: Coordinate count mismatch: candidate has 41, plant 'mujoco' expects 44 |
| **drake**     | `unavailable` | -                    | -                    | -               | -              | -              | Engine 'drake' is not installed or registered in the active environment                            |
| **pinocchio** | `unavailable` | -                    | -                    | -               | -              | -              | Engine 'pinocchio' is not installed or registered in the active environment                        |

## 2. Native-Model Observable Agreement

Evaluates observable kinematic paths (markers, yaw, ground forces) across distinct coordinate architectures.
Non-identical coordinate models do not inherit identical torque requirements.

| Engine       | Status        | Whole Marker RMSE (mm) | Pelvis Yaw RMSE (deg) | Model Name | Reason / Notes                                                             |
| :----------- | :------------ | :--------------------- | :-------------------- | :--------- | :------------------------------------------------------------------------- |
| **opensim**  | `unavailable` | -                      | -                     | `native`   | Engine 'opensim' is not installed or registered in the active environment  |
| **myosuite** | `unavailable` | -                      | -                     | `native`   | Engine 'myosuite' is not installed or registered in the active environment |
| **simscape** | `unavailable` | -                      | -                     | `native`   | Engine 'simscape' is not installed or registered in the active environment |

## 3. Assumptions and Model Invariants
