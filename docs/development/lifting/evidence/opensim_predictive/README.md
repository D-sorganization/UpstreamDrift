# Predictive Moco Deadlift Simulation (OSL-7, #13044)

Predictive musculoskeletal simulation of a conventional barbell deadlift
using OpenSim Moco with foot contact, bar carried at hands, and effort goal.

## Result Status: Exploratory

### Stage 1: Torque-Driven Solve

- **Actuation:** `CoordinateActuator`s (1000 N·m optimal force) on free coordinates; pelvis residuals (optimal force 1, weight 100); muscles removed
- **Solver Status:** `Timeout_Interrupted`
- **Converged:** `False`
- **Iterations:** 1056
- **Objective:** 20049.777
- **Scaled Constraint Violation:** None (last-iterate inf_pr: 654.0)
- **Wall Clock Time:** None s
- **Total Weight (m g):** 1765.197 N
- **Residual Peak:** None N
- **Residual RMS:** None N
- **Residual Ratio:** None
- **Residual Assisted:** None

### Stage 2: Muscle-Driven Solve

- **Actuation:** 80 Rajagopal lower-limb muscles (DeGroote-Fregly 2016)
- **Status:** `not run`
- **Reason:** Stage 1 torque-driven solve did not converge (Timeout_Interrupted)

## Kinematic Comparison

| Quantity          | Bundle Kinematics   | Predictive Solution | Unit |
| :---------------- | :------------------ | :------------------ | :--- |
| Bar Start Height  | 0.43201984819805705 | None                | m    |
| Bar Final Height  | 0.9094192166845657  | None                | m    |
| Bar Vertical Rise | 0.47739936848650866 | None                | m    |
| Peak Bar Velocity | 0.5605000335510915  | None                | m/s  |
| Lift Duration     | 2.0                 | None                | s    |

### Tracked OpenSim Solution Status

- **Status:** `unavailable`
- **Reason:** unavailable: no OSL-3 deadlift IK/ID receipts on main yet (OSL-3 #13041 is still open)

## Reproduction Command

```bash
python scripts/measure_opensim_predictive_deadlift.py
```
