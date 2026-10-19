# Predictive Moco Deadlift Simulation (OSL-7, #13044)

Predictive musculoskeletal simulation of a conventional barbell deadlift
using OpenSim Moco with foot contact, bar carried at hands, and effort goal.

## Result Status: Exploratory

### Stage 1: Torque-Driven Solve
- **Actuation:** `CoordinateActuator`s (500 N·m optimal force) on free coordinates; muscles removed
- **Solver Status:** `Maximum_Iterations_Exceeded`
- **Converged:** `False`
- **Iterations:** 100
- **Objective:** 1844.187478122922
- **Scaled Constraint Violation:** 74.1596931108721
- **Wall Clock Time:** 231.37 s
- **Solution File:** `predictive_solution_stage1.sto` (unsealed Moco solution iterate)

### Stage 2: Muscle-Driven Solve
- **Actuation:** 80 Rajagopal lower-limb muscles (DeGroote-Fregly 2016)
- **Status:** `not run`
- **Reason:** Stage 1 torque-driven solve did not converge (Maximum_Iterations_Exceeded)

## Kinematic Comparison

| Quantity | Bundle Kinematics | Predictive Solution | Unit |
| :--- | :--- | :--- | :--- |
| Bar Start Height | 0.43201984819805705 | None | m |
| Bar Final Height | 0.9094192166845657 | None | m |
| Bar Vertical Rise | 0.47739936848650866 | None | m |
| Peak Bar Velocity | 0.5605000335510915 | None | m/s |
| Lift Duration | 2.0 | None | s |

### Tracked OpenSim Solution Status
- **Status:** `unavailable`
- **Reason:** unavailable: no OSL-3 deadlift IK/ID receipts on main yet (OSL-3 #13041 is still open)

## Reproduction Command

```bash
python scripts/measure_opensim_predictive_deadlift.py
```
