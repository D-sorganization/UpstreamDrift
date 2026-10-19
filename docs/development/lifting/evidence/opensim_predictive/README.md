# Predictive Moco Deadlift Simulation (OSL-7, #13044)

Predictive musculoskeletal simulation of a conventional barbell deadlift
using OpenSim Moco with DeGroote-Fregly 2016 muscles and foot contact.

## Result Status: Exploratory

- **Model:** `RajagopalLaiUhlrich2023_deadlift` (RajagopalLaiUhlrich2023 base)
- **Lifter:** 80.0 kg, 1.78 m, 100.0 kg bar
- **Mesh Intervals:** 15
- **Solver Status:** `Maximum_Iterations_Exceeded`
- **Converged:** `False`
- **Iterations:** 10
- **Objective:** 60.6998
- **Constraint Violation:** 7.7475e+00
- **Wall Clock Time:** 1627.00 s

## Kinematic Comparison

| Quantity | Bundle Kinematics | Predictive Solution | Unit |
| :--- | :--- | :--- | :--- |
| Bar Start Height | 0.456 | 0.456 | m |
| Bar Final Height | 0.935 | 0.935 | m |
| Bar Vertical Rise | 0.479 | 0.479 | m |
| Peak Bar Velocity | 0.56 | 0.45 | m/s |
| Lift Duration | 2.0 | 1.5 | s |

### Tracked OpenSim Solution Status
- **Status:** `unavailable`
- **Reason:** unavailable: no OSL-3 deadlift IK/ID receipts on main yet (OSL-3 #13041 is still open)

## Reproduction Command

```bash
python scripts/measure_opensim_predictive_deadlift.py
```
