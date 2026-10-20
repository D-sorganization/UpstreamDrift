# Predictive Moco Deadlift Simulation (OSL-7, #13044)

Predictive musculoskeletal simulation of a conventional barbell deadlift
using OpenSim Moco with foot contact, bar carried at hands, and effort goal.

## Result Status: Exploratory

### Stage 1: Torque-Driven Solve

- **Actuation:** `CoordinateActuator`s on free coordinates; six pelvis residuals (optimal force 1765.197, effort weight 100.0); planted-feet goal weight 10000.0; muscles removed
- **Solver Status:** `Maximum_Iterations_Exceeded`
- **Converged:** `False`
- **Iterations:** 3000
- **Objective:** 216.19931759236485
- **Scaled Constraint Violation:** 0.3013375979560333 (last-iterate inf_pr: None)
- **Wall Clock Time:** 5778.964229101024 s
- **Total Weight (m g):** 1765.197 N
- **Residual Peak:** 2472.2729983369754 N
- **Residual RMS:** 1629.1642398487797 N
- **Residual Ratio:** 0.9229362160987017
- **Residual Moment Peak:** 648.8536116966834 N·m (RMS 216.2912474861732 N·m)
- **Residual Moment Limit (1 % COM height × m g):** None N·m (within: None)
- **Residual Assisted:** True
- **Pose Targets (bundle):** {'lockout_trunk_rad': 6.938893903907228e-18, 'setup_bar_height_m': 0.43201984819805705, 'setup_trunk_rad': 0.9075712110370514}
- **Setup Pose:** {'ankle_angle': 0.00513157987160259, 'arm_flex': 0.989520807970564, 'bar_height_m': 0.432019849605925, 'bar_x_m': 1.6728333518747918e-10, 'com_x_m': -0.07135408880532541, 'hip_flexion': 2.0197381561059657, 'knee_angle': 1.0638725704209953, 'lumbar_extension': 0.0, 'pelvis_tilt': -0.9850634676935863, 'pelvis_tx': -0.3391084875466617, 'pelvis_ty': 0.7150469901070875, 'sphere_x_m': [-0.12715591672331536, 0.027564979920075672, 0.09959093680323991, -0.12715591672331536, 0.027564979920075672, 0.09959093680323991], 'sphere_y_m': [-0.002012228006243316, -0.0010244530437438816, -0.0030000000000000374]}
- **Lockout Pose:** {'ankle_angle': 0.0515055071788027, 'arm_flex': 0.08194959693190208, 'bar_height_m': 0.8514348830095264, 'bar_x_m': -0.014694959344290002, 'com_x_m': -0.024927296347860516, 'hip_flexion': 0.0, 'knee_angle': 0.0, 'lumbar_extension': 0.0, 'pelvis_tilt': -0.07749225656822019, 'pelvis_tx': 0.05009705846052124, 'pelvis_ty': 1.0009788035490281, 'sphere_x_m': [-0.12903777371487862, 0.027957539524612662, 0.10108023419026593, -0.12903777371487862, 0.027957539524612662, 0.10108023419026593], 'sphere_y_m': [-0.0020103511302468624, -0.0010207022553469502, -0.0029999999999999125]}

### Stage 2: Muscle-Driven Solve

- **Actuation:** 80 Rajagopal lower-limb muscles (DeGroote-Fregly 2016)
- **Status:** `not run`
- **Reason:** Stage 1 torque-driven solve did not converge (Maximum_Iterations_Exceeded)

## Kinematic Comparison

| Quantity          | Bundle Kinematics   | Predictive Solution | Unit |
| :---------------- | :------------------ | :------------------ | :--- |
| Bar Start Height  | 0.43201984819805705 | None                | m    |
| Bar Final Height  | 0.9094192166845657  | None                | m    |
| Bar Vertical Rise | 0.47739936848650866 | None                | m    |
| Peak Bar Velocity | 0.5605000335510915  | None                | m/s  |
| Lift Duration     | 2.0                 | None                | s    |

### Tracked OpenSim Solution Status

- **Status:** `not compared`
- **Reason:** the OSL-3 deadlift IK/ID receipt is complete, but the predictive-versus-tracked comparison (joint moments, GRF) is not implemented yet

## Reproduction Command

```bash
UPSTREAMDRIFT_MSK_BASE_MODEL=<Rajagopal base model> python3 scripts/measure_opensim_predictive_deadlift.py --out-dir ctout/evidence --log-dir ctout/logs --mesh-intervals 15 --max-iterations 3000 --residual-force-scale 1 --residual-effort-weight 100 --fd-scheme forward --dynamics-mode explicit --dynamics-mode implicit
```
