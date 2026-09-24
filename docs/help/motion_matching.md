---
title: Motion Matching
tile_id: motion_matching
status: active
---

# Motion Matching

## Purpose

The Motion Matching tool fits articulated multibody biomechanical models to observed 3D marker and kinematic trajectory streams. It solves inverse kinematics and dynamic optimization problems to reproduce human golf swings across supported physics engines.

## Inputs

| Input | Units | Description |
| --- | --- | --- |
| Marker Trajectories | mm | Optical C3D or markerless joint position streams |
| Subject Mass & Height | kg, m | Subject anthropometric measurements for scaling |
| Engine Backend | categorical | Selected solver engine (Pinocchio, MuJoCo, Drake, OpenSim) |
| Convergence Tolerance | dimensionless | Residual threshold for kinematic error |

## Outputs

| Output | Units | Description |
| --- | --- | --- |
| Generalized Coordinates $q(t)$ | rad, m | Segment joint angles and root pelvis position over time |
| Generalized Velocities $\dot{q}(t)$ | rad/s, m/s | Joint angular velocities and linear motion |
| Fit Residual | mm | Root-mean-square distance between modeled and measured markers |
| Joint Torques $\tau(t)$ | N*m | Solved inverse-dynamics actuator forces |

## Method

Dynamic optimization formulates trajectory tracking as an optimal control or nonlinear least-squares problem with physics-based constraints. Forward dynamics verification checks consistency between solved joint torques and observed kinematic accelerations.

## Limitations

High-frequency impact events (club-ball collision) exhibit transient accelerations that require specialised contact models. Marker occlusions are interpolated using kinematic continuity priors.

## See Also

- [Motion Capture](motion_capture.md)
- [Tour Matching Viewer](tour_matching_viewer.md)
- [Matched Swing Browser](matched_swing_browser.md)
