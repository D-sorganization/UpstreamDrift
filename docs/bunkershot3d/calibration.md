# BunkerShot3D Calibration Data Sources

This document tracks the data sources and experimental procedures used to calibrate granular media models in BunkerShot3D.

## Angle of Repose

The `AngleOfReposeExperiment` class provides two paths:

1. **Analytical Mock (`backend="mock"`)**:
   Used exclusively for fast unit tests. Uses a hard-coded affine formula: `theta = 20.0 + friction * 24.0`.

2. **MuJoCo DEM Experiment (`backend="mujoco"`)**:
   The physical calibration path. It drops a specified number of rigid spherical grains (`n_grains`) of a given `grain_radius` into a cylindrical hopper (radius 0.10m, height 0.30m), allows them to settle under gravity (`0 0 -9.81`) for `settle_steps`, and then computes the macroscopic pile half-angle.
   
   - **Data Source**: First-principles physical rigid-body simulation via MuJoCo.
   - **Implementation**: See `_mujoco_angle_of_repose` in `bunkershot3d/calibration/angle_of_repose.py`.
   - **Reference**: Follows standard DEM hopper discharge/settling experiments.
