# Simscape 2D Motion-Matching Plumbing Audit

## Current State
- The Simscape 2D Golf Model is a planar, reduced-DOF (3-5 DOFs) representation of the golf swing intended for fast parametric studies and educational use.
- The 2D model does not currently contain the programmatic `motion_matching` directory or `fit_swing` architecture present in the 3D model.
- Optimization workflows currently rely on Simulink Design Optimization (SDO) GUI sessions (e.g., `2D Optimization/*.mat`) to maximize club head speed via timing offsets, rather than tracking measured 3D trajectories.

## Architectural Gap
1. **Dimensionality Mismatch**: The canonical `MultiSourceTarget` contains 3D spatial trajectories (6-DOF club, 3D body markers). The 2D model operates entirely in the sagittal-like swing plane. 
2. **Missing Projection Layer**: To perform motion matching, the 3D target must first be mathematically projected onto the best-fit 2D swing plane.
3. **Missing Cost Function**: A 2D-specific `compute_cost` must be written to evaluate tracking error in the planar space, alongside a Python adapter `simscape_2d_adapter.py`.

## Recommended Path (Phase 2)
1. **Target Projection**: Implement a preprocessing utility that takes a `MultiSourceTarget` and projects its club and body markers onto a 2D plane defined by the clubhead trajectory.
2. **2D Python Bridge**: Port the `simscape_adapter.py` logic to interface with `golf_swing_2d.slx`.
3. **Registry Inclusion**: Create `src/engines/physics_engines/simscape_2d/python/motion_matching/provider.py` (once the wrapper exists) that encapsulates this 2D-projection and optimization.

*Follow-up task*: Implement the 3D-to-2D projection utility and the 2D `compute_cost` MATLAB function.
