---
title: Tour Matching Viewer
tile_id: tour_matching_viewer
status: active
---

# Tour Matching Viewer

## Purpose

The Tour Matching Viewer visualizes matched Tour-player swing baselines alongside forward-simulation candidates. It allows engineers to inspect segment kinematics, joint torques, ground reaction forces, and clubhead delivery metrics across professional capture datasets.

## Inputs

| Input | Units | Description |
| --- | --- | --- |
| Tour Baseline Receipt | json / npz | Certified Tour baseline capture dataset |
| Candidate Simulation | npz | Model forward-dynamics trajectory rollout |
| Alignment Landmark | event | Temporal synchronization keyframe (typically impact) |

## Outputs

| Output | Units | Description |
| --- | --- | --- |
| Kinematic Comparison | mm, deg | Residual trajectory differences between baseline and candidate |
| Kinetic Overlay | N, N*m | Ground reaction force profiles and segment torque comparisons |
| Delivery Metrics | m/s, deg | Face angle, path angle, and attack angle at impact |
| Parity Score | % | Normalized agreement metric against Tour baseline standards |

## Method

Synchronizes measured and simulated states using landmark time-warping or impact-relative alignment. Renders 3D skeletal avatars and force vectors using PyVista / VTK rendering pipelines.

## Limitations

Ground truth optical data is subject to marker placement uncertainty (~5 mm) and soft-tissue artifact. Simulated outputs depend on engine model fidelity.

## See Also

- [Matched Swing Browser](matched_swing_browser.md)
- [Motion Matching](motion_matching.md)
- [C3D Viewer](c3d_viewer.md)
