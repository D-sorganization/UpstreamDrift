---
title: Internal libraries that correctly have no tile (classification audit)
labels: documentation, priority/none
---

## Summary

The following modules are internal libraries consumed by other features. They correctly have no launcher tile and should remain internal:

| Module               | Path                                   | Consumed By                                   |
| -------------------- | -------------------------------------- | --------------------------------------------- |
| Biomechanics library | `src/shared/python/biomechanics/`      | MuJoCo, OpenSim, MyoSuite, Exercise Dashboard |
| Screw theory         | `src/shared/python/screw_theory/`      | Physics engines                               |
| Spatial algebra      | `src/shared/python/spatial_algebra/`   | Physics engines                               |
| Data I/O             | `src/shared/python/data_io/`           | Data Explorer, export                         |
| Plot engine          | `src/shared/python/plot_engine/`       | Dashboards, analysis                          |
| Reporting            | `src/shared/python/reporting/`         | Analysis/export flows                         |
| Realtime transport   | `src/shared/python/realtime/`          | Simulation WS, realtime API                   |
| Scripting env        | `src/shared/python/scripting/`         | Internal                                      |
| Pose editor widgets  | `src/shared/python/pose_editor/`       | Pose Studio                                   |
| Chat framework       | `src/shared/python/chat/`              | AI Sidekick                                   |
| Rust physics kernels | `rust_core/`                           | Physics engines                               |
| Robotics control     | `src/robotics/`                        | Drake, MuJoCo                                 |
| Freemocap ingest     | `src/motion_capture/freemocap_ingest/` | Motion Capture                                |
| Deployment modules   | `src/deployment/`                      | Production deployment                         |

These modules should be documented in capabilities arrays of the tiles that consume them, not given their own tiles.

## Acceptance Criteria

- [ ] Add capability tags to consuming tiles (e.g., `screw_theory` on MuJoCo tile)
- [ ] Update PROJECT_MAP.md to reflect these as internal modules
- [ ] No tiles needed
