---
title: Robotics module has no launcher presence
labels: feature, priority/P3
---

## Problem

The robotics module (`src/robotics/`) provides:

- Motion planning: RRT, RRT\* planners
- Control: impedance, admittance, hybrid force-position
- Sensing: sensor models
- Whole-body control
- Locomotion: footstep planning
- Collision detection

These capabilities are referenced in SPEC.md (control schemes F12, constraint analysis) but have no UI entry point.

## Classification

**Internal library**: The robotics module is consumed by the physics engines (Drake, MuJoCo) for control and planning. Users access these features through simulation configuration, not through a standalone tool.

However, the control scheme selection (impedance vs. admittance vs. operational space) is something users might want to configure per-simulation.

## Acceptance Criteria

- [ ] Add `control_schemes` to MuJoCo and Drake tile `capabilities` arrays
- [ ] Expose control scheme selection in the Simulation page
- [ ] No standalone tile needed; these are configuration options
