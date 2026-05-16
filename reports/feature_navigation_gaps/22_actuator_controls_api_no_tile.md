---
title: Actuator Controls API has no tile (simulation sub-feature)
labels: documentation, priority/P2
---

## Problem

The actuator controls API (`src/api/routes/actuator_controls.py`) provides:

- `GET /simulation/actuators` -- list actuators
- `POST /simulation/actuators` -- configure actuator controls
- `POST /simulation/actuators/control` -- send control commands

This is a simulation sub-feature, not a standalone workflow. It should be documented as a capability of the Simulation page.

## Classification

**Internal library**: Actuator controls are used within simulation sessions. No tile needed.

## Acceptance Criteria

- [ ] Add `actuator_controls` to MuJoCo and Drake tile `capabilities` arrays
- [ ] Expose actuator controls in the Simulation page UI
- [ ] No separate tile needed
