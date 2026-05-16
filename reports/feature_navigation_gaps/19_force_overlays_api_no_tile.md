---
title: Force Overlays API has no tile (physics visualization feature hidden)
labels: feature, ui, priority/P2
---

## Problem

The force overlays API (`src/api/routes/force_overlays.py`) provides force/torque vector visualization for simulations. This is a distinct visualization feature that users would want to enable independently, but it has no launcher presence.

## Classification

**Internal library**: Force overlays are used by the simulation views (MuJoCo, Drake) internally. They don't need their own tile but should be documented as a capability of the simulation pages.

## Acceptance Criteria

- [ ] Add `force_overlays` to the MuJoCo and Drake tile `capabilities` arrays
- [ ] Document the force overlay API in the simulation page UI
- [ ] No separate tile needed
