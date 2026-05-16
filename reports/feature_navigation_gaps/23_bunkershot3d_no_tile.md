---
title: BunkerShot3D has no launcher tile
labels: feature, ui, priority/P3
---

## Problem

BunkerShot3D (`src/bunkershot3d/`) is a complete sand-shot simulation with:

- MPM, LIGGGHTS, and Chrono backends
- Calibration system (angle of repose, drained shear cell)
- Clubhead geometry and kinematics
- Configuration system (`configs/bunkershot3d/canonical.yaml`)
- I/O and post-processing

It has no manifest tile and no way for users to discover or launch it.

## Classification

**Tile-worthy**: BunkerShot3D is a distinct simulation domain (sand bunkers). Golf researchers would seek this out specifically. It's not just a sub-feature of MuJoCo.

## Acceptance Criteria

- [ ] Add a `bunkershot3d` tile to `launcher_manifest.json`
- [ ] Add a handler for launching the BunkerShot3D simulator
- [ ] Assign to the `physics_engine` category
- [ ] Add appropriate SVG logo
