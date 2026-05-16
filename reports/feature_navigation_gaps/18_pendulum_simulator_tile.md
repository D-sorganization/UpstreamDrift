---
title: Pendulum Simulator has no launcher tile (educational models unreachable)
labels: feature, ui, priority/P2
---

## Problem

The pendulum simulator (`src/shared/python/pendulum_simulator/`) has a CLI entry point (`__main__.py`) and was reachable from the deprecated `golf_suite_launcher.py`. The SPEC.md explicitly lists educational 2-DOF pendulums as a supported model type.

Users looking for simple educational models have no way to launch them from the tiled UI.

## Classification

**Tile-worthy**: The pendulum is explicitly called out in SPEC.md as a supported model complexity tier. It provides an approachable entry point for new users.

## Acceptance Criteria

- [ ] Add a `pendulum` tile to `launcher_manifest.json`
- [ ] Add a handler for the pendulum simulator type
- [ ] Assign to `physics_engine` category (or a new `educational` category)
- [ ] Consider grouping with other educational/quick-start models
