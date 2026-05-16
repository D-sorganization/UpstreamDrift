---
title: Swing Optimization has no UI entry (core analysis feature invisible)
labels: feature, ui, priority/P1
---

## Problem

Swing optimization (src/shared/python/optimization/) is a fully implemented module with:

- `swing_optimizer.py` -- trajectory optimization with constraint support
- `_swing_constraints.py`, `_swing_kinematics.py`, `_swing_models.py`, `_swing_objectives.py`
- `swing_bridge.py` -- bridge to engine simulations
- `examples/optimize_arm.py` -- runnable example

This is a core analysis capability (trajectory optimization is F11 in SPEC.md), but it has no launcher tile, no dedicated API route, and no web route. Users cannot discover it.

## Classification

**Tile-worthy**: Trajectory optimization is a first-class feature in the spec. Researchers would seek it out independently.

## Acceptance Criteria

- [ ] Add a `swing_optimization` tile to `launcher_manifest.json`
- [ ] Create a launcher entry point (CLI or GUI wrapper)
- [ ] Add an API route for optimization endpoints (`/analysis/optimize`)
- [ ] Assign to the `simulation` or `biomechanics` category
