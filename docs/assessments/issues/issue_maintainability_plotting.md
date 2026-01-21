---
title: "Refactor: Split plotting_core.py Monolith"
labels: ["jules:assessment", "needs-attention", "refactor"]
assignees: ["jules-architect"]
---

## Problem
`shared/python/plotting_core.py` is approximately **4500 lines** long. This "God Class" violates the Single Responsibility Principle, makes code navigation difficult, and increases the risk of merge conflicts.

## Evidence
- `wc -l shared/python/plotting_core.py` -> 4569 lines.
- Contains mixed logic for trajectories, energy plots, phase diagrams, etc.

## Proposed Solution
1. Create a package `shared/python/plotting/`.
2. Move logic into focused modules:
   - `trajectories.py`
   - `energy.py`
   - `phase_diagrams.py`
   - `comparison.py`
3. Update `shared/python/plotting/__init__.py` to expose the API.
