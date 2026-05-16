---
title: Golf Simulation Suite has a handler but no launcher tile
labels: bug, ui, priority/P1
---

## Problem

The Golf Simulation Suite (src/tools/golf_simulation_suite/) has a full CLI entry point and a GolfSimulationSuiteHandler registered in launcher_model_handlers.py (model type golf_simulation). It's a complete application that should be launchable from the tiled UI.

## Evidence

- src/tools/golf_simulation_suite/**main**.py -- CLI entry with main()
- src/launchers/launcher_model_handlers.py:293 -- GolfSimulationSuiteHandler registered
- No tile in launcher_manifest.json

## Acceptance Criteria

- [ ] Add a golf_simulation_suite tile to launcher_manifest.json
- [ ] Assign to the simulation category
- [ ] Add SVG logo
- [ ] Verify launch through the tile
