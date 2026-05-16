---
title: Shot Tracer GUI has no launcher tile (only reachable from deprecated launcher)
labels: feature, ui, priority/P1
---

## Problem

The MultiModelShotTracerWindow (src/launchers/\_shot_tracer_gui.py) is a complete PyQt6 application for trajectory visualization and comparison. It's a user-facing tool that researchers would actively seek out.

Currently it's only launchable from golf_suite_launcher.py (deprecated), which shows a deprecation warning pointing users to the unified launcher. But the unified launcher manifest has no Shot Tracer tile.

## Evidence

- src/launchers/\_shot_tracer_gui.py -- MultiModelShotTracerWindow(QMainWindow) with full GUI
- src/launchers/shot_tracer.py -- CLI wrapper
- src/launchers/golf_suite_launcher.py -- deprecated launcher has \_launch_shot_tracer()
- src/launchers/launcher_model_handlers.py -- no handler for shot_tracer type
- No tile in launcher_manifest.json

## Acceptance Criteria

- [ ] Add a shot_tracer tile to launcher_manifest.json
- [ ] Add ShotTracerHandler to launcher_model_handlers.py
- [ ] Assign to the simulation category
- [ ] Add SVG logo (a trajectory/arc icon)
- [ ] Remove \_launch_shot_tracer from deprecated golf_suite_launcher.py or redirect it
