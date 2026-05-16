---
title: Exercise Dashboard has a handler but no launcher tile
labels: bug, ui, priority/P0
---

## Problem

The ExerciseDashboard class (src/launchers/exercise_dashboard.py) is a complete PyQt6 application for biomechanics exercise workflows. It has a registered BiomechExerciseHandler in src/launchers/launcher_model_handlers.py (model type iomech_exercise).

The sidebar has a **Biomechanics** category button that currently filters to zero tiles. The Exercise Dashboard belongs in that category but cannot be launched from the UI.

## Evidence

- src/launchers/exercise_dashboard.py -- full ExerciseDashboard(QMainWindow)
- src/launchers/launcher_model_handlers.py:243 -- BiomechExerciseHandler registered
- src/launchers/launcher_ui_setup.py:295 -- tn_biomechanics sidebar button exists
- No iomech_exercise tile in launcher_manifest.json

## Acceptance Criteria

- [ ] Add a iomech_exercise tile to launcher_manifest.json
- [ ] Assign it to the iomechanics category
- [ ] Add appropriate SVG logo
- [ ] Verify launch from the Biomechanics sidebar category
