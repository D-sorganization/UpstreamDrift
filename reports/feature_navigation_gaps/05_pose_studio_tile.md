---
title: Pose Studio has no launcher tile (standalone GUI completely hidden)
labels: feature, ui, priority/P1
---

## Problem

Pose Studio (src/tools/pose_studio/) is a standalone PyQt6 application for interactive pose editing. It has a full GUI (gui.py), CLI entry point (**main**.py), and documentation. Users who want to manipulate joint angles or retarget poses would look for this feature, but it's completely invisible in the launcher.

## Evidence

- src/tools/pose_studio/gui.py -- full PoseStudioApp with main window
- src/tools/pose_studio/**main**.py -- CLI entry point
- No entry in launcher_manifest.json
- No handler in launcher_model_handlers.py

## Acceptance Criteria

- [ ] Add a pose_studio tile to launcher_manifest.json
- [ ] Add PoseStudioHandler to launcher_model_handlers.py (as special_app type)
- [ ] Assign to the ool category
- [ ] Add SVG logo
- [ ] Verify standalone launch from both PyQt and Tauri launchers
