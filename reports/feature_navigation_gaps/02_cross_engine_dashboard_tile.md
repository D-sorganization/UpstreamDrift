---
title: Cross-Engine Dashboard has no launcher tile (core feature F6 invisible)
labels: bug, ui, priority/P0
---

## Problem

Cross-engine validation is a **core feature (F6)** in SPEC.md with full ✅ status. The CrossEngineDashboardWindow class (src/launchers/cross_engine_dashboard.py) is a complete PyQt6 application for multi-engine comparison. It has a dedicated UnifiedDashboardWindow base class, tests, and CLI entry points.

Despite being fully implemented, it has **no tile in the launcher manifest** and no way for users to discover or launch it from the main UI.

## Evidence

- src/launchers/cross_engine_dashboard.py -- full CrossEngineDashboardWindow class
- src/launchers/base.py -- BaseLauncher with dashboard infrastructure
-     ests/launchers/test_cross_engine_dashboard.py -- full test suite
-     ests/launchers/test_cross_engine_dashboard_cli.py -- CLI tests
- SPEC.md F6: "Cross-engine validation ✅"
- No entry in src/config/launcher_manifest.json

## Acceptance Criteria

- [ ] Add a cross_engine tile to launcher_manifest.json
- [ ] Assign it to the simulation category
- [ ] Add appropriate SVG logo to src/launchers/assets/
- [ ] Verify tile launches the Cross-Engine Dashboard
- [ ] Verify Tauri dashboard renders the new tile
