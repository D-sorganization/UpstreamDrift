---
title: Engine-specific dashboards unreachable from launcher (Drake, MuJoCo, Pinocchio)
labels: feature, ui, priority/P1
---

## Problem

Three engine-specific dashboard classes exist but have no launcher tiles:

1. **MuJoCo Dashboard** (src/launchers/mujoco_dashboard.py) -- MuJoCoDashboard(UnifiedDashboardWindow)
2. **Drake Dashboard** (src/launchers/drake_dashboard.py) -- DrakeDashboard(UnifiedDashboardWindow)
3. **Pinocchio Dashboard** (src/launchers/pinocchio_dashboard.py) -- PinocchioDashboard(UnifiedDashboardWindow)

These are richer analytics views than the basic engine launchers. Currently, clicking a MuJoCo tile launches the Humanoid Golf simulation, but there's no way to access the dedicated analytics dashboard that shows force vectors, energy plots, and advanced metrics.

The unified launcher tiles each launch a single entry point (e.g., mujoco_unified_launcher.py), but the dashboards are separate, more feature-rich views.

## Options

1. **Add dashboard tiles** for each engine alongside the existing simulation tiles
2. **Merge dashboards into the unified launcher** -- the unified launcher could offer a "Switch to Dashboard" mode
3. **Add a dashboard toggle** within each engine tile (e.g., a dropdown or secondary click option)

## Acceptance Criteria

- [ ] Decide on approach (separate tiles vs. merged vs. toggle)
- [ ] Expose engine dashboards through the launcher
- [ ] Add tests verifying dashboard launch paths
