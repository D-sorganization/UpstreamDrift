---
title: Empty sidebar categories show no tiles (Biomechanics, Simulation, Motion Matching)
labels: bug, ui, priority/P0
---

## Problem

The PyQt6 launcher sidebar defines 7 category filter buttons (Home, Physics Engines, Biomechanics, Simulation, Motion Matching, Motion Capture, Tools & Data). Three of these categories currently filter to **zero tiles**:

| Category        | Tiles in Manifest                   | User Experience |
| --------------- | ----------------------------------- | --------------- |
| Biomechanics    | 0                                   | Empty grid      |
| Simulation      | 0 (Putting Green is physics_engine) | Empty grid      |
| Motion Matching | 0                                   | Empty grid      |

When a user clicks any of these three sidebar buttons, they see an entirely empty launcher grid. This is a broken navigation path.

## Source

- Sidebar categories are defined in src/launchers/launcher_ui_setup.py lines ~295-413
- Tile categories come from src/config/launcher_manifest.json
- Category definitions are in src/config/launcher_manifest_loader.py lines 47-71

## Acceptance Criteria

- [ ] Every sidebar category button filters to at least one visible tile
- [ ] Either add tiles that map to these categories, or remove/merge the empty categories
- [ ] Verify the Tauri React dashboard also handles empty categories gracefully (show a message instead of a blank grid)

## Suggested Fixes

**Option A -- Add tiles (preferred):** Add manifest entries for features that belong in these categories:

- **Biomechanics**: Exercise Dashboard, Injury Risk Analysis, Swing Comparison
- **Simulation**: Cross-Engine Dashboard, Golf Simulation Suite, Shot Tracer
- **Motion Matching**: Motion-Match Preview already exists; add Motion Pipeline and Surrogate Training

**Option B -- Merge categories:** If these features aren't ready for standalone tiles, merge them into existing categories and remove the empty sidebar buttons to avoid dead-end navigation.
