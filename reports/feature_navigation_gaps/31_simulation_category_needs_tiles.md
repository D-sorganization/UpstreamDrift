---
title: Simulation sidebar category needs tiles (Putting Green + others)
labels: bug, ui, priority/P2
---

## Problem

This is a companion to issues #01, #02, #06, and #30. The `simulation` sidebar category currently has zero tiles because:

1. Putting Green is categorized as `physics_engine` instead of `simulation`
2. Cross-Engine Dashboard has no tile at all
3. Golf Simulation Suite has no tile
4. Shot Tracer has no tile

Even after fixing #30 (re-categorizing Putting Green), the category would have only one tile. It needs more.

## Acceptance Criteria

- [ ] Move Putting Green to `simulation` category
- [ ] Add Cross-Engine Dashboard tile to `simulation`
- [ ] Add Golf Simulation Suite tile to `simulation`
- [ ] Add Shot Tracer tile to `simulation`
- [ ] Verify Simulation sidebar shows all tiles
