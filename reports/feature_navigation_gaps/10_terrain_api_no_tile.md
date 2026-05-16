---
title: Terrain API has 6 endpoints but no launcher tile or web route
labels: feature, ui, priority/P1
---

## Problem

The terrain engine (src/api/routes/terrain.py) exposes a full REST API:

- `GET /terrain/presets` -- preset terrain configurations
- `POST /terrain/load` -- load terrain data
- `POST /terrain/query` -- query terrain properties at coordinates
- `GET /terrain/materials` -- surface material types
- `GET /terrain/types` -- terrain type catalog
- `GET /terrain/active` -- currently active terrain

The underlying module (src/shared/python/physics/terrain_engine.py, topography, terrain physics, representations) is complete.

There's also a comprehensive terrain implementation under `src/shared/python/physics/` (terrain*engine.py, topography.py, terrain_representation.py, terrain_physics.py, \_terrain*\*.py).

Yet there's no tile and no `web_route` in the manifest. The Putting Green tile uses terrain but doesn't expose the general terrain system.

## Classification

**Tile-worthy**: Terrain configuration is a distinct workflow (design greens, configure surfaces, import elevation data). Researchers configuring simulation environments need this independently.

## Acceptance Criteria

- [ ] Add a `terrain` tile to `launcher_manifest.json` with `web_route: /tools/terrain`
- [ ] Create a Tauri route for the terrain configuration page
- [ ] Assign to the `tool` category
