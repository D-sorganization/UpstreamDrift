---
title: Character Builder / URDF Generator has no launcher tile
labels: feature, ui, priority/P2
---

## Problem

The humanoid character builder (`src/shared/python/humanoid_character_builder/`) is a complete system:

- Anthropometric body parameter scaling
- URDF generation with mesh, collision, and physics validation
- MakeHuman and SMPL-X mesh generators
- Collision geometry (convex hull, decimation, primitive fitting)
- Preview and quick-build modes
- CLI entry point (`src/shared/python/model_generation/cli/main.py`)

The Model Explorer tile allows browsing URDFs but not building them. Users who want to create custom humanoid models have no way to discover or launch the builder.

## Classification

**Tile-worthy with caveats**: Building a character is a distinct workflow from browsing models. The builder has both a GUI preview mode and a CLI. It could be exposed as a mode within Model Explorer (`Open in Builder`) or as a separate tile.

## Acceptance Criteria

- [ ] Add either a `character_builder` tile or a `Build Model` action in Model Explorer
- [ ] If separate tile: add to manifest with path to `model_generation` CLI
- [ ] If integrated: add a `Create Model` button in Model Explorer that launches the builder
