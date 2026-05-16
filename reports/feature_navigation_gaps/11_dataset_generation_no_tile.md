---
title: Dataset Generation API has full endpoints but no tile or frontend
labels: feature, ui, priority/P1
---

## Problem

The dataset generation API (src/api/routes/dataset.py) has comprehensive endpoints:

- `POST /dataset/generate` -- generate synthetic swing datasets
- `POST /dataset/import-swing` -- import swing data
- `GET /dataset/control/state` -- dataset generator state
- `POST /dataset/control/configure` -- configure generation
- `GET /dataset/control/strategies` -- available strategies
- `GET /dataset/features` -- feature catalog
- `POST /dataset/execute` -- execute generation
- `GET /dataset/plots/types` -- plot type catalog
- `GET /dataset/export/formats` -- export format catalog

This is a significant feature for researchers who need training data, but it has no launcher tile and no Tauri frontend page.

## Classification

**Tile-worthy**: Dataset generation is a distinct workflow that researchers use independently of data exploration.

## Acceptance Criteria

- [ ] Add a `dataset_generator` tile to `launcher_manifest.json` with `web_route: /tools/dataset`
- [ ] Create a Tauri DatasetGenerator page component
- [ ] Add route in `ui/src/App.tsx`
- [ ] Assign to the `tool` category
