---
title: Analysis Tools API has 6 endpoints but no tile or frontend page
labels: feature, ui, priority/P1
---

## Problem

The analysis tools API (src/api/routes/analysis_tools.py) exposes:

- `GET /analysis/metrics` -- biomechanical metrics
- `GET /analysis/statistics` -- statistical summaries
- `POST /analysis/export` -- export analysis data
- `POST /simulation/position` -- set body position
- `POST /simulation/measure` -- take measurements
- `GET /simulation/measurements` -- available measurement tools

These are distinct from the generic `/analyze/biomechanics` route in `analysis.py`. They provide interactive measurement and analysis tooling with no launcher presence.

## Classification

**Borderline -- Tile-worthy**: The analysis tools are interactive measurement features that complement the simulation experience. They could be a sub-feature of the Simulation page or a standalone tile.

## Acceptance Criteria

- [ ] Expose analysis tools through either a dedicated tile or the Simulation page UI
- [ ] If tile: add `analysis_tools` to manifest with `web_route: /tools/analysis`
- [ ] If sub-feature: add measurement/analysis panels to the Simulation page
