---
title: Injury Risk Analysis has no UI entry
labels: feature, ui, priority/P1
---

## Problem

The injury analysis module (src/shared/python/injury/) is fully implemented with:

- `injury_risk.py` -- injury risk computation
- `joint_stress.py` -- joint stress analysis
- `spinal_load_analysis.py` -- spinal loading during swing
- `swing_modifications.py` -- swing modifications to reduce injury risk

This is a unique and valuable feature -- sports medicine researchers would specifically seek injury risk analysis. It has no launcher tile, no API route, and no dashboard.

## Classification

**Tile-worthy**: Injury risk is a distinct analysis workflow that researchers seek independently. It's not just an internal library -- it computes and reports on injury metrics.

## Acceptance Criteria

- [ ] Add an `injury_analysis` tile to `launcher_manifest.json`
- [ ] Create a launcher entry point or integrate into Exercise Dashboard
- [ ] Assign to the `biomechanics` category
- [ ] Add API routes for injury computation endpoints
