---
title: Perturbation Analysis (cross-engine robustness) has no UI
labels: feature, ui, priority/P2
---

## Problem

The perturbation analysis module (`src/shared/python/perturbation/`) provides cross-engine robustness scoring:

- `cross_engine_runner.py` -- run perturbed simulations across engines
- `noise.py` -- noise injection (Gaussian, uniform, etc.)
- `robustness_score.py` -- compute robustness metrics
- `statistics.py` -- statistical analysis of perturbation results
- `analyzer_base.py` -- extensible analyzer framework

This directly supports the cross-engine validation feature (F6) and the `Simulation` sidebar category, but has no UI entry point.

## Classification

**Internal library** with **borderline tile potential**: Perturbation analysis is primarily a programmatic feature, but researchers doing robustness studies would want a way to configure and launch it. Could be a mode within the Cross-Engine Dashboard rather than a standalone tile.

## Acceptance Criteria

- [ ] Add perturbation analysis configuration to the Cross-Engine Dashboard
- [ ] Or: add a `perturbation` tile if the workflow is distinct enough
- [ ] Add API route for configuring and running perturbation studies
