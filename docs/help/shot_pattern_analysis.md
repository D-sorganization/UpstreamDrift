---
title: Shot Pattern Analysis
tile_id: shot_pattern_analysis
status: complete
---

# Shot Pattern Analysis

## Purpose

Shot Pattern Analysis is a local GUI and batch analysis tool for comparing
paired straight, draw, and fade flight patterns across standard club selections
(Driver, 7-Iron, Pitching Wedge). It investigates lateral dispersion, carry
trade-offs, delivery variance sensitivity, and scoring expectations under
geometric and physical impact constraints.

Entry point `src/tools/shot_pattern_analysis/__main__.py`, which delegates
to `src/tools/shot_pattern_analysis/gui.py`. Run it with
`python -m src.tools.shot_pattern_analysis`, or launch the tile from the
UpstreamDrift launcher. Declared capabilities: `shot_pattern_analysis`,
`shot_dispersion`, `impact_physics`.

## Inputs

| Input                         | Description                                                        | Unit          |
| ----------------------------- | ------------------------------------------------------------------ | ------------- |
| Club selection                | Driver, 7-Iron, or Pitching Wedge                                  | -             |
| Face angle standard deviation | Normal standard deviation of face delivery angle (e.g. 1.0°, 2.0°) | degrees       |
| Curve multiplier              | Nominal or doubled curvature scaling factor                        | dimensionless |
| Delivery assumption           | Fixed-loft or assumed shaft-rotation delivery kinematics           | -             |
| Sample count                  | Number of simulated shots per pattern (e.g. 10,000)                | integer       |
| Target range / landing target | Nominal aiming and distance target coordinates                     | meters        |

## Outputs

| Output                  | Description                                                             |
| ----------------------- | ----------------------------------------------------------------------- |
| Lateral / carry scatter | 2D landing distribution and dispersion ellipses                         |
| Dispersion metrics      | Radial standard deviation, lateral spread, longitudinal spread          |
| Target score            | Benchmark tee/fairway/rough or approach/green/rough scoring expectation |
| Raw shot CSVs           | Flight endpoints, apex, landing angle, and impact parameters            |
| High-resolution plots   | 1080p overhead dispersion and trajectory plots                          |
