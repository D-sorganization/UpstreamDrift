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

The tool has two tabs.

### Independent Pattern Comparison

| Input                    | Description                                                                                                        | Unit          |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ | ------------- |
| Illustrative Club Preset | Driver, 7-Iron or Pitching Wedge starting values; every field below can be edited                                  | -             |
| Shots                    | Simulated shots per pattern                                                                                        | integer       |
| Seed                     | Random seed; the same seed reproduces the same draws                                                               | integer       |
| Face Variation           | Standard deviation of the face-angle error (0–5°)                                                                  | degrees       |
| Curve Magnitude          | Scale on the nominal draw/fade face and path (0.1–2×)                                                              | dimensionless |
| Face-to-Loft Model       | Fixed Delivered Loft, or Shaft Rotation Coupling (face error rotates the shaft, which also changes delivered loft) | -             |
| Lie, Shaft Lean          | Static lie and forward shaft lean at impact                                                                        | degrees       |
| Speed, Nominal Loft, AoA | Club speed, static loft and attack angle                                                                           | m/s, degrees  |
| Head Mass                | Clubhead mass                                                                                                      | kg            |
| Save Results To          | Output folder for the files below                                                                                  | -             |

### Matched Delivery Study

Straight, Draw and Fade are each matched to one zero-error landing point, then
all three patterns receive the same indexed delivery errors (paired shots).
Inputs are the output folder (it must be new or empty), shots per pattern, seed, curvature scale, an
optional fixed target distance, the match tolerance and target radius, face and
speed search bounds, the club (speed in mph, loft, shaft lean, attack angle,
head mass, contact law), the standard deviation of each of the seven noise
inputs (shaft twist, path, speed, horizontal and vertical strike, shaft lean,
attack angle) and an approximate face/path correlation.

## Outputs

| Output                                         | Tab         | Description                                                                                                      |
| ---------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------- |
| Summary table                                  | Independent | Per pattern: nominal delivered loft, lateral SD, aimed hit rate, aimed RMSE, mean carry and model strokes gained |
| `shots.csv`                                    | Independent | One row per shot: face, path, launch azimuth, spin, spin-axis tilt, carry X/Y/total and aimed X/Y                |
| `summary.json`, `receipt.json`                 | Both        | Summary statistics, inputs and the source hashes that produced them                                              |
| `overhead_flight.png`                          | Independent | Overhead flight paths                                                                                            |
| `dispersion.png`, `dispersion_equal_range.png` | Independent | Landing scatter, and the same scatter on equal axis ranges                                                       |
| `shots.csv.gz`                                 | Matched     | Every attempted paired shot, including unsupported deliveries and why                                            |
| `flight_samples.json`, `manifest.json`         | Matched     | Sampled trajectories and a SHA-256 manifest of every output file                                                 |
| Strokes-gained scoring                         | Both        | Hypothetical course scoring against a historical tour baseline; not a player-score prediction                    |

The tool does not draw confidence ellipses or report apex or landing angle.
Confidence intervals in the summaries cover Monte Carlo sampling only.
