---
title: Shot Pattern Analysis
tile_id: shot_pattern_analysis
status: complete
---

# Shot Pattern Analysis

## Purpose

Compare simulated straight, draw, and fade shot distributions under explicit
delivery variation and model assumptions. Open the tile in the launcher to use
the historical analysis controls or the Matched Study tab. The launcher attaches
this help page to the embedded tool's F1 help action.

## Requirements

Install the GUI dependencies with `pip install -e '.[gui-tools]'` and install
the native `upstream-physics` wheel. See the
[Tool Guide](../../src/tools/shot_pattern_analysis/README.md) for headless
commands and optional acceleration. Missing native physics support must be
resolved before running an analysis.

## Configure a Matched Study

1. Select Driver, 7-Iron, Pitching Wedge, or Custom. Review club speed in mph,
   static loft and shaft lean in degrees, and head mass in kilograms.
2. Choose curvature scale and target downrange distance in metres. A zero target
   selects matching within lean. Matching solves nominal face, path, and speed
   inside the displayed bounds; it does not rotate completed trajectories.
3. Set the delivery standard deviations in their displayed units. Shaft rotation
   SD is an input; delivered face SD is a derived result. Strike variation applies
   to heel/toe and high/low positions separately.
4. Enter a symmetric seven-variable correlation matrix with unit diagonal.
   The tool rejects a matrix that is not positive semidefinite.
5. Set shots per pattern, seed, and a new output directory. Run the study and
   inspect nominal matching status and supported populations before comparing
   distributions. Unsupported samples remain visible in the exports.

## Read the Results

Review nominal deliveries, landing distributions, realized covariance, and
supported-shot counts together. The matched results also show conditional course
scoring, paired differences against Straight, Monte Carlo intervals, and common
supported-population counts. These summaries use assumed course geometry and a
historical scoring baseline. Saved results and configuration exports support
reproduction; cancellation stops the background run.

## Limitations

These are geometric, model-conditional simulations, not measured golfer data.
Club properties beyond documented loft and lie remain research assumptions.
Matching changes nominal face, path, and speed, so it does not isolate a causal
curvature effect. The contact-law alternatives have not been qualified against
measured club-ball impacts. Results establish neither observed player benefit,
per-round strokes gained, nor a universal best shot shape.

## References

- [Matched Study Design](../research/shot_pattern_analysis/phase2/DESIGN.md)
- [Scientific Report](../research/shot_pattern_analysis/phase2/REPORT.md)
- [Saved-Data Reproduction](../research/shot_pattern_analysis/phase2/REPRODUCING.md)
- [Shot Pattern Tool Guide](../../src/tools/shot_pattern_analysis/README.md)
