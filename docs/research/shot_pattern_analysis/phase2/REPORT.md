# Matched Shot Patterns: Final Conditional Simulation Report

## Scope and Status

The primary within-lean matrix completed 270 / 324 planned cells; 54 were unavailable. The across-lean fixed-target control completed 52 / 72 planned cells; 20 were unavailable. Each completed cell attempted 10,000 paired Gaussian delivery errors for each of Straight, Draw, and Fade. Unsupported shots remain in the population accounting and are never retried.

These are model-conditional sensitivities, not measured player outcomes or a universal advantage for a shot shape. Positive paired strokes gained favors the named curve; negative paired radial-error difference favors the named curve. The 95% intervals quantify Monte Carlo sampling only.

## Design and Physical Interpretation

The [Scientific Design](DESIGN.md) states the assumptions, equations, sourced static geometry, and numerical gates. Straight, draw, and fade nominals are solved to one physical zero-error landing point within each lean. The control attempts the same fixed target across leans. Neither endpoints nor paths are rotated after simulation. Matching landing X/Y does not equate ball speed, launch, peak height, descent, or spin.

The model uses illustrative club masses, inertias, CG, face curvature, friction, and energetic restitution, plus a rigid shaft/face orientation. It omits roll, hazards, turf interaction, shaft deflection, compliant contact, and measured human delivery distributions. Landing is treated as final lie for hypothetical course scoring.

### Relation to Phase One

[Phase One](../README.md) included both a fixed-delivered-loft face-error experiment and earlier shaft-rotation geometric controls. Those historical outputs remain separate. The original face-angle SD is not interchangeable with this phase's primitive shaft-twist SD. This phase pitches the complete club, matches raw nominal landings, adds path/speed/strike noise, and uses the independently reviewed two-body incremental contact law. Numeric phase-one and phase-two effects must not be pooled as if they were the same experiment.

### Deterministic Head-Assumption Sensitivity

The [Deterministic Head-Assumption Sensitivity](DESIGN.md#deterministic-head-assumption-sensitivity) section of the scientific design, supported by the [Saved Sensitivity Summary](validation/independent_head_sensitivity_summary.json) and [Full Execution Receipt](validation/independent_head_sensitivity.json), documents a separate deterministic evaluation of off-center impact sensitivity. The chosen ±20% perturbations to clubhead parameters (CG depth, principal inertias, face radii) and contact-law comparison (energetic restitution versus Newton velocity restitution) are model sensitivities, not measured parameter uncertainty. Narrow Monte Carlo confidence intervals quantify only random delivery draw sampling error and do not validate impact model accuracy or off-center delivery realism.

One unavailable planned cell is `driver_lean10_curve1_rotation1_shaft_only`: `nonairborne_launch`. The complete inventory preserves every reason.

## Controlled Inputs

Each completed cell uses 10,000 indexed errors per physical pattern with seed `20261009`. Nominal matching accepts raw X/Y residual at most 0.05 m. Under original protocols (protocol original_primary/1 and original_control/1), nominal speed is searched within 75–125% of each listed reference speed; no post-simulation aiming rotation is applied. Flight timestep, contact resolution, runtime backend and exact source hashes are frozen in the per-cell execution receipts.

| Club           | Static Loft (°) | Lie at Zero Lean (°) | Reference Speed (m/s) | Attack Angle (°) | Assumed Head Mass (kg) |
| -------------- | --------------: | -------------------: | --------------------: | ---------------: | ---------------------: |
| driver         |              10 |                 58.5 |                    45 |             -0.9 |                  0.200 |
| seven_iron     |              34 |                   63 |                    36 |               -4 |                  0.272 |
| pitching_wedge |              46 |                   64 |                    32 |               -5 |                  0.300 |

The static loft/lie citations and every assumed head property are listed in the [Scientific Design](DESIGN.md). The score model treats carry endpoints as final lie on hypothetical flat course geometry.

## Primary Within-Lean Results

The source-bound [Primary Matrix Plan](final_v1/matrix_plan.json) and per-cell manifests identify every input and output. Every pattern in every cell, including explicit unavailable cases, appears in the [Complete Cell Inventory](REPORT_CELLS.csv). Supported-shot statistics and course scores are conditional whenever a cell has unsupported deliveries. Curves are compared against Straight on common supported shot indices within each cell, rather than pooling different target distances.

### Nine-Profile Input Controls

All rows also use shaft-rotation SD 1° or 2° according to the cell. Speed SD is shown in mph; contact SD is per axis, not radial. Correlations below describe primitive shaft rotation and path errors, not the derived face/path correlation. The full seven-dimensional covariance and realized output covariance remain in the source-bound cell receipts.

| Profile               | Path SD (°) | Speed SD (mph) | Strike SD (mm Each Axis) | Primitive Shaft/Path ρ |
| --------------------- | ----------: | -------------: | -----------------------: | ---------------------: |
| shaft_only            |           0 |            0.0 |              0.00 / 0.00 |                  +0.00 |
| shaft_path            |         0.5 |            0.0 |              0.00 / 0.00 |                  +0.00 |
| shaft_speed1          |           0 |            1.0 |              0.00 / 0.00 |                  +0.00 |
| shaft_speed2          |           0 |            2.0 |              0.00 / 0.00 |                  +0.00 |
| shaft_strike          |           0 |            0.0 |              6.35 / 6.35 |                  +0.00 |
| combined1             |         0.5 |            1.0 |              6.35 / 6.35 |                  +0.00 |
| combined2             |         0.5 |            2.0 |              6.35 / 6.35 |                  +0.00 |
| cov_facepath_positive |         0.5 |            1.0 |              6.35 / 6.35 |                  -0.60 |
| cov_facepath_negative |         0.5 |            1.0 |              6.35 / 6.35 |                  +0.60 |

### Condition-Resolved Carry and Dispersion

Each row names the actual cell. Mean carry, signed X/Y bias, spread, and RMSE use that pattern's individually supported endpoints. The shared nominal target removes initial range mismatch within a cell; random errors can still shift realized carry. Primary lean targets can differ, while fixed-target controls rematch face, path, and speed and do not isolate a causal distance or lean effect.

| Condition ID                                     | Pattern  | Mean Carry / Target X (m) | X Bias / SD (m) | Y Bias / SD (m) | Target RMSE (m) |   Supported / Attempted [95% CI] |
| ------------------------------------------------ | -------- | ------------------------: | --------------: | --------------: | --------------: | -------------------------------: |
| driver_lean0_curve1_rotation2_combined2          | Straight |           175.23 / 178.95 |   -4.14 / 12.24 |   +0.95 / 12.13 |           17.75 | 10,000 / 10,000 [100.0%, 100.0%] |
| driver_lean0_curve1_rotation2_combined2          | Draw     |           175.10 / 178.95 |   -4.25 / 13.07 |   +1.25 / 11.80 |           18.16 | 10,000 / 10,000 [100.0%, 100.0%] |
| driver_lean0_curve1_rotation2_combined2          | Fade     |           175.39 / 178.95 |   -4.00 / 11.46 |   +0.68 / 12.33 |           17.31 | 10,000 / 10,000 [100.0%, 100.0%] |
| driver_lean5_curve1_rotation2_combined2          | Straight |           102.50 / 108.03 |   -5.69 / 25.28 |    +1.55 / 6.12 |           26.68 |    9,940 / 10,000 [99.2%, 99.5%] |
| driver_lean5_curve1_rotation2_combined2          | Draw     |           102.64 / 108.03 |   -5.53 / 26.64 |    +1.74 / 5.85 |           27.88 |    9,940 / 10,000 [99.2%, 99.5%] |
| driver_lean5_curve1_rotation2_combined2          | Fade     |           102.76 / 108.03 |   -5.44 / 24.06 |    +1.35 / 6.32 |           25.49 |    9,940 / 10,000 [99.2%, 99.5%] |
| driver_lean10_curve1_rotation2_combined2         | Straight |             Not Attempted |               — |               — |               — |                                — |
| driver_lean10_curve1_rotation2_combined2         | Draw     |             Not Attempted |               — |               — |               — |                                — |
| driver_lean10_curve1_rotation2_combined2         | Fade     |             Not Attempted |               — |               — |               — |                                — |
| pitching_wedge_lean0_curve1_rotation2_combined2  | Straight |             75.57 / 77.40 |    -1.86 / 4.43 |    -0.09 / 2.14 |            5.26 |   9,999 / 10,000 [99.9%, 100.0%] |
| pitching_wedge_lean0_curve1_rotation2_combined2  | Draw     |             75.57 / 77.40 |    -1.86 / 4.38 |    -0.07 / 2.17 |            5.23 |   9,999 / 10,000 [99.9%, 100.0%] |
| pitching_wedge_lean0_curve1_rotation2_combined2  | Fade     |             75.58 / 77.40 |    -1.85 / 4.48 |    -0.11 / 2.10 |            5.29 |   9,999 / 10,000 [99.9%, 100.0%] |
| pitching_wedge_lean5_curve1_rotation2_combined2  | Straight |             88.27 / 90.43 |    -2.20 / 4.94 |    -0.10 / 2.57 |            5.99 |   9,999 / 10,000 [99.9%, 100.0%] |
| pitching_wedge_lean5_curve1_rotation2_combined2  | Draw     |             88.27 / 90.43 |    -2.20 / 4.89 |    -0.07 / 2.61 |            5.97 |   9,999 / 10,000 [99.9%, 100.0%] |
| pitching_wedge_lean5_curve1_rotation2_combined2  | Fade     |             88.27 / 90.43 |    -2.19 / 4.99 |    -0.13 / 2.53 |            6.01 |   9,999 / 10,000 [99.9%, 100.0%] |
| pitching_wedge_lean10_curve1_rotation2_combined2 | Straight |           100.33 / 102.82 |    -2.53 / 5.41 |    -0.11 / 3.03 |            6.70 |   9,999 / 10,000 [99.9%, 100.0%] |
| pitching_wedge_lean10_curve1_rotation2_combined2 | Draw     |           100.32 / 102.82 |    -2.54 / 5.37 |    -0.07 / 3.08 |            6.69 |   9,999 / 10,000 [99.9%, 100.0%] |
| pitching_wedge_lean10_curve1_rotation2_combined2 | Fade     |           100.33 / 102.82 |    -2.53 / 5.46 |    -0.14 / 2.98 |            6.71 |   9,999 / 10,000 [99.9%, 100.0%] |
| seven_iron_lean0_curve1_rotation2_combined2      | Straight |           125.33 / 127.88 |    -2.65 / 5.45 |    -0.17 / 4.99 |            7.85 |   9,996 / 10,000 [99.9%, 100.0%] |
| seven_iron_lean0_curve1_rotation2_combined2      | Draw     |           125.32 / 127.88 |    -2.66 / 5.39 |    -0.11 / 5.06 |            7.86 |   9,996 / 10,000 [99.9%, 100.0%] |
| seven_iron_lean0_curve1_rotation2_combined2      | Fade     |           125.33 / 127.88 |    -2.65 / 5.51 |    -0.23 / 4.91 |            7.84 |   9,996 / 10,000 [99.9%, 100.0%] |
| seven_iron_lean5_curve1_rotation2_combined2      | Straight |           137.06 / 139.90 |    -2.96 / 5.74 |    -0.18 / 5.81 |            8.70 |   9,996 / 10,000 [99.9%, 100.0%] |
| seven_iron_lean5_curve1_rotation2_combined2      | Draw     |           137.05 / 139.90 |    -2.98 / 5.69 |    -0.10 / 5.90 |            8.72 |   9,996 / 10,000 [99.9%, 100.0%] |
| seven_iron_lean5_curve1_rotation2_combined2      | Fade     |           137.07 / 139.90 |    -2.95 / 5.80 |    -0.26 / 5.71 |            8.66 |   9,996 / 10,000 [99.9%, 100.0%] |
| seven_iron_lean10_curve1_rotation2_combined2     | Straight |           146.58 / 149.74 |    -3.32 / 6.09 |    -0.21 / 6.74 |            9.67 |   9,996 / 10,000 [99.9%, 100.0%] |
| seven_iron_lean10_curve1_rotation2_combined2     | Draw     |           146.56 / 149.74 |    -3.35 / 6.05 |    -0.10 / 6.84 |            9.72 |   9,996 / 10,000 [99.9%, 100.0%] |
| seven_iron_lean10_curve1_rotation2_combined2     | Fade     |           146.59 / 149.74 |    -3.30 / 6.15 |    -0.31 / 6.61 |            9.61 |   9,996 / 10,000 [99.9%, 100.0%] |

### Delivered Face, Loft, and Simulated Directional Outcomes

| Condition ID                                     | Pattern  | Nominal Face / Path (°) | Mean Delivered Face / Loft (°) | Face–Loft Covariance (°²) | Mean Population |
| ------------------------------------------------ | -------- | ----------------------: | -----------------------------: | ------------------------: | --------------: |
| driver_lean0_curve1_rotation2_combined2          | Straight |           +0.00 / +0.00 |                  +0.00 / 10.00 |                    +1.760 |          10,000 |
| driver_lean0_curve1_rotation2_combined2          | Draw     |           +2.03 / +3.53 |                  +2.03 / 10.00 |                    +1.760 |          10,000 |
| driver_lean0_curve1_rotation2_combined2          | Fade     |           -2.03 / -3.53 |                  -2.03 / 10.00 |                    +1.760 |          10,000 |
| driver_lean5_curve1_rotation2_combined2          | Straight |           +0.00 / +0.00 |                   +0.01 / 5.01 |                    +1.723 |           9,940 |
| driver_lean5_curve1_rotation2_combined2          | Draw     |           +1.29 / +2.79 |                   +1.31 / 5.01 |                    +1.723 |           9,940 |
| driver_lean5_curve1_rotation2_combined2          | Fade     |           -1.29 / -2.79 |                   -1.28 / 5.01 |                    +1.723 |           9,940 |
| driver_lean10_curve1_rotation2_combined2         | Straight |           Not Attempted |                              — |                         — |               — |
| driver_lean10_curve1_rotation2_combined2         | Draw     |           Not Attempted |                              — |                         — |               — |
| driver_lean10_curve1_rotation2_combined2         | Fade     |           Not Attempted |                              — |                         — |               — |
| pitching_wedge_lean0_curve1_rotation2_combined2  | Straight |           +0.00 / +0.00 |                  +0.01 / 46.00 |                    +1.557 |           9,999 |
| pitching_wedge_lean0_curve1_rotation2_combined2  | Draw     |           -0.13 / +1.37 |                  -0.12 / 46.00 |                    +1.557 |           9,999 |
| pitching_wedge_lean0_curve1_rotation2_combined2  | Fade     |           +0.13 / -1.37 |                  +0.14 / 46.00 |                    +1.557 |           9,999 |
| pitching_wedge_lean5_curve1_rotation2_combined2  | Straight |           +0.00 / +0.00 |                  +0.01 / 41.00 |                    +1.433 |           9,999 |
| pitching_wedge_lean5_curve1_rotation2_combined2  | Draw     |           +0.04 / +1.54 |                  +0.04 / 41.00 |                    +1.433 |           9,999 |
| pitching_wedge_lean5_curve1_rotation2_combined2  | Fade     |           -0.04 / -1.54 |                  -0.03 / 41.00 |                    +1.433 |           9,999 |
| pitching_wedge_lean10_curve1_rotation2_combined2 | Straight |           +0.00 / +0.00 |                  +0.00 / 36.00 |                    +1.337 |           9,999 |
| pitching_wedge_lean10_curve1_rotation2_combined2 | Draw     |           +0.21 / +1.71 |                  +0.21 / 36.00 |                    +1.337 |           9,999 |
| pitching_wedge_lean10_curve1_rotation2_combined2 | Fade     |           -0.21 / -1.71 |                  -0.20 / 36.00 |                    +1.337 |           9,999 |
| seven_iron_lean0_curve1_rotation2_combined2      | Straight |           +0.00 / +0.00 |                  +0.01 / 34.00 |                    +1.599 |           9,996 |
| seven_iron_lean0_curve1_rotation2_combined2      | Draw     |           +0.44 / +1.94 |                  +0.45 / 34.00 |                    +1.599 |           9,996 |
| seven_iron_lean0_curve1_rotation2_combined2      | Fade     |           -0.44 / -1.94 |                  -0.44 / 34.00 |                    +1.599 |           9,996 |
| seven_iron_lean5_curve1_rotation2_combined2      | Straight |           +0.00 / +0.00 |                  +0.01 / 29.00 |                    +1.516 |           9,996 |
| seven_iron_lean5_curve1_rotation2_combined2      | Draw     |           +0.67 / +2.17 |                  +0.67 / 29.00 |                    +1.516 |           9,996 |
| seven_iron_lean5_curve1_rotation2_combined2      | Fade     |           -0.67 / -2.17 |                  -0.66 / 29.00 |                    +1.516 |           9,996 |
| seven_iron_lean10_curve1_rotation2_combined2     | Straight |           +0.00 / +0.00 |                  +0.00 / 24.00 |                    +1.451 |           9,996 |
| seven_iron_lean10_curve1_rotation2_combined2     | Draw     |           +0.93 / +2.43 |                  +0.93 / 24.00 |                    +1.451 |           9,996 |
| seven_iron_lean10_curve1_rotation2_combined2     | Fade     |           -0.93 / -2.43 |                  -0.93 / 24.00 |                    +1.451 |           9,996 |

| Condition ID                                     | Pattern  |     Left-Long | Left-Short | Right-Long | Right-Short |
| ------------------------------------------------ | -------- | ------------: | ---------: | ---------: | ----------: |
| driver_lean0_curve1_rotation2_combined2          | Straight |         0.069 |      0.428 |      0.336 |       0.167 |
| driver_lean0_curve1_rotation2_combined2          | Draw     |         0.065 |      0.428 |      0.348 |       0.159 |
| driver_lean0_curve1_rotation2_combined2          | Fade     |         0.071 |      0.430 |      0.328 |       0.171 |
| driver_lean5_curve1_rotation2_combined2          | Straight |         0.072 |      0.419 |      0.388 |       0.121 |
| driver_lean5_curve1_rotation2_combined2          | Draw     |         0.077 |      0.396 |      0.386 |       0.142 |
| driver_lean5_curve1_rotation2_combined2          | Fade     |         0.070 |      0.429 |      0.391 |       0.109 |
| driver_lean10_curve1_rotation2_combined2         | Straight | Not Attempted |          — |          — |           — |
| driver_lean10_curve1_rotation2_combined2         | Draw     | Not Attempted |          — |          — |           — |
| driver_lean10_curve1_rotation2_combined2         | Fade     | Not Attempted |          — |          — |           — |
| pitching_wedge_lean0_curve1_rotation2_combined2  | Straight |         0.243 |      0.255 |      0.093 |       0.409 |
| pitching_wedge_lean0_curve1_rotation2_combined2  | Draw     |         0.240 |      0.255 |      0.095 |       0.410 |
| pitching_wedge_lean0_curve1_rotation2_combined2  | Fade     |         0.248 |      0.253 |      0.091 |       0.408 |
| pitching_wedge_lean5_curve1_rotation2_combined2  | Straight |         0.231 |      0.269 |      0.098 |       0.402 |
| pitching_wedge_lean5_curve1_rotation2_combined2  | Draw     |         0.225 |      0.270 |      0.100 |       0.404 |
| pitching_wedge_lean5_curve1_rotation2_combined2  | Fade     |         0.234 |      0.269 |      0.097 |       0.400 |
| pitching_wedge_lean10_curve1_rotation2_combined2 | Straight |         0.214 |      0.284 |      0.107 |       0.395 |
| pitching_wedge_lean10_curve1_rotation2_combined2 | Draw     |         0.211 |      0.282 |      0.108 |       0.400 |
| pitching_wedge_lean10_curve1_rotation2_combined2 | Fade     |         0.217 |      0.285 |      0.105 |       0.392 |
| seven_iron_lean0_curve1_rotation2_combined2      | Straight |         0.219 |      0.280 |      0.097 |       0.404 |
| seven_iron_lean0_curve1_rotation2_combined2      | Draw     |         0.214 |      0.281 |      0.099 |       0.406 |
| seven_iron_lean0_curve1_rotation2_combined2      | Fade     |         0.223 |      0.280 |      0.095 |       0.402 |
| seven_iron_lean5_curve1_rotation2_combined2      | Straight |         0.202 |      0.296 |      0.106 |       0.396 |
| seven_iron_lean5_curve1_rotation2_combined2      | Draw     |         0.197 |      0.297 |      0.108 |       0.399 |
| seven_iron_lean5_curve1_rotation2_combined2      | Fade     |         0.207 |      0.296 |      0.104 |       0.393 |
| seven_iron_lean10_curve1_rotation2_combined2     | Straight |         0.181 |      0.318 |      0.117 |       0.384 |
| seven_iron_lean10_curve1_rotation2_combined2     | Draw     |         0.176 |      0.318 |      0.121 |       0.385 |
| seven_iron_lean10_curve1_rotation2_combined2     | Fade     |         0.186 |      0.317 |      0.114 |       0.383 |

The rigid geometry predicts face/loft coupling for these stated club setups. Quadrants are simulated endpoint frequencies conditional on supported shots; they do not identify a closed-face mechanism or establish that left shots travel longer. Path, strike, speed and spin can also change the outcome.

### Absolute Course Scoring by Actual Context

| Cell                                             | Course Context                                                                 | Pattern  | Absolute Mean SG (Common Three) | Individual Endpoint Supported / Attempted | Common-Three Scored / Attempted |
| ------------------------------------------------ | ------------------------------------------------------------------------------ | -------- | ------------------------------: | ----------------------------------------: | ------------------------------: |
| driver_lean0_curve1_rotation2_combined2          | tee; tee_fairway_rough; start 400.0 m; hole 400.0 m; fairway half-width 15.0 m | Straight |                          -0.458 |                           10,000 / 10,000 |                 10,000 / 10,000 |
| driver_lean0_curve1_rotation2_combined2          | tee; tee_fairway_rough; start 400.0 m; hole 400.0 m; fairway half-width 15.0 m | Draw     |                         -0.4555 |                           10,000 / 10,000 |                 10,000 / 10,000 |
| driver_lean0_curve1_rotation2_combined2          | tee; tee_fairway_rough; start 400.0 m; hole 400.0 m; fairway half-width 15.0 m | Fade     |                          -0.459 |                           10,000 / 10,000 |                 10,000 / 10,000 |
| driver_lean5_curve1_rotation2_combined2          | tee; tee_fairway_rough; start 400.0 m; hole 400.0 m; fairway half-width 15.0 m | Straight |                          -0.783 |                            9,940 / 10,000 |                  9,940 / 10,000 |
| driver_lean5_curve1_rotation2_combined2          | tee; tee_fairway_rough; start 400.0 m; hole 400.0 m; fairway half-width 15.0 m | Draw     |                          -0.782 |                            9,940 / 10,000 |                  9,940 / 10,000 |
| driver_lean5_curve1_rotation2_combined2          | tee; tee_fairway_rough; start 400.0 m; hole 400.0 m; fairway half-width 15.0 m | Fade     |                          -0.783 |                            9,940 / 10,000 |                  9,940 / 10,000 |
| driver_lean10_curve1_rotation2_combined2         | Not Attempted                                                                  | —        |                               — |                                         — |                               — |
| pitching_wedge_lean0_curve1_rotation2_combined2  | fairway; approach_green_rough; start 77.4 m; green radius 15.0 m               | Straight |                         +0.1011 |                            9,999 / 10,000 |                  9,999 / 10,000 |
| pitching_wedge_lean0_curve1_rotation2_combined2  | fairway; approach_green_rough; start 77.4 m; green radius 15.0 m               | Draw     |                         +0.1016 |                            9,999 / 10,000 |                  9,999 / 10,000 |
| pitching_wedge_lean0_curve1_rotation2_combined2  | fairway; approach_green_rough; start 77.4 m; green radius 15.0 m               | Fade     |                         +0.1008 |                            9,999 / 10,000 |                  9,999 / 10,000 |
| pitching_wedge_lean5_curve1_rotation2_combined2  | fairway; approach_green_rough; start 90.4 m; green radius 15.0 m               | Straight |                         +0.0823 |                            9,999 / 10,000 |                  9,999 / 10,000 |
| pitching_wedge_lean5_curve1_rotation2_combined2  | fairway; approach_green_rough; start 90.4 m; green radius 15.0 m               | Draw     |                         +0.0825 |                            9,999 / 10,000 |                  9,999 / 10,000 |
| pitching_wedge_lean5_curve1_rotation2_combined2  | fairway; approach_green_rough; start 90.4 m; green radius 15.0 m               | Fade     |                         +0.0824 |                            9,999 / 10,000 |                  9,999 / 10,000 |
| pitching_wedge_lean10_curve1_rotation2_combined2 | fairway; approach_green_rough; start 102.8 m; green radius 15.0 m              | Straight |                         +0.0679 |                            9,999 / 10,000 |                  9,999 / 10,000 |
| pitching_wedge_lean10_curve1_rotation2_combined2 | fairway; approach_green_rough; start 102.8 m; green radius 15.0 m              | Draw     |                         +0.0677 |                            9,999 / 10,000 |                  9,999 / 10,000 |
| pitching_wedge_lean10_curve1_rotation2_combined2 | fairway; approach_green_rough; start 102.8 m; green radius 15.0 m              | Fade     |                         +0.0683 |                            9,999 / 10,000 |                  9,999 / 10,000 |
| seven_iron_lean0_curve1_rotation2_combined2      | fairway; approach_green_rough; start 127.9 m; green radius 15.0 m              | Straight |                         +0.0722 |                            9,996 / 10,000 |                  9,996 / 10,000 |
| seven_iron_lean0_curve1_rotation2_combined2      | fairway; approach_green_rough; start 127.9 m; green radius 15.0 m              | Draw     |                         +0.0714 |                            9,996 / 10,000 |                  9,996 / 10,000 |
| seven_iron_lean0_curve1_rotation2_combined2      | fairway; approach_green_rough; start 127.9 m; green radius 15.0 m              | Fade     |                         +0.0733 |                            9,996 / 10,000 |                  9,996 / 10,000 |
| seven_iron_lean5_curve1_rotation2_combined2      | fairway; approach_green_rough; start 139.9 m; green radius 15.0 m              | Straight |                         +0.0736 |                            9,996 / 10,000 |                  9,996 / 10,000 |
| seven_iron_lean5_curve1_rotation2_combined2      | fairway; approach_green_rough; start 139.9 m; green radius 15.0 m              | Draw     |                          +0.072 |                            9,996 / 10,000 |                  9,996 / 10,000 |
| seven_iron_lean5_curve1_rotation2_combined2      | fairway; approach_green_rough; start 139.9 m; green radius 15.0 m              | Fade     |                         +0.0751 |                            9,996 / 10,000 |                  9,996 / 10,000 |
| seven_iron_lean10_curve1_rotation2_combined2     | fairway; approach_green_rough; start 149.7 m; green radius 15.0 m              | Straight |                         +0.0668 |                            9,996 / 10,000 |                  9,996 / 10,000 |
| seven_iron_lean10_curve1_rotation2_combined2     | fairway; approach_green_rough; start 149.7 m; green radius 15.0 m              | Draw     |                         +0.0642 |                            9,996 / 10,000 |                  9,996 / 10,000 |
| seven_iron_lean10_curve1_rotation2_combined2     | fairway; approach_green_rough; start 149.7 m; green radius 15.0 m              | Fade     |                         +0.0706 |                            9,996 / 10,000 |                  9,996 / 10,000 |

Absolute SG is a conditional simulated endpoint score against a historical-tour baseline. Driver tee and iron/wedge approach contexts are different opportunities. The absolute SG mean uses the common three-pattern intersection; individual endpoint support can be larger. No values here are measured player strokes gained or per-round effects.

### Direct Curve and Shaft-Variability Interactions

Only reviewed same-index six-population paired contrasts appear here. A shaft-twist SD of 1° or 2° is an assumed primitive input, not a measured face-angle SD or a player precision estimate. Positive interaction means the named curve gained relative to Straight in condition B versus A; this can differ from a lateral-width ranking because target bias, downrange error, and lie scoring also matter.

**Ordered Contrast Key**

**E01** — shaft_rotation_sd:

A: `original_control/1`; `driver_lean0_curve1_rotation1_shaft_only`.

B: `original_control/1`; `driver_lean0_curve1_rotation2_shaft_only`.

**E02** — curve_strength:

A: `original_control/1`; `driver_lean0_curve1_rotation2_shaft_only`.

B: `original_control/1`; `driver_lean0_curve2_rotation2_shaft_only`.

**E03** — shaft_rotation_sd:

A: `original_control/1`; `driver_lean0_curve1_rotation1_combined2`.

B: `original_control/1`; `driver_lean0_curve1_rotation2_combined2`.

**E04** — curve_strength:

A: `original_control/1`; `driver_lean0_curve1_rotation2_combined2`.

B: `original_control/1`; `driver_lean0_curve2_rotation2_combined2`.

**E05** — shaft_rotation_sd:

A: `original_control/1`; `seven_iron_lean0_curve1_rotation1_shaft_only`.

B: `original_control/1`; `seven_iron_lean0_curve1_rotation2_shaft_only`.

**E06** — curve_strength:

A: `original_control/1`; `seven_iron_lean0_curve1_rotation2_shaft_only`.

B: `original_control/1`; `seven_iron_lean0_curve2_rotation2_shaft_only`.

**E07** — shaft_rotation_sd:

A: `original_control/1`; `seven_iron_lean0_curve1_rotation1_combined2`.

B: `original_control/1`; `seven_iron_lean0_curve1_rotation2_combined2`.

**E08** — curve_strength:

A: `original_control/1`; `seven_iron_lean0_curve1_rotation2_combined2`.

B: `original_control/1`; `seven_iron_lean0_curve2_rotation2_combined2`.

**E09** — shaft_rotation_sd:

A: `original_control/1`; `pitching_wedge_lean0_curve1_rotation1_shaft_only`.

B: `original_control/1`; `pitching_wedge_lean0_curve1_rotation2_shaft_only`.

**E10** — curve_strength:

A: `original_control/1`; `pitching_wedge_lean0_curve1_rotation2_shaft_only`.

B: `pw-primary-face-bounds/1`; `pitching_wedge_lean0_curve2_rotation2_shaft_only`.

**E11** — shaft_rotation_sd:

A: `original_control/1`; `pitching_wedge_lean0_curve1_rotation1_combined2`.

B: `original_control/1`; `pitching_wedge_lean0_curve1_rotation2_combined2`.

**E12** — curve_strength:

A: `original_control/1`; `pitching_wedge_lean0_curve1_rotation2_combined2`.

B: `pw-primary-face-bounds/1`; `pitching_wedge_lean0_curve2_rotation2_combined2`.

| Contrast | Curved Pattern |          B − A Relative SG [95% MC] | Common Six / Attempted |
| -------- | -------------- | ----------------------------------: | ---------------------: |
| E01      | Draw           |      +0.00282 [+0.002346, +0.00329] |        10,000 / 10,000 |
| E01      | Fade           |   -0.001293 [-0.001708, -8.956e-04] |        10,000 / 10,000 |
| E02      | Draw           |      +0.004905 [+0.00438, +0.00545] |        10,000 / 10,000 |
| E02      | Fade           | -2.354e-04 [-5.740e-04, +1.124e-04] |        10,000 / 10,000 |
| E03      | Draw           |       +0.00217 [+0.0016, +0.002706] |        10,000 / 10,000 |
| E03      | Fade           |  -9.699e-04 [-0.001502, -5.028e-04] |        10,000 / 10,000 |
| E04      | Draw           |       +0.00449 [+0.00394, +0.00507] |        10,000 / 10,000 |
| E04      | Fade           | +9.299e-05 [-2.735e-04, +4.642e-04] |        10,000 / 10,000 |
| E05      | Draw           | +6.449e-06 [-7.089e-05, +4.934e-05] |        10,000 / 10,000 |
| E05      | Fade           | +2.000e-05 [-5.979e-05, +1.358e-04] |        10,000 / 10,000 |
| E06      | Draw           | +2.109e-04 [+9.164e-05, +3.578e-04] |        10,000 / 10,000 |
| E06      | Fade           |    +0.001515 [+0.001433, +0.001629] |        10,000 / 10,000 |
| E07      | Draw           | -6.028e-05 [-2.966e-04, +1.540e-04] |         9,996 / 10,000 |
| E07      | Fade           | +2.175e-04 [-7.965e-05, +5.113e-04] |         9,996 / 10,000 |
| E08      | Draw           | -2.476e-04 [-4.948e-04, +2.780e-05] |         9,996 / 10,000 |
| E08      | Fade           |   +0.001183 [+9.817e-04, +0.001365] |         9,996 / 10,000 |
| E09      | Draw           |    +0.001093 [+0.001026, +0.001164] |        10,000 / 10,000 |
| E09      | Fade           |   -0.001037 [-0.001105, -9.711e-04] |        10,000 / 10,000 |
| E10      | Draw           |     +0.00559 [+0.005529, +0.005643] |        10,000 / 10,000 |
| E10      | Fade           |     -0.005479 [-0.00553, -0.005418] |        10,000 / 10,000 |
| E11      | Draw           | +3.893e-04 [+2.185e-04, +5.506e-04] |         9,999 / 10,000 |
| E11      | Fade           | -4.276e-04 [-5.452e-04, -3.150e-04] |         9,999 / 10,000 |
| E12      | Draw           | +5.621e-04 [+4.164e-04, +7.167e-04] |         9,999 / 10,000 |
| E12      | Fade           | -2.935e-04 [-4.911e-04, -8.723e-05] |         9,999 / 10,000 |

B − A is the change in (curved − Straight) SG on the same six-population supported-index intersection, with A and B identified above. It is not a causal distance contribution. Intervals are pointwise Monte Carlo sampling intervals; shared seeded errors correlate cells and do not validate player performance.

### Fixed-Target Across-Lean Bounds Controls

The fixed-target across-lean controls evaluate whether adjusting shaft lean from 0° to 5° alters relative curve scoring when nominal target distance is held constant. These comparisons evaluate solver bounds adapted after original search; original unavailable cells remain unchanged in the complete matrix accounting. Rematched speed/face/path prevents causal claims.

Under original protocols (protocol original_primary/1 and original_control/1), nominal speed is searched within 75–125% of reference speed. To resolve the reviewed 5°-lean fixed-target nominal search, the driver control supplement (protocol driver-control-speed-bounds/1) expanded nominal speed search to 75–130%. Additionally, qualified supplemental protocols (driver-control-speed-bounds/1, pw-control-face-bounds/1, and pw-primary-face-bounds/1) evaluated delivery solutions within an explicit absolute face box of [-6.0°, +6.0°]. These adapted bounds were not preregistered in the original factorial protocol and do not imply physical impossibility outside them.

**Ordered Contrast Key**

**E01** — across_lean:

A: `original_control/1`; `driver_lean0_curve2_rotation2_combined2`.

B: `driver-control-speed-bounds/1`; `driver_lean5_curve2_rotation2_combined2`.

**E02** — across_lean:

A: `pw-primary-face-bounds/1`; `pitching_wedge_lean0_curve2_rotation2_combined2`.

B: `pw-control-face-bounds/1`; `pitching_wedge_lean5_curve2_rotation2_combined2`.

| Contrast | Curved Pattern |          B − A Relative SG [95% MC] | Common Six / Attempted |
| -------- | -------------- | ----------------------------------: | ---------------------: |
| E01      | Draw           |         +0.00598 [+0.00483, +0.007] |         9,940 / 10,000 |
| E01      | Fade           |    -0.001397 [-0.00234, -4.807e-04] |         9,940 / 10,000 |
| E02      | Draw           | -3.513e-04 [-5.603e-04, -1.424e-04] |         9,999 / 10,000 |
| E02      | Fade           | +6.868e-04 [+4.851e-04, +8.859e-04] |         9,999 / 10,000 |

B − A is the change in (curved − Straight) SG on the same six-population supported-index intersection, with A and B identified above. It is not a causal distance contribution. Intervals are pointwise Monte Carlo sampling intervals; shared seeded errors correlate cells and do not validate player performance.

### Conclusions and Geometric Mechanism Synthesis

The following descriptive ranges and interval-direction counts cover only the eight selected zero-lean conditions per club (curve ×1/×2, shaft-twist SD 1°/2°, shaft-only/combined2). Positive SG favors the named curve over Straight. Ranges summarize cell point estimates, not pooled confidence intervals. Shared draws make these cells dependent; interval directions are pointwise Monte Carlo statements, not empirical or multiplicity-adjusted discoveries.

| Club           | Pattern | Selected SG Estimate Range (Strokes) | Positive / Negative / Zero-Spanning 95% Intervals | Common-Three Scored n Range / 10,000 |
| -------------- | ------- | -----------------------------------: | ------------------------------------------------: | -----------------------------------: |
| driver         | Draw    |                +2.691e-05 to +0.0078 |                                         6 / 0 / 2 |               10,000–10,000 / 10,000 |
| driver         | Fade    |               -0.00146 to +2.344e-04 |                                         1 / 4 / 3 |               10,000–10,000 / 10,000 |
| seven_iron     | Draw    |              -0.001356 to -2.883e-04 |                                         0 / 8 / 0 |                9,996–10,000 / 10,000 |
| seven_iron     | Fade    |               +8.934e-04 to +0.00258 |                                         8 / 0 / 0 |                9,996–10,000 / 10,000 |
| pitching_wedge | Draw    |               +9.508e-05 to +0.01112 |                                         6 / 0 / 2 |                9,999–10,000 / 10,000 |
| pitching_wedge | Fade    |               -0.01097 to +2.919e-04 |                                         1 / 6 / 1 |                9,999–10,000 / 10,000 |

The relative scoring pattern depends on club and delivery-noise assumptions; these selected rows do not establish a universal best shape. An interval spanning zero does not establish equivalence. The direct paired interaction tables above answer SD 1→2 and curve ×1→×2 on the same six-population intersection; their intervals must not be obtained by subtracting the marginal intervals summarized here.

For the Straight, curve ×1, shaft-only subset below, nominal targets are matched before noise. The realized downrange errors and correlations use each pattern's individual supported endpoints; SD changes need not preserve mean carry.

| Club           | Shaft-Twist SD (°) | Lateral/Downrange Correlation | Mean Downrange Error (m) | Individual Supported / Attempted |
| -------------- | -----------------: | ----------------------------: | -----------------------: | -------------------------------: |
| driver         |                  1 |                       +0.9788 |                   -0.505 |                  10,000 / 10,000 |
| driver         |                  2 |                       +0.9186 |                   -2.005 |                  10,000 / 10,000 |
| seven_iron     |                  1 |                       -0.9960 |                   -0.055 |                  10,000 / 10,000 |
| seven_iron     |                  2 |                       -0.9841 |                   -0.220 |                  10,000 / 10,000 |
| pitching_wedge |                  1 |                       -0.9982 |                   -0.023 |                  10,000 / 10,000 |
| pitching_wedge |                  2 |                       -0.9930 |                   -0.095 |                  10,000 / 10,000 |

Positive lateral coordinates mean right. A positive lateral/downrange correlation associates rightward errors with longer downrange endpoints; a negative correlation associates leftward errors with longer endpoints. This describes the selected simulated population, not a causal face-angle attribution.

At zero shaft twist, delivered loft is L − lean and the rigid geometry gives d(loft)/d(face) = cot(lie) cos(L − lean)/cos(L), positive for these setups. Opening the face by shaft rotation locally increases delivered loft; closing locally decreases it. Fixed primitive twist SD approximately preserves the first-order loft SD across lean while face SD changes. Launch, speed, spin and aerodynamics determine whether that loft change increases carry. See the [Scientific Design](DESIGN.md#frames-and-rigid-delivery-geometry) and the existing editable [Shot-Pattern Reference](../shot_pattern_analysis.tex).

Fixed-target controls remove nominal range mismatch, but include rematched face, path and speed and retain conditional support. No causal fraction of dispersion or scoring is assigned to distance, lean or curvature. Narrow Monte Carlo intervals exclude assumed head-property and contact-law errors and unmeasured golfer variability.

### Verified Academic and Social Figures

Figure links below come from source-bound inventories and include their actual protocol and population type. Individual cell densities use individually supported endpoints, while paired densities use the shared six-population supported-index intersection; displayed point overlays and overhead paths are sampled, as labeled in each image. A 95% highest-density region is not a confidence interval for the mean.

The verified figure links are grouped by club with their resolved comparison rows:

- [Driver Results and Figure Inventory](REPORT_DRIVER.md)
- [Seven Iron Results and Figure Inventory](REPORT_SEVEN_IRON.md)
- [Pitching Wedge Results and Figure Inventory](REPORT_PITCHING_WEDGE.md)

- [Complete Original Figure Availability: final_figures_v1](final_figures_v1/availability.csv)
- [Complete Original Figure Availability: control_final_figures_v1](control_final_figures_v1/availability.csv)

### Separately Qualified Bounds Supplement

The additive supplement inventories 26 unique planned cells and verified 4 control aliases without duplicating those Monte Carlo shots. It is an adapted bounds protocol after the original bounded-search audit, not part of the original factorial preregistration. Its original-versus-supplement results retain explicit versioned protocol identities, common supported indices, and unavailable conditions. [Full Paired Comparison Inventory](supplement_report_v1/REPORT_SUPPLEMENT_COMPARISONS.csv) and [Supplement Figure Inventory](supplement_report_v1/REPORT_SUPPLEMENT_FIGURES.json) contain every reviewed result and asset.

Supplement publication receipt SHA-256: `035c65b249937a639d0ddfd5bd5df3aef8b46aaae351b98a4c6ae9e29e0b064a`. Parent numerical pipeline receipt SHA-256: `035c65b249937a639d0ddfd5bd5df3aef8b46aaae351b98a4c6ae9e29e0b064a`.

### Zero-Lean Curve and Variability Comparison

This predeclared subset varies shaft-twist SD (1°/2°), curve strength (1×/2×), and isolated versus combined delivery errors for each club. Differences are matched shot by shot against Straight inside each cell. The common paired count may be smaller than every pattern's individual supported count. SG is per simulated opportunity; multiply by 100 for the same hypothetical effect over 100 identical opportunities, not per round.

| Club           | Profile    | Twist SD (°) | Curve | Common Paired n | Draw Δ SG [95%]                           | Fade Δ SG [95%]                     |
| -------------- | ---------- | -----------: | ----: | --------------: | ----------------------------------------- | ----------------------------------- |
| driver         | shaft_only |            1 |     1 |          10,000 | +7.462e-05 [-7.104e-05, +2.190e-04]       | +6.826e-05 [-5.228e-05, +1.705e-04] |
| driver         | combined2  |            1 |     1 |          10,000 | +2.691e-05 [-1.642e-04, +2.204e-04]       | +4.218e-05 [-1.739e-04, +2.646e-04] |
| driver         | shaft_only |            2 |     1 |          10,000 | +0.00289 [+0.00246, +0.00338]             | -0.001225 [-0.00162, -8.629e-04]    |
| driver         | combined2  |            2 |     1 |          10,000 | +0.002195 [+0.001695, +0.002636]          | -9.277e-04 [-0.001382, -5.147e-04]  |
| driver         | shaft_only |            1 |     2 |          10,000 | +3.250e-04 [+1.042e-04, +5.513e-04]       | +2.344e-04 [+5.011e-05, +4.290e-04] |
| driver         | combined2  |            1 |     2 |          10,000 | +5.626e-04 [+2.392e-04, +8.978e-04]       | +9.462e-05 [-2.218e-04, +4.419e-04] |
| driver         | shaft_only |            2 |     2 |          10,000 | +0.0078 [+0.00713, +0.00852]              | -0.00146 [-0.001974, -9.630e-04]    |
| driver         | combined2  |            2 |     2 |          10,000 | +0.00668 [+0.00594, +0.00744]             | -8.347e-04 [-0.001478, -2.621e-04]  |
| seven_iron     | shaft_only |            1 |     1 |          10,000 | -5.056e-04 [-5.137e-04, -4.968e-04]       | +0.001047 [+0.0010332, +0.001059]   |
| seven_iron     | combined2  |            1 |     1 |           9,996 | -7.450e-04 [-8.996e-04, -5.795e-04]       | +8.934e-04 [+7.090e-04, +0.001078]  |
| seven_iron     | shaft_only |            2 |     1 |          10,000 | -4.992e-04 [-5.776e-04, -4.544e-04]       | +0.001067 [+9.861e-04, +0.001182]   |
| seven_iron     | combined2  |            2 |     1 |           9,996 | -8.053e-04 [-0.001038, -5.918e-04]        | +0.00111 [+8.442e-04, +0.001355]    |
| seven_iron     | shaft_only |            1 |     2 |          10,000 | -3.978e-04 [-4.106e-04, -3.851e-04]       | +0.0025526 [+0.00252, +0.00258]     |
| seven_iron     | combined2  |            1 |     2 |           9,996 | -0.001356 [-0.00165, -0.001025]           | +0.001975 [+0.001623, +0.002286]    |
| seven_iron     | shaft_only |            2 |     2 |          10,000 | -2.883e-04 [-4.426e-04, -1.306e-04]       | +0.002582 [+0.002453, +0.002736]    |
| seven_iron     | combined2  |            2 |     2 |           9,996 | -0.001053 [-0.001428, -6.655e-04]         | +0.002294 [+0.00193, +0.002645]     |
| pitching_wedge | shaft_only |            1 |     1 |          10,000 | +0.004434 [+0.004363, +0.004502]          | -0.00445 [-0.004517, -0.00438]      |
| pitching_wedge | combined2  |            1 |     1 |           9,999 | +9.508e-05 [-6.221e-05, +2.800e-04]       | +9.390e-05 [-3.145e-05, +2.411e-04] |
| pitching_wedge | shaft_only |            2 |     1 |          10,000 | +0.005527 [+0.005466, +0.005576]          | -0.005487 [-0.005537, -0.005427]    |
| pitching_wedge | combined2  |            2 |     1 |           9,999 | +4.844e-04 [+3.075e-04, +6.523e-04]       | -3.337e-04 [-5.016e-04, -1.721e-04] |
| pitching_wedge | shaft_only |            1 |     2 |     Unavailable | Nominal Match Unavailable (See Inventory) | —                                   |
| pitching_wedge | combined2  |            1 |     2 |     Unavailable | Nominal Match Unavailable (See Inventory) | —                                   |
| pitching_wedge | shaft_only |            2 |     2 |     Unavailable | Nominal Match Unavailable (See Inventory) | —                                   |
| pitching_wedge | combined2  |            2 |     2 |     Unavailable | Nominal Match Unavailable (See Inventory) | —                                   |

For the same selected combined-2, 2° shaft variation, curve scale 1 cells, paired radial error gives the geometric endpoint comparison independently of the hypothetical score:

| Club           | Common Paired n | Draw Δ Radial Error (m) [95%] | Fade Δ Radial Error (m) [95%] |
| -------------- | --------------: | ----------------------------- | ----------------------------- |
| driver         |          10,000 | +0.3023 [+0.2945, +0.3088]    | -0.3236 [-0.3304, -0.3159]    |
| seven_iron     |           9,996 | +0.00927 [+0.00811, +0.01057] | -0.01391 [-0.01522, -0.01276] |
| pitching_wedge |           9,999 | -0.01728 [-0.01855, -0.01602] | +0.01672 [+0.01546, +0.01798] |

### Resolved Zero-Lean Curve and Variability Comparison

The complete condition keys and paired scoring tables are preserved in the club appendices. C01–C24 retain their original global identifiers.

- [Driver resolved results (C01–C08)](REPORT_DRIVER.md)
- [Seven iron resolved results (C09–C16)](REPORT_SEVEN_IRON.md)
- [Pitching wedge resolved results (C17–C24)](REPORT_PITCHING_WEDGE.md)

### Primary Matrix Direction Counts

Counts below cover only 270 completed cells; 54 unavailable planned cells have no interval and remain in the complete inventory.

| Metric                           | Pattern | Positive Interval | Negative Interval | Crosses Zero | Missing Metric in Completed Cells |
| -------------------------------- | ------- | ----------------: | ----------------: | -----------: | --------------------------------: |
| Mean Radial Error Difference     | Draw    |               175 |                89 |            6 |                                 0 |
| Mean Radial Error Difference     | Fade    |                84 |               180 |            6 |                                 0 |
| Paired Strokes Gained Difference | Draw    |               125 |               120 |           25 |                                 0 |
| Paired Strokes Gained Difference | Fade    |               177 |                72 |           21 |                                 0 |

A direction count describes this finite grid of assumed cells. It is not a pooled effect, an independent replication count, or a multiple-comparison-adjusted discovery.

## Across-Lean Control Cell Accounting

The [Control Matrix Plan](control_final_v1/matrix_plan.json) fixes target distance to the zero-lean reference. Unavailable cells are retained and named in the complete CSV inventory. This does not by itself identify a pure lean effect because the optimizer rematches delivery inputs. Driver lean can shorten or eliminate a physically airborne nominal; all comparisons are conditioned on successful nominal matching under the stated bounds.

### Control Matrix Direction Counts

Counts below cover only 52 completed cells; 20 unavailable planned cells have no interval and remain in the complete inventory.

| Metric                           | Pattern | Positive Interval | Negative Interval | Crosses Zero | Missing Metric in Completed Cells |
| -------------------------------- | ------- | ----------------: | ----------------: | -----------: | --------------------------------: |
| Mean Radial Error Difference     | Draw    |                36 |                16 |            0 |                                 0 |
| Mean Radial Error Difference     | Fade    |                14 |                36 |            2 |                                 0 |
| Paired Strokes Gained Difference | Draw    |                20 |                24 |            8 |                                 0 |
| Paired Strokes Gained Difference | Fade    |                32 |                15 |            5 |                                 0 |

A direction count describes this finite grid of assumed cells. It is not a pooled effect, an independent replication count, or a multiple-comparison-adjusted discovery.

## Across-Lean Fixed-Target Control

The fixed-target control planned 72 lean pairs: 20 fully supported, 20 conditional, and 32 unavailable. All comparisons use the intersection of supported shot indices across six physical populations: three patterns at each of two lean settings. Lean differences include rematched nominal speed, face, and path; they are not an isolated causal effect of shaft lean.

The complete paired comparison inventory, including every unavailable pair, is in [REPORT_LEAN.csv](REPORT_LEAN.csv). The compact table below shows combined-2, 2° shaft variation, curve scale 1, for each club. Positive differences mean the later lean is higher; paired SG and radial error have different desirable signs.

| Club           | Lean Pair (°) | Common n | Pattern     | Δ Radial Error (m) [95%]      | Δ SG [95%]                    |
| -------------- | ------------- | -------: | ----------- | ----------------------------- | ----------------------------- |
| driver         | 0→5           |    9,940 | Straight    | +10.980 [+10.730, +11.221]    | -0.01824 [-0.0197, -0.01658]  |
| driver         | 0→5           |    9,940 | Draw        | +12.597 [+12.322, +12.858]    | -0.01646 [-0.01806, -0.01464] |
| driver         | 0→5           |    9,940 | Fade        | +9.431 [+9.214, +9.640]       | -0.0198 [-0.02116, -0.0182]   |
| driver         | 0→10          |        — | Unavailable | one_or_both_cells_unavailable | —                             |
| driver         | 5→10          |        — | Unavailable | one_or_both_cells_unavailable | —                             |
| seven_iron     | 0→5           |    9,996 | Straight    | +0.4006 [+0.3893, +0.4111]    | -0.02405 [-0.02508, -0.02298] |
| seven_iron     | 0→5           |    9,996 | Draw        | +0.4062 [+0.395, +0.4167]     | -0.02414 [-0.02513, -0.02313] |
| seven_iron     | 0→5           |    9,996 | Fade        | +0.393 [+0.3816, +0.4034]     | -0.02362 [-0.02464, -0.02253] |
| seven_iron     | 0→10          |    9,996 | Straight    | +0.9187 [+0.896, +0.9405]     | -0.05347 [-0.05537, -0.05153] |
| seven_iron     | 0→10          |    9,996 | Draw        | +0.935 [+0.9125, +0.9573]     | -0.05434 [-0.05628, -0.05243] |
| seven_iron     | 0→10          |    9,996 | Fade        | +0.897 [+0.8743, +0.919]      | -0.0527 [-0.05474, -0.05073]  |
| seven_iron     | 5→10          |    9,996 | Straight    | +0.518 [+0.5066, +0.5298]     | -0.02941 [-0.03064, -0.02822] |
| seven_iron     | 5→10          |    9,996 | Draw        | +0.529 [+0.5172, +0.5409]     | -0.0302 [-0.03142, -0.02904]  |
| seven_iron     | 5→10          |    9,996 | Fade        | +0.5038 [+0.4924, +0.5156]    | -0.02908 [-0.0304, -0.02794]  |
| pitching_wedge | 0→5           |    9,999 | Straight    | +0.223 [+0.212, +0.2324]      | -0.01895 [-0.01994, -0.01792] |
| pitching_wedge | 0→5           |    9,999 | Draw        | +0.2277 [+0.2169, +0.2371]    | -0.01915 [-0.02013, -0.01812] |
| pitching_wedge | 0→5           |    9,999 | Fade        | +0.2176 [+0.2068, +0.227]     | -0.01864 [-0.01962, -0.0176]  |
| pitching_wedge | 0→10          |    9,999 | Straight    | +0.4933 [+0.4714, +0.512]     | -0.03934 [-0.04124, -0.0375]  |
| pitching_wedge | 0→10          |    9,999 | Draw        | +0.5026 [+0.481, +0.5213]     | -0.03992 [-0.0418, -0.03806]  |
| pitching_wedge | 0→10          |    9,999 | Fade        | +0.4828 [+0.4608, +0.502]     | -0.03876 [-0.04068, -0.0369]  |
| pitching_wedge | 5→10          |    9,999 | Straight    | +0.2704 [+0.259, +0.280]      | -0.0204 [-0.0213, -0.01943]   |
| pitching_wedge | 5→10          |    9,999 | Draw        | +0.275 [+0.2636, +0.2844]     | -0.02077 [-0.02164, -0.01984] |
| pitching_wedge | 5→10          |    9,999 | Fade        | +0.2652 [+0.2538, +0.2748]    | -0.02012 [-0.02103, -0.01915] |

## Course-Dimension Sensitivity

Each case re-scores the same validated, commonly supported endpoints from lean-0, curve-scale-1, shaft-SD-1°, combined-2 cells; no new flights are simulated. Driver dimensions are fairway half-width and hole distance. Iron and wedge dimensions are green radius. Changes in score with course dimensions do not change the flight distribution.

| Club           | Scenario        | Paired n | Draw SG − Straight [95%]            | Fade SG − Straight [95%]            |
| -------------- | --------------- | -------: | ----------------------------------- | ----------------------------------- |
| driver         | width10_hole350 |   10,000 | +0.001707 [+0.001144, +0.00226]     | -6.568e-04 [-0.001238, -1.308e-04]  |
| driver         | width10_hole400 |   10,000 | +0.001348 [+9.025e-04, +0.001787]   | -4.611e-04 [-9.092e-04, -4.579e-05] |
| driver         | width10_hole450 |   10,000 | +0.00111 [+8.064e-04, +0.001412]    | -5.059e-04 [-8.136e-04, -2.189e-04] |
| driver         | width15_hole350 |   10,000 | -6.451e-06 [-2.371e-04, +2.428e-04] | +3.009e-05 [-2.460e-04, +3.107e-04] |
| driver         | width15_hole400 |   10,000 | +2.691e-05 [-1.642e-04, +2.204e-04] | +4.218e-05 [-1.739e-04, +2.646e-04] |
| driver         | width15_hole450 |   10,000 | +2.309e-04 [+1.054e-04, +3.662e-04] | -1.712e-04 [-3.189e-04, -1.870e-05] |
| driver         | width20_hole350 |   10,000 | -2.014e-04 [-3.201e-04, -7.197e-05] | +3.146e-04 [+1.762e-04, +4.526e-04] |
| driver         | width20_hole400 |   10,000 | -1.016e-04 [-2.141e-04, +1.252e-05] | +2.386e-04 [+1.181e-04, +3.622e-04] |
| driver         | width20_hole450 |   10,000 | +1.424e-04 [+7.159e-05, +2.176e-04] | -3.593e-05 [-1.165e-04, +3.960e-05] |
| pitching_wedge | radius10        |    9,999 | +1.673e-04 [-5.909e-05, +4.235e-04] | +5.642e-05 [-1.532e-04, +2.733e-04] |
| pitching_wedge | radius15        |    9,999 | +9.508e-05 [-6.221e-05, +2.800e-04] | +9.390e-05 [-3.145e-05, +2.411e-04] |
| pitching_wedge | radius20        |    9,999 | +2.078e-05 [-1.213e-04, +1.422e-04] | +9.397e-05 [-3.137e-05, +2.412e-04] |
| seven_iron     | radius10        |    9,996 | -8.173e-04 [-0.001117, -5.177e-04]  | +9.299e-04 [+6.219e-04, +0.001276]  |
| seven_iron     | radius15        |    9,996 | -7.450e-04 [-8.996e-04, -5.795e-04] | +8.934e-04 [+7.090e-04, +0.001078]  |
| seven_iron     | radius20        |    9,996 | -7.823e-04 [-9.184e-04, -6.568e-04] | +9.323e-04 [+7.836e-04, +0.001093]  |

## Reproducibility and Limitations

Primary plan SHA-256: `ae84ed2ac2ee62bb9be3e0742d1c1d447b8a21e40493c74dfa27a03db4ac8ce0`. Control plan SHA-256: `ed182a08a2b2afa80b6e726b3c0f5730c7f19001791c3f22ad2d3d4429cb8d9d`. Source and native-binary identities, numerical settings, seeds, and artifact hashes are frozen in each `run_start.json`, `receipt.json`, and `manifest.json`; the cell loader validates them before aggregation. Review the [Independent Validation Record](validation/README.md) and [Final Run Acceptance](FINAL_RUN_ACCEPTANCE.md).

Companion inventory SHA-256: `REPORT_CELLS.csv` `f1f5abf8d256e62982506a8085c3132f3f15777fdb2caa734ed430dfeeea5042`; `REPORT_LEAN.csv` `baa74e7792a10a245c940ea2c355dd0bc8a4049f51e18056744f5a450dcbd0a7`. These appendices preserve all planned and unavailable rows while the main report selects readable comparisons.

An interval excluding zero is not empirical proof of a golfer's best shot shape. Course SG relies on a historical, source-backed but approximate baseline and assumed green/fairway geometry. Unsupported terminal lies prevent full-population SG; conditional estimates use common supported indices. Absolute carry and across-lean target feasibility constrain interpretation, especially at 10° driver lean.

## Rebuild Commands

```bash
python -m scripts.research.shot_pattern_analysis.build_final_report docs/research/shot_pattern_analysis/phase2/final_v1 docs/research/shot_pattern_analysis/phase2/control_final_v1 docs/research/shot_pattern_analysis/phase2/REPORT.md
python -m scripts.research.shot_pattern_analysis.build_pdf docs/research/shot_pattern_analysis/phase2/REPORT.md docs/research/shot_pattern_analysis/phase2/REPORT.pdf
```
