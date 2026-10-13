# Partial Report Composition Preview

Layout diagnostic only: selected numerical sections, not the complete final report or publication acceptance.

## Selected Numerical Sections

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

### Resolved Zero-Lean Curve and Variability Comparison

This resolved view integrates the qualified supplemental face-bounds protocol (pw-primary-face-bounds/1) for the four strong-curve pitching wedge conditions while preserving original primary protocol accounting (original_primary/1) above. Differences are matched shot by shot against Straight inside each cell. The common paired count may be smaller than every pattern's individual supported count. SG is per simulated opportunity; multiply by 100 for the same hypothetical effect over 100 identical opportunities, not per round.

| Club           | Profile    | Twist SD (°) | Curve | Protocol                 | Individual Supported / Attempted (Straight; Draw; Fade) | Common-Three Scored / Attempted | Draw Δ SG [95%]                     | Fade Δ SG [95%]                     |
| -------------- | ---------- | -----------: | ----: | ------------------------ | ------------------------------------------------------- | ------------------------------: | ----------------------------------- | ----------------------------------- |
| driver         | shaft_only |            1 |     1 | original_primary/1       | 10,000 / 10,000; 10,000 / 10,000; 10,000 / 10,000       |                 10,000 / 10,000 | +7.462e-05 [-7.104e-05, +2.190e-04] | +6.826e-05 [-5.228e-05, +1.705e-04] |
| driver         | combined2  |            1 |     1 | original_primary/1       | 10,000 / 10,000; 10,000 / 10,000; 10,000 / 10,000       |                 10,000 / 10,000 | +2.691e-05 [-1.642e-04, +2.204e-04] | +4.218e-05 [-1.739e-04, +2.646e-04] |
| driver         | shaft_only |            2 |     1 | original_primary/1       | 10,000 / 10,000; 10,000 / 10,000; 10,000 / 10,000       |                 10,000 / 10,000 | +0.00289 [+0.00246, +0.00338]       | -0.001225 [-0.00162, -8.629e-04]    |
| driver         | combined2  |            2 |     1 | original_primary/1       | 10,000 / 10,000; 10,000 / 10,000; 10,000 / 10,000       |                 10,000 / 10,000 | +0.002195 [+0.001695, +0.002636]    | -9.277e-04 [-0.001382, -5.147e-04]  |
| driver         | shaft_only |            1 |     2 | original_primary/1       | 10,000 / 10,000; 10,000 / 10,000; 10,000 / 10,000       |                 10,000 / 10,000 | +3.250e-04 [+1.042e-04, +5.513e-04] | +2.344e-04 [+5.011e-05, +4.290e-04] |
| driver         | combined2  |            1 |     2 | original_primary/1       | 10,000 / 10,000; 10,000 / 10,000; 10,000 / 10,000       |                 10,000 / 10,000 | +5.626e-04 [+2.392e-04, +8.978e-04] | +9.462e-05 [-2.218e-04, +4.419e-04] |
| driver         | shaft_only |            2 |     2 | original_primary/1       | 10,000 / 10,000; 10,000 / 10,000; 10,000 / 10,000       |                 10,000 / 10,000 | +0.0078 [+0.00713, +0.00852]        | -0.00146 [-0.001974, -9.630e-04]    |
| driver         | combined2  |            2 |     2 | original_primary/1       | 10,000 / 10,000; 10,000 / 10,000; 10,000 / 10,000       |                 10,000 / 10,000 | +0.00668 [+0.00594, +0.00744]       | -8.347e-04 [-0.001478, -2.621e-04]  |
| seven_iron     | shaft_only |            1 |     1 | original_primary/1       | 10,000 / 10,000; 10,000 / 10,000; 10,000 / 10,000       |                 10,000 / 10,000 | -5.056e-04 [-5.137e-04, -4.968e-04] | +0.001047 [+0.0010332, +0.001059]   |
| seven_iron     | combined2  |            1 |     1 | original_primary/1       | 9,996 / 10,000; 9,996 / 10,000; 9,996 / 10,000          |                  9,996 / 10,000 | -7.450e-04 [-8.996e-04, -5.795e-04] | +8.934e-04 [+7.090e-04, +0.001078]  |
| seven_iron     | shaft_only |            2 |     1 | original_primary/1       | 10,000 / 10,000; 10,000 / 10,000; 10,000 / 10,000       |                 10,000 / 10,000 | -4.992e-04 [-5.776e-04, -4.544e-04] | +0.001067 [+9.861e-04, +0.001182]   |
| seven_iron     | combined2  |            2 |     1 | original_primary/1       | 9,996 / 10,000; 9,996 / 10,000; 9,996 / 10,000          |                  9,996 / 10,000 | -8.053e-04 [-0.001038, -5.918e-04]  | +0.00111 [+8.442e-04, +0.001355]    |
| seven_iron     | shaft_only |            1 |     2 | original_primary/1       | 10,000 / 10,000; 10,000 / 10,000; 10,000 / 10,000       |                 10,000 / 10,000 | -3.978e-04 [-4.106e-04, -3.851e-04] | +0.0025526 [+0.00252, +0.00258]     |
| seven_iron     | combined2  |            1 |     2 | original_primary/1       | 9,996 / 10,000; 9,996 / 10,000; 9,996 / 10,000          |                  9,996 / 10,000 | -0.001356 [-0.00165, -0.001025]     | +0.001975 [+0.001623, +0.002286]    |
| seven_iron     | shaft_only |            2 |     2 | original_primary/1       | 10,000 / 10,000; 10,000 / 10,000; 10,000 / 10,000       |                 10,000 / 10,000 | -2.883e-04 [-4.426e-04, -1.306e-04] | +0.002582 [+0.002453, +0.002736]    |
| seven_iron     | combined2  |            2 |     2 | original_primary/1       | 9,996 / 10,000; 9,996 / 10,000; 9,996 / 10,000          |                  9,996 / 10,000 | -0.001053 [-0.001428, -6.655e-04]   | +0.002294 [+0.00193, +0.002645]     |
| pitching_wedge | shaft_only |            1 |     1 | original_primary/1       | 10,000 / 10,000; 10,000 / 10,000; 10,000 / 10,000       |                 10,000 / 10,000 | +0.004434 [+0.004363, +0.004502]    | -0.00445 [-0.004517, -0.00438]      |
| pitching_wedge | combined2  |            1 |     1 | original_primary/1       | 9,999 / 10,000; 9,999 / 10,000; 9,999 / 10,000          |                  9,999 / 10,000 | +9.508e-05 [-6.221e-05, +2.800e-04] | +9.390e-05 [-3.145e-05, +2.411e-04] |
| pitching_wedge | shaft_only |            2 |     1 | original_primary/1       | 10,000 / 10,000; 10,000 / 10,000; 10,000 / 10,000       |                 10,000 / 10,000 | +0.005527 [+0.005466, +0.005576]    | -0.005487 [-0.005537, -0.005427]    |
| pitching_wedge | combined2  |            2 |     1 | original_primary/1       | 9,999 / 10,000; 9,999 / 10,000; 9,999 / 10,000          |                  9,999 / 10,000 | +4.844e-04 [+3.075e-04, +6.523e-04] | -3.337e-04 [-5.016e-04, -1.721e-04] |
| pitching_wedge | shaft_only |            1 |     2 | pw-primary-face-bounds/1 | 10,000 / 10,000; 10,000 / 10,000; 10,000 / 10,000       |                 10,000 / 10,000 | +0.008878 [+0.008738, +0.009015]    | -0.008934 [-0.009068, -0.008798]    |
| pitching_wedge | combined2  |            1 |     2 | pw-primary-face-bounds/1 | 9,999 / 10,000; 9,999 / 10,000; 9,999 / 10,000          |                  9,999 / 10,000 | +2.513e-04 [-3.605e-05, +5.326e-04] | +2.919e-04 [+3.620e-05, +5.922e-04] |
| pitching_wedge | shaft_only |            2 |     2 | pw-primary-face-bounds/1 | 10,000 / 10,000; 10,000 / 10,000; 10,000 / 10,000       |                 10,000 / 10,000 | +0.011117 [+0.010994, +0.01122]     | -0.010966 [-0.011068, -0.010844]    |
| pitching_wedge | combined2  |            2 |     2 | pw-primary-face-bounds/1 | 9,999 / 10,000; 9,999 / 10,000; 9,999 / 10,000          |                  9,999 / 10,000 | +0.001046 [+7.406e-04, +0.001332]   | -6.272e-04 [-9.681e-04, -2.936e-04] |

| Contrast          | Curved Pattern | B − A Relative SG (95% MC Interval) | Common Six / Attempted | Source Conditions (A → B)                                                                                                                         |
| ----------------- | -------------- | ----------------------------------: | ---------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| shaft_rotation_sd | Draw           |      +0.00282 [+0.002346, +0.00329] |        10,000 / 10,000 | original_control/1: driver_lean0_curve1_rotation1_shaft_only → original_control/1: driver_lean0_curve1_rotation2_shaft_only                       |
| shaft_rotation_sd | Fade           |   -0.001293 [-0.001708, -8.956e-04] |        10,000 / 10,000 | original_control/1: driver_lean0_curve1_rotation1_shaft_only → original_control/1: driver_lean0_curve1_rotation2_shaft_only                       |
| curve_strength    | Draw           |      +0.004905 [+0.00438, +0.00545] |        10,000 / 10,000 | original_control/1: driver_lean0_curve1_rotation2_shaft_only → original_control/1: driver_lean0_curve2_rotation2_shaft_only                       |
| curve_strength    | Fade           | -2.354e-04 [-5.740e-04, +1.124e-04] |        10,000 / 10,000 | original_control/1: driver_lean0_curve1_rotation2_shaft_only → original_control/1: driver_lean0_curve2_rotation2_shaft_only                       |
| shaft_rotation_sd | Draw           |       +0.00217 [+0.0016, +0.002706] |        10,000 / 10,000 | original_control/1: driver_lean0_curve1_rotation1_combined2 → original_control/1: driver_lean0_curve1_rotation2_combined2                         |
| shaft_rotation_sd | Fade           |  -9.699e-04 [-0.001502, -5.028e-04] |        10,000 / 10,000 | original_control/1: driver_lean0_curve1_rotation1_combined2 → original_control/1: driver_lean0_curve1_rotation2_combined2                         |
| curve_strength    | Draw           |       +0.00449 [+0.00394, +0.00507] |        10,000 / 10,000 | original_control/1: driver_lean0_curve1_rotation2_combined2 → original_control/1: driver_lean0_curve2_rotation2_combined2                         |
| curve_strength    | Fade           | +9.299e-05 [-2.735e-04, +4.642e-04] |        10,000 / 10,000 | original_control/1: driver_lean0_curve1_rotation2_combined2 → original_control/1: driver_lean0_curve2_rotation2_combined2                         |
| shaft_rotation_sd | Draw           | +6.449e-06 [-7.089e-05, +4.934e-05] |        10,000 / 10,000 | original_control/1: seven_iron_lean0_curve1_rotation1_shaft_only → original_control/1: seven_iron_lean0_curve1_rotation2_shaft_only               |
| shaft_rotation_sd | Fade           | +2.000e-05 [-5.979e-05, +1.358e-04] |        10,000 / 10,000 | original_control/1: seven_iron_lean0_curve1_rotation1_shaft_only → original_control/1: seven_iron_lean0_curve1_rotation2_shaft_only               |
| curve_strength    | Draw           | +2.109e-04 [+9.164e-05, +3.578e-04] |        10,000 / 10,000 | original_control/1: seven_iron_lean0_curve1_rotation2_shaft_only → original_control/1: seven_iron_lean0_curve2_rotation2_shaft_only               |
| curve_strength    | Fade           |    +0.001515 [+0.001433, +0.001629] |        10,000 / 10,000 | original_control/1: seven_iron_lean0_curve1_rotation2_shaft_only → original_control/1: seven_iron_lean0_curve2_rotation2_shaft_only               |
| shaft_rotation_sd | Draw           | -6.028e-05 [-2.966e-04, +1.540e-04] |         9,996 / 10,000 | original_control/1: seven_iron_lean0_curve1_rotation1_combined2 → original_control/1: seven_iron_lean0_curve1_rotation2_combined2                 |
| shaft_rotation_sd | Fade           | +2.175e-04 [-7.965e-05, +5.113e-04] |         9,996 / 10,000 | original_control/1: seven_iron_lean0_curve1_rotation1_combined2 → original_control/1: seven_iron_lean0_curve1_rotation2_combined2                 |
| curve_strength    | Draw           | -2.476e-04 [-4.948e-04, +2.780e-05] |         9,996 / 10,000 | original_control/1: seven_iron_lean0_curve1_rotation2_combined2 → original_control/1: seven_iron_lean0_curve2_rotation2_combined2                 |
| curve_strength    | Fade           |   +0.001183 [+9.817e-04, +0.001365] |         9,996 / 10,000 | original_control/1: seven_iron_lean0_curve1_rotation2_combined2 → original_control/1: seven_iron_lean0_curve2_rotation2_combined2                 |
| shaft_rotation_sd | Draw           |    +0.001093 [+0.001026, +0.001164] |        10,000 / 10,000 | original_control/1: pitching_wedge_lean0_curve1_rotation1_shaft_only → original_control/1: pitching_wedge_lean0_curve1_rotation2_shaft_only       |
| shaft_rotation_sd | Fade           |   -0.001037 [-0.001105, -9.711e-04] |        10,000 / 10,000 | original_control/1: pitching_wedge_lean0_curve1_rotation1_shaft_only → original_control/1: pitching_wedge_lean0_curve1_rotation2_shaft_only       |
| curve_strength    | Draw           |     +0.00559 [+0.005529, +0.005643] |        10,000 / 10,000 | original_control/1: pitching_wedge_lean0_curve1_rotation2_shaft_only → pw-primary-face-bounds/1: pitching_wedge_lean0_curve2_rotation2_shaft_only |
| curve_strength    | Fade           |     -0.005479 [-0.00553, -0.005418] |        10,000 / 10,000 | original_control/1: pitching_wedge_lean0_curve1_rotation2_shaft_only → pw-primary-face-bounds/1: pitching_wedge_lean0_curve2_rotation2_shaft_only |
| shaft_rotation_sd | Draw           | +3.893e-04 [+2.185e-04, +5.506e-04] |         9,999 / 10,000 | original_control/1: pitching_wedge_lean0_curve1_rotation1_combined2 → original_control/1: pitching_wedge_lean0_curve1_rotation2_combined2         |
| shaft_rotation_sd | Fade           | -4.276e-04 [-5.452e-04, -3.150e-04] |         9,999 / 10,000 | original_control/1: pitching_wedge_lean0_curve1_rotation1_combined2 → original_control/1: pitching_wedge_lean0_curve1_rotation2_combined2         |
| curve_strength    | Draw           | +5.621e-04 [+4.164e-04, +7.167e-04] |         9,999 / 10,000 | original_control/1: pitching_wedge_lean0_curve1_rotation2_combined2 → pw-primary-face-bounds/1: pitching_wedge_lean0_curve2_rotation2_combined2   |
| curve_strength    | Fade           | -2.935e-04 [-4.911e-04, -8.723e-05] |         9,999 / 10,000 | original_control/1: pitching_wedge_lean0_curve1_rotation2_combined2 → pw-primary-face-bounds/1: pitching_wedge_lean0_curve2_rotation2_combined2   |

B − A is the change in (curved − Straight) SG on the same six-population supported-index intersection. It is not a causal distance contribution. Intervals are pointwise Monte Carlo sampling intervals; shared seeded errors correlate cells and do not validate player performance.
