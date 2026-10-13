# Superseded Historical Control Report

The following record preserves the original analysis and its numerical outputs. It is not the corrected-model conclusion. The missing tangential impulse invalidated its use as the final shape comparison. See [Current Study](README.md).

# Straight, Draw, and Fade: Shot Pattern Analysis

**Status:** The four original driver result bundles below are superseded
impact-approximation controls. Independent review found a missing tangential
translational impulse. Corrected driver, 7-iron, and wedge experiments are in
progress; the historical precision and scoring numbers are not final conclusions.

This experiment compares 10,000 shots per pattern with independent normally
distributed face errors at both 1° and 2° standard deviation and fixed path. It is a
conditional simulation study, not a measured golfer study or physics qualification.

## Expanded Goal and Acceptance

The active goal now includes driver, 7-iron, and wedge cases with
club-appropriate tee/approach scoring, an independent Astra review, and
correction of the inconsistent impact impulse. It also includes face–loft coupling, rotation about the shaft,
shaft-lean sensitivity, and the observed long-left/short-right pattern. The
fixed-delivered-loft experiments below are retained as controls; their results
must not be generalized to players whose face error covaries with delivered
loft, attack angle, speed, or strike. The analysis is complete only after:

- The 3D rotation geometry, frames, and shaft-lean effects are derived and tested.
- Coupled delivery trials preserve matched random errors and report actual delivered loft.
- Carry–lateral covariance and long-left/short-right behavior are quantified rather than assumed.
- Carry-versus-loft sensitivity is checked for both low- and higher-loft cases.
- Dispersion, equal-range controls, approach scoring, graphics, and tile controls reflect the expanded model.
- Implementation errors are corrected with failing tests first; unresolved physical validation remains explicit.
- Every finding is tracked in the [Analysis Issue Register](issues/README.md), with GitHub publication and closure reported truthfully.

## Corrected Deterministic Carry Sensitivity

The [deterministic sensitivity receipt](corrected_sensitivity.json) uses the
corrected central-contact physics and assumed shaft rotation. At zero nominal
lean, closing by 2° lowers delivered loft in all three presets, but carry responds
differently:

| Illustrative Club | Closed-Face Loft | Closed-Face Carry Change | Open-Face Loft | Open-Face Carry Change |
| --- | ---: | ---: | ---: | ---: |
| Driver | 11.572° | −6.746 m | 14.022° | +2.423 m |
| 7-Iron | 22.977° | +1.468 m | 25.015° | −1.961 m |
| Pitching Wedge | 35.719° | +2.223 m | 37.669° | −2.456 m |

These are deterministic face-error ±2° sensitivities around straight delivery,
not averages over the 10,000-shot populations. The iron and PW examples show
long-left/short-right behavior. The low-loft driver example shows short-left/
long-right behavior: de-lofting lowers launch enough to reduce carry. Roll is
excluded, so these results do not describe total distance. Exact artifact values
and source hashes govern the rounded values displayed here.

Face closure does not uniquely determine loft. The effect depends on the axis
of rotation. The primary coupled mode assumes rotation about a rigid shaft;
vertical-axis yaw instead preserves loft. Shaft lean changes the shaft-axis
coupling, while pitching an otherwise fixed club also changes nominal loft.
Those are separate controls in the receipt. No measured player covariance is
available, and manufacturer static lie is only an assumed delivered shaft
orientation.

## Experiment Definition

| Pattern | Mean Face Relative to Target | Fixed Path Relative to Target |
| --- | ---: | ---: |
| Straight | 0° | 0° |
| Draw | +1.5° | +3° |
| Fade | −1.5° | −3° |

The larger-curve trials double both means: draw face +3° / path +6°, and
fade face −3° / path −6°. Straight remains 0°/0°. Each magnitude is evaluated
at face SD 1° and 2° with the same paired random errors.

Positive means right for a right-handed golfer. The draw face is 1.5° left of
the path. All three patterns receive the same 10,000 random face errors,
using seed 20261008. The illustrative driver delivery fixes speed at 45 m/s,
dynamic loft at 10.9°, attack at 0°, and strike at center; the atmosphere is
the default sea-level case without wind. Bounce and roll are excluded.
CSV face, path, launch azimuth and lateral position use right-positive signs.
The diagnostic `spin_axis_tilt_deg` retains the native solver's positive tilt
toward draw/left curvature; it is not Trackman's right-positive tilt convention.

Raw results preserve the specified target-relative angles. The aimed comparison
rotates each pattern by a single angle calculated from its zero-error landing.
It uses the straight nominal carry as the fixed target distance and a 15 m
radius target circle. No per-shot aiming or sample-mean correction is applied.

## Fixed-Loft Control Results and Practical Meaning

After each pattern receives its single nominal aiming adjustment:

| Measure | Straight | Draw | Fade |
| --- | ---: | ---: | ---: |
| Shots | 10,000 | 10,000 | 10,000 |
| Lateral Standard Deviation (m) | 9.677 | 9.585 | 9.583 |
| Lateral Variance (m²) | 93.643 | 91.870 | 91.829 |
| Central 90% Lateral Width (m) | 31.799 | 31.499 | 31.482 |
| Landings Within 15 m of Target | 87.56% | 87.88% | 87.72% |
| Target RMSE (m) | 9.704 | 9.652 | 9.651 |
| Mean Carry (m) | 195.154 | 194.749 | 194.746 |
| Carry Standard Deviation (m) | 0.255 | 0.597 | 0.601 |

Curving reduces lateral standard deviation by only 9–10 cm, about 1%, and
central 90% width by about 30 cm. The draw improves target-hit fraction by
0.32 percentage points and fade by 0.16 points in this sample. Curved shots
also have approximately 0.41 m less mean carry and more carry variability.
These are small modeled differences, not evidence of a substantial practical
advantage or a recommendation for golfers.

The aimed lateral variance ratios versus straight are 0.98107 for draw
(paired-bootstrap 95% interval 0.98007–0.98199) and 0.98062 for fade
(0.97970–0.98162). Thus the small variance reduction is resolved against
Monte Carlo sampling noise, but those intervals do not include model error.

Without aiming adjustment, the draw averages 4.25 m left and fade 4.31 m right;
their target-hit fractions fall to 84.26% and 84.04%, versus 87.56% straight.
The experiment therefore shows that aiming compensation matters much more
than the roughly 1% difference in spread. Opposite curvature occurs in 6.92%
of draw-intended shots and 6.59% of fade-intended shots. The small draw/fade
asymmetry is consistent with finite sampling of a symmetric model.

## The 2° Face-Variability Trial

| Measure | Straight | Draw | Fade |
| --- | ---: | ---: | ---: |
| Lateral Standard Deviation (m) | 18.940 | 18.772 | 18.763 |
| Landings Within 15 m of Target | 56.04% | 56.34% | 56.31% |
| Mean Carry (m) | 194.613 | 194.211 | 194.204 |

Doubling face SD almost doubles landing SD and reduces target-hit fraction
from about 88% to 56%. The curve-related SD reduction remains below 1%:
approximately 0.88% for draw and 0.93% for fade. Greater impact variability
therefore does not create a substantially larger proportional curve benefit
in this conditional model. Target-hit gains are 0.30 and 0.27 percentage points.

## Doubled-Curve Trials

The larger draw uses face +3° / path +6°; fade mirrors these angles.

| Face SD | Measure | Straight | Draw | Fade |
| --- | --- | ---: | ---: | ---: |
| 1° | Lateral Standard Deviation (m) | 9.677 | 9.313 | 9.308 |
| 1° | Central 90% Lateral Width (m) | 31.799 | 30.602 | 30.570 |
| 1° | Landings Within 15 m | 87.56% | 88.31% | 88.33% |
| 1° | Target RMSE (m) | 9.704 | 9.608 | 9.605 |
| 1° | Mean Carry (m) | 195.154 | 193.531 | 193.525 |
| 2° | Lateral Standard Deviation (m) | 18.940 | 18.270 | 18.252 |
| 2° | Central 90% Lateral Width (m) | 62.338 | 60.111 | 59.990 |
| 2° | Landings Within 15 m | 56.04% | 56.95% | 56.95% |
| 2° | Target RMSE (m) | 19.149 | 18.795 | 18.782 |
| 2° | Mean Carry (m) | 194.613 | 193.001 | 192.988 |

Doubling curve magnitude increases the modeled lateral-SD reduction to
3.5–3.8%, but costs approximately 1.62 m mean carry. The target RMSE improves
less than lateral SD because it also counts longitudinal error. At equal range,
the larger-curve SD benefit is approximately 2.9–3.0%, versus 0.7–0.8% for the
original curves. Thus part of the apparent precision benefit is shorter range.
Greater face variability still does not produce a larger proportional benefit.
The straight-shot rows are identical across curve magnitudes within each SD,
providing a matched control rather than a fresh random baseline.

## Carry Distance and the Equal-Range Check

Shots do not all travel the same distance. Face error changes impact and spin,
so carry varies within each pattern; curved patterns also average about 0.41 m
shorter in the 1° trial. To check this geometric advantage, each aimed endpoint
is projected radially onto the same 195.334 m range, preserving its landing
bearing. This is a geometric diagnostic, not a second physical flight model.

| 1° Trial: Lateral Standard Deviation (m) | Straight | Draw | Fade |
| --- | ---: | ---: | ---: |
| Actual Carry | 9.677 | 9.585 | 9.583 |
| Common Range | 9.704 | 9.630 | 9.629 |

The curved-pattern SD advantage falls from approximately 0.95% to 0.76%.
Shorter carry therefore explains part of the apparent improvement, while a
small angular-spread difference remains in this model. This normalization
cannot identify a causal physical mechanism or validate the impact model.

## Approach Strokes-Gained Scenario

This estimate treats the shot as a fairway approach from 195.334 m to a hole
at the center of a circular green of radius 15 m, with rough surrounding it.
The carry endpoint is assumed to be the final resting position: no bounce,
roll, slopes, bunkers, water, or out-of-bounds. The saved scoring outputs also
include 10 m and 20 m green radii to expose layout sensitivity.

Every endpoint is evaluated with the pinned Tools public source-backed
strokes-gained API. Fairway and rough expected strokes are factual numerical
values from [Broadie's Historical PGA Tour Benchmark, Table 9](https://www.columbia.edu/~mnb2/broadie/Assets/strokes_gained_pga_broadie_20110408.pdf),
based on 2003–2010 data. Putting uses the paper's one-putt expression and an
explicitly approximate three-putt reconciliation. The printed coefficient
order produces invalid probabilities; the chosen approximation agrees with
the stated 33-foot/two-putt and 40-foot/10% three-putt anchors, with a joint
probability bound for tap-ins. This is not an independently verified exact
reproduction of the author's fit. The baseline artifact stores every knot,
source PDF hash, interpolation support, and its own verified digest.

Strokes gained equals expected strokes at the start minus one shot minus
expected strokes remaining. Because every pattern starts at the same point,
the relative benefit comes entirely from its finishing outcomes. Positive
paired differences favor the curved pattern. These values are per approach,
not per round, and are not predictions of a real player's score.

For the original 1° trial, mean SG is +0.4460 straight, +0.4436 draw and
+0.4431 fade. The curved-minus-straight changes are −0.00239 strokes for draw
(95% paired Monte Carlo interval −0.00339 to −0.00150) and −0.00291 for fade
(−0.00390 to −0.00208). A slightly narrower lateral pattern therefore need
not improve scoring. Model, course-layout, and benchmark uncertainty are
excluded from these sampling intervals and can exceed these tiny differences.

## Data and Shareable Graphics

The four bundles are `results/` (1° SD, 1× curve), `results_sd2/` (2° SD,
1× curve), `results_large_curve_sd1/` and `results_large_curve_sd2/` (2× curve).
Each contains the following files: `shots.csv` contains all 30,000 landings,
`summary.json` contains variance, standard deviation, percentiles, target RMSE,
target-hit fractions, Wilson intervals and paired-bootstrap variance ratios,
and `receipt.json` records model parameters and source/runtime hashes.
The overview `curve_magnitude_comparison.png` compares all four trials.
The Facebook graphics are 1920 × 1080 PNG files:

- `overhead_flight.png`: nominal and nine representative flights per pattern.
- `dispersion.png`: all landings after nominal aiming adjustment on shared scales.
- `dispersion_equal_range.png`: the geometric common-range comparison.

The overhead and dispersion plots deliberately label their different aiming
conditions. The displayed 90% spread is empirical, not a fitted ellipse.

## Model Review and Limits

The implementation reuses the public `ImpactSolverAPI` rigid-body model and
`BallFlightSimulator` Rust RK4 solver. Units and frame conversions are explicit.
Drag uses the kernel's Reynolds-dependent drag-crisis curve; its single
spin force includes both vertical lift and lateral curvature, with spin decay.
It is a different coefficient pathway from `flight_models.py`'s constant-spin
Waterloo/Penner model, and no coefficients were changed for this study.

The rigid-body impact solver adds friction-generated angular momentum while
omitting the matching tangential translational impulse. Consequently, ball
launch direction equals face angle exactly in this implementation. This is a
material model limitation, especially for interpreting small differences among
patterns. No empirical launch-monitor validation of these shot populations was
performed. The model also excludes face/path covariance, strike/speed/loft error,
wind, ground conditions and a player's ability to execute a chosen shape.
Because only face angle is random, the landing cloud lies on a narrow curved
one-dimensional locus. It is not a realistic two-dimensional golfer dispersion
ellipse; adding independent speed, path and strike errors would change its shape.

The normal-error assumption gives the following theoretical chance of crossing
zero face-to-path and curving opposite the intended direction:

| Curve Magnitude | Face SD 1° | Face SD 2° |
| --- | ---: | ---: |
| Original: 1.5° Mean Face-to-Path Offset | 6.68% | 22.66% |
| Doubled: 3° Mean Face-to-Path Offset | 0.135% | 6.68% |

These are direction probabilities, not accuracy or scoring improvements. Curving intention does not eliminate one
curvature direction under the assumed error distribution. Draw/fade differences
in a finite sample can arise from sampling; the still-air centered-strike model
is symmetric under mirroring all horizontal angles.

## Numerical and Software Evidence

[Numerical Validation](numerical_validation.json) records 15 deliveries at
nominal and ±1°/±3° face deviations. The maximum Euclidean landing change was
0.0003863 m for time steps 0.02→0.01 s and 0.0001112 m for 0.01→0.005 s,
against a predeclared 0.05 m limit. This establishes numerical consistency
for those cases, not physical accuracy.

Tests cover paired sampling, target aiming, invalid inputs, finite output,
known bootstrap variance ratios, export/schema behavior, signs of curvature,
mirror symmetry, time-step refinement and failure to land within the horizon.
The focused model/API and new analysis validation passed 59 tests before
launcher integration. Later integration evidence is recorded in the final
execution handoff.

## Reproduction

Use Python 3.11+ and install the project's numerical dependencies and the native
Rust wheel from the same checkout:

```bash
git submodule update --init vendor/ud-tools
python3 -m pip install -e '.[gui-tools]' maturin
VIRTUAL_ENV=/path/to/venv maturin develop --release \
  --manifest-path rust_core/upstream-physics/Cargo.toml --features python
python3 -m src.tools.shot_pattern_analysis.matrix \
  docs/research/shot_pattern_analysis/corrected_results
python3 -m src.tools.shot_pattern_analysis.sensitivity \
  --output docs/research/shot_pattern_analysis/corrected_sensitivity.json
python3 -m src.tools.shot_pattern_analysis.numerics \
  --output docs/research/shot_pattern_analysis/corrected_numerical_validation.json
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy \
  python3 -m pytest tests/tools/shot_pattern_analysis \
  tests/ui/tools/shot_pattern_analysis
```

The [Editable LaTeX Experiment Reference](shot_pattern_analysis.tex) records
equations and calculation assumptions. It is separate from the canonical
engineering design manual in `manuals/upstreamdrift`; that manual's calculation
inventory and scientific publication gates remain blocked. Built-in LaTeX
compilation was attempted but could not obtain its uncached Tectonic bundle;
compilation is unverified. The source remains editable in the native editor.

The repository's required `python3 -m scripts.pre_pr` command is unavailable in
this checkout (`No module named scripts.pre_pr`). The full document title audit
reports 2,505 existing violations across 1,816 documents; this change's document
titles are checked separately. Neither limitation is represented as a passing
gate. GitHub publication requires Codex bot credentials; the host currently
authenticates as the owner account, which repository policy prohibits for agent
publication.
