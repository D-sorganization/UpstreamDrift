# Straight, Draw, and Fade: Shot Pattern Analysis

This experiment compares 10,000 shots per pattern with independent normally
distributed face errors at both 1° and 2° standard deviation and fixed path. It is a
conditional simulation study, not a measured golfer study or physics qualification.

## Experiment Definition

| Pattern | Mean Face Relative to Target | Fixed Path Relative to Target |
| --- | ---: | ---: |
| Straight | 0° | 0° |
| Draw | +1.5° | +3° |
| Fade | −1.5° | −3° |

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

## Results and Practical Meaning

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

## Data and Shareable Graphics

The result bundle lives in `results/`: `shots.csv` contains all 30,000 landings,
`summary.json` contains variance, standard deviation, percentiles, target RMSE,
target-hit fractions, Wilson intervals and paired-bootstrap variance ratios,
and `receipt.json` records model parameters and source/runtime hashes.
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

A 1.5° face-to-path offset with 1° standard deviation leaves a theoretical
6.68% chance of opposite curvature. Curving intention does not eliminate one
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
python3 -m src.tools.shot_pattern_analysis --shots 10000 --seed 20261008 \
  --output docs/research/shot_pattern_analysis/results
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
