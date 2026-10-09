# Shot Pattern Analysis

Compare 10,000 simulated shots each for straight, draw, and fade patterns.
The reusable Python package delegates impact and flight to the existing
UpstreamDrift solvers; it does not duplicate their physics. It is a Python
submodule within this repository, rather than a separately hosted Git submodule.

## Run an Analysis

From the repository root, with the native `upstream-physics` wheel installed:

```bash
python3 -m src.tools.shot_pattern_analysis --shots 10000 --seed 20261008 \
  --output docs/research/shot_pattern_analysis/results
python3 -m src.tools.shot_pattern_analysis --shots 10000 --seed 20261008 \
  --face-sd 2 --curve-scale 2 \
  --output docs/research/shot_pattern_analysis/results_large_curve_sd2
```

The bundle contains per-shot CSV, summary JSON, a provenance receipt, and
1920 × 1080 PNG graphics for Facebook sharing. The overhead view shows nominal
and representative flights; all shots enter the dispersion and statistics.
All distances describe carry at first ground contact, excluding roll.

## Inputs and Interpretation

Angles are relative to the target, positive right for a right-handed golfer:

| Pattern | Mean Face | Fixed Path | Face-to-Path |
| --- | ---: | ---: | ---: |
| Straight | 0° | 0° | 0° |
| Draw | +1.5° | +3° | −1.5° |
| Fade | −1.5° | −3° | +1.5° |

`spin_axis_tilt_deg` is positive for native +world-z spin and draw/left curvature,
opposite a right-positive TrackMan-style tilt sign; face, path, and carry are
positive right.

The trials vary face angle with a normal standard deviation of 1° or 2°.
`--face-sd` sets that standard deviation; `--curve-scale 2` doubles the mean
draw angles to face +3° and path +6°, with the fade mirrored. The straight
baseline stays at 0°/0°. Both controls are available in the launcher tile.
It uses the same random deviations for each pattern. Speed is 45 m/s, loft
10.9°, attack angle 0°, contact centered, and wind zero. Both raw outcomes and
outcomes after a single nominal aiming rotation are reported. Aiming uses the
zero-error shot, never the random sample's mean. The target circle has a 15 m
radius at the straight nominal carry distance.

The rigid-body impact model launches translation along the face normal while
adding friction spin separately; it omits the corresponding tangential
translational impulse. Results are conditional on this simplified model.
They do not establish how well a golfer can control each shot pattern.

See the [Experiment Reference](../../../docs/research/shot_pattern_analysis/README.md)
and its [Editable LaTeX Source](../../../docs/research/shot_pattern_analysis/shot_pattern_analysis.tex)
for equations, evidence, results and reproduction details.

## Development Contracts

The pure data API is `AnalysisConfig` and `run_analysis` in `core.py`.
`ShotPhysics` adapts target-relative angles to the solver frame and rejects
nonfinite input or a flight that does not reach the ground. `export_analysis`
writes the data and graphic bundle. The GUI and embed adapter load optional
dependencies lazily. Configure GUI dependencies with `pip install -e '.[gui-tools]'`.

Run focused tests headlessly:

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy \
  python3 -m pytest tests/tools/shot_pattern_analysis \
  tests/ui/tools/shot_pattern_analysis
```
