# Shot Pattern Analysis

Compare 10,000 simulated shots each for straight, draw, and fade patterns.
The reusable Python package delegates impact and flight to the existing
UpstreamDrift solvers; it does not duplicate their physics. It is a Python
submodule within this repository, rather than a separately hosted Git submodule.

## Run an Analysis

From the repository root, with the native `upstream-physics` wheel installed,
select the Shot Pattern Analysis tile or run
`python3 -m src.tools.shot_pattern_analysis` to open the tool. For headless export:

```bash
python3 -m src.tools.shot_pattern_analysis --club-preset driver \
  --shots 10000 --seed 20261008 --output shot_pattern_results/driver_sd1
python3 -m src.tools.shot_pattern_analysis --club-preset seven_iron \
  --face-sd 2 --curve-scale 2 --delivery-mode shaft_rotation \
  --shots 10000 --seed 20261008 --output shot_pattern_results/seven_iron_coupled_sd2
python3 -m src.tools.shot_pattern_analysis.matrix shot_pattern_results/full_matrix
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
It uses the same random deviations for each pattern. Illustrative driver,
7-iron, and pitching-wedge presets supply explicit speed, nominal delivered loft,
attack angle, head mass, and assumed shaft elevation. They are hybrid research
inputs, not measured player averages. Controls can be edited while retaining
the selected club's scoring context.

Fixed-loft mode holds delivered loft constant as face varies. Shaft-rotation
mode couples face error and loft through an assumed rigid shaft axis around
each pattern nominal. All shapes share the same zero-error loft. Lie and lean
control that geometry; no measured closure covariance or shaft bending is
inferred. Contact is central and wind is zero.

Both raw endpoints and outcomes after one nominal aiming rotation are reported.
Aiming uses the zero-error shot, never the sample mean. The generic 15 m target
circle is centered at straight nominal carry. Driver scoring separately uses a
hypothetical 400 m tee hole and 30 m fairway corridor; iron/PW scoring uses an
approach to a 15 m green with rough outside. Historical PGA benchmarks and an
approximate putting fit provide conditional SG, not player predictions.

The corrected rigid-body model applies matching tangential linear and spin
impulses, includes finite club mass in the sticking condition, and rejects
separating contact. Independent conservation tests verify central-contact
consistency. Off-center gear effect and measured player accuracy remain
unqualified. Face-only randomness is not a calibrated 2D golfer dispersion.

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
