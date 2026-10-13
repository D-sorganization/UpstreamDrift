# Shot Pattern Analysis

Compare 10,000 simulated shots each for straight, draw, and fade patterns.
The reusable Python package delegates impact and flight to the existing
UpstreamDrift solvers; it does not duplicate their physics. It is a Python
submodule within this repository, rather than a separately hosted Git submodule.

## Matched Delivery Extension

The **Matched Study** GUI tab compares raw nominal landings without post-flight
aiming. It exposes static loft, shaft lean, shaft-twist/path/speed/strike SDs,
and a validated seven-variable correlation matrix. Nominal face, path and
speed are solved and displayed. The primary contact law is incremental Coulomb
with energetic restitution; the Newton impulse law is a separate comparator.
Install optional acceleration with `pip install -e '.[shot-pattern-acceleration]'`.
All head properties beyond documented loft/lie remain research assumptions.

See the [Matched Study Design](../../../docs/research/shot_pattern_analysis/phase2/DESIGN.md),
the [Reviewed Design PDF](../../../docs/research/shot_pattern_analysis/phase2/matched_delivery_reference.pdf),
the [Nominal Matching Bounds Supplement](../../../docs/research/shot_pattern_analysis/phase2/BOUNDS_SUPPLEMENT.md)
([editable LaTeX](../../../docs/research/shot_pattern_analysis/phase2/BOUNDS_SUPPLEMENT.tex)),
and [Turnover](../../../docs/research/shot_pattern_analysis/phase2/TURNOVER.md).
The phase-one section below preserves the historical controls.
The original and supplemental final matrices are complete, and their saved-data
independent audits and archive-restoration checks have passed. The revised
39-page illustrated report has been reviewed for local document delivery, with
lateral dispersion on the horizontal axis and downrange distance on the vertical
axis. Its expanded methods and separate verified 100-shot flight replays preserve
the original simulation statistics. The finding is conditional: no shot shape
wins across all modeled clubs and delivery distributions. Repository release
gates and publication of the revised edition remain open. The linked design PDF
is a design snapshot; the editable results source is the separate
[Illustrated Research Article](../../../docs/research/shot_pattern_analysis/shot_pattern_analysis.tex).
See the [Current Refinement Status](../../../docs/research/shot_pattern_analysis/phase2/validation/refinement_v3/progress.json),
and the
[Saved-Data Reproduction Guide](../../../docs/research/shot_pattern_analysis/phase2/REPRODUCING.md)
and [Final Report Status](../../../docs/research/shot_pattern_analysis/phase2/REPORT.md).

### Configure the Matched Study

1. Select Driver, 7-Iron, or Pitching Wedge and set static loft and forward lean.
   Static loft is a club-face geometry input; delivered loft is calculated after
   lean and shaft rotation. Driver 10° loft at 10° forward lean can be nonairborne.
2. Choose curve strength and the nominal target. Matching adjusts face, path and
   speed within declared bounds; it never rotates the finished trajectories.
   The nominal-search controls show reference-centered face bounds by default
   (each pattern face ±3°) and speed factors 0.75–1.25 times reference club
   speed. Enable absolute face bounds to edit a common interval; both the face
   interval and speed factors survive config JSON import/export. The separate
   driver bounds sensitivity uses a 1.30 upper speed factor explicitly.
3. Set each delivery SD in its displayed units. Shaft rotation SD is not face-angle
   SD: delivered face angle is a derived output. Strike SD applies separately to
   heel/toe and high/low position. Zero SD holds a variable exactly fixed.
4. Enter correlations in the symmetric seven-variable matrix. The diagonal is one;
   a non-positive-semidefinite matrix is rejected. A shaft/path correlation is not
   an exact delivered face/path correlation; inspect realized output covariance.
5. Run into a new output directory. Review matching status and supported-shot
   counts before interpreting density or conditional strokes gained. Unsupported
   samples remain in the export and are never replaced with extra draws. The
   results view displays nominal deliveries alongside saved conditional course
   scoring (three-pattern mean strokes gained, paired differences vs Straight with
   95% Monte Carlo intervals, and common supported population counts) under
   the hypothetical flat course model. Display precision adapts to the saved
   effect and interval width so small effects and nonzero uncertainty remain
   visible. Starting another study, failure, or cancellation clears previous SG
   from the current-result display.

Save/load configuration controls retain the full input specification. A cancelled
or failed export does not publish a complete result directory. Both contact laws
are research assumptions; the incremental energetic law is the reviewed primary
pathway and the Newton impulse law is a sensitivity comparator.

For a small headless matched matrix:

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \
  python -m src.tools.shot_pattern_analysis.matched_matrix /tmp/matched-pilot \
  --stage pilot --shots 32 --profiles shaft_only combined2 --figures selected
```

The final research matrix uses exactly 10,000 shots per pattern and requires a
reviewed acceptance JSON matching current source, runtime and evidence hashes.
See the phase-two Turnover for final run commands and archive restoration.

### Publication Identity and Numerical Parity

The supplemental publisher derives filenames from the ordered source protocols,
cell identities and exact target distance. It rejects planned collisions before
statistics, uses suffix-safe cell names, and verifies output hashes and path
uniqueness before writing the final receipt. Pure naming helpers live in
`matched_figure_identity.py` and are exposed through the figure module.
For recovery against unchanged source locations, `--expected-numeric-hashes`
binds all seven numerical artifacts before rendering; see the reproduction guide.

## Run a Historical Phase-One Analysis

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

| Pattern  | Mean Face | Fixed Path | Face-to-Path |
| -------- | --------: | ---------: | -----------: |
| Straight |        0° |         0° |           0° |
| Draw     |     +1.5° |        +3° |        −1.5° |
| Fade     |     −1.5° |        −3° |        +1.5° |

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
