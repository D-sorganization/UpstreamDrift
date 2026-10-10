# Independent Review of the Corrected Shot Pattern Study

The final V2 results pass this independent internal-consistency review. This
accepts the recorded numerical experiment under its stated assumptions; it does
not qualify the model against measured golfers, launch monitors, or course play.
Review completed on 9 October 2026 UTC by the independent Astra review agent.

## Completed Gates

- All 24 cells contain 10,000 finite shots per pattern, or 720,000 shots total.
  All 216 artifact hashes recorded by their manifests matched the inspected files.
- Lateral sample SD, mean carry, target RMSE, equal-range lateral SD, and
  downrange–lateral covariance were independently recomputed from every CSV.
  Maximum discrepancy from saved summaries was 2.84 × 10⁻¹⁴ in the relevant units.
- All cells use baseline `table9-full-plus-anchor-reconciled-putting/2`.
  Every unholed green knot is at least one expected stroke; bounded linear
  interpolation therefore preserves that physical lower bound. Saved flight CSV
  hashes match the rescoring receipts. V1 scoring is retained in a separate archive.
- All 24 summary scoring objects agree with their standalone scoring reports.
  Paired SG point estimates agree with differences of saved pattern means within
  10⁻¹² strokes. Driver starts are tee states; iron and PW starts are fairway states.
- The final focused scoring suite passed 19 tests, including all knots,
  midpoints, and seeded random API-parity cases, unsupported-distance rejection,
  the unholed tap-in regression, and historical parity using historical baselines.
  The historical parity check does not require corrected V2 scores to equal V1.
- Earlier independent impact checks passed momentum, restitution, energy,
  sticking, pre-existing-spin, and rigid-rotation tests for central contact.
  Native landing/full-trajectory parity passed to 10⁻¹² m. The recorded
  270-flight refinement study passed its unchanged 0.05 m gate, with maximum
  0.02→0.01 s change of 0.000573215 m.
- All 72 per-cell figures meet the checked pixel-size floor. The two overview
  SG figures and representative carry/flight figures were visually inspected for
  labels, sign, units, context, and legibility. Overview legends identify both
  patterns; intervals are labeled as Monte Carlo intervals. This is representative
  visual inspection, not a claim that every figure received a separate visual review.

The [Audit Receipt](independent_final_audit.json) records the exact reviewed
artifact and cell-manifest SHA-256 values. The source archive digest is
`e6de6acc7979f0cc7affffc4ccb73b6429c74d0ed4e3ba097e65871e2926dd73`;
the archived pre-correction scoring digest is
`9c5130e845b91442fe427d9166a7c89c4b48655a50a849ff7abc5129df7da9ec`.

## Interpretation Checks

The corrected README and calculation reference distinguish actual carry from
the geometric equal-range diagnostic. They correctly restrict mirror symmetry
to fixed-loft delivery: a fixed right-handed shaft axis changes loft differently
under positive and negative face errors. Coupled draw/fade differences can thus
be structural, as well as affected by finite sampling.

The PW example confirms why lateral width alone cannot predict scoring. At face
SD 1° and curve scale 1×, shaft-coupled straight/draw/fade downrange variances are
1.3691/1.1831/1.5638 m², while lateral variances are 3.8956/3.9958/3.7660 m².
The draw's smaller distance spread more than offsets its larger lateral spread;
the reverse occurs for the fade. Target RMSE is 2.2953/2.2788/2.3115 m, consistent
with draw/fade relative SG of +0.0023001/−0.0023691 per shot. The two curved
patterns have similar short biases. Their correlations near −0.996 describe a
tilted long-left/short-right locus; covariance is not itself an extra term in
the mean squared radial target error.

The deterministic loft sensitivity also supports the conditional wording:
closing and delofting produces long-left/short-right for these iron/PW presets,
but short-left/long-right for the selected driver operating point. A universal
distance response to delofting is not supported.

## Remaining Qualification Limits

The presets are illustrative mixtures of sourced geometry and explicit delivery
assumptions. Manufacturer static lie is not measured delivered shaft elevation;
the PW loft source is an optimizer example, and the 7-iron delivered loft and
head masses are assumptions. No empirical population calibration was performed.

The central impact and native aerodynamic model remain conditional approximations.
Off-center club rotation, gear effect, shaft bending, changing strike, realistic
face/path covariance, wind, roll, and wedge stopping are outside this experiment.
The 2011 benchmark and reconciled putting model are historical and approximate.
The course geometry is hypothetical. Sampling intervals exclude these uncertainties.

The study is suitable as a reproducible geometric and model-sensitivity analysis.
It does not establish a preferred shot shape for a golfer. Repository-wide gates, document compilation, and any PR
lifecycle steps are tracked by the parent agent separately; this review does not
claim those have passed merely because the scientific artifact checks passed.

## Resumed Committed-Evidence Check

At commit `a2eb2ace3629dd362db5230c958a83a8f61fec45`, all 24 CSVs read directly
from Git HEAD matched their frozen manifest SHA-256 values. All seven recorded
artifact/archive hashes and all 24 cell-manifest hashes in the audit receipt
still matched. The seven audited artifacts also matched their Git HEAD blobs;
all 33 source-archive and 144 historical-scoring-archive member hashes verified.
No flight rerun was needed. The 18 issue acceptance records
continue to distinguish local implementation from merged-PR closure; the
canonical type/release gate is not accepted solely from focused checks.

The resumed reading identified stale completion wording in the main README and
reference, and old reference reproduction commands targeting historical output
directories. These were reported to the parent for correction before publication.
The corrected completion wording and fresh-output matrix/refinement reproduction
commands were subsequently verified, as was the added primary iron-lie source.
These corrections do not change the audited numerical results.

## Focused Reproduction

From the repository root, with its configured Python environment:

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy \
python -m pytest -q \
  tests/tools/shot_pattern_analysis/test_scoring_cache_independent.py \
  tests/tools/shot_pattern_analysis/test_scoring_cache.py \
  tests/tools/shot_pattern_analysis/test_scoring.py \
  tests/tools/shot_pattern_analysis/test_scenario_scoring.py
```

Full experiment, numerical-refinement, and sensitivity commands are retained in
the [Study README](README.md) and its editable LaTeX reference.
