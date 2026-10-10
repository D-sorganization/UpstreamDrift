# Reproducing the Matched Study Publication

This workflow rebuilds figures, statistics, and the narrative from the preserved
final shot rows. It does not run new Monte Carlo trials or replace the accepted
simulation evidence. Use the [Archive Restoration Instructions](data_archives/README.md)
to restore `final_v1`, `control_final_v1`, and `supplement_final_v1` first.
The [Turnover](TURNOVER.md) identifies current publication acceptance and blockers.

## Environment and Source Identity

Run from the repository root with the pinned Tools submodule initialized and the
study dependencies installed. The accepted environment used Python 3.12.13,
NumPy 2.2.6, SciPy 1.18.1, Numba 0.68.0, and llvmlite 0.50.0. Exact simulation
source, native-library, runtime, input, and protocol hashes remain in the saved
receipts; installing similarly named packages does not reproduce those identities.

The supplemental numerical snapshot was verified at commit `256745676e`.
Subsequent renderer and GUI changes are reporting revisions, not new execution
snapshots. Keep the original and adapted-bound protocols separate. Never rewrite
historical manifests to make them match newer code. See the
[Supplemental Completion Review](validation/supplement_final_independent_completion_review_v1/REVIEW.md).

## Rebuild From Saved Final Data

Use a separate working copy for reproduction. Preserve its existing published
figure directories, supplemental-report directory, report and companion inventories
in a backup location before running these commands. The figure builders reject
nonempty destinations. Keep the study directory layout below: the narrative
links its scientific design, original plan, and validation references relative
to that directory. These commands are for rebuilding, not for appending to an
active publication run.

```bash
set -e
export QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy OPENBLAS_NUM_THREADS=1
study_phase=docs/research/shot_pattern_analysis/phase2
study_output="$study_phase"
test ! -e "$study_output/final_figures_v1"
test ! -e "$study_output/control_final_figures_v1"
test ! -e "$study_output/supplement_report_v1"

python -m scripts.research.shot_pattern_analysis.build_final_figures \
  "$study_phase/final_v1" "$study_output/final_figures_v1"
python -m scripts.research.shot_pattern_analysis.build_final_figures \
  "$study_phase/control_final_v1" "$study_output/control_final_figures_v1"

python -m src.tools.shot_pattern_analysis.matched_supplement_pipeline \
  --original-primary "$study_phase/final_v1" \
  --original-control "$study_phase/control_final_v1" \
  --pw-primary-face-bounds "$study_phase/supplement_final_v1/pw_primary_face" \
  --pw-control-face-bounds "$study_phase/supplement_final_v1/pw_control_face" \
  --driver-control-speed-bounds "$study_phase/supplement_final_v1/driver_control_speed" \
  --inventory "$study_phase/review_followups/SUPPLEMENTAL_EXECUTION_INVENTORY.json" \
  --transition-receipt "$study_phase/source_transition_supplement.json" \
  --transition-sha256 cf622694964b0dc6c2ca0c24b92f9fa8ee5365721219069283aca1f6804a8981 \
  --output-dir "$study_output/supplement_report_v1"

python -m scripts.research.shot_pattern_analysis.build_final_report \
  "$study_phase/final_v1" "$study_phase/control_final_v1" \
  "$study_output/REPORT.md" \
  --supplement-dir "$study_output/supplement_report_v1" \
  --primary-figures "$study_output/final_figures_v1" \
  --control-figures "$study_output/control_final_figures_v1"
python -m scripts.research.shot_pattern_analysis.build_pdf \
  "$study_output/REPORT.md" "$study_output/REPORT.pdf"
```

The report builder checks saved input identities and figure hashes and produces
`REPORT_CELLS.csv`, `REPORT_LEAN.csv`, and `REPORT_PUBLICATION_RECEIPT.json`.
The supplemental receipt binds its separate inventories, common-population
contrasts, figures, and postprocessing sources. Preserve failed or unavailable
conditions and unsupported rows: a conditional result is not a result for all
attempted deliveries.

The PDF builder uses Pandoc and ReportLab, separately from the app's LaTeX editor.
Its success cannot resolve or establish acceptance of the recorded Tectonic
bundle-download failure. Keep the user's existing LaTeX editor open and record
its actual compilation result separately.

## Acceptance After Rebuilding

A process exit or matching manifest is necessary but insufficient. Review the
[Scientific Narrative Checklist](review_followups/FINAL_NARRATIVE_ACCEPTANCE.md),
inspect rendered academic and social figures and every PDF page, and verify the
actual output population and condition labels. Keep the immutable original
execution receipts and the new reporting-source receipt together. Independent
empirical qualification remains deferred in
[DV-11984](../../../development/planning/DV-11984.md).
