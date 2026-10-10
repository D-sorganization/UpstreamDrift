# Reproducing the Matched Study Publication

This workflow rebuilds figures, statistics, and the narrative from the preserved
final shot rows. It does not run new Monte Carlo trials or replace the accepted
simulation evidence. Use the [Archive Restoration Instructions](data_archives/README.md)
to restore `final_v1`, `control_final_v1`, `supplement_final_v1`, and the
`control_pilot_v1` provenance prerequisite first. The archive bundle names differ
from the first two restored directory names; use the explicit mapping in that
guide. Keep tracked `common_targets.json`, `FINAL_RUN_ACCEPTANCE.md`, and
`validation/independent_control_pilot_audit.json` in the study directory layout.
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

The current publication regeneration started from `4bb2be7362`. Its
[Start Receipt and Frozen Source Inventory](validation/publication_regeneration_v4_start/start.json)
and adjacent executable wrapper copies document the actual invocation. A start
receipt proves neither completion nor graphical acceptance. The original-figure
wrapper checks the accepted execution snapshot, native library and runtime;
only `academic_plots.py` and `matched_report_plots.py` are permitted reporting
differences in that execution snapshot. All study Python is frozen for the live
jobs. Do not change code or start a second writer in their output directories.

The caption-only original-figure revision uses BOUNDS candidate `c9c0fb194f` and the [V6 Preflight and Wrapper](validation/original_final_figures_rendering_v6_preflight/README.md). The first v5 launch failed its output-name contract before rendering; the corrected invocation keeps canonical names. Historical v4 outputs are retained under `presentation_preview_rejected_final_v4`, with [Exact-Byte Relocation Evidence](validation/original_final_figures_v4_preservation/relocation.json). A source/input preflight does not establish final graphical acceptance.

The supplemental pipeline separately binds 14 postprocessing source files,
including the shared `matched_formatting.py` precision helper, in
`REPORT_SUPPLEMENT_PIPELINE.json`. This reporting identity does not replace the
historical 54-source execution identity. The final receipt is written only after
all statistics and figures finish and the source/input checks pass. A directory
containing tables but lacking that receipt is incomplete.

For this saved inventory, expect 384 paired comparisons: 164 curve-strength,
164 shaft-SD and 56 across-lean contrasts. The across-lean records now include
the direct change in Draw/Fade-versus-Straight scoring benefit on the same
six-population intersection. For this across-lean interaction addition, existing
per-pattern estimates and populations must remain unchanged from the preserved
earlier tables. New interaction fields and
their reporting provenance are expected additions; subtracting separately
conditioned means or confidence limits is not an equivalent reproduction.

## Rebuild From Saved Final Data

Use a separate working copy for reproduction. Preserve its existing published
figure directories, supplemental-report directory, report and companion inventories
in a backup location before running these commands. The figure builders reject
nonempty destinations. Keep the study directory layout below: the narrative
links its scientific design, original plan, and validation references relative
to that directory. These commands are for rebuilding, not for appending to an
active publication run.

The CLI recipe below invokes the builders directly. It does not reproduce the
outer original-figure wrapper's runtime/native snapshot checks by itself. To
reproduce those guarantees, apply the snapshot comparison in the preserved
wrapper before rendering and verify the frozen Python inventory again after
rendering. Its absolute paths identify the recorded run; adapt those paths to
the separate reproduction checkout while retaining the comparison logic and
unchanged accepted hashes. A different runtime is a new reproduction environment
and must be disclosed rather than silently accepted as the original one.

The current jobs preserve superseded original graphics under
`presentation_preview_rejected_final_v3/{final_figures_v1,control_final_figures_v1}`
and the intentionally stopped supplemental output under
`presentation_preview_rejected_supplement_v2/supplement_report_v1`. These are
rejected or partial evidence, not publication inputs. Their exact file hashes
and stop receipts are retained in
[Preservation Evidence](validation/reporting_revision_preservation_v4/).
Historical receipts may still name the original absolute output locations;
use the preservation inventory to locate the identical bytes without rewriting
those receipts. Restore sparse historical checkouts before running commands
that depend on their omitted files.

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

The supplemental pipeline computes all paired statistics before creating its
output directory, then renders each unique supplemental cell and every available
comparison. A quiet log or an absent output directory during that calculation
does not establish a stalled or terminated job. Check the actual process handle;
do not restart from an observation timeout. Allow the full run to produce its
terminal receipt before invoking the narrative builder.

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
