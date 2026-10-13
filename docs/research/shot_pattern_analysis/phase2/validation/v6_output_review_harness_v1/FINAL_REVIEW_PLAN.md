# V6 Full Publication Review Plan

## Boundary and Scope

This is a plan, not acceptance. No live v6 graphic was read while preparing it. Wait for the source-bound successful terminal receipt; a quiet directory, complete-looking filenames, or a finished numerical stage is insufficient. Never fabricate a final receipt or rerun statistics/simulations for review.

The frozen producer is PRESENTATION commit `4023111d2aaf8b20449a36f8691a295f4a94ceff`. The qualified operational harness is `phase2/validation/v6_output_review_harness_v1/review_batch.py` in MAIN, SHA256 `1db7be642e097377afeb44874687e41639a29c32c0440f13b3055eb3457d0bf6`. Its current helper transition passes 114 tests, including all 22 original harness cases; see `validation/repository_readiness_repair_v1/independent_acceptance/`. Historical qualification remains preserved; do not repeat tests absent a change. Its contacts are inventories, never automatic visual acceptance.

## Exact Ledger Coverage

| Component                 | Records | Formats per Record           | Assets |  PNGs |
| ------------------------- | ------: | ---------------------------- | -----: | ----: |
| Across-Lean Pairs         |      56 | Six PNG, Five PDF, Five SVG  |    896 |   336 |
| Curve-Strength Pairs      |     164 | Six PNG, Five PDF, Five SVG  |  2,624 |   984 |
| Shaft-Twist-SD Pairs      |     164 | Six PNG, Five PDF, Five SVG  |  2,624 |   984 |
| Unique Supplemental Cells |      26 | Four PNG, Four PDF, Four SVG |    312 |   104 |
| Total                     |     410 |                              |  6,456 | 2,408 |

Pair suffixes are `density.{png,pdf,svg}`, `overhead.{png,pdf,svg}`, `overhead_social.png`, and `density_{straight,draw,fade}_social.{png,pdf,svg}`. There is no pair overhead-social PDF/SVG. Cell suffixes are `{density,density_social,overhead,overhead_social}.{png,pdf,svg}`. Totals are 2,408 PNG, 2,024 PDF and 2,024 SVG. These are expected production counts, not a substitute for the actual final ledger. Any explicit unavailable record requires reconciliation and a scope decision; never silently drop it to hit a count.

Names must come from the qualified public `pair_comparison_identity`/`pair_comparison_slug` and cell `_slug` façade. Never reconstruct old cell-only stems. Pair identities include ordered source protocols, cell IDs and exact target; protocol aliases are not independent simulations.

## Terminal Verification and Commands

Use the frozen producer checkout, not a later MAIN module version. First inspect terminal JSON and call the strict public loader once; it verifies all bound inputs, artifacts, numerical reference, sources and unavailable-source records. Record returned receipt identity and hashes. Do not run the producer CLI again.

```bash
REVIEW_PY=/tmp/shot-pattern-validation-venv-20261009/bin/python
STUDY_MAIN=/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift
STUDY_PRODUCER=/home/dieterolson/.codex/worktrees/shot-pattern-presentation/UpstreamDrift
STUDY_PHASE="$STUDY_MAIN/docs/research/shot_pattern_analysis/phase2"
REVIEW_OUT="$STUDY_PHASE/validation/supplement_v6_final_graphics_review_v1"
cd "$STUDY_PRODUCER"
"$REVIEW_PY" - <<'PY'
import json
from pathlib import Path
from src.tools.shot_pattern_analysis.matched_supplement_pipeline import load_verified_supplement_artifacts
phase = Path('/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift/docs/research/shot_pattern_analysis/phase2')
terminal = json.loads((phase/'validation/supplement_rendering_revision_v6/exit.json').read_text())
if terminal.get('process_exit_code') != 0 or terminal.get('status') != 'completed_pending_full_independent_acceptance':
    raise ValueError('No successful completed producer boundary')
result = load_verified_supplement_artifacts(phase/'supplement_report_v1')
print('Strict public loader passed; visual acceptance remains pending.')
PY
```

Then use the qualified harness for each batch below. This is the exact command template; substitute only the listed group/kind/offset/limit/output suffix. The pinned start SHA was independently read from the actual committed start, not calculated from an arbitrary new success receipt.

```bash
"$REVIEW_PY" "$STUDY_PHASE/validation/v6_output_review_harness_v1/review_batch.py" \
  --repo "$STUDY_PRODUCER" \
  --report "$STUDY_PHASE/supplement_report_v1" \
  --completion-receipt "$STUDY_PHASE/validation/supplement_rendering_revision_v6/exit.json" \
  --start-receipt "$STUDY_PHASE/validation/supplement_rendering_revision_v6/start.json" \
  --start-sha256 ab2d3b97b8be26a0fbe4bfa7f2fabaccefb8de602493f76a6d07d2682595bfc0 \
  --expected-numeric "$STUDY_PHASE/validation/protocol_name_recovery_v6/expected_numeric_hashes.json" \
  --output "$REVIEW_OUT/across_lean_000" \
  --group pairs --kind across_lean --offset 0 --limit 12
```

Use fresh output directories. Exact batch schedule, with records sorted by the harness's public identity:

- Across lean: offsets 0,12,24,36 (limit12), then48 (limit8).
- Curve strength: offsets0,12,24,36,48,60,72,84,96,108,120,132,144 (limit12), then156 (limit8).
- Shaft-twist SD: the same fourteen offsets/limits as curve strength.
- Cells: `--group cells`, omit `--kind`; offsets0/12 (limit12), then24 (limit2).

There are36 batches. Run one bounded batch at a time, or at most two independent review-only workers if disk/memory allows. No busy polling. Each pair produces three legible two-chart contact sheets; each cell produces two. Without reuse this is1,204 contacts covering every PNG. Batch receipts record exact selected records, all16/12 asset hashes, extracted SVG text, source/terminal/ledger hashes, and pending visual status.

## Reusing Prior Visual Work Safely

Build a review-coverage ledger keyed by **new canonical asset path + SHA256 + ordered scientific identity + artifact kind**. Match old evidence by scientific identity and kind, never just old path or digest alone. Byte equality does not repair a caption that was already wrong. Each reused acceptance must cite the exact prior receipt and its scope/status; a whole rejected publication cannot be called accepted.

Candidate prior evidence under `phase2/validation`:

1. `supplement_v5_cell_graphics_review_v1/{review.json,cell_checks.json}`: all26 cells/312 artifacts,104 PNGs; source-qualified captions. Reuse exact byte-matched accepted individual assets only. Names for these actual26 cells were proven unchanged by the new identity helper.
2. `supplement_v5_across_lean_graphics_review_v1/{review.json,inventory.json}`: all56 pairs inspected. **Academic density row-gutter defect was retained**, so those old density images cannot supply acceptance. Social-density acceptance can be reused only with exact ordered protocol/target/count identity and bytes. Overhead scope must be read from the individual review, not assumed.
3. `supplement_v5_curve_graphics_first12_v1/{review.json,inventory.json}`: only those exact12 identities; social density was legible, long-caption academic/overhead-social defects were not accepted. Do not reuse rejected kinds.
4. `selected_renderer_representative_v1`, `selected_renderer_canvas_binding_v1`, and `protocol_identity_final_qualification_v1/representative_review.json`: four real representative pairs,28 selected outputs/12 PNGs with repaired gutter/captions and numerical artist parity. Reuse only exact source-identity/kind/byte matches. This is not acceptance of all384 layouts.
5. Any later partial reused-density review must first be checked for its protocol-collision finding. A failed batch has no blanket approval. Its explicitly accepted, noncolliding, hash-bound files may be considered individually; otherwise review anew.

PDF timestamps can legitimately prevent byte equality. Do not strip timestamps and label the result byte-identical. Such vectors follow the format-validation path below. Old v3/v5 paths now live under preserved rejected-publication directories; use relocation receipts and old bound hashes, not presumed current canonical paths.

Efficient order: match accepted hashes first; avoid re-opening an unchanged already-accepted PNG solely to fill another contact sheet. The harness itself currently makes all contacts; leave it unchanged. A separate coverage index can mark which contacts need first inspection and which already have precise accepted evidence. Every one of2,408 PNGs must end with either new explicit visual review or valid exact-byte review reuse.

## Visual and Format Checks

For each newly reviewed contact, log actual filenames and findings, not just 'contact generated'. Open full-size images whenever labels cannot be read at contact scale. Mandatory full-size representatives span all three clubs, all three contrast kinds, both noise/SD settings, strong curves, narrow/high-range axes, long correlated-noise captions, cross-protocol Driver speed controls, PW aliases, and minimum-support cases. Every flagged image receives full-size inspection; there is no representative-only exemption for the remaining PNG inventory.

Check ordered A/B conditions and source protocols, numeric target and within-lean/fixed-target basis; common-six density conditioning versus raw overhead trajectory population; individual cell versus common-three SG counts; actual sampled-path/dot limits; units; quantitative axis/tick legibility and shared physical scales; HDR/density legends; main/marginal neighbor and boundary separation; row-label/title gutters; covariance captions and social canvas clipping. Never infer that a narrower pattern automatically gives higher SG.

The batch harness automatically checks social-density protocol/target/count captions in SVG, but **does not fully validate overhead captions, target-basis wording, every PDF page or all SVG geometry**. Those remain explicit review tasks. Verify all2,024 PDFs open and expected page counts, all2,024 SVGs parse with nonzero viewports, and extract scientific labels/metadata for identity/count checks on each available vector. Render representative vectors for every exporter/kind and every detected structural/text anomaly; if their layout differs from the already-reviewed PNG, expand to all vectors of that affected exporter. Record clearly which formats were visually inspected versus structurally checked under a qualified shared-figure exporter. Do not claim every vector was visually viewed if it was not. No asset is skipped in byte/inventory/format verification.

## Closing Coverage Without False Acceptance

Aggregate the36 batch receipts and per-asset visual decisions. Require exact disjoint union of all384 ordered pairs and26 unique cells, all6,456 final-ledger paths and hashes, no duplicate normalized path, no missing/extra planned artifact, and all seven numerical files matching the committed expected map. Reconcile caption identities to the numerical JSON, not the filename alone. Record all aliases as reuse, not extra cells.

Recheck final receipt/ledger/source/input hashes and every output hash after review; compare to initial accepted terminal snapshot. Then call the strict public loader again only if review duration or new evidence warrants the end-boundary verification. A source/asset mutation invalidates affected acceptance rather than being silently rehashed.

Final graphics acceptance is separate from report composition/PDF page review, existing editable-LaTeX compilation, archive/transport packaging and PR delivery. Existing primary/control graphical acceptance is reusable through its unchanged manifest and all-output hashes in `original_final_graphics_independent_review_v6`; it does not qualify supplemental graphics. Pending or failed components stay named in the final review. No universal full-publication claim until those remaining gates are actually closed.
