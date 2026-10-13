# Illustrated Article Review

The canonical editable standalone source is `../../../shot_pattern_analysis.tex`.
The illustrated edition contains 13 figure plates and 33 pages. `review.json`
binds the source, compiled PDF, and all final page renders. The separate Astra
and Sol receipts plus `page_reconciliation.json` retain the visual review scope.
No simulation or numerical result was recomputed for this presentation update.

## Reproduction

From `docs/research/shot_pattern_analysis`, with the saved figure assets restored:

```bash
mkdir -p phase2/validation/illustrated_article_v1/final_build
tectonic --keep-logs --keep-intermediates \
  --outdir phase2/validation/illustrated_article_v1/final_build \
  shot_pattern_analysis.tex
```

The local compiler is Tectonic 0.15.0; it completed with zero warnings.
`source_bundle.json` identifies the portable source-and-13-figure archive.
PDFs, page renders, and the ZIP remain external artifacts rather than tracked binaries.
The report is copied to `phase2/ILLUSTRATED_REPORT.pdf` and the user's Desktop.

## Remaining Limits

The built-in editor still fails before source parsing while fetching its
Tectonic dependency bundle (#12120). Local success does not clear that issue.
Empirical qualification (#11984) and repository release gates remain open.
The earlier text/table `REPORT.pdf` is historical and is not the illustrated edition.
