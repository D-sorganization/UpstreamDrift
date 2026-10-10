# Matched Shot Patterns: Final Conditional Simulation Report

## Results Pending

The original 10,000-shot-per-pattern primary and fixed-target control matrices are complete and passed full independent saved-data audits (324 and 72 planned cells). Their lossless archives also passed fresh restoration. This editable report remains a framework until the supplemental trials and final publication review are complete.

The primary comparison matches Straight, Draw, and Fade to one physical zero-error landing point within each club and lean setting. The fixed-target control attempts the zero-lean target across lean settings. Unsupported shots and unsuccessful bounded nominal searches remain visible in the final population accounting. A failed bounded search does not establish physical impossibility. Phase-one constant-delivered-loft results are a separate historical experiment.

The [Search-Bounds Review](review_followups/WEDGE_MATCH_BOUNDS_REVIEW.md) identifies feasible larger-curve wedge and driver comparisons excluded by the original search bounds. A separate [Supplemental Inventory](review_followups/SUPPLEMENTAL_EXECUTION_INVENTORY.json) declares 26 additional cells and four references to reused data. Its three 32-shot pilots and full audits passed, as did the predeclared numerical refinement. Independent final-run acceptance and the full supplemental simulations remain pending. Supplemental results must retain their protocol labels and cannot silently replace the original matrix records.

The [Scientific Design](DESIGN.md) records the model, assumptions, and acceptance gates. Per-cell source-bound manifests and the independent validation record will support the final result tables. Simulation intervals describe sampling uncertainty; they do not establish empirical player performance or a universal preferred shot shape.

## Interpretation Required for the Final Results

Matched zero-error landings do not mean every noisy shot carries the same distance. The report must compare the actual carry distributions and target errors, alongside nominal face, path, speed, launch and spin. Matching removes the initial landing-distance difference as an explanation for dispersion changes; changing several delivery conditions together does not isolate curvature as a causal variable.

The final comparisons must answer whether increasing shaft-twist SD from 1° to 2°, doubling curve strength, changing shaft lean, or adding correlated delivery errors changes the relative pattern outcomes. Left-long and right-short behavior must be evaluated from the simulated endpoints and the documented face/loft geometry for each club. Static loft, delivered loft and rotation about the shaft are different quantities.

Strokes-gained differences use the declared hypothetical tee-shot context for driver and approach context for irons/wedge. The report must state course geometry, supported population, paired count, effect per opportunity and uncertainty. Any incomplete physical population remains explicit; a result conditional on supported shots is not an estimate for all attempted shots. Model assumptions, course sensitivity and unavailable empirical validation must remain distinct from Monte Carlo sampling intervals.

## Rebuild Commands

```bash
python -m scripts.research.shot_pattern_analysis.build_final_report docs/research/shot_pattern_analysis/phase2/final_v1 docs/research/shot_pattern_analysis/phase2/control_final_v1 docs/research/shot_pattern_analysis/phase2/REPORT.md
python -m scripts.research.shot_pattern_analysis.build_pdf docs/research/shot_pattern_analysis/phase2/REPORT.md docs/research/shot_pattern_analysis/phase2/REPORT.pdf
```
