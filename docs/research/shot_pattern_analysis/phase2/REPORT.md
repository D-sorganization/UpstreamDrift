# Matched Shot Patterns: Final Conditional Simulation Report

## Results Pending

The 10,000-shot-per-pattern primary and fixed-target control matrices are being generated. This editable report is a framework, not a results claim. The final report will be built only after every planned cell has a validated completed or explicitly unavailable status.

The primary comparison matches Straight, Draw, and Fade to one physical zero-error landing point within each club and lean setting. The fixed-target control attempts the zero-lean target across lean settings. Unsupported shots and infeasible nominal matches remain visible in the final population accounting. Phase-one constant-delivered-loft results are a separate historical experiment.

The [Scientific Design](DESIGN.md) records the model, assumptions, and acceptance gates. Per-cell source-bound manifests and the independent validation record will support the final result tables. Simulation intervals describe sampling uncertainty; they do not establish empirical player performance or a universal preferred shot shape.

## Rebuild Commands

```bash
python -m scripts.research.shot_pattern_analysis.build_final_report docs/research/shot_pattern_analysis/phase2/final_v1 docs/research/shot_pattern_analysis/phase2/control_final_v1 docs/research/shot_pattern_analysis/phase2/REPORT.md
python -m scripts.research.shot_pattern_analysis.build_pdf docs/research/shot_pattern_analysis/phase2/REPORT.md docs/research/shot_pattern_analysis/phase2/REPORT.pdf
```
