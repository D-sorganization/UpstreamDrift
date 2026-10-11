# Conventional Shot-Pattern Presentation

This directory contains presentation-only replacements for the 13 figures embedded in `shot_pattern_analysis.tex`, plus the one-page graphical summary. The renderer reads frozen `final_v1`, `control_final_v1`, and `supplement_final_v1` shot data. It does not simulate or alter study results. The source files remain in their original `[downrange, lateral]` metre frame; display axes are horizontal lateral position (right positive) and vertical downrange distance.

Run from the repository root after restoring the final data archives:

```bash
MPLBACKEND=Agg python -m docs.research.shot_pattern_analysis.phase2.presentation_v2.render
```

`mapping.json` names every former embedded figure, its replacement PDF, its source file hashes, and hashes of both `render.py` and `summary.py`. `original_paths.json` freezes the former figure order so rerendering does not depend on subsequent LaTeX edits. Every input cell's `shots.csv.gz` and `flight_samples.json` must match that cell's saved manifest before rendering. The six-panel comparisons use the same shot-index intersection supported in all six populations and check its count against the supplement comparison receipt. The summary's six relative-SG ranges are checked against the numerical table in the LaTeX source at render time.

Density panels use all supported endpoints, common panel bounds and colour scale, equal metre scaling, and binned empirical highest-density contours. Flight panels use saved illustrative trajectories with stated lateral magnification. The graphical summary shows selected-condition ranges; they are not pooled confidence intervals or per-round gains. All PDFs have companion PNGs for review and social sharing.

Validation: the first axis-mapping test was red on import before the renderer existed; after implementation, the seven focused tests passed. `ruff check` and `ruff format --check` passed. The rendered driver density, six-panel variability comparison, forward-lean flight plate, and graphical summary were inspected visually at publication size.
