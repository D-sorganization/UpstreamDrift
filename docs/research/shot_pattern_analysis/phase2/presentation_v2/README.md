# Conventional Shot-Pattern Presentation

This directory contains presentation-only replacements for the 13 figures embedded in `shot_pattern_analysis.tex`, plus the one-page graphical summary. The renderer reads frozen `final_v1`, `control_final_v1`, and `supplement_final_v1` shot data. It does not simulate or alter study results. The source files remain in their original `[downrange, lateral]` metre frame; display axes are horizontal lateral position (right positive) and vertical downrange distance.

Run from the repository root after restoring the final data archives:

```bash
MPLBACKEND=Agg python -m docs.research.shot_pattern_analysis.phase2.presentation_v2.render
```

`mapping.json` names every former embedded figure, its replacement PDF, its source file hashes, and hashes of both `render.py` and `summary.py`. `original_paths.json` freezes the former figure order so rerendering does not depend on subsequent LaTeX edits. Every input cell's `shots.csv.gz` and `flight_samples.json` must match that cell's saved manifest before rendering. The six-panel comparisons use the same shot-index intersection supported in all six populations and check its count against the supplement comparison receipt. The summary's six relative-SG ranges are checked against the numerical table in the LaTeX source at render time.

Density panels use all supported endpoints, common panel bounds and colour scale, equal metre scaling, and binned empirical highest-density contours. Flight panels use saved illustrative trajectories with stated lateral magnification. The graphical summary shows selected-condition ranges; they are not pooled confidence intervals or per-round gains. All PDFs have companion PNGs for review and social sharing.

Validation: the first axis-mapping test was red on import before the renderer existed; after implementation, the seven focused tests passed. `ruff check` and `ruff format --check` passed. The rendered driver density, six-panel variability comparison, forward-lean flight plate, and graphical summary were inspected visually at publication size.

## Supplemental Flight Replay

The optional `supplemental_traces_v3` receipt records 100 supported shot indices common to all five selected cells and all three patterns. The replay excludes shot indices 0–4, which independently reproduce the five frozen sampled paths; each cell's nominal saved path is also reproduced. The original `flight_samples.json` files and their manifests stay unchanged. Replay uses the saved delivery inputs and matched physics solver, verifies the recorded native binary and numerical runtime, and requires saved-path and endpoint agreement within 1e-6 m. The output is illustrative, model-conditional flight geometry, not an empirical confidence interval.

From the repository root, with the recorded Python 3.12.13, NumPy 2.2.6, SciPy 1.18.1, Numba 0.68.0, llvmlite 0.50.0, and hash-matched native binary available, reproduce into a fresh directory:

```bash
MPLBACKEND=Agg /tmp/shot-pattern-validation-venv-20261009/bin/python - <<'PY'
from pathlib import Path

from docs.research.shot_pattern_analysis.phase2.presentation_v2.replay import (
    execute_replay,
    plan_replay,
)
from docs.research.shot_pattern_analysis.phase2.presentation_v2.supplemental import (
    EXPECTED_CELLS,
)
from src.tools.shot_pattern_analysis.matched_export import matched_source_snapshot
from src.tools.shot_pattern_analysis.matched_physics import MatchedShotPhysics

plan = plan_replay(EXPECTED_CELLS, target=100)
execute_replay(
    plan,
    MatchedShotPhysics(),
    Path("docs/research/shot_pattern_analysis/phase2/supplemental_traces_v3_reproduction"),
    runtime=matched_source_snapshot(),
)
PY
```

The destination must not already exist. The replay receipt binds its trajectories, selected IDs, frozen inputs, solver source and runtime, and stated parity tolerances. To render the verified supplemental paths into the 13 presentation plates, pass the explicit flag:

```bash
MPLBACKEND=Agg /tmp/shot-pattern-validation-venv-20261009/bin/python -m docs.research.shot_pattern_analysis.phase2.presentation_v2.render --supplemental docs/research/shot_pattern_analysis/phase2/supplemental_traces_v3
```
