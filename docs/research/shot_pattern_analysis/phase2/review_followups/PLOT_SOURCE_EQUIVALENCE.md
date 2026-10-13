# Plot Source Equivalence Review

## Scope

This change is prepared in the isolated bounds worktree. It does not modify the
original running worktree, its accepted source snapshot, or historical receipts.
Integration still requires the reviewed source transition after the original
workers finish.

`academic_plots.py` and `matched_report_plots.py` have formatting changes only;
their parsed abstract syntax trees are identical to the recorded baseline.
`density.py` makes NumPy's existing float64 grid-edge dtype explicit in the
`linspace` call. The two incompatible-array-type errors were reproduced before
the edit; scoped mypy subsequently passes for all three source files.

## Verification

The [Machine Receipt](PLOT_SOURCE_EQUIVALENCE.json) records before/after source
hashes and the verification script hash. The density AST differs only by the
explicit `dtype=np.float64` keyword. Eighty-four deterministic comparisons cover
constant, singular, one-point and three-pattern clouds with square/rectangular
grids. Every returned array dtype, grid edge, density, HDR threshold, achieved
mass, resolution and bound is exactly equal. This evidence supports the narrow
format/type correction; it does not qualify unrelated source changes.

The 14 existing academic and matrix-plot tests pass. Their output is retained in
[Test Log](PLOT_SOURCE_TEST_LOG.txt). Ruff passes for the three source files and
the verification script. No physics, noise, scoring, tolerance or sample-count
change is part of this correction.

## Reproduction

The historical 84-comparison proof can be reproduced from the exact baseline
and post-typing Git commits below. Both commits and all three post-typing source
digests are fixed in the verifier before it executes density code. Without
`--candidate`, the verifier checks working-tree source freshness and currently
fails closed: later density refactoring changed the AST beyond this narrow
float64 spelling proof. A historical pass does not renew current-source or
scientific qualification, and the original machine receipt remains unchanged.

Run from the repository root using the qualified environment:

```bash
OPENBLAS_NUM_THREADS=1 /tmp/shot-pattern-validation-venv-20261009/bin/python docs/research/shot_pattern_analysis/phase2/review_followups/check_plot_source_equivalence.py --baseline 4fc22c694404217208534b78308d31ea8985e7d3 --candidate ecc82796cec7e9119e2dd2fb0e8bb2582851761f --output /tmp/plot-source-equivalence-historical.json
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 /tmp/shot-pattern-validation-venv-20261009/bin/python -m pytest tests/tools/shot_pattern_analysis/test_academic_plots.py tests/tools/shot_pattern_analysis/test_matched_report_plots.py -o addopts='' -q
```
