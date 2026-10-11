# Model-Aware Matching (MOSAIC) Reference

Start with the [maintained LaTeX reference](model_aware_matching.tex). It is a
separate research and experiment record (see `AGENTS.md`, Modeling Reference
Documentation); the engineering design manual source remains
`manuals/upstreamdrift`.

The document specifies MOSAIC — Model-aware, Observation-consistent,
Simultaneous Anthropometry, Inertia and Control estimation — the
variable-projection inverse-dynamics collocation estimator implemented in
`src/shared/python/estimation/mosaic/`, its identifiability results, the
canonical ZTCF/ZVCF nomenclature, the qualification gates, measured planar
reference results, the engine integration path, and the audit of the DIME epic
(#11421).

| Item | Status |
| ---- | ------ |
| Equations and kernels | Verified by 36 behavioural tests against closed-form dynamics and synthetic truth |
| Planar reference results | Measured; reproducible with `python3 -m pytest tests/unit/estimation/mosaic -q` |
| Human capture fits (capture-A, capture-O) | Not started; gated by the follow-up epic |
| Engine-backed regressors, floating base, contact | Not started; roadmap in the reference |

Compile with `pdflatex model_aware_matching.tex` (two passes). The generated
PDF is not committed.

See also the sibling [Same-Input Cross-Engine Dynamics Parity Reference](../same_input_parity/same_input_parity.tex).

## Build Locally

CI typesets this reference with a pinned Tectonic (`.github/workflows/latex-references.yml`)
whenever a `docs/research/**/*.tex` file changes. To reproduce it:

```bash
tectonic -X compile --outdir build docs/research/model_aware_matching/model_aware_matching.tex
# or, with a TeX Live install:
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build docs/research/model_aware_matching/model_aware_matching.tex
```
