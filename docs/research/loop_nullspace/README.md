# Closed-Loop Null-Space Exploration Reference

Start with the [maintained LaTeX reference](loop_nullspace.tex). It is a
separate, editable research record (see `AGENTS.md`, Modeling Reference
Documentation); the engineering design manual source remains
`manuals/upstreamdrift`, where the pathway is tracked by the registry blocker
`UP-NSX-loop-nullspace-exploration` and the provisional chapter
`65-loop-null-space.qmd`.

The document specifies the mathematics for epic #12593 (NSX): the general
allocation equation and its dimension counts, the three-case zero-side
analysis, the exact grasp-map kernel on the bare club, the invariants and their
validity gates, the gross-power identity, constrained counterfactuals, envelope
geometry, reaction-restriction semantics, the canonical closure convention, a
rank audit of the pendulum constraints and the citation verification table.

| Item                         | Status                                                                  |
| ---------------------------- | ----------------------------------------------------------------------- |
| Formulas and goldens         | Reproduced by exact sympy derivations; regenerated and compared by tests |
| Citations                    | Verification table in the reference; unverified sources are not used     |
| Scientific qualification     | None; no numerical result about any golfer                               |
| External validation          | Deferred (NSX-05)                                                        |

## Contents of This Directory

- `loop_nullspace.tex` is the reference.
- `publication_gate.md` is the checklist that must pass before any NSX result is published externally.
- `hit_and_run_sampling.md` is the NSX-70 research spike (hit-and-run on a
  low-rank temporal basis); `hit_and_run_spike.py` reproduces it and
  `hit_and_run_results.json` holds the measured diagnostics. Exploratory.
- `contracts.md` is the per-child contract appendix for NSX-02 to NSX-62:
  signatures, preconditions, postconditions and golden names. It is split into
  `contracts_trajectory.md` (NSX-23 to NSX-61 and NSX-62 Ensembles) and
  `contracts_variability.md` (NSX-80 to NSX-98 and NSX-62 Inertial Parameter
  Ensemble) to stay under the documentation size budget.
- `muscle_level_view.md` is the exploratory scoping report for a muscle-redundancy layer above the envelopes (NSX-73, #12591).
- `closure_alternatives_report.md` compares grip models that each have their own
  closure kinematics (NSX-72); its records are in `records/`.
- `inverse_optimal_control_report.md` is the NSX-71 research report: inverse
  optimal control of the allocation weights with a pre-registered held-out
  evaluation (result: the effort-plus-rate family is rejected on the anthro
  record; one swing, no measured torques).
- `derivations/` holds the sympy scripts. They import no repository production
  code. `generate_goldens.py` writes the golden JSON files.
- The golden files are in `tests/fixtures/loop_nullspace/goldens/`.
- The seed registry of hypotheses, conventions and presets is
  `src/config/loop_nullspace_registry.json`; the decision record is
  `docs/adr/0054-closed-loop-null-space-exploration.md`.

## Reproduce

```bash
python3 docs/research/loop_nullspace/derivations/generate_goldens.py          # rewrite goldens
python3 docs/research/loop_nullspace/derivations/generate_goldens.py --check  # fail if stale
python3 -m pytest tests/unit/loop_nullspace -n 0 -q
```

Requires `sympy`, `numpy` and `scipy`. The generated PDF is not committed.

## Build Locally

CI typesets references with a pinned Tectonic whenever a
`docs/research/**/*.tex` file changes. To reproduce it:

```bash
tectonic -X compile --outdir build docs/research/loop_nullspace/loop_nullspace.tex
```

## Non-Claims

No numerical result about any golfer. No claim that any single hand split is
the golfer's. No stiffness or co-contraction inference from net joint torques.
