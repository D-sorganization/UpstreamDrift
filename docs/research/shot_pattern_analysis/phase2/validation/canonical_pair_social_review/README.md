# Canonical Paired Social Component Review

## Scope and Decision

The wrapper `04c6f17aa4` reuses the accepted canonical density renderer for
three two-condition figures. Five tests failed on the initially absent module
and then passed; Astra independently repeated all five. Ruff and scoped mypy
passed for the wrapper. This is a reporting component, not final-study or
empirical acceptance.

The actual saved Driver, Lean 5°, Curve ×1, Combined-2 pair contains 10,000
attempted shots per physical pattern per condition. The figures use exactly
9,939 indices supported in all six populations. The two conditions are shaft
twist SD 1° and 2°; path SD is 0.5°, speed SD is 2 mph and strike SD is
0.25 in on each axis. Coordinates remain raw simulator output.

Each physical-pattern figure shares histogram bins, axes and density/marginal
scales between conditions. Scales can differ between the three files; their
captions disclose this. All common endpoints enter density/HDR/marginals;
only the colored dot overlay is capped at 600.

Root inspected the three initial pattern images and the explicit-unit draw
revision. Astra independently inspected the draw image. The retained previews
are byte-identical to the explicit-unit revision. The images clearly show
population, units, sample-derived HDR and model-conditional limitations.
**Final visual qualification remains pending:** the rightmost main-axis tick
and adjacent marginal zero tick slightly overlap in the draw view. Issue
#11988 tracks correction, final integration and visual review. The canonical
renderer is execution-hashed; repair it after numerical execution or through
a separately accepted source transition, preserving existing receipts.

## Provenance

`review_receipt.json` binds the original matrix plan and both input manifests,
the unchanged accepted 54-source/native/runtime execution snapshot, wrapper,
saved-data readers, executed reproducer and all nine output hashes. Input,
report-source and execution stability were checked after rendering. The three
PNG previews are retained here; intermediate PDF/SVG outputs remain in the
recorded temporary directory until final packaging. The independent test log
is `astra_contract_tests.txt`. No new shots were simulated.

The first tracked reproducer run succeeded but had one Ruff complaint about
its CLI `print` call. It was replaced with `sys.stdout.write`; the final
source-bound reproduction passed and its PNG hashes matched the preceding
explicit-unit render. Earlier receipts remain historical, not overwritten.

## Reproduction

From the presentation worktree, choose a fresh output directory:

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
/tmp/shot-pattern-validation-venv-20261009/bin/python \
docs/research/shot_pattern_analysis/phase2/validation/render_canonical_pair_social.py \
/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift \
/tmp/canonical-pair-social-new-review
```

The first argument contains the saved original cells and immutable transition
receipt. Source changes require a new review receipt; a successful rerender
does not qualify the complete experiment or resolve the cosmetic finding.

The exact executed reproducer is preserved at `76eb0eff77`. Its later typing-only
annotation correction passes mypy and has normalized execution-AST equivalence
in `reproducer_typing_followup.json`; it does not rewrite the historical
graphic receipt or claim another executed render.
