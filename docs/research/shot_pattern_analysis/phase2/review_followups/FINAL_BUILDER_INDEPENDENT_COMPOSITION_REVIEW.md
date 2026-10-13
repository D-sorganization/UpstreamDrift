# Final Builder Independent Composition Review

Status: **Accepted for Component Contracts**; no final publication or visual acceptance.

At presentation commit `200fbf028f319b475e5cba6d6144b28ec94afb21`, both builder modules and both report test files exactly match MAIN `f265cf7839` bytes. Independent focused tests passed (32 passed, 2 slow tests deselected); the slow publication endguard was then run separately and passed. The remaining deselected slow test concerns rendering and is outside this source-contract review.

The builder preserves comparison kind, exact ordered cell/protocol identities, and Straight/Draw/Fade social pattern metadata. Using the actual v3 producer ledger, the selector returned ten distinct PNG links, including six paired links and all three physical-pattern social densities. It excludes unrelated condition pairs and rejects inconsistent presentation/pattern labels or escaping paths. Academic and social overhead remain distinct from endpoint density.

Source inspection confirms receipt hashing before supplement loading, generation-time binding of original and supplemental source records including unavailable cells, post-source-validation image hash checks, and publication-endguard revalidation. Original control figures retain the control protocol despite sharing the `final` stage. Output publication receipts bind report outputs, figure manifests, reporting source hashes, original plans, and the supplemental receipt. No new actionable composition defect was found.

All 54 execution source hashes and the native binary hash still match the accepted supplemental source transition. Runtime equivalence was not rerun. No accepted execution source, inventory fixture, immutable receipt, or data file was changed by this review.

## Evidence and Limits

`final_builder_independent_composition_review.json` binds the reviewed files and transition receipt. The two `FINAL_BUILDER_INDEPENDENT_*_TESTS.txt` logs preserve process results; `final_builder_real_ledger_probe.json` preserves the real-ledger selector result. These are new records; prior acceptance evidence is unchanged.

The actual final report cannot yet be qualified without completed controls, qualified supplemental runs and final source-bound asset inventories. Known canonical marginal tick adjacency and phone-scale density readability remain visual publication gates. The v3-ledger probe proves metadata compatibility only; it does not approve v3 captions or final scientific conclusions.

## Reproduction

From the presentation worktree, with the exact committed inventory and its hash-bound plans restored:

```bash
OPENBLAS_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy /tmp/shot-pattern-validation-venv-20261009/bin/python -m pytest -q -o addopts='' tests/research/test_shot_pattern_final_report.py tests/research/test_shot_pattern_final_report_assets.py tests/tools/shot_pattern_analysis/test_matched_supplement_pipeline.py -m 'not slow'
OPENBLAS_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy /tmp/shot-pattern-validation-venv-20261009/bin/python -m pytest -q -o addopts='' tests/research/test_shot_pattern_final_report.py::test_publication_boundary_rejects_mutated_png_and_supplement_receipt
```
