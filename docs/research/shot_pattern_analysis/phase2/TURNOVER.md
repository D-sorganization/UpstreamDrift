# Matched Delivery Study Turnover

## Current State

The phase-two goal is active and incomplete. Worktree: `/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift`; branch: `codex/shot-pattern-matched-delivery`, based on phase-one PR [#11941](https://github.com/D-sorganization/UpstreamDrift/pull/11941) at `10c7d61aa686e456b7962f1d9aa9ecfda4e8405f`. The prior PR is open and held; phase two has not been pushed. Do not edit the primary checkout or vendor submodule.

Read [Design](DESIGN.md), [Astra Review](ASTRA_REVIEW.md), [Validation](validation/README.md), and [Issue Records](issues/). AGY Gemini 3.8 Flash High filed #11972–#11976, #11984 and #11988 with the user's authorized personal GitHub identity. #11984 is an open external measurement obligation, recorded in `docs/development/planning/DV-11984.md`; do not close it from software evidence.

## Implemented and Verified

- Static lofts 10°/34°/46°, whole-club lean, raw endpoint face/speed matching without aiming rotation, paired seven-variable Gaussian covariance, complete unsupported-population accounting, conditional scoring and atomic source-bound exports.
- Full finite-head incremental Coulomb contact with energetic restitution is primary; maximum-dissipation/Newton is a comparator. Numba/Python parity, momentum/work/energy/frame checks pass. Zero-spin airborne shots are supported with an undefined spin-axis flag/null serialized tilt.
- Matched GUI, complete covariance controls, input/configuration round-trip, cancellation, nominal matching display, two contact laws, density/HDR/marginal/overhead figures and matrix overview plots are implemented. Figures explicitly identify illustrative simulation and assumed equipment.
- Shared saved-cell integrity loading, course-geometry sensitivity and across-lean paired comparisons are implemented. Across-lean estimates use one supported-index intersection across all six physical populations in a cell pair.
- Independent refinement: 243 deliveries, 225 supported and 18 consistently nonairborne; 128→512 contact change ≤0.00002013 m; dt 0.02→0.005 landing change ≤0.042923 m. All pass the predeclared 0.05 m gate. Ninety nominal searches produced 72 matches, 10 domain failures and 8 bounded-search failures; these are not proofs of global unreachability.
- Independent head-parameter sensitivity and analytical geometry derivatives are saved in `validation/`. Fixed twist SD does not imply fixed face SD. Driver closure gives left/short for these low-launch presets; iron/wedge closure gives left/long. Neither is a universal player claim.
- Broad focused suite: 256 passed. Frozen physics/GUI/figure acceptance suite: 124 passed after the SG heatmap sample-count correction. The exact current source snapshot is accepted by Astra in immutable `FINAL_RUN_ACCEPTANCE.md` and validated by `final_acceptance.json`.

## Publication Review Progress

The GUI fits a 1000 × 760 launcher window with scrollable controls and readable correlation headers; 27 UI tests, focused Ruff and mypy pass. Offscreen screenshots and the review receipt are in `validation/gui_layout_review.json`. Final report and figure builders are implemented; Astra and Sol corrected publication-only labeling, protocol guards, input integrity and planned-versus-attempted count reporting; the combined 20 publication tests and both builders’ Ruff/mypy checks pass (Astra review `a4d9502a95`). These changes do not modify the frozen execution snapshot. The preliminary pre-PR run remains failed/incomplete; inventory and audit-script typing findings are fixed, while frozen-source formatting/type-only fixes wait until production completes. Unrelated API authentication fixture failures are tracked in [#11995](https://github.com/D-sorganization/UpstreamDrift/issues/11995), with source-equivalent baseline evidence in `review_followups/`.

## Completed Pilots

`pilot_v1`: all 324 planned cells accounted for; 270 completed, 54 unavailable; all 25,920 sampled rows supported. Independent audit checked 1,770 artifact hashes, exact seeded input pairing, statistics/covariance and direct public-API scoring. Original tool sessions returned 143 despite complete artifacts; the audit accepts artifacts, not successful process exit.

`control_pilot_v1`: explicit 72-cell common-distance plan using shaft-only and combined-2-mph profiles, both twist SDs/curve strengths, all clubs/leans. It completed with exit code 0 (`/tmp/p2-control-exit.json`): 52 completed and 20 unavailable. Common targets are saved in `common_targets.json`. Independent control audit and 480 additional nonzero-lean scoring checks passed.

Numerical sources are unchanged between pilots. Only declared plotting/export-snapshot/matrix-selection sources changed; `validation/source_transition_review.json` records the exact union-of-paths diff. The new snapshot includes the final SG-count annotation fix and is independently accepted. Never rewrite historical execution hashes.

Pilot data and figures are committed losslessly as bounded archives in `data_archives/`; see its README for verification and restoration. Unpacked local roots remain present and ignored by Git. Pilot figure reviews are under `pilot_v1/figure_review/`. Use the final simulation-labelled rerenders; earlier layouts are explicitly superseded. Final 10,000-shot matrices are running; do not treat partial outputs as final results.

## Immediate Next Steps

1. Execution acceptance is complete, with 11 hashed evidence files. Keep the immutable acceptance note, evidence and all scientific snapshot sources unchanged.
2. Run the final within-lean 324-cell matrix and 72-cell common-distance control with 10,000 shots per pattern. Expected nominally completed cells from pilots are 270 and 52, but retain every final unsupported sample and matching failure. No resampling, clipping or bounds widening.
3. Use at most two logged detached worker processes on this four-CPU shared host. Each job must write an explicit exit receipt. Matrix resume verifies manifests, inputs and execution. Keep all hashed sources frozen while jobs run. Foreground tool-session lifetime is not a reliable long-job receipt.
4. Recompute final statistics, common-population lean comparisons, course sensitivities and selected academic/social figures. Complete polished linked Markdown/HTML/PDF/LaTeX reporting and an independent final-data/claims audit. Keep statistical intervals separate from model and course uncertainty.
5. Package raw results with hashes, update current handoffs/change fragment, run required gates, and publish through ready PR(s)/merge-queue guard. Do not close issues before merged acceptance evidence.

## Running Jobs

Started 2026-10-09 at approximately 17:22 UTC (10:22 PDT). Detached wrapper `/tmp/p2-final-worker.py`: worker 0 PID 1155177, worker 1 PID 1155178. Logs `/tmp/p2-final-worker{0,1}.log`; each worker first runs its `final_v1` shard, then its `control_final_v1` shard. Each command writes `/tmp/p2-{final_v1,control_final_v1}-shard{0,1}-exit.json`; aggregate worker receipts end in `-exit.json`. Read logs and inspect processes before resuming anything. No more than two workers. A completed manifest is authoritative only after integrity validation; any failed command must be investigated.

## Environment and Reproduction

Use `/tmp/shot-pattern-validation-venv-20261009/bin/python`: Python 3.12.13, NumPy 2.2.6, Numba 0.68.0, llvmlite 0.50.0. Optional acceleration extra is `shot-pattern-acceleration`. The optional `shot-pattern-report` extra and `scripts.research.shot_pattern_analysis.build_pdf` renderer are implemented and tested; they do not alter scientific simulation sources.

Set `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy OPENBLAS_NUM_THREADS=1`. Never launch GUI processes.

```bash
python -m src.tools.shot_pattern_analysis.matched_matrix OUTPUT \
  --stage final --shots 10000 --acceptance ACCEPTANCE_JSON \
  --shard-index 0 --shard-count 2 --figures selected

python -m src.tools.shot_pattern_analysis.matched_matrix CONTROL_OUTPUT \
  --stage final --shots 10000 --acceptance ACCEPTANCE_JSON \
  --profiles shaft_only combined2 --common-targets \
  docs/research/shot_pattern_analysis/phase2/common_targets.json \
  --shard-index 0 --shard-count 2 --figures none
```

Run the second shard with index 1. Review current `--help` before execution. Source snapshots include numerical sources, rendering/orchestration, native binary identity and scientific runtime versions. Preserve every accepted source snapshot and its evidence files unchanged during execution.

The five-page `matched_delivery_reference.pdf` was generated with the tested ReportLab renderer and all five pages visually checked; see `validation/design_pdf_review.json`. It is separate from LaTeX compiler acceptance.

The standalone LaTeX reference is regenerated from `DESIGN.md` via `pandoc --from=gfm+tex_math_dollars --to=latex --standalone`. Built-in compilation is unverified because Tectonic cannot fetch its uncached bundle; the separate ReportLab PDF must be rendered and visually reviewed before delivery.

## Final Publication Preparation

- `scripts/research/shot_pattern_analysis/build_final_report.py` (latest correction `2de9960b77`) validates exactly 324 primary and 72 control planned cells, final stage and 10,000 planned shots per pattern before generating concise `REPORT.md`, `REPORT_CELLS.csv` and `REPORT_LEAN.csv`. Seven tests pass; unavailable rows distinguish planned 10,000 from attempted zero. The checked-in report is an explicitly pending scaffold until complete data are available.
- A 16-cell partial final audit is running in process 1226990, output `/tmp/p2-final-partial-16.json` and log `/tmp/p2-final-partial-16.log`. Command: `PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 /tmp/shot-pattern-validation-venv-20261009/bin/python docs/research/shot_pattern_analysis/phase2/validation/independent_final_audit.py docs/research/shot_pattern_analysis/phase2/final_v1 --allow-partial --max-cells 16 --output /tmp/p2-final-partial-16.json`. The first direct-script invocation without `PYTHONPATH` failed at import before auditing; the corrected command is active. Partial success does not establish full final-data acceptance.
- `validation/independent_final_audit.py` (commit `ed60f608f6`) was verified on the full control pilot, corrupted-input rejection, and the first two final cells (60,000 rows). `FINAL_DATA_REVIEW.md` correctly withholds full acceptance; partial checks cannot produce a final pass.
- Deterministic archive builder now has 12 passing tests after three independently reproduced integrity/portability findings were fixed. AGY added them to #11988. Existing pilot bundles retain all original bytes and hashes.
- Launcher/manifest checks passed 54 tests; architecture-map contract passes. The expanded CLI/GUI guide distinguishes matched shaft-twist noise from the historical face-only trial.
- Preliminary canonical pre-PR finished failed/incomplete; receipt and exact log are `validation/pre_pr_preliminary.json` and `pre_pr_preliminary_log.txt` (commit `3fa688e2bc`). Broad pytest/collection were interrupted to preserve production CPU. Inventory is regenerated and verifies (`b12d550f96`); independent audit-script typing fixes have normalized-AST equivalence evidence (`0eb2e36d15`). Two frozen `density.py` float-array typing errors and formatting of frozen academic/matrix plot files remain. **Do not edit accepted sources while runs are active.** After completion apply minimal typing/formatting corrections with explicit equivalence/source-transition review, preserving historical execution hashes. API 401 failures have a source-equivalent baseline review and follow-up #11995.
- `build_final_figures.py` and its tests are committed (`f86caf2389`, Astra fixes `0574ba1582`); 13 tests pass; all temporary preview roots are outside the immutable pilots and ignored. Final figure production should cover selected shaft-only/combined2 detailed views and all matrix strata, plus direct SD1/SD2 and curve1/curve2 two-panel density comparisons on common axes/scales. Coordinate exact paths with the report builder.

## Repository Baselines

Design-manual structural governance passes, but release remains `blocked-inventory-required`. Architecture-map contract passes. The preliminary full tracked title audit reports 2,521 violations across 1,867 documents (earlier baseline: 2,505 across 1,861); final changed-document review remains required; development-log validation has existing owner/state/WIP/size failures. Preserve unrelated handoff content; root `AGENT_HANDOFF.md` already exceeds the fleet size target. Prior baseline issues #11934/#11935/#11940 must be checked at final pre-PR validation. Do not bypass hooks, change tolerances, force-push, or edit unrelated fleet work to make the study appear accepted.
