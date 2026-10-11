# Shot Pattern Study: Agent Resume Handoff

## Current Resumed State

The user resumed on October 10 after usage reset. The goal is active. Sol owns
`validation/supplement_v6_final_graphics_review_v2`, Luna owns vector review v2,
and Astra reviews the narrative. Root restarted frozen report construction with
fresh evidence in `validation/final_report_production_v7`; inspect its terminal
receipt before using the generated report. No simulations are being repeated.
The stop checkpoint below remains historical evidence.

## Prior User-Requested Stop

Goal pursuit is paused at the user's request to conserve credits. No completion,
publication acceptance, remote push, or new PR is claimed. Resume only when asked.
All checked report/review processes have stopped; no monitoring is scheduled.
This document supersedes historical live-process descriptions in Turnover.

## Checkouts and Saved State

All checkouts are under `/home/dieterolson/.codex/worktrees/`:

| Role | Checkout | Branch and Checkpoint |
| --- | --- | --- |
| Study and Data | `shot-pattern-analysis/UpstreamDrift` | `codex/shot-pattern-matched-delivery`; use latest local commit after `2584d3fa13` |
| Delivery Integration | `shot-pattern-main-integration/UpstreamDrift` | `codex/shot-pattern-main-integration`; integration checkpoint `89fe6d3efa`, plus stop documentation |
| Frozen Producer | `shot-pattern-presentation/UpstreamDrift` | `codex/shot-pattern-density-gutter`; `4023111d2aaf8b20449a36f8691a295f4a94ceff` |
| Qualified Pending Guard | `shot-pattern-bounds/UpstreamDrift` | `codex/shot-pattern-protocol-preconditions`; `c7a2f41fc8` |

Study commits after `dd2c235a0a` still need integration into the delivery branch.
Do not touch armed phase-one PR #11941. Phase two is local and unpublished.

## Completed Evidence

All 10.44 million attempted simulation rows are complete. Do not rerun simulations.
V6 exporter completed with exit 0 at 2026-10-10T21:46:30.187679+00:00;
see `validation/supplement_rendering_revision_v6/exit.json`.
Its pipeline receipt SHA256 is
`035c65b249937a639d0ddfd5bd5df3aef8b46aaae351b98a4c6ae9e29e0b064a`.
Sol reported a successful strict public-loader exit with 384 ordered comparisons,
26 unique cells and 6,456 assets (2,408 PNG, 2,024 PDF, 2,024 SVG).
Loader session 85362 exited 0, but no durable loader receipt was written; its
returned object was truncated. This reported check is not a substitute for a
source-bound saved receipt on resumption and is not visual acceptance.
Original primary/control graphics and vector checks are accepted within their
recorded scope. Integration has actual root evidence for 282 passing tests and
Astra evidence for 71 launcher-boundary tests. Rejected AGY test claims are not evidence.

## Interrupted Work and Exact Resume Points

The first of 36 graphics batches, `across_lean_000`, generated an inventory and
contact sheets but was stopped before terminal qualification. Preserve its partial
outputs; no visual approval follows. Exact bytes and hashes are committed in
`validation/user_requested_stop_20261010_final/partial_graphics_review.zip` and
`checkpoint.json`. Continue using a fresh review directory after reading
`validation/v6_output_review_harness_v1/FINAL_REVIEW_PLAN.md` and `VECTOR_REVIEW_PLAN.md`.

Final report construction was intentionally stopped with SIGTERM (actual exit -15).
`validation/final_report_production_v6/` contains its exact command, frozen checkout,
start/exit receipts, logs, prior scaffold and explicit user-stop receipt.
`REPORT.md` is still the prior scaffold. Resume construction under frozen
PRESENTATION 4023111d2a in a fresh evidence directory; do not overwrite this stop record.
The report uses primary `final_v1`, control `control_final_v1`, supplement
`supplement_report_v1`, and their final figure directories, all under this phase2 folder.

Supplement vector review has not started. Empty output directories convey no evidence.
All 4,048 vectors need qualified inspection and the predeclared selection/raster review.
Use `validation/v6_vector_selection_plan_v1/` and qualified vector helpers.
The original runtime uses `dependencies/bin/override/pdftoppm` v26.05.0,
Node v24.19.0 and Sharp v0.35.4 under
`/home/dieterolson/.cache/codex-runtimes/codex-primary-runtime/`.
Validation Python is `/tmp/shot-pattern-validation-venv-20261009/bin/python`.
Do not install or alter the frozen runtime/source bindings.

## Remaining Acceptance and Delivery

1. Finish all 36 review batches, exact coverage reconciliation, vector checks and
   actual visual acceptance. Contacts and successful rendering alone are insufficient.
2. Finish frozen report construction and inspect all assembled PDF pages and claims
   against `review_followups/FINAL_NARRATIVE_ACCEPTANCE.md`.
3. Preserve geometric-only assumptions, conditional scoring, common-support
   populations and limitations. No universal curvature or empirical player benefit
   has been established. Empirical validation #11984 remains deferred.
4. Build and verify supplemental graphics bundles; original primary/control local
   bundles exist in `/tmp/shot-pattern-publication-bundles-v1`. Follow committed
   `review_followups/ARTIFACT_DELIVERY_PLAN.md` for unique release filenames and
   actual remote download/restoration. Nothing has been uploaded this resume.
5. Only after frozen report verification, integrate qualified guard `c7a2f41fc8`
   and remaining study commits into isolated delivery integration.
6. Update change fragments/handoff, run canonical RM `python -m scripts.pre_pr`,
   resolve owned failures, push and open a ready PR; attach and arm it through
   `scripts/automerge_guard.py`. Do not bypass hooks/protection or change peer PRs.

## Preserved Local Artifacts and Compiler Blocker

Large final graphics remain locally in `final_figures_v1`, `control_final_figures_v1`
and `supplement_report_v1`, deliberately outside Git for verified bundle transport.
The partial contact directory also remains locally and has a committed exact-byte
archive. Do not clean these worktrees or `/tmp`, run `git add -A`, or archive the
worktrees before durable artifact delivery. Code and handoff/evidence changes are
committed; final graphics are not remotely backed up.

The user-open `../shot_pattern_analysis.tex` is preserved in place. Its numerical
source repair passed 577 checks, but the built-in compiler fails before parsing
at Tectonic `src/config.rs:154` fetching an uncached TeX bundle. Issue #12120 remains
open. No compiled article acceptance or replacement PDF is claimed. Leave the
editor open. See [Turnover](TURNOVER.md) for detailed scientific and integration
receipts; older live-process statements are historical, not resume instructions.
