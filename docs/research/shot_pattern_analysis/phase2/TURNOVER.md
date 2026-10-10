# Matched Delivery Study Turnover

## Current State

The phase-two goal is active and incomplete. Worktree: `/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift`; branch `codex/shot-pattern-matched-delivery`. Phase-one PR [#11941](https://github.com/D-sorganization/UpstreamDrift/pull/11941) remains open/held. Phase two has not been pushed. Do not edit the primary checkout or vendor submodule.

Read [Design](DESIGN.md), [Astra Review](ASTRA_REVIEW.md), [Final Narrative Acceptance](review_followups/FINAL_NARRATIVE_ACCEPTANCE.md), and [Supplemental Acceptance](SUPPLEMENT_FINAL_RUN_ACCEPTANCE.md). Commit history holds earlier checkpoints; this file records current state.

## Completed Original Data

All four original simulation shards finished with exit0. Full independent saved-data audits pass all324 primary and72 control records:270/52 completed and54/20 explicitly unavailable, respectively;8.1million/1.56million saved shot rows. Receipts: `validation/independent_primary_final_audit.json` and `validation/independent_control_final_audit.json`. Original unavailable records must remain intact.

Lossless archives and independent fresh restorations are committed: primary2,097files/15ZIPparts, control427files/3parts. Exact bytes, inventory and full restored reports agree. See `validation/PRIMARY_FINAL_ARCHIVE_RESTORATION.md` and `validation/CONTROL_FINAL_ARCHIVE_RESTORATION.md`. Original pilot archives are also preserved. Packaging acceptance is separate from scientific and publication acceptance.

## Accepted Supplemental Execution

Bounds source `2e0a74f8ca` was independently accepted in immutable `source_transition_supplement.json` (SHA256 `cf622694964b0dc6c2ca0c24b92f9fa8ee5365721219069283aca1f6804a8981`). It was merged as `b21a57f2d0`; reviewed reporting composition `a40cf86dc0` was merged as `3f5945edff`. Exact54-source/native/runtime equality passes;97 composition tests pass. Source-transition evidence remains unchanged.

All three32-shot pilots completed and full independent audits pass26unique cells/2,496rows. Four PW0 control views are verified reuse aliases, not new executions. Refinement completed333deliveries/1,998evaluations:318supported at all settings,15consistently unsupported, no unresolved comparison. Maximum primary-to-fine endpoint change0.009001249870284642m and refined nominal residual0.0011381967888155486m are below the unchanged0.05m gate. Pilot archives passed fresh restoration of188files. See `validation/SUPPLEMENT_PILOT_VALIDATION.md`.

Astra independently accepted the exact26-cell final experiment in `e55638e6db`. `supplement_final_acceptance.json` SHA256 is `4afeeec8fc1e3a73b18ef87b25b2082078b826646ab8a3e33c441b99b2958d95`;67durable evidence files are bound. Root and Astra both passed generic, three production-protocol and three independent-protocol validators. Six pilot PNGs/all18asset hashes passed diagnostic review. Preserve this receipt, its evidence and the54execution sources unchanged during runs.

## Active Jobs

Started2026-10-10 at01:19:38UTC: two detached supplemental final workers, PIDs1934292/1934293. Wrapper: `/tmp/p2-supplement-final-worker.py`; worker inventory `/tmp/p2-supplement-final-workers.json`; logs `/tmp/p2-supplement-final-worker{0,1}.log`. Each sequentially runs `pw_primary_face`, `pw_control_face`, `driver_control_speed`, shard0/1 of2,10,000shots per pattern, no figures. Outputs: `supplement_final_v1/{protocol}`. Per-command exit receipts: `/tmp/p2-supplement-final-{protocol}-shard{index}-exit.json`; aggregate receipts `/tmp/p2-supplement-final-worker{index}-exit.json`. Revalidate process/log/receipt state before acting; never blindly restart a live or completed run.

Original final-figure rendering runs separately in the **reporting-only** worktree `/home/dieterolson/.codex/worktrees/shot-pattern-final-layout/UpstreamDrift`, branch `codex/shot-pattern-final-layout`, source `edfae39f00`. Wrapper PID1937409, `/tmp/p2-render-original-final-figures.py`; log `/tmp/p2-original-final-figures.log`. It sequentially creates MAIN `final_figures_v1` and `control_final_figures_v1` from saved original data only. Per-set exit receipts `/tmp/p2-{final_figures_v1,control_final_figures_v1}-exit.json`; final provenance `validation/original_final_figures_rendering_revision.json`. Freeze this reporting worktree while it renders. No Monte Carlo belongs there.

## Reporting Revision

Isolated layout `1e4d70d0dd` fixes phone readability, main/marginal ticks and footer clearance. Regression evidence is red-to-green;19tests pass. Astra inspected nine1080px previews and verified27artifact hashes across Driver/7-Iron/PW. Root also inspected the7-Iron draw. This qualifies the reviewed component, not the complete final publication.

Layout `edfae39f00` additionally composes MAIN's validated endpoint loader in `build_final_figures.py` and applies Ruff-only formatting with identical normalized AST;20tests pass. Preserve both changes when later merging. Its scientific snapshot differs from accepted execution only in `academic_plots.py`; record this explicitly as a rendering revision. Do **not** merge that renderer into MAIN before supplemental numerical runs and audits finish.

Final report builders preserve exact source/protocol/ordered-condition identities, all three physical-pattern social density assets, input receipts before loading and figure bytes after source rechecks. Existing interpretation gates distinguish individual-supported endpoints, common-three scoring and common-six paired comparisons. Existing `REPORT.md` is still a pending scaffold.

## Next Steps

1. Verify all six supplemental command exit receipts and complete manifests. Run the full independent auditor separately on each protocol with `--supplement-stage final --supplement-acceptance .../supplement_final_acceptance.json`. Generate and validate four final alias references with existing `load_supplement_sources`; pass the alias envelope to the auditor. Preserve all unsupported shots and failures.
2. Package supplemental final raw data into bounded archives and verify fresh restoration. Do not overwrite historical pilots or original unavailable records.
3. After numerical audits, merge the qualified reporting-only branch, preserving MAIN's current documents and all accepted execution evidence. Check original final figure manifests/source hashes against the integrated reporting code. Independently inspect final assets; component acceptance does not automatically qualify all layouts.
4. Run the supplemental report pipeline against all five original/supplemental roots, inventory and immutable source-transition receipt. Generate the complete final report, statistics, scoring sensitivities, common-six effects, figures and linked editable references. Distinguish original protocols from adapted supplemental bounds and driver speed sensitivity.
5. Complete independent data/claims/visual review, repository gates, current handoff/change fragment, ready PR and merge-queue guard. Follow PR lifecycle: no CI polling/check-in automation after PR open. No issue closure without merged acceptance evidence.

## Scientific Boundaries

Assumed static lofts10°/34°/46°; lean0°/5°/10°. Raw zero-error landings are matched by physical face/path/speed, without post-flight aiming rotation. Shaft-twist SD is not face-angle SD. Finite-head energetic Coulomb contact is primary; Newton maximum-dissipation is a comparator. Coupled-contact impact replaces additive gear-spin in this study; separate legacy gear-sign repair#12006 is included and reviewed.

Noisy shots do not all carry the same distance. Nominal matching removes initial landing mismatch but does not isolate a causal curvature or lean effect when physical delivery conditions change together. Report actual carry distributions and common populations. Driver tee and iron/wedge approach scoring is conditional on assumed course geometry, carry-as-stop and approximate baseline; no per-round extrapolation or calibrated-player claim. Preserve shared-seed dependence, pointwise intervals and unsupported mass. Empirical qualification#11984 remains open in `docs/development/planning/DV-11984.md`.

## Environment and Commands

Use `/tmp/shot-pattern-validation-venv-20261009/bin/python`:Python3.12.13,NumPy2.2.6,Numba0.68.0,llvmlite0.50.0. Tools submodule is exactly`3678409fc51024150ab28970b72e3b468935f345`. Set `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy OPENBLAS_NUM_THREADS=1`. Never launch GUI processes. Tests use `-o addopts=''`; scoped Mypy uses `--python-version 3.12 --follow-imports=silent`.

Sol/Luna quotas are exhausted until October13; do not repeatedly retry them. AGY CLI Gemini3.8 Flash High and Astra remain available. User authorized personal GitHub identity for issues/PRs. Latest AGY issue11988 update is mirrored in `issues/P2-007.json`.

## LaTeX and Repository Gates

The user-open `../shot_pattern_analysis.tex` remains unchanged/open. Built-in Tectonic fails before parsing at `src/config.rs:154` while fetching its uncached bundle. Full host GET succeeds but does not prove compiler-environment access. Three attempts have reproduced the failure; do not blindly retry without new evidence or claim compilation. See `validation/latex_editor_compiler_followup.json`. No replacement PDF was compiled for that request. The separate ReportLab design PDF is separately reviewed, not LaTeX acceptance.

Final pre-PR has not run at publication SHA. Prior pre-PR found an unrelated API401 baseline#11995, existing document-title/development-log debt#11935 and other recorded baselines#11934/#11940. Design-manual structure passes but release remains inventory-blocked. Preserve unrelated handoffs; root `AGENT_HANDOFF.md` already exceeds fleet length policy. Run required final gates and report genuine blockers without tolerance, hook, identity-policy or branch-protection bypasses. Never force-push or push directly to main.
