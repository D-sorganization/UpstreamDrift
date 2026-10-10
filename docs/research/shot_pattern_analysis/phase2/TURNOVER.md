# Matched Delivery Study Turnover

## Current State

The goal is active and incomplete. MAIN is `/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift`, branch `codex/shot-pattern-matched-delivery`. Phase-one PR [#11941](https://github.com/D-sorganization/UpstreamDrift/pull/11941) remains open; phase two is not pushed. Work only in owned worktrees. Start with [Design](DESIGN.md), [Astra Review](ASTRA_REVIEW.md), [Final Narrative Acceptance](review_followups/FINAL_NARRATIVE_ACCEPTANCE.md), [Supplemental Acceptance](SUPPLEMENT_FINAL_RUN_ACCEPTANCE.md), and [Reproduction](REPRODUCING.md). History lives in git and immutable validation receipts.

## Live Jobs and Mandatory Source Freeze

Both saved-data jobs started from `9ea4c14bed`. Revalidate actual handles before acting; never restart a live job.

| Job | Session | Wrapper / Child PID | Log | Terminal Receipt |
| --- | --- | --- | --- | --- |
| Original Graphics v3 — Complete | 78157 (exit 0) | Both producers terminal | `/tmp/p2-original-final-figures-v3.log` | `/tmp/p2-v3-*-exit.json` |
| Supplemental Report v2 | 83056 | 2509004 / 2509013 | `/tmp/p2-supplement-report-v2.log` | `/tmp/p2-supplement-report-v2-exit.json` |

**Freeze every MAIN Python file under `src/tools/shot_pattern_analysis` and `scripts/research/shot_pattern_analysis` until both jobs finish.** This includes adding files: the original wrapper binds the complete Python inventory. Docs/evidence may change. Do not cherry-pick the isolated repairs while either job is live.

Primary completed with exit 0 at 06:32:48 UTC and control at 07:05:12 UTC. The final wrapper exited 0. Control audit/visual review is in `validation/control_final_graphics_independent_review_v3/`; its 72 source manifests and five inventory CSVs exactly match preserved v2. The primary audit validates all 689 output hashes, 324 cell manifests and five inventories, and all five inventory CSVs exactly match the preserved v2 output. All 288 PNGs are indexed in 18 contact sheets; Astra reviewed all contact sheets and found a remaining main-axis decimal-label collision in the 7-iron lean-10 / curve-1 / SD-1 / shaft-only density. Root confirmed it; the completed visual review covers 288 contacts and nine full-size figures, while a 60-PDF screen identifies the same suspect. Whole-set acceptance is withheld pending regeneration with the qualified general spacing repair `8142c78776`. See issue #11988 comment 6094738523. Evidence: `validation/original_final_graphics_independent_review_v3/`. The original wrapper renders primary then control into canonical `final_figures_v1` and `control_final_figures_v1`; its final receipt is `validation/original_final_figures_rendering_revision_v3.json`. The supplemental pipeline has emitted all 384 paired comparisons in `supplement_report_v1` and is now rendering figures. Its receipt is not yet final. There is no reuse/resume CLI. These jobs use saved shots; they do not rerun Monte Carlo. Wrapper copies and start identities are in `validation/final_publication_regeneration_start.json`, `validation/original_v3_renderer.py.txt`, and `validation/supplement_v2_wrapper.py.txt`.

## Completed Numerical Data

All simulation workers, full saved-data audits and fresh archive restorations are finished. Do not restart them.

| Dataset | Planned / Completed / Unavailable Cells | Attempted Shot Rows | Archive |
| --- | --- | ---: | --- |
| Original Primary | 324 / 270 / 54 | 8,100,000 | 2,097 files, 15 ZIP parts |
| Original Fixed-Target Control | 72 / 52 / 20 | 1,560,000 | 427 files, 3 ZIP parts |
| Bounds Supplement | 26 / 26 / 0 | 780,000 | 191 files, 2 ZIP parts |

The supplement retains 779,712 supported and 288 unsupported rows. Four PW zero-lean control aliases reuse verified executions and add no shots. Original unavailable records remain unchanged. Audits, exact-byte restoration and restored full-report equality pass. See `validation/independent_primary_final_audit.json`, `validation/independent_control_final_audit.json`, `validation/PRIMARY_FINAL_ARCHIVE_RESTORATION.md`, `validation/CONTROL_FINAL_ARCHIVE_RESTORATION.md`, and `validation/SUPPLEMENT_FINAL_VALIDATION.md`.

Astra independently accepted supplemental completion at `a63278a0fc`; evidence is in `validation/supplement_final_independent_completion_review_v1/`. This acceptance checks receipts/manifests/archive inventory and does not claim a second repetition of the full CSV audit or extraction.

## Immutable Supplemental Provenance

Accepted execution source is `2e0a74f8ca`, merged as `b21a57f2d0`; preserve its 54-source/native/runtime identity. The historical snapshot was verified at `256745676e`. Do not rewrite it for later reporting changes.

- `source_transition_supplement.json`: SHA-256 `cf622694964b0dc6c2ca0c24b92f9fa8ee5365721219069283aca1f6804a8981`.
- `supplement_final_acceptance.json`: SHA-256 `4afeeec8fc1e3a73b18ef87b25b2082078b826646ab8a3e33c441b99b2958d95`.
- Protocols: `pw-primary-face-bounds/1`, `pw-control-face-bounds/1`, `driver-control-speed-bounds/1`.
- Pilots/refinement passed: 26 unique pilot cells, 2,496 rows; 333 deliveries / 1,998 refinement evaluations, 318 consistently supported and 15 consistently unsupported. Maximum endpoint change 0.00900125 m remains below the unchanged 0.05 m gate. See `validation/SUPPLEMENT_PILOT_VALIDATION.md`.

## Graphics and Comparison Repairs

Earlier rejected outputs are preserved, not publication assets: 90 partial v1 assets under `presentation_preview_rejected_final_v1`, and both complete v2 trees under `presentation_preview_rejected_final_v2`. Relocation and original manifests are bound in `validation/original_v2_figure_relocation.json`. FINAL_LAYOUT is inactive; its older source must not be used for new rendering.

All required graphics repairs are integrated into MAIN: direct-comparison controls/units (`cc09eec2cf`, `82e017f00d`), marginal boundary clearance (`a400f5de95`), matrix units/correlation/scoring context/contrast (`ca80a32c87`), covariance-header height (`56ea967f53`), and circular containment radius (`8820b13fd0`). Component reviews include actual saved-data figures, unchanged histogram values/limits, 23 boundary tests, 15 final matrix tests and 48 integration tests before the last narrow radius change. See `validation/new_main_marginal_boundary_review_v1/`, `validation/matrix_covariance_header_v1/`, `validation/matrix_radius_independent_review_v1/`, and `validation/publication_repair_integration_v1/`. **Whole regenerated-set acceptance is still pending.**

The first supplemental report failed a reviewed Driver curve-pair bounds transition. Repair `20d462ad2b` permits only the exact Driver 5° original-control curve-1 → supplemented curve-2 face/speed-bound change; arbitrary changes remain rejected. All 384 source/config/nominal pairs pass preflight, 30 tests pass, and the actual 10,000-shot pair has 9,940 common supported rows. See `validation/driver_curve_bounds_reporting_review_v1/` and retained `validation/supplement_report_first_failure_v1/`.

## Isolated Precision and Narrative Work

BOUNDS is `/home/dieterolson/.codex/worktrees/shot-pattern-bounds/UpstreamDrift`, branch `codex/shot-pattern-report-precision`. MAIN bounds prerequisite `20d462ad2b` was copied into BOUNDS as `4b53effe73`; do not cherry-pick that duplicate back. Integrate only owned repair commits, never replace the whole tree. Its sparse checkout excludes bulky tracked data archives.

- `41ad7e3893`: AGY's shared formatter, improved and accepted by Astra. Seven new red regressions were fixed; 94 focused tests pass. All 696 saved paired SG intervals and 1,044 absolute means preserve signs/nonzero widths: maximum width display error 0.9184%, value error 0.099713%. Portable fixtures and formatter source binding are included. Evidence: `validation/report_precision_independent_review_v1/`.
- `3f023b7ad9`: supplemental default table delegates to that formatter and binds it as the 14th reporting source. Two intended red tests → 46 passing tests; Ruff/Mypy pass. Only `_estimate_ci` and `_report_source_hashes` changed at function level. Evidence: `validation/supplement_precision_candidate_v1/`.
- `f34e1a6cd9`: AGY narrative completion, repaired and accepted by Astra. The selected Driver and PW controls both use curve 2, shaft SD 2° and combined-2 noise, with exact ordered protocol identities and common-six counts 9,940 / 9,999 respectively. Four red regressions → 35 passing tests; redundant parsing removed and duplicate comparisons rejected. See `validation/report_narrative_independent_review_v1/`. All three isolated commits await MAIN freeze release.
- `8142c78776`: Astra's adjacent main-axis tick-label repair reuses the existing four-point spacing helper. Two red cases → 20 passing layout tests, plus nine invariant tests; Ruff/Mypy pass. Six actual saved-data figures retain exact production limits, with minimum neighboring-label clearance 8.2605 pt. Root confirmed the repaired narrow 7-iron plot. See `validation/main_tick_neighbor_candidate_v1/`. Integrate only after both MAIN producers finish.

Astra's provisional real-table review found a remaining calculation gap: all 56 across-lean records omit relative Draw/Fade SG interactions, so the selected-control narrative would show unavailable effects despite valid populations. The prior synthetic narrative tests did not catch the missing producer fields. AGY Gemini 3.8 Flash High is repairing the existing same-six scored-vector pathway in BOUNDS (`/tmp/p2-lean-interaction-agy.log`, exec session 3797), with tests first and independent actual-control review required. The 164 curve-strength and 164 shaft-SD records already contain interaction fields. See issue #11988 comment 6094919525.

The current live pipeline still has 13 reporting sources and fixed-decimal Markdown. After both live jobs finish, preserve that output and receipt unchanged, integrate the qualified repairs, and run a fresh saved-data pipeline into the canonical output location. Compare its numeric JSON/CSV with the preserved prior output: existing estimates/populations must remain unchanged; the new across-lean relative interactions and their provenance are the only intended added numerical results. No marginal-interval subtraction is permitted. Do not silently rehash old outputs, bypass source validation or add a caching framework. No shot simulation rerun is needed.

## GUI and Scientific Interpretation

GUI scoring #12067 is integrated at `1ac7342872`: saved conditional absolute SG, paired intervals, correct denominators and lifecycle clearing. Independent and root post-merge 51-test checks pass; see `validation/gui_scoring_independent_review_v1/` and `validation/gui_scoring_main_integration_v1/`. The pending formatter extraction must preserve that behavior.

Assumed static lofts are 10°/34°/46°; lean is 0°/5°/10°. Physical face/path/speed matching aligns raw zero-error landings without aiming transforms. Noisy shots still differ in carry. Shaft-twist SD is not delivered face-angle SD. Rematching changes multiple delivery variables, so neither curvature nor lean has an isolated causal interpretation. Keep individual-supported geometry, common-three scoring and common-six contrasts distinct.

Driver tee and iron/wedge approach scores assume course dimensions, carry-as-stop and an approximate baseline. No measured-player calibration, universal best pattern or per-round SG extrapolation is justified. Contact uses the energetic Coulomb finite-head model; legacy additive gear-spin repair #12006 is separate. Empirical qualification #11984 remains deferred.

Design and both editable references now disclose deterministic head sensitivity: 268 valid / 32 unsupported evaluations, fixed delivery, chosen ±20% parameters and a contact-law comparator. Astra verified all ten table maxima and limitations. This is distinct from Monte Carlo uncertainty; see `validation/independent_head_sensitivity_summary.json` and `validation/supplement_precision_candidate_v1/sensitivity_document_review.json`.

## LaTeX Panel and Final Publication

The existing user-open `../shot_pattern_analysis.tex` remains open and now distinguishes current phase two from historical phase one. Built-in Tectonic fails **before parsing** at `src/config.rs:154:18` while fetching its uncached bundle. Full host GET succeeds but does not establish compiler-environment access. The path-only tool exposes no cache/bundle configuration. Latest edited-source checks failed identically for both references; title checks pass. See `validation/latex_editor_head_sensitivity_check.json` and `validation/latex_editor_compiler_followup.json`. Do not claim compilation or substitute a separately produced PDF for this panel failure.

`REPORT.md` remains a scaffold. Complete source-bound final Markdown/statistical inventories, editable LaTeX findings, final PDF and full page/graphic/claims review after qualified inputs are available. The PDF operation marker was already executed for this publication; do not repeat it. A separate ReportLab PDF does not establish LaTeX compilation.

## Next Steps and Gates

1. Revalidate live jobs and preserve the MAIN source freeze until both terminate. Review their complete manifests, inventories, statistics and graphics.
2. Integrate all four qualified BOUNDS commits after freeze. Run appropriate MAIN integration tests and the fresh supplemental report pipeline; preserve historical receipts and verify numeric equality.
3. Build and independently review complete scientific/social deliverables, including the selected fixed-target controls and model-sensitivity discussion. Keep reproducibility instructions current.
4. Run final `python -m scripts.pre_pr` from MAIN with Repository_Management on `PYTHONPATH`; update handoff/change fragment and ready PR, arm only through the merge-queue guard, then end the session. No post-PR polling/check-ins. Check owned PR #11941's actual remote/queue state before deciding how to deliver the expanded branch; no force push or another session's PR changes.

Prior gates identified unrelated API 401 baseline #11995, document-title/development-log debt #11935, and recorded baselines #11934/#11940. Design-manual structure passes; its release inventory remains blocked. Root `AGENT_HANDOFF.md` inherits excessive length and peer work; do not delete peer sections to mask that baseline. No hook, tolerance, identity-policy or branch-protection bypass. Issues require merged acceptance before closure; #11984 stays deferred.

## Environment

Use `/tmp/shot-pattern-validation-venv-20261009/bin/python` (Python 3.12.13, NumPy 2.2.6, Numba 0.68.0, llvmlite 0.50.0). Tools pin: `3678409fc51024150ab28970b72e3b468935f345`. Set `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy OPENBLAS_NUM_THREADS=1`. Pytest uses `-o addopts=''`; scoped Mypy uses `--python-version 3.12 --follow-imports=silent`. Do not launch GUI processes.

Sol/Luna quotas are exhausted until October 13; do not retry. AGY Gemini 3.8 Flash High and Astra remain available. The user authorized personal GitHub identity; never expose credentials. Obsolete task-owned Mypy caches totaling 493 MB were removed after validating their cache tags; no evidence or simulation data was removed. Last disk check: approximately 2.5 GB free; recheck before large publication commits.
