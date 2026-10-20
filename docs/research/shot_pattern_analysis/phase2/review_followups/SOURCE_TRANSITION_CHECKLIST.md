# Supplemental Source-Transition Qualification Checklist

## Pending Status

`source_transition_preparation_v1.json` is a candidate inventory, **not acceptance**. At capture, the candidate differed from frozen `final_acceptance.json` on exactly the 12 agreed paths. Every other source hash and all non-source execution metadata were equal; the actual native binary hash matched. The 50 parent-repository frozen source hashes were independently checked against Git `d5f59256ef`, with no mismatches. The vendored Tools scorer remains hash-identical between snapshots and is governed by its upstream pin.

Plot sources remain under active revision. The candidate inventory must be regenerated after the final commit; a prior preparation record must never be renamed into acceptance. The preparer always emits `pending_independent_qualification_not_acceptance` and uses a separate schema rejected by the comparison validator.

## Per-Change Qualification

| Source                                              | Change and Evidence                                                                                                                                          | Remaining Gate                                                                        |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `matched_runner.py`                                 | Adds explicit search bounds and delegates legacy defaults unchanged; reviewed native default parity and PW match tests in `BOUNDS_IMPLEMENTATION_REVIEW.md`. | Bind final source hash and qualified 32-shot pilot/refinement.                        |
| `matched_matrix.py`                                 | Selected-cell protocol plans and exact supplemental final receipt gate; independently tested with the supplement/GUI suite.                                  | Bind final source and complete pilot plans/audits.                                    |
| `matched_supplement.py`                             | New explicit 18+4+4 catalog; original matrices retained.                                                                                                     | Exact committed inventory, targets, seeds, profiles and 4 non-execution aliases.      |
| `matched_export.py`                                 | Adds supplemental module to source inventory and replaces overstrong infeasibility wording.                                                                  | Verify final diff remains limited to these changes.                                   |
| `nominal_matching.py`                               | Exception/documentation wording only; optimizer equations unchanged.                                                                                         | Final hash and diff check.                                                            |
| `density.py`                                        | Explicit float64 construction plus formatting; 84 exact comparisons in `PLOT_SOURCE_EQUIVALENCE.json`.                                                       | Recheck evidence matches final source hash.                                           |
| `academic_plots.py`                                 | Presentation changes are still being implemented. Earlier formatting-only proof no longer suffices for new edits.                                            | New tests, representative rendered reviews, no endpoint/density/statistic changes.    |
| `matched_report_plots.py`                           | Earlier formatting/type proof recorded; final presentation scope must be inspected.                                                                          | Exact final diff, tests and figure evidence.                                          |
| `provenance.py`                                     | Adds scorer helper and legacy utility coverage.                                                                                                              | Record coverage additions without implying old snapshot contained those keys.         |
| `scenario_scoring.py` and `course_score_vectors.py` | DRY extraction; exact historical parity on 6 saved and 30 randomized reports plus 3 errors, independently recorded in `course_scoring_refactor_review.json`. | Final hashes must match evidence or rerun characterization.                           |
| Legacy `impact_model/utils.py`                      | Separately repaired RH toe-spin sign (#12006), unreachable from matched contact; independent nine-test direction check and parent 58-test patch gate.        | Bind committed repair evidence, including public matched-path monkeypatch regression. |

The utility had no key in the original snapshot. Its `old_sha256` in the transition map is therefore null. The actual original Git-source SHA-256 is `06a79b5414370f502d32f36f99ba04eabb5ff2f6dd6587f5f284ea4beba9eca9` (verified from original base `10c7d61aa686e456b7962f1d9aa9ecfda4e8405f`). This records a coverage extension; the legacy helper's output genuinely changed, while the qualified scope excludes that legacy pathway.

## Receipt and Execution Gates

Use the settled `matched-delivery-source-transition/1` contract only after review: `status=independently_accepted`, `qualified_scope=matched-delivery-comparisons`, exact `old_execution`/`new_execution`, exact derived `changed_sources[path]={old_sha256,new_sha256,classification}`, and `evidence_files` mapping contained relative paths to SHA-256. Classification is `approved_nonprojection` only for the 11 named shot-pattern paths and `unreachable_legacy` only for the utility. All other source paths and every non-source execution field must be unchanged. Callers supply the receipt hash separately. Classification strings in the pending preparer are proposals, not approvals.

Before issuing the real source-transition receipt: freeze source; rerun focused software gates; finish pair-API integration/drift tests; inspect rendered figures; and verify all evidence hashes and source projection. After that source qualification, independently qualify the three 32-shot pilots and deterministic rematched-nominal refinement. Final simulation acceptance is a separate artifact with the full 26-cell protocol catalog and all pilot/refinement gates. This checklist neither authorizes simulations nor accepts incomplete original final data.

## Reproduction

From the candidate worktree, save `matched_source_snapshot()` from `src.tools.shot_pattern_analysis.matched_export` with the qualified runtime. Then run the preparer from the main study worktree:

```sh
python3 docs/research/shot_pattern_analysis/phase2/review_followups/prepare_source_transition.py --acceptance docs/research/shot_pattern_analysis/phase2/final_acceptance.json --snapshot NEW_CANDIDATE_SNAPSHOT.json --candidate-root /home/dieterolson/.codex/worktrees/shot-pattern-bounds/UpstreamDrift --output NEW_PENDING_INVENTORY.json
```

The preparer checks actual candidate source/native bytes against the snapshot and refuses to overwrite an output. Ruff and scoped Mypy pass. No flights were executed for this inventory review.
