# Interpretation and Pipeline Independent Review

## Scope and Status

Reviewed interpretation commit `8e73636ba9` and the data/pair contracts of pipeline checkpoint `559c345e8c`. This is not final dataset acceptance, supplemental execution authorization, or approval of the evolving figure layouts. Original numerical sources and immutable source-transition evidence were not modified.

## Interpretation Findings Requiring Correction

1. **Absolute SG population mislabeled.** `matched_runner._paired_results` intersects the three supported populations before calling `score_course_endpoints`. Consequently `course_scoring.patterns[name].n` is the common-three scored count. The initial interpretation module calls it individual support. Obtain individual endpoint support from `summary[name].supported_n`; label the absolute SG mean with its common-three population and require each scorer count to equal `paired_n`. Existing fabricated tests with scorer n 9,000 and paired n 8,500 do not represent the producer contract.
2. **Condition identities lost.** Club/lean alone does not identify curve strength, primitive shaft SD or noise profile. Cell/geometry tables and direct interaction tables must preserve exact condition identities and ordered A/B source references.
3. **Asset presentation relabeling.** The initial selector validates the requested `presentation` string but does not verify that the selected asset belongs to that presentation. It can label an academic asset social. Require an explicit, verified inventory category or presentation field.
4. **Saved-cell read stability.** A final manifest-consistency check alone does not detect coherent manifest/CSV replacement after loading. Bind the initial manifest and CSV bytes and recheck them before returning delivered means.
5. **Paired-result contracts.** Require status/count consistency, unavailable estimates to remain unavailable, complete ordered source identities and source hashes. A display adapter must not promote arbitrary dictionaries into qualified six-population results. Actual qualification remains the pair producer's responsibility.
6. **Insufficient support.** Undefined statistics/covariance for fewer than two supported rows must render unavailable rather than fail through `float(None)` or imply a measured zero. Retain actual support counts.

Sol acknowledged the scorer-population defect and owns test-first interpretation fixes. These findings remain open until the corrected commit is independently reviewed. The module's explicit noncausal distance and geometric-versus-empirical caveats are appropriate.

## Pair and Pipeline Review

The renewed explicit-source pair path retains six-population common-index conditioning, centered primitive input errors, normalized shaft-twist draws for SD comparisons, ordered condition contrasts, and direct paired SG difference-of-differences. Source-transition validation uses the closed changed-source inventory, full remaining execution equality and hash-bound evidence; active scorer/source and input bytes are rechecked. The focused four-module test run exited zero after the shared-validator refactor.

One additional pipeline issue remains: `_recheck_source_inputs` skips every non-completed reference. Materialized unavailable-cell manifests and summaries supply published reasons/status and must also be bound and rechecked. Only the CSV requirement should depend on completed status. Luna received this narrow regression request.

Original protocol validation now checks the canonical full configuration inventory and binds common targets to the accepted control pilot evidence. The published original-source inventory retains unavailable rows. Final receipt writing follows input/evidence/postprocessing checks. Final rendered presentation is separately undergoing correction following the root's actual 10,000-shot component review.

## Remaining Acceptance

Re-review interpretation fixes and the unavailable-artifact guard, then qualify the stable postprocessing sources. Full original/supplemental dataset audits, supplemental pilot/refinement and execution acceptance, and final rendered scientific narrative review remain required. No result is inferred from incomplete production matrices.
