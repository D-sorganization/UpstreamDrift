# Explicit Source-Pair Review

## Reviewed Commit and Scope

Independent review of `5df7335879dc910dbdc32d2598f3e92791c01108` covers `matched_lean_control.py`, `matched_supplement_report.py`, and their focused tests. This is a read-only saved-data comparison API review; it does not accept unfinished final matrices or authorize supplemental simulations. The initial independent run of the two focused modules passed 21 tests (`/tmp/astra-pair-review-tests.log`). No flight or Monte Carlo execution was involved.

## Verified Statistical Contract

The API centers delivered path and speed on each pattern's saved nominal, and lean/AoA on the club preset. Shaft-SD comparisons normalize shaft rotation by its configured SD while retaining the remaining primitive errors. It checks pairing across patterns and conditions, including unsupported rows. Outcome calculations use the sorted intersection supported in all six pattern-by-condition populations, disclose common and total denominators, and preserve individual unsupported counts.

The interaction is computed rowwise as `(curved − Straight)_second − (curved − Straight)_first`, followed by the shared paired bootstrap. It uses the public source-backed score-vector helper reviewed separately in `COURSE_SCORING_REFACTOR_REVIEW.md`. Positive values mean increased relative curved-pattern benefit. This is a conditional comparison of complete rematched delivery conditions, not a causal curvature coefficient or an unconditional estimate when support differs.

Actual lean comparisons accept ordered 0→5°, 0→10°, and 5→10° pairs at a common physical target. SD comparisons require 1→2°. Nominal target residuals and same seed/noise contracts are checked. Generic first/second-condition labels replace misleading lean labels for SD and curve comparisons.

## Findings Requiring Follow-Up

1. **Approved Driver Pair Still Rejected:** `_config_differences` emits `match_speed_factors[1]` for the approved upper-bound change, while the allowed set and exception check use `match_speed_factors`. The predicate unit test does not exercise this failure. Require a complete driver 0→5° explicit-pair regression and a narrow bound-vector classification fix.
2. **Final Evidence Recheck Missing:** Source artifacts are rechecked after calculation, but the accepted transition/evidence and active scoring sources are not. Require a scorer-source guard (including the vector helper) and final transition/evidence revalidation, with mutation-during-calculation tests.
3. **Protected Transition Coverage:** The implementation checks the exact derived changed-path set, a closed allowlist, and all non-source execution metadata. Existing transition tests mainly demonstrate missing receipt or stale receipt hash. Add full-API tests with coherent new source manifests and correctly rehashed receipts that falsely approve protected contact/noise/native/runtime changes.

The first two findings block API qualification pending actual fixes and independent re-review. The third is a required focused acceptance check. No accepted transition receipt has been manufactured. The future real receipt must bind the exact snapshots and reviewed evidence, including the scoring parity and explicit legacy-helper reachability qualification.

## Follow-Up Qualification

Independently re-reviewed `38a38090787612dc53ec30b5ebb7367b4a75d6e9` and its additive provenance follow-up `eac423b7337a7a8dec6c536d22df34a71f07538f`. The three findings above are resolved at the code/test boundary:

- Only matcher-bound vectors use whole-field differences, so the approved driver comparison reaches its narrow protocol exception; other scientific vector differences retain indexed rejection. A complete public-API driver regression passes.
- Scorer/provider/helper hashes and transition receipt/evidence are rechecked after calculation. The captured analysis source map is also exported in pair results and comparison CSVs. The common-index membership operation uses a set.
- Full-API tests with internally consistent, rehashed manifests and receipts reject protected contact-source and native-runtime changes. Separate tests detect evidence mutation and scorer-source drift during analysis. The transition schema and changed-source records reject extra fields.

The stable independent run passed **30 tests, exit 0**, including real public course-scoring integration, six-population exclusions, SD/curve interactions, ordered lean comparisons, and source-integrity guards. The first rerun had 29 passes and one source-drift rejection because an API edit occurred concurrently; the guard correctly stopped publication. Both logs are retained as `EXPLICIT_PAIR_REVIEW_TESTS.txt` and `EXPLICIT_PAIR_SOURCE_DRIFT_TESTS.txt.gz` (lossless compression; original content hash retained). The stable run used a coordinated source hold; source hashes and exact scope are recorded in `explicit_pair_independent_review.json`.

**Disposition:** the reviewed pair API is qualified for the stated saved-data comparison contract. This does not accept the separate pipeline under development, authorize new simulations, accept any actual cross-snapshot transition receipt, or qualify incomplete final datasets. Those independent gates remain open.
