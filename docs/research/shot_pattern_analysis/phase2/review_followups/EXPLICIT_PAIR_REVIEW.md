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
