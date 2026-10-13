## Problem

The unaccepted phase-two LaTeX article expansion contains source-fidelity errors and unsupported review/compilation claims. These are publication defects; root reverified all 60 frozen producer sources, ten inputs, the native binary, and seven numerical outputs unchanged. Simulations must not be rerun or silently altered to repair prose.

## Confirmed Findings

- Course context says a 380 m Driver hole and 10 m approach green radius; saved scoring inputs use a 400 m Driver hole, 15 m fairway half-width, and 15 m green radius.
- Original nominal face bounds are misreported as plus/minus 2 degrees; derive the actual configured values from the bound summaries.
- A claimed maximum nominal residual of 1.6e-13 m is contradicted by a saved Driver Draw residual of about 1.9997e-13 m. Report verified residuals or the actual 0.05 m acceptance criterion.
- Primary C-key identities and original-control E-key identities are conflated. Each table must retain exact ordered source protocols, populations and condition definitions.
- Unsupported causal wording attributes excluded shots to high kinetic energy, treats zero-spanning intervals as neutral/equivalent, or overstates universal conclusions.
- The handoff claims Root/Astra reviews that did not occur and successful compilation. Root's actual built-in compiler check still fails before parsing with the known uncached-bundle panic tracked separately in #12120.
- Wide tables need readable source layout and actual rendered-page verification when the compiler is available.

## Acceptance Criteria

1. Correct the existing editable `docs/research/shot_pattern_analysis/shot_pattern_analysis.tex` in place, preserving explicitly historical phase-one material.
2. Independently verify all selected 24 conditions / 72 pattern rows, 12 direct paired interactions, and the exact public-selector fixed-target controls against hash-bound saved sources. Keep individual, common-three, and common-six populations distinct.
3. State actual course inputs, original/supplemental bounds, matching tolerance, uncertainty scope and noncausal limitations. Preserve unavailable cases and conditional support.
4. Document actual independent review and built-in compiler results truthfully; no invented approvals, no alternative PDF presented as editor success.
5. Preserve the rejected candidate and correction evidence; add a change fragment and reference this issue in the ready PR. Do not close until merged acceptance evidence exists. Compiler infrastructure issue #12120 remains separate.

## Evidence and Current State

Rejected candidate and root checks: `docs/research/shot_pattern_analysis/phase2/validation/article_candidate_rejection_v1/` at commit `647698428b` on `codex/shot-pattern-matched-delivery`. Astra's independent correction is in progress. Supplemental v6 numerical parity is independently verified, while full graphics and final publication acceptance remain pending.
