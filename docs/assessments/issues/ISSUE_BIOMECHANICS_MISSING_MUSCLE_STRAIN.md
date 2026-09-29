# Critical Gap: Injury Risk Scorer Lacks Muscle Strain Modeling

**Status:** Open
**Priority:** High
**Labels:** biomechanics-gap, high
**Date Identified:** 2026-02-16
**Author:** JULES (Gap Analysis)

## Description

The `InjuryRiskScorer` in `src/shared/python/injury/injury_risk.py` calculates risk based on peak spinal loads, joint stresses, and cumulative workload. However, it currently lacks any integration with Hill-type muscle-tendon models to estimate muscle strain or tendon loading directly.

Muscle strains (especially eccentric loading during deceleration phases) are a major cause of injury in golf. The current implementation relies on generic joint stress proxies rather than physiological tissue strain.

Although a `HillMuscleModel` exists in `src/shared/python/hill_muscle.py`, it is not utilized by the injury risk assessment module.

## Impact

- **Risk Blindness:** The system cannot detect risks associated with rapid eccentric muscle lengthening (e.g., "pulled muscle" scenarios).
- **Specificity:** Recommendations are generic ("reduce stress") rather than targeted ("limit eccentric velocity of lead hamstring").
- **Accuracy:** Joint stress alone does not capture the soft tissue risk if muscle activation/length dynamics are ignored.

## Affected Files

- `src/shared/python/injury/injury_risk.py`
- `src/shared/python/hill_muscle.py` (unused in risk scoring)

## Recommended Fix

1.  Update `InjuryRiskScorer.score` to accept `muscle_states` (outputs from a muscle-driven simulation).
2.  Implement a `_score_muscle_risks` method that checks for high eccentric velocities combined with high activation (strain risk).
3.  Integrate `HillMuscleModel` usage into the biomechanics pipeline to generate the necessary muscle state data.
