---
title: Patent Infringement Risk in Kinematic Sequence Efficiency Score
labels: physics-error, critical, legal, patent-risk
assignees: physics-team
---

## Description
The `efficiency_score` calculated in `src/shared/python/analysis/pca_analysis.py` (`matches / len(expected_order)`) and its nomenclature overlap significantly with patent claims from Zepp Labs and Blast Motion regarding "kinematic sequence scoring."

## Expected Behavior
The metric should be renamed to a non-infringing term (e.g., "Sequence Adherence") and the implementation must be legally reviewed to avoid patent infringement while maintaining scientific validity.

## Impact
High risk of patent infringement lawsuits.

## Recommended Fix
Rename `efficiency_score` to "Sequence Adherence" (or another approved term) across the codebase and consult legal counsel.
