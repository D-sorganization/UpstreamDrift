---
title: Patent Infringement Risk in Kinematic Sequence Scoring
labels: physics-gap, high, legal-risk
assignees: legal-team, physics-team
---

## Description
The `efficiency_score` implementation in `src/shared/python/analysis/pca_analysis.py` (Line 160) calculates a metric based on the number of sequential peaks matching an expected order:

```python
matches = sum(1 for e, a in zip(expected_order, actual_order) if e == a)
efficiency_score = matches / len(expected_order)
```

This specific methodology of scoring a kinematic sequence (pelvis -> thorax -> lead arm -> club) overlaps with claims in patents held by **Zepp Labs** and **Blast Motion**, which cover systems and methods for capturing and scoring athletic swings based on sequence timing.

## Impact
- **Legal Liability**: Potential lawsuit for patent infringement if this metric is surfaced to users as a core feature.
- **Scientific Validity**: The "match count" metric is overly simplistic and does not reflect the continuous nature of energy transfer efficiency.

## Recommended Fix
1. **Consult Legal Counsel**: Review specific claims of Zepp/Blast patents regarding "kinematic sequence scoring."
2. **Rename and Redesign**: Change the metric name to "Sequence Adherence" or similar.
3. **Use Open Standards**: Implement academic metrics (e.g., standard deviation of peak times relative to TPI norms) rather than proprietary-style scores.
