---
title: "[CRITICAL] PCA Analysis Efficiency Score Patent Risk"
labels: ["legal", "patent-risk", "critical"]
assignees: ["legal-team"]
---

## Description
The `efficiency_score` calculation in `src/shared/python/analysis/pca_analysis.py` (`matches / len(expected_order)`) directly implements a sequence-adherence metric that overlaps with patented methodologies by **Zepp Labs** and **Blast Motion** for scoring kinematic sequences.

## Action Required
- Redesign the efficiency metric to avoid direct sequence-matching formulas.
- Consult with legal counsel to ensure compliance.
