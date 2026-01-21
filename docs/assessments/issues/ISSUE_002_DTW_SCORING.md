---
title: "Legal Risk: DTW Motion Scoring infringes Zepp/Blast patents"
labels: ["legal", "patent-risk", "critical", "high-priority"]
assignee: "legal-team"
status: "open"
---

# Issue Description

The motion comparison logic in `shared/python/swing_comparison.py` uses Dynamic Time Warping (DTW) to generate a user score.

## Technical Detail
The function `compute_kinematic_similarity` calculates a normalized DTW distance and converts it to a 0-100 score:
```python
score = SIMILARITY_SCORE_CONSTANT * np.exp(-norm_dist)
```

## Legal Context
Patents from **Zepp Labs** (now Blast Motion?) and **K-Motion** cover methods of "evaluating athletic motion" by time-warping a user's motion to a professional reference and deriving a score from the alignment cost.

## Required Actions
1.  **Isolate Logic**: Move DTW scoring to a separate module/plugin that can be disabled.
2.  **Implement Alternative**: Develop a scoring metric based on **Discrete Keyframes** (P1-P10) or **Spatial Hausdorff Distance** which does not rely on temporal warping.
3.  **Legal Review**: Submit `swing_comparison.py` for formal IP review.
