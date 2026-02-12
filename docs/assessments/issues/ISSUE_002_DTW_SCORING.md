---
title: "Legal Risk: DTW Motion Scoring infringes Zepp/Blast patents"
labels: ["legal", "patent-risk", "critical", "high-priority"]
assignee: "legal-team"
status: "open"
---

# Issue Description

The motion comparison logic in `src/shared/python/validation_pkg/comparative_analysis.py` (previously `swing_comparison.py`) uses Dynamic Time Warping (DTW) to generate a user score.

## Technical Detail

The function `compute_dtw_distance` calculates a normalized DTW distance, which is often converted to a 0-100 similarity score.

```python
# In src/shared/python/validation_pkg/comparative_analysis.py
signal_processing.compute_dtw_path(data_a, data_b, window=radius)
```

## Legal Context

Patents from **Zepp Labs** (now Blast Motion?) and **K-Motion** cover methods of "evaluating athletic motion" by time-warping a user's motion to a professional reference and deriving a score from the alignment cost.

## Required Actions

1.  **Isolate Logic**: Move DTW scoring to a separate module/plugin that can be disabled.
2.  **Implement Alternative**: Develop a scoring metric based on **Discrete Keyframes** (P1-P10) or **Spatial Hausdorff Distance** which does not rely on temporal warping.
3.  **Legal Review**: Submit `comparative_analysis.py` for formal IP review.
