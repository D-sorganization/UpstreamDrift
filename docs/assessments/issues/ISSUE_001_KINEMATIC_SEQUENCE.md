---
title: "Legal Risk: Kinematic Sequence Implementation infringes TPI patents"
labels: ["legal", "patent-risk", "critical", "high-priority"]
assignee: "legal-team"
status: "open"
---

# Issue Description

The current implementation of `KinematicSequenceAnalyzer` in `src/shared/python/biomechanics/kinematic_sequence.py` and the `efficiency_score` calculation in `src/shared/python/analysis/pca_analysis.py` pose a high patent infringement risk.

## Technical Detail

The code explicitly validates and scores a specific segment firing order:

```python
expected_order = ["Pelvis", "Torso", "Arm", "Club"]
```

In `src/shared/python/analysis/pca_analysis.py`, the `efficiency_score` is calculated based on adherence to this order:

```python
efficiency_score = matches / len(expected_order)
```

## Legal Context

This methodology (quantifying kinematic sequence efficiency based on proximal-to-distal sequencing) is covered by patents held by **Titleist Performance Institute (TPI)** and **K-Motion Interactive**.

## Required Actions

1.  **Refactor Class**: Rename `KinematicSequenceAnalyzer` to `SegmentTimingAnalyzer`.
2.  **Remove Scoring**: Remove the `sequence_consistency` score in `kinematic_sequence.py` and `efficiency_score` in `pca_analysis.py` that imply a "correct" order.
3.  **UI Update**: Display timing data as neutral graphs/Gantt charts without normative "Good/Bad" coloring based on the TPI model.
