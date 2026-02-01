---
title: "Critical Patent Risk: Kinematic Sequence & DTW Scoring"
labels: ["legal", "patent-risk", "critical", "technical-debt"]
assignees: ["legal-team", "lead-developer"]
---

### Description
The 2026-01-26 Patent Risk Audit identified two HIGH RISK implementations that directly conflict with known competitor patents.

### Findings

1.  **Kinematic Sequence Order Check (TPI Risk)**
    *   **File**: `src/shared/python/kinematic_sequence.py`
    *   **Issue**: The `KinematicSequenceAnalyzer` class has a hardcoded `expected_order` of `["Pelvis", "Torso", "Arm", "Club"]` and scores the user's swing based on adherence to this specific order. This is the core claim of Titleist Performance Institute (TPI) patents regarding "Kinematic Sequence Efficiency".
    *   **Action Required**:
        *   Refactor `KinematicSequenceAnalyzer` to remove the default hardcoded order.
        *   Remove the "Efficiency Score" that is derived solely from this ordering.
        *   Rename "Kinematic Sequence" in the UI to "Segment Timing".

2.  **DTW Similarity Scoring (Zepp/Blast Risk)**
    *   **File**: `src/shared/python/swing_comparison.py`
    *   **Issue**: The `SwingComparator` calculates a single scalar "Similarity Score" (0-100) using Dynamic Time Warping (DTW) distance with an exponential decay formula. This method of comparing an amateur swing to a pro model is heavily patented by Zepp/Blast Motion.
    *   **Action Required**:
        *   Deprecate `compute_kinematic_similarity` using DTW for scoring.
        *   Implement a non-time-warped comparison method (e.g., discrete Keyframe Analysis or simple spatial deviation).

### Reference
See `docs/legal/patents/patent_alerts_2026-01-26_update.md` for exact code locations.
