# Patent Alert: 2026-01-26

**Severity:** CRITICAL
**Reviewer:** Jules

## 🚨 Critical Findings

### 1. Kinematic Sequence Infringement Risk
*   **Component**: `shared/python/kinematic_sequence.py`
*   **Issue**: The `KinematicSequenceAnalyzer` explicitly scores the "Pelvis -> Torso -> Arm -> Club" firing order. This methodology is heavily patented by **Titleist Performance Institute (TPI)** and **K-Motion**.
*   **Recommendation**: Immediately refactor to "Segment Timing Analysis" and remove the normative "Efficiency" score that enforces this specific order.

### 2. Motion Comparison Scoring (DTW)
*   **Component**: `shared/python/swing_comparison.py`
*   **Issue**: The `compute_kinematic_similarity` method uses Dynamic Time Warping (DTW) to generate a 0-100 score. This "time-warped comparison scoring" is a core claim of **Zepp** and **Blast Motion** patents.
*   **Recommendation**: Switch to a non-temporal scoring method (e.g., spatial keyframe overlap) or purely statistical correlation.

## ✅ Remediated Items

*   **Swing DNA**: Code inspection confirms the UI now uses "Swing Profile" instead of "Swing DNA". The previous violation in `window.py` has been resolved.

## Action Required

*   [ ] Legal review of `kinematic_sequence.py`.
*   [ ] Legal review of `swing_comparison.py`.
