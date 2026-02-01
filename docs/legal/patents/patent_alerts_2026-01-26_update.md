# Patent Risk Alerts - Update

**Date:** 2026-01-26
**Reviewer:** Jules (Patent Reviewer Agent)
**Status:** CRITICAL ATTENTION REQUIRED

## Summary of Findings

This alert documents specific code segments identified as high-risk for patent infringement during the comprehensive audit on 2026-01-26.

### 1. Kinematic Sequence Order (TPI Risk)
*   **Risk Level:** **HIGH**
*   **File:** `src/shared/python/kinematic_sequence.py`
*   **Location:** Line 56-60
*   **Code:**
    ```python
    self.expected_order = expected_order or [
        "Pelvis",
        "Torso",
        "Arm",
        "Club",
    ]
    ```
*   **Issue:** The code explicitly defines the "Proximal-to-Distal" sequence order (Pelvis->Torso->Arm->Club). Comparing a user's swing against this specific order to derive a "validity" or "efficiency" score is the core claim of Titleist Performance Institute (TPI) patents and common law methodology.
*   **Recommendation:** Remove the default hardcoded order. Allow the user to define order, or simply report peak times without scoring "correctness" based on this specific template.

### 2. DTW Swing Scoring (Zepp/Blast Risk)
*   **Risk Level:** **HIGH**
*   **File:** `src/shared/python/swing_comparison.py`
*   **Location:** Line 135-140
*   **Code:**
    ```python
    dist, path = signal_processing.compute_dtw_path(s1_norm, s2_norm)
    # ...
    score = SIMILARITY_SCORE_CONSTANT * np.exp(-norm_dist)
    ```
*   **Issue:** Generating a single scalar "Score" (0-100) by comparing a user's motion to a professional's motion using Dynamic Time Warping (DTW) is heavily patented by Zepp Labs (now Blast Motion/others). The specific formula using exponential decay of the normalized distance is a common implementation of "similarity scoring".
*   **Recommendation:** Deprecate the single "Similarity Score". Provide "deviation heatmaps" or "key position comparison" (P-system) instead of a continuous time-warped score.

### 3. Gear Effect Magic Numbers (TrackMan Risk)
*   **Risk Level:** **MEDIUM**
*   **File:** `src/shared/python/impact_model.py`
*   **Location:** Line 125-126
*   **Code:**
    ```python
    gear_effect_h_scale: float = 100.0
    gear_effect_v_scale: float = 50.0
    ```
*   **Issue:** These constants appear to be empirical fitting parameters, likely derived to match data from a specific launch monitor (TrackMan). Using reverse-engineered coefficients without license may constitute IP risk or Terms of Service violation if derived from scraping.
*   **Recommendation:** Verify the source of these numbers. If they are from public literature (e.g., Cochran & Stobbs), document the citation. If empirical, document the clean-room data collection method used to derive them.
