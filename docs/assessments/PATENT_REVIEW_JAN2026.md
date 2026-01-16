# Patent Risk Review & Infringement Documentation

**Date:** January 16, 2026
**Scope:** `shared/python/swing_comparison.py`, `shared/python/impact_model.py`, `shared/python/dashboard/window.py`, `shared/python/statistical_analysis.py`, `shared/python/kinematic_sequence.py`.

## 1. Executive Summary
This document identifies current patent and trademark risks within the codebase. While significant remediation has occurred (renaming "Swing DNA"), several critical algorithms remain that closely mirror proprietary methodologies from K-Motion, Zepp, and major biomechanics providers.

**High Risk Areas:**
1.  **Biofeedback Scoring (DTW):** The specific formula `100 / (1 + distance)` and use of Dynamic Time Warping for "Kinematic Similarity" is highly risky.
2.  **Kinematic Sequence Efficiency:** The logic for scoring "efficiency" based on peak velocity timing order is a core claim in several expired and active patents.
3.  **Gear Effect Model:** The use of specific empirical constants (`100`, `50`) in the linear approximation model.

**Remediated Areas:**
*   **"Swing DNA" Trademark:** The term has been largely replaced with "Swing Profile" in the UI and backend logic. Residual risk is low but requires vigilance.

---

## 2. Detailed Findings

### A. Biofeedback & Scoring Algorithms (High Risk)
**File:** `shared/python/swing_comparison.py`
**Method:** `compute_kinematic_similarity`

*   **Infringement Risk:** The implementation uses Dynamic Time Warping (DTW) to align student vs. pro swings and computes a score using the specific formula:
    ```python
    score = 100.0 / (1.0 + norm_dist)
    ```
    This approach mimics scoring methodologies protected by patents held by **K-Motion** and **Zepp (Blast Motion)**. These patents cover the method of "comparing a user motion to a model motion using time-warping" and "generating a compliance score".

*   **Evidence in Code:**
    *   Lines 118-125: Normalization and DTW call.
    *   Line 131: `score = 100.0 / (1.0 + norm_dist)`

*   **Recommendation:**
    *   **Change the Scoring Formula:** Use a non-linear mapping or a completely different metric (e.g., Cross-Correlation coefficient).
    *   **Renaming:** Ensure "Kinematic Similarity" doesn't overlap with specific "Kinematic Sequence" trademarked products.

### B. Gear Effect Empirical Model (Medium-High Risk)
**File:** `shared/python/impact_model.py`
**Method:** `compute_gear_effect_spin`

*   **Infringement Risk:** The function calculates gear effect spin using linear approximations with specific "empirical" constants:
    ```python
    horizontal_spin = -gear_factor * h_offset * speed * 100
    vertical_spin = gear_factor * v_offset * speed * 50
    ```
    If these constants (`100`, `50`) are derived from a specific research paper or competitor's white paper (e.g., TrackMan, Foresight), implementing them directly may constitute infringement of a "method for simulating golf ball flight" patent.

*   **Evidence in Code:**
    *   Lines 420-421: Explicit use of `100` and `50` constants.

*   **Recommendation:**
    *   **Obfuscate or Cite:** If these are from a public paper, add the citation to prove it's Prior Art/Public Domain.
    *   **Parametrize:** Move `100` and `50` into the `ImpactParameters` dataclass so they are configuration values, not hardcoded "secret sauce".

### C. Kinematic Sequence Efficiency (Medium Risk)
**File:** `shared/python/kinematic_sequence.py`
**Method:** `analyze`

*   **Infringement Risk:** The concept of "Kinematic Sequence" (Proximal-to-Distal sequencing) is widely researched but specific methods of *scoring* it (e.g., checking the order of peak velocities: Pelvis -> Thorax -> Arm -> Club) are patented by **TPI (Titleist Performance Institute)** and others in the biofeedback space.
*   **Evidence in Code:**
    *   Lines 105-125: Logic to strictly check order of peaks and calculate an `efficiency_score`.
*   **Recommendation:**
    *   Ensure the term "Efficiency Score" is generic.
    *   Avoid using specific color-coded "Green/Red" output logic if it mimics TPI's "Kinematic Sequence Report" exactly.

### D. "Swing DNA" Trademark (Low Risk - Remediated)
**File:** `shared/python/dashboard/window.py`
**File:** `shared/python/statistical_analysis.py`

*   **Status:** **REMEDIATED**.
*   **Observation:** The term "Swing DNA" has been replaced with "Swing Profile".
    *   `window.py`: Uses `Swing Profile (Radar)`.
    *   `statistical_analysis.py`: Class is `SwingProfileMetrics`, method is `compute_swing_profile`.
*   **Residual Risk:** `docs/assessments/archive/PATENT_RISK_ASSESSMENT.md` still contains text *describing* the risk, which might be flagged by text searches, but the *code* is clean.

### E. Chaos & Stability Metrics (Low-Medium Risk)
**File:** `shared/python/statistical_analysis.py`

*   **Infringement Risk:** Algorithms for `compute_lyapunov_exponent` and `compute_fractal_dimension` are generally scientific public domain (Rosenstein algorithm, Higuchi method). However, *applying* them to "Sport Movement Consistency" is a claim in some modern sports tech patents.
*   **Recommendation:** Keep the implementation generic and scientific. Avoid marketing terms like "Chaos Score" or "Talent Index".

---

## 3. Action Plan

1.  **Immediate:** Refactor `score = 100.0 / (1.0 + norm_dist)` in `swing_comparison.py`.
2.  **Immediate:** Move Gear Effect constants to `ImpactParameters`.
3.  **Review:** Verify `KinematicSequenceResult` logic against TPI patent claims.
