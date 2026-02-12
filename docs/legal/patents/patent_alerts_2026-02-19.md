# Patent Risk Alerts: 2026-02-19

## Summary
A comprehensive review of the codebase was conducted to verify patent risk remediation status and identify new risks.

## Key Findings

### 1. Swing DNA (Mizuno) - **REMEDIATED**
- **Status**: The term "Swing DNA" has been successfully removed from the UI and logic.
- **Evidence**: `src/shared/python/dashboard/window.py` now uses "Swing Profile (Radar)".
- **Action**: No further action required. Monitor for regression.

### 2. Kinematic Sequence Efficiency (TPI) - **HIGH RISK (ACTIVE)**
- **Status**: Active infringement risk identified in `src/shared/python/analysis/pca_analysis.py`.
- **Detail**: The `efficiency_score` logic explicitly calculates a score based on adherence to a specific segment order (`matches / len(expected_order)`). This maps directly to TPI's core patent claims.
- **Action**: **CRITICAL**. This logic must be removed or replaced with an energy-based metric (like the one in `reporting.py`).
- **Tracker**: [`ISSUE_001`](../../assessments/issues/ISSUE_001_KINEMATIC_SEQUENCE.md)

### 3. DTW Motion Scoring (Zepp/Blast) - **HIGH RISK (ACTIVE)**
- **Status**: Logic located in `src/shared/python/validation_pkg/comparative_analysis.py`.
- **Detail**: The function `compute_dtw_distance` implements Dynamic Time Warping to compare swing signals. While the specific Zepp formula (`100 * exp...`) was not found in this file, the use of DTW for athletic motion scoring remains the core risk.
- **Action**: Isolate and potentially deprecate this method in favor of keyframe analysis.
- **Tracker**: [`ISSUE_002`](../../assessments/issues/ISSUE_002_DTW_SCORING.md)

### 4. Magic Numbers (Medium Risk)
- **Gear Effect**: `src/shared/python/physics/impact_model.py` uses heuristic constants.
- **Ball Flight**: `src/shared/python/physics/ball_flight_physics.py` uses uncited coefficients.
- **Action**: Add citations or externalize to config.

## Recommendations
Immediate engineering effort should be directed towards refactoring `pca_analysis.py` to remove the order-based efficiency score.
