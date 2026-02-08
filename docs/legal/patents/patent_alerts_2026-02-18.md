# Patent Alert: 2026-02-18

**Author:** Jules (Patent Reviewer Agent)
**Status:** ALERT

## Summary
Routine patent risk assessment update. No new critical infringements found. Previous critical risks (Kinematic Sequence, DTW) remain open.

## New Findings

### 1. Gear Effect Constants (Confirmed Medium Risk)
- **File:** `shared/python/impact_model.py`
- **Issue:** Use of `gear_effect_h_scale=100.0`, `gear_effect_v_scale=50.0`.
- **Context:** These constants appear to be empirical heuristics. There is a risk they are derived from TrackMan proprietary data/models.
- **Action:** [Issue_011](../../assessments/issues/Issue_011_Physics_Gear_Effect_Heuristic.md) tracks the need to replace this heuristic with a first-principles physics model.

### 2. Swing Comparator Formula Update
- **File:** `shared/python/swing_comparison.py`
- **Update:** Formula changed to `100 * exp(-norm_dist)`.
- **Analysis:** This mitigates risks associated with specific algebraic forms claimed in patents, but the underlying use of Dynamic Time Warping (DTW) for motion scoring remains a High Risk area (Zepp/Blast).
- **Action:** Continue monitoring [ISSUE_002](../../assessments/issues/ISSUE_002_DTW_SCORING.md).

### 3. Anthropometric Data Clearance
- **File:** `shared/python/data_fitting.py`
- **Finding:** Hardcoded coefficients found, but verified as cited from public academic sources (Dempster, Winter, de Leva).
- **Status:** CLEARED (Low Risk).

## Remediations Confirmed

- **Swing DNA:** No instances found in codebase. Remediated.

## Pending Critical Actions

- **Kinematic Sequence:** Still using TPI-style proximal-to-distal check. Needs refactoring.
- **Ball Flight Coefficients:** Uncited aerodynamic coefficients in `ball_flight_physics.py`. Needs citation or externalization.
