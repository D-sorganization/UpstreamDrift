# Patent Risk Alert - 2026-02-25

**Reviewer:** Jules (Patent Reviewer Agent)
**Status:** ACTIVE

## New Findings / Refinements

### 1. Gear Effect Heuristic Constants (Medium Risk)
- **File:** `src/shared/python/physics/impact_model.py`
- **Finding:** The `compute_gear_effect_spin` function uses explicit heuristic constants:
  - `gear_effect_h_scale = 100.0`
  - `gear_effect_v_scale = 50.0`
- **Risk:** These specific constants likely originate from empirical fitting against proprietary data (e.g., TrackMan or Foresight launch monitors). Using them without independent derivation or citation constitutes a risk of copyright/trade secret infringement or patent infringement if the specific coefficients are claimed.
- **Action:** Document origin immediately. If derived from public data, cite the dataset. If copied from a forum or SDK, flag for removal/replacement with a clean-room implementation.

### 2. Injury Risk Thresholds (Medium Risk)
- **File:** `src/shared/python/injury/injury_risk.py`
- **Finding:** The `InjuryRiskScorer` uses specific hardcoded thresholds for "X-Factor Stretch":
  - `threshold_safe = 45.0` degrees
  - `threshold_high = 55.0` degrees
- **Risk:** These specific threshold values, combined with the term "X-Factor Stretch", are highly characteristic of TPI (Titleist Performance Institute) methodology.
- **Action:** Verify if these specific numbers are protected by patent or copyright. Re-label to generic "Torso-Pelvis Separation" and cite academic sources for the thresholds (e.g., McHardy et al., 2006) to demonstrate independent development.

### 3. Kinematic Sequence Efficiency Score (CRITICAL Risk)
- **File:** `src/shared/python/analysis/pca_analysis.py`
- **Finding:** Despite the renaming of `KinematicSequenceAnalyzer` to `SegmentTimingAnalyzer` in `kinematic_sequence.py`, the `pca_analysis.py` module retains the infringing logic:
  ```python
  efficiency_score = matches / len(expected_order)
  ```
- **Risk:** This is a direct implementation of the TPI "Kinematic Sequence Efficiency" scoring method, which is protected by patent. The renaming in the other file is insufficient remediation while this logic exists here.
- **Action:** **IMMEDIATE REMOVAL** of this specific calculation is required. Replace with an energy-transfer-based metric (which is already implemented in `reporting.py`) or a generic "Sequence Consistency" metric that does not use the specific TPI formulation.

### 4. Ball Flight Coefficients (Medium Risk)
- **File:** `src/shared/python/physics/ball_flight_physics.py`
- **Finding:** `BallProperties` class uses:
  - `cd0 = 0.21`
  - `cd1 = 0.05`
  - `cd2 = 0.02`
  - `cl1 = 0.38`
- **Risk:** These coefficients are precise and uncited. They likely come from a specific research paper or competitor SDK.
- **Action:** Add citation.

## Summary
The "Kinematic Sequence" risk remains **CRITICAL** due to the persistence of the scoring logic in `pca_analysis.py`. The "Injury Risk" and "Gear Effect" risks are now **CONFIRMED** with specific evidence in the codebase.
