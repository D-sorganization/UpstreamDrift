# Patent Risk Alerts: 2026-02-24

## Summary
A targeted review of the `injury` and `physics` packages was conducted to identify potential IP risks in new implementations.

## Key Findings

### 1. Injury Risk Scoring (TPI / Various) - **MEDIUM RISK (NEW)**
- **Status**: Identified potential risk in `src/shared/python/injury/injury_risk.py`.
- **Detail**: The `InjuryRiskScorer` class implements a scoring system based on specific biomechanical thresholds, including "X-Factor Stretch". The term "X-Factor" and specific methods for quantifying injury risk based on kinematic separation are subjects of various patents (including TPI's).
- **Action**:
    - Verify all thresholds (e.g., > 55 degrees) against public academic sources.
    - Rename "X-Factor" to "Torso-Pelvis Separation" or similar generic term.
    - Avoid using a "proprietary" scoring formula; stick to summing up evidence-based risk factors.
- **Tracker**: [`ISSUE_016`](../../assessments/issues/ISSUE_016_INJURY_RISK_SCORING.md)

### 2. Shaft Flexibility Modeling - **LOW RISK (CLEARED)**
- **Status**: Reviewed `src/shared/python/physics/flexible_shaft.py`.
- **Detail**: The module implements standard Euler-Bernoulli beam theory using Finite Element Method. This is standard engineering practice and does not appear to infringe on specific "shaft fitting" patents, provided no claims are made about "EI Profile Matching" for specific commercial shafts.
- **Action**: Ensure documentation emphasizes the use of standard physics models.

### 3. Kinematic Sequence Efficiency (TPI) - **HIGH RISK (PERSISTENT)**
- **Status**: Still active in `src/shared/python/analysis/pca_analysis.py`.
- **Detail**: The code still calculates `efficiency_score` based on sequence order.
- **Action**: **CRITICAL**. This remains the highest priority legal risk. The code must be refactored to remove this calculation.
- **Tracker**: [`ISSUE_001`](../../assessments/issues/ISSUE_001_KINEMATIC_SEQUENCE.md)

### 4. DTW Motion Scoring (Zepp/Blast) - **HIGH RISK (PERSISTENT)**
- **Status**: Still active in `src/shared/python/validation_pkg/comparative_analysis.py`.
- **Detail**: DTW usage for swing comparison is a known patent minefield.
- **Action**: Deprecate `compute_dtw_distance`.
- **Tracker**: [`ISSUE_002`](../../assessments/issues/ISSUE_002_DTW_SCORING.md)

## Recommendations
1.  **Create Issue 016** for Injury Risk Scoring.
2.  **Prioritize Issue 001** for immediate engineering remediation.
