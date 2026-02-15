# ISSUE 016: Injury Risk Scoring (X-Factor Stretch)

**Status:** ACTIVE
**Priority:** MEDIUM
**Component:** `src/shared/python/injury/injury_risk.py`
**Created:** 2026-02-24

## Description

The module `src/shared/python/injury/injury_risk.py` implements an `InjuryRiskScorer` class that calculates a comprehensive injury risk score based on various biomechanical factors. Specifically, it uses the term "X-Factor Stretch" and applies specific thresholds (e.g., > 55 degrees is High Risk) to calculate a weighted score.

## Patent Risk Context

- **"X-Factor"**: This term is widely associated with Jim McLean and popularized by TPI (Titleist Performance Institute). While the term itself may be trademarked or common law, the *method* of using kinematic separation (pelvis-thorax angle) to assess efficiency or injury risk is subject to various patents in the golf instruction space.
- **Risk Scoring Algorithms**: Patents exist for "systems and methods for assessing athletic motion" which often cover the specific algorithm of combining multiple biomechanical inputs into a single "Risk Score" or "Efficiency Score".

## Technical Details

- **File**: `src/shared/python/injury/injury_risk.py`
- **Class**: `InjuryRiskScorer`
- **Method**: `_score_spinal_risks`
- **Specific Concern**:
    ```python
    # X-factor risk
    self.risk_factors.append(
        RiskFactor(
            name="x_factor_stretch",
            value=stretch,
            threshold_safe=45.0,
            threshold_high=55.0,
            weight=0.8,
            # ...
        )
    )
    ```

## Mitigation Plan

1.  **Literature Verification**: Verify the specific thresholds (45.0, 55.0) against public academic literature (e.g., *McHardy et al., 2006*, *Gluck et al., 2008*). If these numbers appear in public research papers, cite them explicitly in the code to establish "Prior Art" defense.
2.  **Terminology Update**: Rename "X-Factor" to a generic biomechanical term like "Torso-Pelvis Separation" or "Axial Rotation Difference".
3.  **Algorithm Transparency**: Ensure the scoring algorithm is transparently a sum of weighted factors based on published medical guidelines, rather than a "black box" proprietary formula that might mimic a competitor's patent.
4.  **Legal Review**: Consult with legal counsel regarding the specific use of "X-Factor" in a commercial product.

## Action Items

- [ ] **Engineering**: Rename "X-Factor" variable names and strings to "Torso-Pelvis Separation".
- [ ] **Research**: Add citations for the 45/55 degree thresholds in `injury_risk.py`.
- [ ] **Documentation**: Update `PATENT_REVIEW.md` with citation details once found.
