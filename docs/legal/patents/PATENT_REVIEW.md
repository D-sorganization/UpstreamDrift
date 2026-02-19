# Patent & Legal Risk Assessment

**Last Updated:** 2026-02-24
**Status:** ACTIVE
**Reviewer:** Jules (Patent Reviewer Agent)

## 1. Known Patent Risks

### High Risk Areas

| Area                     | Patent(s) / Assignee                                                                           | Our Implementation                                                                                                                                                                                                                              | Risk Level              | Mitigation                                                                                                                                                                 | Issue Tracker |
| ------------------------ | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| **Kinematic Sequence**   | **TPI (Titleist)**, **K-Motion**<br>Methods for analyzing proximal-to-distal sequencing        | `KinematicSequenceAnalyzer` in `src/shared/python/biomechanics/kinematic_sequence.py`.<br>`efficiency_score` logic in `src/shared/python/analysis/pca_analysis.py` (explicitly scores based on order).                                          | **HIGH (Active)**       | Rename "Kinematic Sequence" to "Segment Timing". **CRITICAL:** Remove `efficiency_score` calculation in `pca_analysis.py` which enforces TPI order. **Legal Review Required.** | [`ISSUE_001`](../../assessments/issues/ISSUE_001_KINEMATIC_SEQUENCE.md) |
| **Motion Scoring (DTW)** | **Zepp**, **Blast Motion**, **K-Motion**<br>Scoring athletic motion via time-warped comparison | `compute_dtw_distance` in `src/shared/python/validation_pkg/comparative_analysis.py`.<br>Calculates distance metric for swing comparison which is the core of Zepp's patent claims.                                                             | **HIGH (Active)**       | Replace DTW scoring with discrete keyframe correlation or strictly spatial comparison (Hausdorff distance). The formula change is partial mitigation but DTW usage remains risk. | [`ISSUE_002`](../../assessments/issues/ISSUE_002_DTW_SCORING.md) |
| **Swing DNA**            | **Mizuno** (Trademark/Method)<br>Club fitting based on specific metric vector                  | `SwingProfileMetrics` in `src/shared/python/analysis/reporting.py`. Visualization as Radar Chart in `src/shared/python/dashboard/window.py`.                                                                                                    | **VERIFIED REMEDIATED** | Renamed to "Swing Profile". Metrics changed to: Speed, Sequence, Stability, Efficiency, Power (distinct from Mizuno's). Audit confirmed no "Swing DNA" strings in UI code. | N/A           |

### Medium Risk Areas

| Area                         | Patent(s) / Assignee                                                | Our Implementation                                                                                                       | Risk Level | Mitigation                                                                                                                                                                              | Issue Tracker |
| ---------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| **Injury Risk Scoring**      | **TPI**, **Various**<br>Methods for assessing injury risk (X-Factor) | `InjuryRiskScorer` in `src/shared/python/injury/injury_risk.py`. Uses specific thresholds for "X-Factor Stretch" and risk weighting. | **MEDIUM** | Verify thresholds against public academic literature (e.g., McHardy et al., 2006). Cite sources explicitly. Avoid proprietary "Risk Score" formulas if possible.                        | [`ISSUE_016`](../../assessments/issues/ISSUE_016_INJURY_RISK_SCORING.md) |
| **Gear Effect Simulation**   | **TrackMan**, **Foresight**<br>Methods for simulating ball flight   | `compute_gear_effect_spin` in `src/shared/python/physics/impact_model.py`. Uses heuristic constants `100.0`, `50.0`.     | **MEDIUM** | Document source of constants. If derived from TrackMan data, ensure "fair use" or clean-room implementation.                                                                            | [`Issue_011`](../../assessments/issues/Issue_011_Physics_Gear_Effect_Heuristic.md) |
| **Ball Flight Coefficients** | **Unknown / Proprietary SDKs**<br>Specific aerodynamic coefficients | `BallProperties` in `src/shared/python/physics/ball_flight_physics.py`. Uses specific values (`cd0=0.21`, `cl1=0.38`).   | **MEDIUM** | Citation missing. If these match a specific competitor's SDK or proprietary research, it could be infringement. **Action:** Cite source (e.g., Smits & Smith) or externalize to config. | [`ISSUE_046`](../../assessments/issues/ISSUE_046_BALL_FLIGHT_COEFFICIENTS.md) |

### Low Risk Areas

| Area                    | Patent(s) / Assignee                                 | Our Implementation                                                                                    | Risk Level | Mitigation                                                                                                               |
| ----------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Chaos Analysis**      | **Various / Niche**<br>Non-linear dynamics in sports | `estimate_lyapunov_exponent`, `compute_fractal_dimension` in `src/shared/python/analysis/nonlinear_dynamics.py` (assumed). | **LOW**    | Mathematical methods are public domain. Ensure application claims don't infringe specific "method of talent ID" patents. |
| **Shaft Flexibility**   | Public Domain / Standard Engineering                 | `FiniteElementShaftModel` in `src/shared/python/physics/flexible_shaft.py`. Uses standard Euler-Bernoulli beam theory. | **LOW**    | Standard engineering practice. Ensure no specific "EI Profile Matching" method claims are made.                          |
| **Physics Engine**      | Public Domain                                        | Standard rigid body dynamics (Newtonian/Lagrangian).                                                  | **LOW**    | Standard physics.                                                                                                        |
| **Ball Flight ODE**     | Public Domain                                        | Implementation of standard aerodynamic equations (drag, lift, Magnus).                                | **LOW**    | Equations are standard; coefficients are the only risk (see Medium).                                                     |
| **Statistical Metrics** | Public Domain                                        | Standard deviations, correlations, PCA.                                                               | **LOW**    | Generic math.                                                                                                            |
| **Photometric Measure** | **Foresight**                                        | N/A                                                                                                   | **N/A**    | No implementation of camera-based ball tracking logic. We use simulation only.                                           |
| **Radar Tracking**      | **TrackMan**                                         | N/A                                                                                                   | **N/A**    | No implementation of radar signal processing. We use simulation only.                                                    |
| **Ground Reaction**     | Public Domain                                        | `src/shared/python/physics/ground_reaction_forces.py` implements standard Newtonian mechanics.        | **LOW**    | Standard physics guideline E5 implementation.                                                                            |

## 2. Code-Specific Analysis

### `src/shared/python/biomechanics/kinematic_sequence.py` & `src/shared/python/analysis/pca_analysis.py`

- **Risk**: `KinematicSequenceAnalyzer.analyze` and `efficiency_score` calculation.
- **Finding**:
    - `pca_analysis.py` explicitly calculates `efficiency_score = matches / len(expected_order)`. This is a DIRECT implementation of TPI's patented efficiency metric based on sequence order.
    - `kinematic_sequence.py` defines `expected_order = ["Pelvis", "Torso", "Arm", "Club"]`.
- **Action**: **CRITICAL**. Remove the `efficiency_score` logic in `pca_analysis.py` that relies on order. The energy-based `efficiency_score` in `reporting.py` is safe.
- **Tracker**: [`ISSUE_001`](../../assessments/issues/ISSUE_001_KINEMATIC_SEQUENCE.md)

### `src/shared/python/validation_pkg/comparative_analysis.py` (formerly `swing_comparison.py`)

- **Risk**: `compute_dtw_distance`
- **Finding**: Uses Dynamic Time Warping (DTW) to compute distance between two swings.
- **Context**: Zepp and K-Motion have patents on "comparing user motion to pro motion using time-warping".
- **Action**: Isolate this logic. Consider making it an optional plugin or replacing with "Key Position Analysis" (P1-P10 positions).
- **Tracker**: [`ISSUE_002`](../../assessments/issues/ISSUE_002_DTW_SCORING.md)

### `src/shared/python/injury/injury_risk.py`

- **Risk**: `InjuryRiskScorer` and "X-Factor Stretch"
- **Finding**: Uses specific thresholds (e.g. X-Factor Stretch > 55 deg = High Risk) and weighted scoring.
- **Context**: "X-Factor" and specific injury risk algorithms are often subject to patents (e.g., TPI).
- **Action**: Verify thresholds against public domain literature. Ensure "Risk Score" is generic.
- **Tracker**: [`ISSUE_016`](../../assessments/issues/ISSUE_016_INJURY_RISK_SCORING.md)

### `src/shared/python/physics/impact_model.py`

- **Risk**: `compute_gear_effect_spin`
- **Finding**: Uses hardcoded scalars `h_scale=100.0`, `v_scale=50.0`.
- **Context**: "Magic numbers" suggest empirical tuning against a reference (likely TrackMan).
- **Action**: Validate these constants against open literature (e.g., Cochran & Stobbs) to establish independent derivation.
- **Tracker**: [`Issue_011`](../../assessments/issues/Issue_011_Physics_Gear_Effect_Heuristic.md)

### `src/shared/python/physics/ball_flight_physics.py`

- **Risk**: `BallProperties` class
- **Finding**: Hardcoded aerodynamic coefficients: `cd0=0.21`, `cd1=0.05`, `cd2=0.02`, `cl1=0.38`, etc.
- **Context**: These specific numbers likely come from a specific study or competitor model.
- **Action**: Add citation immediately.
- **Tracker**: [`ISSUE_046`](../../assessments/issues/ISSUE_046_BALL_FLIGHT_COEFFICIENTS.md)

### `src/shared/python/analysis/reporting.py` (Efficiency Score)

- **Risk**: `efficiency_score` calculation.
- **Finding**: Uses `peak_ke / total_work`.
- **Status**: **SAFE**. This is based on energy efficiency, not kinematic sequence order. Distinct from TPI patent.

## 3. Trademark Concerns

| Term                   | Owner             | Status                  | Alternative      | Notes                                                                                   |
| ---------------------- | ----------------- | ----------------------- | ---------------- | --------------------------------------------------------------------------------------- |
| **Swing DNA**          | Mizuno            | **Verified Remediated** | "Swing Profile"  | Code audit 2026-02-19 confirmed removal from UI and code logic.                         |
| **Kinematic Sequence** | TPI (Common Law?) | **Active**              | "Segment Timing" | Term is widely used in academia, might be genericized, but risky in commercial product. |
| **Smash Factor**       | TrackMan (Orig.)  | **Generic**             | Keep             | Widely accepted as generic golf term now.                                               |
| **TrackMan**           | TrackMan A/S      | **Ref Only**            | N/A              | Used only for data validation references. Acceptable nominative use.                    |
| **X-Factor**           | TPI / Various     | **Medium Risk**         | "Torso-Pelvis Separation" | Used in `injury_risk.py`. Consider renaming. |

## 4. Prior Art & Defense

- **Kinematic Sequence**: Cheetham, P. J. (2014). "A Simple Model of the Pelvis-Thorax Kinematic Sequence." (Academic citation used in docs).
- **Ball Flight**: Uses standard aerodynamic models (MacDonald & Hanzely). _Verify coefficient source._
- **Chaos Theory**: Implementations are standard mathematical algorithms (Rosenstein for LLE, Higuchi for FD) available in open scientific libraries.
- **Anthropometry**: Explicitly based on Dempster, Winter, and de Leva (see `src/shared/python/validation_pkg/data_fitting.py`).
- **Shaft Dynamics**: Finite Element Method is standard engineering practice (Euler-Bernoulli Beam Theory).

## 5. Risk Mitigation Actions

1.  **Kinematic Sequence**:
    - [ ] **Refactor**: Rename `KinematicSequenceAnalyzer` to `SegmentTimingAnalyzer`.
    - [ ] **Code Change**: Remove `efficiency_score` calculation in `pca_analysis.py`.
2.  **DTW Scoring**:
    - [ ] **Refactor**: Abstract the scoring mechanism.
    - [ ] **Alternative**: Implement `KeyframeSimilarity` (P-positions) as default.
3.  **Injury Risk**:
    - [ ] **Review**: Verify "X-Factor Stretch" thresholds.
    - [ ] **Rename**: Consider "Torso-Pelvis Separation" instead of "X-Factor".
4.  **Ball Flight**:
    - [ ] **Documentation**: Cite source for `0.21` etc. coefficients.
5.  **Documentation**:
    - [ ] Add "Methodology" citations to all analysis classes to prove reliance on public academic research rather than competitor patents.

## 6. Change Log

- **2026-02-24**: Updated review.
    - Added **Injury Risk Scoring** (Medium Risk) due to `injury_risk.py`.
    - Flagged "X-Factor" as potential trademark/patent risk.
    - Confirmed **Shaft Flexibility** uses standard FEM (Low Risk).
    - Re-confirmed High Risk status of `pca_analysis.py` and `comparative_analysis.py`.
- **2026-02-19**: Updated review.
    - Confirmed **Swing DNA** remediation is complete.
    - Identified active **High Risk** in `pca_analysis.py` (TPI efficiency score).
    - Located DTW logic in `src/shared/python/validation_pkg/comparative_analysis.py` (High Risk).
    - Updated file paths for Physics and Validation modules.
- **2026-02-18**: Previous review.
    - Updated `SwingComparator` formula description.
    - Explicitly linked all major risks to issue tracker files.
- **2026-02-05**: Updated review.
    - Added **Ball Flight Coefficients** as Medium Risk.
    - Verified "Swing DNA" remediation.
- **2026-01-26**: Initial comprehensive review.
    - Confirmed remediation of "Swing DNA" in UI code.
    - Flagged "Kinematic Sequence" and "DTW Scoring" as Critical/High risks.
