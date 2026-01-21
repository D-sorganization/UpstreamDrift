# Patent & Legal Risk Assessment

**Last Updated:** 2026-01-26
**Status:** ACTIVE
**Reviewer:** Jules (Patent Reviewer Agent)

## 1. Known Patent Risks

### High Risk Areas

| Area | Patent(s) / Assignee | Our Implementation | Risk Level | Mitigation |
|------|----------------------|-------------------|------------|------------|
| **Kinematic Sequence** | **TPI (Titleist)**, **K-Motion**<br>Methods for analyzing proximal-to-distal sequencing | `KinematicSequenceAnalyzer` in `shared/python/kinematic_sequence.py`. Explicitly checks order: Pelvis -> Thorax -> Arm -> Club. | **HIGH** | Rename "Kinematic Sequence" to "Segment Timing". Use relative timing gaps instead of absolute efficiency score. **Legal Review Required.** |
| **Motion Scoring (DTW)** | **Zepp**, **Blast Motion**, **K-Motion**<br>Scoring athletic motion via time-warped comparison | `SwingComparator.compute_kinematic_similarity` in `shared/python/swing_comparison.py`. Uses `100 / (1 + DTW_dist)` formula. | **HIGH** | Replace DTW scoring with discrete keyframe correlation or strictly spatial comparison (Hausdorff distance). |
| **Swing DNA** | **Mizuno** (Trademark/Method)<br>Club fitting based on specific metric vector | `SwingProfileMetrics` in `shared/python/statistical_analysis.py`. Visualization as Radar Chart. | **REMEDIATED** | Renamed to "Swing Profile". Metrics changed to: Speed, Sequence, Stability, Efficiency, Power (distinct from Mizuno's). |

### Medium Risk Areas

| Area | Patent(s) / Assignee | Our Implementation | Risk Level | Mitigation |
|------|----------------------|-------------------|------------|------------|
| **Gear Effect Simulation** | **TrackMan**, **Foresight**<br>Methods for simulating ball flight | `compute_gear_effect_spin` in `shared/python/impact_model.py`. Uses linear approximation with constants `100.0`, `50.0`. | **MEDIUM** | Document source of constants. If derived from TrackMan data, ensure "fair use" or clean-room implementation. |
| **Chaos Analysis** | **Various / Niche**<br>Non-linear dynamics in sports | `estimate_lyapunov_exponent`, `compute_fractal_dimension` in `shared/python/statistical_analysis.py`. | **LOW/MED** | Mathematical methods are public domain. Ensure application claims don't infringe specific "method of talent ID" patents. |

### Low Risk Areas
*   **Physics Engine**: Standard rigid body dynamics (Newtonian/Lagrangian) are public domain.
*   **Ball Flight ODE**: Implementation of standard aerodynamic equations (drag, lift, Magnus) is safe if using public coefficients (e.g., Smits & Smith, Bearman).
*   **Statistical Metrics**: Standard deviations, correlations, PCA are generic math.

## 2. Code-Specific Analysis

### `shared/python/kinematic_sequence.py`
*   **Risk**: `KinematicSequenceAnalyzer.analyze`
*   **Finding**: Logic explicitly defines `expected_order = ["Pelvis", "Torso", "Arm", "Club"]`. This specific order check is the core claim of several TPI patents.
*   **Action**: Avoid binary "Efficiency" scoring based on this order. Present data as raw timing charts without "Good/Bad" judgment based on TPI norms.

### `shared/python/swing_comparison.py`
*   **Risk**: `compute_kinematic_similarity`
*   **Finding**: Uses Dynamic Time Warping (DTW) to normalize temporal differences and generate a similarity score (0-100).
*   **Context**: Zepp and K-Motion have patents on "comparing user motion to pro motion using time-warping".
*   **Action**: Isolate this logic. Consider making it an optional plugin or replacing with "Key Position Analysis" (P1-P10 positions).

### `shared/python/impact_model.py`
*   **Risk**: `compute_gear_effect_spin`
*   **Finding**: Uses hardcoded scalars `h_scale=100.0`, `v_scale=50.0`.
*   **Context**: "Magic numbers" suggest empirical tuning against a reference (likely TrackMan).
*   **Action**: Validate these constants against open literature (e.g., Cochran & Stobbs) to establish independent derivation.

## 3. Trademark Concerns

| Term | Owner | Status | Alternative | Notes |
|------|-------|--------|-------------|-------|
| **Swing DNA** | Mizuno | **Remediated** | "Swing Profile" | Code updated in `window.py`. Monitor for regression. |
| **Kinematic Sequence** | TPI (Common Law?) | **Active** | "Segment Timing" | Term is widely used in academia, might be genericized, but risky in commercial product. |
| **Smash Factor** | TrackMan (Orig.) | **Generic** | Keep | Widely accepted as generic golf term now. |
| **TrackMan** | TrackMan A/S | **Ref Only** | N/A | Used only for data validation references. Acceptable nominative use. |

## 4. Prior Art & Defense

*   **Kinematic Sequence**: Cheetham, P. J. (2014). "A Simple Model of the Pelvis-Thorax Kinematic Sequence." (Academic citation used in docs).
*   **Ball Flight**: Uses standard aerodynamic models (MacDonald & Hanzely).
*   **Chaos Theory**: Implementations are standard mathematical algorithms (Rosenstein for LLE, Higuchi for FD) available in open scientific libraries.

## 5. Risk Mitigation Actions

1.  **Kinematic Sequence**:
    *   [ ] **Refactor**: Rename `KinematicSequenceAnalyzer` to `SegmentTimingAnalyzer`.
    *   [ ] **UI Change**: Remove "Efficiency Score" derived from TPI order.
2.  **DTW Scoring**:
    *   [ ] **Refactor**: Abstract the scoring mechanism.
    *   [ ] **Alternative**: Implement `KeyframeSimilarity` (P-positions) as default.
3.  **Documentation**:
    *   [ ] Add "Methodology" citations to all analysis classes to prove reliance on public academic research rather than competitor patents.

## 6. Change Log

*   **2026-01-26**: Initial comprehensive review.
    *   Confirmed remediation of "Swing DNA" in UI code.
    *   Flagged "Kinematic Sequence" and "DTW Scoring" as Critical/High risks.
    *   Identified "Gear Effect" constants as potential IP leakage (Trade Secret/Copyright).
