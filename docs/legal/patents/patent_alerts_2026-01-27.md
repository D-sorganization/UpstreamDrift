# Patent Alert: 2026-01-27

**Severity:** MEDIUM
**Reviewer:** Jules

## 🔍 New Findings

### 1. Gear Effect Magic Numbers
*   **Component**: `src/shared/python/impact_model.py`
*   **Issue**: The `compute_gear_effect_spin` function uses hardcoded scaling factors:
    ```python
    h_scale: float = 100.0
    v_scale: float = 50.0
    ```
*   **Risk**: These "magic numbers" strongly suggest empirical tuning against a proprietary dataset (likely TrackMan or Foresight). Without a public source citation, this could be flagged as "misappropriation of trade secrets" or "clean-room violation" if we cannot prove independent derivation.
*   **Action**: Locate a public physics paper (e.g., Cochran & Stobbs, Science of Golf) that justifies these values, or replace them with a documented geometric derivation.

### 2. Ambiguity in "Efficiency" Terminology
*   **Component**: `src/shared/python/statistical_analysis.py`
*   **Issue**: `StatisticalAnalyzer.compute_swing_profile` calculates an `efficiency_score` based on `peak_ke / total_work`.
*   **Risk**: While "Energy Efficiency" is a safe physics concept, the term "Efficiency" is heavily overloaded in golf. TPI holds patents on "Kinematic Sequence Efficiency" (firing order). Using the bare term "Efficiency" in the UI (as seen in `metrics["Efficiency"]`) creates a risk of confusion and false patent infringement claims.
*   **Action**: Rename this metric to "Energy Efficiency" or "Work-Energy Ratio" in all UI and Code references to clearly distinguish it from Kinematic Efficiency.

### 3. Residual "Swing DNA" Variable Names
*   **Component**: `src/shared/python/dashboard/window.py`
*   **Issue**: While the feature was renamed to "Swing Profile", the variable name `dna` persists:
    ```python
    dna = analyzer.compute_swing_profile()
    ```
*   **Risk**: Low, but discoverable in a code audit. It implies the original intent was to copy Mizuno's "Swing DNA".
*   **Action**: Rename the variable `dna` to `profile` or `swing_profile` during the next refactor cycle.

## Action Items

*   [ ] Physics Team: Validate `impact_model.py` constants.
*   [ ] Frontend Team: Update "Efficiency" label to "Energy Eff."
*   [ ] Dev Team: Refactor `dna` variable names.
