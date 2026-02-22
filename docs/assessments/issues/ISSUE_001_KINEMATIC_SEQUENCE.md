---
title: "Legal Risk: Kinematic Sequence Implementation infringes TPI patents"
labels: ["legal", "patent-risk", "critical", "high-priority"]
assignees: ["jules"]
---

## Description

The implementation of "Kinematic Sequence" analysis in the codebase has been identified as a **CRITICAL** patent infringement risk. Specifically, the method of scoring efficiency based on the proximal-to-distal order of peak velocities is a core claim of Titleist Performance Institute (TPI) and K-Motion patents.

### Problem Areas

1.  **`src/shared/python/analysis/pca_analysis.py`**:
    - Contains `efficiency_score = matches / len(expected_order)`.
    - This is a direct implementation of the patented sequencing score.
    - **Status:** Active / Critical.

2.  **`src/shared/python/biomechanics/kinematic_sequence.py`**:
    - Renamed to `SegmentTimingAnalyzer` (Good).
    - But alias `KinematicSequenceAnalyzer` still exists (Acceptable for compat, but risky).

3.  **UI Strings (New Finding 2026-02-25)**:
    - The term "Kinematic Sequence" is still used in user-facing UI strings.
    - `src/shared/python/dashboard/window.py`: `"Kinematic Sequence (Bars)"`
    - `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/advanced_gui_methods.py`: `"Kinematic Sequence"`
    - **Risk:** Trademark / Method Claim confirmation.

## Acceptance Criteria

- [ ] **Code Removal:** Remove the `efficiency_score` calculation from `pca_analysis.py`.
- [ ] **Refactor:** Implement a non-infringing scoring metric (e.g., "Energy Transfer Efficiency" or "Timing Consistency").
- [ ] **UI Updates:** Rename all user-facing strings from "Kinematic Sequence" to "Segment Timing" or "Movement Sequence".
    - [ ] Update `src/shared/python/dashboard/window.py`
    - [ ] Update `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/advanced_gui_methods.py`
- [ ] **Documentation:** Ensure no claims of "Kinematic Sequence Efficiency" are made in docs.

## References

- TPI Patents (various) regarding Kinematic Sequence.
- K-Motion Patents regarding Biofeedback.
