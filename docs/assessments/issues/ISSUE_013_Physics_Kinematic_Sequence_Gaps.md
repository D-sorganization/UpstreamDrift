# High Priority Issue: Kinematic Sequence Gaps (Speed Gain & Braking)

**Status:** Open
**Priority:** High
**Labels:** physics-gap, high
**Date Identified:** 2026-01-31
**Author:** PHYSICS AUDITOR

## Description
The `KinematicSequenceAnalyzer` in `src/shared/python/kinematic_sequence.py` identifies the timing of peak velocities but lacks two critical biomechanical metrics standard in modern golf analysis (e.g., TPI, AMM):

1.  **Speed Gain (Efficiency):** The ratio of peak velocity of a distal segment to its proximal neighbor (e.g., Club Speed / Hand Speed). This quantifies the efficiency of energy transfer across the joint.
2.  **Deceleration (Braking):** The rate at which a proximal segment decelerates after its peak. Proper "bracing" or deceleration is crucial for transferring energy to the next segment (whip effect). The current analyzer only looks at peak magnitudes.

## Impact
- **Diagnostic Value:** Coaching insights regarding "why" speed is low (e.g., poor transfer vs. low input) are missing.
- **Completeness:** Fails to meet industry standard expectations for 3D kinematic analysis.

## Affected Files
- `src/shared/python/kinematic_sequence.py`

## Recommended Fix
1.  Add `speed_gain` calculation to `SegmentPeak` or `KinematicSequenceResult`.
    - `gain = peak_distal / peak_proximal`
2.  Add `deceleration_rate` metric.
    - Calculate slope of velocity curve immediately post-peak (e.g., average slope over next 20-50ms).
3.  Differentiate between Angular Velocity (deg/s) and Linear Velocity (m/s) inputs to ensure gains are unit-consistent or explicitly dimensionless ratios.
