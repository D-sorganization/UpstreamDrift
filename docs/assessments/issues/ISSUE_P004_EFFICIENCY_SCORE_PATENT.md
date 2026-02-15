# Legal Issue: Patent Infringement Risk in Efficiency Score

**ID**: ISSUE_P004
**Category**: Legal Risk / Biomechanics Accuracy
**Status**: Open
**Severity**: Critical

## Description
The file `src/shared/python/analysis/pca_analysis.py` calculates an "efficiency score" based on the kinematic sequence of peak velocities. The formula `matches / len(expected_order)` directly ties adherence to a specific order (proximal-to-distal) to "efficiency".

## Legal Implication
This specific definition of efficiency, as a direct function of kinematic sequence order, is closely related to patented methods (e.g., TPI's Kinematic Sequence patents) which claim methods for assessing swing efficiency based on sequence timing.

## Physics Violation
Efficiency is physically defined as $P_{out} / P_{in}$ (power output / power input). Kinematic sequence order is a *style* or *coordination pattern*, not a direct measure of energy efficiency. A golfer with a perfect sequence might have low power output due to poor strength or technique.

## Recommended Fix
1.  **Rename**: Change `efficiency_score` to `sequence_adherence` or `coordination_index`.
2.  **Remove Claims**: Ensure documentation does not equate sequence adherence with energy efficiency.
3.  **Redefine**: If efficiency is needed, calculate it using kinetic energy transfer ratios (Work-Energy Theorem).
