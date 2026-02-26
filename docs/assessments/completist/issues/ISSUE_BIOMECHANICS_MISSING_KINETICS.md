---
title: Missing Kinetic Metrics in Biomechanical Analysis
labels: physics-gap, high
assignees: biomechanics-team
---

## Description
The `kinematic_sequence.py` module currently analyzes the timing and velocity of body segments (kinematics) but completely omits kinetic metrics (forces, torques, powers) that are essential for explaining *why* the sequence occurs.

Explicit TODOs include:
- `TODO: Implement Proximal Braking Efficiency calculation.`
- `TODO: Implement Inter-segmental Power Flow calculation.`

## Expected Behavior
The biomechanics module should compute:
1.  **Joint Torques**: Net joint moments at the hips, torso, and shoulders.
2.  **Power Flow**: Energy transfer between segments ($P = M \cdot \omega$) to quantify "Inter-segmental Power Flow".
3.  **Proximal Braking**: The magnitude and rate of deceleration of proximal segments (hips/torso) that allows energy transfer to distal segments (arms/club).

## Impact
- **Diagnostic Incompleteness**: Can describe *what* happened (velocities) but not *how* (forces/torques).
- **Efficiency Metrics**: Cannot compute mechanical efficiency or work done.
- **Injury Risk**: Joint loading (torque/force magnitude) is the primary predictor of injury risk, which is currently unmeasurable.

## Recommended Fix
1.  Integrate inverse dynamics (requires mass properties and accelerations) to compute joint torques.
2.  Implement power calculations ($P = M \cdot \omega$) for each joint.
3.  Calculate "Braking Efficiency" as the ratio of energy transferred to energy dissipated/absorbed by proximal segments.
