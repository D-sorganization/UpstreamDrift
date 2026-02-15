# Physics Issue: Incorrect Ground Reaction Force Fallback

**ID**: ISSUE_P002
**Category**: Critical Physics Error
**Status**: Open
**Severity**: Critical

## Description
In `src/shared/python/physics/ground_reaction_forces.py`, the `extract_grf_from_contacts` function contains a fallback mechanism for when the physics engine does not report contact data. This fallback incorrectly sums the gravity force of the contact bodies: `total_force[2] += abs(np.sum(g))`.

## Physics Violation
This implementation assumes that the ground reaction force (GRF) equals the static weight of the golfer ($W = mg$). In a dynamic golf swing, the golfer accelerates their center of mass vertically and rotationally, causing GRF to fluctuate significantly. Peak vertical GRF typically exceeds 1.5-2.0x body weight during the downswing (due to $F = m(g + a)$). The current fallback completely ignores the dynamic inertial term ($ma$).

## Impact
-   **Biomechanics**: Drastically underestimates peak forces and power generation.
-   **Plausibility**: Results will show physically impossible swing mechanics (high clubhead speed with static GRF).

## Recommended Fix
1.  **Remove the fallback**: If contact data is missing, raise an error or return `NaN` rather than a physically incorrect value.
2.  **Inverse Dynamics**: If kinematic data (motion capture) is available, implement an inverse dynamics solver to compute the required GRF to produce the observed motion.
