"""Tier metadata for the MyoSuite engine package."""

TIER = "experimental"

# MyoSuite is a musculoskeletal simulation framework focused on muscle dynamics.
# It does not implement the fit_swing motion-matching interface because:
# 1. MyoSuite uses muscle-driven control (activations) rather than joint-space tracking
# 2. The optimization problem differs fundamentally from rigid-body IK/matching
# 3. Users should use MuJoCo's fit_swing for trajectory optimization, then
#    apply resulting motions to MyoSuite for muscle analysis
FIT_INCAPABLE = True
FIT_INCAPABLE_REASON = """
MyoSuite is designed for muscle-driven forward dynamics simulation, not joint-space
motion matching. The fit_swing interface assumes direct joint control, whereas MyoSuite
requires muscle activation patterns to produce motion.

For motion matching workflows:
1. Use MuJoCo's fit_swing to optimize joint trajectories
2. Apply the resulting motion to MyoSuite for muscle analysis
3. Use induced acceleration analysis to understand muscle contributions
"""
