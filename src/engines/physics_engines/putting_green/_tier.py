"""Tier metadata for the Putting Green engine package."""

TIER = "core"

# Putting Green is a golf-specific simulation for putting dynamics.
# It does not implement fit_swing because:
# 1. Putting Green models putting stroke dynamics, not full golf swing
# 2. The physics model is specialized for low-speed ball-green interaction
# 3. Users should use MuJoCo's fit_swing for full swing trajectory optimization
FIT_INCAPABLE = True
FIT_INCAPABLE_REASON = """
Putting Green is designed for simulating putting stroke dynamics and ball-green
interaction, not full golf swing motion matching. The fit_swing interface is
designed for full-body swing trajectory optimization.

For putting analysis:
- Use Putting Green's native putting stroke interface
- For full swing analysis, use MuJoCo's fit_swing
"""
