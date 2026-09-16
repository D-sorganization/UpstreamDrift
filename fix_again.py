import subprocess
import os

# We need to make sure we don't break the drake test in `test_cross_engine_grip_agreement`.
# Let's revert src/engines/physics_engines/mujoco/python/full_body_markers.py back to what it was
# Or run the full motion_matching test locally to see what fails exactly.
# Drake outputs all-NaNs and we'll check how it handles it.
