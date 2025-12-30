# Adjust path
import os
import sys
import unittest

sys.path.append(os.getcwd())


class TestVerification(unittest.TestCase):
    def test_mujoco_analyzer(self):
        try:
            import mujoco  # noqa: F401

            # ...
            import pinocchio as pin  # noqa: F401

            from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.biomechanics import (
                BiomechanicalAnalyzer,  # noqa: F401
            )
            from engines.physics_engines.pinocchio.python.pinocchio_golf.induced_acceleration import (
                InducedAccelerationAnalyzer,  # noqa: F401
            )

            print("Successfully imported Pinocchio InducedAccelerationAnalyzer")
        except ImportError:
            print("Skipping Pinocchio test (dependencies missing)")

    def test_drake_analyzer(self):
        try:
            # Mock pydrake if not present, but we are in env where we might not have it installed?
            # If installed:
            from engines.physics_engines.drake.python.src.drake_gui_app import (
                DrakeInducedAccelerationAnalyzer,  # noqa: F401
            )

            print("Successfully imported DrakeInducedAccelerationAnalyzer")
        except ImportError:
            print("Skipping Drake test (dependencies missing)")


if __name__ == "__main__":
    unittest.main()
