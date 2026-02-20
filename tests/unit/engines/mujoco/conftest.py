"""Pytest configuration for MuJoCo physics engine tests.

Path configuration is centralized in pyproject.toml [tool.pytest.ini_options].
This follows DRY principles from The Pragmatic Programmer.
"""

import importlib.util
import sys
from unittest.mock import MagicMock

# Mock mujoco to allow importing modules that depend on it, only if it's missing
if importlib.util.find_spec("mujoco") is None:
    sys.modules["mujoco"] = MagicMock()
    # Also mock viewer if needed
    sys.modules["mujoco.viewer"] = MagicMock()
