"""Pytest configuration for tests.

This file sets up the test environment, including PYTHONPATH configuration
to ensure imports work correctly in both local and CI environments.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add python/src to PYTHONPATH for test imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Add python/ to PYTHONPATH for mujoco_humanoid_golf imports
python_root = Path(__file__).parent.parent
if str(python_root) not in sys.path:
    sys.path.insert(0, str(python_root))

# Mock mujoco to allow importing modules that depend on it, only if it's missing
if importlib.util.find_spec("mujoco") is None:
    sys.modules["mujoco"] = MagicMock()
    # Also mock viewer if needed
    sys.modules["mujoco.viewer"] = MagicMock()
