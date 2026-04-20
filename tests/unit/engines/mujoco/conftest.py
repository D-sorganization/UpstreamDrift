"""Pytest configuration for MuJoCo physics engine tests.

Path configuration is centralized in pyproject.toml [tool.pytest.ini_options].
This follows DRY principles from The Pragmatic Programmer.
"""

import sys
from importlib.util import find_spec
from unittest.mock import MagicMock

_MUJOCO_MOCKED = False


def pytest_configure(config):
    """Install mujoco stub early so test-file imports succeed when mujoco is absent."""
    global _MUJOCO_MOCKED
    if find_spec("mujoco") is None:
        sys.modules["mujoco"] = MagicMock()
        sys.modules["mujoco.viewer"] = MagicMock()
        _MUJOCO_MOCKED = True


def pytest_unconfigure(config):
    """Remove mujoco stubs installed by pytest_configure."""
    if _MUJOCO_MOCKED:
        sys.modules.pop("mujoco", None)
        sys.modules.pop("mujoco.viewer", None)
