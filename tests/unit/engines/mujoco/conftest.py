"""Pytest configuration for MuJoCo physics engine tests.

Path configuration is centralized in pyproject.toml [tool.pytest.ini_options].
This follows DRY principles from The Pragmatic Programmer.
"""

from __future__ import annotations

import contextlib
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from unittest.mock import MagicMock, patch

import pytest

# If mujoco is not installed, install a mock at collection time so that test
# modules that do ``import mujoco`` at the top level can be collected.
# patch.dict via contextlib.ExitStack ensures the mock is removed cleanly when
# pytest exits via pytest_unconfigure, preventing leakage to other processes.
_mujoco_mock_stack = contextlib.ExitStack()

if find_spec("mujoco") is None:
    _mujoco_mock = MagicMock()
    # importlib.util.find_spec() raises ValueError when __spec__ is a MagicMock.
    # Set a proper ModuleSpec so that other test modules that call find_spec("mujoco")
    # at collection time (e.g. to skip tests) do not raise ValueError.
    _mujoco_mock.__spec__ = ModuleSpec("mujoco", None)
    _mujoco_viewer_mock = MagicMock()
    _mujoco_viewer_mock.__spec__ = ModuleSpec("mujoco.viewer", None)
    _mujoco_mock_stack.enter_context(
        patch.dict(
            "sys.modules",
            {
                "mujoco": _mujoco_mock,
                "mujoco.viewer": _mujoco_viewer_mock,
            },
        )
    )


def pytest_unconfigure(config: pytest.Config) -> None:
    """Remove mujoco mock installed at collection time."""
    _mujoco_mock_stack.close()
