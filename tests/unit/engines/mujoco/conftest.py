"""Pytest configuration for MuJoCo physics engine tests.

Path configuration is centralized in pyproject.toml [tool.pytest.ini_options].
This follows DRY principles from The Pragmatic Programmer.

Optional-dependency stubs (``cv2``, ``imageio``, ``mujoco``) are installed in
``pytest_configure`` so they are available before any test module imports
``src.engines.physics_engines.mujoco.*`` — which in turn does ``import cv2``
and ``import imageio`` lazily. Stubs are removed in ``pytest_unconfigure``
to avoid leaking across sessions. This replaces the module-level
``sys.modules[...] = MagicMock()`` pattern banned by CLAUDE.md.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec
from unittest.mock import MagicMock

import pytest

from tests._mocks.physics_stubs import mujoco_cv_stubs

_installed_keys: list[str] = []


def pytest_configure(config: pytest.Config) -> None:
    """Install MuJoCo video-export optional-dependency stubs."""
    stubs: dict = dict(mujoco_cv_stubs())
    if find_spec("mujoco") is None:
        stubs.setdefault("mujoco", MagicMock())
        stubs.setdefault("mujoco.viewer", MagicMock())
    for key, value in stubs.items():
        if key not in sys.modules:
            sys.modules[key] = value
            _installed_keys.append(key)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Remove stubs installed by :func:`pytest_configure`."""
    while _installed_keys:
        key = _installed_keys.pop()
        sys.modules.pop(key, None)
