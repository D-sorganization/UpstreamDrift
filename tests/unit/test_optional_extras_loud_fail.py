"""Smoke coverage for optional engine extras (issue #5914).

These checks distinguish the two states that matter for optional engine
dependencies:

1. the extra is not installed on this machine, so the lane should skip; or
2. the extra is installed, but its import surface is broken, so the lane
   should fail loudly instead of masking the problem behind a skip.
"""

from __future__ import annotations

import importlib

import pytest
from src.shared.python.engine_core.engine_availability import (
    EngineStatus,
    get_engine_error,
    get_engine_status,
)

pytestmark = [pytest.mark.unit]


def _import_optional_engine(engine_name: str, import_name: str) -> object:
    """Import an optional engine extra, skipping only when it is absent."""
    status = get_engine_status(engine_name)
    if status == EngineStatus.NOT_INSTALLED:
        pytest.skip(f"{engine_name} not installed")
    if status == EngineStatus.BROKEN:
        error = get_engine_error(engine_name)
        pytest.fail(f"{engine_name} is installed but broken: {error}")
    return importlib.import_module(import_name)


@pytest.mark.requires_drake
def test_drake_extra_imports_cleanly_when_enabled() -> None:
    """The Drake extra must expose the canonical ``pydrake.all`` surface."""
    module = _import_optional_engine("drake", "pydrake.all")
    assert module is not None


@pytest.mark.requires_opensim
def test_opensim_extra_imports_cleanly_when_enabled() -> None:
    """The OpenSim extra must expose the SWIG-backed ``opensim`` module."""
    module = _import_optional_engine("opensim", "opensim")
    assert module is not None


@pytest.mark.requires_pinocchio
def test_pinocchio_extra_imports_cleanly_when_enabled() -> None:
    """The Pinocchio extra must expose the real binding helpers we depend on."""
    module = _import_optional_engine("pinocchio", "pinocchio")
    assert hasattr(module, "buildModelFromUrdf")
