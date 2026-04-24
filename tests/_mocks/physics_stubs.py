"""Reusable sys.modules stub dictionaries for optional third-party packages.

Each helper returns a fresh dict of ``{module_name: MagicMock}`` so callers
can install the mocks via ``monkeypatch.setitem(sys.modules, ...)`` or
``unittest.mock.patch.dict("sys.modules", ...)``. Both approaches clean up
the entries when the test scope exits, preventing the cross-test pollution
described in ``CLAUDE.md`` ("Test pollution: Never ``sys.modules['pydrake']
= MagicMock()`` at module level").

Usage example::

    import pytest
    import sys

    from tests._mocks.physics_stubs import drake_stubs


    @pytest.fixture
    def stubbed_pydrake(monkeypatch):
        for name, mock in drake_stubs().items():
            monkeypatch.setitem(sys.modules, name, mock)
        yield
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


def drake_stubs() -> dict[str, Any]:
    """Return a dict of Drake-related sys.modules stubs."""
    root = MagicMock()
    return {
        "pydrake": root,
        "pydrake.all": root,
        "pydrake.geometry": MagicMock(),
        "pydrake.math": MagicMock(),
        "pydrake.multibody": MagicMock(),
        "pydrake.multibody.plant": MagicMock(),
        "pydrake.multibody.parsing": MagicMock(),
        "pydrake.multibody.tree": MagicMock(),
        "pydrake.systems": MagicMock(),
        "pydrake.systems.framework": MagicMock(),
        "pydrake.systems.analysis": MagicMock(),
    }


def pinocchio_stubs() -> dict[str, Any]:
    """Return a dict of Pinocchio-related sys.modules stubs."""
    return {
        "pinocchio": MagicMock(),
        "pinocchio.casadi": MagicMock(),
        "casadi": MagicMock(),
    }


def mujoco_cv_stubs() -> dict[str, Any]:
    """Return a dict of stubs for optional MuJoCo video-export dependencies."""
    return {
        "cv2": MagicMock(),
        "imageio": MagicMock(),
    }


def hatchling_stubs(hook_interface: Any | None = None) -> dict[str, Any]:
    """Return a dict of Hatchling-related sys.modules stubs.

    If ``hook_interface`` is provided, it is attached as ``BuildHookInterface``
    on the ``hatchling.builders.hooks.plugin.interface`` stub so that
    ``build_hooks`` sees the expected attribute on import.
    """
    interface = MagicMock()
    if hook_interface is not None:
        interface.BuildHookInterface = hook_interface
    return {
        "hatchling": MagicMock(),
        "hatchling.builders": MagicMock(),
        "hatchling.builders.hooks": MagicMock(),
        "hatchling.builders.hooks.plugin": MagicMock(),
        "hatchling.builders.hooks.plugin.interface": interface,
    }


def google_genai_stubs() -> dict[str, Any]:
    """Return stubs for the ``google-generativeai`` package."""
    genai = MagicMock()
    return {
        "google": MagicMock(),
        "google.generativeai": genai,
        "google.generativeai.types": MagicMock(),
    }


def openai_stubs() -> dict[str, Any]:
    """Return stubs for the ``openai`` package."""
    mock = MagicMock()
    mock.OpenAI = MagicMock()
    mock.Anthropic = MagicMock()
    return {"openai": mock}


def anthropic_stubs() -> dict[str, Any]:
    """Return stubs for the ``anthropic`` package."""
    mock = MagicMock()
    mock.OpenAI = MagicMock()
    mock.Anthropic = MagicMock()
    return {"anthropic": mock}


def httpx_stubs() -> dict[str, Any]:
    """Return stubs for the ``httpx`` package used by the Ollama adapter."""

    class _MockConnectError(OSError):
        pass

    class _MockTimeoutException(OSError):
        pass

    mock = MagicMock()
    mock.OpenAI = MagicMock()
    mock.Anthropic = MagicMock()
    mock.ConnectError = _MockConnectError
    mock.TimeoutException = _MockTimeoutException
    return {"httpx": mock}
