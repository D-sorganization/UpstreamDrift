"""Loader-level guard for the optional JaxSim backend (#6880).

``load_jaxsim_engine`` must validate the JaxSim runtime import surface before
reporting success, raising a ``GolfModelingError`` with an install hint when the
``jaxsim`` package cannot be imported. These tests do not require ``jax`` or
``jaxsim`` to be installed; the import surface is monkeypatched.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from src.engines.loaders import load_jaxsim_engine
from src.shared.python.data_io.common_utils import GolfModelingError


def test_load_jaxsim_engine_raises_when_runtime_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing ``jaxsim.api`` import surface must raise with an install hint."""
    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "jaxsim" or name.startswith("jaxsim."):
            raise ImportError("No module named 'jaxsim'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(GolfModelingError) as excinfo:
        load_jaxsim_engine(Path("."))

    message = str(excinfo.value)
    assert "jaxsim" in message.lower()
    assert "upstream-drift[jaxsim]" in message


def test_load_jaxsim_engine_rejects_non_path() -> None:
    """DbC: a non-Path suite_root is rejected before any import work."""
    with pytest.raises(TypeError):
        load_jaxsim_engine("not-a-path")  # type: ignore[arg-type]
