"""Tests for src.engines.simscape._engine_pool."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.engines.simscape import _engine_pool
from src.engines.simscape._errors import (
    SimscapeEngineStartupError,
    SimscapeNotInstalledError,
)


@pytest.fixture(autouse=True)
def _reset_engine_singleton() -> None:
    # Reset module-level singleton between tests
    _engine_pool._engine = None  # type: ignore[attr-defined]
    yield
    _engine_pool._engine = None  # type: ignore[attr-defined]


def test_is_matlab_available_force_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UD_SIMSCAPE_FORCE_NO_MATLAB", "1")
    assert _engine_pool.is_matlab_available() is False


def test_is_matlab_available_true_when_module_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UD_SIMSCAPE_FORCE_NO_MATLAB", raising=False)
    import importlib.machinery

    fake_matlab = MagicMock()
    fake_matlab.__spec__ = importlib.machinery.ModuleSpec("matlab", loader=None)
    with patch.dict(sys.modules, {"matlab": fake_matlab}):
        assert _engine_pool.is_matlab_available() is True


def test_is_matlab_available_false_without_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UD_SIMSCAPE_FORCE_NO_MATLAB", raising=False)
    # Real env has no matlab installed
    if "matlab" in sys.modules:
        pytest.skip("matlab module is genuinely installed on this host")
    assert _engine_pool.is_matlab_available() is False


def test_get_shared_engine_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UD_SIMSCAPE_FORCE_NO_MATLAB", "1")
    with pytest.raises(SimscapeNotInstalledError):
        _engine_pool.get_shared_engine()


def _install_fake_matlab() -> tuple[MagicMock, MagicMock]:
    """Install a fake matlab + matlab.engine in sys.modules with valid specs."""
    import importlib.machinery

    fake_engine_mod = MagicMock()
    fake_engine_mod.__spec__ = importlib.machinery.ModuleSpec(
        "matlab.engine", loader=None
    )
    fake_matlab = MagicMock()
    fake_matlab.__spec__ = importlib.machinery.ModuleSpec("matlab", loader=None)
    fake_matlab.engine = fake_engine_mod
    return fake_matlab, fake_engine_mod


def test_get_shared_engine_starts_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UD_SIMSCAPE_FORCE_NO_MATLAB", raising=False)
    fake_engine = MagicMock(name="MatlabEngine")
    fake_matlab, fake_engine_mod = _install_fake_matlab()
    fake_engine_mod.start_matlab.return_value = fake_engine

    with patch.dict(
        sys.modules,
        {"matlab": fake_matlab, "matlab.engine": fake_engine_mod},
    ):
        eng = _engine_pool.get_shared_engine(startup_timeout_s=5.0)
        assert eng is fake_engine
        eng2 = _engine_pool.get_shared_engine(startup_timeout_s=5.0)
        assert eng2 is fake_engine
        assert fake_engine_mod.start_matlab.call_count == 1


def test_get_shared_engine_license_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UD_SIMSCAPE_FORCE_NO_MATLAB", raising=False)
    fake_matlab, fake_engine_mod = _install_fake_matlab()
    fake_engine_mod.start_matlab.side_effect = RuntimeError(
        "MATLAB license checkout failed"
    )
    with (
        patch.dict(
            sys.modules,
            {"matlab": fake_matlab, "matlab.engine": fake_engine_mod},
        ),
        pytest.raises(SimscapeEngineStartupError, match="license"),
    ):
        _engine_pool.get_shared_engine(startup_timeout_s=5.0)


def test_get_shared_engine_generic_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UD_SIMSCAPE_FORCE_NO_MATLAB", raising=False)
    fake_matlab, fake_engine_mod = _install_fake_matlab()
    fake_engine_mod.start_matlab.side_effect = RuntimeError("boom")
    with (
        patch.dict(
            sys.modules,
            {"matlab": fake_matlab, "matlab.engine": fake_engine_mod},
        ),
        pytest.raises(SimscapeEngineStartupError, match="boom"),
    ):
        _engine_pool.get_shared_engine(startup_timeout_s=5.0)


def test_shutdown_shared_engine_quits_and_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UD_SIMSCAPE_FORCE_NO_MATLAB", raising=False)
    fake_engine = MagicMock()
    fake_matlab, fake_engine_mod = _install_fake_matlab()
    fake_engine_mod.start_matlab.return_value = fake_engine
    with patch.dict(
        sys.modules,
        {"matlab": fake_matlab, "matlab.engine": fake_engine_mod},
    ):
        _engine_pool.get_shared_engine(startup_timeout_s=5.0)
        _engine_pool.shutdown_shared_engine()
        fake_engine.quit.assert_called_once()
        _engine_pool.shutdown_shared_engine()


def test_shutdown_when_no_engine_is_noop() -> None:
    _engine_pool._engine = None  # type: ignore[attr-defined]
    _engine_pool.shutdown_shared_engine()
    assert _engine_pool._engine is None  # type: ignore[attr-defined]


def test_shutdown_swallows_quit_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UD_SIMSCAPE_FORCE_NO_MATLAB", raising=False)
    fake_engine = MagicMock()
    fake_engine.quit.side_effect = RuntimeError("quit boom")
    _engine_pool._engine = fake_engine  # type: ignore[attr-defined]
    # Should not raise
    _engine_pool.shutdown_shared_engine()
    assert _engine_pool._engine is None  # type: ignore[attr-defined]
