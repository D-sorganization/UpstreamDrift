"""Tests for src.engines.simscape._errors."""

from __future__ import annotations

import pytest

from src.engines.simscape._errors import (
    SimscapeEngineStartupError,
    SimscapeModelNotFoundError,
    SimscapeNotInstalledError,
    SimscapeSimulationError,
    SimscapeStateError,
)
from src.shared.python.core.error_utils import SimulationError


def test_not_installed_default_message() -> None:
    err = SimscapeNotInstalledError()
    assert "MATLAB Engine for Python is not installed" in str(err)
    assert isinstance(err, SimulationError)


def test_not_installed_with_hint() -> None:
    err = SimscapeNotInstalledError(hint="extra context")
    assert "extra context" in str(err)


def test_model_not_found_stores_path_and_reason() -> None:
    err = SimscapeModelNotFoundError("/some/path.slx", reason="missing")
    assert err.path == "/some/path.slx"
    assert "missing" in str(err)
    assert "/some/path.slx" in str(err)


def test_model_not_found_without_reason() -> None:
    err = SimscapeModelNotFoundError("foo.slx")
    assert err.path == "foo.slx"
    assert "foo.slx" in str(err)


def test_simulation_error_carries_metadata() -> None:
    err = SimscapeSimulationError(
        "boom", matlab_error_id="MATLAB:foo", matlab_traceback="tb"
    )
    assert err.matlab_error_id == "MATLAB:foo"
    assert err.matlab_traceback == "tb"
    assert "boom" in str(err)


def test_simulation_error_defaults() -> None:
    err = SimscapeSimulationError("x")
    assert err.matlab_error_id == ""
    assert err.matlab_traceback == ""


def test_engine_startup_error_carries_id() -> None:
    err = SimscapeEngineStartupError("startup boom", matlab_error_id="MATLAB:license:x")
    assert err.matlab_error_id == "MATLAB:license:x"


def test_state_error_formats_message() -> None:
    err = SimscapeStateError(
        "step", current_state="uninitialized", required_state="loaded"
    )
    assert err.operation == "step"
    assert err.current_state == "uninitialized"
    assert err.required_state == "loaded"
    assert "step" in str(err)
    assert "uninitialized" in str(err)
    assert "loaded" in str(err)


@pytest.mark.parametrize(
    "cls",
    [
        SimscapeNotInstalledError,
        SimscapeModelNotFoundError,
        SimscapeSimulationError,
        SimscapeEngineStartupError,
        SimscapeStateError,
    ],
)
def test_all_subclass_simulation_error(cls: type) -> None:
    assert issubclass(cls, SimulationError)
