"""Unit tests for the Simscape adapter error hierarchy (#4005)."""

from __future__ import annotations

import pytest
from src.engines.simscape._errors import (
    SimscapeModelNotFoundError,
    SimscapeNotInstalledError,
    SimscapeStateError,
)
from src.shared.python.core.error_utils import SimulationError


@pytest.mark.unit
def test_simscape_not_installed_subclasses_simulation_error() -> None:
    err = SimscapeNotInstalledError()
    assert isinstance(err, SimulationError)
    assert "matlabengine" in str(err).lower()


@pytest.mark.unit
def test_simscape_not_installed_accepts_hint() -> None:
    err = SimscapeNotInstalledError(hint="MATLAB R2024a required")
    assert "R2024a" in str(err)


@pytest.mark.unit
def test_simscape_model_not_found_carries_path() -> None:
    err = SimscapeModelNotFoundError("Foo.slx", reason="file not found")
    assert err.path == "Foo.slx"
    assert "Foo.slx" in str(err)
    assert "file not found" in str(err)
    assert isinstance(err, SimulationError)


@pytest.mark.unit
def test_simscape_state_error_carries_operation_and_states() -> None:
    err = SimscapeStateError(
        "step",
        current_state="uninitialized",
        required_state="loaded|running",
    )
    assert err.operation == "step"
    assert err.current_state == "uninitialized"
    assert err.required_state == "loaded|running"
    assert "step" in str(err)
    assert isinstance(err, SimulationError)


@pytest.mark.unit
def test_errors_are_distinct_classes() -> None:
    assert SimscapeNotInstalledError is not SimscapeModelNotFoundError
    assert SimscapeStateError is not SimscapeModelNotFoundError
    assert SimscapeNotInstalledError is not SimscapeStateError
