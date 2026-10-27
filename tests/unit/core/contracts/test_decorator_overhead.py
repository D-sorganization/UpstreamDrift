"""Tests verifying that contract decorators do not re-introspect signatures on every call (UD-13902)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

import src.shared.python._contracts_decorators as contracts_decorators
import src.shared.python.core.contracts.decorators as core_decorators
from src.shared.python._contracts_level import _ContractState
from src.shared.python.core.contracts.exceptions import PreconditionError
from src.shared.python.core.contracts.level import (
    ContractLevel,
    get_contract_level,
    set_contract_level,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def enforce_contracts():
    prev_level = get_contract_level()
    prev_state_level = _ContractState.level
    set_contract_level(ContractLevel.ENFORCE)
    _ContractState.level = contracts_decorators.ContractLevel.ENFORCE
    try:
        yield
    finally:
        set_contract_level(prev_level)
        _ContractState.level = prev_state_level


def test_precondition_does_not_introspect_per_call(monkeypatch: pytest.MonkeyPatch):
    """Sync wrapper in core.contracts.decorators must not inspect signature on each call."""
    orig_sig = inspect.signature
    spy = MagicMock(side_effect=orig_sig)
    monkeypatch.setattr(inspect, "signature", spy)

    @core_decorators.precondition(lambda x: x > 0, "x must be positive")
    def double(x: int) -> int:
        """Double x."""
        return x * 2

    # Reset spy after decoration
    spy.reset_mock()

    for i in range(1, 101):
        assert double(i) == i * 2

    assert spy.call_count <= 1


@pytest.mark.asyncio
async def test_precondition_async_does_not_introspect_per_call(
    monkeypatch: pytest.MonkeyPatch,
):
    """Async wrapper in core.contracts.decorators must not inspect signature on each call."""
    orig_sig = inspect.signature
    spy = MagicMock(side_effect=orig_sig)
    monkeypatch.setattr(inspect, "signature", spy)

    @core_decorators.precondition(lambda x: x > 0, "x must be positive")
    async def async_double(x: int) -> int:
        return x * 2

    # Reset spy after decoration
    spy.reset_mock()

    for i in range(1, 101):
        assert await async_double(i) == i * 2

    assert spy.call_count <= 1


def test_contracts_decorators_module_precondition_does_not_introspect_per_call(
    monkeypatch: pytest.MonkeyPatch,
):
    """_contracts_decorators.precondition must not inspect signatures per call."""
    orig_sig = inspect.signature
    spy = MagicMock(side_effect=orig_sig)
    monkeypatch.setattr(inspect, "signature", spy)

    @contracts_decorators.precondition(lambda x: x > 0, "x must be positive")
    def square(x: int) -> int:
        return x * x

    spy.reset_mock()

    for i in range(1, 101):
        assert square(i) == i * i

    assert spy.call_count <= 1


def test_precondition_behavior_violation_raises():
    """Violating a precondition raises PreconditionError when enforced."""

    @core_decorators.precondition(lambda x: x > 0, "x must be positive")
    def inverse(x: float) -> float:
        return 1.0 / x

    assert inverse(2.0) == 0.5
    with pytest.raises(PreconditionError):
        inverse(-1.0)


def test_precondition_behavior_bad_args_raises_typeerror():
    """Calling with wrong arguments raises TypeError from the function signature."""

    @core_decorators.precondition(lambda *args, **kwargs: True, "always true")
    def add(x: int, y: int) -> int:
        return x + y

    with pytest.raises(TypeError):
        add(1)  # type: ignore[call-arg]


def test_precondition_preserves_metadata():
    """functools.wraps metadata (__name__, __doc__, __annotations__) is preserved."""

    @core_decorators.precondition(lambda x: x > 0, "x positive")
    def documented(x: int) -> int:
        """Documented func."""
        return x

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Documented func."
    assert documented.__annotations__["x"] in (int, "int")
