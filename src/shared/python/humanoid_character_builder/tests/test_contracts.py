from typing import Any

"""test_contracts.py module."""

import pytest
from humanoid_character_builder.contracts import (
    ContractViolationError,
    invariant,
    postcondition,
    precondition,
)


def test_precondition_valid() -> Any:
    @precondition(lambda x: x > 0)
    def func(x) -> Any:
        return x

    assert func(5) == 5


def test_precondition_invalid() -> Any:
    @precondition(lambda x: x > 0)
    def func(x) -> Any:
        return x

    with pytest.raises(ContractViolationError):
        func(-1)


def test_precondition_argument_binding() -> Any:
    @precondition(lambda y: y > 0)
    def func(x, y) -> Any:
        return x + y

    assert func(1, 2) == 3

    with pytest.raises(ContractViolationError):
        func(1, -2)


def test_precondition_with_kwargs() -> Any:
    @precondition(lambda y: y > 0)
    def func(x, y=10) -> Any:
        return x + y

    assert func(1) == 11
    assert func(1, y=5) == 6

    with pytest.raises(ContractViolationError):
        func(1, y=-1)


def test_postcondition_valid() -> Any:
    @postcondition(lambda r: r > 0)
    def func(x) -> Any:
        return x * x

    assert func(2) == 4


def test_postcondition_invalid() -> Any:
    @postcondition(lambda r: r > 0)
    def func(x) -> Any:
        return x

    with pytest.raises(ContractViolationError):
        func(-1)


def test_invariant() -> Any:
    @invariant(lambda self: self.value > 0)
    class Counter:
        def __init__(self, value):
            self.value = value

        def increment(self) -> Any:
            self.value += 1
            return self.value

        def decrement(self) -> Any:
            self.value -= 1
            return self.value

    c = Counter(5)
    assert c.increment() == 6

    with pytest.raises(ContractViolationError):
        Counter(-1)

    c = Counter(1)
    with pytest.raises(ContractViolationError):
        c.decrement()  # Becomes 0, fails invariant > 0


def test_precondition_message() -> Any:
    @precondition(lambda x: x > 0, "X must be positive")
    def func(x) -> Any:
        return x

    with pytest.raises(ContractViolationError, match="X must be positive"):
        func(-1)
