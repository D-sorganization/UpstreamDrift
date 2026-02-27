"""test_contracts.py module."""

import pytest
from humanoid_character_builder.contracts import (
    ContractViolationError,
    invariant,
    postcondition,
    precondition,
)


def test_precondition_valid():
    @precondition(lambda x: x > 0)
    def func(x):
        return x

    assert func(5) == 5


def test_precondition_invalid():
    @precondition(lambda x: x > 0)
    def func(x):
        return x

    with pytest.raises(ContractViolationError):
        func(-1)


def test_precondition_argument_binding():
    @precondition(lambda y: y > 0)
    def func(x, y):
        return x + y

    assert func(1, 2) == 3

    with pytest.raises(ContractViolationError):
        func(1, -2)


def test_precondition_with_kwargs():
    @precondition(lambda y: y > 0)
    def func(x, y=10):
        return x + y

    assert func(1) == 11
    assert func(1, y=5) == 6

    with pytest.raises(ContractViolationError):
        func(1, y=-1)


def test_postcondition_valid():
    @postcondition(lambda r: r > 0)
    def func(x):
        return x * x

    assert func(2) == 4


def test_postcondition_invalid():
    @postcondition(lambda r: r > 0)
    def func(x):
        return x

    with pytest.raises(ContractViolationError):
        func(-1)


def test_invariant():
    @invariant(lambda self: self.value > 0)
    class Counter:
        def __init__(self, value):
            self.value = value

        def increment(self):
            self.value += 1
            return self.value

        def decrement(self):
            self.value -= 1
            return self.value

    c = Counter(5)
    assert c.increment() == 6

    with pytest.raises(ContractViolationError):
        Counter(-1)

    c = Counter(1)
    with pytest.raises(ContractViolationError):
        c.decrement()  # Becomes 0, fails invariant > 0


def test_precondition_message():
    @precondition(lambda x: x > 0, "X must be positive")
    def func(x):
        return x

    with pytest.raises(ContractViolationError, match="X must be positive"):
        func(-1)
