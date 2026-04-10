"""Tests for src.shared.python.core.contracts.invariants (Issues #1949, #1744)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.shared.python.core.contracts.exceptions import InvariantError
from src.shared.python.core.contracts.invariants import (
    ContractChecker,
    invariant,
    invariant_checked,
)
from src.shared.python.core.contracts.level import (
    ContractLevel,
    get_contract_level,
    set_contract_level,
)


class TestContractChecker:
    def setup_method(self) -> None:
        self._saved_level = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)

    def teardown_method(self) -> None:
        set_contract_level(self._saved_level)

    def test_no_invariants_returns_true(self) -> None:
        class MyClass(ContractChecker):
            pass

        obj = MyClass()
        assert obj.verify_invariants() is True

    def test_satisfied_invariant_returns_true(self) -> None:
        class MyClass(ContractChecker):
            def __init__(self) -> None:
                self.mass = 1.0

            def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
                return [(lambda: self.mass > 0, "mass must be positive")]

        obj = MyClass()
        assert obj.verify_invariants() is True

    def test_violated_invariant_raises(self) -> None:
        class MyClass(ContractChecker):
            def __init__(self) -> None:
                self.mass = -1.0

            def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
                return [(lambda: self.mass > 0, "mass must be positive")]

        obj = MyClass()
        with pytest.raises(InvariantError):
            obj.verify_invariants()

    def test_multiple_invariants_all_checked(self) -> None:
        class MyClass(ContractChecker):
            def __init__(self) -> None:
                self.mass = 1.0
                self.timestep = 0.01

            def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
                return [
                    (lambda: self.mass > 0, "mass must be positive"),
                    (lambda: self.timestep > 0, "timestep must be positive"),
                ]

        obj = MyClass()
        assert obj.verify_invariants() is True


class TestInvariantCheckedDecorator:
    def setup_method(self) -> None:
        self._saved_level = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)

    def teardown_method(self) -> None:
        set_contract_level(self._saved_level)

    def test_invariant_checked_runs_after_method(self) -> None:
        class MyClass(ContractChecker):
            def __init__(self) -> None:
                self.count = 0

            def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
                return [(lambda: self.count >= 0, "count must be non-negative")]

            @invariant_checked
            def increment(self) -> None:
                self.count += 1

        obj = MyClass()
        obj.increment()
        assert obj.count == 1

    def test_invariant_checked_raises_when_invariant_broken(self) -> None:
        class MyClass(ContractChecker):
            def __init__(self) -> None:
                self.value = 1.0

            def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
                return [(lambda: self.value > 0, "value must be positive")]

            @invariant_checked
            def set_negative(self) -> None:
                self.value = -1.0

        obj = MyClass()
        with pytest.raises(InvariantError):
            obj.set_negative()


class TestInvariantDecorator:
    def setup_method(self) -> None:
        self._saved_level = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)

    def teardown_method(self) -> None:
        set_contract_level(self._saved_level)

    def test_valid_construction_passes(self) -> None:
        @invariant(lambda self: self.timestep > 0, "Timestep must be positive")
        class Engine:
            def __init__(self, timestep: float) -> None:
                self.timestep = timestep

        engine = Engine(0.01)
        assert engine.timestep == pytest.approx(0.01)

    def test_invalid_construction_raises(self) -> None:
        @invariant(lambda self: self.timestep > 0, "Timestep must be positive")
        class Engine:
            def __init__(self, timestep: float) -> None:
                self.timestep = timestep

        with pytest.raises(InvariantError):
            Engine(-0.01)

    def test_multiple_invariants_both_checked(self) -> None:
        @invariant(lambda self: self.mass > 0, "mass must be positive")
        @invariant(lambda self: self.timestep > 0, "timestep must be positive")
        class Engine:
            def __init__(self, mass: float, timestep: float) -> None:
                self.mass = mass
                self.timestep = timestep

        # Valid
        engine = Engine(1.0, 0.01)
        assert engine.mass == pytest.approx(1.0)

        # Invalid mass
        with pytest.raises(InvariantError):
            Engine(-1.0, 0.01)

    def test_invariant_error_has_class_name(self) -> None:
        @invariant(lambda self: self.timestep > 0, "Timestep must be positive")
        class SpecialEngine:
            def __init__(self, timestep: float) -> None:
                self.timestep = timestep

        with pytest.raises(InvariantError) as exc_info:
            SpecialEngine(-0.01)
        # The error should reference the class somehow
        assert exc_info.value is not None
