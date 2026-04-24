"""Tests for src.shared.python.core.contracts.decorators (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts.decorators import (
    finite_result,
    non_empty_result,
    postcondition,
    precondition,
    require_state,
)
from src.shared.python.core.contracts.exceptions import (
    PostconditionError,
    PreconditionError,
    StateError,
)
from src.shared.python.core.contracts.level import (
    ContractLevel,
    get_contract_level,
    set_contract_level,
)


class TestPreconditionDecorator:
    def setup_method(self) -> None:
        self._saved_level = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)

    def teardown_method(self) -> None:
        set_contract_level(self._saved_level)

    def test_passes_when_condition_met(self) -> None:
        @precondition(lambda x: x > 0, "x must be positive")
        def sqrt(x: float) -> float:
            return x**0.5

        assert sqrt(4.0) == pytest.approx(2.0)

    def test_raises_when_condition_fails(self) -> None:
        @precondition(lambda x: x > 0, "x must be positive")
        def sqrt(x: float) -> float:
            return x**0.5

        with pytest.raises(PreconditionError):
            sqrt(-1.0)

    def test_disabled_decorator_skips_check(self) -> None:
        @precondition(lambda x: x > 0, "x must be positive", enabled=False)
        def sqrt(x: float) -> float:
            return x**0.5

        # Should not raise even though condition fails
        result = sqrt(-1.0)
        assert np.isnan(result) or result is not None  # just shouldn't raise

    def test_method_receives_self_correctly(self) -> None:
        class MyClass:
            def __init__(self, val: float) -> None:
                self.val = val

            @precondition(lambda self: self.val > 0, "val must be positive")
            def do_work(self) -> float:
                return self.val * 2

        obj = MyClass(5.0)
        assert obj.do_work() == pytest.approx(10.0)

        obj2 = MyClass(-1.0)
        with pytest.raises(PreconditionError):
            obj2.do_work()


class TestPostconditionDecorator:
    def setup_method(self) -> None:
        self._saved_level = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)

    def teardown_method(self) -> None:
        set_contract_level(self._saved_level)

    def test_passes_when_result_satisfies_condition(self) -> None:
        @postcondition(lambda r: r >= 0, "result must be non-negative")
        def positive_sqrt(x: float) -> float:
            return abs(x) ** 0.5

        assert positive_sqrt(4.0) == pytest.approx(2.0)

    def test_raises_when_result_fails_condition(self) -> None:
        @postcondition(lambda r: r > 0, "result must be positive")
        def bad_func() -> float:
            return -1.0

        with pytest.raises(PostconditionError):
            bad_func()

    def test_disabled_skips_check(self) -> None:
        @postcondition(lambda r: r > 0, "result must be positive", enabled=False)
        def bad_func() -> float:
            return -1.0

        # Should not raise
        assert bad_func() == pytest.approx(-1.0)


class TestRequireStateDecorator:
    def setup_method(self) -> None:
        self._saved_level = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)

    def teardown_method(self) -> None:
        set_contract_level(self._saved_level)

    def test_passes_when_state_valid(self) -> None:
        class Engine:
            def __init__(self) -> None:
                self._is_initialized = True

            @require_state(lambda self: self._is_initialized, "initialized")
            def step(self, dt: float) -> float:
                return dt * 2

        engine = Engine()
        assert engine.step(0.1) == pytest.approx(0.2)

    def test_raises_when_state_not_met(self) -> None:
        class Engine:
            def __init__(self) -> None:
                self._is_initialized = False

            @require_state(lambda self: self._is_initialized, "initialized")
            def step(self, dt: float) -> float:
                return dt * 2

        engine = Engine()
        with pytest.raises(StateError):
            engine.step(0.1)


class TestFiniteResultDecorator:
    def setup_method(self) -> None:
        self._saved_level = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)

    def teardown_method(self) -> None:
        set_contract_level(self._saved_level)

    def test_passes_for_finite_array(self) -> None:
        @finite_result
        def make_array() -> np.ndarray:
            return np.array([1.0, 2.0, 3.0])

        result = make_array()
        assert np.all(np.isfinite(result))

    def test_raises_for_nan_array(self) -> None:
        @finite_result
        def make_nan() -> np.ndarray:
            return np.array([np.nan, 1.0])

        with pytest.raises(PostconditionError):
            make_nan()


class TestNonEmptyResultDecorator:
    def setup_method(self) -> None:
        self._saved_level = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)

    def teardown_method(self) -> None:
        set_contract_level(self._saved_level)

    def test_passes_for_non_empty_array(self) -> None:
        @non_empty_result
        def make_array() -> np.ndarray:
            return np.array([1.0])

        result = make_array()
        assert len(result) > 0

    def test_raises_for_empty_array(self) -> None:
        @non_empty_result
        def make_empty() -> np.ndarray:
            return np.array([])

        with pytest.raises(PostconditionError):
            make_empty()
