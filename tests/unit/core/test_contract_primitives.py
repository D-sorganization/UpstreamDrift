"""Tests for src.shared.python.core.contracts.primitives (Issues #1949, #1744)."""

from __future__ import annotations

import pytest

from src.shared.python.core.contracts.exceptions import (
    PostconditionError,
    PreconditionError,
)
from src.shared.python.core.contracts.level import (
    ContractLevel,
    disable_contracts,
    enable_contracts,
    get_contract_level,
    set_contract_level,
)
from src.shared.python.core.contracts.primitives import ensure, require


class TestRequire:
    def setup_method(self) -> None:
        self._original = get_contract_level()
        enable_contracts()

    def teardown_method(self) -> None:
        set_contract_level(self._original)

    def test_true_condition_does_not_raise(self) -> None:
        require(True, "should not raise")

    def test_false_condition_raises_precondition_error(self) -> None:
        with pytest.raises(PreconditionError):
            require(False, "value must be positive")

    def test_error_message_in_exception(self) -> None:
        with pytest.raises(PreconditionError, match="value must be positive"):
            require(False, "value must be positive")

    def test_off_level_no_raise(self) -> None:
        disable_contracts()
        require(False, "this would raise if enabled")  # should not raise

    def test_warn_level_no_raise(self) -> None:
        set_contract_level(ContractLevel.WARN)
        require(False, "just a warning")  # should not raise

    def test_with_value_kwarg(self) -> None:
        with pytest.raises(PreconditionError):
            require(False, "bad input", value=-5)

    def test_expression_result_true(self) -> None:
        x = 5
        require(x > 0, "x must be positive")  # no exception

    def test_expression_result_false(self) -> None:
        x = -1
        with pytest.raises(PreconditionError):
            require(x > 0, "x must be positive")


class TestEnsure:
    def setup_method(self) -> None:
        self._original = get_contract_level()
        enable_contracts()

    def teardown_method(self) -> None:
        set_contract_level(self._original)

    def test_true_condition_does_not_raise(self) -> None:
        ensure(True, "should not raise")

    def test_false_condition_raises_postcondition_error(self) -> None:
        with pytest.raises(PostconditionError):
            ensure(False, "result must be positive")

    def test_error_message_in_exception(self) -> None:
        with pytest.raises(PostconditionError, match="result must be positive"):
            ensure(False, "result must be positive")

    def test_off_level_no_raise(self) -> None:
        disable_contracts()
        ensure(False, "this would raise if enabled")  # should not raise

    def test_warn_level_no_raise(self) -> None:
        set_contract_level(ContractLevel.WARN)
        ensure(False, "just a warning")  # should not raise

    def test_with_value_kwarg(self) -> None:
        with pytest.raises(PostconditionError):
            ensure(False, "bad result", value=None)

    def test_expression_result_true(self) -> None:
        result = 42
        ensure(result > 0, "result must be positive")  # no exception

    def test_expression_result_false(self) -> None:
        result = -1
        with pytest.raises(PostconditionError):
            ensure(result > 0, "result must be positive")
