"""Tests for contracts module (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.contracts import (
    DBC_LEVEL,
    ContractLevel,
    ContractViolationError,
    check_non_negative,
    check_positive,
    check_range,
    ensure,
    get_contract_level,
    precondition,
    require,
    require_positive,
    set_contract_level,
)

_needs_contracts = pytest.mark.skipif(
    DBC_LEVEL != ContractLevel.ENFORCE,
    reason="DBC_LEVEL is not 'enforce'; enforcement tests require ENFORCE mode",
)


@pytest.fixture(autouse=True)
def _enforce_contracts():
    """Force ENFORCE mode by patching the exact module dict that require/ensure read.

    set_contract_level() updates sys.modules[__name__], but in a namespace-package
    environment the module may be loaded under two names, so require.__globals__ can
    be a different dict.  Patching __globals__ directly is always correct.
    """
    _g = require.__globals__  # the actual dict require/ensure/_handle_violation read
    original_dbc = _g["DBC_LEVEL"]
    _g["DBC_LEVEL"] = _g["ContractLevel"].ENFORCE
    _g["_ContractState"].level = _g["ContractLevel"].ENFORCE
    yield
    _g["DBC_LEVEL"] = original_dbc
    _g["_ContractState"].level = original_dbc


class TestRequire:
    def test_passes_when_true(self) -> None:
        require(True, "should pass")  # No exception

    def test_raises_when_false(self) -> None:
        with pytest.raises((ContractViolationError, AssertionError, ValueError)):
            require(False, "must be positive")

    def test_message_in_exception(self) -> None:
        with pytest.raises((ContractViolationError, AssertionError, ValueError)) as exc:
            require(False, "custom error message")
        assert "custom error message" in str(exc.value)


class TestEnsure:
    def test_passes_when_true(self) -> None:
        ensure(True, "post-condition ok")

    def test_raises_when_false(self) -> None:
        with pytest.raises((ContractViolationError, AssertionError, ValueError)):
            ensure(False, "post-condition violated")


class TestContractLevel:
    def test_enum_values_exist(self) -> None:
        assert ContractLevel.ENFORCE is not None

    def test_get_returns_contract_level(self) -> None:
        level = get_contract_level()
        # Use value comparison to avoid namespace-package class identity issues
        # (same class imported via two paths compares unequal with isinstance)
        assert hasattr(level, "value")
        assert level.value in ("enforce", "warn", "off")

    def test_contracts_set_and_get(self) -> None:
        original = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)
        assert get_contract_level() == ContractLevel.ENFORCE
        set_contract_level(original)  # restore


class TestCheckHelpers:
    def test_check_positive_passes(self) -> None:
        check_positive(1.0)

    def test_check_positive_fails(self) -> None:
        with pytest.raises((ContractViolationError, AssertionError, ValueError)):
            check_positive(-1.0)

    def test_check_non_negative_passes(self) -> None:
        check_non_negative(0.0)
        check_non_negative(5.0)

    def test_check_non_negative_fails(self) -> None:
        with pytest.raises((ContractViolationError, AssertionError, ValueError)):
            check_non_negative(-0.1)

    def test_check_range_passes(self) -> None:
        check_range(5.0, 0.0, 10.0)

    def test_check_range_fails_below(self) -> None:
        with pytest.raises((ContractViolationError, AssertionError, ValueError)):
            check_range(-1.0, 0.0, 10.0)

    def test_check_range_fails_above(self) -> None:
        with pytest.raises((ContractViolationError, AssertionError, ValueError)):
            check_range(11.0, 0.0, 10.0)

    def test_require_positive_passes(self) -> None:
        require_positive(1.0)

    def test_require_positive_fails(self) -> None:
        with pytest.raises((ContractViolationError, AssertionError, ValueError)):
            require_positive(0.0)


class TestPreconditionDecorator:
    def test_decorator_allows_valid_input(self) -> None:
        @precondition(lambda self, x: x > 0, "x must be positive")
        def compute(self, x):
            return x * 2

        assert compute(None, 5) == 10

    def test_decorator_raises_on_violation(self) -> None:
        original = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)
        try:

            @precondition(lambda self, x: x > 0, "x must be positive")
            def compute(self, x):
                return x * 2

            with pytest.raises((ContractViolationError, AssertionError, ValueError)):
                compute(None, -1)
        finally:
            set_contract_level(original)
