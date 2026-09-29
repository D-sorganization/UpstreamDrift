"""Tests for src.shared.python.contracts (Issues #1949, #1744).

Note: Tests for src.shared.python.core.contracts are in tests/unit/dbc/.
This file tests the outer src.shared.python.contracts module.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

# ---------------------------------------------------------------------------
# Helpers — enforce mode for all tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def enforce_contracts(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setenv("DBC_LEVEL", "enforce")
    import src.shared.python.contracts as _contracts

    # Use set_contract_level() instead of importlib.reload() to avoid
    # destroying class identity. reload() creates new class objects that break
    # isinstance() checks in other test modules that captured the old classes
    # at collection time (e.g., test_dbc_decorators.py).
    original_level = _contracts.get_contract_level()
    _contracts.set_contract_level(_contracts.ContractLevel.ENFORCE)
    yield
    _contracts.set_contract_level(original_level)


def _get_contracts():
    """Import after env var is set."""
    import src.shared.python.contracts as c

    return c


# ---------------------------------------------------------------------------
# ContractLevel enum
# ---------------------------------------------------------------------------


class TestContractLevel:
    def test_off_exists(self) -> None:
        c = _get_contracts()
        assert c.ContractLevel.OFF

    def test_warn_exists(self) -> None:
        c = _get_contracts()
        assert c.ContractLevel.WARN

    def test_enforce_exists(self) -> None:
        c = _get_contracts()
        assert c.ContractLevel.ENFORCE

    def test_contracts_shared_three_levels(self) -> None:
        c = _get_contracts()
        assert len(c.ContractLevel) == 3


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    def test_contract_violation_is_assertion_error(self) -> None:
        c = _get_contracts()
        assert issubclass(c.ContractViolationError, AssertionError)

    def test_precondition_error_is_contract_violation(self) -> None:
        c = _get_contracts()
        assert issubclass(c.PreconditionError, c.ContractViolationError)

    def test_postcondition_error_is_contract_violation(self) -> None:
        c = _get_contracts()
        assert issubclass(c.PostconditionError, c.ContractViolationError)

    def test_invariant_error_is_contract_violation(self) -> None:
        c = _get_contracts()
        assert issubclass(c.InvariantError, c.ContractViolationError)


# ---------------------------------------------------------------------------
# require()
# ---------------------------------------------------------------------------


class TestRequire:
    def test_passing_condition_does_not_raise(self) -> None:
        c = _get_contracts()
        c.set_contract_level(c.ContractLevel.ENFORCE)
        c.require(True, "must be true")  # should not raise

    def test_failing_condition_raises(self) -> None:
        c = _get_contracts()
        c.set_contract_level(c.ContractLevel.ENFORCE)
        with pytest.raises(c.ContractViolationError):
            c.require(False, "failure message")

    def test_off_mode_skips_check(self) -> None:
        c = _get_contracts()
        c.set_contract_level(c.ContractLevel.OFF)
        c.require(False, "ignored")  # should not raise

    def test_warn_mode_does_not_raise(self) -> None:
        c = _get_contracts()
        c.set_contract_level(c.ContractLevel.WARN)
        c.require(False, "warned")  # should not raise


# ---------------------------------------------------------------------------
# ensure()
# ---------------------------------------------------------------------------


class TestEnsure:
    def test_passing_condition_does_not_raise(self) -> None:
        c = _get_contracts()
        c.set_contract_level(c.ContractLevel.ENFORCE)
        c.ensure(True, "ok")

    def test_failing_condition_raises(self) -> None:
        c = _get_contracts()
        c.set_contract_level(c.ContractLevel.ENFORCE)
        with pytest.raises(c.ContractViolationError):
            c.ensure(False, "postcondition failed")


# ---------------------------------------------------------------------------
# precondition decorator
# ---------------------------------------------------------------------------


class TestPreconditionDecorator:
    def test_passing_precondition_runs_function(self) -> None:
        c = _get_contracts()
        c.set_contract_level(c.ContractLevel.ENFORCE)

        @c.precondition(lambda x: x > 0, "x must be positive")
        def compute(x: float) -> float:
            return x * 2.0

        assert compute(5.0) == 10.0

    def test_failing_precondition_raises(self) -> None:
        c = _get_contracts()
        c.set_contract_level(c.ContractLevel.ENFORCE)

        @c.precondition(lambda x: x > 0, "x must be positive")
        def compute(x: float) -> float:
            return x * 2.0

        with pytest.raises(c.ContractViolationError):
            compute(-1.0)

    def test_contracts_shared_preserves_function_name(self) -> None:
        c = _get_contracts()
        c.set_contract_level(c.ContractLevel.ENFORCE)

        @c.precondition(lambda x: x > 0, "positive")
        def my_named_fn(x: float) -> float:
            return x

        assert my_named_fn.__name__ == "my_named_fn"


# ---------------------------------------------------------------------------
# postcondition decorator
# ---------------------------------------------------------------------------


class TestPostconditionDecorator:
    def test_passing_postcondition_returns_value(self) -> None:
        c = _get_contracts()
        c.set_contract_level(c.ContractLevel.ENFORCE)

        @c.postcondition(lambda r: r >= 0, "result non-negative")
        def square(x: float) -> float:
            return x**2

        assert square(-3.0) == 9.0

    def test_failing_postcondition_raises(self) -> None:
        c = _get_contracts()
        c.set_contract_level(c.ContractLevel.ENFORCE)

        @c.postcondition(lambda r: r > 0, "result must be positive")
        def bad_fn() -> float:
            return -1.0

        with pytest.raises(c.ContractViolationError):
            bad_fn()


# ---------------------------------------------------------------------------
# contract() combined
# ---------------------------------------------------------------------------


class TestContractDecorator:
    def test_pre_and_post_passing(self) -> None:
        c = _get_contracts()
        c.set_contract_level(c.ContractLevel.ENFORCE)

        @c.contract(
            pre=lambda x: x >= 0,
            post=lambda r: r >= 0,
        )
        def sqrt_approx(x: float) -> float:
            return x**0.5

        assert sqrt_approx(4.0) == pytest.approx(2.0)

    def test_pre_violation_raises(self) -> None:
        c = _get_contracts()
        c.set_contract_level(c.ContractLevel.ENFORCE)

        @c.contract(pre=lambda x: x >= 0)
        def sqrt_approx(x: float) -> float:
            return x**0.5

        with pytest.raises(c.ContractViolationError):
            sqrt_approx(-1.0)
