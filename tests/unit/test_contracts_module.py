"""Unit tests for src/shared/python/contracts.py.

Tests cover the DbC (Design by Contract) helpers: ContractLevel enum,
require/ensure/invariant primitives, precondition/postcondition/contract
decorators, set_contract_level/get_contract_level, and all exception types.

All tests are headless-safe (no display server, no heavy deps).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ContractLevel enum
# ---------------------------------------------------------------------------


class TestContractLevel:
    """Tests for ContractLevel enum."""

    def test_contracts_module_three_levels(self) -> None:
        """ContractLevel has exactly three variants."""
        from src.shared.python.contracts import ContractLevel

        levels = [ContractLevel.OFF, ContractLevel.WARN, ContractLevel.ENFORCE]
        assert len(levels) == 3

    def test_values(self) -> None:
        """Enum values match expected strings."""
        from src.shared.python.contracts import ContractLevel

        assert ContractLevel.OFF.value == "off"
        assert ContractLevel.WARN.value == "warn"
        assert ContractLevel.ENFORCE.value == "enforce"


# ---------------------------------------------------------------------------
# set_contract_level / get_contract_level
# ---------------------------------------------------------------------------


class TestContractLevelGetSet:
    """Tests for set_contract_level and get_contract_level."""

    def setup_method(self) -> None:
        """Save the original contract level before each test."""
        from src.shared.python.contracts import get_contract_level

        self._original = get_contract_level()

    def teardown_method(self) -> None:
        """Restore the original contract level after each test."""
        from src.shared.python.contracts import set_contract_level

        set_contract_level(self._original)

    def test_get_contract_level_returns_level(self) -> None:
        """get_contract_level returns a ContractLevel instance."""
        from src.shared.python.contracts import ContractLevel, get_contract_level

        level = get_contract_level()
        assert isinstance(level, ContractLevel)

    def test_set_and_get_roundtrip(self) -> None:
        """Setting a level and reading it back gives the same level."""
        from src.shared.python.contracts import (
            ContractLevel,
            get_contract_level,
            set_contract_level,
        )

        for target in (ContractLevel.OFF, ContractLevel.WARN, ContractLevel.ENFORCE):
            set_contract_level(target)
            assert get_contract_level() == target

    def test_set_level_updates_module_alias(self) -> None:
        """After set_contract_level, the module-level DBC_LEVEL alias updates too."""
        import src.shared.python.contracts as contracts_module
        from src.shared.python.contracts import ContractLevel, set_contract_level

        set_contract_level(ContractLevel.WARN)
        assert contracts_module.DBC_LEVEL == ContractLevel.WARN


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Tests for ContractViolationError and its subclasses."""

    def test_contract_violation_is_assertion_and_value_error(self) -> None:
        """ContractViolationError inherits from both AssertionError and ValueError."""
        from src.shared.python.contracts import ContractViolationError

        exc = ContractViolationError("pre-condition", "test error", 42)
        assert isinstance(exc, AssertionError)
        assert isinstance(exc, ValueError)

    def test_precondition_error(self) -> None:
        """PreconditionError is a ContractViolationError."""
        from src.shared.python.contracts import (
            ContractViolationError,
            PreconditionError,
        )

        exc = PreconditionError("x must be positive", -1)
        assert isinstance(exc, ContractViolationError)
        assert "pre-condition" in str(exc)

    def test_postcondition_error(self) -> None:
        """PostconditionError is a ContractViolationError."""
        from src.shared.python.contracts import (
            ContractViolationError,
            PostconditionError,
        )

        exc = PostconditionError("result must be non-negative", -5.0)
        assert isinstance(exc, ContractViolationError)
        assert "post-condition" in str(exc)

    def test_invariant_error(self) -> None:
        """InvariantError is a ContractViolationError."""
        from src.shared.python.contracts import ContractViolationError, InvariantError

        exc = InvariantError("mass must remain positive")
        assert isinstance(exc, ContractViolationError)
        assert "invariant" in str(exc)

    def test_exception_message_includes_value(self) -> None:
        """ContractViolationError message includes the bad value when provided."""
        from src.shared.python.contracts import PreconditionError

        exc = PreconditionError("must be positive", -3)
        assert "-3" in str(exc)

    def test_exception_message_without_value(self) -> None:
        """ContractViolationError works without a value argument."""
        from src.shared.python.contracts import PreconditionError

        exc = PreconditionError("condition failed")
        assert "condition failed" in str(exc)


# ---------------------------------------------------------------------------
# require / ensure / invariant primitives
# ---------------------------------------------------------------------------


class TestRequire:
    """Tests for the require() primitive."""

    def setup_method(self) -> None:
        from src.shared.python.contracts import get_contract_level

        self._original = get_contract_level()

    def teardown_method(self) -> None:
        from src.shared.python.contracts import set_contract_level

        set_contract_level(self._original)

    def test_passing_condition_does_not_raise(self) -> None:
        """require(True, ...) does not raise."""
        from src.shared.python.contracts import (
            ContractLevel,
            require,
            set_contract_level,
        )

        set_contract_level(ContractLevel.ENFORCE)
        require(True, "this is fine")

    def test_failing_condition_raises_precondition_error(self) -> None:
        """require(False, ...) raises PreconditionError in ENFORCE mode."""
        from src.shared.python.contracts import (
            ContractLevel,
            PreconditionError,
            require,
            set_contract_level,
        )

        set_contract_level(ContractLevel.ENFORCE)
        with pytest.raises(PreconditionError):
            require(False, "value must be positive", -1)

    def test_off_mode_skips_check(self) -> None:
        """require(False, ...) does not raise in OFF mode."""
        from src.shared.python.contracts import (
            ContractLevel,
            require,
            set_contract_level,
        )

        set_contract_level(ContractLevel.OFF)
        require(False, "this would normally fail")  # should not raise

    def test_warn_mode_does_not_raise(self) -> None:
        """require(False, ...) logs warning but does not raise in WARN mode."""
        from src.shared.python.contracts import (
            ContractLevel,
            require,
            set_contract_level,
        )

        set_contract_level(ContractLevel.WARN)
        require(False, "this is a warning only")  # should not raise


class TestEnsure:
    """Tests for the ensure() primitive."""

    def setup_method(self) -> None:
        from src.shared.python.contracts import get_contract_level

        self._original = get_contract_level()

    def teardown_method(self) -> None:
        from src.shared.python.contracts import set_contract_level

        set_contract_level(self._original)

    def test_passing_condition_does_not_raise(self) -> None:
        """ensure(True, ...) does not raise."""
        from src.shared.python.contracts import (
            ContractLevel,
            ensure,
            set_contract_level,
        )

        set_contract_level(ContractLevel.ENFORCE)
        ensure(True, "result is valid")

    def test_failing_condition_raises_postcondition_error(self) -> None:
        """ensure(False, ...) raises PostconditionError in ENFORCE mode."""
        from src.shared.python.contracts import (
            ContractLevel,
            PostconditionError,
            ensure,
            set_contract_level,
        )

        set_contract_level(ContractLevel.ENFORCE)
        with pytest.raises(PostconditionError):
            ensure(False, "result must be non-negative", -5.0)

    def test_off_mode_skips_check(self) -> None:
        """ensure(False, ...) does not raise in OFF mode."""
        from src.shared.python.contracts import (
            ContractLevel,
            ensure,
            set_contract_level,
        )

        set_contract_level(ContractLevel.OFF)
        ensure(False, "skipped")  # should not raise


class TestInvariant:
    """Tests for the invariant() primitive."""

    def setup_method(self) -> None:
        from src.shared.python.contracts import get_contract_level

        self._original = get_contract_level()

    def teardown_method(self) -> None:
        from src.shared.python.contracts import set_contract_level

        set_contract_level(self._original)

    def test_passing_invariant_does_not_raise(self) -> None:
        """invariant(True, ...) does not raise."""
        from src.shared.python.contracts import (
            ContractLevel,
            invariant,
            set_contract_level,
        )

        set_contract_level(ContractLevel.ENFORCE)
        invariant(True, "system is consistent")

    def test_failing_invariant_raises_invariant_error(self) -> None:
        """invariant(False, ...) raises InvariantError in ENFORCE mode."""
        from src.shared.python.contracts import (
            ContractLevel,
            InvariantError,
            invariant,
            set_contract_level,
        )

        set_contract_level(ContractLevel.ENFORCE)
        with pytest.raises(InvariantError):
            invariant(False, "invariant violated")


# ---------------------------------------------------------------------------
# precondition decorator
# ---------------------------------------------------------------------------


class TestPreconditionDecorator:
    """Tests for the @precondition decorator."""

    def setup_method(self) -> None:
        from src.shared.python.contracts import get_contract_level

        self._original = get_contract_level()

    def teardown_method(self) -> None:
        from src.shared.python.contracts import set_contract_level

        set_contract_level(self._original)

    def test_valid_call_passes_through(self) -> None:
        """Function with satisfied precondition runs normally."""
        from src.shared.python.contracts import (
            ContractLevel,
            precondition,
            set_contract_level,
        )

        set_contract_level(ContractLevel.ENFORCE)

        @precondition(lambda x: x > 0, "x must be positive")
        def sqrt_approx(x: float) -> float:
            """Compute approximate square root."""
            return x**0.5

        result = sqrt_approx(4.0)
        assert abs(result - 2.0) < 1e-9

    def test_violated_precondition_raises(self) -> None:
        """Function with violated precondition raises PreconditionError."""
        from src.shared.python.contracts import (
            ContractLevel,
            PreconditionError,
            precondition,
            set_contract_level,
        )

        set_contract_level(ContractLevel.ENFORCE)

        @precondition(lambda x: x > 0, "x must be positive")
        def sqrt_approx(x: float) -> float:
            """Compute approximate square root."""
            return x**0.5

        with pytest.raises(PreconditionError):
            sqrt_approx(-1.0)

    def test_off_mode_bypasses_decorator(self) -> None:
        """In OFF mode, precondition check is skipped entirely."""
        from src.shared.python.contracts import (
            ContractLevel,
            precondition,
            set_contract_level,
        )

        set_contract_level(ContractLevel.OFF)

        @precondition(lambda x: x > 0, "x must be positive")
        def neg_allowed(x: float) -> float:
            """Return x."""
            return x

        # Would raise in ENFORCE mode, but OFF mode skips the check
        result = neg_allowed(-5.0)
        assert result == -5.0


# ---------------------------------------------------------------------------
# postcondition decorator
# ---------------------------------------------------------------------------


class TestPostconditionDecorator:
    """Tests for the @postcondition decorator."""

    def setup_method(self) -> None:
        from src.shared.python.contracts import get_contract_level

        self._original = get_contract_level()

    def teardown_method(self) -> None:
        from src.shared.python.contracts import set_contract_level

        set_contract_level(self._original)

    def test_valid_return_passes_through(self) -> None:
        """Function returning valid result passes postcondition."""
        from src.shared.python.contracts import (
            ContractLevel,
            postcondition,
            set_contract_level,
        )

        set_contract_level(ContractLevel.ENFORCE)

        @postcondition(lambda r: r >= 0, "result must be non-negative")
        def absolute_value(x: float) -> float:
            """Return absolute value."""
            return abs(x)

        assert absolute_value(-3.0) == 3.0

    def test_violated_postcondition_raises(self) -> None:
        """Function returning invalid result raises PostconditionError."""
        from src.shared.python.contracts import (
            ContractLevel,
            PostconditionError,
            postcondition,
            set_contract_level,
        )

        set_contract_level(ContractLevel.ENFORCE)

        @postcondition(lambda r: r >= 0, "must be non-negative")
        def always_negative(x: float) -> float:
            """Always return negative (intentionally bad)."""
            return -abs(x)

        with pytest.raises(PostconditionError):
            always_negative(5.0)


# ---------------------------------------------------------------------------
# contract decorator (combined pre + post)
# ---------------------------------------------------------------------------


class TestContractDecorator:
    """Tests for the @contract decorator (combined pre + post)."""

    def setup_method(self) -> None:
        from src.shared.python.contracts import get_contract_level

        self._original = get_contract_level()

    def teardown_method(self) -> None:
        from src.shared.python.contracts import set_contract_level

        set_contract_level(self._original)

    def test_combined_contract_passes(self) -> None:
        """Function satisfying both pre and post conditions works."""
        from src.shared.python.contracts import (
            ContractLevel,
            contract,
            set_contract_level,
        )

        set_contract_level(ContractLevel.ENFORCE)

        @contract(
            pre=lambda x: x > 0,
            post=lambda r: r > 0,
            pre_msg="x must be positive",
            post_msg="result must be positive",
        )
        def double(x: float) -> float:
            """Double the input."""
            return x * 2.0

        assert double(3.0) == 6.0

    def test_precondition_violation_raises(self) -> None:
        """Violated pre-condition raises PreconditionError."""
        from src.shared.python.contracts import (
            ContractLevel,
            PreconditionError,
            contract,
            set_contract_level,
        )

        set_contract_level(ContractLevel.ENFORCE)

        @contract(
            pre=lambda x: x > 0,
            post=lambda r: r > 0,
            pre_msg="x must be positive",
            post_msg="result must be positive",
        )
        def double(x: float) -> float:
            """Double the input."""
            return x * 2.0

        with pytest.raises(PreconditionError):
            double(-1.0)
