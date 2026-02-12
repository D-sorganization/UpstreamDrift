"""Unit tests for Design by Contract infrastructure.

Tests the contracts module including:
- Precondition decorator
- Postcondition decorator
- StateError exceptions
- ContractChecker mixin
- Invariant verification
- Function-call style require/ensure
- Tri-state DBC_LEVEL enforcement
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

from src.shared.python.core.contracts import (
    ContractChecker,
    ContractLevel,
    ContractViolationError,
    InvariantError,
    PostconditionError,
    PreconditionError,
    StateError,
    check_finite,
    check_positive,
    check_positive_definite,
    check_symmetric,
    disable_contracts,
    enable_contracts,
    ensure,
    finite_result,
    get_contract_level,
    invariant_checked,
    postcondition,
    precondition,
    require,
    require_state,
    set_contract_level,
)


@pytest.fixture(autouse=True)
def _enforce_contracts() -> None:
    """Ensure contracts are in ENFORCE mode for all tests."""
    original = get_contract_level()
    set_contract_level(ContractLevel.ENFORCE)
    yield  # type: ignore[misc]
    set_contract_level(original)


class TestPreconditionDecorator:
    """Tests for the @precondition decorator."""

    def test_precondition_passes_when_satisfied(self):
        """Precondition should allow execution when condition is True."""

        @precondition(lambda x: x > 0, "x must be positive")
        def sqrt(x: float) -> float:
            return x**0.5

        result = sqrt(4.0)
        assert result == 2.0

    def test_precondition_raises_when_violated(self):
        """Precondition should raise PreconditionError when condition is False."""

        @precondition(lambda x: x > 0, "x must be positive")
        def sqrt(x: float) -> float:
            return x**0.5

        with pytest.raises(PreconditionError) as exc_info:
            sqrt(-1.0)

        assert "x must be positive" in str(exc_info.value)
        assert "Precondition" in str(exc_info.value)

    def test_precondition_with_method(self):
        """Precondition should work with class methods."""

        class Calculator:
            def __init__(self, value: float = 0):
                self.value = value
                self._is_ready = True

            @precondition(lambda self, x: self._is_ready, "Calculator must be ready")
            def compute(self, x: float) -> float:
                return self.value + x

        calc = Calculator(10)
        assert calc.compute(5) == 15

        calc._is_ready = False
        with pytest.raises(PreconditionError):
            calc.compute(5)

    def test_precondition_multiple_conditions(self):
        """Multiple preconditions should all be checked."""

        @precondition(lambda x, y: x > 0, "x must be positive")
        @precondition(lambda x, y: y > 0, "y must be positive")
        def divide(x: float, y: float) -> float:
            return x / y

        assert divide(10, 2) == 5.0

        with pytest.raises(PreconditionError) as exc_info:
            divide(-1, 2)
        assert "x must be positive" in str(exc_info.value)

        with pytest.raises(PreconditionError) as exc_info:
            divide(1, -2)
        assert "y must be positive" in str(exc_info.value)


class TestPostconditionDecorator:
    """Tests for the @postcondition decorator."""

    def test_postcondition_passes_when_satisfied(self):
        """Postcondition should allow return when condition is True."""

        @postcondition(lambda result: result >= 0, "result must be non-negative")
        def absolute(x: float) -> float:
            return abs(x)

        assert absolute(-5) == 5
        assert absolute(5) == 5

    def test_postcondition_raises_when_violated(self):
        """Postcondition should raise PostconditionError when condition is False."""

        @postcondition(lambda result: result > 0, "result must be positive")
        def bad_function() -> int:
            return -1

        with pytest.raises(PostconditionError) as exc_info:
            bad_function()

        assert "result must be positive" in str(exc_info.value)
        assert "Postcondition" in str(exc_info.value)

    def test_postcondition_with_numpy_array(self):
        """Postcondition should work with numpy array results."""

        @postcondition(
            lambda arr: np.all(arr >= 0), "all elements must be non-negative"
        )
        def compute_squares(values: np.ndarray) -> np.ndarray:
            return values**2

        result = compute_squares(np.array([-2, -1, 0, 1, 2]))
        np.testing.assert_array_equal(result, np.array([4, 1, 0, 1, 4]))


class TestRequireStateDecorator:
    """Tests for the @require_state decorator."""

    def test_require_state_passes_when_satisfied(self):
        """Should allow execution when state requirement is met."""

        class Engine:
            def __init__(self):
                self._is_initialized = False

            def initialize(self):
                self._is_initialized = True

            @require_state(lambda self: self._is_initialized, "initialized")
            def step(self):
                return "stepped"

        engine = Engine()
        engine.initialize()
        assert engine.step() == "stepped"

    def test_require_state_raises_when_not_met(self):
        """Should raise StateError when state requirement is not met."""

        class Engine:
            def __init__(self):
                self._is_initialized = False

            @require_state(lambda self: self._is_initialized, "initialized")
            def step(self):
                return "stepped"

        engine = Engine()
        with pytest.raises(StateError) as exc_info:
            engine.step()

        assert "not initialized" in str(exc_info.value)
        assert "State" in str(exc_info.value)


class TestContractChecker:
    """Tests for the ContractChecker mixin class."""

    def test_verify_invariants_passes_when_all_hold(self):
        """verify_invariants should return True when all invariants hold."""

        class TestClass(ContractChecker):
            def __init__(self):
                self.value = 10

            def _get_invariants(self):
                return [
                    (lambda: self.value > 0, "value must be positive"),
                    (lambda: self.value < 100, "value must be less than 100"),
                ]

        obj = TestClass()
        assert obj.verify_invariants() is True

    def test_verify_invariants_raises_when_violated(self):
        """verify_invariants should raise InvariantError when any invariant fails."""

        class TestClass(ContractChecker):
            def __init__(self):
                self.value = 10

            def _get_invariants(self):
                return [
                    (lambda: self.value > 0, "value must be positive"),
                ]

        obj = TestClass()
        obj.value = -5

        with pytest.raises(InvariantError) as exc_info:
            obj.verify_invariants()

        assert "value must be positive" in str(exc_info.value)

    def test_invariant_checked_decorator(self):
        """@invariant_checked should verify invariants after method execution."""

        class TestClass(ContractChecker):
            def __init__(self):
                self.value = 10

            def _get_invariants(self):
                return [
                    (lambda: self.value > 0, "value must be positive"),
                ]

            @invariant_checked
            def set_value(self, new_value: int):
                self.value = new_value

        obj = TestClass()
        obj.set_value(20)  # Should pass
        assert obj.value == 20

        with pytest.raises(InvariantError):
            obj.set_value(-5)  # Should fail invariant check


class TestStateError:
    """Tests for StateError exception."""

    def test_state_error_message(self):
        """StateError should include state information in message."""
        error = StateError(
            "Cannot step",
            current_state="uninitialized",
            required_state="initialized",
            operation="step",
        )

        assert "Cannot step" in str(error)
        assert "State" in str(error)

    def test_state_error_attributes(self):
        """StateError should store state information."""
        error = StateError(
            "Cannot step",
            current_state="uninitialized",
            required_state="initialized",
            operation="step",
        )

        assert error.current_state == "uninitialized"
        assert error.required_state == "initialized"
        assert error.operation == "step"


class TestContractHelpers:
    """Tests for contract helper functions."""

    def test_check_finite(self):
        """check_finite should detect NaN and Inf values."""
        assert check_finite(np.array([1, 2, 3])) is True
        assert check_finite(np.array([1, np.nan, 3])) is False
        assert check_finite(np.array([1, np.inf, 3])) is False
        assert check_finite(np.array([1, -np.inf, 3])) is False
        assert check_finite(None) is False

    def test_check_positive(self):
        """check_positive should verify all values are positive."""
        assert check_positive(5) is True
        assert check_positive(-5) is False
        assert check_positive(0) is False
        assert check_positive(np.array([1, 2, 3])) is True
        assert check_positive(np.array([1, -2, 3])) is False

    def test_check_symmetric(self):
        """check_symmetric should verify matrix symmetry."""
        symmetric = np.array([[1, 2], [2, 1]])
        asymmetric = np.array([[1, 2], [3, 1]])

        assert check_symmetric(symmetric) is True
        assert check_symmetric(asymmetric) is False

    def test_check_positive_definite(self):
        """check_positive_definite should verify positive definiteness."""
        pd_matrix = np.array([[2, 1], [1, 2]])  # Eigenvalues: 1, 3
        npd_matrix = np.array([[1, 2], [2, 1]])  # Eigenvalues: -1, 3

        assert check_positive_definite(pd_matrix) is True
        assert check_positive_definite(npd_matrix) is False


class TestFiniteResultDecorator:
    """Tests for the @finite_result decorator."""

    def test_finite_result_passes_for_finite_values(self):
        """Should pass when all values are finite."""

        @finite_result
        def compute() -> np.ndarray:
            return np.array([1.0, 2.0, 3.0])

        result = compute()
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_finite_result_raises_for_nan(self):
        """Should raise when result contains NaN."""

        @finite_result
        def compute_bad() -> np.ndarray:
            return np.array([1.0, np.nan, 3.0])

        with pytest.raises(PostconditionError):
            compute_bad()

    def test_finite_result_allows_none(self):
        """Should allow None return values."""

        @finite_result
        def compute_none() -> np.ndarray | None:
            return None

        result = compute_none()
        assert result is None


class TestContractEnableDisable:
    """Tests for enabling/disabling contracts."""

    def test_contracts_can_be_disabled(self):
        """Contracts should not be enforced when disabled."""
        import src.shared.python.core.contracts as contracts_module

        original_state = contracts_module.CONTRACTS_ENABLED

        try:
            disable_contracts()

            @precondition(lambda x: x > 0, "x must be positive")
            def test_func(x: float) -> float:
                return x

        finally:
            if original_state:
                enable_contracts()
            else:
                disable_contracts()


class TestContractViolationErrorHierarchy:
    """Tests for the contract exception hierarchy."""

    def test_precondition_error_is_contract_violation(self):
        """PreconditionError should be a ContractViolationError."""
        error = PreconditionError("test")
        assert isinstance(error, ContractViolationError)

    def test_postcondition_error_is_contract_violation(self):
        """PostconditionError should be a ContractViolationError."""
        error = PostconditionError("test")
        assert isinstance(error, ContractViolationError)

    def test_invariant_error_is_contract_violation(self):
        """InvariantError should be a ContractViolationError."""
        error = InvariantError("test")
        assert isinstance(error, ContractViolationError)

    def test_state_error_is_contract_violation(self):
        """StateError should be a ContractViolationError."""
        error = StateError("test")
        assert isinstance(error, ContractViolationError)

    def test_can_catch_all_contract_violations(self):
        """Should be able to catch all contract errors with ContractViolationError."""
        errors = [
            PreconditionError("test"),
            PostconditionError("test"),
            InvariantError("test"),
            StateError("test"),
        ]

        for error in errors:
            try:
                raise error
            except ContractViolationError:
                pass  # Expected
            except Exception:
                pytest.fail(
                    f"{type(error).__name__} was not caught as ContractViolationError"
                )


# ─── New: Function-call style contracts ───────────────────────


class TestRequirePrimitive:
    """Tests for the require() function-call style contract."""

    def test_passes_on_true(self) -> None:
        require(True, "should not fail")

    def test_raises_on_false(self) -> None:
        with pytest.raises(PreconditionError, match="Precondition"):
            require(False, "value must be positive", -1)

    def test_skipped_when_off(self) -> None:
        set_contract_level(ContractLevel.OFF)
        require(False, "should not raise")

    def test_warns_when_warn(self) -> None:
        set_contract_level(ContractLevel.WARN)
        require(False, "warn only")  # Should not raise


class TestEnsurePrimitive:
    """Tests for the ensure() function-call style contract."""

    def test_passes_on_true(self) -> None:
        ensure(True, "ok")

    def test_raises_on_false(self) -> None:
        with pytest.raises(PostconditionError, match="Postcondition"):
            ensure(False, "result must be finite", float("nan"))


# ─── New: Tri-state enforcement level ─────────────────────────


class TestContractLevelControls:
    """Tests for the tri-state enforcement level system."""

    def test_set_and_get(self) -> None:
        set_contract_level(ContractLevel.WARN)
        assert get_contract_level() == ContractLevel.WARN

    def test_off_skips_all_checks(self) -> None:
        set_contract_level(ContractLevel.OFF)
        require(False, "should not raise")
        ensure(False, "should not raise")

    def test_warn_does_not_raise(self) -> None:
        set_contract_level(ContractLevel.WARN)
        require(False, "warn only")
        ensure(False, "warn only")

    def test_enforce_raises(self) -> None:
        set_contract_level(ContractLevel.ENFORCE)
        with pytest.raises(PreconditionError):
            require(False, "should raise")

    def test_env_var_controls_level(self) -> None:
        with patch.dict(os.environ, {"DBC_LEVEL": "warn"}):
            from src.shared.python.core.contracts import (
                _resolve_contract_level,
            )

            level = _resolve_contract_level()
            assert level == ContractLevel.WARN
