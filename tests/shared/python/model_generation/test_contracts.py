"""Tests for Design by Contract decorators."""

import numpy as np
import pytest
from model_generation.core.contracts import (
    InvariantError,
    PostconditionError,
    PreconditionError,
    contract,
    invariant,
    postcondition,
    precondition,
    require_finite,
    require_positive,
    require_unit_vector,
    set_contracts_enabled,
)


class TestPrecondition:
    """Tests for @precondition decorator."""

    def test_precondition_passes(self) -> None:
        """Test precondition passes for valid input."""

        @precondition(lambda x: x > 0, "x must be positive")
        def sqrt(x: float) -> float:
            return x**0.5

        result = sqrt(4.0)
        assert result == 2.0

    def test_precondition_fails(self) -> None:
        """Test precondition raises for invalid input."""

        @precondition(lambda x: x > 0, "x must be positive")
        def sqrt(x: float) -> float:
            return x**0.5

        with pytest.raises(PreconditionError) as exc:
            sqrt(-1.0)

        assert "x must be positive" in str(exc.value)

    def test_precondition_multiple_args(self) -> None:
        """Test precondition with multiple arguments."""

        @precondition(lambda a, b: a < b, "a must be less than b")
        def range_check(a: int, b: int) -> bool:
            return True

        assert range_check(1, 5)

        with pytest.raises(PreconditionError):
            range_check(5, 1)


class TestPostcondition:
    """Tests for @postcondition decorator."""

    def test_postcondition_passes(self) -> None:
        """Test postcondition passes for valid output."""

        @postcondition(lambda result: result >= 0, "result must be non-negative")
        def abs_value(x: float) -> float:
            return abs(x)

        assert abs_value(-5) == 5

    def test_postcondition_fails(self) -> None:
        """Test postcondition raises for invalid output."""

        @postcondition(lambda result: result >= 0, "result must be non-negative")
        def bad_function(x: float) -> float:
            return -abs(x)  # Bug: returns negative

        with pytest.raises(PostconditionError) as exc:
            bad_function(5)

        assert "result must be non-negative" in str(exc.value)


class TestContract:
    """Tests for @contract combined decorator."""

    def test_contract_both_pass(self) -> None:
        """Test contract with valid input and output."""

        @contract(
            pre=lambda x: x >= 0,
            post=lambda result: result >= 0,
            pre_msg="input must be non-negative",
            post_msg="output must be non-negative",
        )
        def sqrt(x: float) -> float:
            return x**0.5

        assert sqrt(4) == 2.0

    def test_contract_pre_fails(self) -> None:
        """Test contract with invalid input."""

        @contract(
            pre=lambda x: x >= 0,
            post=lambda result: result >= 0,
        )
        def sqrt(x: float) -> float:
            return x**0.5

        with pytest.raises(PreconditionError):
            sqrt(-1)

    def test_contract_post_fails(self) -> None:
        """Test contract with invalid output."""

        @contract(
            pre=lambda x: True,  # Always pass
            post=lambda result: result > 100,  # Will fail
        )
        def small_number(x: float) -> float:
            return x

        with pytest.raises(PostconditionError):
            small_number(5)


class TestInvariant:
    """Tests for @invariant class decorator."""

    def test_invariant_maintained(self) -> None:
        """Test invariant is maintained after method call."""

        @invariant(lambda self: self.count >= 0, "count must be non-negative")
        class Counter:
            def __init__(self) -> None:
                self.count = 0

            def increment(self) -> None:
                self.count += 1

            def decrement(self) -> None:
                self.count = max(0, self.count - 1)

        counter = Counter()
        counter.increment()
        counter.increment()
        counter.decrement()
        assert counter.count == 1

    def test_invariant_violated(self) -> None:
        """Test invariant violation raises error."""

        @invariant(lambda self: self.value >= 0, "value must be non-negative")
        class Container:
            def __init__(self) -> None:
                self.value = 0

            def set_value(self, v: int) -> None:
                self.value = v

        container = Container()
        with pytest.raises(InvariantError):
            container.set_value(-1)


class TestConvenienceFunctions:
    """Tests for convenience validation functions."""

    def test_require_positive_passes(self) -> None:
        """Test require_positive with valid input."""
        require_positive(1.0, "test")  # Should not raise
        require_positive(0.001, "test")  # Should not raise

    def test_require_positive_fails(self) -> None:
        """Test require_positive with invalid input."""
        with pytest.raises(PreconditionError):
            require_positive(0.0, "test")
        with pytest.raises(PreconditionError):
            require_positive(-1.0, "test")

    def test_require_finite_passes(self) -> None:
        """Test require_finite with valid input."""
        require_finite(np.array([1.0, 2.0, 3.0]), "test")

    def test_require_finite_fails(self) -> None:
        """Test require_finite with NaN/Inf."""
        with pytest.raises(PreconditionError):
            require_finite(np.array([1.0, np.nan, 3.0]), "test")
        with pytest.raises(PreconditionError):
            require_finite(np.array([1.0, np.inf, 3.0]), "test")

    def test_require_unit_vector_passes(self) -> None:
        """Test require_unit_vector with valid input."""
        require_unit_vector(np.array([1.0, 0.0, 0.0]), "test")
        require_unit_vector(np.array([0.0, 1.0, 0.0]), "test")
        require_unit_vector(np.array([1 / np.sqrt(3)] * 3), "test")

    def test_require_unit_vector_fails(self) -> None:
        """Test require_unit_vector with invalid input."""
        with pytest.raises(PreconditionError):
            require_unit_vector(np.array([2.0, 0.0, 0.0]), "test")
        with pytest.raises(PreconditionError):
            require_unit_vector(np.array([0.0, 0.0, 0.0]), "test")


class TestContractsToggle:
    """Tests for global contracts toggle."""

    def test_disable_contracts(self) -> None:
        """Test that disabling contracts skips checks."""

        @precondition(lambda x: x > 0, "x must be positive")
        def sqrt(x: float) -> float:
            return abs(x) ** 0.5

        # Should fail when enabled
        set_contracts_enabled(True)
        with pytest.raises(PreconditionError):
            sqrt(-1)

        # Should pass when disabled
        set_contracts_enabled(False)
        result = sqrt(-1)  # No error
        assert result == 1.0

        # Re-enable for other tests
        set_contracts_enabled(True)
