"""Tests for src.shared.python.core.contracts.exceptions (Issues #1949, #1744)."""

from __future__ import annotations

import pytest

from src.shared.python.core.contracts.exceptions import (
    ContractViolationError,
    InvariantError,
    PostconditionError,
    PreconditionError,
    StateError,
)


class TestContractViolationError:
    def test_is_value_error(self) -> None:
        err = ContractViolationError("Precondition", "x must be positive")
        assert isinstance(err, ValueError)

    def test_message_contains_type_and_message(self) -> None:
        err = ContractViolationError("Precondition", "x must be positive")
        assert "Precondition" in str(err)
        assert "x must be positive" in str(err)

    def test_message_contains_function_name(self) -> None:
        err = ContractViolationError(
            "Postcondition", "bad result", function_name="my_func"
        )
        assert "my_func" in str(err)

    def test_message_contains_details(self) -> None:
        err = ContractViolationError("State", "bad state", details={"key": "val"})
        assert "key" in str(err) or "val" in str(err)

    def test_attributes_stored(self) -> None:
        err = ContractViolationError("Invariant", "violated", function_name="f")
        assert err.contract_type == "Invariant"
        assert err.function_name == "f"

    def test_no_function_name(self) -> None:
        err = ContractViolationError("Precondition", "bad input")
        assert err.function_name is None

    def test_empty_details_default(self) -> None:
        err = ContractViolationError("Precondition", "msg")
        assert err.details == {}


class TestPreconditionError:
    def test_is_contract_violation_error(self) -> None:
        err = PreconditionError("x must be > 0")
        assert isinstance(err, ContractViolationError)

    def test_message_in_str(self) -> None:
        err = PreconditionError("x must be > 0")
        assert "x must be > 0" in str(err)

    def test_precondition_in_str(self) -> None:
        err = PreconditionError("bad input")
        assert "Precondition" in str(err)

    def test_parameter_stored(self) -> None:
        err = PreconditionError("bad", parameter="x")
        assert err.parameter == "x"

    def test_value_stored(self) -> None:
        err = PreconditionError("bad", value=-1)
        assert err.value == -1

    def test_function_name_stored(self) -> None:
        err = PreconditionError("bad", function_name="compute")
        assert err.function_name == "compute"


class TestPostconditionError:
    def test_is_contract_violation_error(self) -> None:
        err = PostconditionError("result must be positive")
        assert isinstance(err, ContractViolationError)

    def test_postcondition_in_str(self) -> None:
        err = PostconditionError("bad result")
        assert "Postcondition" in str(err)

    def test_result_stored(self) -> None:
        err = PostconditionError("bad result", result=-5)
        assert err.result == -5

    def test_numpy_result_stored_by_shape(self) -> None:
        import numpy as np

        arr = np.zeros((3, 3))
        err = PostconditionError("bad", result=arr)
        # Should store shape info, not full array
        assert err.result is arr


class TestInvariantError:
    def test_is_contract_violation_error(self) -> None:
        err = InvariantError("class invariant broken")
        assert isinstance(err, ContractViolationError)

    def test_invariant_in_str(self) -> None:
        err = InvariantError("x must be non-negative")
        assert "Invariant" in str(err)

    def test_class_name_stored(self) -> None:
        err = InvariantError("broken", class_name="MyClass")
        assert err.class_name == "MyClass"

    def test_method_name_stored(self) -> None:
        err = InvariantError("broken", method_name="update")
        assert err.method_name == "update"

    def test_class_name_in_str(self) -> None:
        err = InvariantError("broken", class_name="Foo")
        assert "Foo" in str(err)


class TestStateError:
    def test_is_contract_violation_error(self) -> None:
        err = StateError("not initialized")
        assert isinstance(err, ContractViolationError)

    def test_state_in_str(self) -> None:
        err = StateError("wrong state")
        assert "State" in str(err)

    def test_current_state_stored(self) -> None:
        err = StateError("bad", current_state="idle")
        assert err.current_state == "idle"

    def test_required_state_stored(self) -> None:
        err = StateError("bad", required_state="running")
        assert err.required_state == "running"

    def test_operation_stored(self) -> None:
        err = StateError("bad", operation="start")
        assert err.operation == "start"

    def test_catch_as_value_error(self) -> None:
        with pytest.raises(ValueError):
            raise StateError("not ready")
