import logging
from typing import Any

import numpy as np
import pytest

from src.shared.python.contracts import (
    ContractChecker,
    ContractLevel,
    ContractViolationError,
    InvariantError,
    PostconditionError,
    PreconditionError,
    _ContractState,
    _handle_violation,
    _resolve_contract_level,
    check_non_negative,
    check_positive,
    check_pressure,
    check_range,
    check_temperature,
    class_invariant,
    contract,
    ensure,
    ensure_valid_result,
    get_contract_level,
    has_finite_elements,
    invariant,
    invariant_checked,
    is_non_negative,
    is_positive,
    is_valid_result,
    postcondition,
    precondition,
    require,
    require_finite,
    require_positive,
    require_unit_vector,
    set_contract_level,
    set_contracts_enabled,
)


@pytest.fixture(autouse=True)
def reset_contract_level() -> Any:
    """Reset the contract level to ENFORCE before and after each test."""
    original_level = get_contract_level()
    set_contract_level(ContractLevel.ENFORCE)
    yield
    set_contract_level(original_level)


class TestContractLevels:
    def test_resolve_contract_level(self, monkeypatch) -> Any:
        monkeypatch.setenv("DBC_LEVEL", "warn")
        assert _resolve_contract_level() == ContractLevel.WARN

        monkeypatch.setenv("DBC_LEVEL", "off")
        assert _resolve_contract_level() == ContractLevel.OFF

        monkeypatch.setenv("DBC_LEVEL", "enforce")
        assert _resolve_contract_level() == ContractLevel.ENFORCE

        monkeypatch.setenv("DBC_LEVEL", "UNKNOWN")
        # should fallback to ENFORCE since __debug__ is true in testing
        assert _resolve_contract_level() == ContractLevel.ENFORCE

    def test_get_set_contract_level(self) -> Any:
        set_contract_level(ContractLevel.WARN)
        assert get_contract_level() == ContractLevel.WARN
        assert _ContractState.level == ContractLevel.WARN

        # Manually evaluate the property to hit line 73
        assert _ContractState.__dict__["enabled"].__func__.fget(_ContractState) is True

        set_contract_level(ContractLevel.OFF)
        assert get_contract_level() == ContractLevel.OFF
        assert _ContractState.__dict__["enabled"].__func__.fget(_ContractState) is False

        set_contracts_enabled(True)
        assert get_contract_level() == ContractLevel.ENFORCE

        set_contracts_enabled(False)
        assert get_contract_level() == ContractLevel.OFF


class TestContractExceptions:
    def test_contract_violation_error(self) -> Any:
        err = ContractViolationError("pre-condition", "test msg", 42)
        assert err.condition_type == "pre-condition"
        assert err.message == "test msg"
        assert err.value == 42
        assert str(err) == "[DbC pre-condition] test msg (got: 42)"

        # without value
        err2 = ContractViolationError("post-condition", "test msg 2")
        assert err2.value is None
        assert str(err2) == "[DbC post-condition] test msg 2"

    def test_precondition_error(self) -> Any:
        err = PreconditionError("msg", 1)
        assert err.condition_type == "pre-condition"

    def test_postcondition_error(self) -> Any:
        err = PostconditionError("msg")
        assert err.condition_type == "post-condition"

    def test_invariant_error(self) -> Any:
        err = InvariantError("msg")
        assert err.condition_type == "invariant"


class TestHandleViolation:
    def test_handle_violation_enforce(self) -> Any:
        set_contract_level(ContractLevel.ENFORCE)
        with pytest.raises(PreconditionError, match="test"):
            _handle_violation("pre-condition", "test", 1)

    def test_handle_violation_warn(self, caplog) -> Any:
        set_contract_level(ContractLevel.WARN)
        with caplog.at_level(logging.WARNING):
            _handle_violation("pre-condition", "test warning msg", 1)
        assert "test warning msg (got: 1)" in caplog.text


class TestPrimitives:
    def test_require(self) -> Any:
        require(True, "msg")
        with pytest.raises(PreconditionError):
            require(False, "msg")

        set_contract_level(ContractLevel.OFF)
        require(False, "msg")  # Shouldn't raise

    def test_ensure(self) -> Any:
        set_contract_level(ContractLevel.ENFORCE)
        ensure(True, "msg")
        with pytest.raises(PostconditionError):
            ensure(False, "msg")

        set_contract_level(ContractLevel.OFF)
        ensure(False, "msg")

    def test_invariant(self) -> Any:
        set_contract_level(ContractLevel.ENFORCE)
        invariant(True, "msg")
        with pytest.raises(InvariantError):
            invariant(False, "msg")

        set_contract_level(ContractLevel.OFF)
        invariant(False, "msg")


class TestDecorators:
    def test_precondition_decorator_positional(self) -> Any:
        @precondition(lambda x: x > 0, "x positive")
        def func(x) -> Any:
            return x

        assert func(1) == 1
        with pytest.raises(PreconditionError):
            func(0)

    def test_precondition_decorator_kwargs(self) -> Any:
        @precondition(lambda x: x > 0, "x positive")
        def func(y, x=0) -> Any:
            return y + x

        assert func(y=1, x=2) == 3
        with pytest.raises(PreconditionError):
            func(y=1, x=-1)

    def test_precondition_decorator_type_error(self) -> Any:
        # Trigger TypeError in precondition evaluation (missing positional arg)
        @precondition(lambda nonexistent: True, "msg")
        def func(x) -> Any:
            return x

        with pytest.raises(PreconditionError, match="Failed to evaluate precondition"):
            func(x=1)

    def test_precondition_decorator_type_error_warn(self) -> Any:
        set_contract_level(ContractLevel.WARN)

        # Should gracefully return function result since violations are just logged
        @precondition(lambda nonexistent: True, "msg")
        def func(x) -> Any:
            return x

        assert func(x=42) == 42

    def test_precondition_decorator_off(self) -> Any:
        set_contract_level(ContractLevel.OFF)

        @precondition(lambda x: x > 0, "x positive")
        def func(x) -> Any:
            return x

        assert func(0) == 0

    def test_postcondition_decorator(self) -> Any:
        @postcondition(lambda r: r > 0, "r positive")
        def func(x) -> Any:
            return x

        assert func(1) == 1
        with pytest.raises(PostconditionError):
            func(-1)

        # when condition evaluation fails with TypeError (requires 2 args, gets 1)
        # The two-arg lambda is intentionally malformed to exercise the
        # evaluation-error path, hence the type: ignore.
        @postcondition(lambda r, extras: True)  # type: ignore[arg-type, misc]
        def func2(x) -> Any:
            return x

        with pytest.raises(
            PostconditionError, match="Failed to evaluate postcondition"
        ):
            func2(1)

    def test_postcondition_decorator_type_error_warn(self) -> Any:
        set_contract_level(ContractLevel.WARN)

        # Should gracefully return function result. The two-arg lambda is
        # intentionally malformed to exercise the evaluation-error path.
        @postcondition(lambda r, extras: True)  # type: ignore[arg-type, misc]
        def func(x) -> Any:
            return x

        assert func(42) == 42

    def test_postcondition_decorator_off(self) -> Any:
        set_contract_level(ContractLevel.OFF)

        def func(x) -> Any:
            return x

        # Ensure we hit line 285 by calling decorator with DBC_LEVEL=OFF
        decorated = postcondition(lambda r: r > 0)(func)
        assert decorated is func  # The exact same function is returned

        assert decorated(-1) == -1

    def test_contract_decorator_combined(self) -> Any:
        @contract(pre=lambda x: x > 0, post=lambda r: r < 10)
        def func(x) -> Any:
            return x * 2

        assert func(2) == 4
        with pytest.raises(PreconditionError):
            func(-1)
        with pytest.raises(PostconditionError):
            func(6)


class TestClassInvariants:
    def test_class_invariant_decorator(self) -> Any:
        @class_invariant(lambda self: self.val > 0, "val positive")
        class MyClass:
            def __init__(self, val):
                self.val = val

            def set_val(self, val) -> Any:
                self.val = val
                return "ok"

        # init check
        obj = MyClass(1)
        with pytest.raises(InvariantError, match="val positive \\(after __init__\\)"):
            MyClass(-1)

        # method check (success hits return result statement)
        assert obj.set_val(2) == "ok"

        # method check (failure)
        with pytest.raises(InvariantError, match="val positive \\(after set_val\\)"):
            obj.set_val(-1)

    def test_class_invariant_evaluation_error(self) -> Any:
        @class_invariant(lambda self: self.val.foo > 0, "foo positive")
        class MyClass:
            def __init__(self):
                self.val = None

        with pytest.raises(InvariantError, match="Error checking invariant"):
            MyClass()

    def test_class_invariant_off(self) -> Any:
        set_contract_level(ContractLevel.OFF)

        @class_invariant(lambda self: self.val > 0)
        class MyClass:
            def __init__(self, val):
                self.val = val

        obj = MyClass(-1)
        assert obj.val == -1

    def test_contract_checker_mixin(self) -> Any:
        # Base method check
        assert ContractChecker()._get_invariants() == []

        class MyChecker(ContractChecker):
            def __init__(self, val):
                self.val = val

            def _get_invariants(self) -> Any:
                return [(lambda: self.val > 0, "val positive")]

            @invariant_checked
            def set_val(self, val) -> Any:
                self.val = val

        obj = MyChecker(1)
        # Not checked automatically on __init__ with mixin unless decorated!

        obj.set_val(2)
        with pytest.raises(InvariantError):
            obj.set_val(-1)

    def test_contract_checker_mixin_warn(self, caplog) -> Any:
        set_contract_level(ContractLevel.WARN)

        class MyChecker(ContractChecker):
            def _get_invariants(self) -> Any:
                return [(lambda: False, "val positive")]

        obj = MyChecker()
        with caplog.at_level(logging.WARNING):
            obj.verify_invariants()
        assert "val positive" in caplog.text

    def test_contract_checker_mixin_evaluation_error(self) -> Any:
        class MyChecker(ContractChecker):
            def _get_invariants(self) -> Any:
                def bad_cond() -> Any:
                    raise ValueError("eval error")

                return [(bad_cond, "val positive")]

        obj = MyChecker()
        with pytest.raises(InvariantError, match="Failed to evaluate invariant"):
            obj.verify_invariants()

    def test_contract_checker_mixin_off(self) -> Any:
        set_contract_level(ContractLevel.OFF)

        class MyChecker(ContractChecker):
            def _get_invariants(self) -> Any:
                return [(lambda: False, "never holds")]

            @invariant_checked
            def nop(self) -> Any:
                pass

        obj = MyChecker()
        assert obj.verify_invariants() is True
        obj.nop()


class TestDomainHelpers:
    def test_check_positive(self) -> Any:
        check_positive(1)
        with pytest.raises(PreconditionError):
            check_positive(0)

    def test_check_non_negative(self) -> Any:
        check_non_negative(0)
        with pytest.raises(PreconditionError):
            check_non_negative(-1)

    def test_check_range(self) -> Any:
        check_range(5, 0, 10)
        with pytest.raises(PreconditionError):
            check_range(11, 0, 10)

    def test_check_temperature(self) -> Any:
        check_temperature(1)
        with pytest.raises(PreconditionError):
            check_temperature(0)

    def test_check_pressure(self) -> Any:
        check_pressure(1)
        with pytest.raises(PreconditionError):
            check_pressure(0)

    def test_require_positive(self) -> Any:
        require_positive(1)
        with pytest.raises(PreconditionError):
            require_positive(0)

        set_contracts_enabled(False)
        require_positive(0)

    def test_require_finite(self) -> Any:
        require_finite([1, 2, 3])
        with pytest.raises(PreconditionError):
            require_finite([1, np.nan, 3])

        set_contracts_enabled(False)
        require_finite([1, np.nan, 3])

    def test_require_unit_vector(self) -> Any:
        require_unit_vector([1, 0, 0])
        with pytest.raises(PreconditionError):
            require_unit_vector([1, 1, 0])

        set_contracts_enabled(False)
        require_unit_vector([1, 1, 0])

    def test_ensure_valid_result(self) -> Any:
        class MockResult:
            def __init__(self, is_valid):
                self.is_valid = is_valid

            def get_error_messages(self) -> Any:
                return ["error1", "error2"]

        ensure_valid_result(MockResult(True))
        with pytest.raises(
            PostconditionError, match="Validation failed: error1; error2"
        ):
            ensure_valid_result(MockResult(False))

        set_contracts_enabled(False)
        ensure_valid_result(MockResult(False))

    def test_predicates(self) -> Any:
        assert is_positive(1) is True
        assert is_positive(0) is False

        assert is_non_negative(0) is True
        assert is_non_negative(-1) is False

        class MockResult:
            is_valid = True

        assert is_valid_result(MockResult()) is True

        assert has_finite_elements([1, 2, 3]) is True
        assert has_finite_elements([1, np.inf, 3]) is False
