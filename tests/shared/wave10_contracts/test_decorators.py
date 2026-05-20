"""Exhaustive tests for precondition/postcondition/contract/class_invariant decorators."""

from __future__ import annotations

import pytest

from src.shared.python._contracts_decorators import (
    _check_class_invariant,
    _evaluate_precondition,
    _wrap_method_with_invariant,
    class_invariant,
    contract,
    postcondition,
    precondition,
)
from src.shared.python._contracts_exceptions import (
    ContractEvaluationError,
    InvariantError,
    PostconditionError,
    PreconditionError,
)
from src.shared.python._contracts_level import ContractLevel, set_contract_level


class TestEvaluatePrecondition:
    def test_name_based_subset(self):
        def func(a, b, c):  # noqa: ARG001
            return None

        cond = lambda b: b > 0  # noqa: E731
        assert _evaluate_precondition(cond, func, (1, 2, 3), {}) is True
        assert _evaluate_precondition(cond, func, (1, -1, 3), {}) is False

    def test_full_positional(self):
        def func(x, y):  # noqa: ARG001
            return None

        cond = lambda x, y: x + y > 0  # noqa: E731
        assert _evaluate_precondition(cond, func, (1, 2), {}) is True

    def test_keyword_args_bound(self):
        def func(x, y=10):  # noqa: ARG001
            return None

        cond = lambda y: y == 10  # noqa: E731
        assert _evaluate_precondition(cond, func, (1,), {}) is True

    def test_condition_no_params_falls_through_positional(self):
        # condition with no params -> name-based skipped; falls back to positional
        def func():
            return None

        cond = lambda: True  # noqa: E731
        assert _evaluate_precondition(cond, func, (), {}) is True

    def test_signature_bind_failure_raises_evaluation_error(self):
        def func(a):  # noqa: ARG001
            return None

        cond = lambda a: a > 0  # noqa: E731
        # Missing required positional arg `a` triggers TypeError in bind
        with pytest.raises(ContractEvaluationError):
            _evaluate_precondition(cond, func, (), {})

    def test_positional_fallback_typeerror_wrapped(self):
        # A builtin lacks inspectable signature, force positional path; if it
        # raises TypeError it must be wrapped.
        cond = lambda x: x  # noqa: E731

        # Patch signature inspection by passing a builtin in lieu of func.
        class _Builtin:
            __qualname__ = "X"

        # Build a func with no params so name-based call has no overlap, then
        # positional fallback receives an extra arg.
        def func():
            return None

        with pytest.raises(ContractEvaluationError):
            _evaluate_precondition(cond, func, (1,), {})


class TestPreconditionDecorator:
    def test_passes(self):
        @precondition(lambda x: x > 0)
        def f(x):
            return x * 2

        assert f(5) == 10

    def test_fails(self):
        @precondition(lambda x: x > 0, "must be positive")
        def f(x):
            return x

        with pytest.raises(PreconditionError) as exc:
            f(-1)
        assert "must be positive" in str(exc.value)

    def test_off_returns_original(self):
        set_contract_level(ContractLevel.OFF)

        @precondition(lambda x: x > 0)
        def f(x):
            return x

        # In OFF mode, even invalid args pass through
        assert f(-1) == -1

    def test_default_message(self):
        @precondition(lambda x: False)
        def f(x):
            return x

        with pytest.raises(PreconditionError) as exc:
            f(1)
        assert "Precondition failed" in str(exc.value)


class TestPostconditionDecorator:
    def test_passes(self):
        @postcondition(lambda r: r > 0)
        def f(x):
            return x + 1

        assert f(2) == 3

    def test_fails(self):
        @postcondition(lambda r: r > 0, "must be positive result")
        def f(x):
            return x - 1

        with pytest.raises(PostconditionError):
            f(0)

    def test_off_returns_original(self):
        set_contract_level(ContractLevel.OFF)

        @postcondition(lambda r: r > 0)
        def f():
            return -1

        assert f() == -1

    def test_evaluation_error_wraps_typeerror(self):
        @postcondition(lambda r: r.no_such_method())  # AttributeError
        def f():
            return 1

        with pytest.raises(ContractEvaluationError):
            f()

    def test_evaluation_error_wraps_zerodivision(self):
        @postcondition(lambda r: (1 / 0) > 0)
        def f():
            return 1

        with pytest.raises(ContractEvaluationError):
            f()


class TestContractDecorator:
    def test_pre_and_post(self):
        @contract(pre=lambda x: x >= 0, post=lambda r: r >= 0)
        def sqrt_like(x):
            return x**0.5

        assert sqrt_like(4) == 2

    def test_pre_violation(self):
        @contract(pre=lambda x: x >= 0, pre_msg="non-neg")
        def f(x):
            return x

        with pytest.raises(PreconditionError):
            f(-1)

    def test_post_violation(self):
        @contract(post=lambda r: r >= 0, post_msg="non-neg-out")
        def f():
            return -1

        with pytest.raises(PostconditionError):
            f()

    def test_no_conditions_runs_unchanged(self):
        @contract()
        def f(x):
            return x

        assert f(7) == 7

    def test_only_post(self):
        @contract(post=lambda r: r == 1)
        def f():
            return 1

        assert f() == 1


class TestCheckClassInvariant:
    def test_passes(self):
        _check_class_invariant(object(), lambda s: True, "m", "ctx")

    def test_fails_raises(self):
        with pytest.raises(InvariantError) as exc:
            _check_class_invariant(object(), lambda s: False, "broken", "in test")
        assert "broken" in str(exc.value)
        assert "in test" in str(exc.value)

    def test_condition_raises_wraps(self):
        def bad(_self):
            raise ValueError("boom")

        with pytest.raises(InvariantError) as exc:
            _check_class_invariant(object(), bad, "checking", "ctx")
        assert "boom" in str(exc.value)

    def test_propagates_invariant_error_unchanged(self):
        def cond(_self):
            raise InvariantError("inner")

        with pytest.raises(InvariantError) as exc:
            _check_class_invariant(object(), cond, "x", "y")
        assert "inner" in str(exc.value)


class TestWrapMethodWithInvariant:
    def test_returns_value_and_checks(self):
        class C:
            count = 0

        c = C()

        def method(self, n):
            self.count = n
            return n * 2

        wrapped = _wrap_method_with_invariant(
            method, "method", lambda s: s.count >= 0, "non-neg count"
        )
        assert wrapped(c, 3) == 6

    def test_invariant_failure_after_method(self):
        class C:
            count = 0

        c = C()

        def method(self):
            self.count = -1

        wrapped = _wrap_method_with_invariant(
            method, "method", lambda s: s.count >= 0, "non-neg count"
        )
        with pytest.raises(InvariantError):
            wrapped(c)


class TestClassInvariantDecorator:
    def test_init_checked(self):
        @class_invariant(lambda self: self.x >= 0, "x must be non-negative")
        class C:
            def __init__(self, x):
                self.x = x

        c = C(1)
        assert c.x == 1
        with pytest.raises(InvariantError):
            C(-1)

    def test_method_checked(self):
        @class_invariant(lambda self: self.x >= 0)
        class C:
            def __init__(self):
                self.x = 0

            def decrement(self):
                self.x -= 1

        c = C()
        with pytest.raises(InvariantError):
            c.decrement()

    def test_private_methods_not_wrapped(self):
        sentinel = []

        @class_invariant(lambda self: True)
        class C:
            def __init__(self):
                pass

            def _private(self):
                sentinel.append("called")

        C()._private()
        assert sentinel == ["called"]

    def test_off_returns_original(self):
        set_contract_level(ContractLevel.OFF)

        @class_invariant(lambda self: False)
        class C:
            def __init__(self):
                pass

        # Would raise if wrapped; OFF means decorator passthrough
        C()
