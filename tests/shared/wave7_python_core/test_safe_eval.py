"""Tests for src/shared/python/safe_eval.py — AST-based safe evaluator.

These tests exercise the public API and verify the security properties of the
evaluator: only whitelisted AST nodes execute, names resolve solely from the
caller-supplied namespace, and no builtins/attribute access leak through.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from safe_eval import (
    NUMPY_MATH_NAMESPACE,
    SCALAR_MATH_NAMESPACE,
    safe_eval,
    safe_eval_math,
    validate_expression,
)


# ── validate_expression ────────────────────────────────────────────────────


class TestValidateExpression:
    def test_simple_expression(self) -> None:
        tree = validate_expression("1 + 2")
        assert tree is not None

    def test_with_allowed_names(self) -> None:
        validate_expression("x + 1", allowed_names={"x"})

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_expression("")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_expression("   \t\n  ")

    def test_rejects_syntax_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid syntax"):
            validate_expression("1 +")

    def test_rejects_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown variable"):
            validate_expression("y + 1", allowed_names={"x"})

    def test_rejects_unknown_function(self) -> None:
        with pytest.raises(ValueError, match="Unknown function"):
            validate_expression("foo(1)", allowed_names={"x"})

    def test_rejects_attribute_call(self) -> None:
        with pytest.raises(ValueError, match="Attribute-based"):
            validate_expression("os.system('ls')", allowed_names={"os"})

    def test_rejects_lambda(self) -> None:
        with pytest.raises(ValueError, match="Unsafe operation"):
            validate_expression("lambda x: x")

    def test_rejects_walrus(self) -> None:
        with pytest.raises(ValueError, match="Unsafe operation"):
            validate_expression("(x := 5)")

    def test_rejects_comprehension(self) -> None:
        with pytest.raises(ValueError, match="Unsafe operation"):
            validate_expression("[i for i in range(3)]", allowed_names={"range", "i"})

    def test_allows_subscript(self) -> None:
        validate_expression("a[0]", allowed_names={"a"})

    def test_allows_slice(self) -> None:
        validate_expression("a[1:3]", allowed_names={"a"})

    def test_allows_starred(self) -> None:
        validate_expression("f(*args)", allowed_names={"f", "args"})

    def test_allows_ifexp(self) -> None:
        validate_expression("1 if x else 2", allowed_names={"x"})

    def test_skip_name_check_when_none(self) -> None:
        # When allowed_names is None, name checks are skipped at validate time.
        validate_expression("anything_goes + here")


# ── safe_eval — arithmetic / logic / literals ───────────────────────────────


class TestSafeEvalArithmetic:
    def test_addition(self) -> None:
        assert safe_eval("1 + 2", {}) == 3

    def test_subtraction(self) -> None:
        assert safe_eval("10 - 4", {}) == 6

    def test_multiplication(self) -> None:
        assert safe_eval("3 * 4", {}) == 12

    def test_division(self) -> None:
        assert safe_eval("10 / 4", {}) == 2.5

    def test_floor_division(self) -> None:
        assert safe_eval("10 // 3", {}) == 3

    def test_modulo(self) -> None:
        assert safe_eval("10 % 3", {}) == 1

    def test_power(self) -> None:
        assert safe_eval("2 ** 8", {}) == 256

    def test_bitwise(self) -> None:
        assert safe_eval("0b1100 & 0b1010", {}) == 0b1000
        assert safe_eval("0b1100 | 0b1010", {}) == 0b1110
        assert safe_eval("0b1100 ^ 0b1010", {}) == 0b0110

    def test_shifts(self) -> None:
        assert safe_eval("1 << 4", {}) == 16
        assert safe_eval("32 >> 2", {}) == 8

    def test_unary_plus_minus(self) -> None:
        assert safe_eval("-5", {}) == -5
        assert safe_eval("+5", {}) == 5
        assert safe_eval("--3", {}) == 3

    def test_not(self) -> None:
        assert safe_eval("not True", {}) is False

    def test_invert(self) -> None:
        assert safe_eval("~5", {}) == -6

    def test_precedence(self) -> None:
        assert safe_eval("2 + 3 * 4", {}) == 14
        assert safe_eval("(2 + 3) * 4", {}) == 20


class TestSafeEvalCompareAndBool:
    def test_eq_neq(self) -> None:
        assert safe_eval("1 == 1", {}) is True
        assert safe_eval("1 != 2", {}) is True

    def test_lt_gt(self) -> None:
        assert safe_eval("1 < 2", {}) is True
        assert safe_eval("2 > 1", {}) is True
        assert safe_eval("1 <= 1", {}) is True
        assert safe_eval("1 >= 1", {}) is True

    def test_chained_compare(self) -> None:
        assert safe_eval("1 < 2 < 3", {}) is True
        assert safe_eval("1 < 3 < 2", {}) is False

    def test_in_notin(self) -> None:
        assert safe_eval("1 in [1, 2, 3]", {}) is True
        assert safe_eval("4 not in [1, 2, 3]", {}) is True

    def test_is_isnot(self) -> None:
        assert safe_eval("None is None", {"None": None}) is True
        assert safe_eval("1 is not None", {"None": None}) is True

    def test_and_or(self) -> None:
        assert safe_eval("True and False", {}) is False
        assert safe_eval("True or False", {}) is True

    def test_short_circuit(self) -> None:
        # If short-circuit broke, 1/0 would raise.
        assert safe_eval("False and 1/0", {}) is False
        assert safe_eval("True or 1/0", {}) is True


class TestSafeEvalLiteralsAndContainers:
    def test_constants(self) -> None:
        assert safe_eval("42", {}) == 42
        assert safe_eval("3.14", {}) == 3.14
        assert safe_eval("'hello'", {}) == "hello"
        assert safe_eval("True", {}) is True

    def test_list(self) -> None:
        assert safe_eval("[1, 2, 3]", {}) == [1, 2, 3]

    def test_tuple(self) -> None:
        assert safe_eval("(1, 2, 3)", {}) == (1, 2, 3)

    def test_subscript(self) -> None:
        ns = {"a": [10, 20, 30]}
        assert safe_eval("a[0]", ns) == 10
        assert safe_eval("a[-1]", ns) == 30

    def test_slice(self) -> None:
        ns = {"a": [1, 2, 3, 4, 5]}
        assert safe_eval("a[1:4]", ns) == [2, 3, 4]
        assert safe_eval("a[::2]", ns) == [1, 3, 5]
        assert safe_eval("a[:3]", ns) == [1, 2, 3]
        assert safe_eval("a[2:]", ns) == [3, 4, 5]

    def test_ifexp(self) -> None:
        assert safe_eval("1 if True else 2", {}) == 1
        assert safe_eval("1 if False else 2", {}) == 2


class TestSafeEvalCalls:
    def test_bare_call(self) -> None:
        assert safe_eval("abs(-5)", {"abs": abs}) == 5

    def test_call_with_kwargs(self) -> None:
        def fn(a: int, b: int = 10) -> int:
            return a + b

        assert safe_eval("fn(1, b=20)", {"fn": fn}) == 21

    def test_starred_args(self) -> None:
        assert safe_eval("max(*xs)", {"max": max, "xs": [3, 1, 4, 1, 5, 9]}) == 9

    def test_call_unknown_function_via_namespace(self) -> None:
        # Validation skips name check when allowed_names=None inside safe_eval
        # uses namespace keys; calling unknown name should raise NameError.
        with pytest.raises(ValueError, match="Unknown"):
            safe_eval("foo(1)", {})


class TestSafeEvalNames:
    def test_name_lookup(self) -> None:
        assert safe_eval("x + 1", {"x": 41}) == 42

    def test_undefined_name_at_eval(self) -> None:
        # Validation catches it first.
        with pytest.raises(ValueError, match="Unknown"):
            safe_eval("y", {"x": 1})

    def test_namespace_not_dict(self) -> None:
        with pytest.raises(TypeError, match="namespace must be a dict"):
            safe_eval("1", object())  # type: ignore[arg-type]

    def test_none_expression(self) -> None:
        with pytest.raises(ValueError, match="expression must be provided"):
            safe_eval(None, {})  # type: ignore[arg-type]

    def test_explicit_allowed_names_override(self) -> None:
        # allowed_names disjoint from namespace blocks usage.
        with pytest.raises(ValueError, match="Unknown"):
            safe_eval("x", {"x": 1}, allowed_names=set())


class TestSafeEvalSecurity:
    def test_import_blocked(self) -> None:
        with pytest.raises(ValueError):
            safe_eval("__import__('os')", {})

    def test_dunder_attribute_blocked(self) -> None:
        with pytest.raises(ValueError, match="Unsafe operation"):
            safe_eval("().__class__.__bases__", {})

    def test_assignment_blocked(self) -> None:
        # parse-mode 'eval' rejects assignment as syntax error
        with pytest.raises(ValueError, match="Invalid syntax"):
            safe_eval("x = 1", {})


# ── safe_eval_math convenience wrapper ─────────────────────────────────────


class TestSafeEvalMath:
    def test_numpy_default(self) -> None:
        # np.sqrt of array works
        result = safe_eval_math("sqrt(x)", {"x": np.array([4.0, 9.0])})
        np.testing.assert_array_equal(result, np.array([2.0, 3.0]))

    def test_scalar_mode(self) -> None:
        result = safe_eval_math("sqrt(16)", use_numpy=False)
        assert result == 4.0

    def test_constants_available(self) -> None:
        assert safe_eval_math("pi", use_numpy=False) == math.pi

    def test_caller_vars_override(self) -> None:
        # User var 'pi' overrides default
        assert safe_eval_math("pi", {"pi": 99}, use_numpy=False) == 99

    def test_trig(self) -> None:
        assert safe_eval_math("sin(0)", use_numpy=False) == 0.0
        assert safe_eval_math("cos(0)", use_numpy=False) == 1.0

    def test_no_variables(self) -> None:
        assert safe_eval_math("2 + 3", None, use_numpy=False) == 5

    def test_none_expression(self) -> None:
        with pytest.raises(ValueError, match="expression must be provided"):
            safe_eval_math(None)  # type: ignore[arg-type]

    def test_numpy_namespace_contents(self) -> None:
        # The known math name set is available.
        for k in ("sin", "cos", "sqrt", "log", "exp", "pi", "e", "np_sqrt"):
            assert k in NUMPY_MATH_NAMESPACE

    def test_scalar_namespace_contents(self) -> None:
        for k in ("sin", "cos", "sqrt", "log", "exp", "pi", "e", "math"):
            assert k in SCALAR_MATH_NAMESPACE
