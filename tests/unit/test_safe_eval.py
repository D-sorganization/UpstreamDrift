"""Tests for src.shared.python.safe_eval (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.safe_eval import (
    NUMPY_MATH_NAMESPACE,
    SCALAR_MATH_NAMESPACE,
    safe_eval,
    safe_eval_math,
    validate_expression,
)

# ---------------------------------------------------------------------------
# validate_expression
# ---------------------------------------------------------------------------


class TestValidateExpression:
    def test_simple_arithmetic(self) -> None:
        tree = validate_expression("1 + 2")
        assert tree is not None

    def test_safe_eval_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_expression("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_expression("   ")

    def test_disallowed_import(self) -> None:
        # __import__ is blocked when an explicit allowed_names set is provided
        with pytest.raises(ValueError, match="Unknown function"):
            validate_expression("__import__('os')", allowed_names={"sin", "cos"})

    def test_disallowed_attribute_access(self) -> None:
        with pytest.raises(ValueError, match="Attribute-based"):
            validate_expression("os.system('ls')")

    def test_disallowed_function_def(self) -> None:
        with pytest.raises(ValueError, match="Unsafe"):
            validate_expression("lambda x: x")

    def test_allowed_names_accepted(self) -> None:
        tree = validate_expression("x + y", allowed_names={"x", "y"})
        assert tree is not None

    def test_disallowed_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown variable"):
            validate_expression("secret_var", allowed_names={"x"})

    def test_safe_eval_syntax_error_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid syntax"):
            validate_expression("1 +* 2")

    def test_call_with_unknown_function_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown function"):
            validate_expression("hack()", allowed_names=set())

    def test_ternary_expression_allowed(self) -> None:
        tree = validate_expression("1 if x > 0 else 0", allowed_names={"x"})
        assert tree is not None

    def test_comparison_allowed(self) -> None:
        tree = validate_expression("a < b", allowed_names={"a", "b"})
        assert tree is not None


# ---------------------------------------------------------------------------
# safe_eval
# ---------------------------------------------------------------------------


class TestSafeEval:
    def test_basic_arithmetic(self) -> None:
        assert safe_eval("2 + 3", {}) == 5

    def test_uses_namespace(self) -> None:
        assert safe_eval("x * 2", {"x": 7}) == 14

    def test_builtins_blocked(self) -> None:
        # __builtins__ is set to {} — open() should not be callable
        with pytest.raises((NameError, ValueError)):
            safe_eval("open('file')", {})

    def test_string_input_not_allowed(self) -> None:
        # 'os' not in namespace — should raise ValueError (unknown var) or NameError
        with pytest.raises((ValueError, NameError)):
            safe_eval("os", {})

    def test_division(self) -> None:
        result = safe_eval("10 / 4", {})
        assert abs(result - 2.5) < 1e-12

    def test_power_operator(self) -> None:
        assert safe_eval("2 ** 8", {}) == 256

    def test_boolean_logic(self) -> None:
        assert safe_eval("True and False", {}) is False

    def test_comparison(self) -> None:
        assert safe_eval("3 > 2", {}) is True

    def test_nested_expression(self) -> None:
        result = safe_eval("(a + b) * c", {"a": 1, "b": 2, "c": 3})
        assert result == 9

    def test_ternary(self) -> None:
        result = safe_eval("1 if flag else 0", {"flag": True})
        assert result == 1

    def test_safe_eval_does_not_call_python_eval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_eval(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("eval must not be called")

        monkeypatch.setattr("builtins.eval", fail_eval)

        assert safe_eval("x + 1", {"x": 2}) == 3

    def test_safe_eval_supports_star_args_without_eval(self) -> None:
        assert safe_eval("sum(*values)", {"sum": sum, "values": [[1, 2, 3]]}) == 6


# ---------------------------------------------------------------------------
# safe_eval_math
# ---------------------------------------------------------------------------


class TestSafeEvalMath:
    def test_numpy_sqrt(self) -> None:
        result = safe_eval_math("sqrt(4)", use_numpy=True)
        assert abs(result - 2.0) < 1e-12

    def test_scalar_sqrt(self) -> None:
        result = safe_eval_math("sqrt(9)", use_numpy=False)
        assert abs(result - 3.0) < 1e-12

    def test_pi_constant(self) -> None:
        import math

        result = safe_eval_math("pi")
        assert abs(result - math.pi) < 1e-12

    def test_with_variables(self) -> None:
        result = safe_eval_math("x * sin(0)", variables={"x": 99})
        assert abs(result - 0.0) < 1e-9

    def test_trig_functions(self) -> None:
        result = safe_eval_math("cos(0)", use_numpy=False)
        assert abs(result - 1.0) < 1e-12

    def test_exp_function(self) -> None:
        import math

        result = safe_eval_math("exp(1)", use_numpy=False)
        assert abs(result - math.e) < 1e-10

    def test_empty_expression_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            safe_eval_math("")

    def test_unsafe_expression_raises(self) -> None:
        with pytest.raises(ValueError):
            safe_eval_math("__import__('os')")


# ---------------------------------------------------------------------------
# Namespace integrity
# ---------------------------------------------------------------------------


class TestNamespaces:
    def test_numpy_namespace_has_trig(self) -> None:
        for fn in ("sin", "cos", "tan", "sqrt", "log", "exp"):
            assert fn in NUMPY_MATH_NAMESPACE

    def test_scalar_namespace_has_trig(self) -> None:
        for fn in ("sin", "cos", "tan", "sqrt", "log", "exp"):
            assert fn in SCALAR_MATH_NAMESPACE

    def test_numpy_namespace_has_constants(self) -> None:
        assert "pi" in NUMPY_MATH_NAMESPACE
        assert "e" in NUMPY_MATH_NAMESPACE

    def test_scalar_namespace_has_constants(self) -> None:
        assert "pi" in SCALAR_MATH_NAMESPACE
        assert "e" in SCALAR_MATH_NAMESPACE
