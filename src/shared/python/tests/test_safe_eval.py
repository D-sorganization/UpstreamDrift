import ast
from typing import Any

import numpy as np
import pytest
from safe_eval import safe_eval, safe_eval_math, validate_expression


class TestValidateExpression:
    def test_valid_math_expression(self) -> Any:
        tree = validate_expression("2 * x + 3 * y", {"x", "y"})
        assert isinstance(tree, ast.Expression)

    def test_empty_expression_raises(self) -> Any:
        with pytest.raises(ValueError, match="Expression must not be empty"):
            validate_expression("")
        with pytest.raises(ValueError, match="Expression must not be empty"):
            validate_expression("  ")

    def test_invalid_syntax_raises(self) -> Any:
        with pytest.raises(ValueError, match="Invalid syntax"):
            validate_expression("2 * +")

    def test_unsafe_operation_raises(self) -> Any:
        with pytest.raises(ValueError, match="Unsafe operation detected: List"):
            validate_expression("[1, 2, 3]")
        with pytest.raises(ValueError, match="Unsafe operation detected: Lambda"):
            validate_expression("lambda x: x")

    def test_unknown_name_raises(self) -> Any:
        with pytest.raises(ValueError, match="Unknown variable or function: z"):
            validate_expression("x + z", {"x"})

    def test_function_calls(self) -> Any:
        # allowed call
        tree = validate_expression("sin(x)", {"sin", "x"})
        assert isinstance(tree, ast.Expression)

        # unknown function
        with pytest.raises(ValueError, match="Unknown function: cos"):
            validate_expression("cos(x)", {"sin", "x"})

        # attribute call
        with pytest.raises(
            ValueError, match="Attribute-based function calls not allowed"
        ):
            validate_expression("math.sin(x)", {"math", "x"})

    def test_starred_call_args_rejected(self) -> Any:
        with pytest.raises(ValueError, match="Unsafe operation detected: Starred"):
            validate_expression("sum(*x)", {"sum", "x"})

    def test_allowed_names_none_bypasses_check(self) -> Any:
        # Should parse without error since name checking is disabled
        tree = validate_expression("unknown_func(unknown_var)", allowed_names=None)
        assert isinstance(tree, ast.Expression)


class TestSafeEval:
    def test_safe_eval_basic(self) -> Any:
        result = safe_eval("2 * x", {"x": 21})
        assert result == 42

    def test_safe_eval_explicit_allowlist(self) -> Any:
        # Only 'x' is in allowlist, even though namespace also has 'y'
        with pytest.raises(ValueError, match="Unknown variable or function: y"):
            safe_eval("x + y", {"x": 10, "y": 20}, allowed_names={"x"})

    def test_safe_eval_builtins_removed(self) -> Any:
        # Even if allowed_names = None, eval shouldn't find builtins
        # Pass let abs pass name validation, so it reaches eval()
        with pytest.raises(NameError):
            safe_eval("abs(-5)", {}, allowed_names={"abs"})


class TestSafeEvalMath:
    def test_safe_eval_math_numpy(self) -> Any:
        x = np.array([1, 2, 3])
        result = safe_eval_math("x * 2 + sum(x)", {"x": x})
        np.testing.assert_array_equal(result, np.array([8, 10, 12]))

        # Test math functions
        res_sin = safe_eval_math("sin(0)")
        assert res_sin == 0.0

    def test_safe_eval_math_scalar(self) -> Any:
        # Scalar math module evaluation
        result = safe_eval_math("sin(pi / 2) * x", {"x": 10}, use_numpy=False)
        assert result == 10.0

    def test_safe_eval_math_aliases(self) -> Any:
        result = safe_eval_math("np_sqrt(16)")
        assert result == 4.0

    def test_safe_eval_math_math_module_access(self) -> Any:
        # 'math' module should NOT be exposed in the namespace (security fix)
        # Attribute-based function calls like math.sin are blocked
        with pytest.raises(
            ValueError, match="Attribute-based function calls not allowed"
        ):
            safe_eval_math("math.sin(0)", {}, use_numpy=False)

        # Using 'math' as a bare name is also blocked now
        with pytest.raises(ValueError, match="Unknown variable or function: math"):
            safe_eval_math("math", {}, use_numpy=False)
