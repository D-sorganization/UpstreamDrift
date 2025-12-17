import math

import pytest

from double_pendulum_model.safe_eval import SafeEvaluator


def test_safe_eval_basic_math() -> None:
    """Test basic mathematical expressions."""
    evaluator = SafeEvaluator()
    assert math.isclose(evaluator.evaluate("sin(pi/2) + 1"), 2.0)


def test_safe_eval_variables() -> None:
    """Test evaluation with variables."""
    evaluator = SafeEvaluator(allowed_variables={"x", "y"})
    assert math.isclose(evaluator.evaluate("x + y", {"x": 1.0, "y": 2.0}), 3.0)


def test_safe_eval_rejects_imports() -> None:
    """Test that imports are rejected."""
    evaluator = SafeEvaluator()
    # "import os" is a statement, not an expression.
    # ast.parse(..., mode='eval') raises SyntaxError.
    with pytest.raises(ValueError, match="Invalid syntax"):
        evaluator.validate("import os")


def test_safe_eval_rejects_builtins() -> None:
    """Test that usage of builtins is rejected."""
    evaluator = SafeEvaluator()
    # This involves Attribute access or Call of non-allowed name
    error_pattern = (
        "Disallowed syntax|Only direct function calls|Function .* is not permitted"
    )
    with pytest.raises(ValueError, match=error_pattern):
        evaluator.validate("__import__('os').system('ls')")


def test_safe_eval_rejects_attributes() -> None:
    """Test that attribute access is rejected."""
    evaluator = SafeEvaluator()
    # We disallowed ast.Attribute
    with pytest.raises(ValueError, match="Only direct function calls are permitted"):
        evaluator.validate("math.sin(0)")


def test_safe_eval_rejects_unknown_variables() -> None:
    """Test that unknown variables are rejected."""
    evaluator = SafeEvaluator(allowed_variables={"x"})
    with pytest.raises(ValueError, match="Unknown variable 'z'"):
        evaluator.validate("z")


def test_safe_eval_rejects_complex_nodes() -> None:
    """Test that complex nodes (like list comprehensions) are rejected."""
    evaluator = SafeEvaluator()
    with pytest.raises(ValueError, match="Disallowed syntax"):
        evaluator.validate("[x for x in range(10)]")


def test_safe_eval_power_operator() -> None:
    """Test that the power operator (^) is rejected to prevent confusion."""
    evaluator = SafeEvaluator()
    # Check if we want to allow **
    assert evaluator.evaluate("2**3") == 8.0

    # Using ^ should raise a ValueError
    error_msg = "Use '\\*\\*' for exponentiation instead of '\\^'"
    with pytest.raises(ValueError, match=error_msg):
        evaluator.evaluate("2^3")
