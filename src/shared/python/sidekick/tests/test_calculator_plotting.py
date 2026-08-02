"""Security + behavior tests for calculator plot expression evaluation (#3274).

``_evaluate_expression`` previously used raw ``eval`` with a cleared
``__builtins__``, which does NOT prevent attribute-based sandbox escapes. It now
delegates to the repo's AST-validated ``safe_eval``, which rejects attribute
access / dunder traversal at parse time.
"""

from __future__ import annotations

import math

import pytest
from sidekick.ui.tools_sidebar.calculator_plotting import _evaluate_expression

pytestmark = pytest.mark.unit


def test_evaluate_expression_blocks_attribute_escape() -> None:
    """A known attribute-traversal escape payload must be rejected."""
    with pytest.raises(ValueError):
        _evaluate_expression("(1.0).__class__.__mro__[1].__subclasses__()", 1.0)


@pytest.mark.parametrize(
    "payload",
    [
        "().__class__.__bases__[0].__subclasses__()",
        "x.__class__",
        "__import__('os').system('echo hi')",
    ],
)
def test_evaluate_expression_rejects_escape_payloads(payload: str) -> None:
    with pytest.raises(ValueError):
        _evaluate_expression(payload, 1.0)


@pytest.mark.parametrize(
    ("expression", "x_value", "expected"),
    [
        ("sin(x) + pi", 0.0, math.pi),
        ("sqrt(x)", 4.0, 2.0),
        ("x**2", 3.0, 9.0),
        ("cos(x)", 0.0, 1.0),
    ],
)
def test_evaluate_expression_allows_legitimate_math(
    expression: str, x_value: float, expected: float
) -> None:
    """Legitimate calculator expressions still evaluate correctly."""
    result = _evaluate_expression(expression, x_value)
    assert result == pytest.approx(expected)


def test_evaluate_expression_requires_scalar_result() -> None:
    with pytest.raises(ValueError, match="numeric scalar"):
        # ``pi`` alone is scalar; force a non-scalar via a tuple-like name access
        # is blocked, so use a boolean which _is_scalar_number rejects.
        _evaluate_expression("x > 0", 1.0)
