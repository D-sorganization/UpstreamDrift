"""Tests for ExpressionFunction safe expression evaluator.

Covers AST validation rejection paths, allowed-name handling, and runtime
evaluation against ``DoublePendulumState`` context.
"""

from __future__ import annotations

import math

import pytest
from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumState,
    ExpressionFunction,
    compile_forcing_functions,
)


@pytest.fixture()
def zero_state() -> DoublePendulumState:
    return DoublePendulumState(theta1=0.0, theta2=0.0, omega1=0.0, omega2=0.0)


class TestExpressionFunctionEvaluation:
    def test_constant_expression(self, zero_state: DoublePendulumState) -> None:
        fn = ExpressionFunction("1.5")
        assert fn(0.0, zero_state) == pytest.approx(1.5)

    def test_uses_state_variables(self) -> None:
        fn = ExpressionFunction("theta1 + 2 * omega2")
        state = DoublePendulumState(theta1=1.0, theta2=0.0, omega1=0.0, omega2=2.0)
        assert fn(0.0, state) == pytest.approx(5.0)

    def test_uses_time(self, zero_state: DoublePendulumState) -> None:
        fn = ExpressionFunction("sin(t)")
        assert fn(math.pi / 2, zero_state) == pytest.approx(1.0)

    def test_uses_pi_and_tau(self, zero_state: DoublePendulumState) -> None:
        fn = ExpressionFunction("pi + tau")
        assert fn(0.0, zero_state) == pytest.approx(math.pi + math.tau)

    def test_strips_whitespace(self, zero_state: DoublePendulumState) -> None:
        fn = ExpressionFunction("  2 + 3  ")
        assert fn(0.0, zero_state) == pytest.approx(5.0)

    def test_allowed_math_functions(self, zero_state: DoublePendulumState) -> None:
        fn = ExpressionFunction("sqrt(fabs(log(exp(2.0))))")
        assert fn(0.0, zero_state) == pytest.approx(math.sqrt(2.0))


class TestExpressionFunctionValidation:
    @pytest.mark.parametrize(
        "expr",
        [
            "[1, 2]",  # List
            "{1: 2}",  # Dict
            "{1, 2}",  # Set
            "[x for x in [1]]",  # ListComp
            "(x for x in [1])",  # GeneratorExp
            "lambda x: x",  # Lambda
            "1 if True else 2",  # IfExp
        ],
    )
    def test_rejects_disallowed_nodes(self, expr: str) -> None:
        with pytest.raises(ValueError, match="Disallowed syntax"):
            ExpressionFunction(expr)

    def test_rejects_attribute_access(self) -> None:
        with pytest.raises(ValueError, match="Attribute"):
            ExpressionFunction("sin.__doc__")

    def test_rejects_syntax_error(self) -> None:
        with pytest.raises(ValueError, match="Syntax error"):
            ExpressionFunction("1 + ")

    def test_rejects_unknown_function(self) -> None:
        with pytest.raises(ValueError, match="not permitted"):
            ExpressionFunction("eval('1')")

    def test_rejects_unknown_variable(self) -> None:
        with pytest.raises(ValueError, match="unknown variable"):
            ExpressionFunction("foobar + 1")

    def test_rejects_indirect_call(self) -> None:
        # A call where func is not a direct Name (e.g. via subscript) — synthesize one
        # via attribute since attribute itself is blocked first; use call on call result:
        with pytest.raises(ValueError):
            # "sin(1)(2)" - func is a Call, not a Name
            ExpressionFunction("sin(1)(2)")


class TestCompileForcingFunctions:
    def test_returns_two_callables(self) -> None:
        shoulder, wrist = compile_forcing_functions("1.0", "2.0")
        state = DoublePendulumState(theta1=0.0, theta2=0.0, omega1=0.0, omega2=0.0)
        assert shoulder(0.0, state) == pytest.approx(1.0)
        assert wrist(0.0, state) == pytest.approx(2.0)
