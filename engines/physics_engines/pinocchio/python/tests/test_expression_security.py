import math

import pytest
from double_pendulum_model.physics.double_pendulum import (
    DoublePendulumState,
    ExpressionFunction,
)


def test_valid_expressions() -> None:
    # Setup a dummy state
    state = DoublePendulumState(theta1=1.0, theta2=2.0, omega1=0.5, omega2=0.5)
    t = 0.0

    # Simple arithmetic
    ef = ExpressionFunction("theta1 + theta2")
    assert ef(t, state) == 3.0

    # Math function
    ef = ExpressionFunction("sin(0)")
    assert ef(t, state) == 0.0

    # Math constant (pi is allowed in names)
    ef = ExpressionFunction("pi")
    assert ef(t, state) == math.pi

    # BitXor (if intended as such)
    ef = ExpressionFunction("1 ^ 0")  # 1 XOR 0 = 1
    assert ef(t, state) == 1.0


def test_attribute_access_blocked() -> None:
    # Should raise ValueError because Attribute node is disallowed
    with pytest.raises(ValueError, match="Disallowed syntax in expression: Attribute"):
        ExpressionFunction("sin.__doc__")

    with pytest.raises(ValueError, match="Disallowed syntax in expression: Attribute"):
        ExpressionFunction("theta1.__class__")


def test_import_blocked() -> None:
    # __import__ is not in allowed names
    with pytest.raises(ValueError, match="Function '__import__' is not permitted"):
        ExpressionFunction("__import__('os')")


def test_call_on_attribute_blocked() -> None:
    # Example: attribute call.
    # This parses as Call(func=Attribute(value=Name(foo), attr=bar))
    # _validate_ast visits the Call node first.
    # Since child.func is Attribute (not Name), it raises:
    # "Only direct function calls are permitted".
    with pytest.raises(ValueError, match="Only direct function calls are permitted"):
        ExpressionFunction("theta1.as_integer_ratio()")


def test_disallowed_syntax_nodes() -> None:
    # List creation is not allowed
    with pytest.raises(ValueError, match="Disallowed syntax"):
        ExpressionFunction("[x for x in range(10)]")

    # Dict type(not allowed)
    with pytest.raises(ValueError, match="Disallowed syntax"):
        ExpressionFunction("{'a': 1}")


def test_unknown_variable() -> None:
    with pytest.raises(ValueError, match="Use of unknown variable 'foo'"):
        ExpressionFunction("foo")


def test_unknown_function() -> None:
    with pytest.raises(ValueError, match="Function 'eval' is not permitted"):
        ExpressionFunction("eval('1')")
