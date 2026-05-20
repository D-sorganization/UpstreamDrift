"""Coverage for src/shared/python/safe_eval.py.

The safe expression evaluator is security-critical — exercise both the
happy path and every rejection path of the AST validator.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.safe_eval import (
    NUMPY_MATH_NAMESPACE,
    SCALAR_MATH_NAMESPACE,
    safe_eval,
    safe_eval_math,
    validate_expression,
)


class TestValidateExpression:
    def test_empty_expression_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            validate_expression("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            validate_expression("   \t\n")

    def test_syntax_error_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid syntax"):
            validate_expression("1 +")

    def test_disallowed_node_lambda(self) -> None:
        with pytest.raises(ValueError):
            validate_expression("(lambda: 1)()")

    def test_disallowed_attribute_access(self) -> None:
        with pytest.raises(ValueError, match="Unsafe operation"):
            validate_expression("a.b")

    def test_disallowed_dict_literal(self) -> None:
        with pytest.raises(ValueError, match="Unsafe operation"):
            validate_expression("{1: 2}")

    def test_disallowed_setcomp(self) -> None:
        with pytest.raises(ValueError, match="Unsafe operation"):
            validate_expression("[x for x in [1, 2]]")

    def test_unknown_name_with_allowlist(self) -> None:
        with pytest.raises(ValueError, match="Unknown variable"):
            validate_expression("x + y", allowed_names={"x"})

    def test_unknown_function_with_allowlist(self) -> None:
        with pytest.raises(ValueError, match="Unknown function"):
            validate_expression("f(1)", allowed_names={"x"})

    def test_attribute_function_call_rejected(self) -> None:
        # ast.parse permits the syntax but our validator must reject it
        # because the attribute access is the disallowed step.
        with pytest.raises(ValueError):
            validate_expression("os.system('ls')")

    def test_allowed_name_passes(self) -> None:
        tree = validate_expression("x + 1", allowed_names={"x"})
        assert tree is not None


class TestSafeEval:
    def test_basic_arithmetic(self) -> None:
        assert safe_eval("1 + 2 * 3", {}) == 7
        assert safe_eval("(1 + 2) * 3", {}) == 9
        assert safe_eval("10 / 4", {}) == 2.5
        assert safe_eval("10 // 4", {}) == 2
        assert safe_eval("10 % 3", {}) == 1
        assert safe_eval("2 ** 10", {}) == 1024

    def test_unary(self) -> None:
        assert safe_eval("-5", {}) == -5
        assert safe_eval("+5", {}) == 5
        assert safe_eval("not True", {"True": True}) is False
        assert safe_eval("~0", {}) == -1

    def test_compare_chain(self) -> None:
        assert safe_eval("1 < 2 < 3", {}) is True
        assert safe_eval("1 < 2 > 5", {}) is False
        assert safe_eval("1 == 1", {}) is True
        assert safe_eval("1 != 2", {}) is True

    def test_bool_ops(self) -> None:
        assert safe_eval("True and False", {"True": True, "False": False}) is False
        assert safe_eval("True or False", {"True": True, "False": False}) is True

    def test_namespace_lookup(self) -> None:
        assert safe_eval("x * y", {"x": 3, "y": 4}) == 12

    def test_name_allowed_but_missing_from_namespace(self) -> None:
        # When validation accepts a name but execution can't find it.
        with pytest.raises(NameError):
            safe_eval("x", {}, allowed_names={"x"})

    def test_undefined_name_with_default_allowlist(self) -> None:
        # Default behavior: allowed_names defaults to namespace.keys(), so
        # undefined names are rejected at validation as ValueError.
        with pytest.raises(ValueError, match="Unknown variable"):
            safe_eval("undefined_var", {})

    def test_function_call(self) -> None:
        assert safe_eval("add(2, 3)", {"add": lambda a, b: a + b}) == 5

    def test_call_with_keywords(self) -> None:
        def f(*, x: int, y: int) -> int:
            return x - y

        assert safe_eval("f(x=10, y=3)", {"f": f}) == 7

    def test_call_with_starred(self) -> None:
        def f(a: int, b: int, c: int) -> int:
            return a + b + c

        assert safe_eval("f(*args)", {"f": f, "args": [1, 2, 3]}) == 6

    def test_list_tuple_literals(self) -> None:
        assert safe_eval("[1, 2, 3]", {}) == [1, 2, 3]
        assert safe_eval("(1, 2)", {}) == (1, 2)

    def test_subscript_and_slice(self) -> None:
        assert safe_eval("xs[1]", {"xs": [10, 20, 30]}) == 20
        assert safe_eval("xs[0:2]", {"xs": [10, 20, 30]}) == [10, 20]
        assert safe_eval("xs[::-1]", {"xs": [1, 2, 3]}) == [3, 2, 1]

    def test_ifexp(self) -> None:
        assert safe_eval("1 if cond else 2", {"cond": True}) == 1
        assert safe_eval("1 if cond else 2", {"cond": False}) == 2

    def test_none_expression_raises(self) -> None:
        with pytest.raises(ValueError, match="must be provided"):
            safe_eval(None, {})  # type: ignore[arg-type]

    def test_non_dict_namespace_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a dict"):
            safe_eval("1", "not a dict")  # type: ignore[arg-type]

    def test_default_allowed_names_uses_namespace(self) -> None:
        # When allowed_names is None it defaults to namespace.keys()
        assert safe_eval("x", {"x": 42}) == 42
        with pytest.raises(ValueError, match="Unknown variable"):
            safe_eval("y", {"x": 42})


class TestSafeEvalMath:
    def test_numpy_namespace_default(self) -> None:
        result = safe_eval_math("sqrt(16)")
        assert np.isclose(result, 4.0)

    def test_scalar_namespace(self) -> None:
        result = safe_eval_math("sqrt(9)", use_numpy=False)
        assert math.isclose(result, 3.0)

    def test_variables_merged(self) -> None:
        result = safe_eval_math("sin(0) + offset", {"offset": 5})
        assert np.isclose(result, 5.0)

    def test_constants(self) -> None:
        assert np.isclose(safe_eval_math("pi"), np.pi)
        assert np.isclose(safe_eval_math("e", use_numpy=False), math.e)

    def test_none_expression(self) -> None:
        with pytest.raises(ValueError, match="must be provided"):
            safe_eval_math(None)  # type: ignore[arg-type]

    def test_array_operations(self) -> None:
        arr = np.array([1.0, 4.0, 9.0])
        result = safe_eval_math("sqrt(x)", {"x": arr})
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


def test_namespaces_have_expected_keys() -> None:
    for key in ("sin", "cos", "sqrt", "pi", "e"):
        assert key in NUMPY_MATH_NAMESPACE
        assert key in SCALAR_MATH_NAMESPACE
