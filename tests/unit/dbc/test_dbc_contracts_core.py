"""DbC tests for the core contracts module itself.

Validates that the contracts infrastructure works correctly:
- require() raises PreconditionError on False
- ensure() raises PostconditionError on False
- ContractChecker.verify_invariants() catches violations
- @precondition, @postcondition, @invariant decorators fire correctly
- check_* helper functions work correctly
- ContractLevel toggling works
- require_state decorator works
"""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ["DBC_LEVEL"] = "enforce"


class TestRequirePrimitive(unittest.TestCase):
    """require() must raise PreconditionError on False."""

    def test_true_passes(self) -> None:
        from src.shared.python.core.contracts import require

        require(True, "should pass")

    def test_false_raises(self) -> None:
        from src.shared.python.core.contracts import PreconditionError, require

        with self.assertRaises(PreconditionError):
            require(False, "must fail")

    def test_error_message_included(self) -> None:
        from src.shared.python.core.contracts import PreconditionError, require

        try:
            require(False, "custom message")
            self.fail("Should have raised")
        except PreconditionError as e:
            self.assertIn("custom message", str(e))

    def test_value_diagnostics(self) -> None:
        from src.shared.python.core.contracts import PreconditionError, require

        try:
            require(False, "bad value", value=-42)
            self.fail("Should have raised")
        except PreconditionError as e:
            self.assertEqual(e.value, -42)


class TestEnsurePrimitive(unittest.TestCase):
    """ensure() must raise PostconditionError on False."""

    def test_true_passes(self) -> None:
        from src.shared.python.core.contracts import ensure

        ensure(True, "should pass")

    def test_false_raises(self) -> None:
        from src.shared.python.core.contracts import PostconditionError, ensure

        with self.assertRaises(PostconditionError):
            ensure(False, "must fail")

    def test_error_message_included(self) -> None:
        from src.shared.python.core.contracts import PostconditionError, ensure

        try:
            ensure(False, "postcondition msg")
            self.fail("Should have raised")
        except PostconditionError as e:
            self.assertIn("postcondition msg", str(e))


class TestPreconditionDecorator(unittest.TestCase):
    """@precondition decorator."""

    def test_decorator_passes_valid(self) -> None:
        from src.shared.python.core.contracts import precondition

        @precondition(lambda x: x > 0, "x must be positive")
        def square(x: float) -> float:
            return x * x

        self.assertAlmostEqual(square(3.0), 9.0)

    def test_decorator_raises_invalid(self) -> None:
        from src.shared.python.core.contracts import (
            ContractViolationError,
            precondition,
        )

        @precondition(lambda x: x > 0, "x must be positive")
        def square(x: float) -> float:
            return x * x

        with self.assertRaises(ContractViolationError):
            square(-1.0)

    def test_multiple_preconditions(self) -> None:
        from src.shared.python.core.contracts import (
            ContractViolationError,
            precondition,
        )

        @precondition(lambda x, y: x > 0, "x must be positive")
        @precondition(lambda x, y: y > 0, "y must be positive")
        def divide(x: float, y: float) -> float:
            return x / y

        self.assertAlmostEqual(divide(10.0, 2.0), 5.0)
        with self.assertRaises(ContractViolationError):
            divide(-1.0, 2.0)
        with self.assertRaises(ContractViolationError):
            divide(1.0, -2.0)

    def test_preserves_function_name(self) -> None:
        from src.shared.python.core.contracts import precondition

        @precondition(lambda x: True, "always passes")
        def my_function(x: float) -> float:
            return x

        self.assertEqual(my_function.__name__, "my_function")


class TestPostconditionDecorator(unittest.TestCase):
    """@postcondition decorator."""

    def test_valid_result_passes(self) -> None:
        from src.shared.python.core.contracts import postcondition

        @postcondition(lambda result: result >= 0, "result must be non-negative")
        def abs_val(x: float) -> float:
            return abs(x)

        self.assertAlmostEqual(abs_val(-5.0), 5.0)

    def test_invalid_result_raises(self) -> None:
        from src.shared.python.core.contracts import (
            ContractViolationError,
            postcondition,
        )

        @postcondition(lambda result: result >= 0, "result must be non-negative")
        def negate(x: float) -> float:
            return -x

        with self.assertRaises(ContractViolationError):
            negate(5.0)  # Returns -5, violates postcondition


class TestInvariantDecorator(unittest.TestCase):
    """@invariant class decorator."""

    def test_valid_construction_passes(self) -> None:
        from src.shared.python.core.contracts import invariant

        @invariant(lambda self: self.value > 0, "value must be positive")
        class PositiveValue:
            def __init__(self, value: float) -> None:
                self.value = value

        obj = PositiveValue(5.0)
        self.assertEqual(obj.value, 5.0)

    def test_invalid_construction_raises(self) -> None:
        from src.shared.python.core.contracts import InvariantError, invariant

        @invariant(lambda self: self.value > 0, "value must be positive")
        class PositiveValue:
            def __init__(self, value: float) -> None:
                self.value = value

        with self.assertRaises(InvariantError):
            PositiveValue(-1.0)

    def test_stacked_invariants(self) -> None:
        from src.shared.python.core.contracts import InvariantError, invariant

        @invariant(lambda self: self.x > 0, "x must be positive")
        @invariant(lambda self: self.y > 0, "y must be positive")
        class Point:
            def __init__(self, x: float, y: float) -> None:
                self.x = x
                self.y = y

        Point(1.0, 2.0)  # OK
        with self.assertRaises(InvariantError):
            Point(-1.0, 2.0)
        with self.assertRaises(InvariantError):
            Point(1.0, -2.0)


class TestContractChecker(unittest.TestCase):
    """ContractChecker mixin invariant checking."""

    def test_verify_invariants_passes(self) -> None:
        from src.shared.python.core.contracts import ContractChecker

        class TestChecker(ContractChecker):
            def __init__(self) -> None:
                self.mass = 1.0

            def _get_invariants(self):  # type: ignore[no-untyped-def]
                return [(lambda: self.mass > 0, "mass must be positive")]

        checker = TestChecker()
        self.assertTrue(checker.verify_invariants())

    def test_verify_invariants_raises(self) -> None:
        from src.shared.python.core.contracts import ContractChecker, InvariantError

        class TestChecker(ContractChecker):
            def __init__(self) -> None:
                self.mass = -1.0

            def _get_invariants(self):  # type: ignore[no-untyped-def]
                return [(lambda: self.mass > 0, "mass must be positive")]

        checker = TestChecker()
        with self.assertRaises(InvariantError):
            checker.verify_invariants()


class TestCheckHelpers(unittest.TestCase):
    """check_finite, check_shape, check_positive, check_non_negative, check_symmetric."""

    def _run_check_finite_cases(self) -> None:
        """Parametrized via subTest for check_finite."""
        from src.shared.python.core.contracts import check_finite

        cases = [
            (np.array([1.0, 2.0, 3.0]), True, "valid-array"),
            (np.array([1.0, float("nan"), 3.0]), False, "nan-in-array"),
            (np.array([1.0, float("inf"), 3.0]), False, "inf-in-array"),
            (None, False, "none-input"),
        ]
        for value, expected, label in cases:
            with self.subTest(label=label):
                self.assertEqual(check_finite(value), expected)

    def test_check_finite_cases(self) -> None:
        self._run_check_finite_cases()

    def _run_check_shape_cases(self) -> None:
        """Parametrized via subTest for check_shape."""
        from src.shared.python.core.contracts import check_shape

        cases = [
            (np.zeros((3, 3)), (3, 3), True, "correct-shape"),
            (np.zeros((3, 3)), (3, 4), False, "wrong-shape"),
            (None, (3, 3), False, "none-input"),
        ]
        for value, shape, expected, label in cases:
            with self.subTest(label=label):
                self.assertEqual(check_shape(value, shape), expected)

    def test_check_shape_cases(self) -> None:
        self._run_check_shape_cases()

    def _run_check_positive_cases(self) -> None:
        """Parametrized via subTest for check_positive."""
        from src.shared.python.core.contracts import check_positive

        cases = [
            (5.0, True, "positive-scalar"),
            (0.0, False, "zero-scalar"),
            (-1.0, False, "negative-scalar"),
            (np.array([1.0, 2.0, 3.0]), True, "all-positive-array"),
            (np.array([1.0, 0.0, 3.0]), False, "zero-in-array"),
        ]
        for value, expected, label in cases:
            with self.subTest(label=label):
                self.assertEqual(check_positive(value), expected)

    def test_check_positive_cases(self) -> None:
        self._run_check_positive_cases()

    def _run_check_non_negative_cases(self) -> None:
        """Parametrized via subTest for check_non_negative."""
        from src.shared.python.core.contracts import check_non_negative

        cases = [
            (0.0, True, "zero-scalar"),
            (5.0, True, "positive-scalar"),
            (-1.0, False, "negative-scalar"),
            (np.array([0.0, 1.0, 2.0]), True, "non-negative-array"),
            (np.array([0.0, -1.0, 2.0]), False, "negative-in-array"),
        ]
        for value, expected, label in cases:
            with self.subTest(label=label):
                self.assertEqual(check_non_negative(value), expected)

    def test_check_non_negative_cases(self) -> None:
        self._run_check_non_negative_cases()

    def test_check_symmetric_true(self) -> None:
        from src.shared.python.core.contracts import check_symmetric

        M = np.array([[1.0, 2.0], [2.0, 3.0]])
        self.assertTrue(check_symmetric(M))

    def test_check_symmetric_false(self) -> None:
        from src.shared.python.core.contracts import check_symmetric

        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertFalse(check_symmetric(M))

    def test_check_positive_definite_true(self) -> None:
        from src.shared.python.core.contracts import check_positive_definite

        M = np.array([[2.0, -1.0], [-1.0, 2.0]])
        self.assertTrue(check_positive_definite(M))

    def test_check_positive_definite_false(self) -> None:
        from src.shared.python.core.contracts import check_positive_definite

        M = np.array([[1.0, 2.0], [2.0, 1.0]])  # Eigenvalues: 3, -1
        self.assertFalse(check_positive_definite(M))


class TestContractLevelSwitching(unittest.TestCase):
    """Contract level can be switched at runtime."""

    def test_set_and_get(self) -> None:
        from src.shared.python.core.contracts import (
            ContractLevel,
            get_contract_level,
            set_contract_level,
        )

        original = get_contract_level()
        try:
            set_contract_level(ContractLevel.WARN)
            self.assertEqual(get_contract_level(), ContractLevel.WARN)
            set_contract_level(ContractLevel.OFF)
            self.assertEqual(get_contract_level(), ContractLevel.OFF)
        finally:
            set_contract_level(original)

    def test_enable_disable(self) -> None:
        from src.shared.python.core.contracts import (
            ContractLevel,
            disable_contracts,
            enable_contracts,
            get_contract_level,
            set_contract_level,
        )

        original = get_contract_level()
        try:
            disable_contracts()
            self.assertEqual(get_contract_level(), ContractLevel.OFF)
            enable_contracts()
            self.assertEqual(get_contract_level(), ContractLevel.ENFORCE)
        finally:
            set_contract_level(original)


class TestExceptionHierarchy(unittest.TestCase):
    """Exception hierarchy is correct."""

    def test_all_subclass_contract_violation(self) -> None:
        """All contract exception types must be subclasses of ContractViolationError."""
        from src.shared.python.core.contracts import (
            ContractViolationError,
            InvariantError,
            PostconditionError,
            PreconditionError,
            StateError,
        )

        for exc_cls in (
            PreconditionError,
            PostconditionError,
            InvariantError,
            StateError,
        ):
            with self.subTest(exc=exc_cls.__name__):
                self.assertTrue(issubclass(exc_cls, ContractViolationError))


class TestRequireState(unittest.TestCase):
    """require_state decorator."""

    def test_valid_state_passes(self) -> None:
        from src.shared.python.core.contracts import require_state

        @require_state(lambda self: self._ready, "ready")
        def do_work(self) -> str:
            return "done"

        class Worker:
            _ready = True

        w = Worker()
        # Since require_state wraps a method, we need to bind it
        bound = do_work.__get__(w, Worker)
        self.assertEqual(bound(), "done")

    def test_invalid_state_raises(self) -> None:
        from src.shared.python.core.contracts import StateError, require_state

        @require_state(lambda self: self._ready, "ready")
        def do_work(self) -> str:
            return "done"

        class Worker:
            _ready = False

        w = Worker()
        bound = do_work.__get__(w, Worker)
        with self.assertRaises(StateError):
            bound()


class TestFiniteResultDecorator(unittest.TestCase):
    """finite_result decorator."""

    def test_finite_result_passes(self) -> None:
        from src.shared.python.core.contracts import finite_result

        @finite_result
        def compute() -> np.ndarray:
            return np.array([1.0, 2.0, 3.0])

        result = compute()
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_nan_result_raises(self) -> None:
        from src.shared.python.core.contracts import (
            ContractViolationError,
            finite_result,
        )

        @finite_result
        def compute() -> np.ndarray:
            return np.array([1.0, float("nan"), 3.0])

        with self.assertRaises(ContractViolationError):
            compute()

    def test_none_result_passes(self) -> None:
        """None result should pass (finite_result allows None)."""
        from src.shared.python.core.contracts import finite_result

        @finite_result
        def compute() -> np.ndarray | None:
            return None

        result = compute()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
