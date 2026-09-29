"""Tests for src.shared.python.core.contracts.validators (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from src.shared.python.core.contracts.validators import (
    check_finite,
    check_non_negative,
    check_positive,
    check_positive_definite,
    check_shape,
    check_symmetric,
)


class TestCheckFinite:
    def test_finite_array_returns_true(self) -> None:
        assert check_finite(np.array([1.0, 2.0, 3.0])) is True

    def test_nan_returns_false(self) -> None:
        assert check_finite(np.array([1.0, np.nan])) is False

    def test_inf_returns_false(self) -> None:
        assert check_finite(np.array([1.0, np.inf])) is False

    def test_none_returns_false(self) -> None:
        assert check_finite(None) is False

    def test_empty_array_returns_true(self) -> None:
        assert check_finite(np.array([])) is True

    def test_2d_finite_array(self) -> None:
        assert check_finite(np.ones((3, 3))) is True

    def test_2d_with_nan(self) -> None:
        arr = np.ones((2, 2))
        arr[0, 0] = np.nan
        assert check_finite(arr) is False


class TestCheckShape:
    def test_correct_shape_returns_true(self) -> None:
        arr = np.zeros((3, 2))
        assert check_shape(arr, (3, 2)) is True

    def test_wrong_shape_returns_false(self) -> None:
        arr = np.zeros((3, 2))
        assert check_shape(arr, (2, 3)) is False

    def test_none_returns_false(self) -> None:
        assert check_shape(None, (3,)) is False

    def test_1d_shape(self) -> None:
        arr = np.zeros(5)
        assert check_shape(arr, (5,)) is True

    def test_scalar_shape(self) -> None:
        arr = np.array(42.0)
        assert check_shape(arr, ()) is True


class TestCheckPositive:
    def test_positive_scalar(self) -> None:
        assert check_positive(5.0) is True

    def test_zero_returns_false(self) -> None:
        assert check_positive(0.0) is False

    def test_negative_returns_false(self) -> None:
        assert check_positive(-1.0) is False

    def test_positive_array_returns_true(self) -> None:
        assert check_positive(np.array([1.0, 2.0, 3.0])) is True

    def test_array_with_zero_returns_false(self) -> None:
        assert check_positive(np.array([1.0, 0.0, 3.0])) is False

    def test_array_with_negative_returns_false(self) -> None:
        assert check_positive(np.array([1.0, -1.0])) is False

    def test_integer_positive(self) -> None:
        assert check_positive(3) is True


class TestCheckNonNegative:
    def test_positive_scalar(self) -> None:
        assert check_non_negative(5.0) is True

    def test_zero_returns_true(self) -> None:
        assert check_non_negative(0.0) is True

    def test_negative_returns_false(self) -> None:
        assert check_non_negative(-0.1) is False

    def test_zero_array(self) -> None:
        assert check_non_negative(np.zeros(3)) is True

    def test_mixed_sign_array_false(self) -> None:
        assert check_non_negative(np.array([1.0, -0.5])) is False


class TestCheckSymmetric:
    def test_contract_validators_symmetric_matrix(self) -> None:
        M = np.array([[1.0, 2.0], [2.0, 3.0]])
        assert check_symmetric(M) is True

    def test_identity_symmetric(self) -> None:
        assert check_symmetric(np.eye(3)) is True

    def test_asymmetric_returns_false(self) -> None:
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert check_symmetric(M) is False

    def test_non_square_returns_false(self) -> None:
        M = np.ones((2, 3))
        assert check_symmetric(M) is False

    def test_1d_returns_false(self) -> None:
        arr = np.array([1.0, 2.0])
        assert check_symmetric(arr) is False


class TestCheckPositiveDefinite:
    def test_identity_is_positive_definite(self) -> None:
        assert check_positive_definite(np.eye(3)) is True

    def test_singular_not_positive_definite(self) -> None:
        M = np.zeros((2, 2))
        assert check_positive_definite(M) is False

    def test_diagonal_positive(self) -> None:
        M = np.diag([1.0, 2.0, 3.0])
        assert check_positive_definite(M) is True

    def test_indefinite_matrix(self) -> None:
        M = np.array([[1.0, 0.0], [0.0, -1.0]])
        assert check_positive_definite(M) is False

    def test_non_square_returns_false(self) -> None:
        M = np.ones((2, 3))
        assert check_positive_definite(M) is False
