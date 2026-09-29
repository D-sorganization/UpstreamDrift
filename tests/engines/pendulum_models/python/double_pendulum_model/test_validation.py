"""Tests for UI input validation helpers."""

from __future__ import annotations

from src.engines.pendulum_models.python.double_pendulum_model.ui.validation import (
    validate_polynomial_text,
    validate_torque_text,
)


class TestValidatePolynomialText:
    def test_empty_string_valid(self) -> None:
        assert validate_polynomial_text("") is None

    def test_whitespace_only_valid(self) -> None:
        assert validate_polynomial_text("   ") is None

    def test_single_number_valid(self) -> None:
        assert validate_polynomial_text("1.5") is None

    def test_sum_of_numbers_valid(self) -> None:
        assert validate_polynomial_text("1.0 + 2.5 + 3") is None

    def test_invalid_text_returns_error(self) -> None:
        result = validate_polynomial_text("abc")
        assert result is not None
        assert "Invalid polynomial" in result

    def test_garbage_token_returns_error(self) -> None:
        result = validate_polynomial_text("1+foo")
        assert result is not None


class TestValidateTorqueText:
    def test_empty_string_valid(self) -> None:
        assert validate_torque_text("") is None

    def test_whitespace_only_valid(self) -> None:
        assert validate_torque_text("   ") is None

    def test_valid_expression(self) -> None:
        assert validate_torque_text("sin(t) + theta1") is None

    def test_invalid_syntax_returns_error(self) -> None:
        result = validate_torque_text("1 + ")
        assert result is not None
        assert "Invalid syntax" in result
