"""Tests for symbolic_engine.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("sympy", reason="sympy not installed")

from sidekick.symbolic_engine import (
    SymbolicEngineError,
    is_sympy_available,
    symbolic_diff,
    symbolic_expand,
    symbolic_integrate,
    symbolic_simplify,
    symbolic_solve,
    symbolic_to_latex,
)


def test_is_sympy_available() -> None:
    """Test is_sympy_available."""
    assert isinstance(is_sympy_available(), bool)


def test_require_sympy_raises_when_missing() -> None:
    """Test that functions raise SymbolicEngineError when sympy is missing."""
    with patch("sidekick.symbolic_engine._SYMPY_AVAILABLE", False):
        with pytest.raises(SymbolicEngineError, match="sympy is not installed"):
            symbolic_solve("x**2 - 4")


def test_symbolic_solve_success() -> None:
    """Test solving expressions."""
    solutions = symbolic_solve("x**2 - 4", "x")
    assert set(solutions) == {"-2", "2"}

    # Solve for y
    solutions_y = symbolic_solve("y**2 - 9", "y")
    assert set(solutions_y) == {"-3", "3"}


def test_symbolic_solve_validation() -> None:
    """Test input validation for solve."""
    # invalid expr_str type
    with pytest.raises(TypeError, match="expr_str must be str"):
        symbolic_solve(123)  # type: ignore[arg-type]

    # empty expr_str
    with pytest.raises(ValueError, match="expr_str must not be empty"):
        symbolic_solve("")

    # invalid symbol type
    with pytest.raises(TypeError, match="symbol must be str"):
        symbolic_solve("x**2 - 4", 123)  # type: ignore[arg-type]

    # empty symbol
    with pytest.raises(ValueError, match="symbol must not be empty"):
        symbolic_solve("x**2 - 4", "")


def test_symbolic_solve_failure() -> None:
    """Test solving unparseable expression."""
    with pytest.raises(ValueError, match="Cannot parse expression"):
        symbolic_solve("x ** -")


def test_symbolic_diff_success() -> None:
    """Test differentiating expressions."""
    assert symbolic_diff("x**3", "x") == "3*x**2"
    assert symbolic_diff("x**3", "x", order=2) == "6*x"


def test_symbolic_diff_validation() -> None:
    """Test input validation for diff."""
    # order is not int
    with pytest.raises(TypeError, match="order must be int"):
        symbolic_diff("x**3", "x", order="1")  # type: ignore[arg-type]

    # order < 1
    with pytest.raises(ValueError, match="order must be >= 1"):
        symbolic_diff("x**3", "x", order=0)

    # symbol validation
    with pytest.raises(TypeError, match="symbol must be str"):
        symbolic_diff("x**3", 123)  # type: ignore[arg-type]


def test_symbolic_diff_failure() -> None:
    """Test differentiation failure on bad input."""
    # Wait, diffing an invalid expression raises ValueError at parse time
    with pytest.raises(ValueError, match="Cannot parse expression"):
        symbolic_diff("x ** -")


def test_symbolic_integrate_success() -> None:
    """Test integrating expressions."""
    # Indefinite
    assert symbolic_integrate("x**2", "x") == "x**3/3"

    # Definite
    assert symbolic_integrate("x**2", "x", lower="0", upper="1") == "1/3"


def test_symbolic_integrate_validation() -> None:
    """Test validation in integration."""
    # mismatched bounds
    with pytest.raises(
        ValueError, match="Both lower and upper bounds must be provided"
    ):
        symbolic_integrate("x**2", "x", lower="0")

    with pytest.raises(
        ValueError, match="Both lower and upper bounds must be provided"
    ):
        symbolic_integrate("x**2", "x", upper="1")

    # invalid symbol type
    with pytest.raises(TypeError, match="symbol must be str"):
        symbolic_integrate("x**2", 123)  # type: ignore[arg-type]


def test_symbolic_simplify_success() -> None:
    """Test simplifying expressions."""
    assert symbolic_simplify("sin(x)**2 + cos(x)**2") == "1"


def test_symbolic_simplify_validation() -> None:
    """Test simplify validation."""
    with pytest.raises(TypeError, match="expr_str must be str"):
        symbolic_simplify(123)  # type: ignore[arg-type]


def test_symbolic_expand_success() -> None:
    """Test expanding expressions."""
    assert symbolic_expand("(x + 1)**2") == "x**2 + 2*x + 1"


def test_symbolic_expand_validation() -> None:
    """Test expand validation."""
    with pytest.raises(TypeError, match="expr_str must be str"):
        symbolic_expand(123)  # type: ignore[arg-type]


def test_symbolic_to_latex_success() -> None:
    """Test converting expressions to LaTeX."""
    assert symbolic_to_latex("x**2 / (x + 1)") == "\\frac{x^{2}}{x + 1}"


def test_symbolic_to_latex_validation() -> None:
    """Test to_latex validation."""
    with pytest.raises(TypeError, match="expr_str must be str"):
        symbolic_to_latex(123)  # type: ignore[arg-type]
