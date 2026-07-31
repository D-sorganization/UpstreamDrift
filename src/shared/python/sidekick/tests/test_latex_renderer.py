"""Tests for latex_renderer.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("PyQt6", reason="PyQt6 not installed")
pytest.importorskip("sympy", reason="sympy not installed")

from PyQt6.QtWidgets import QLabel
from sidekick.latex_renderer import (
    LatexRenderError,
    _expr_to_latex,
    _make_label_widget,
    is_latex_ready,
    is_qt_available,
    render_expr_label,
    render_latex_label,
)


def test_is_qt_available() -> None:
    """Test is_qt_available function."""
    assert isinstance(is_qt_available(), bool)


def test_is_latex_ready() -> None:
    """Test is_latex_ready function."""
    assert isinstance(is_latex_ready(), bool)


def test_expr_to_latex_success() -> None:
    """Test conversion of expression to latex."""
    res = _expr_to_latex("x**2")
    assert "x^{2}" in res


def test_expr_to_latex_invalid_input() -> None:
    """Test input validation for _expr_to_latex."""
    with pytest.raises(TypeError):
        _expr_to_latex(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        _expr_to_latex("")

    with pytest.raises(ValueError):
        _expr_to_latex("   ")


def test_expr_to_latex_parse_failure() -> None:
    """Test parsing failure in _expr_to_latex."""
    with pytest.raises(LatexRenderError):
        _expr_to_latex("x ** -")


def test_expr_to_latex_no_sympy() -> None:
    """Test _expr_to_latex behavior when SymPy is unavailable."""
    with patch("sidekick.latex_renderer._SYMPY_AVAILABLE", False):
        with pytest.raises(LatexRenderError, match="sympy is not installed"):
            _expr_to_latex("x**2")


def test_make_label_widget_success(qapp: Any) -> None:
    """Test creating a label widget successfully."""
    label = _make_label_widget("test_text", monospace=True)
    assert isinstance(label, QLabel)
    assert label.text() == "test_text"
    assert label.objectName() == "SidekickLatexLabel"


def test_make_label_widget_no_qt() -> None:
    """Test _make_label_widget when Qt is unavailable."""
    with patch("sidekick.latex_renderer._QT_AVAILABLE", False):
        with pytest.raises(LatexRenderError, match="PyQt6 is not installed"):
            _make_label_widget("test_text")


from sidekick.latex_renderer import _MATPLOTLIB_AVAILABLE


def test_render_latex_label_success(qapp: Any) -> None:
    """Test rendering latex label successfully."""
    label = render_latex_label(r"x^{2}")
    assert isinstance(label, QLabel)
    assert label.toolTip() == r"LaTeX: x^{2}"
    if _MATPLOTLIB_AVAILABLE:
        assert label.pixmap() is not None
        assert not label.text()
    else:
        assert label.text() == r"$  x^{2}  $"


def test_render_latex_label_invalid_input() -> None:
    """Test input validation for render_latex_label."""
    with pytest.raises(TypeError):
        render_latex_label(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        render_latex_label("")

    with pytest.raises(ValueError):
        render_latex_label("   ")


def test_render_expr_label_success(qapp: Any) -> None:
    """Test rendering expression label successfully."""
    label = render_expr_label("x**2")
    assert isinstance(label, QLabel)
    if _MATPLOTLIB_AVAILABLE:
        assert label.pixmap() is not None
        assert not label.text()
    else:
        assert "x^{2}" in label.text()


def test_render_expr_label_invalid_input() -> None:
    """Test input validation for render_expr_label."""
    with pytest.raises(TypeError):
        render_expr_label(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        render_expr_label("")

    with pytest.raises(ValueError):
        render_expr_label("   ")
