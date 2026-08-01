"""LaTeX rendering helpers for the Sidekick calculator tab (issue #2934).

Provides a Qt-backed renderer that turns LaTeX math strings into pixel maps
for display in a :class:`~PyQt6.QtWidgets.QLabel`.  Falls back gracefully
to a monospace plain-text label when PyQt6 or SymPy is not available.

Design
------
- **DbC**: public functions validate inputs and raise ``TypeError``/
  ``ValueError`` on violation.
- **LOD**: Qt widget internals are accessed only via the returned widget;
  callers receive a ``QWidget`` they can embed directly.
- **DRY**: both the plain-text and rich-math paths share a common
  ``_make_label_widget`` helper.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — module is always importable even without Qt / SymPy.
# ---------------------------------------------------------------------------

try:
    from PyQt6 import QtWidgets

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False

try:
    from sympy import latex as _sympy_latex
    from sympy import sympify as _sympify

    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False

try:
    import numpy as _np
    from matplotlib.mathtext import MathTextParser as _MathTextParser

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


class LatexRenderError(RuntimeError):
    """Raised when a LaTeX render operation fails."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _expr_to_latex(expr_str: str) -> str:
    """Convert an expression string to LaTeX via SymPy.

    Args:
        expr_str: A mathematical expression string.

    Returns:
        LaTeX representation of the expression.

    Raises:
        LatexRenderError: If SymPy is unavailable or parsing fails.
        TypeError: If *expr_str* is not a string.
        ValueError: If *expr_str* is empty.
    """
    if not isinstance(expr_str, str):
        raise TypeError(f"expr_str must be str, got {type(expr_str).__name__!r}")
    if not expr_str.strip():
        raise ValueError("expr_str must not be empty")
    if not _SYMPY_AVAILABLE:
        raise LatexRenderError(
            "sympy is not installed; install it with: pip install sympy"
        )
    try:
        expr = _sympify(expr_str, evaluate=True)
        return str(_sympy_latex(expr))
    except Exception as exc:
        raise LatexRenderError(f"Cannot render {expr_str!r} to LaTeX: {exc}") from exc


def _make_label_widget(
    text: str,
    *,
    monospace: bool = False,
    parent: Any = None,
) -> Any:
    """Create a ``QLabel`` with *text*.

    Args:
        text: Label text.
        monospace: If ``True`` use a monospace font.
        parent: Optional Qt parent widget.

    Returns:
        A :class:`~PyQt6.QtWidgets.QLabel`.

    Raises:
        LatexRenderError: If Qt is not available.
    """
    if not _QT_AVAILABLE:
        raise LatexRenderError(
            "PyQt6 is not installed; install it with: pip install PyQt6"
        )
    from PyQt6.QtCore import Qt

    label = QtWidgets.QLabel(text, parent)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    if monospace:
        from PyQt6.QtGui import QFont

        font = QFont("Courier")
        font.setStyleHint(QFont.StyleHint.Monospace)
        label.setFont(font)
    label.setObjectName("SidekickLatexLabel")
    return label


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_latex_label(
    latex_str: str,
    *,
    parent: Any = None,
) -> Any:
    """Create a Qt label that displays *latex_str* as rich text.

    Because Qt does not natively render LaTeX, we display the raw LaTeX
    string in a styled monospace ``QLabel``.  A future upgrade can swap this
    out for ``matplotlib.backends.backend_qtagg`` rendering or MathJax in a
    ``QWebEngineView`` without changing the caller interface.

    Args:
        latex_str: A LaTeX math string (e.g. ``r"x^{2}"``).
        parent: Optional Qt parent widget.

    Returns:
        A :class:`~PyQt6.QtWidgets.QLabel` whose text is *latex_str*.

    Raises:
        TypeError: If *latex_str* is not a string.
        ValueError: If *latex_str* is empty.
        LatexRenderError: If Qt is not available.

    Example::

        label = render_latex_label(r"x^{2} + \\frac{1}{x}")
        layout.addWidget(label)
    """
    if not isinstance(latex_str, str):
        raise TypeError(f"latex_str must be str, got {type(latex_str).__name__!r}")
    if not latex_str.strip():
        raise ValueError("latex_str must not be empty")
    if not _QT_AVAILABLE:
        raise LatexRenderError(
            "PyQt6 is not installed; install it with: pip install PyQt6"
        )

    label = _make_label_widget("", parent=parent)
    label.setToolTip(f"LaTeX: {latex_str}")

    if _MATPLOTLIB_AVAILABLE:
        try:
            clean_formula = latex_str.strip()
            if clean_formula.startswith("$") and clean_formula.endswith("$"):
                clean_formula = clean_formula[1:-1].strip()

            parser = _MathTextParser("agg")
            res = parser.parse(clean_formula, dpi=120)
            arr = _np.asarray(res.image)
            h, w = arr.shape

            if h > 0 and w > 0:
                palette = label.palette()
                fg_color = palette.color(label.foregroundRole())
                r, g, b = fg_color.red(), fg_color.green(), fg_color.blue()

                rgba = _np.zeros((h, w, 4), dtype=_np.uint8)
                rgba[..., 0] = r
                rgba[..., 1] = g
                rgba[..., 2] = b
                rgba[..., 3] = arr

                from PyQt6.QtGui import QImage, QPixmap

                qimg = QImage(rgba.data, w, h, QImage.Format.Format_RGBA8888)
                qimg._rgba_buffer = rgba  # Keep reference alive
                pm = QPixmap.fromImage(qimg)
                label.setPixmap(pm)
                _log.debug("render_latex_label(%r) rendered as QPixmap", latex_str)
                return label
        except Exception as exc:
            _log.warning(
                "Could not render LaTeX %r as QPixmap, falling back: %s",
                latex_str,
                exc,
            )

    display_text = f"$  {latex_str}  $"
    label.setText(display_text)
    from PyQt6.QtGui import QFont

    font = QFont("Courier")
    font.setStyleHint(QFont.StyleHint.Monospace)
    label.setFont(font)
    _log.debug("render_latex_label(%r) fallback to monospace text", latex_str)
    return label


def render_expr_label(
    expr_str: str,
    *,
    parent: Any = None,
) -> Any:
    """Parse *expr_str* with SymPy and render its LaTeX form in a Qt label.

    Combines :func:`~sidekick.symbolic_engine.symbolic_to_latex` and
    :func:`render_latex_label` into a single call.

    Args:
        expr_str: A mathematical expression string (e.g. ``"x**2 / (x + 1)"``).
        parent: Optional Qt parent widget.

    Returns:
        A :class:`~PyQt6.QtWidgets.QLabel` whose text is the LaTeX
        representation of *expr_str*.

    Raises:
        TypeError: If *expr_str* is not a string.
        ValueError: If *expr_str* is empty or cannot be parsed.
        LatexRenderError: If SymPy or Qt is not available, or conversion fails.
    """
    if not isinstance(expr_str, str):
        raise TypeError(f"expr_str must be str, got {type(expr_str).__name__!r}")
    if not expr_str.strip():
        raise ValueError("expr_str must not be empty")

    latex_str = _expr_to_latex(expr_str)
    return render_latex_label(latex_str, parent=parent)


def is_qt_available() -> bool:
    """Return ``True`` when PyQt6 is importable."""
    return _QT_AVAILABLE


def is_latex_ready() -> bool:
    """Return ``True`` when both PyQt6 and SymPy are importable."""
    return _QT_AVAILABLE and _SYMPY_AVAILABLE


__all__ = [
    "LatexRenderError",
    "is_latex_ready",
    "is_qt_available",
    "render_expr_label",
    "render_latex_label",
]
