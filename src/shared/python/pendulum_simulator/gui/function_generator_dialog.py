"""
FunctionGeneratorDialog — wraps the shared Signal Toolkit widgets
inside a tabbed dialog so users can either:

  1. **Design tab**: Draw points / freehand / type equations → polynomial fit
     (PolynomialGeneratorWidget from signal_toolkit.polynomial_generator)
  2. **Analyze tab**: Generate waveforms, filter, add noise → polynomial fit
     (SignalToolkitWidget from signal_toolkit.widget)

Both tabs emit compatible signals that are mapped to torque coefficients
for the selected pendulum joint.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ── Ensure shared/python (Tools/src/shared/python) is on sys.path ──────
# signal_toolkit needs two path entries:
#   1. shared/python   — so `import signal_toolkit` works
#   2. src/            — so `from shared.python.safe_eval import ...` works
#                        (signal_toolkit uses this internally)
_this_file = Path(__file__).resolve()

# ── Try to import the shared widgets ──────────────────────────────────────
_HAS_POLY_WIDGET = False
_HAS_SIGNAL_WIDGET = False
_IMPORT_ERRORS: list[str] = []
_PolyWidget: type | None = None
_SignalWidget: type | None = None

try:
    from signal_toolkit.polynomial_generator import PolynomialGeneratorWidget as _PW

    _PolyWidget = _PW
    _HAS_POLY_WIDGET = True
    logger.info("PolynomialGeneratorWidget loaded successfully")
except ImportError as _exc:  # pragma: no cover
    _IMPORT_ERRORS.append(f"PolynomialGeneratorWidget: {_exc}")
    logger.warning("PolynomialGeneratorWidget import failed: %s", _exc)

try:
    from signal_toolkit.widget import SignalToolkitWidget as _SW

    _SignalWidget = _SW
    _HAS_SIGNAL_WIDGET = True
    logger.info("SignalToolkitWidget loaded successfully")
except ImportError as _exc:  # pragma: no cover
    _IMPORT_ERRORS.append(f"SignalToolkitWidget: {_exc}")
    logger.warning("SignalToolkitWidget import failed: %s", _exc)

_WIDGET_AVAILABLE = _HAS_POLY_WIDGET or _HAS_SIGNAL_WIDGET
_WIDGET_IMPORT_ERROR = "; ".join(_IMPORT_ERRORS) if _IMPORT_ERRORS else None

if _WIDGET_AVAILABLE:
    logger.info(
        "Function generator available: poly=%s, signal=%s",
        _HAS_POLY_WIDGET,
        _HAS_SIGNAL_WIDGET,
    )
else:
    logger.error(  # pragma: no cover
        "Function generator UNAVAILABLE: %s (searched from %s)",
        _WIDGET_IMPORT_ERROR,
        _this_file,
    )


class FunctionGeneratorDialog(QDialog):
    """Tabbed dialog for torque profile design.

    Tab 1 — **Design**: PolynomialGeneratorWidget (draw / click / drag / equation → fit)
    Tab 2 — **Analyze**: SignalToolkitWidget (waveforms, filters, noise → fit)

    Both tabs allow "Apply to Joint" which emits ``torque_imported``.

    Signals
    -------
    torque_imported(str, list[float])
        Joint name and the fitted polynomial coefficients.
    """

    torque_imported = pyqtSignal(str, object)

    def __init__(
        self,
        parent: QWidget | None = None,
        joint_names: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Signal Toolkit — Torque Profile Designer")
        self.setModal(True)
        self.resize(1200, 800)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("QDialog { background: #12121c; color: #d0d0e8; }")

        self._joint_names = joint_names or ["Shoulder", "Wrist"]
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(8)

        # Header
        hdr = QLabel("Design or generate a torque waveform and apply it to a joint")
        hdr.setFont(QFont("Sans", 11))
        hdr.setStyleSheet("color: #9090b8; padding: 2px 0;")
        layout.addWidget(hdr)

        has_any_widget = _HAS_POLY_WIDGET or _HAS_SIGNAL_WIDGET

        if has_any_widget:
            tabs = QTabWidget()
            tabs.setStyleSheet(
                "QTabWidget::pane { border: 1px solid #2a2a48; background: #12121c; }"
                "QTabBar::tab { background: #1a1a30; color: #8080a0; padding: 8px 16px;"
                "  border-top-left-radius: 4px; border-top-right-radius: 4px; }"
                "QTabBar::tab:selected { background: #252548; color: #c0c0e0;"
                "  border-bottom: 2px solid #4888c8; }"
                "QTabBar::tab:hover:!selected { background: #202040; }"
            )

            # Tab 1: Polynomial Designer (draw / click / equation → fit)
            if _HAS_POLY_WIDGET and _PolyWidget is not None:
                self._poly_widget = _PolyWidget(use_builtin_theme=False)
                self._poly_widget.set_joints(self._joint_names)
                self._poly_widget.polynomial_generated.connect(self._on_signal_applied)
                tabs.addTab(self._poly_widget, "🎨 Design (Draw / Click / Equation)")

            # Tab 2: Signal Processing (generate waveform → filter → fit → apply)
            if _HAS_SIGNAL_WIDGET and _SignalWidget is not None:
                self._signal_widget = _SignalWidget(use_builtin_theme=False)
                self._signal_widget.set_joints(self._joint_names)
                self._signal_widget.signal_generated.connect(self._on_signal_applied)
                tabs.addTab(self._signal_widget, "📊 Analyze (Waveforms / Filters)")

            layout.addWidget(tabs, stretch=1)
        else:
            from PyQt6.QtWidgets import QTextEdit

            error_detail = _WIDGET_IMPORT_ERROR or "Unknown reason"
            note = QTextEdit()
            note.setReadOnly(True)
            note.setPlainText(
                f"⚠ Signal Toolkit widgets not available.\n\n"
                f"Reason: {error_detail}\n\n"
                f"Searched from: {_this_file}\n\n"
                "Ensure the signal_toolkit package is on the Python path.\n"
                "You can select and copy this text for troubleshooting."
            )
            note.setStyleSheet(
                "QTextEdit { color: #e0a060; background: #1a1a2e;"
                " padding: 8px; font-size: 12px; border: 1px solid #404060; }"
            )
            layout.addWidget(note)

        # Close button
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_signal_applied(self, joint_name: str, coefficients: list) -> None:
        """Handle signal emissions from either widget."""
        if not (joint_name is not None):
            raise ValueError("joint_name must be provided")
        logger.info("Signal applied to %s: %s", joint_name, coefficients)
        # Normalize joint name to lowercase for consistency
        self.torque_imported.emit(joint_name.lower(), list(coefficients))
        self.accept()
