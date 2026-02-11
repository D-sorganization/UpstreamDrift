"""PyQt6 GUI Widget for Signal Processing Toolkit.

This module provides a comprehensive visual interface for signal generation,
fitting, filtering, and analysis. It can be used standalone or integrated
into other applications.

Refactored: UI construction in ``signal_toolkit_ui_mixin.py``.
Signal processing logic in ``signal_toolkit_processing_mixin.py``.
"""

from __future__ import annotations

import logging
import sys

import numpy as np

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PyQt6 import QtWidgets
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import QApplication, QWidget

    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from src.shared.python.signal_toolkit.core import Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dark theme stylesheet
# ---------------------------------------------------------------------------
DARK_STYLESHEET = """
QWidget {
    background-color: #2b2b2b;
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}
QGroupBox {
    border: 1px solid #444;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: bold;
    color: #e0e0e0;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QPushButton {
    background-color: #3d3d3d;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px 12px;
    color: #fff;
}
QPushButton:hover {
    background-color: #4d4d4d;
    border: 1px solid #666;
}
QPushButton:pressed {
    background-color: #2b2b2b;
}
QPushButton:disabled {
    background-color: #333;
    color: #666;
}
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
    background-color: #1e1e1e;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 4px;
    color: #fff;
}
QComboBox::drop-down {
    border: none;
}
QTextEdit {
    background-color: #1e1e1e;
    border: 1px solid #555;
    border-radius: 4px;
    color: #fff;
}
QTabWidget::pane {
    border: 1px solid #444;
    background: #2b2b2b;
}
QTabBar::tab {
    background: #333;
    color: #ccc;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: #0078d4;
    color: white;
}
QTabBar::tab:hover:!selected {
    background: #444;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #444;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px;
    margin: -5px 0;
    background: #0078d4;
    border-radius: 8px;
}
QScrollArea {
    border: none;
}
QLabel {
    color: #cccccc;
}
"""


if HAS_MATPLOTLIB and HAS_PYQT:
    from .signal_toolkit_processing_mixin import SignalToolkitProcessingMixin
    from .signal_toolkit_ui_mixin import SignalToolkitUIMixin

    class MplCanvas(FigureCanvasQTAgg):
        """Matplotlib canvas for PyQt6."""

        def __init__(
            self,
            parent: QWidget | None = None,
            width: float = 5,
            height: float = 4,
            dpi: int = 100,
        ) -> None:
            """Initialize the canvas."""
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes = self.fig.add_subplot(111)
            super().__init__(self.fig)
            self.setParent(parent)
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            self.updateGeometry()

        def setup_dark_theme(self) -> None:
            """Apply dark theme to the plot."""
            self.fig.patch.set_facecolor("#2b2b2b")
            self.axes.set_facecolor("#1e1e1e")
            self.axes.tick_params(colors="#aaaaaa", which="both")
            self.axes.xaxis.label.set_color("#aaaaaa")
            self.axes.yaxis.label.set_color("#aaaaaa")
            self.axes.title.set_color("#ffffff")
            for spine in self.axes.spines.values():
                spine.set_edgecolor("#555555")
            self.axes.grid(
                True, color="#444444", linestyle="--", linewidth=0.5
            )

    class SignalToolkitWidget(  # type: ignore[misc]
        SignalToolkitProcessingMixin,
        SignalToolkitUIMixin,
        QWidget,
    ):
        """Comprehensive signal processing toolkit widget."""

        signal_generated = pyqtSignal(str, list)
        signal_updated = pyqtSignal(object)

        def __init__(self, parent: QWidget | None = None) -> None:
            """Initialize the widget."""
            super().__init__(parent)

            self.setWindowTitle("Signal Processing Toolkit")
            self.resize(1200, 800)
            self.setStyleSheet(DARK_STYLESHEET)

            # State
            self.current_signal: Signal | None = None
            self.original_signal: Signal | None = None
            self.derivative_signal: Signal | None = None
            self.integral_signal: Signal | None = None
            self.joint_names: list[str] = []

            # Default time array
            self.t_default = np.linspace(0, 10, 1000)

            # Build UI (from UIMixin) and wire connections (from ProcessingMixin)
            self._setup_ui()
            self._setup_connections()

            # Start with a default signal
            self._generate_default_signal()

    def main() -> None:
        """Run the widget as a standalone application."""
        app = QApplication(sys.argv)
        window = SignalToolkitWidget()
        window.show()
        sys.exit(app.exec())

else:
    # Stub class when dependencies are not available
    class SignalToolkitWidget:  # type: ignore[no-redef]
        """Stub class when PyQt6 or matplotlib is not available."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            msg = "SignalToolkitWidget requires PyQt6 and matplotlib"
            raise ImportError(msg)

    def main() -> None:
        """Stub main function."""
        logger.info("SignalToolkitWidget requires PyQt6 and matplotlib")


if __name__ == "__main__":
    main()
