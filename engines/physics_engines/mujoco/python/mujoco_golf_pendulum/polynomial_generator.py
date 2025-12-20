"""Polynomial Function Generator Module.

This module provides a visual interface for generating 6th-order polynomial functions
for joint control. It allows users to:
- Draw trends manually
- Add control points
- Input equations
- Drag/manipulate curves
- Fit polynomials to the visual data
"""

from __future__ import annotations

import logging
import sys
from functools import partial

import matplotlib
import numpy as np
import sympy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtWidgets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for PyQt6."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
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


class PolynomialGeneratorWidget(QtWidgets.QWidget):
    """Widget for visually generating polynomial functions."""

    # Signals
    polynomial_generated = QtCore.pyqtSignal(str, list)  # joint_name, coefficients

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the widget."""
        super().__init__(parent)

        self.setWindowTitle("Polynomial Function Generator")
        self.resize(1000, 700)

        # State
        self.joint_names: list[str] = []
        self.current_points: list[tuple[float, float]] = []
        self.drawn_points: list[tuple[float, float]] = []
        self.polynomial_coeffs: np.ndarray | None = None
        self.dragging_curve = False
        self.drag_start_pos: tuple[float, float] | None = None
        self.drag_start_coeffs: np.ndarray | None = None
        self.drag_start_points: list[tuple[float, float]] = []
        self.mode = "view"  # view, draw, add_points, drag

        # UI Setup
        self._setup_ui()
        self._setup_connections()

        # Initial plot
        self._update_plot()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        layout = QtWidgets.QHBoxLayout(self)

        # Left Panel: Controls
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_panel.setFixedWidth(300)

        # Joint Selection
        joint_group = QtWidgets.QGroupBox("Target Joint")
        joint_layout = QtWidgets.QVBoxLayout(joint_group)
        self.joint_combo = QtWidgets.QComboBox()
        self.joint_combo.addItems(["Joint 1", "Joint 2", "Joint 3"])  # Defaults
        joint_layout.addWidget(self.joint_combo)
        left_layout.addWidget(joint_group)

        # Plot Settings (Scale)
        scale_group = QtWidgets.QGroupBox("Plot Scale")
        scale_layout = QtWidgets.QGridLayout(scale_group)

        self.x_min_spin = self._create_spinbox(-100, 100, 0, "X Min")
        self.x_max_spin = self._create_spinbox(-100, 100, 10, "X Max")
        self.y_min_spin = self._create_spinbox(-1000, 1000, -10, "Y Min")
        self.y_max_spin = self._create_spinbox(-1000, 1000, 10, "Y Max")

        scale_layout.addWidget(QtWidgets.QLabel("X Range:"), 0, 0)
        scale_layout.addWidget(self.x_min_spin, 0, 1)
        scale_layout.addWidget(self.x_max_spin, 0, 2)
        scale_layout.addWidget(QtWidgets.QLabel("Y Range:"), 1, 0)
        scale_layout.addWidget(self.y_min_spin, 1, 1)
        scale_layout.addWidget(self.y_max_spin, 1, 2)

        self.apply_scale_btn = QtWidgets.QPushButton("Apply Scale")
        scale_layout.addWidget(self.apply_scale_btn, 2, 0, 1, 3)

        left_layout.addWidget(scale_group)

        # Input Methods
        input_group = QtWidgets.QGroupBox("Input Method")
        input_layout = QtWidgets.QVBoxLayout(input_group)

        self.mode_group = QtWidgets.QButtonGroup(self)

        self.btn_equation = QtWidgets.QRadioButton("Equation")
        self.btn_draw = QtWidgets.QRadioButton("Draw Line")
        self.btn_points = QtWidgets.QRadioButton("Add Points")
        self.btn_drag = QtWidgets.QRadioButton("Drag Trend")

        self.mode_group.addButton(self.btn_equation)
        self.mode_group.addButton(self.btn_draw)
        self.mode_group.addButton(self.btn_points)
        self.mode_group.addButton(self.btn_drag)

        self.btn_points.setChecked(True)  # Default
        self.mode = "add_points"

        input_layout.addWidget(self.btn_equation)
        self.equation_input = QtWidgets.QLineEdit()
        self.equation_input.setPlaceholderText("e.g. 0.5*x**2 + 2*x")
        self.equation_input.setEnabled(False)
        input_layout.addWidget(self.equation_input)
        self.generate_eq_btn = QtWidgets.QPushButton("Generate from Equation")
        self.generate_eq_btn.setEnabled(False)
        input_layout.addWidget(self.generate_eq_btn)

        input_layout.addWidget(self.btn_draw)
        input_layout.addWidget(self.btn_points)
        input_layout.addWidget(self.btn_drag)

        left_layout.addWidget(input_group)

        # Actions
        action_group = QtWidgets.QGroupBox("Actions")
        action_layout = QtWidgets.QVBoxLayout(action_group)

        self.clear_btn = QtWidgets.QPushButton("Clear Points")
        self.fit_btn = QtWidgets.QPushButton("Fit Polynomial (6th Order)")
        self.fit_btn.setStyleSheet(
            "font-weight: bold; background-color: #4CAF50; color: white;"
        )

        action_layout.addWidget(self.clear_btn)
        action_layout.addWidget(self.fit_btn)
        left_layout.addWidget(action_group)

        # Results
        result_group = QtWidgets.QGroupBox("Result")
        result_layout = QtWidgets.QVBoxLayout(result_group)
        self.result_text = QtWidgets.QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(100)
        result_layout.addWidget(self.result_text)
        left_layout.addWidget(result_group)

        left_layout.addStretch()
        layout.addWidget(left_panel)

        # Right Panel: Plot
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.canvas, stretch=1)

    def _create_spinbox(
        self, min_val: float, max_val: float, val: float, tooltip: str
    ) -> QtWidgets.QDoubleSpinBox:
        """Create a configured double spin box."""
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(val)
        spin.setToolTip(tooltip)
        return spin

    def _setup_connections(self) -> None:
        """Setup signal-slot connections."""
        self.apply_scale_btn.clicked.connect(self._update_plot)
        self.clear_btn.clicked.connect(self._clear_data)
        self.fit_btn.clicked.connect(self._fit_polynomial)
        self.generate_eq_btn.clicked.connect(self._generate_from_equation)

        self.btn_equation.toggled.connect(partial(self._set_mode, "equation"))
        self.btn_draw.toggled.connect(partial(self._set_mode, "draw"))
        self.btn_points.toggled.connect(partial(self._set_mode, "add_points"))
        self.btn_drag.toggled.connect(partial(self._set_mode, "drag"))

        # Matplotlib events
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)

    def _set_mode(self, mode: str, checked: bool) -> None:
        """Set the current interaction mode."""
        if not checked:
            return
        self.mode = mode
        self.equation_input.setEnabled(mode == "equation")
        self.generate_eq_btn.setEnabled(mode == "equation")
        logger.info(f"Mode set to: {mode}")

    def _update_plot(self) -> None:
        """Redraw the plot with current data."""
        self.canvas.axes.clear()
        self.canvas.axes.grid(True)
        self.canvas.axes.set_title("Joint Function Generator")
        self.canvas.axes.set_xlabel("Time / Input")
        self.canvas.axes.set_ylabel("Value")

        # Set limits
        self.canvas.axes.set_xlim(self.x_min_spin.value(), self.x_max_spin.value())
        self.canvas.axes.set_ylim(self.y_min_spin.value(), self.y_max_spin.value())

        # Plot points
        if self.current_points:
            xs, ys = zip(*self.current_points, strict=True)
            self.canvas.axes.scatter(xs, ys, c="red", marker="o", label="Points")

        # Plot drawn line
        if self.drawn_points:
            dx, dy = zip(*self.drawn_points, strict=True)
            self.canvas.axes.plot(dx, dy, "g--", alpha=0.5, label="Drawn Path")

        # Plot fitted polynomial
        if self.polynomial_coeffs is not None:
            x_range = np.linspace(self.x_min_spin.value(), self.x_max_spin.value(), 500)
            poly_func = np.poly1d(self.polynomial_coeffs)
            y_poly = poly_func(x_range)
            self.canvas.axes.plot(
                x_range, y_poly, "b-", linewidth=2, label="Polynomial Fit"
            )

        self.canvas.axes.legend()
        self.canvas.draw()

    def _on_canvas_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """Handle mouse click events on the canvas."""
        if event.inaxes != self.canvas.axes:
            return

        if self.mode == "add_points":
            if event.button == 1:  # Left click
                self.current_points.append((event.xdata, event.ydata))
                self._update_plot()

        elif self.mode == "draw":
            if event.button == 1:
                self.drawn_points = [(event.xdata, event.ydata)]

        elif self.mode == "drag":
            if event.button == 1 and self.polynomial_coeffs is not None:
                self.dragging_curve = True
                self.drag_start_pos = (event.xdata, event.ydata)
                self.drag_start_coeffs = self.polynomial_coeffs.copy()
                self.drag_start_points = list(self.current_points)

    def _on_canvas_release(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """Handle mouse release events."""
        if self.mode == "draw" and self.drawn_points:
            # When drawing finishes, convert drawn path to points for fitting
            # Simplify drawn points to avoid overcrowding
            if len(self.drawn_points) > 20:
                indices = np.linspace(0, len(self.drawn_points) - 1, 20, dtype=int)
                sampled = [self.drawn_points[i] for i in indices]
                self.current_points.extend(sampled)
            else:
                self.current_points.extend(self.drawn_points)
            self.drawn_points = []
            self._update_plot()

        elif self.mode == "drag":
            if self.dragging_curve:
                self._display_results()
                joint = self.joint_combo.currentText()
                if self.polynomial_coeffs is not None:
                    self.polynomial_generated.emit(joint, list(self.polynomial_coeffs))

            self.dragging_curve = False
            self.drag_start_pos = None
            self.drag_start_coeffs = None
            self.drag_start_points = []

    def _on_canvas_motion(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """Handle mouse motion events."""
        if event.inaxes != self.canvas.axes:
            return

        if self.mode == "draw" and event.button == 1:
            self.drawn_points.append((event.xdata, event.ydata))
            # Optimization: only redraw every N points or use blitting?
            # For now, simple redraw is fine for low frequency
            if len(self.drawn_points) % 5 == 0:
                self._update_plot()

        elif self.mode == "drag" and self.dragging_curve and self.drag_start_pos:
            # Horizontal shift not implemented (dx unused)
            # dx = event.xdata - self.drag_start_pos[0]
            dy = event.ydata - self.drag_start_pos[1]

            # Update polynomial coefficients (shift)
            # Vertical shift is easy: just add dy to the constant term (last coeff)
            # Horizontal shift is harder for general polynomial coefficients
            # For now, let's just do vertical shift as it's safer visually

            if self.drag_start_coeffs is not None:
                new_coeffs = self.drag_start_coeffs.copy()
                # poly1d coeffs are [cn, ..., c1, c0]
                new_coeffs[-1] += dy

                self.polynomial_coeffs = new_coeffs

                # Sync points with vertical shift
                if self.drag_start_points:
                    self.current_points = [
                        (x, y + dy) for x, y in self.drag_start_points
                    ]

                self._update_plot()

    def _clear_data(self) -> None:
        """Clear all points and fits."""
        self.current_points = []
        self.drawn_points = []
        self.polynomial_coeffs = None
        self.result_text.clear()
        self._update_plot()

    def _fit_polynomial(self) -> None:
        """Fit a 6th order polynomial to the current points."""
        if len(self.current_points) < 7:
            QtWidgets.QMessageBox.warning(
                self, "Insufficient Data", "Need at least 7 points for a 6th order fit."
            )
            return

        try:
            xs, ys = zip(*self.current_points, strict=True)
            # Fit 6th order
            self.polynomial_coeffs = np.polyfit(xs, ys, 6)
            self._update_plot()
            self._display_results()

            # Emit signal
            joint = self.joint_combo.currentText()
            self.polynomial_generated.emit(joint, list(self.polynomial_coeffs))

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Fit Error", str(e))
            logger.error(f"Fitting error: {e}")

    def _generate_from_equation(self) -> None:
        """Generate points from the user-provided equation."""
        eq_str = self.equation_input.text()
        if not eq_str:
            return

        try:
            x = sympy.symbols("x")
            # Parse equation safely
            expr = sympy.sympify(eq_str)

            # Generate points
            x_vals = np.linspace(self.x_min_spin.value(), self.x_max_spin.value(), 20)
            f_lambdified = sympy.lambdify(x, expr, "numpy")
            y_vals = f_lambdified(x_vals)

            # Check for complex results or errors
            if np.iscomplexobj(y_vals):
                raise ValueError("Equation resulted in complex numbers")

            self.current_points = list(zip(x_vals, y_vals, strict=True))
            self._update_plot()
            # Auto fit
            self._fit_polynomial()

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Equation Error", f"Invalid equation: {e}"
            )

    def _display_results(self) -> None:
        """Display the polynomial coefficients."""
        if self.polynomial_coeffs is None:
            return

        # Format as string
        terms = []
        order = len(self.polynomial_coeffs) - 1
        for i, c in enumerate(self.polynomial_coeffs):
            power = order - i
            if abs(c) > 1e-10:
                if power == 0:
                    terms.append(f"{c:.4f}")
                elif power == 1:
                    terms.append(f"{c:.4f}*x")
                else:
                    terms.append(f"{c:.4f}*x^{power}")

        poly_str = " + ".join(terms).replace("+ -", "- ")
        self.result_text.setText(
            f"Polynomial:\n{poly_str}\n\nCoefficients:\n{self.polynomial_coeffs}"
        )

    def set_joints(self, joints: list[str]) -> None:
        """Set the list of available joints."""
        self.joint_names = joints
        self.joint_combo.clear()
        self.joint_combo.addItems(joints)


def main() -> None:
    """Run the widget as a standalone application."""
    app = QtWidgets.QApplication(sys.argv)
    window = PolynomialGeneratorWidget()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
