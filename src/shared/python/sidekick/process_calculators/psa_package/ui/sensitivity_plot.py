import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..psa_model import (
    DEFAULT_COMPONENTS,
    ComponentData,
    calculate_o2_safety_analysis,
    calculate_sensitivity,
)

__all__ = [
    "MplCanvas",
    "SensitivityPlotWidget",
]

matplotlib.use("QtAgg")


class MplCanvas(FigureCanvas):
    """Matplotlib canvas widget for embedding in PyQt6."""

    def __init__(
        self, parent: QWidget | None = None, width: float = 8, height: float = 6
    ) -> None:
        if width is None:
            raise ValueError("width must be provided")
        self.fig = Figure(figsize=(width, height), dpi=100)

        from shared.python.theme.integration import get_theme_manager
        from shared.python.theme.matplotlib_style import apply_plot_theme

        _tm = get_theme_manager()
        apply_plot_theme(self.fig, _tm.get_current_colors())
        _tm.themeChanged.connect(
            lambda name: apply_plot_theme(
                self.fig, _tm.get_theme_colors(name) or _tm.get_current_colors()
            )
        )

        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


class SensitivityPlotWidget(QWidget):
    """Widget for sensitivity analysis plots."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Controls
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(
            [
                "H2 Recovery vs Recycle",
                "Net Product vs Recycle",
                "O2 Safety Analysis",
                "3D Recovery Surface",
                "Contour Map",
            ]
        )
        controls_layout.addWidget(self.plot_type_combo)

        # Line/Marker options
        controls_layout.addWidget(QLabel("  "))
        self.show_lines_check = QCheckBox("Lines")
        self.show_lines_check.setChecked(True)
        controls_layout.addWidget(self.show_lines_check)

        self.show_markers_check = QCheckBox("Markers")
        self.show_markers_check.setChecked(False)
        controls_layout.addWidget(self.show_markers_check)

        # Number of points
        controls_layout.addWidget(QLabel("Points:"))
        self.num_points_spin = QSpinBox()
        self.num_points_spin.setRange(11, 101)
        self.num_points_spin.setValue(51)
        self.num_points_spin.setSingleStep(10)
        controls_layout.addWidget(self.num_points_spin)

        self.update_button = QPushButton("Update Plot")
        controls_layout.addWidget(self.update_button)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Canvas
        self.canvas = MplCanvas(self, width=10, height=7)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Connect signals
        self.update_button.clicked.connect(self._update_plot)
        self.plot_type_combo.currentIndexChanged.connect(self._update_plot)
        self.show_lines_check.stateChanged.connect(self._update_plot)
        self.show_markers_check.stateChanged.connect(self._update_plot)
        self.num_points_spin.valueChanged.connect(self._update_plot)

        # Store components for later use
        self._components: list[ComponentData] = list(DEFAULT_COMPONENTS)
        # Whether the displayed plot reflects the current components.
        self._plot_dirty: bool = True

    def set_components(self, components: list[ComponentData]) -> None:
        """Set component data for sensitivity calculations.

        Marks the plot dirty and, if the widget is currently visible, re-plots
        immediately so the curves reflect the latest inputs (issue #3105 F1 —
        previously this only stored the list and the tab kept showing the prior
        run's data). When hidden, the tab-change handler honours the dirty flag.
        """
        self._components = components
        self._plot_dirty = True
        if self.isVisible():
            self._update_plot()

    def _update_plot(self) -> None:
        """Update the sensitivity plot based on selected type."""
        self._plot_dirty = False
        plot_type = self.plot_type_combo.currentText()
        self.canvas.fig.clear()

        if plot_type == "H2 Recovery vs Recycle":
            self._plot_recovery_vs_recycle()
        elif plot_type == "Net Product vs Recycle":
            self._plot_product_vs_recycle()
        elif plot_type == "O2 Safety Analysis":
            self._plot_o2_safety()
        elif plot_type == "3D Recovery Surface":
            self._plot_3d_surface()
        elif plot_type == "Contour Map":
            self._plot_contour()

        self.canvas.draw()

    def _get_plot_style(self) -> tuple[str, str]:
        """Get the line and marker style based on checkbox states."""
        show_lines = self.show_lines_check.isChecked()
        show_markers = self.show_markers_check.isChecked()

        if show_lines and show_markers:
            return "-", "o"
        elif show_lines:
            return "-", ""
        elif show_markers:
            return "", "o"
        else:
            return "-", ""  # Default to lines

    def _plot_recovery_vs_recycle(self) -> None:
        """Plot H2 recovery vs recycle fractions."""
        num_points = self.num_points_spin.value()
        s2_range = np.linspace(0, 1, num_points)
        prod_range = np.array([0.0, 0.1, 0.2])

        sensitivity = calculate_sensitivity(
            s2_tail_recycle_range=s2_range,
            product_recycle_range=prod_range,
            components=self._components,
        )

        ax = self.canvas.fig.add_subplot(111)
        linestyle, marker = self._get_plot_style()
        markers = ["o", "s", "^"]

        for j, r_prod in enumerate(prod_range):
            ax.plot(
                s2_range * 100,
                sensitivity["h2_recovery"][:, j],
                linestyle=linestyle,
                marker=markers[j] if marker else "",
                markersize=5,
                linewidth=2,
                label=f"Product Recycle = {r_prod * 100:.0f}%",
            )

        ax.set_xlabel("Stage 2 Tail Recycle (%)")
        ax.set_ylabel("H2 Recovery (%)")
        ax.set_title("H2 Recovery vs Recycle Fractions")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_product_vs_recycle(self) -> None:
        """Plot net product vs recycle fractions."""
        num_points = self.num_points_spin.value()
        s2_range = np.linspace(0, 1, num_points)
        prod_range = np.array([0.0, 0.1, 0.2])

        sensitivity = calculate_sensitivity(
            s2_tail_recycle_range=s2_range,
            product_recycle_range=prod_range,
            components=self._components,
        )

        ax = self.canvas.fig.add_subplot(111)
        linestyle, marker = self._get_plot_style()
        markers = ["s", "^", "D"]

        for j, r_prod in enumerate(prod_range):
            ax.plot(
                s2_range * 100,
                sensitivity["net_product"][:, j],
                linestyle=linestyle,
                marker=markers[j] if marker else "",
                markersize=5,
                linewidth=2,
                label=f"Product Recycle = {r_prod * 100:.0f}%",
            )

        ax.set_xlabel("Stage 2 Tail Recycle (%)")
        ax.set_ylabel("Net Product Flow (SCFM)")
        ax.set_title("Net Product Flow vs Recycle Fractions")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_o2_safety(self) -> None:
        """Plot O2 safety analysis."""
        num_points = min(self.num_points_spin.value(), 51)  # Cap at 51 for O2 analysis
        inlet_o2_values = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float64)
        s1_removal_range = np.linspace(50.0, 95.0, num_points, dtype=np.float64)

        o2_analysis = calculate_o2_safety_analysis(
            inlet_o2_pcts=inlet_o2_values,
            stage1_o2_removal_range=s1_removal_range,
            components=self._components,
        )

        ax = self.canvas.fig.add_subplot(111)
        linestyle, marker = self._get_plot_style()
        markers_list = ["o", "s", "^", "D"]

        for j, inlet_o2 in enumerate(inlet_o2_values):
            ax.plot(
                s1_removal_range,
                o2_analysis["s2_tail_o2"][:, j],
                linestyle=linestyle,
                marker=markers_list[j] if marker else "",
                linewidth=2,
                markersize=5,
                label=f"Inlet O2 = {inlet_o2}%",
            )

        ax.axhline(y=2.0, color="red", linestyle="--", linewidth=2, label="DANGER (2%)")
        # Size the hazard band from the actual plotted data rather than reading
        # ax.get_ylim() before autoscale has settled (issue #3105 F3 — the old
        # code could draw the band to the default (0, 1) limit, making the
        # safety-critical shading wrong or invisible).
        data_max = float(np.nanmax(o2_analysis["s2_tail_o2"]))
        if not np.isfinite(data_max):
            data_max = 2.0
        band_top = max(data_max, 2.0) * 1.05
        ax.set_ylim(0.0, band_top)
        ax.fill_between(
            s1_removal_range,
            2.0,
            band_top,
            alpha=0.2,
            color="red",
        )
        ax.axvline(x=95, color="green", linestyle=":", alpha=0.7, label="Current (95%)")
        ax.axvline(
            x=80, color="orange", linestyle=":", alpha=0.7, label="Concern (80%)"
        )

        ax.set_xlabel("Stage 1 O2 Removal (%)")
        ax.set_ylabel("Stage 2 Tail O2 (%)")
        ax.set_title("O2 Safety Analysis: S2 Tail O2 vs S1 Removal")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim((50.0, 95.0))

    def _plot_3d_surface(self) -> None:
        """Plot 3D surface of H2 recovery."""
        num_points = self.num_points_spin.value()
        s2_range = np.linspace(0, 1, num_points)
        prod_range = np.linspace(0, 0.5, max(11, num_points // 2))

        sensitivity = calculate_sensitivity(
            s2_tail_recycle_range=s2_range,
            product_recycle_range=prod_range,
            components=self._components,
        )

        S2: np.ndarray
        PROD: np.ndarray
        S2, PROD = np.meshgrid(s2_range, prod_range, indexing="ij")

        ax = self.canvas.fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            S2 * 100, PROD * 100, sensitivity["h2_recovery"], cmap="viridis", alpha=0.8
        )
        ax.set_xlabel("S2 Tail Recycle (%)")
        ax.set_ylabel("Product Recycle (%)")
        ax.set_zlabel("H2 Recovery (%)")
        ax.set_title("H2 Recovery Surface")
        self.canvas.fig.colorbar(surf, ax=ax, shrink=0.5, label="H2 Recovery (%)")

    def _plot_contour(self) -> None:
        """Plot contour map of H2 recovery."""
        num_points = self.num_points_spin.value()
        s2_range = np.linspace(0, 1, num_points)
        prod_range = np.linspace(0, 0.5, max(11, num_points // 2))

        sensitivity = calculate_sensitivity(
            s2_tail_recycle_range=s2_range,
            product_recycle_range=prod_range,
            components=self._components,
        )

        S2: np.ndarray
        PROD: np.ndarray
        S2, PROD = np.meshgrid(s2_range, prod_range, indexing="ij")

        ax = self.canvas.fig.add_subplot(111)
        cs = ax.contourf(
            S2 * 100, PROD * 100, sensitivity["h2_recovery"], levels=20, cmap="viridis"
        )
        ax.contour(
            S2 * 100,
            PROD * 100,
            sensitivity["h2_recovery"],
            levels=[75, 77, 79, 80],
            colors="white",
            linewidths=1,
        )
        self.canvas.fig.colorbar(cs, ax=ax, label="H2 Recovery (%)")
        ax.set_xlabel("S2 Tail Recycle (%)")
        ax.set_ylabel("Product Recycle (%)")
        ax.set_title("H2 Recovery Contour Map")
        ax.plot([100], [0], "r*", markersize=15, label="Current Operation")
        ax.legend()
