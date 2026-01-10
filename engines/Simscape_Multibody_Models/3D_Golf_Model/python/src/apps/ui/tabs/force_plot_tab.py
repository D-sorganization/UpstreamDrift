"""Force plate visualization tab for C3D viewer.

Implements Task 2.3: Force-Plate Visualization per Phase 2 roadmap.
Provides GRF time-series plots and COP trajectory visualization.
"""

from PyQt6 import QtWidgets

from ...core.models import C3DDataModel
from ..widgets.mpl_canvas import MplCanvas


class ForcePlotTab(QtWidgets.QWidget):
    """Force plate GRF and COP visualization tab.

    Features:
    - GRF component time-series (Fx, Fy, Fz, Mx, My, Mz)
    - COP trajectory trace on ground plane
    - Multi-plate support
    """

    def __init__(self) -> None:
        """Initialize the force plot tab."""
        super().__init__()
        self.model: C3DDataModel | None = None
        self._force_plate_data: dict[int, dict[str, list[float]]] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        """Set up the user interface."""
        layout = QtWidgets.QVBoxLayout(self)

        # Top row: plate selection and controls
        control_row = QtWidgets.QHBoxLayout()

        control_row.addWidget(QtWidgets.QLabel("Force Plate:"))
        self.plate_combo = QtWidgets.QComboBox()
        self.plate_combo.currentIndexChanged.connect(self._update_plots)
        control_row.addWidget(self.plate_combo)

        control_row.addWidget(QtWidgets.QLabel("Component:"))
        self.component_combo = QtWidgets.QComboBox()
        self.component_combo.addItems(
            ["All Forces", "All Moments", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]
        )
        self.component_combo.currentIndexChanged.connect(self._update_plots)
        control_row.addWidget(self.component_combo)

        self.show_cop_checkbox = QtWidgets.QCheckBox("Show COP Trajectory")
        self.show_cop_checkbox.setChecked(True)
        self.show_cop_checkbox.toggled.connect(self._update_plots)
        control_row.addWidget(self.show_cop_checkbox)

        control_row.addStretch()
        layout.addLayout(control_row)

        # Main content area: splitter with plots
        splitter = QtWidgets.QSplitter()

        # Left: Time-series plot
        self.time_series_canvas = MplCanvas(self, width=6, height=4, dpi=100)
        splitter.addWidget(self.time_series_canvas)

        # Right: COP trajectory plot
        self.cop_canvas = MplCanvas(self, width=4, height=4, dpi=100)
        splitter.addWidget(self.cop_canvas)

        splitter.setSizes([600, 400])
        layout.addWidget(splitter)

        # Status bar
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.status_label)

    def update_from_model(self, model: C3DDataModel | None) -> None:
        """Update UI with data from the model.

        Args:
            model: The loaded C3D data model.
        """
        self.model = model
        self.plate_combo.clear()
        self._force_plate_data = {}

        if model is None:
            self.time_series_canvas.clear_axes()
            self.cop_canvas.clear_axes()
            self.status_label.setText("")
            return

        # Try to extract force plate data from analog channels
        if not self._load_force_plate_data():
            self.status_label.setText("No force plate channels detected")
            return

        # Populate plate selector
        plate_numbers = sorted(self._force_plate_data.keys())
        for plate_num in plate_numbers:
            self.plate_combo.addItem(f"Plate {plate_num}", plate_num)

        if plate_numbers:
            self.plate_combo.setCurrentIndex(0)
            self._update_plots()

        self.status_label.setText(f"Detected {len(plate_numbers)} force plate(s)")

    def _load_force_plate_data(self) -> bool:
        """Load force plate data from analog channels.

        Returns:
            True if force plate data was found.
        """
        if self.model is None:
            return False

        # Detect force plate channels from analog data
        analog_names = self.model.analog_names()

        # Standard force plate patterns
        import re

        pattern = re.compile(r"^(?:Force\.)?([FfMm])([xyzXYZ])(\d+)$")

        plates: dict[int, dict[str, list[float]]] = {}

        for name in analog_names:
            match = pattern.match(name)
            if match:
                force_or_moment = match.group(1).lower()
                axis = match.group(2).lower()
                plate_num = int(match.group(3))

                key = f"{force_or_moment}{axis}"  # 'fx', 'fy', etc.

                if plate_num not in plates:
                    plates[plate_num] = {}

                channel = self.model.analog.get(name)
                if channel is not None:
                    plates[plate_num][key] = list(channel.values)

        # Also try to get time data
        if self.model.analog_time is not None:
            for plate_num in plates:
                plates[plate_num]["time"] = list(self.model.analog_time)

        self._force_plate_data = plates
        return len(plates) > 0

    def _update_plots(self) -> None:
        """Update all plots based on current selection."""
        self._update_time_series()
        self._update_cop_trajectory()

    def _update_time_series(self) -> None:
        """Update GRF time-series plot."""
        plate_idx = self.plate_combo.currentData()
        if plate_idx is None or plate_idx not in self._force_plate_data:
            self.time_series_canvas.clear_axes()
            return

        data = self._force_plate_data[plate_idx]
        time = data.get("time", list(range(len(data.get("fx", [])))))

        component = self.component_combo.currentText()

        self.time_series_canvas.fig.clear()

        if component == "All Forces":
            axes = [self.time_series_canvas.add_subplot(3, 1, i + 1) for i in range(3)]
            for ax, key, label in zip(
                axes, ["fx", "fy", "fz"], ["Fx", "Fy", "Fz"], strict=True
            ):
                if key in data:
                    ax.plot(time, data[key], label=label)
                    ax.set_ylabel(f"{label} [N]")
                    ax.grid(True)
                    ax.legend()
            axes[-1].set_xlabel("Time [s]")
            axes[0].set_title(f"Plate {plate_idx} - Force Components")

        elif component == "All Moments":
            axes = [self.time_series_canvas.add_subplot(3, 1, i + 1) for i in range(3)]
            for ax, key, label in zip(
                axes, ["mx", "my", "mz"], ["Mx", "My", "Mz"], strict=True
            ):
                if key in data:
                    ax.plot(time, data[key], label=label)
                    ax.set_ylabel(f"{label} [N·m]")
                    ax.grid(True)
                    ax.legend()
            axes[-1].set_xlabel("Time [s]")
            axes[0].set_title(f"Plate {plate_idx} - Moment Components")

        else:
            # Single component
            key = component.lower()
            ax = self.time_series_canvas.add_subplot(111)
            if key in data:
                ax.plot(time, data[key], label=component)
                unit = "[N]" if key.startswith("f") else "[N·m]"
                ax.set_ylabel(f"{component} {unit}")
                ax.set_xlabel("Time [s]")
                ax.set_title(f"Plate {plate_idx} - {component}")
                ax.grid(True)
                ax.legend()

        self.time_series_canvas.fig.tight_layout()
        self.time_series_canvas.draw()  # type: ignore

    def _update_cop_trajectory(self) -> None:
        """Update COP trajectory plot."""
        if not self.show_cop_checkbox.isChecked():
            self.cop_canvas.clear_axes()
            return

        plate_idx = self.plate_combo.currentData()
        if plate_idx is None or plate_idx not in self._force_plate_data:
            self.cop_canvas.clear_axes()
            return

        data = self._force_plate_data[plate_idx]

        # Compute COP if we have forces and moments
        # Force components (fx, fy) available but not used for COP
        _ = data.get("fx", [])
        _ = data.get("fy", [])
        fz = data.get("fz", [])
        mx = data.get("mx", [])
        my = data.get("my", [])

        if not all([fz, mx, my]):
            self.cop_canvas.clear_axes()
            return

        import numpy as np

        fz_arr = np.array(fz)
        mx_arr = np.array(mx)
        my_arr = np.array(my)

        # COP calculation: COP_x = -My/Fz, COP_y = Mx/Fz
        min_force = 10.0  # [N] threshold
        valid = np.abs(fz_arr) > min_force

        cop_x = np.where(valid, -my_arr / fz_arr, np.nan)
        cop_y = np.where(valid, mx_arr / fz_arr, np.nan)

        self.cop_canvas.fig.clear()
        ax = self.cop_canvas.add_subplot(111)

        # Plot COP trajectory with color indicating time
        time = data.get("time")
        if time is not None:
            t_arr = np.array(time)
            scatter = ax.scatter(cop_x, cop_y, c=t_arr, cmap="viridis", s=2, alpha=0.7)
            self.cop_canvas.fig.colorbar(scatter, ax=ax, label="Time [s]")
        else:
            ax.plot(cop_x, cop_y, "b-", linewidth=0.5, alpha=0.7)

        # Mark start and end
        valid_indices = np.where(valid)[0]
        if len(valid_indices) > 0:
            start_idx = valid_indices[0]
            end_idx = valid_indices[-1]
            ax.plot(
                cop_x[start_idx], cop_y[start_idx], "go", markersize=10, label="Start"
            )
            ax.plot(cop_x[end_idx], cop_y[end_idx], "ro", markersize=10, label="End")

        ax.set_xlabel("COP X [m]")
        ax.set_ylabel("COP Y [m]")
        ax.set_title(f"Plate {plate_idx} - COP Trajectory")
        ax.set_aspect("equal")
        ax.grid(True)
        ax.legend()

        self.cop_canvas.fig.tight_layout()
        self.cop_canvas.draw()  # type: ignore
