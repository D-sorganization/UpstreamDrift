from __future__ import annotations

import logging
import typing

from PyQt6 import QtCore, QtWidgets

from ...plotting import GolfSwingPlotter, MplCanvas
from ...sim_widget import MuJoCoSimWidget

if typing.TYPE_CHECKING:
    from ..advanced_gui import AdvancedGolfAnalysisWindow

logger = logging.getLogger(__name__)


class PlottingTab(QtWidgets.QWidget):
    """Tab for advanced plotting and data visualization."""

    def __init__(
        self,
        sim_widget: MuJoCoSimWidget,
        main_window: AdvancedGolfAnalysisWindow,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.sim_widget = sim_widget
        self.main_window = main_window

        self.current_plot_canvas: MplCanvas | None = None

        self._setup_ui()
        self.update_joint_list()

    def _setup_ui(self) -> None:
        """Create the plotting interface."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Plot selection
        plot_group = QtWidgets.QGroupBox("Plot Type")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)

        self.plot_combo = QtWidgets.QComboBox()
        self.plot_combo.addItems(
            [
                "Summary Dashboard",
                "Joint Angles",
                "Joint Velocities",
                "Joint Torques",
                "Induced Accelerations",
                "Actuator Powers",
                "Energy Analysis",
                "Club Head Speed",
                "Club Head Trajectory (3D)",
                "Swing Plane Analysis",
                "Phase Diagram",
                "Torque Comparison",
                "Counterfactual Comparison",
            ]
        )
        plot_layout.addWidget(self.plot_combo)

        # Settings Stack
        self.settings_stack = QtWidgets.QStackedWidget()
        plot_layout.addWidget(self.settings_stack)

        # Empty page
        self.empty_page = QtWidgets.QWidget()
        self.settings_stack.addWidget(self.empty_page)

        # Joint selection (for phase diagram)
        self.joint_select_widget = QtWidgets.QWidget()
        js_layout = QtWidgets.QFormLayout(self.joint_select_widget)
        self.joint_select_combo = QtWidgets.QComboBox()
        js_layout.addRow("Joint:", self.joint_select_combo)
        self.settings_stack.addWidget(self.joint_select_widget)

        # Induced Accel Settings
        self.induced_widget = QtWidgets.QWidget()
        ind_layout = QtWidgets.QFormLayout(self.induced_widget)
        self.induced_source_combo = QtWidgets.QComboBox()
        self.induced_source_combo.addItems(["gravity", "actuator"])
        ind_layout.addRow("Source:", self.induced_source_combo)
        self.settings_stack.addWidget(self.induced_widget)

        # Counterfactual Settings
        self.cf_widget = QtWidgets.QWidget()
        cf_layout = QtWidgets.QFormLayout(self.cf_widget)
        self.cf_combo = QtWidgets.QComboBox()
        self.cf_combo.addItems(["ztcf", "zvcf"])
        cf_layout.addRow("Counterfactual:", self.cf_combo)
        self.settings_stack.addWidget(self.cf_widget)

        self.plot_combo.currentTextChanged.connect(self.on_plot_type_changed)

        self.generate_plot_btn = QtWidgets.QPushButton("Generate Plot")
        self.generate_plot_btn.clicked.connect(self.on_generate_plot)
        self.generate_plot_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2ca02c;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #238c23;
            }
        """
        )
        plot_layout.addWidget(self.generate_plot_btn)

        self.btn_advanced_dialog = QtWidgets.QPushButton("Open Advanced Analysis...")
        self.btn_advanced_dialog.clicked.connect(
            self.main_window.show_advanced_plots_dialog
        )
        self.btn_advanced_dialog.setStyleSheet(
            """
            QPushButton {
                background-color: #9467bd;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #8c564b;
            }
            """
        )
        plot_layout.addWidget(self.btn_advanced_dialog)

        layout.addWidget(plot_group)

        # Plot canvas container
        self.plot_container = QtWidgets.QWidget()
        self.plot_container_layout = QtWidgets.QVBoxLayout(self.plot_container)
        self.plot_container_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.plot_container)
        layout.addWidget(scroll_area, stretch=1)

    def update_joint_list(self) -> None:
        """Update the list of joints for phase diagram selection."""
        self.joint_select_combo.clear()
        if self.sim_widget.model is None:
            return

        import mujoco

        for i in range(self.sim_widget.model.njnt):
            name = mujoco.mj_id2name(
                self.sim_widget.model, mujoco.mjtObj.mjOBJ_JOINT, i
            )
            if name:
                self.joint_select_combo.addItem(name)
            else:
                self.joint_select_combo.addItem(f"Joint {i}")

    def on_plot_type_changed(self, plot_type: str) -> None:
        """Handle plot type selection change."""
        if plot_type == "Phase Diagram":
            self.settings_stack.setCurrentWidget(self.joint_select_widget)
            if self.joint_select_combo.count() == 0:
                self.update_joint_list()
        elif plot_type == "Induced Accelerations":
            self.settings_stack.setCurrentWidget(self.induced_widget)
        elif plot_type == "Counterfactual Comparison":
            self.settings_stack.setCurrentWidget(self.cf_widget)
        else:
            self.settings_stack.setCurrentWidget(self.empty_page)

    def on_generate_plot(self) -> None:
        """Generate the selected plot."""
        recorder = self.sim_widget.get_recorder()

        if recorder.get_num_frames() == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No recorded data available. Please record some data first.",
            )
            return

        # Clear existing plot
        if self.current_plot_canvas is not None:
            self.plot_container_layout.removeWidget(self.current_plot_canvas)
            self.current_plot_canvas.deleteLater()
            self.current_plot_canvas = None

        # Create new canvas
        canvas = MplCanvas(width=8, height=6, dpi=100)

        # We should pass joint names if possible
        joint_names = []
        if self.sim_widget.model:
            import mujoco

            for i in range(self.sim_widget.model.njnt):
                name = mujoco.mj_id2name(
                    self.sim_widget.model, mujoco.mjtObj.mjOBJ_JOINT, i
                )
                joint_names.append(name or f"Joint {i}")

        plotter = GolfSwingPlotter(recorder)

        # Generate appropriate plot
        plot_type = self.plot_combo.currentText()

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            if plot_type == "Summary Dashboard":
                plotter.plot_summary_dashboard(canvas.fig)
            elif plot_type == "Joint Angles":
                plotter.plot_joint_angles(canvas.fig)
            elif plot_type == "Joint Velocities":
                plotter.plot_joint_velocities(canvas.fig)
            elif plot_type == "Joint Torques":
                plotter.plot_joint_torques(canvas.fig)
            elif plot_type == "Actuator Powers":
                plotter.plot_actuator_powers(canvas.fig)
            elif plot_type == "Energy Analysis":
                plotter.plot_energy_analysis(canvas.fig)
            elif plot_type == "Club Head Speed":
                plotter.plot_club_head_speed(canvas.fig)
            elif plot_type == "Club Head Trajectory (3D)":
                plotter.plot_club_head_trajectory(canvas.fig)
            elif plot_type == "Swing Plane Analysis":
                plotter.plot_swing_plane(canvas.fig)
            elif plot_type == "Phase Diagram":
                joint_idx = self.joint_select_combo.currentIndex()
                plotter.plot_phase_diagram(canvas.fig, joint_idx)
            elif plot_type == "Torque Comparison":
                plotter.plot_torque_comparison(canvas.fig)
            elif plot_type == "Induced Accelerations":
                source = self.induced_source_combo.currentText()
                plotter.plot_induced_acceleration(canvas.fig, source)
            elif plot_type == "Counterfactual Comparison":
                cf_name = self.cf_combo.currentText()
                plotter.plot_counterfactual_comparison(canvas.fig, cf_name)

            canvas.draw()
            self.current_plot_canvas = canvas
            self.plot_container_layout.addWidget(canvas)

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Plot Error",
                f"Error generating plot: {e!s}",
            )
            logger.error("Plot generation failed: %s", e)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
