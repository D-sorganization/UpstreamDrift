"""PyQt6 GUI components for the Training Controller dashboard.

Provides MainWindow, MainWidget, and SubmitDialog to interact with the
headless TrainingDashboardController.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.training import (
    Dataset,
    DatasetRegistry,
    JobId,
    TrainingConfig,
    TrainingFramework,
)

from .controller import TrainingDashboardController
from .view_model import DashboardModel, MetricSeries, ResourceSnapshot

# Setup Logger per PEP 8 & GEMINI.md
logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Sleek Dark QSS Theme
DARK_STYLE = """
QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: 13px;
}
QTableWidget {
    background-color: #252526;
    alternate-background-color: #1e1e1e;
    gridline-color: #2d2d2d;
    border: 1px solid #3e3e42;
    border-radius: 4px;
}
QTableWidget::item:selected {
    background-color: #0A84FF;
    color: #ffffff;
}
QHeaderView::section {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #3e3e42;
    padding: 6px;
    font-weight: bold;
}
QListWidget {
    background-color: #252526;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 4px;
}
QListWidget::item:hover {
    background-color: #2d2d2d;
}
QListWidget::item:selected {
    background-color: #0A84FF;
    color: #ffffff;
}
QTabWidget::pane {
    border: 1px solid #3e3e42;
    background-color: #1e1e1e;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #2d2d2d;
    color: #888888;
    padding: 8px 16px;
    margin-right: 4px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border: 1px solid #3e3e42;
    border-bottom: none;
}
QTabBar::tab:hover {
    background-color: #333333;
    color: #d4d4d4;
}
QTabBar::tab:selected {
    background-color: #1e1e1e;
    color: #ffffff;
    border-bottom: 2px solid #0A84FF;
}
QPushButton {
    background-color: #333333;
    border: 1px solid #444444;
    border-radius: 4px;
    color: #ffffff;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #444444;
    border-color: #555555;
}
QPushButton:pressed {
    background-color: #0A84FF;
    border-color: #0A84FF;
}
QPushButton:disabled {
    background-color: #222222;
    border-color: #333333;
    color: #666666;
}
QPushButton#submit-btn {
    background-color: #0A84FF;
    border-color: #0A84FF;
    font-weight: bold;
}
QPushButton#submit-btn:hover {
    background-color: #2997FF;
}
QLineEdit, QComboBox {
    background-color: #2d2d2d;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 6px;
    color: #ffffff;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #0A84FF;
}
QDialog {
    background-color: #1e1e1e;
}
"""


class MainWindow(QMainWindow):
    """Standalone/unified main window for the Training Controller."""

    def __init__(
        self,
        controller: TrainingDashboardController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Training Controller")
        self.setMinimumSize(1024, 720)
        self.setStyleSheet(DARK_STYLE)

        self.main_widget = MainWidget(controller, parent=self)
        self.setCentralWidget(self.main_widget)

    def closeEvent(self, event: Any) -> None:
        """Handle close event and run cleanup on widgets."""
        self.main_widget.cleanup()
        super().closeEvent(event)


class MainWidget(QWidget):
    """The central widget dashboard holding tabs, tables, and docks."""

    update_ui_signal = pyqtSignal()

    def __init__(
        self,
        controller: TrainingDashboardController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self._unsubscribe: Callable[[], None] | None = None

        self._init_layout()
        self._connect_signals()

        # Render first initial state
        self.update_ui()

    def _init_layout(self) -> None:
        main_h_layout = QHBoxLayout(self)
        main_h_layout.setContentsMargins(10, 10, 10, 10)
        main_h_layout.setSpacing(10)

        # Splitter to allow resizing the side panels
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_h_layout.addWidget(self.splitter)

        # Left Panel (Actions + Jobs)
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Actions Toolbar
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(8)

        self.submit_button = QPushButton("Submit Job", self)
        self.submit_button.setObjectName("submit-btn")
        self.submit_button.setToolTip("Queue a new training run configuration")

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setToolTip("Abort the selected active job")

        self.pause_button = QPushButton("Pause", self)
        self.pause_button.setToolTip("Pause execution of the running job")

        self.resume_button = QPushButton("Resume", self)
        self.resume_button.setToolTip("Resume execution of the paused job")

        actions_layout.addWidget(self.submit_button)
        actions_layout.addWidget(self.cancel_button)
        actions_layout.addWidget(self.pause_button)
        actions_layout.addWidget(self.resume_button)
        actions_layout.addStretch(1)
        left_layout.addLayout(actions_layout)

        # Job List Table
        self.job_table = QTableWidget(self)
        self.job_table.setColumnCount(6)
        self.job_table.setHorizontalHeaderLabels(
            ["Job ID", "Framework", "Status", "Dataset ID", "Elapsed", "Error"]
        )
        header = self.job_table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        self.job_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.job_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.job_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.job_table.setAlternatingRowColors(True)
        left_layout.addWidget(self.job_table)

        # Resource status strip at bottom left
        self.resource_label = QLabel("System resource monitoring unavailable", self)
        self.resource_label.setStyleSheet(
            "color: #888888; font-size: 12px; padding: 4px;"
        )
        left_layout.addWidget(self.resource_label)

        self.splitter.addWidget(left_widget)

        # Right Panel (Tabs + Dataset dock)
        right_splitter = QSplitter(Qt.Orientation.Vertical, self)

        # Tabs container
        self.tabs = QTabWidget(self)

        # Tab 1: Live Metrics Graph
        self.plot_stack = QStackedWidget(self)

        self.plot_placeholder = QLabel(
            "Select an active/completed job from the table\n"
            "to view real-time training metric curves.",
            self,
        )
        self.plot_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_placeholder.setStyleSheet("color: #666666; font-size: 14px;")
        self.plot_stack.addWidget(self.plot_placeholder)

        if HAS_MATPLOTLIB:
            self.plot_widget = QWidget(self)
            plot_layout = QVBoxLayout(self.plot_widget)
            plot_layout.setContentsMargins(5, 5, 5, 5)

            self.figure = Figure(figsize=(6, 4), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            plot_layout.addWidget(self.canvas)
            self.plot_stack.addWidget(self.plot_widget)
        else:
            self.plot_widget = QLabel("Matplotlib is unavailable.", self)
            self.plot_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.plot_stack.addWidget(self.plot_widget)

        self.tabs.addTab(self.plot_stack, "Live Metrics")

        # Tab 2: Job Summary & Config Details
        self.summary_scroll = QScrollArea(self)
        self.summary_scroll.setWidgetResizable(True)
        self.summary_label = QLabel("", self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.RichText)
        self.summary_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.summary_scroll.setWidget(self.summary_label)

        self.tabs.addTab(self.summary_scroll, "Job Detail Summary")
        right_splitter.addWidget(self.tabs)

        # Dataset Dock (as a collapsible/resizable list at the bottom right)
        dataset_dock = QWidget(self)
        dock_layout = QVBoxLayout(dataset_dock)
        dock_layout.setContentsMargins(0, 5, 0, 0)
        dock_layout.setSpacing(5)

        dock_title = QLabel("Dataset Library Dock", self)
        dock_title.setStyleSheet("font-weight: bold; color: #ffffff;")
        dock_layout.addWidget(dock_title)

        self.dataset_list = QListWidget(self)
        dock_layout.addWidget(self.dataset_list)
        right_splitter.addWidget(dataset_dock)

        right_splitter.setSizes([450, 200])
        self.splitter.addWidget(right_splitter)

        # Give left side more space default
        self.splitter.setSizes([600, 400])

    def _connect_signals(self) -> None:
        self.submit_button.clicked.connect(self._on_submit_clicked)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.pause_button.clicked.connect(self._on_pause_clicked)
        self.resume_button.clicked.connect(self._on_resume_clicked)

        self.job_table.itemSelectionChanged.connect(self._on_selection_changed)

        self.update_ui_signal.connect(self.update_ui)
        self._unsubscribe = self.controller.on_model_change(self._trigger_update)

    def _trigger_update(self) -> None:
        self.update_ui_signal.emit()

    def cleanup(self) -> None:
        """Unsubscribe from controller observers on closed tab."""
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except (ValueError, RuntimeError) as e:
                logger.debug(f"Failed to unsubscribe cleanly: {e}")
            self._unsubscribe = None

    def update_ui(self) -> None:
        """Update widgets state from the latest DashboardModel."""
        model = self.controller.current_model()

        # Update Job Table
        self.job_table.blockSignals(True)
        self.job_table.setRowCount(len(model.jobs))
        for row_idx, job in enumerate(model.jobs):
            self.job_table.setItem(row_idx, 0, QTableWidgetItem(job.job_id))
            self.job_table.setItem(row_idx, 1, QTableWidgetItem(job.framework))
            self.job_table.setItem(row_idx, 2, QTableWidgetItem(job.status))
            self.job_table.setItem(
                row_idx, 3, QTableWidgetItem(job.dataset_id or "None")
            )

            # Format elapsed time
            elapsed = job.elapsed_s
            if elapsed < 60:
                elapsed_str = f"{elapsed:.1f}s"
            else:
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                elapsed_str = f"{mins}m {secs}s"
            self.job_table.setItem(row_idx, 4, QTableWidgetItem(elapsed_str))
            self.job_table.setItem(
                row_idx, 5, QTableWidgetItem(job.error_message or "")
            )

        # Restore table selection based on controller.selected_job_id
        selected_job_id = model.selected_job_id
        if selected_job_id is not None:
            for r in range(self.job_table.rowCount()):
                item = self.job_table.item(r, 0)
                if item and item.text() == selected_job_id.value:
                    self.job_table.selectRow(r)
                    break
        else:
            self.job_table.clearSelection()
        self.job_table.blockSignals(False)

        # Update lifecycle button states
        self._update_button_states(model)

        # Update resource status strip
        self._update_resources(model.resources)

        # Update datasets list
        self._update_datasets()

        # Update right detail panel
        self._update_right_panel(model)

    def _update_button_states(self, model: DashboardModel) -> None:
        selected_row = model.selected_row
        if selected_row is None:
            self.cancel_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(False)
            return

        # Enable actions when a job is selected, letting the controller validate transitions
        self.cancel_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(True)

    def _update_resources(self, resources: ResourceSnapshot) -> None:
        if not resources.available:
            self.resource_label.setText("System resource monitoring unavailable")
            return

        text = f"CPU: {resources.cpu_percent:.1f}% | Memory: {resources.memory_percent:.1f}%"
        if resources.gpus:
            gpus_text = []
            for gpu in resources.gpus:
                util = (
                    f"{gpu.utilization_percent:.1f}%"
                    if gpu.utilization_percent is not None
                    else "N/A"
                )
                mem = f"{gpu.memory_used_mb}/{gpu.memory_total_mb} MB"
                gpus_text.append(
                    f"GPU {gpu.index} ({gpu.name}): Util {util}, Mem {mem}"
                )
            text += " | " + " | ".join(gpus_text)
        self.resource_label.setText(text)

    def _update_datasets(self) -> None:
        self.dataset_list.clear()
        datasets = self.controller.dataset_registry.list()
        for ds in datasets:
            self.dataset_list.addItem(f"{ds.name} ({ds.dataset_id})")

    def _update_right_panel(self, model: DashboardModel) -> None:
        selected_id = model.selected_job_id

        # Update Summary text
        if selected_id is None:
            self.summary_label.setText(
                "<div style='color: #888888; font-size: 14px; text-align: center; padding: 20px;'>"
                "Select a job to view details and summaries."
                "</div>"
            )
            self.plot_stack.setCurrentIndex(0)
            return

        # Summary Tab
        job = self.controller.scheduler.registry.get(selected_id)
        if job:
            elapsed = model.selected_row.elapsed_s if model.selected_row else 0.0
            html = f"""
            <div style="font-family: 'Segoe UI', Arial; padding: 10px; color: #d4d4d4;">
                <h3 style="color: #ffffff; margin-top: 0; border-bottom: 1px solid #3e3e42; padding-bottom: 5px;">
                    Job Details: {job.job_id.value}
                </h3>
                <table cellpadding="4" cellspacing="0" style="width: 100%;">
                    <tr><td style="color: #888888; width: 120px;"><b>Framework:</b></td><td>{job.config.framework.value.upper()}</td></tr>
                    <tr><td style="color: #888888;"><b>Status:</b></td><td>{job.status.value.upper()}</td></tr>
                    <tr><td style="color: #888888;"><b>Dataset ID:</b></td><td>{job.config.dataset_id or "None"}</td></tr>
                    <tr><td style="color: #888888;"><b>Elapsed Time:</b></td><td>{elapsed:.2f} seconds</td></tr>
            """
            if job.started_at is not None:
                dt = datetime.datetime.fromtimestamp(
                    job.started_at, tz=datetime.timezone.utc
                )
                html += f"<tr><td style='color: #888888;'><b>Started At:</b></td><td>{dt.strftime('%Y-%m-%d %H:%M:%S UTC')}</td></tr>"
            if job.completed_at is not None:
                dt = datetime.datetime.fromtimestamp(
                    job.completed_at, tz=datetime.timezone.utc
                )
                html += f"<tr><td style='color: #888888;'><b>Completed At:</b></td><td>{dt.strftime('%Y-%m-%d %H:%M:%S UTC')}</td></tr>"

            if job.error_message:
                html += f"""
                    <tr><td style="color: #ff3333; valign: top;"><b>Error:</b></td><td style="color: #ff3333;">{job.error_message}</td></tr>
                """
            html += "</table>"

            if job.config.hyperparameters:
                html += """
                <h4 style="color: #ffffff; margin-top: 15px; margin-bottom: 5px; border-bottom: 1px solid #3e3e42; padding-bottom: 3px;">
                    Hyperparameters
                </h4>
                <table cellpadding="4" cellspacing="0" style="width: 100%;">
                """
                for k, v in job.config.hyperparameters.items():
                    html += f"<tr><td style='color: #888888; width: 150px;'>{k}:</td><td>{v}</td></tr>"
                html += "</table>"

            html += "</div>"
            self.summary_label.setText(html)

        # Plot Tab
        series = model.metric_series_for_selected
        if HAS_MATPLOTLIB and series:
            self._update_plot(series)
            self.plot_stack.setCurrentIndex(1)
        else:
            self.plot_stack.setCurrentIndex(0)

    def _update_plot(self, series_list: tuple[MetricSeries, ...]) -> None:
        self.figure.clear()
        if not series_list:
            self.canvas.draw()
            return

        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="#aaaaaa")
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.title.set_color("#ffffff")
        ax.spines["bottom"].set_color("#2d2d2d")
        ax.spines["top"].set_color("#2d2d2d")
        ax.spines["left"].set_color("#2d2d2d")
        ax.spines["right"].set_color("#2d2d2d")

        for series in series_list:
            if not series.steps:
                continue
            (line,) = ax.plot(
                series.steps,
                series.values,
                label=series.name,
                linewidth=2,
            )
            if series.smoothed is not None:
                ax.plot(
                    series.steps,
                    series.smoothed,
                    linestyle="--",
                    color=line.get_color(),
                    alpha=0.7,
                    label=f"{series.name} (smoothed)",
                )

        ax.legend(
            loc="upper left",
            frameon=True,
            facecolor="#2d2d2d",
            edgecolor="#3e3e42",
            labelcolor="#ffffff",
        )
        ax.set_xlabel("Steps")
        ax.set_ylabel("Values")
        self.figure.tight_layout()
        self.canvas.draw()

    def _on_selection_changed(self) -> None:
        selected_ranges = self.job_table.selectedRanges()
        if not selected_ranges:
            self.controller.select_job(None)
            return
        row = selected_ranges[0].topRow()
        item = self.job_table.item(row, 0)
        if item:
            self.controller.select_job(JobId(item.text()))
        else:
            self.controller.select_job(None)

    def _on_submit_clicked(self) -> None:
        dialog = SubmitDialog(self.controller, self)
        dialog.exec()

    def _on_cancel_clicked(self) -> None:
        selected_id = self.controller.selected_job_id
        if selected_id is not None:
            from src.shared.python.training import TrainingError

            try:
                self.controller.cancel_job(selected_id)
            except (TrainingError, TypeError, ValueError) as e:
                logger.error(f"Failed to cancel job: {e}")

    def _on_pause_clicked(self) -> None:
        selected_id = self.controller.selected_job_id
        if selected_id is not None:
            from src.shared.python.training import TrainingError

            try:
                self.controller.pause_job(selected_id)
            except (TrainingError, TypeError, ValueError) as e:
                logger.error(f"Failed to pause job: {e}")

    def _on_resume_clicked(self) -> None:
        selected_id = self.controller.selected_job_id
        if selected_id is not None:
            from src.shared.python.training import TrainingError

            try:
                self.controller.resume_job(selected_id)
            except (TrainingError, TypeError, ValueError) as e:
                logger.error(f"Failed to resume job: {e}")


class SubmitDialog(QDialog):
    """Dialogue window to configure and submit training configurations."""

    def __init__(
        self,
        controller: TrainingDashboardController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setObjectName("training-submit-dialog")
        self.setWindowTitle("Submit Training Job")
        self.setMinimumWidth(450)
        self.setStyleSheet(DARK_STYLE)

        self._init_layout()

    def _init_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        # Framework Selector
        self.framework_combo = QComboBox(self)
        for fw in TrainingFramework:
            self.framework_combo.addItem(fw.value.upper(), fw)
        form_layout.addRow("Framework:", self.framework_combo)

        # Entry Point Edit
        self.entry_edit = QLineEdit(self)
        self.entry_edit.setText("module:train")
        self.entry_edit.setToolTip("Path to the entry point (e.g. module:train)")
        form_layout.addRow("Entry Point:", self.entry_edit)

        # Output Dir Edit
        self.output_edit = QLineEdit(self)
        self.output_edit.setText(str(Path("/tmp/training-controller-gui")))
        self.output_edit.setToolTip("Directory where training outputs will be saved")
        form_layout.addRow("Output Directory:", self.output_edit)

        # Dataset Selector
        self.dataset_combo = QComboBox(self)
        datasets = self.controller.dataset_registry.list()
        for ds in datasets:
            self.dataset_combo.addItem(ds.name, ds.dataset_id)
        form_layout.addRow("Dataset:", self.dataset_combo)

        layout.addLayout(form_layout)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)

        self.submit_button = QPushButton("Submit", self)
        self.submit_button.setObjectName("submit-btn")

        self.cancel_btn = QPushButton("Cancel", self)

        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.submit_button)
        layout.addLayout(button_layout)

        # Signals
        self.submit_button.clicked.connect(self._on_submit_clicked)
        self.cancel_btn.clicked.connect(self.reject)

    def _on_submit_clicked(self) -> None:
        framework = self.framework_combo.currentData()
        entry_point = self.entry_edit.text().strip()
        output_dir_str = self.output_edit.text().strip()
        dataset_id = self.dataset_combo.currentData()

        # Basic inputs validation
        if not entry_point:
            logger.warning("Submission blocked: empty entry point")
            return
        if not output_dir_str:
            logger.warning("Submission blocked: empty output directory")
            return

        try:
            config = TrainingConfig(
                framework=framework,
                entry_point=entry_point,
                output_dir=Path(output_dir_str),
                dataset_id=dataset_id,
            )
            self.controller.submit_job(config)
            self.accept()
        except Exception as e:
            logger.exception(f"Failed to submit training job: {e}")


def build_default_controller() -> TrainingDashboardController:
    """Construct headless controller using a default Scheduler configuration."""
    from src.shared.python.training import (
        CompatibilityChecker,
        JobRegistry,
        Scheduler,
    )
    from src.shared.python.training.runtime import InProcessDriver, RunnerRegistry

    runners = RunnerRegistry()
    scheduler = Scheduler(
        registry=JobRegistry(),
        runners=runners,
        driver=InProcessDriver(runners, max_workers=1),
    )
    datasets = DatasetRegistry(
        initial=(
            Dataset(
                dataset_id="dataset-1",
                name="Dataset 1",
                path=Path("/tmp/dataset-1"),
                format="custom",
            ),
        )
    )
    return TrainingDashboardController(
        scheduler,
        datasets,
        CompatibilityChecker(),
    )
