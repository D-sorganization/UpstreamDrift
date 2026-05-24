"""PyQt6 widget surface for the training-controller dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDockWidget,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger
from training import (
    CompatibilityChecker,
    CompatibilityError,
    DatasetRegistry,
    JobId,
    JobRegistry,
    RunResult,
    Scheduler,
    TrainingConfig,
    TrainingConfigError,
    TrainingError,
    TrainingFramework,
    TrainingStatus,
    best_per_metric,
    new_run_id,
)
from training.contracts import CancelToken, ProgressSink
from training.metrics import TrainingMetric
from training.resource_monitor import ResourceMonitor, ResourceMonitorUnavailableError
from training.runtime import InProcessDriver, RunnerRegistry

from .controller import TrainingDashboardController
from .live_subscriber import TrainingJobLiveSubscriber
from .view_model import DashboardModel, JobRow, MetricSeries

logger = get_logger(__name__)
_RESOURCE_MONITORS: dict[int, ResourceMonitor] = {}

__all__ = [
    "MainWidget",
    "MainWindow",
    "SubmitJobDialog",
    "build_default_controller",
    "main",
]


class _NoopRunner:
    framework = TrainingFramework.PYTORCH

    def can_run(self, config: TrainingConfig) -> bool:
        return config.framework is self.framework

    def prepare(self, config: TrainingConfig) -> None:
        del config

    def run(
        self,
        config: TrainingConfig,
        *,
        progress: ProgressSink,
        cancel: CancelToken,
    ) -> RunResult:
        del config, progress, cancel
        return RunResult(
            run_id=new_run_id(),
            status=TrainingStatus.COMPLETED,
            duration_s=0.0,
        )


def build_default_controller() -> TrainingDashboardController:
    """Build the default controller used by the standalone launcher."""

    runners = RunnerRegistry()
    runners.register(_NoopRunner())
    scheduler = Scheduler(
        registry=JobRegistry(),
        runners=runners,
        driver=InProcessDriver(runners, max_workers=1),
    )
    resource_monitor = _try_start_resource_monitor()
    provider = None if resource_monitor is None else lambda: resource_monitor.latest
    controller = TrainingDashboardController(
        scheduler,
        DatasetRegistry(),
        CompatibilityChecker(),
        resource_provider=provider,
    )
    if resource_monitor is not None:
        _RESOURCE_MONITORS[id(controller)] = resource_monitor
    return controller


def _try_start_resource_monitor() -> ResourceMonitor | None:
    try:
        monitor = ResourceMonitor()
    except ResourceMonitorUnavailableError:
        return None
    monitor.start()
    return monitor


class SubmitJobDialog(QDialog):
    """Modal editor that builds and submits a :class:`TrainingConfig`."""

    def __init__(
        self,
        controller: TrainingDashboardController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("training-submit-dialog")
        self.setWindowTitle("Submit Training Job")
        self.controller = controller
        self.created_job_id: JobId | None = None
        self.error_banner = QLabel("")
        self.error_banner.setWordWrap(True)
        self.error_banner.setVisible(False)

        self.framework_combo = QComboBox()
        for framework in TrainingFramework:
            self.framework_combo.addItem(framework.value, framework)
        self.engine_combo = QComboBox()
        for engine_name in _engine_names(controller.compatibility_checker):
            self.engine_combo.addItem(engine_name)
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("(none)", None)
        for dataset in controller.dataset_registry.list():
            self.dataset_combo.addItem(dataset.name, dataset.dataset_id)
        self.entry_edit = QLineEdit("module:train")
        self.output_edit = QLineEdit(str(Path.cwd() / "output" / "training"))
        self.hyperparams_edit = QPlainTextEdit("{}")

        form = QFormLayout()
        form.addRow("Framework", self.framework_combo)
        form.addRow("Target engine", self.engine_combo)
        form.addRow("Dataset", self.dataset_combo)
        form.addRow("Entry point", self.entry_edit)
        form.addRow("Output dir", self.output_edit)
        form.addRow("Hyperparameters", self.hyperparams_edit)

        self.submit_button = QPushButton("Submit")
        cancel_button = QPushButton("Cancel")
        self.submit_button.clicked.connect(self._submit)
        cancel_button.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(cancel_button)
        buttons.addWidget(self.submit_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.error_banner)
        layout.addLayout(form)
        layout.addLayout(buttons)

    def _submit(self) -> None:
        try:
            config = self._build_config()
            engine = self.engine_combo.currentText().strip() or None
            job = self.controller.submit_job(config, target_engine=engine)
        except (
            CompatibilityError,
            TrainingConfigError,
            TrainingError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            self._show_error(str(exc))
            return
        self.created_job_id = job.job_id
        self.accept()

    def _build_config(self) -> TrainingConfig:
        framework = self.framework_combo.currentData()
        if not isinstance(framework, TrainingFramework):
            raise ValueError("select a training framework")
        raw = self.hyperparams_edit.toPlainText().strip() or "{}"
        hyperparams = json.loads(raw)
        if not isinstance(hyperparams, dict):
            raise ValueError("hyperparameters must be a JSON object")
        dataset_id = self.dataset_combo.currentData()
        return TrainingConfig(
            framework=framework,
            entry_point=self.entry_edit.text().strip(),
            output_dir=Path(self.output_edit.text()).expanduser(),
            hyperparameters=hyperparams,
            dataset_id=dataset_id,
        )

    def _show_error(self, message: str) -> None:
        self.error_banner.setText(message)
        self.error_banner.setVisible(True)


class MainWidget(QWidget):
    """Training-controller dashboard widget."""

    model_change_requested = pyqtSignal()

    def __init__(
        self,
        controller: TrainingDashboardController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self._subscribers: dict[str, TrainingJobLiveSubscriber] = {}
        self._unsubscribe = controller.on_model_change(self._request_render)
        self.model_change_requested.connect(self.render)
        self._build_ui()
        self.render()

    def _build_ui(self) -> None:
        self.toolbar = QToolBar("Training actions")
        self.submit_button = QPushButton("Submit")
        self.cancel_button = QPushButton("Cancel")
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")
        self.open_output_button = QPushButton("Open Output Dir")
        for button in (
            self.submit_button,
            self.cancel_button,
            self.pause_button,
            self.resume_button,
            self.open_output_button,
        ):
            self.toolbar.addWidget(button)
        self.submit_button.clicked.connect(self._open_submit_dialog)
        self.cancel_button.clicked.connect(self._cancel_selected)
        self.pause_button.clicked.connect(self._pause_selected)
        self.resume_button.clicked.connect(self._resume_selected)
        self.open_output_button.clicked.connect(self._open_output_dir)

        self.status_filter = QComboBox()
        self.status_filter.addItem("all")
        for status in TrainingStatus:
            self.status_filter.addItem(status.value)
        self.status_filter.currentTextChanged.connect(lambda _text: self.render())

        self.job_table = QTableWidget(0, 5)
        self.job_table.setHorizontalHeaderLabels(
            ["id", "framework", "status", "dataset", "elapsed"]
        )
        self.job_table.setSortingEnabled(True)
        self.job_table.itemSelectionChanged.connect(self._select_current_row)

        left = QVBoxLayout()
        left.addWidget(QLabel("Status filter"))
        left.addWidget(self.status_filter)
        left.addWidget(self.job_table)

        self.detail_tabs = QTabWidget()
        self.metrics_panel = _MetricPanel()
        self.summary_list = QListWidget()
        self.detail_tabs.addTab(self.metrics_panel, "Live metrics")
        self.detail_tabs.addTab(self.summary_list, "Summary")
        self.dataset_list = QListWidget()

        body = QHBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left)
        body.addWidget(left_widget, 1)
        body.addWidget(self.detail_tabs, 2)

        self.resource_label = QLabel()
        self.resource_label.setObjectName("training-resource-strip")

        root = QVBoxLayout(self)
        root.addWidget(self.toolbar)
        root.addLayout(body, 1)
        root.addWidget(self.resource_label)

    def render(self) -> None:
        model = self.controller.current_model()
        self._sync_subscribers(model)
        self._render_jobs(model)
        self.metrics_panel.render(model.metric_series_for_selected)
        self._render_summary(model)
        self._render_resources(model)
        self._render_datasets()

    def cleanup(self) -> None:
        self._unsubscribe()
        for subscriber in tuple(self._subscribers.values()):
            subscriber.stop()
        self._subscribers.clear()
        monitor = _RESOURCE_MONITORS.pop(id(self.controller), None)
        if monitor is not None:
            monitor.stop()
        self.controller.close()
        self.controller.scheduler.shutdown(wait=False)

    def _request_render(self) -> None:
        self.model_change_requested.emit()

    def _render_jobs(self, model: DashboardModel) -> None:
        selected = (
            model.selected_job_id.value if model.selected_job_id is not None else None
        )
        rows = [
            row
            for row in model.jobs
            if self.status_filter.currentText() in ("all", row.status)
        ]
        self.job_table.setSortingEnabled(False)
        self.job_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            self._set_row(index, row)
            if row.job_id == selected:
                self.job_table.selectRow(index)
        self.job_table.setSortingEnabled(True)

    def _set_row(self, index: int, row: JobRow) -> None:
        values = (
            row.job_id[:12],
            row.framework,
            row.status,
            row.dataset_id or "",
            _format_elapsed(row.elapsed_s),
        )
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setData(Qt.ItemDataRole.UserRole, row.job_id)
            self.job_table.setItem(index, col, item)

    def _render_summary(self, model: DashboardModel) -> None:
        self.summary_list.clear()
        selected = model.selected_job_id
        if selected is None:
            return
        best_metrics = best_per_metric(self.controller.metrics_for(selected))
        for name, best in sorted(best_metrics.items()):
            self.summary_list.addItem(f"{name}: {best.value:g} at step {best.step}")
        row = model.selected_row
        if row is not None and row.error_message:
            self.summary_list.addItem(f"Error: {row.error_message}")

    def _render_resources(self, model: DashboardModel) -> None:
        resources = model.resources
        if not resources.available:
            self.resource_label.setText("monitoring unavailable; install psutil")
            return
        parts = [
            f"CPU {resources.cpu_percent:.1f}%",
            f"Memory {resources.memory_percent:.1f}%",
        ]
        for gpu in resources.gpus:
            util = (
                "n/a"
                if gpu.utilization_percent is None
                else f"{gpu.utilization_percent:.1f}%"
            )
            parts.append(
                f"GPU {gpu.index} {gpu.name}: {util}, "
                f"{gpu.memory_used_mb}/{gpu.memory_total_mb} MiB"
            )
        self.resource_label.setText(" | ".join(parts))

    def _render_datasets(self) -> None:
        self.dataset_list.clear()
        for dataset in self.controller.dataset_registry.list():
            item = QListWidgetItem(f"{dataset.name} ({dataset.dataset_id})")
            item.setToolTip(str(dataset.path))
            self.dataset_list.addItem(item)

    def _sync_subscribers(self, model: DashboardModel) -> None:
        live = {row.job_id for row in model.jobs if row.status in {"running", "paused"}}
        for job_id in live - self._subscribers.keys():
            subscriber = TrainingJobLiveSubscriber(
                job_id,
                on_metric=lambda metric, jid=job_id: self._ingest_metric(jid, metric),
                on_status=lambda _status, _message: self._request_render(),
            )
            subscriber.start()
            self._subscribers[job_id] = subscriber
        for job_id in set(self._subscribers) - live:
            self._subscribers.pop(job_id).stop()

    def _ingest_metric(self, job_id: str, metric: TrainingMetric) -> None:
        self.controller.ingest_metric(JobId(job_id), metric)

    def _selected_job_id(self) -> JobId | None:
        items = self.job_table.selectedItems()
        if not items:
            return None
        return JobId(str(items[0].data(Qt.ItemDataRole.UserRole)))

    def _select_current_row(self) -> None:
        job_id = self._selected_job_id()
        try:
            self.controller.select_job(job_id)
        except KeyError:
            self.controller.select_job(None)

    def _open_submit_dialog(self) -> None:
        dialog = SubmitJobDialog(self.controller, self)
        dialog.exec()

    def _cancel_selected(self) -> None:
        job_id = self._selected_job_id()
        if job_id is not None:
            self.controller.cancel_job(job_id)

    def _pause_selected(self) -> None:
        job_id = self._selected_job_id()
        if job_id is not None:
            self.controller.pause_job(job_id)

    def _resume_selected(self) -> None:
        job_id = self._selected_job_id()
        if job_id is not None:
            self.controller.resume_job(job_id)

    def _open_output_dir(self) -> None:
        job_id = self._selected_job_id()
        if job_id is None:
            return
        job = self.controller.scheduler.get(job_id)
        path = job.config.output_dir
        if not path.exists():
            QMessageBox.information(self, "Output directory", str(path))
            return
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


class _MetricPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.layout = QVBoxLayout(self)
        self._canvas: Any | None = None
        self._figure: Any | None = None
        self._fallback = QLabel("No metrics yet")
        self.layout.addWidget(self._fallback)
        self._init_canvas()

    def _init_canvas(self) -> None:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except (ImportError, OSError):
            return
        self._figure = Figure(figsize=(5, 4))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self.layout.addWidget(self._canvas)
        self._fallback.hide()

    def render(self, series: tuple[MetricSeries, ...]) -> None:
        if self._canvas is None or self._figure is None:
            self._fallback.setText(_metric_text(series))
            self._fallback.show()
            return
        self._figure.clear()
        if not series:
            self._canvas.draw_idle()
            return
        axes_by_kind: dict[str, Any] = {}
        for metric in series:
            key = metric.kind.value
            axis = axes_by_kind.get(key)
            if axis is None:
                axis = self._figure.add_subplot(
                    len({s.kind for s in series}),
                    1,
                    len(axes_by_kind) + 1,
                )
                axis.set_title(key)
                axes_by_kind[key] = axis
            axis.plot(metric.steps, metric.values, label=metric.name)
            if metric.smoothed is not None:
                axis.plot(
                    metric.steps, metric.smoothed, label=f"{metric.name} smoothed"
                )
            axis.legend(loc="best")
        self._figure.tight_layout()
        self._canvas.draw_idle()


class MainWindow(QMainWindow):
    """Top-level Training Controller window."""

    def __init__(
        self,
        controller: TrainingDashboardController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Training Controller")
        self.resize(1100, 720)
        self.main_widget = MainWidget(controller or build_default_controller(), self)
        self.setCentralWidget(self.main_widget)
        dock = QDockWidget("Dataset library", self)
        dock.setWidget(self.main_widget.dataset_list)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.main_widget.render()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        self.main_widget.cleanup()
        event.accept()


def _engine_names(checker: CompatibilityChecker) -> list[str]:
    try:
        from src.shared.python.engine_core.engine_registry import get_registry

        registered = [engine.value for engine in get_registry().all_types()]
    except (ImportError, AttributeError, ValueError):
        registered = []
    names = sorted(set(registered) | set(checker.known_engines))
    return names or ["mujoco"]


def _format_elapsed(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


def _metric_text(series: tuple[MetricSeries, ...]) -> str:
    if not series:
        return "No metrics yet"
    lines = []
    for metric in series:
        last = metric.values[-1] if metric.values else 0.0
        lines.append(f"{metric.kind.value} / {metric.name}: {last:g}")
    return "\n".join(lines)


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    QTimer.singleShot(0, window.main_widget.render)
    return int(app.exec())
