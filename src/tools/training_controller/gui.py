"""PyQt6 GUI components for the Training Controller dashboard.

Provides MainWindow, MainWidget, and SubmitDialog to interact with the
headless TrainingDashboardController.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
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

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.training import (
    Dataset,
    DatasetRegistry,
    JobId,
    TrainingConfig,
    TrainingError,
    TrainingFramework,
    best_per_metric,
)

from ._style import DARK_STYLE
from .controller import TrainingDashboardController
from .view_model import DashboardModel, MetricSeries, ResourceSnapshot

# Setup Logger per CLAUDE.md (shared logging facade).
logger = get_logger(__name__)

# Sentinel meaning "do not run the engine-compatibility preflight".
_NO_ENGINE = ""


def _available_engine_names() -> list[str]:
    """Return engine identifiers for the submit dialog's target dropdown.

    Sourced from :meth:`EngineRegistry.all_types`. When the global
    registry has no engines registered (common in a headless GUI-only
    process), fall back to the full :class:`EngineType` enumeration so
    the dropdown is never empty and the compatibility preflight has
    something to validate against.
    """

    from src.shared.python.engine_core.engine_registry import (  # noqa: PLC0415
        EngineType,
        get_registry,
    )

    registered = [t.value for t in get_registry().all_types()]
    if registered:
        return sorted(registered)
    return sorted(t.value for t in EngineType)


def _open_in_file_browser(path: Path) -> None:
    """Open ``path`` in the OS file browser (best-effort, cross-platform).

    The directory is created if missing — a job's output dir is created
    lazily by the worker, so on a freshly-queued job it may not exist
    yet. All failures are logged and swallowed: opening a folder is a
    convenience, never a critical path.
    """

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Open Output Dir: could not create %s: %s", path, exc)
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)  # noqa: S603, S607
        else:
            subprocess.run(["xdg-open", str(path)], check=False)  # noqa: S603, S607
    except (OSError, ValueError) as exc:
        logger.warning("Open Output Dir: failed to open %s: %s", path, exc)


def _folder_size_bytes(path: Path) -> int:
    """Best-effort recursive byte count for ``path``.

    Returns ``0`` when the path does not exist or cannot be walked —
    :class:`Dataset` accepts a ``0`` (unknown) size, so failures degrade
    gracefully rather than blocking registration.
    """

    if not path.exists():
        return 0
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    continue
    except OSError as exc:
        logger.warning("Could not size dataset folder %s: %s", path, exc)
        return 0
    return total


try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


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
        # Live realtime subscribers, keyed by job id, populated on
        # resume() and torn down on pause()/cleanup().
        self._subscribers: dict[str, Any] = {}
        self._backgrounded = False

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

        self.open_output_button = QPushButton("Open Output Dir", self)
        self.open_output_button.setToolTip(
            "Open the selected job's output directory in the file browser"
        )

        actions_layout.addWidget(self.submit_button)
        actions_layout.addWidget(self.cancel_button)
        actions_layout.addWidget(self.pause_button)
        actions_layout.addWidget(self.resume_button)
        actions_layout.addWidget(self.open_output_button)
        actions_layout.addStretch(1)
        left_layout.addLayout(actions_layout)

        # Status filter for the job list.
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(8)
        filter_label = QLabel("Filter status:", self)
        filter_label.setStyleSheet("color: #aaaaaa;")
        self.status_filter = QComboBox(self)
        self.status_filter.addItem("All", None)
        from src.shared.python.training import TrainingStatus  # noqa: PLC0415

        for status in TrainingStatus:
            self.status_filter.addItem(status.value, status.value)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.status_filter)
        filter_layout.addStretch(1)
        left_layout.addLayout(filter_layout)

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
        self.job_table.setSortingEnabled(True)
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
        self.summary_label.setOpenExternalLinks(False)
        self.summary_label.linkActivated.connect(self._on_summary_link)
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

        # Dataset library controls: add / remove / re-scan a folder.
        dataset_btn_layout = QHBoxLayout()
        dataset_btn_layout.setSpacing(6)
        self.dataset_add_button = QPushButton("Add Folder", self)
        self.dataset_add_button.setToolTip(
            "Register a folder as a dataset in the library"
        )
        self.dataset_remove_button = QPushButton("Remove", self)
        self.dataset_remove_button.setToolTip("Remove the selected dataset")
        self.dataset_rescan_button = QPushButton("Re-scan", self)
        self.dataset_rescan_button.setToolTip(
            "Re-scan the selected dataset's folder to refresh its on-disk size"
        )
        dataset_btn_layout.addWidget(self.dataset_add_button)
        dataset_btn_layout.addWidget(self.dataset_remove_button)
        dataset_btn_layout.addWidget(self.dataset_rescan_button)
        dataset_btn_layout.addStretch(1)
        dock_layout.addLayout(dataset_btn_layout)

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
        self.open_output_button.clicked.connect(self._on_open_output_clicked)
        self.status_filter.currentIndexChanged.connect(self.update_ui)
        self.dataset_add_button.clicked.connect(self._on_dataset_add_clicked)
        self.dataset_remove_button.clicked.connect(self._on_dataset_remove_clicked)
        self.dataset_rescan_button.clicked.connect(self._on_dataset_rescan_clicked)

        self.job_table.itemSelectionChanged.connect(self._on_selection_changed)

        self.update_ui_signal.connect(self.update_ui)
        self._unsubscribe = self.controller.on_model_change(self._trigger_update)

    def _trigger_update(self) -> None:
        self.update_ui_signal.emit()

    def cleanup(self) -> None:
        """Unsubscribe from controller observers on closed tab."""
        self._stop_live_subscribers()
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except (ValueError, RuntimeError) as e:
                logger.debug(f"Failed to unsubscribe cleanly: {e}")
            self._unsubscribe = None

    # ------------------------------------------------------ backgrounding

    def pause(self) -> None:
        """Background hook (#6013): detach live realtime subscriptions.

        The scheduler keeps running, so this is a *cheapness* measure
        only — it tears down the per-job realtime watchers so a hidden
        tab is not paying for IPC it cannot display. The authoritative
        job state remains in the scheduler registry and is re-read on
        :meth:`resume`.
        """

        self._backgrounded = True
        self._stop_live_subscribers()

    def resume(self) -> None:
        """Background hook (#6013): re-bind to scheduler status + progress.

        Re-renders the dashboard from the (still-bound) controller and
        re-establishes a :class:`TrainingJobLiveSubscriber` for every
        non-terminal job so live metrics resume flowing into the
        controller's in-memory series.
        """

        self._backgrounded = False
        self._sync_live_subscribers()
        self.update_ui()

    def _sync_live_subscribers(self) -> None:
        """Ensure exactly one live subscriber per non-terminal job."""

        from src.shared.python.training import JobId  # noqa: PLC0415

        from .live_subscriber import TrainingJobLiveSubscriber  # noqa: PLC0415

        active: set[str] = set()
        for job in self.controller.scheduler.registry.list():
            if job.status.is_terminal:
                continue
            active.add(job.job_id.value)
            if job.job_id.value in self._subscribers:
                continue
            job_value = job.job_id.value

            def _on_metric(metric: Any, _jid: str = job_value) -> None:
                self.controller.ingest_metric(JobId(_jid), metric)

            subscriber = TrainingJobLiveSubscriber(job_value, on_metric=_on_metric)
            try:
                subscriber.start()
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning(
                    "Live subscriber start failed for %s: %s", job_value, exc
                )
                continue
            self._subscribers[job_value] = subscriber
        # Drop subscribers whose jobs are gone or terminal.
        for job_value in list(self._subscribers):
            if job_value not in active:
                self._stop_subscriber(job_value)

    def _stop_subscriber(self, job_value: str) -> None:
        subscriber = self._subscribers.pop(job_value, None)
        if subscriber is None:
            return
        try:
            subscriber.stop()
        except (RuntimeError, OSError, ValueError) as exc:
            logger.debug("Subscriber stop failed for %s: %s", job_value, exc)

    def _stop_live_subscribers(self) -> None:
        for job_value in list(self._subscribers):
            self._stop_subscriber(job_value)

    def update_ui(self) -> None:
        """Update widgets state from the latest DashboardModel."""
        model = self.controller.current_model()

        # Apply the status filter (None == show all).
        status_filter = self.status_filter.currentData()
        visible_jobs = [
            job
            for job in model.jobs
            if status_filter is None or job.status == status_filter
        ]

        # Update Job Table. Disable sorting while we repopulate so row
        # indices stay stable, then re-enable it afterwards.
        self.job_table.blockSignals(True)
        self.job_table.setSortingEnabled(False)
        self.job_table.setRowCount(len(visible_jobs))
        for row_idx, job in enumerate(visible_jobs):
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
        self.job_table.setSortingEnabled(True)
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
            self.open_output_button.setEnabled(False)
            return

        # Enable actions when a job is selected, letting the controller
        # validate transitions.
        self.cancel_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(True)
        self.open_output_button.setEnabled(True)

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
        from PyQt6.QtWidgets import QListWidgetItem  # noqa: PLC0415

        self.dataset_list.clear()
        datasets = self.controller.dataset_registry.list()
        for ds in datasets:
            label = f"{ds.name} ({ds.dataset_id}) — {ds.format}"
            if ds.size_bytes:
                label += f", {ds.size_bytes / (1024 * 1024):.1f} MB"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, ds.dataset_id)
            self.dataset_list.addItem(item)

    def _selected_dataset_id(self) -> str | None:
        item = self.dataset_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _on_dataset_add_clicked(self) -> None:
        """Register a chosen folder as a ``custom`` dataset."""

        folder = QFileDialog.getExistingDirectory(self, "Select dataset folder")
        if not folder:
            return
        path = Path(folder)
        dataset_id = path.name or "dataset"
        # De-duplicate the id if the registry already has one.
        registry = self.controller.dataset_registry
        candidate = dataset_id
        suffix = 1
        while registry.has(candidate):
            suffix += 1
            candidate = f"{dataset_id}-{suffix}"
        try:
            registry.register(
                Dataset(
                    dataset_id=candidate,
                    name=path.name or candidate,
                    path=path,
                    format="custom",
                    size_bytes=_folder_size_bytes(path),
                )
            )
        except TrainingError as exc:
            logger.warning("Add dataset failed: %s", exc)
            return
        self.update_ui()

    def _on_dataset_remove_clicked(self) -> None:
        dataset_id = self._selected_dataset_id()
        if dataset_id is None:
            return
        try:
            self.controller.dataset_registry.remove(dataset_id)
        except TrainingError as exc:
            logger.warning("Remove dataset failed: %s", exc)
            return
        self.update_ui()

    def _on_dataset_rescan_clicked(self) -> None:
        """Re-scan the selected dataset's folder to refresh its size."""

        dataset_id = self._selected_dataset_id()
        if dataset_id is None:
            return
        registry = self.controller.dataset_registry
        try:
            existing = registry.get(dataset_id)
        except TrainingError as exc:
            logger.warning("Re-scan dataset failed: %s", exc)
            return
        from dataclasses import replace as _replace  # noqa: PLC0415

        refreshed = _replace(existing, size_bytes=_folder_size_bytes(existing.path))
        registry.replace(refreshed)
        self.update_ui()

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

            html += self._best_so_far_html(selected_id)
            html += self._artifacts_html(job)
            html += "</div>"
            self.summary_label.setText(html)

        # Plot Tab
        series = model.metric_series_for_selected
        if HAS_MATPLOTLIB and series:
            self._update_plot(series)
            self.plot_stack.setCurrentIndex(1)
        else:
            self.plot_stack.setCurrentIndex(0)

    def _best_so_far_html(self, job_id: JobId) -> str:
        """Render the "best so far" card from buffered metrics.

        Uses :func:`best_per_metric`, which honours each
        :class:`MetricKind`'s direction-of-improvement, so the card
        shows the optimal value seen for minimised metrics (loss) and
        maximised ones (reward / accuracy) alike.
        """

        best = best_per_metric(self.controller.metrics_for(job_id))
        if not best:
            return ""
        rows = "".join(
            f"<tr><td style='color: #888888; width: 150px;'>{name}:</td>"
            f"<td>{bm.value:.4g} <span style='color: #666666;'>"
            f"(step {bm.step})</span></td></tr>"
            for name, bm in sorted(best.items())
        )
        return (
            '<h4 style="color: #ffffff; margin-top: 15px; margin-bottom: 5px; '
            'border-bottom: 1px solid #3e3e42; padding-bottom: 3px;">'
            "Best so far</h4>"
            '<table cellpadding="4" cellspacing="0" style="width: 100%;">'
            f"{rows}</table>"
        )

    def _artifacts_html(self, job: Any) -> str:
        """Render the output directory as a clickable artifact link.

        :class:`RunResult.artifacts` paths are written under the job's
        configured ``output_dir``; surfacing that directory as a single
        click-to-open link gives the user access to checkpoints,
        ``metrics.json``, and logs without leaking absolute paths into
        the table.
        """

        output_dir = job.config.output_dir
        return (
            '<h4 style="color: #ffffff; margin-top: 15px; margin-bottom: 5px; '
            'border-bottom: 1px solid #3e3e42; padding-bottom: 3px;">'
            "Artifacts</h4>"
            "<p><a style='color: #0A84FF;' href='artifact:output_dir'>"
            f"{output_dir}</a></p>"
        )

    def _on_summary_link(self, href: str) -> None:
        """Handle clicks on artifact links in the summary panel."""

        if href != "artifact:output_dir":
            return
        selected_id = self.controller.selected_job_id
        if selected_id is None:
            return
        try:
            job = self.controller.scheduler.registry.get(selected_id)
        except (TrainingError, KeyError) as exc:
            logger.warning("Artifact link: job lookup failed: %s", exc)
            return
        _open_in_file_browser(job.config.output_dir)

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

    def _run_lifecycle_action(
        self,
        verb: str,
        action: Callable[[JobId], object],
    ) -> None:
        """Run a controller lifecycle call against the selection, logging errors.

        Centralises the cancel/pause/resume try/except so each handler
        is a one-liner and the narrowed exception tuple lives in one
        place (DRY per CLAUDE.md).
        """

        selected_id = self.controller.selected_job_id
        if selected_id is None:
            return
        try:
            action(selected_id)
        except (TrainingError, TypeError, ValueError) as exc:
            logger.warning("Failed to %s job: %s", verb, exc)

    def _on_cancel_clicked(self) -> None:
        self._run_lifecycle_action("cancel", self.controller.cancel_job)

    def _on_pause_clicked(self) -> None:
        self._run_lifecycle_action("pause", self.controller.pause_job)

    def _on_resume_clicked(self) -> None:
        self._run_lifecycle_action("resume", self.controller.resume_job)

    def _on_open_output_clicked(self) -> None:
        """Open the selected job's configured output directory."""

        selected_id = self.controller.selected_job_id
        if selected_id is None:
            return
        try:
            job = self.controller.scheduler.registry.get(selected_id)
        except (TrainingError, KeyError) as exc:
            logger.warning("Open Output Dir: job lookup failed: %s", exc)
            return
        _open_in_file_browser(job.config.output_dir)


class SubmitDialog(QDialog):
    """Dialogue window to configure and submit training configurations.

    Builds a :class:`TrainingConfig` from the form. When a target engine
    is chosen, an idiot-proof *preflight* runs
    :meth:`CompatibilityChecker.check` and surfaces any error-severity
    issues in a non-dismissable banner; the Submit button stays disabled
    while errors are present, so an incompatible (config, engine) pair
    can never be dispatched.
    """

    def __init__(
        self,
        controller: TrainingDashboardController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setObjectName("training-submit-dialog")
        self.setWindowTitle("Submit Training Job")
        self.setMinimumWidth(480)
        self.setStyleSheet(DARK_STYLE)

        self._init_layout()
        self._run_preflight()

    def _init_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        # Non-dismissable preflight banner (hidden until a compat error).
        self.preflight_banner = QLabel("", self)
        self.preflight_banner.setObjectName("preflight-banner")
        self.preflight_banner.setWordWrap(True)
        self.preflight_banner.setStyleSheet(
            "background-color: #4a1015; border: 1px solid #FF375F; "
            "border-radius: 4px; color: #ffd0d6; padding: 8px;"
        )
        self.preflight_banner.setVisible(False)
        layout.addWidget(self.preflight_banner)

        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        # Framework Selector
        self.framework_combo = QComboBox(self)
        for fw in TrainingFramework:
            self.framework_combo.addItem(fw.value.upper(), fw)
        form_layout.addRow("Framework:", self.framework_combo)

        # Target-engine Selector (drives the compatibility preflight).
        self.engine_combo = QComboBox(self)
        self.engine_combo.addItem("(no engine — skip preflight)", _NO_ENGINE)
        for engine_name in _available_engine_names():
            self.engine_combo.addItem(engine_name, engine_name)
        self.engine_combo.setToolTip(
            "Target physics engine. Selecting one runs a compatibility "
            "preflight before the job is queued."
        )
        form_layout.addRow("Target Engine:", self.engine_combo)

        # Entry Point Edit
        self.entry_edit = QLineEdit(self)
        self.entry_edit.setText("module:train")
        self.entry_edit.setToolTip("Path to the entry point (e.g. module:train)")
        form_layout.addRow("Entry Point:", self.entry_edit)

        # Output Dir Edit
        self.output_edit = QLineEdit(self)
        self.output_edit.setText(
            str(Path(tempfile.gettempdir()) / "training-controller-gui")
        )
        self.output_edit.setToolTip("Directory where training outputs will be saved")
        form_layout.addRow("Output Directory:", self.output_edit)

        # Dataset Selector
        self.dataset_combo = QComboBox(self)
        self.dataset_combo.addItem("(none)", None)
        datasets = self.controller.dataset_registry.list()
        for ds in datasets:
            self.dataset_combo.addItem(ds.name, ds.dataset_id)
        form_layout.addRow("Dataset:", self.dataset_combo)

        layout.addLayout(form_layout)

        # Hyperparameter JSON editor.
        hp_label = QLabel("Hyperparameters (JSON object):", self)
        hp_label.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(hp_label)
        self.hyperparams_edit = QPlainTextEdit(self)
        self.hyperparams_edit.setPlainText("{}")
        self.hyperparams_edit.setFixedHeight(90)
        self.hyperparams_edit.setStyleSheet(
            "background-color: #2d2d2d; border: 1px solid #3e3e42; "
            "border-radius: 4px; color: #ffffff; font-family: Consolas, monospace;"
        )
        layout.addWidget(self.hyperparams_edit)

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
        self.engine_combo.currentIndexChanged.connect(self._run_preflight)
        self.framework_combo.currentIndexChanged.connect(self._run_preflight)

    def _selected_engine(self) -> str | None:
        """Return the chosen engine name, or ``None`` when preflight is skipped."""

        engine = self.engine_combo.currentData()
        return engine or None

    def _build_config(self) -> TrainingConfig | None:
        """Assemble a :class:`TrainingConfig` from the form, or ``None``.

        Returns ``None`` (and surfaces the reason in the banner) when an
        input is invalid — empty entry point / output dir or malformed
        hyperparameter JSON.
        """

        entry_point = self.entry_edit.text().strip()
        output_dir_str = self.output_edit.text().strip()
        if not entry_point:
            self._show_banner("Entry point must not be empty.")
            return None
        if not output_dir_str:
            self._show_banner("Output directory must not be empty.")
            return None
        raw_hp = self.hyperparams_edit.toPlainText().strip() or "{}"
        try:
            hyperparameters = json.loads(raw_hp)
        except json.JSONDecodeError as exc:
            self._show_banner(f"Hyperparameters must be valid JSON: {exc}")
            return None
        if not isinstance(hyperparameters, dict):
            self._show_banner("Hyperparameters must be a JSON object ({...}).")
            return None
        try:
            return TrainingConfig(
                framework=self.framework_combo.currentData(),
                entry_point=entry_point,
                output_dir=Path(output_dir_str),
                dataset_id=self.dataset_combo.currentData(),
                hyperparameters=hyperparameters,
            )
        except TrainingError as exc:
            self._show_banner(f"Invalid configuration: {exc}")
            return None

    def _run_preflight(self) -> None:
        """Validate (config, engine) and enable/disable Submit accordingly."""

        engine = self._selected_engine()
        if engine is None:
            self._clear_banner()
            return
        config = self._build_config()
        if config is None:
            # _build_config already surfaced the reason and left submit
            # disabled via _show_banner.
            return
        report = self.controller.compatibility_checker.check(config, engine)
        if report.is_compatible:
            self._clear_banner()
            return
        messages = "; ".join(issue.message for issue in report.errors)
        self._show_banner(f"Engine {engine!r} is incompatible: {messages}")

    def _show_banner(self, message: str) -> None:
        self.preflight_banner.setText(message)
        self.preflight_banner.setVisible(True)
        self.submit_button.setEnabled(False)

    def _clear_banner(self) -> None:
        self.preflight_banner.clear()
        self.preflight_banner.setVisible(False)
        self.submit_button.setEnabled(True)

    def _on_submit_clicked(self) -> None:
        config = self._build_config()
        if config is None:
            return
        engine = self._selected_engine()
        try:
            self.controller.submit_job(config, target_engine=engine)
        except TrainingError as exc:
            logger.warning("Submission blocked by backend: %s", exc)
            self._show_banner(str(exc))
            return
        self.accept()


def build_default_controller() -> TrainingDashboardController:
    """Construct headless controller using a default Scheduler configuration."""
    from src.shared.python.training import CompatibilityChecker, JobRegistry, Scheduler
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
                path=Path(tempfile.gettempdir()) / "dataset-1",
                format="custom",
            ),
        )
    )
    return TrainingDashboardController(
        scheduler,
        datasets,
        CompatibilityChecker(),
    )


def main() -> int:
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    controller = build_default_controller()
    win = MainWindow(controller)
    win.show()
    return app.exec()
