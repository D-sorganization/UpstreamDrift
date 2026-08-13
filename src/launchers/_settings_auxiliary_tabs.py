"""Diagnostics and process-management behavior for launcher settings."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.launchers.hover_copy_browser import HoverCopyTextBrowser
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.style_constants import Styles

logger = get_logger(__name__)


class _DiagnosticsWorker(QThread):
    """Run launcher diagnostics without blocking the GUI thread."""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, launcher: Any | None = None) -> None:
        super().__init__()
        self._launcher = launcher

    def run(self) -> None:
        try:
            from src.launchers.launcher_diagnostics import LauncherDiagnostics

            diagnostics = LauncherDiagnostics()
            results = diagnostics.run_all_checks()
            runtime_state = _launcher_runtime_state(self._launcher)
            if runtime_state is not None:
                results["runtime_state"] = runtime_state
            self.finished.emit(results)
        except ImportError as exc:
            self.error.emit(str(exc))


class _SubmoduleSyncWorker(QThread):
    """Synchronize the shared Tools checkout without blocking the UI."""

    finished = pyqtSignal(bool, str)

    def __init__(self, repos_root: Path) -> None:
        super().__init__()
        self._repos_root = repos_root

    def run(self) -> None:
        try:
            from src.launchers.launcher_diagnostics import LauncherDiagnostics

            result = subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=str(self._repos_root),
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                timeout=60.0,
            )
            subprocess.run(
                ["git", "fetch"],
                cwd=str(self._repos_root),
                capture_output=True,
                timeout=15.0,
            )
            submodule = self._repos_root / "vendor" / "ud-tools"
            if submodule.is_dir():
                subprocess.run(
                    ["git", "fetch"],
                    cwd=str(submodule),
                    capture_output=True,
                    timeout=15.0,
                )
            sibling = LauncherDiagnostics._find_sibling_tools_root()
            if sibling:
                subprocess.run(
                    ["git", "fetch"],
                    cwd=str(sibling),
                    capture_output=True,
                    timeout=15.0,
                )
            self.finished.emit(
                True, result.stdout or "Submodules synchronized successfully."
            )
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
            self.finished.emit(False, str(exc))


def _launcher_runtime_state(launcher: Any | None) -> dict[str, Any] | None:
    """Return the diagnostics snapshot for a launcher that exposes model state."""
    if launcher is None or not hasattr(launcher, "available_models"):
        return None
    return {
        "available_models_count": len(launcher.available_models),
        "available_model_ids": list(launcher.available_models.keys()),
        "model_order_count": len(launcher.model_order),
        "model_order": launcher.model_order,
        "model_cards_count": len(launcher.model_cards),
        "selected_model": launcher.selected_model,
        "docker_available": launcher.docker_available,
        "registry_loaded": launcher.registry is not None,
    }


class SettingsAuxiliaryTabsMixin:
    """Own diagnostics, log synchronization, and process-management tabs."""

    def _create_diagnostics_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        self._diag_browser = HoverCopyTextBrowser()
        self._diag_browser.setOpenExternalLinks(False)
        self._diag_browser.setStyleSheet(Styles.CONSOLE_DIAGNOSTICS)
        tab_layout.addWidget(self._diag_browser, stretch=3)

        diagnostics_data = getattr(self, "_diagnostics_data", None)
        if diagnostics_data:
            self._render_diagnostics(diagnostics_data)

        self._proc_log_viewer = self._create_log_viewer(
            tab_layout,
            "Process Output Log (recent)",
            180,
            Styles.CONSOLE_LOG_GREEN,
        )
        self._log_viewer = self._create_log_viewer(
            tab_layout,
            "Application Log (recent)",
            160,
            Styles.CONSOLE_LOG_LIGHT,
        )
        self._load_process_log()
        self._load_app_log()

        button_row = QHBoxLayout()
        button_row.addStretch()
        self.btn_sync_tools = QPushButton("Sync Shared Tools")
        self.btn_sync_tools.setToolTip(
            "Synchronize git submodules and fetch latest updates"
        )
        self.btn_sync_tools.clicked.connect(self._sync_shared_tools)
        button_row.addWidget(self.btn_sync_tools)
        refresh = QPushButton("Re-run Diagnostics")
        refresh.setToolTip("Run all diagnostic checks again")
        refresh.clicked.connect(self._refresh_diagnostics)
        button_row.addWidget(refresh)
        refresh_logs = QPushButton("Refresh Logs")
        refresh_logs.setToolTip("Reload all log files")
        refresh_logs.clicked.connect(self._refresh_all_logs)
        button_row.addWidget(refresh_logs)
        tab_layout.addLayout(button_row)
        return tab

    @staticmethod
    def _create_log_viewer(
        tab_layout: QVBoxLayout, title: str, maximum_height: int, style: str
    ) -> QTextEdit:
        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)
        viewer = QTextEdit()
        viewer.setReadOnly(True)
        viewer.setMaximumHeight(maximum_height)
        viewer.setStyleSheet(style)
        group_layout.addWidget(viewer)
        tab_layout.addWidget(group, stretch=1)
        return viewer

    def _load_app_log(self) -> None:
        log_candidates = [
            Path.cwd() / "app_launch.log",
            Path.home() / ".golf_modeling_suite" / "launcher.log",
        ]
        for log_path in log_candidates:
            if self._load_log_file(log_path, self._log_viewer, 200):
                return
        self._log_viewer.setPlainText("(No log file found)")

    def _load_process_log(self) -> None:
        log_path = Path.home() / ".golf_modeling_suite" / "process_output.log"
        if self._load_log_file(log_path, self._proc_log_viewer, 300):
            return
        self._proc_log_viewer.setPlainText(
            "(No process output log yet — launch a model to generate output)"
        )

    @staticmethod
    def _load_log_file(log_path: Path, viewer: QTextEdit, line_limit: int) -> bool:
        if not log_path.exists():
            return False
        try:
            from PyQt6.QtGui import QTextCursor

            text = log_path.read_text(encoding="utf-8", errors="replace")
            viewer.setPlainText("\n".join(text.strip().splitlines()[-line_limit:]))
            viewer.moveCursor(QTextCursor.MoveOperation.End)
            return True
        except (RuntimeError, ValueError, AttributeError, OSError) as exc:
            logger.debug("Could not display log file %s: %s", log_path, exc)
            return False

    def _refresh_all_logs(self) -> None:
        self._load_process_log()
        self._load_app_log()

    def _render_diagnostics(self, data: dict[str, Any]) -> None:
        if data is None:
            raise ValueError("data must be provided")
        summary = data.get("summary", {})
        checks = data.get("checks", [])
        runtime = data.get("runtime_state", {})
        recommendations = data.get("recommendations", [])
        html = self._render_diag_summary(summary)
        html += self._render_diag_checks(checks)
        html += self._render_diag_engines(checks)
        html += self._render_diag_runtime(runtime)
        html += self._render_diag_recommendations(recommendations)
        self._diag_browser.setHtml(html)

    @staticmethod
    def _render_diag_summary(summary: dict[str, Any]) -> str:
        if summary is None:
            raise ValueError("summary must be provided")
        status = str(summary.get("status", "unknown")).upper()
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        warnings = summary.get("warnings", 0)
        total = summary.get("total_checks", passed + failed + warnings)
        status_color = "#2da44e" if status == "HEALTHY" else "#d29922"
        return f"""
        <div style="margin-bottom: 12px;">
            <h2 style="color:{status_color}; margin: 0;">Status: {status}</h2>
            <p><b>{total} checks:</b>
                <span style="color:#2da44e;">{passed} passed</span>,
                <span style="color:#f85149;">{failed} failed</span>,
                <span style="color:#d29922;">{warnings} warnings</span>
            </p>
        </div>
        """

    @staticmethod
    def _render_diag_checks(checks: list[dict[str, Any]]) -> str:
        if checks is None:
            raise ValueError("checks must be provided")
        html = "<h3>Check Results</h3><table style='width:100%;'>"
        for check in checks:
            status = check["status"]
            icon = {"pass": "&#9989;", "fail": "&#10060;", "warning": "&#9888;"}.get(
                status, "&#8226;"
            )
            color = {
                "pass": "#2da44e",
                "fail": "#f85149",
                "warning": "#d29922",
            }.get(status, "#d4d4d4")
            duration = check.get("duration_ms", 0)
            html += (
                f"<tr><td style='color:{color}; padding:2px 6px;'>{icon}</td>"
                f"<td style='padding:2px 6px;'><b>{check['name']}</b></td>"
                f"<td style='padding:2px 6px; color:#a0a0a0;'>{check['message']}</td>"
                f"<td style='padding:2px 6px; color:#666;'>{duration:.0f}ms</td></tr>"
            )
        return html + "</table>"

    @staticmethod
    def _render_diag_engines(checks: list[dict[str, Any]]) -> str:
        if checks is None:
            raise ValueError("checks must be provided")
        engine_check = next(
            (check for check in checks if check["name"] == "engine_availability"),
            None,
        )
        engines = (
            engine_check.get("details", {}).get("engines", []) if engine_check else []
        )
        if not engines:
            return ""
        html = (
            "<h3>Physics Engines</h3><table style='width:100%; border-collapse:collapse;'>"
            "<tr style='border-bottom:1px solid #333;'>"
            "<th style='padding:4px 8px; text-align:left;'>Engine</th>"
            "<th style='padding:4px 8px; text-align:left;'>Status</th>"
            "<th style='padding:4px 8px; text-align:left;'>Version</th>"
            "<th style='padding:4px 8px; text-align:left;'>Details</th></tr>"
        )
        for engine in engines:
            installed = engine.get("installed", False)
            icon = "&#9989;" if installed else "&#10060;"
            color = "#2da44e" if installed else "#f85149"
            name = engine.get("name", "?").replace("_", " ").title()
            version = engine.get("version") or "-"
            details = engine.get("diagnostic", "")
            missing = engine.get("missing_deps", [])
            if missing and not installed:
                details = f"Missing: {', '.join(missing[:3])}"
            html += (
                f"<tr><td style='padding:3px 8px;'><b>{name}</b></td>"
                f"<td style='padding:3px 8px; color:{color};'>{icon} "
                f"{'Installed' if installed else 'Not installed'}</td>"
                f"<td style='padding:3px 8px; color:#a0a0a0;'>{version}</td>"
                f"<td style='padding:3px 8px; color:#888;'>{details}</td></tr>"
            )
        return html + "</table>"

    @staticmethod
    def _render_diag_runtime(runtime: dict[str, Any]) -> str:
        if runtime is None:
            raise ValueError("runtime must be provided")
        if not runtime:
            return ""
        return (
            "<h3>Runtime State</h3><ul>"
            f"<li>Available models: {runtime.get('available_models_count', '?')}</li>"
            f"<li>Tile order: {runtime.get('model_order_count', '?')}</li>"
            f"<li>Model cards: {runtime.get('model_cards_count', '?')}</li>"
            f"<li>Registry loaded: {runtime.get('registry_loaded', '?')}</li>"
            f"<li>Docker available: {runtime.get('docker_available', '?')}</li></ul>"
        )

    @staticmethod
    def _render_diag_recommendations(recommendations: list[str]) -> str:
        if recommendations is None:
            raise ValueError("recommendations must be provided")
        if not recommendations:
            return ""
        items = "".join(
            f"<li>{recommendation}</li>" for recommendation in recommendations[:8]
        )
        return f"<h3>Recommendations</h3><ul>{items}</ul>"

    def _refresh_diagnostics(self) -> None:
        try:
            from src.launchers.launcher_diagnostics import LauncherDiagnostics

            results = LauncherDiagnostics().run_all_checks()
            runtime_state = _launcher_runtime_state(getattr(self, "_launcher", None))
            if runtime_state is not None:
                results["runtime_state"] = runtime_state
            self._diagnostics_data = results
            self._render_diagnostics(results)
        except ImportError as exc:
            self._diag_browser.setHtml(
                f"<p style='color:#f85149;'>Error running diagnostics: {exc}</p>"
            )

    def _on_tab_changed(self, index: int) -> None:
        if index == self.TAB_PROCESSES:
            self.refresh_processes_ui()
            return
        if index != self.TAB_DIAGNOSTICS or self._diagnostics_loaded:
            return
        self._diagnostics_loaded = True
        self._diag_browser.setHtml(
            "<p style='color:#d29922;'>Running diagnostics...</p>"
        )
        self._diag_worker = _DiagnosticsWorker(getattr(self, "_launcher", None))
        self._diag_worker.finished.connect(self._on_diagnostics_ready)
        self._diag_worker.error.connect(self._on_diagnostics_error)
        self._diag_worker.start()

    def _on_diagnostics_ready(self, results: dict[str, Any]) -> None:
        self._diagnostics_data = results
        self._render_diagnostics(results)

    def _on_diagnostics_error(self, error_msg: str) -> None:
        self._diag_browser.setHtml(
            f"<p style='color:#f85149;'>Error running diagnostics: {error_msg}</p>"
        )

    def _sync_shared_tools(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        from src.shared.python.data_io.path_utils import get_repo_root

        reply = QMessageBox.question(
            self,
            "Sync Shared Tools",
            "Are you sure you want to synchronize git submodules and fetch latest remote updates?\n\n"
            "This will run 'git submodule update --init --recursive' in the background.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.No:
            return
        self.btn_sync_tools.setEnabled(False)
        self.btn_sync_tools.setText("Syncing...")
        self._diag_browser.append(
            "<p style='color:#58a6ff;'>Starting sync of shared tools...</p>"
        )
        self._sync_worker = _SubmoduleSyncWorker(get_repo_root())
        self._sync_worker.finished.connect(self._on_sync_finished)
        self._sync_worker.start()

    def _on_sync_finished(self, success: bool, output: str) -> None:
        from PyQt6.QtWidgets import QMessageBox

        self.btn_sync_tools.setEnabled(True)
        self.btn_sync_tools.setText("Sync Shared Tools")
        if success:
            QMessageBox.information(
                self,
                "Sync Complete",
                "Shared tools synchronization completed successfully.",
            )
            self._refresh_diagnostics()
        else:
            QMessageBox.critical(
                self,
                "Sync Failed",
                f"Failed to synchronize shared tools:\n\n{output}",
            )

    def _create_processes_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(15, 15, 15, 15)
        tab_layout.setSpacing(10)
        self.lbl_processes_status = QLabel("Active Processes")
        self.lbl_processes_status.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #ffffff;"
        )
        tab_layout.addWidget(self.lbl_processes_status)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "background-color: transparent; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px;"
        )
        self.processes_container_widget = QWidget()
        self.processes_container_widget.setStyleSheet("background-color: transparent;")
        self.processes_layout = QVBoxLayout(self.processes_container_widget)
        self.processes_layout.setContentsMargins(10, 10, 10, 10)
        self.processes_layout.setSpacing(6)
        self.processes_layout.addStretch()
        scroll.setWidget(self.processes_container_widget)
        tab_layout.addWidget(scroll)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.btn_kill_all_procs = QPushButton("Kill All Processes")
        self.btn_kill_all_procs.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_kill_all_procs.setStyleSheet(
            "QPushButton { background-color: rgba(220, 53, 69, 0.2); "
            "border: 1px solid rgba(220, 53, 69, 0.4); border-radius: 4px; "
            "padding: 6px 14px; color: #ff6b6b; font-weight: bold; font-size: 11px; } "
            "QPushButton:hover { background-color: rgba(220, 53, 69, 0.4); color: #ffffff; }"
        )
        self.btn_kill_all_procs.clicked.connect(self._on_kill_all_clicked)
        button_layout.addWidget(self.btn_kill_all_procs)
        tab_layout.addLayout(button_layout)
        return tab

    def _on_kill_all_clicked(self) -> None:
        launcher = getattr(self, "_launcher", None)
        if launcher and hasattr(launcher, "_kill_all_processes"):
            launcher._kill_all_processes()
            self.refresh_processes_ui()

    def _on_kill_clicked(self, name: str) -> None:
        launcher = getattr(self, "_launcher", None)
        if launcher and hasattr(launcher, "_kill_process_by_name"):
            launcher._kill_process_by_name(name)
            self.refresh_processes_ui()

    def refresh_processes_ui(self) -> None:
        while self.processes_layout.count() > 1:
            item = self.processes_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        running = self._running_process_names()
        if not running:
            self.lbl_processes_status.setText("No active processes.")
            self.lbl_processes_status.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: gray;"
            )
            self.btn_kill_all_procs.setEnabled(False)
            return
        self.lbl_processes_status.setText(f"Active Processes ({len(running)})")
        self.lbl_processes_status.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #ffffff;"
        )
        self.btn_kill_all_procs.setEnabled(True)
        for name in running:
            self.processes_layout.insertWidget(
                self.processes_layout.count() - 1, self._create_process_row(name)
            )

    def _running_process_names(self) -> list[str]:
        launcher = getattr(self, "_launcher", None)
        if not (
            launcher
            and hasattr(launcher, "running_processes")
            and hasattr(launcher, "process_manager")
        ):
            return []
        running: list[str] = []
        with launcher.process_manager._process_lock:
            for name, process in list(launcher.running_processes.items()):
                if process.poll() is None:
                    running.append(name)
        return running

    def _create_process_row(self, name: str) -> QWidget:
        row = QWidget()
        row.setObjectName("ProcessRow")
        row.setStyleSheet(
            "#ProcessRow { background-color: rgba(255, 255, 255, 0.03); "
            "border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.05); } "
            "#ProcessRow:hover { background-color: rgba(255, 255, 255, 0.06); }"
        )
        layout = QHBoxLayout(row)
        layout.setContentsMargins(10, 8, 10, 8)
        status = QLabel("●")
        status.setStyleSheet("color: #2ecc71; font-size: 14px; margin-right: 6px;")
        layout.addWidget(status)
        label = QLabel(name)
        label.setStyleSheet("color: #e0e0e0; font-size: 12px; font-weight: 500;")
        layout.addWidget(label)
        layout.addStretch()
        kill_button = QPushButton("Kill")
        kill_button.setCursor(Qt.CursorShape.PointingHandCursor)
        kill_button.setStyleSheet(
            "QPushButton { background-color: rgba(220, 53, 69, 0.1); "
            "border: 1px solid rgba(220, 53, 69, 0.3); border-radius: 4px; "
            "padding: 4px 10px; color: #ff6b6b; font-size: 11px; } "
            "QPushButton:hover { background-color: rgba(220, 53, 69, 0.3); color: #ffffff; }"
        )
        kill_button.clicked.connect(
            lambda checked=False, process_name=name: self._on_kill_clicked(process_name)
        )
        layout.addWidget(kill_button)
        return row
