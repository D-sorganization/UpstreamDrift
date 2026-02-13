"""Settings dialog for the UpstreamDrift Launcher.

Provides a tabbed dialog with Layout, Configuration, and Diagnostics tabs.
"""
# mypy: disable-error-code="attr-defined,assignment"

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.launchers.docker_manager import DockerBuildThread
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.style_constants import Styles

from .startup import REPOS_ROOT

logger = get_logger(__name__)

DOCKER_IMAGE_NAME = "robotics_env"


class SettingsDialog(QDialog):
    """Settings dialog with Layout, Configuration, and Diagnostics tabs.

    Tab order:
        0 - Layout: tile arrangement, lock, reset
        1 - Configuration: execution env, simulation opts, Docker rebuild
        2 - Diagnostics: system checks, error logs, terminal output
    """

    reset_layout_requested = pyqtSignal()

    # Tab index constants for external callers
    TAB_LAYOUT = 0
    TAB_CONFIG = 1
    TAB_DIAGNOSTICS = 2

    def __init__(
        self,
        parent: QWidget | None = None,
        diagnostics_data: dict[str, Any] | None = None,
        initial_tab: int = 0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(850, 650)
        self._diagnostics_data = diagnostics_data
        self._setup_ui()
        self.tabs.setCurrentIndex(initial_tab)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        from src.shared.python.gui_pkg.draggable_tabs import DraggableTabWidget

        self.tabs = DraggableTabWidget(
            core_tabs={"Layout", "Configuration", "Diagnostics"}
        )
        self.tabs.setTabsClosable(False)
        layout.addWidget(self.tabs)

        self.tabs.addTab(self._create_layout_tab(), "Layout")
        self.tabs.addTab(self._create_configuration_tab(), "Configuration")
        self.tabs.addTab(self._create_diagnostics_tab(), "Diagnostics")

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    # ── Layout tab ──────────────────────────────────────────────────

    def _create_layout_tab(self) -> QWidget:
        """Layout tab: tile lock, edit tiles, reset to defaults."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        group = QGroupBox("Tile Layout")
        inner = QVBoxLayout(group)

        self._btn_layout_lock = QPushButton("Layout: Locked")
        self._btn_layout_lock.setCheckable(True)
        self._btn_layout_lock.setChecked(False)
        self._btn_layout_lock.setStyleSheet(Styles.BTN_LAYOUT_TOGGLE)
        inner.addWidget(self._btn_layout_lock)

        self._btn_edit_tiles = QPushButton("Edit Tiles (show/hide)")
        self._btn_edit_tiles.setEnabled(False)
        inner.addWidget(self._btn_edit_tiles)

        inner.addSpacing(12)

        btn_reset = QPushButton("Reset Layout to Defaults")
        btn_reset.setToolTip("Restore all tiles and default arrangement")
        btn_reset.clicked.connect(self._on_reset_layout)
        inner.addWidget(btn_reset)

        tab_layout.addWidget(group)

        # Sync with parent launcher
        launcher = self.parent()
        if launcher and hasattr(launcher, "btn_modify_layout"):
            self._btn_layout_lock.setChecked(launcher.btn_modify_layout.isChecked())
            self._btn_layout_lock.toggled.connect(launcher.btn_modify_layout.click)
            self._btn_edit_tiles.clicked.connect(launcher.open_layout_manager)
            self._btn_layout_lock.toggled.connect(self._btn_edit_tiles.setEnabled)

        tab_layout.addStretch()
        return tab

    # ── Configuration tab ───────────────────────────────────────────

    def _create_configuration_tab(self) -> QWidget:
        """Configuration tab: execution env + simulation opts + Docker rebuild."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        # --- Execution environment ---
        env_group = QGroupBox("Execution Environment")
        env_inner = QVBoxLayout(env_group)

        self.chk_docker = QCheckBox("Docker mode")
        self.chk_docker.setToolTip(
            "Run physics engines in Docker containers (requires Docker Desktop)"
        )
        env_inner.addWidget(self.chk_docker)

        self.chk_wsl = QCheckBox("WSL mode")
        self.chk_wsl.setToolTip(
            "Run in WSL2 Ubuntu environment (full Pinocchio/Drake/Crocoddyl support)"
        )
        env_inner.addWidget(self.chk_wsl)

        tab_layout.addWidget(env_group)

        # --- Simulation options ---
        sim_group = QGroupBox("Simulation Options")
        sim_inner = QVBoxLayout(sim_group)

        self.chk_live_viz = QCheckBox("Live Visualization")
        self.chk_live_viz.setToolTip(
            "Enable real-time 3D visualization during simulation"
        )
        sim_inner.addWidget(self.chk_live_viz)

        self.chk_gpu = QCheckBox("GPU Acceleration")
        self.chk_gpu.setToolTip(
            "Use GPU for physics computation (requires supported hardware)"
        )
        sim_inner.addWidget(self.chk_gpu)

        tab_layout.addWidget(sim_group)

        # --- Rebuild Environment (Docker build) ---
        build_group = QGroupBox("Rebuild Environment")
        build_inner = QVBoxLayout(build_group)

        stage_row = QHBoxLayout()
        stage_row.addWidget(QLabel("Target Stage:"))
        self.combo_stage = QComboBox()
        self.combo_stage.addItems(["all", "mujoco", "pinocchio", "drake", "base"])
        stage_row.addWidget(self.combo_stage)
        build_inner.addLayout(stage_row)

        btn_row = QHBoxLayout()
        self._btn_build = QPushButton("Build Environment")
        self._btn_build.clicked.connect(self._start_build)
        btn_row.addWidget(self._btn_build)

        self._btn_cancel_build = QPushButton("Cancel")
        self._btn_cancel_build.setEnabled(False)
        self._btn_cancel_build.clicked.connect(self._cancel_build)
        btn_row.addWidget(self._btn_cancel_build)
        build_inner.addLayout(btn_row)

        self._build_status = QLabel("")
        build_inner.addWidget(self._build_status)

        self.build_console = QTextEdit()
        self.build_console.setReadOnly(True)
        self.build_console.setMaximumHeight(150)
        self.build_console.setStyleSheet(Styles.CONSOLE_BUILD)
        build_inner.addWidget(self.build_console)

        tab_layout.addWidget(build_group)

        # Sync checkboxes with parent launcher state
        launcher = self.parent()
        if launcher and hasattr(launcher, "chk_docker"):
            self.chk_docker.setChecked(launcher.chk_docker.isChecked())
            self.chk_wsl.setChecked(launcher.chk_wsl.isChecked())
            self.chk_live_viz.setChecked(launcher.chk_live.isChecked())
            self.chk_gpu.setChecked(launcher.chk_gpu.isChecked())

            self.chk_docker.toggled.connect(launcher.chk_docker.setChecked)
            self.chk_wsl.toggled.connect(launcher.chk_wsl.setChecked)
            self.chk_live_viz.toggled.connect(launcher.chk_live.setChecked)
            self.chk_gpu.toggled.connect(launcher.chk_gpu.setChecked)

        return tab

    # ── Diagnostics tab ─────────────────────────────────────────────

    def _create_diagnostics_tab(self) -> QWidget:
        """Diagnostics tab: system checks, error log viewer, terminal output."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        # System checks browser
        self._diag_browser = QTextBrowser()
        self._diag_browser.setOpenExternalLinks(False)
        self._diag_browser.setStyleSheet(Styles.CONSOLE_DIAGNOSTICS)
        tab_layout.addWidget(self._diag_browser, stretch=3)

        if self._diagnostics_data:
            self._render_diagnostics(self._diagnostics_data)

        # Process output log viewer
        proc_group = QGroupBox("Process Output Log (recent)")
        proc_inner = QVBoxLayout(proc_group)
        self._proc_log_viewer = QTextEdit()
        self._proc_log_viewer.setReadOnly(True)
        self._proc_log_viewer.setMaximumHeight(180)
        self._proc_log_viewer.setStyleSheet(Styles.CONSOLE_LOG_GREEN)
        proc_inner.addWidget(self._proc_log_viewer)
        tab_layout.addWidget(proc_group, stretch=1)

        # Application log viewer
        log_group = QGroupBox("Application Log (recent)")
        log_inner = QVBoxLayout(log_group)
        self._log_viewer = QTextEdit()
        self._log_viewer.setReadOnly(True)
        self._log_viewer.setMaximumHeight(160)
        self._log_viewer.setStyleSheet(Styles.CONSOLE_LOG_LIGHT)
        log_inner.addWidget(self._log_viewer)
        tab_layout.addWidget(log_group, stretch=1)

        # Load recent log lines
        self._load_process_log()
        self._load_app_log()

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn_refresh = QPushButton("Re-run Diagnostics")
        btn_refresh.setToolTip("Run all diagnostic checks again")
        btn_refresh.clicked.connect(self._refresh_diagnostics)
        btn_row.addWidget(btn_refresh)

        btn_refresh_log = QPushButton("Refresh Logs")
        btn_refresh_log.setToolTip("Reload all log files")
        btn_refresh_log.clicked.connect(self._refresh_all_logs)
        btn_row.addWidget(btn_refresh_log)

        tab_layout.addLayout(btn_row)
        return tab

    def _load_app_log(self) -> None:
        """Load recent lines from the application log file."""
        log_candidates = [
            Path.cwd() / "app_launch.log",
            Path.home() / ".golf_modeling_suite" / "launcher.log",
        ]
        for log_path in log_candidates:
            if log_path.exists():
                try:
                    text = log_path.read_text(encoding="utf-8", errors="replace")
                    lines = text.strip().splitlines()
                    recent = "\n".join(lines[-200:])
                    self._log_viewer.setPlainText(recent)
                    self._log_viewer.moveCursor(
                        self._log_viewer.textCursor().End  # type: ignore[arg-type]
                    )
                    return
                except (RuntimeError, ValueError, AttributeError):
                    pass
        self._log_viewer.setPlainText("(No log file found)")

    def _load_process_log(self) -> None:
        """Load recent lines from the process output log file."""
        log_path = Path.home() / ".golf_modeling_suite" / "process_output.log"
        if log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                lines = text.strip().splitlines()
                recent = "\n".join(lines[-300:])
                self._proc_log_viewer.setPlainText(recent)
                self._proc_log_viewer.moveCursor(
                    self._proc_log_viewer.textCursor().End  # type: ignore[arg-type]
                )
                return
            except (RuntimeError, ValueError, AttributeError):
                pass
        self._proc_log_viewer.setPlainText(
            "(No process output log yet — launch a model to generate output)"
        )

    def _refresh_all_logs(self) -> None:
        """Refresh both log viewers."""
        self._load_process_log()
        self._load_app_log()

    def _render_diagnostics(self, data: dict[str, Any]) -> None:
        """Render diagnostics results as styled HTML."""
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

    def _render_diag_summary(self, summary: dict) -> str:
        status = summary.get("status", "unknown").upper()
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

    def _render_diag_checks(self, checks: list) -> str:
        html = "<h3>Check Results</h3><table style='width:100%;'>"
        for check in checks:
            icon = {"pass": "&#9989;", "fail": "&#10060;", "warning": "&#9888;"}.get(
                check["status"], "&#8226;"
            )
            color = {"pass": "#2da44e", "fail": "#f85149", "warning": "#d29922"}.get(
                check["status"], "#d4d4d4"
            )
            duration = check.get("duration_ms", 0)
            html += (
                f"<tr><td style='color:{color}; padding:2px 6px;'>{icon}</td>"
                f"<td style='padding:2px 6px;'><b>{check['name']}</b></td>"
                f"<td style='padding:2px 6px; color:#a0a0a0;'>{check['message']}</td>"
                f"<td style='padding:2px 6px; color:#666;'>{duration:.0f}ms</td></tr>"
            )
        html += "</table>"
        return html

    def _render_diag_engines(self, checks: list) -> str:
        engine_check = next(
            (c for c in checks if c["name"] == "engine_availability"), None
        )
        engines = (
            engine_check.get("details", {}).get("engines", []) if engine_check else []
        )
        if not engines:
            return ""

        html = "<h3>Physics Engines</h3>"
        html += (
            "<table style='width:100%; border-collapse:collapse;'>"
            "<tr style='border-bottom:1px solid #333;'>"
            "<th style='padding:4px 8px; text-align:left;'>Engine</th>"
            "<th style='padding:4px 8px; text-align:left;'>Status</th>"
            "<th style='padding:4px 8px; text-align:left;'>Version</th>"
            "<th style='padding:4px 8px; text-align:left;'>Details</th>"
            "</tr>"
        )
        for eng in engines:
            installed = eng.get("installed", False)
            icon = "&#9989;" if installed else "&#10060;"
            color = "#2da44e" if installed else "#f85149"
            name = eng.get("name", "?").replace("_", " ").title()
            version = eng.get("version") or "-"
            diag = eng.get("diagnostic", "")
            missing = eng.get("missing_deps", [])
            detail_str = diag
            if missing and not installed:
                detail_str = f"Missing: {', '.join(missing[:3])}"
            html += (
                f"<tr>"
                f"<td style='padding:3px 8px;'><b>{name}</b></td>"
                f"<td style='padding:3px 8px; color:{color};'>{icon} "
                f"{'Installed' if installed else 'Not installed'}</td>"
                f"<td style='padding:3px 8px; color:#a0a0a0;'>{version}</td>"
                f"<td style='padding:3px 8px; color:#888;'>{detail_str}</td>"
                f"</tr>"
            )
        html += "</table>"
        return html

    def _render_diag_runtime(self, runtime: dict) -> str:
        if not runtime:
            return ""
        html = "<h3>Runtime State</h3><ul>"
        html += (
            f"<li>Available models: {runtime.get('available_models_count', '?')}</li>"
        )
        html += f"<li>Tile order: {runtime.get('model_order_count', '?')}</li>"
        html += f"<li>Model cards: {runtime.get('model_cards_count', '?')}</li>"
        html += f"<li>Registry loaded: {runtime.get('registry_loaded', '?')}</li>"
        html += f"<li>Docker available: {runtime.get('docker_available', '?')}</li>"
        html += "</ul>"
        return html

    def _render_diag_recommendations(self, recommendations: list) -> str:
        if not recommendations:
            return ""
        html = "<h3>Recommendations</h3><ul>"
        for rec in recommendations[:8]:
            html += f"<li>{rec}</li>"
        html += "</ul>"
        return html

    def _refresh_diagnostics(self) -> None:
        """Re-run diagnostics and update the display."""
        try:
            from src.launchers.launcher_diagnostics import LauncherDiagnostics

            diag = LauncherDiagnostics()
            results = diag.run_all_checks()

            launcher = self.parent()
            if launcher and hasattr(launcher, "available_models"):
                results["runtime_state"] = {
                    "available_models_count": len(launcher.available_models),
                    "available_model_ids": list(launcher.available_models.keys()),
                    "model_order_count": len(launcher.model_order),
                    "model_order": launcher.model_order,
                    "model_cards_count": len(launcher.model_cards),
                    "selected_model": launcher.selected_model,
                    "docker_available": launcher.docker_available,
                    "registry_loaded": launcher.registry is not None,
                }

            self._diagnostics_data = results
            self._render_diagnostics(results)
        except ImportError as e:
            self._diag_browser.setHtml(
                f"<p style='color:#f85149;'>Error running diagnostics: {e}</p>"
            )

    def _on_reset_layout(self) -> None:
        self.reset_layout_requested.emit()

    def _start_build(self) -> None:
        self.build_console.clear()
        self._btn_build.setEnabled(False)
        self._btn_cancel_build.setEnabled(True)
        self._build_start_time = time.monotonic()
        self._build_timer_id = self.startTimer(1000)
        self._build_status.setText("Building...")

        context = REPOS_ROOT / "src" / "engines" / "physics_engines" / "mujoco"
        self.build_thread = DockerBuildThread(
            target_stage=self.combo_stage.currentText(),
            image_name=DOCKER_IMAGE_NAME,
            context_path=context,
        )
        self.build_thread.log_signal.connect(self._on_build_log)
        self.build_thread.finished_signal.connect(self._on_build_finished)
        self.build_thread.start()

    def _on_build_log(self, line: str) -> None:
        self.build_console.append(line)
        sb = self.build_console.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def _on_build_finished(self, success: bool, message: str) -> None:
        self._btn_build.setEnabled(True)
        self._btn_cancel_build.setEnabled(False)
        if hasattr(self, "_build_timer_id") and self._build_timer_id is not None:
            self.killTimer(self._build_timer_id)
            self._build_timer_id = None
        elapsed = time.monotonic() - self._build_start_time
        status = "SUCCESS" if success else "FAILED"
        self._build_status.setText(f"Build {status} ({elapsed:.0f}s): {message}")
        self.build_console.append(f"\n=== Build {status} ({elapsed:.0f}s) ===")

    def _cancel_build(self) -> None:
        if (
            hasattr(self, "build_thread")
            and self.build_thread
            and self.build_thread.isRunning()
        ):
            self.build_thread.terminate()
            self._build_status.setText("Build cancelled.")
            self._btn_build.setEnabled(True)
            self._btn_cancel_build.setEnabled(False)
            if hasattr(self, "_build_timer_id") and self._build_timer_id is not None:
                self.killTimer(self._build_timer_id)
                self._build_timer_id = None

    def timerEvent(self, event: Any) -> None:
        """Update the build elapsed-time label on each timer tick."""
        if hasattr(self, "_build_start_time"):
            elapsed = time.monotonic() - self._build_start_time
            self._build_status.setText(f"Building... ({elapsed:.0f}s elapsed)")
