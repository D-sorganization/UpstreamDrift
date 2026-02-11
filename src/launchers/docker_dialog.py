"""Docker check and environment management dialogs.

Provides the Docker availability checker thread and the environment
(Docker build) management dialog.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.launchers.docker_manager import DockerBuildThread
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.security.secure_subprocess import secure_run

from .startup import REPOS_ROOT

logger = get_logger(__name__)

DOCKER_IMAGE_NAME = "robotics_env"


class DockerCheckThread(QThread):
    result = pyqtSignal(bool)

    def run(self) -> None:
        """Run docker check."""
        try:
            secure_run(
                ["docker", "--version"],
                timeout=5.0,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.result.emit(True)
        except (OSError, ValueError):
            self.result.emit(False)


class EnvironmentDialog(QDialog):
    """Dialog to manage Docker environment and view dependencies."""

    _DOCKER_CONTEXT = REPOS_ROOT / "src" / "engines" / "physics_engines" / "mujoco"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage Environment")
        self.resize(700, 500)
        self.build_thread: DockerBuildThread | None = None
        self._build_start_time: float = 0.0
        self._elapsed_timer_id: int | None = None
        self.setup_ui()

    def setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Build Tab
        tab_build = QWidget()
        build_layout = QVBoxLayout(tab_build)
        self.combo_stage = QComboBox()
        self.combo_stage.addItems(["all", "mujoco", "pinocchio", "drake", "base"])
        build_layout.addWidget(QLabel("Target Stage:"))
        build_layout.addWidget(self.combo_stage)

        btn_row = QHBoxLayout()
        self.btn_build = QPushButton("Build Environment")
        self.btn_build.clicked.connect(self.start_build)
        btn_row.addWidget(self.btn_build)

        self.btn_cancel = QPushButton("Cancel Build")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_build)
        btn_row.addWidget(self.btn_cancel)
        build_layout.addLayout(btn_row)

        self.build_status_label = QLabel("")
        build_layout.addWidget(self.build_status_label)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet(
            "background-color: #1e1e1e; color: #00ff00; font-family: Consolas;"
        )
        build_layout.addWidget(self.console)
        tabs.addTab(tab_build, "Build Docker")

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def start_build(self) -> None:
        self.console.clear()
        self.btn_build.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._build_start_time = time.monotonic()
        self._elapsed_timer_id = self.startTimer(1000)
        self.build_status_label.setText("Building...")

        self.build_thread = DockerBuildThread(
            target_stage=self.combo_stage.currentText(),
            image_name=DOCKER_IMAGE_NAME,
            context_path=self._DOCKER_CONTEXT,
        )
        self.build_thread.log_signal.connect(self._on_build_log)
        self.build_thread.finished_signal.connect(self._on_build_finished)
        self.build_thread.start()

    def _on_build_log(self, line: str) -> None:
        self.console.append(line)
        # Auto-scroll to bottom
        sb = self.console.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def _on_build_finished(self, success: bool, message: str) -> None:
        self.btn_build.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if self._elapsed_timer_id is not None:
            self.killTimer(self._elapsed_timer_id)
            self._elapsed_timer_id = None
        elapsed = time.monotonic() - self._build_start_time
        status = "SUCCESS" if success else "FAILED"
        self.build_status_label.setText(f"Build {status} ({elapsed:.0f}s): {message}")
        self.console.append(f"\n=== Build {status} ({elapsed:.0f}s) ===")

    def _cancel_build(self) -> None:
        if self.build_thread and self.build_thread.isRunning():
            self.build_thread.terminate()
            self.build_status_label.setText("Build cancelled.")
            self.btn_build.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            if self._elapsed_timer_id is not None:
                self.killTimer(self._elapsed_timer_id)
                self._elapsed_timer_id = None

    def timerEvent(self, event: Any) -> None:
        elapsed = time.monotonic() - self._build_start_time
        self.build_status_label.setText(f"Building... ({elapsed:.0f}s elapsed)")
