"""Docker check and environment management dialogs.

Provides the Docker availability checker thread and the environment
(Docker build) management dialog.
"""

from __future__ import annotations

import time
from typing import Any

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
from src.launchers.docker_manager import DockerCheckThread as SharedDockerCheckThread
from src.launchers.launcher_constants import DOCKER_STAGES
from src.shared.python.docker_config import DOCKER_IMAGE_ENGINE as DOCKER_IMAGE_NAME
from src.shared.python.logging_pkg.logging_config import get_logger

from .startup import REPOS_ROOT

logger = get_logger(__name__)

# Preserve legacy import path while reusing shared implementation.
DockerCheckThread = SharedDockerCheckThread


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
        self._building = False  # Guard against concurrent builds (issue #2715)
        self.setup_ui()

    def setup_ui(self) -> None:
        """Build the Docker build dialog UI layout."""
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Build Tab
        tab_build = QWidget()
        build_layout = QVBoxLayout(tab_build)
        self.combo_stage = QComboBox()
        self.combo_stage.addItems(list(DOCKER_STAGES))
        build_layout.addWidget(QLabel("Target Stage:"))
        build_layout.addWidget(self.combo_stage)

        btn_row = QHBoxLayout()
        # Match the Settings → Configuration → Docker Image button label
        # so users see the same vocabulary in both build entry points.
        self.btn_build = QPushButton("Build Image")
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
        self.console.setProperty("class", "console-dark")
        self.console.style().polish(self.console)
        build_layout.addWidget(self.console)
        tabs.addTab(tab_build, "Build Docker")

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def start_build(self) -> None:
        """Launch the Docker build process in a background thread (issue #2715).

        Prevents concurrent builds and ensures thread cleanup.
        """
        # Serialize: ignore if build already in progress
        if self._building:
            logger.warning("Build already in progress; ignoring request")
            return

        self._building = True
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
        if not (line is not None):
            raise ValueError("line must be provided")
        if not (line is not None):
            raise ValueError("line must be provided")
        self.console.append(line)
        # Auto-scroll to bottom
        sb = self.console.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def _on_build_finished(self, success: bool, message: str) -> None:
        if not (success is not None):
            raise ValueError("success must be provided")
        if not (success is not None):
            raise ValueError("success must be provided")
        self.btn_build.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._building = False
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
            self._building = False
            if self._elapsed_timer_id is not None:
                self.killTimer(self._elapsed_timer_id)
                self._elapsed_timer_id = None

    def closeEvent(self, event: Any) -> None:
        """Handle dialog close event to clean up threads (issue #2715)."""
        # Join the build thread with timeout to prevent orphans
        if self.build_thread and self.build_thread.isRunning():
            logger.info("Waiting for Docker build thread to finish...")
            if not self.build_thread.wait(5000):  # 5 second timeout
                logger.warning("Docker build thread did not exit; terminating")
                self.build_thread.terminate()
                self.build_thread.wait(1000)
        super().closeEvent(event)

    def timerEvent(self, event: Any) -> None:
        """Update the elapsed-time label on each timer tick."""
        elapsed = time.monotonic() - self._build_start_time
        self.build_status_label.setText(f"Building... ({elapsed:.0f}s elapsed)")
