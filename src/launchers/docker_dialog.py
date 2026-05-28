"""Docker check and environment management dialogs.

Provides the Docker availability checker thread and the environment
(Docker build) management dialog.
"""

from __future__ import annotations

import time
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.launchers.docker_manager import DockerBuildThread
from src.launchers.docker_manager import DockerCheckThread as SharedDockerCheckThread
from src.launchers.docker_profile_info import (
    ProfileInfo,
    format_profile_summary,
    load_docker_profiles,
)
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
        tab_build_layout = QVBoxLayout(tab_build)

        container = QWidget()
        build_layout = QVBoxLayout(container)
        build_layout.setContentsMargins(0, 0, 0, 0)

        # Profile metadata for tier details (size, included features, etc.).
        # Loaded once at dialog construction — the YAML is read-only during a
        # session, and reloading on every selection would just add latency.
        self._profile_infos: dict[str, ProfileInfo] = load_docker_profiles()

        build_layout.addWidget(QLabel("Target Stage:"))
        self.combo_stage = QComboBox()
        self.combo_stage.addItems(list(DOCKER_STAGES))
        # Per-item hover tooltips give the user the answer to "what does this
        # tier mean?" without having to expand the details panel.
        self._apply_combo_tooltips()
        build_layout.addWidget(self.combo_stage)

        # Tier-details panel: lists max image size, estimated install size,
        # and every feature the selected profile bakes in. This replaces the
        # previous opaque "Professional", "Standard" labels with explicit
        # package-level information.
        # ``QTextBrowser`` for native scrolling of rich-text content. The
        # previous QLabel-in-QScrollArea pattern with ``widgetResizable``
        # was clipping the content rather than producing a usable
        # scrollbar — Qt sized the label to the viewport so the rich-text
        # rendering had no overflow region.
        from PyQt6.QtWidgets import QTextBrowser as _QTextBrowser

        self.tier_details = _QTextBrowser()
        self.tier_details.setReadOnly(True)
        self.tier_details.setOpenExternalLinks(False)
        self.tier_details.setFrameShape(QFrame.Shape.StyledPanel)
        self.tier_details.setFixedHeight(220)
        self.tier_details.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.tier_details.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.tier_details.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        build_layout.addWidget(self.tier_details)

        # Keep the details panel in sync with the combobox selection.
        self.combo_stage.currentTextChanged.connect(self._refresh_tier_details)
        self._refresh_tier_details(self.combo_stage.currentText())

        # Visible gap so a long features list can't visually overlap the
        # action row below.
        build_layout.addSpacing(12)

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

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(container)

        # DbC postconditions
        assert scroll.widget() is container, (
            "Postcondition: scroll area must wrap the build container"
        )
        assert container.layout() is not None, (
            "Postcondition: build container must have an active layout"
        )

        tab_build_layout.addWidget(scroll)
        tabs.addTab(tab_build, "Build Docker")

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def _apply_combo_tooltips(self) -> None:
        """Attach a per-item hover tooltip describing each profile."""
        for idx in range(self.combo_stage.count()):
            name = self.combo_stage.itemText(idx)
            info = self._profile_infos.get(name)
            if info is None:
                # Unknown profile (e.g. legacy hardcoded fallback name) —
                # leave the default tooltip empty rather than fabricate one.
                continue
            self.combo_stage.setItemData(
                idx, format_profile_summary(info), Qt.ItemDataRole.ToolTipRole
            )

    def _refresh_tier_details(self, profile_name: str) -> None:
        """Update the details panel for the *profile_name* tier selection."""
        info = self._profile_infos.get(profile_name)
        if info is None:
            self.tier_details.setHtml(
                f"<i>No metadata available for profile <b>{profile_name}</b>. "
                "See <code>docker/profiles.yaml</code>.</i>"
            )
            return

        # Render as lightweight HTML so we can bold the headline and indent
        # the feature list without pulling in a richer widget.
        title = profile_name.replace("-", " ").title()
        rows: list[str] = []
        rows.append(f"<b>{title}</b>")
        if info.description:
            rows.append(f'<span style="color: palette(mid);">{info.description}</span>')
        rows.append("")
        if info.max_size_mb:
            rows.append(
                f"<b>Budget:</b> &le; {info.max_size_mb} MB &nbsp;·&nbsp; "
                f"<b>Estimated install:</b> ~{info.approx_total_mb} MB"
            )
        if info.features:
            rows.append(f"<b>Includes {len(info.features)} feature(s):</b>")
            items: list[str] = []
            for f in info.features:
                size = f"{f.approx_size_mb} MB" if f.approx_size_mb else "—"
                items.append(
                    f"<li><b>{f.display_name}</b> "
                    f'<span style="color: palette(mid);">({size}) — '
                    f"{f.description}</span></li>"
                )
            rows.append("<ul style='margin-top: 4px;'>" + "".join(items) + "</ul>")
        elif info.feature_names:
            rows.append("<b>Features:</b> " + ", ".join(info.feature_names))

        self.tier_details.setHtml("<br>".join(rows))

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

        stage = self.combo_stage.currentText()
        from src.launchers.launcher_constants import DOCKER_STAGES

        if stage in DOCKER_STAGES:
            context = REPOS_ROOT
            dockerfile = REPOS_ROOT / "Dockerfile.modular"
            build_args = {"PROFILE": stage}
        else:
            context = self._DOCKER_CONTEXT
            dockerfile = None
            build_args = None

        self.build_thread = DockerBuildThread(
            target_stage=stage,
            image_name=DOCKER_IMAGE_NAME,
            context_path=context,
            dockerfile_path=dockerfile,
            build_args=build_args,
        )
        self.build_thread.log_signal.connect(self._on_build_log)
        self.build_thread.finished_signal.connect(self._on_build_finished)
        self.build_thread.start()

    def _on_build_log(self, line: str) -> None:
        if line is None:
            raise ValueError("line must be provided")
        self.console.append(line)
        # Auto-scroll to bottom
        sb = self.console.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def _on_build_finished(self, success: bool, message: str) -> None:
        if success is None:
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
