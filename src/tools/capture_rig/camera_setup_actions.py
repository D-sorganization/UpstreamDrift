"""Header and wizard entry points for player-owned camera setup revisions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QSettings
from PyQt6.QtWidgets import QPushButton

from . import commands
from .camera_setup import load_plan
from .camera_setup_dialog import CameraSetupDialog
from .layout import default_settings
from .library_actions import library_root
from .record_bar import Phase

if TYPE_CHECKING:
    from .gui import CaptureRigWidget

PLAN_PATH_KEY = "capture/plan_path"
LAB_PLAN = Path("docs/motion_capture/plans/lab_three_view_sonnet.json")


def default_plan_path(settings: QSettings | None = None) -> Path | None:
    """Restore the selected setup; retain the existing checkout sample fallback."""
    selected = settings if settings is not None else default_settings()
    saved = selected.value(PLAN_PATH_KEY, "", type=str)
    if saved:
        return Path(saved)
    sample = commands.repo_root() / LAB_PLAN
    return sample if sample.is_file() else None


class CameraSetupActions(QObject):
    def __init__(
        self, host: CaptureRigWidget, settings: QSettings | None = None
    ) -> None:
        super().__init__(host)
        self.host = host
        self.settings = settings if settings is not None else default_settings()
        self.button = QPushButton("Camera Setup", host)
        self.button.setToolTip(
            "Discover cameras, name their views and save a setup without editing JSON."
        )
        self.button.clicked.connect(self.show)

    def show(self) -> None:
        host = self.host
        if host.runner.busy or host.record_bar.phase is not Phase.IDLE:
            host.journey.notice(
                "Finish or stop the current capture operation before changing camera setup."
            )
            return
        capture = host.capture
        path = capture.plan_edit.text().strip()
        plan, warning = None, ""
        if path:
            try:
                plan = load_plan(Path(path))
            except (ValueError, OSError) as exc:
                warning = f"The current plan could not be opened: {exc}. Load another plan or create a new setup."
        dialog = CameraSetupDialog(library_root(self.settings), host, plan=plan)
        if warning:
            dialog.status.setText(warning)
        if dialog.exec() and dialog.saved_path is not None:
            try:
                self.apply(dialog.saved_path)
            except (ValueError, OSError) as exc:
                host.journey.notice(
                    f"Setup saved but could not be selected: {exc}. Load it again when ready."
                )

    def apply(self, path: Path) -> None:
        """Select a validated revision for the next take, preserving captured data."""
        plan = load_plan(path)
        host = self.host
        if host.runner.busy or host.record_bar.phase is not Phase.IDLE:
            raise ValueError(
                "Finish the current capture operation before selecting a different setup"
            )
        host.preview.stop()
        capture = host.capture
        capture.plan_edit.setText(str(path))
        capture.mode_combo.setCurrentIndex(0)
        for field in (capture.views_edit, capture.exposure_edit, capture.gain_edit):
            field.clear()
        capture.auto_exposure_combo.setCurrentIndex(0)
        self.settings.setValue(PLAN_PATH_KEY, str(path))
        host.journey.notice(
            f"Camera setup '{plan.name}' selected for the next take. Run Plan Check and Preview; "
            "saving a setup does not verify stream modes or calibration."
        )
