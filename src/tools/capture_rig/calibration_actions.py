"""Connect calibration revision review to the existing capture command runner."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QPushButton, QWidget

from src.motion_capture.rig.bundle import load_bundle
from src.motion_capture.rig.plan import RigPlan

from .calibration_dialog import CalibrationDialog
from .commands import PlanSelection


class CalibrationActions(QObject):
    """Keep revision selection available from the header while capture is idle."""

    def __init__(
        self,
        parent: QWidget,
        *,
        selection: Callable[[], PlanSelection],
        session: Callable[[], Path | None],
        library_root: Callable[[], Path],
        apply: Callable[[Path], None],
        recalibrate: Callable[[Path], None],
        busy: Callable[[], bool],
    ) -> None:
        super().__init__(parent)
        self._host = parent
        self._selection, self._session, self._root = selection, session, library_root
        self._apply, self._recalibrate, self._busy = apply, recalibrate, busy
        self.button = QPushButton("Calibration", parent)
        self.button.setToolTip(
            "Review lens and zoom settings, save revisions, or recalibrate a board take."
        )
        self.button.clicked.connect(self.show)

    def refresh(self) -> None:
        self.button.setEnabled(not self._busy())

    def _plan(self) -> RigPlan:
        session = self._session()
        if session is not None:
            plan, index, _ = load_bundle(session)
            sizes = {entry.view: entry for entry in index.recordings}
            cameras = []
            for camera in plan.cameras:
                entry = sizes[camera.view]
                if not entry.width or not entry.height:
                    raise ValueError(f"Recording size is unknown for {camera.view}")
                mode = camera.mode.model_copy(
                    update={"width": entry.width, "height": entry.height}
                )
                cameras.append(camera.model_copy(update={"mode": mode}))
            return plan.model_copy(update={"cameras": tuple(cameras)})
        selected = self._selection()
        return RigPlan.load(selected.plan).with_overrides(
            mode=selected.mode, views=selected.views or None, controls=selected.controls
        )

    def show(self) -> Path | None:
        if self._busy():
            return None
        try:
            root = self._root() / "calibration"
            root.mkdir(parents=True, exist_ok=True)
            dialog = CalibrationDialog(self._plan(), root, self._host)
            dialog.exec()
            if dialog.output_path is not None:
                self._apply(dialog.output_path)
                return dialog.output_path
            if dialog.recalibrate_requested:
                self._repeat()
        except (ValueError, OSError, sqlite3.Error, KeyError) as exc:
            QMessageBox.warning(self._host, "Camera Calibration", str(exc))
        return None

    def _repeat(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self._host, "Select a Recorded Chessboard Session to Recalibrate"
        )
        if folder:
            load_bundle(Path(folder))
            self._recalibrate(Path(folder))
