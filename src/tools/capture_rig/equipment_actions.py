"""Header entry for the capture library's player bag."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QMessageBox, QPushButton, QWidget

from .capture_library import CaptureLibrary
from .equipment_dialog import EquipmentDialog, capture_equipment_text


class EquipmentActions(QObject):
    def __init__(
        self,
        parent: QWidget,
        *,
        library: Callable[[], CaptureLibrary],
        current_session: Callable[[], Path | None],
        busy: Callable[[], bool],
    ) -> None:
        super().__init__(parent)
        self._host, self._library = parent, library
        self._session, self._busy = current_session, busy
        self.button = QPushButton("My Clubs", parent)
        self.button.clicked.connect(self.show_equipment)

    def refresh(self) -> None:
        self.button.setEnabled(not self._busy())
        self.button.setToolTip(capture_equipment_text(self._session()))

    def show_equipment(self) -> None:
        if self._busy():
            return
        try:
            library, capture = self._library(), self._session()
            if capture is not None:
                library.register(capture)
            EquipmentDialog(library.root, capture, self._host).exec()
        except (OSError, ValueError, sqlite3.Error) as exc:
            QMessageBox.warning(self._host, "My Clubs", str(exc))
        self.refresh()
