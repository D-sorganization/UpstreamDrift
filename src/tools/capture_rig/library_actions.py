"""Small header adapter for library/editor navigation and lifecycle guards."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from PyQt6.QtCore import QObject, QSettings, QStandardPaths
from PyQt6.QtWidgets import QMessageBox, QPushButton, QWidget

from src.motion_capture.rig.edits import has_analysis

from .capture_library import CaptureLibrary
from .layout import default_settings
from .library_dialog import LibraryDialog
from .swing_editor import SwingEditor

LIBRARY_ROOT_KEY = "capture/library_root"


def library_root(settings: QSettings | None = None) -> Path:
    """Resolve player storage without creating a catalog or capture directory."""
    selected = settings if settings is not None else default_settings()
    base = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppLocalDataLocation
    )
    fallback = Path(base) if base else Path.home() / "UpstreamDrift"
    root = selected.value(LIBRARY_ROOT_KEY, str(fallback / "capture-library"), type=str)
    return Path(root or fallback / "capture-library").expanduser().resolve()


def new_capture_path(settings: QSettings | None = None) -> Path:
    """Return a unique destination in the player's library without reserving it."""
    return library_root(settings) / "captures" / str(uuid4())


class LibraryActions(QObject):
    def __init__(
        self,
        parent: QWidget,
        *,
        current_session: Callable[[], Path | None],
        open_capture: Callable[[Path], None],
        import_videos: Callable[[Path], None],
        busy: Callable[[], bool],
        settings: QSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self._host = parent
        self._session, self._open, self._import, self._busy = (
            current_session,
            open_capture,
            import_videos,
            busy,
        )
        self._settings = settings if settings is not None else default_settings()
        self._library: CaptureLibrary | None = None
        self.library_button = QPushButton("Library", parent)
        self.library_button.setToolTip(
            "Find captures, save swing notes, manage storage and archive or restore takes."
        )
        self.library_button.clicked.connect(self.show_library)
        self.edit_button = QPushButton("Edit swing", parent)
        self.edit_button.setToolTip(
            "Choose the swing's first/last frames and crop before pose ingestion."
        )
        self.edit_button.clicked.connect(self.edit_swing)

    def library(self) -> CaptureLibrary:
        if self._library is None:
            self._library = CaptureLibrary(library_root(self._settings))
        return self._library

    def refresh(self) -> None:
        idle = not self._busy()
        self.library_button.setEnabled(idle)
        self.edit_button.setEnabled(idle and self._session() is not None)

    def recording_destination(self, requested: Path) -> Path:
        """Keep an empty selected destination; preserve existing takes in place."""
        if requested.exists() and any(requested.iterdir()):
            return new_capture_path(self._settings)
        return requested

    def _error(self, exc: Exception) -> None:
        QMessageBox.warning(self._host, "Capture library", str(exc))

    def show_library(self) -> None:
        if self._busy():
            return
        try:
            library = self.library()
            session = self._session()
            if session:
                library.register(session)
            dialog = LibraryDialog(
                library,
                open_capture=self._open,
                import_videos=self._import_new,
                location_changed=self._set_library,
                parent=self._host,
            )
            dialog.exec()
        except (ValueError, OSError, sqlite3.Error) as exc:
            self._error(exc)

    def _set_library(self, library: CaptureLibrary) -> None:
        self._library = library
        self._settings.setValue(LIBRARY_ROOT_KEY, str(library.root))

    def _import_new(self) -> None:
        target = self.library().root / "captures" / str(uuid4())
        self._import(target)
        self.refresh()

    def command_finished(self, code: int) -> None:
        session = self._session()
        if code == 0 and session is not None:
            try:
                self.library().register(session)
            except (ValueError, OSError, sqlite3.Error) as exc:
                self._error(exc)
        self.refresh()

    def edit_swing(self) -> None:
        root = self._session()
        if self._busy() or root is None:
            return
        try:
            if has_analysis(root):
                answer = QMessageBox.question(
                    self._host,
                    "Preserve previous analysis",
                    "Create an editable copy with the original recordings and no prior analysis results?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
                library = self.library()
                library.register(root)
                root = library.editable_copy(root)
                self._open(root)
            editor = SwingEditor(root, self._host)
            editor.exec()
        except (ValueError, OSError, sqlite3.Error) as exc:
            self._error(exc)
