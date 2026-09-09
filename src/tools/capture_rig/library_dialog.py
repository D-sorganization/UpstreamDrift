"""Search, notes, storage and reversible management of capture bundles (#9861)."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QThread, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.rig.bundle import load_bundle
from src.motion_capture.rig.edits import has_analysis

from . import styling
from .capture_library import CaptureLibrary, LibraryEntry, read_notes
from .flow_layout import FlowLayout
from .swing_editor import SwingEditor
from .coaching_dialog import show_coaching


class LibraryScan(QThread):
    rows = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self, library: CaptureLibrary, query: str, archived: bool, parent: QWidget
    ) -> None:
        super().__init__(parent)
        self.library, self.query, self.archived = library, query, archived

    def run(self) -> None:
        try:
            self.rows.emit(
                self.library.list(
                    query=self.query,
                    archived=self.archived,
                    cancelled=self.isInterruptionRequested,
                )
            )
        except InterruptedError:
            pass
        except (ValueError, OSError, sqlite3.Error) as exc:
            self.failed.emit(str(exc))


def _size(value: int) -> str:
    for unit, scale in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if value >= scale:
            return f"{value / scale:.1f} {unit}"
    return f"{value} B"


class LibraryDialog(QDialog):
    def __init__(
        self,
        library: CaptureLibrary,
        *,
        open_capture: Callable[[Path], None],
        import_videos: Callable[[], None] | None = None,
        location_changed: Callable[[CaptureLibrary], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.library, self._open = library, open_capture
        self._import = import_videos
        self._location_changed = location_changed
        self._worker: LibraryScan | None = None
        self._refresh_pending = False
        self._closing = False
        self._visible_archived = False
        self._rows: list[LibraryEntry] = []
        self._selected: Path | None = None
        self._baseline = ("", "")
        self.setWindowTitle("Capture Library")
        self.resize(900, 650)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search capture names and swing notes")
        self.search.setAccessibleName("Search capture library")
        self.search.returnPressed.connect(self.refresh)
        self.archived = QCheckBox("Archived captures")
        self.archived.toggled.connect(self.refresh)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["Capture", "Session storage", "Linked media"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        assert header is not None
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.cellDoubleClicked.connect(lambda _r, _c: self.open_selected())
        self.title, self.notes = QLineEdit(), QPlainTextEdit()
        self.title.setMaxLength(200)
        self.title.setAccessibleName("Capture title")
        self.notes.setAccessibleName("Swing notes")
        self.notes.setPlaceholderText(
            "Lesson notes, club, swing cues, what to compare next…"
        )
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.archive_button = QPushButton("Archive")
        self.archive_button.clicked.connect(self.archive_selected)
        self._build()
        styling.apply_theme(self)
        self.refresh()

    @property
    def loading(self) -> bool:
        return self._worker is not None

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        self.location = QLineEdit(str(self.library.root))
        self.location.setReadOnly(True)
        self.location.setAccessibleName("Library location")
        location_row = QHBoxLayout()
        location_row.addWidget(self.location, 1)
        location_row.addWidget(self._button("Library folder…", self.choose_location))
        layout.addLayout(location_row)
        search = QHBoxLayout()
        search.addWidget(self.search, 1)
        search.addWidget(self._button("Search / refresh", self.refresh))
        search.addWidget(self.archived)
        layout.addLayout(search)
        splitter = QSplitter()
        splitter.addWidget(self.table)
        detail = QWidget()
        self._details = detail
        form = QVBoxLayout(detail)
        form.addWidget(QLabel("Capture title"))
        form.addWidget(self.title)
        form.addWidget(QLabel("Swing notes"))
        form.addWidget(self.notes, 1)
        form.addWidget(self._button("Save title and notes", self.save_notes))
        splitter.addWidget(detail)
        splitter.setSizes([520, 340])
        layout.addWidget(splitter, 1)
        actions = FlowLayout(spacing=6)
        for text, callback in (
            ("Open capture", self.open_selected),
            ("Edit swing…", self.edit_selected),
            ("Draw References…", self.draw_selected),
            ("Add session folder…", self.add_session),
            ("Open folder", self.open_folder),
            ("Rename recording…", self.rename_selected),
        ):
            actions.add_widget(self._button(text, callback))
        if self._import is not None:
            actions.add_widget(self._button("Import videos…", self._import_clicked))
        actions.add_widget(self.archive_button)
        layout.addLayout(actions)
        hint = QLabel(
            "Archive keeps files and frees no disk space. Linked media lives outside the session folder."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self.status)

    @staticmethod
    def _button(text: str, callback: Callable[..., object]) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(lambda _checked=False: callback())
        return button

    def refresh(self) -> None:
        if not self._leave_selection():
            self.archived.blockSignals(True)
            self.archived.setChecked(self._visible_archived)
            self.archived.blockSignals(False)
            return
        if self.loading:
            self._refresh_pending = True
            return
        self.status.setText("Reading captures and storage…")
        self.table.setEnabled(False)
        self._details.setEnabled(False)
        worker = LibraryScan(
            self.library, self.search.text().strip(), self.archived.isChecked(), self
        )
        self._worker = worker
        worker.rows.connect(self._loaded)
        worker.failed.connect(self.status.setText)
        worker.finished.connect(self._scan_finished)
        worker.start()

    def _loaded(self, rows: list[LibraryEntry]) -> None:
        if self._worker:
            self._visible_archived = self._worker.archived
        self._rows, self._selected, self._baseline = rows, None, ("", "")
        self.title.clear()
        self.notes.clear()
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = (row.title, _size(row.managed_bytes), _size(row.external_bytes))
            for j, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(row.problem or str(row.root))
                self.table.setItem(i, j, item)
        self.table.clearSelection()
        self.table.blockSignals(False)
        problems = sum(bool(row.problem) for row in rows)
        self.status.setText(
            f"{len(rows)} captures · {problems} need attention"
            if rows
            else "No matching captures. Add a session folder or import a video to begin."
        )
        self.archive_button.setText(
            "Restore" if self.archived.isChecked() else "Archive"
        )

    def _scan_finished(self) -> None:
        worker, self._worker = self._worker, None
        if worker:
            worker.deleteLater()
        self.table.setEnabled(True)
        self._details.setEnabled(True)
        if self._closing:
            self.close()
        elif self._refresh_pending:
            self._refresh_pending = False
            self.refresh()

    def _selection_changed(self) -> None:
        row = self.table.currentRow()
        if not 0 <= row < len(self._rows):
            return
        entry = self._rows[row]
        if entry.root == self._selected:
            return
        previous = self._selected
        if not self._leave_selection():
            self.table.blockSignals(True)
            for i, candidate in enumerate(self._rows):
                if candidate.root == previous:
                    self.table.selectRow(i)
            self.table.blockSignals(False)
            return
        self._selected = entry.root
        try:
            notes = read_notes(entry.root)
            self.title.setText(notes.title)
            self.notes.setPlainText(notes.notes)
            self._baseline = (notes.title, notes.notes)
            self.status.setText(entry.problem or str(entry.root))
        except (ValueError, OSError) as exc:
            self._selected = None
            self.title.clear()
            self.notes.clear()
            self.status.setText(str(exc))

    def save_notes(self) -> bool:
        if self._selected is None:
            self.status.setText("Select a capture first")
            return False
        try:
            result = self.library.update(
                self._selected, title=self.title.text(), notes=self.notes.toPlainText()
            )
            self._baseline = (result.title, result.notes)
            self.title.setText(result.title)
            row = self.table.currentRow()
            if row >= 0 and (item := self.table.item(row, 0)) is not None:
                item.setText(result.title)
            self.status.setText("Title and swing notes saved")
            return True
        except (ValueError, OSError, sqlite3.Error) as exc:
            self.status.setText(f"Not saved: {exc}")
            return False

    def _leave_selection(self) -> bool:
        if (
            self._selected is None
            or (self.title.text(), self.notes.toPlainText()) == self._baseline
        ):
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved swing notes",
            "Save your changes before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save_notes()
        if answer == QMessageBox.StandardButton.Discard:
            self.title.setText(self._baseline[0])
            self.notes.setPlainText(self._baseline[1])
            return True
        return False

    def _operate(self, operation: Callable[[Path], None]) -> None:
        if self._selected is None:
            self.status.setText("Select a capture first")
        elif self._leave_selection():
            try:
                operation(self._selected)
            except (ValueError, OSError, sqlite3.Error) as exc:
                self.status.setText(str(exc))

    def open_selected(self) -> None:
        def open_root(root: Path) -> None:
            self.library.recover_rename(root)
            self._open(root)
            self.close()

        self._operate(open_root)

    def archive_selected(self) -> None:
        def archive(root: Path) -> None:
            self.library.update(root, archived=not self.archived.isChecked())
            self.refresh()

        self._operate(archive)

    def edit_selected(self) -> None:
        def edit(root: Path) -> None:
            if has_analysis(root):
                choice = QMessageBox.question(
                    self,
                    "Preserve previous analysis",
                    "Create an editable copy with the original recordings and no prior analysis results?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                )
                if choice != QMessageBox.StandardButton.Yes:
                    return
                root = self.library.editable_copy(root)
            self.library.recover_rename(root)
            editor = SwingEditor(root, self)
            editor.exec()
            self._open(root)
            self.refresh()

        self._operate(edit)

    def add_session(self) -> None:
        if not self._leave_selection():
            return
        folder = QFileDialog.getExistingDirectory(self, "Add capture session folder")
        if folder:
            try:
                self.library.register(Path(folder))
                self.refresh()
            except (ValueError, OSError, sqlite3.Error) as exc:
                self.status.setText(str(exc))

    def draw_selected(self) -> None:
        if self._leave_selection():
            self._operate(lambda root: show_coaching(root, self))

    def choose_location(self) -> None:
        if self.loading:
            self.status.setText(
                "Finish the current scan before changing the library folder"
            )
            return
        if not self._leave_selection():
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Choose library catalog folder", str(self.library.root)
        )
        if folder:
            try:
                self.library = CaptureLibrary(Path(folder))
                self.location.setText(str(self.library.root))
                if self._location_changed:
                    self._location_changed(self.library)
                self.refresh()
            except (ValueError, OSError, sqlite3.Error) as exc:
                self.status.setText(str(exc))

    def open_folder(self) -> None:
        def show(root: Path) -> None:
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(root))):
                self.status.setText("Could not open the capture folder")

        self._operate(show)

    def rename_selected(self) -> None:
        def rename(root: Path) -> None:
            views = [e.view for e in load_bundle(root)[1].recordings]
            view, ok = QInputDialog.getItem(
                self, "Rename recording", "Camera view", views, editable=False
            )
            if not ok:
                return
            old = next(
                e.file for e in load_bundle(root)[1].recordings if e.view == view
            )
            name, ok = QInputDialog.getText(
                self,
                "Rename recording",
                "Filename (keep extension)",
                text=Path(old).name,
            )
            if ok:
                self.library.rename_recording(root, view, name)
                self.refresh()

        self._operate(rename)

    def _import_clicked(self) -> None:
        if self._leave_selection() and self._import:
            self._import()
            self.close()

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802
        if event is None:
            return
        if not self._leave_selection():
            event.ignore()
        elif self._worker:
            self._closing = True
            self._worker.requestInterruption()
            self.status.setText("Finishing library scan…")
            event.ignore()
        else:
            event.accept()

    def reject(self) -> None:
        self.close()
