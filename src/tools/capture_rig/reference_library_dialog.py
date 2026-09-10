"""Reference library: background imports, explicit mapping and portable notes."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

from PyQt6.QtCore import QThread, QUrl
from PyQt6.QtGui import QCloseEvent, QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.reference import Asset, ReferenceMotion, ReferenceVideo
from src.motion_capture.reference.importers import (
    MotionDraft,
    finish_motion_import,
    load_motion_draft,
)
from src.motion_capture.reference.storage import ReferenceLibrary, ReferenceScan

from . import styling
from .flow_layout import FlowLayout
from .reference_import import ReferenceMappingDialog, load_reference_video


class ReferenceTask(QThread):
    """One bounded operation; the owning dialog remains alive until finished."""

    def __init__(self, operation: Callable[[], object], parent: QWidget) -> None:
        super().__init__(parent)
        self.operation = operation
        self.result: object = None
        self.error = ""

    def run(self) -> None:
        try:
            self.result = self.operation()
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            self.error = str(exc)


class ReferenceLibraryDialog(QDialog):
    def __init__(
        self,
        library: ReferenceLibrary,
        parent: QWidget | None = None,
        *,
        capture_root: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.library = library
        self.capture_root = capture_root
        self._rows: list[Asset] = []
        self._asset: Asset | None = None

        self._visible_archived = False
        self._worker: ReferenceTask | None = None
        self._callback: Callable[[object], None] | None = None
        self.setWindowTitle("Expert Reference Library")
        self.resize(740, 660)
        layout = QVBoxLayout(self)
        self.content = QWidget()
        form = QVBoxLayout(self.content)
        description = QLabel(
            "Import an expert video, C3D recording or model marker animation. Videos stay 2D; motion references retain their own clock and require camera registration before overlay."
        )
        description.setWordWrap(True)
        form.addWidget(description)
        actions = FlowLayout(spacing=6)
        for title, callback in (
            ("Import motion…", self.import_motion),
            ("Import video…", self.import_video),
            ("Refresh", self.refresh),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            actions.addWidget(button)
        form.addLayout(actions)
        self.archived = QCheckBox("Archived references")
        self.archived.toggled.connect(self.refresh)
        form.addWidget(self.archived)
        self.items = QListWidget()
        self.items.currentRowChanged.connect(self.select_row)
        form.addWidget(self.items, 1)
        self.title = QLineEdit()
        self.title.setMaxLength(200)
        self.title.setAccessibleName("Reference title")
        form.addWidget(QLabel("Reference title"))
        form.addWidget(self.title)
        self.notes = QPlainTextEdit()
        self.notes.setAccessibleName("Reference notes")
        self.notes.setPlaceholderText(
            "Why this swing is a useful comparison, instructor cues, model provenance…"
        )
        form.addWidget(self.notes, 1)
        self.details = QLabel()
        self.details.setWordWrap(True)
        form.addWidget(self.details)
        self.source = QLineEdit()
        self.source.setReadOnly(True)
        self.source.setAccessibleName("Reference source file")
        form.addWidget(self.source)
        actions = FlowLayout(spacing=6)
        for title, callback in (
            ("Save title and notes", self.save_notes),
            ("Compare with capture…", self.compare_with_capture),
            ("Analyze Model…", self.analyze_model),
            ("Archive / restore", self.archive_selected),
            ("Open source", self.open_source),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            actions.addWidget(button)
        form.addLayout(actions)

        layout.addWidget(self.content)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        styling.apply_theme(self)
        self.refresh()

    def _start(
        self, operation: Callable[[], object], callback: Callable[[object], None]
    ) -> None:
        if self._worker is not None:
            return
        self.content.setEnabled(False)
        self.progress.show()
        self.status.setText("Working…")
        self._callback = callback
        self._worker = ReferenceTask(operation, self)
        self._worker.finished.connect(self._finished)
        self._worker.start()

    def _finished(self) -> None:
        worker, callback = self._worker, self._callback
        assert worker is not None
        self._worker, self._callback = None, None
        self.content.setEnabled(True)
        self.progress.hide()
        worker.deleteLater()
        if worker.error:
            self.status.setText(worker.error)
        elif callback is not None:
            self.status.setText("")
            callback(worker.result)

    def _leave(self) -> bool:
        if self._asset is None or (self.title.text(), self.notes.toPlainText()) == (
            self._asset.title,
            self._asset.notes,
        ):
            return True
        choice = QMessageBox.question(
            self,
            "Unsaved Reference Notes",
            "Discard unsaved changes to this reference?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return choice == QMessageBox.StandardButton.Discard

    def refresh(self) -> None:
        if self._worker is not None:
            return
        if not self._leave():
            self.archived.blockSignals(True)
            self.archived.setChecked(self._visible_archived)
            self.archived.blockSignals(False)
            return
        archived = self.archived.isChecked()
        self._start(lambda: self.library.scan(archived=archived), self._show_rows)

    def _show_rows(self, result: object) -> None:
        self._visible_archived = self.archived.isChecked()
        scan = cast(ReferenceScan, result)
        self._rows = list(scan.assets)
        self._asset = None
        self.items.clear()
        self.items.addItems(
            [
                f"{a.title}  ·  {'3D motion' if a.kind == 'motion' else '2D video'}"
                for a in self._rows
            ]
        )
        self.title.clear()
        self.notes.clear()
        self.details.clear()
        self.source.clear()
        if self._rows:
            self.items.setCurrentRow(0)
        else:
            self.status.setText(
                "No references here yet. Import a motion or video to begin."
            )
        if scan.problems:
            self.status.setText(
                "Some references need attention: " + "\n".join(scan.problems)
            )

    def select_row(self, row: int) -> None:
        if not 0 <= row < len(self._rows):
            return
        if not self._leave():
            self.items.blockSignals(True)
            self.items.setCurrentRow(
                self._rows.index(self._asset) if self._asset else -1
            )
            self.items.blockSignals(False)
            return
        self._asset = self._rows[row]
        self.title.setText(self._asset.title)
        self.notes.setPlainText(self._asset.notes)
        asset = self._asset
        summary = (
            f"{len(asset.time_s)} samples · {len(asset.joint_names)} joints · metres, Z up"
            if isinstance(asset, ReferenceMotion)
            else f"{asset.width}×{asset.height} · {asset.frames} frames · {asset.fps:g} fps · 2D only"
        )
        self.details.setText(summary)
        self.source.setText(asset.source.path)
        self.source.setToolTip(f"Source SHA-256: {asset.source.sha256}")

    def _save(self, asset: Asset) -> None:
        self._start(lambda: self.library.save(asset), lambda _: self._saved(asset))

    def _saved(self, asset: Asset) -> None:
        self._asset = asset
        self.title.setText(asset.title)
        self.notes.setPlainText(asset.notes)
        self.refresh()

    def save_notes(self) -> None:
        if self._asset is None:
            return
        try:
            self._save(
                self._asset.changed(
                    title=self.title.text(), notes=self.notes.toPlainText()
                )
            )
        except ValueError as exc:
            self.status.setText(str(exc))

    def compare_with_capture(self) -> None:
        if self._asset is None or not self._leave():
            return
        capture_path = self.capture_root
        if capture_path is None:
            folder = QFileDialog.getExistingDirectory(
                self, "Select Capture Session Folder"
            )
            if not folder:
                return
            capture_path = Path(folder)
        from .reference_comparison import show_reference_comparison

        show_reference_comparison(capture_path, self.library, self)

    def analyze_model(self) -> None:
        """Open model playback and drawing tools without requiring a capture."""
        if not isinstance(self._asset, ReferenceMotion):
            self.status.setText("Select a motion reference for model analysis.")
            return
        if not self._leave():
            return
        from .model_analysis_dialog import ModelAnalysisDialog

        try:
            ModelAnalysisDialog(
                self._asset, self.library.root / "analysis" / self._asset.id, self
            ).exec()
        except (ValueError, OSError) as exc:
            self.status.setText(f"Model Analysis Unavailable: {exc}")

    def archive_selected(self) -> None:
        if self._asset is not None and self._leave():
            self._save(self._asset.changed(archived=not self._asset.archived))

    def import_motion(self) -> None:
        if not self._leave():
            return
        name, _ = QFileDialog.getOpenFileName(
            self, "Import Motion Reference", "", "Motion references (*.c3d *.json)"
        )
        if name:
            self._start(lambda: load_motion_draft(Path(name)), self._map_motion)

    def _map_motion(self, result: object) -> None:
        draft = cast(MotionDraft, result)
        dialog = ReferenceMappingDialog(draft, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            options = dialog.options()
            self._start(
                lambda: finish_motion_import(draft, **options),
                lambda asset: self._save(cast(ReferenceMotion, asset)),
            )

    def import_video(self) -> None:
        if not self._leave():
            return
        name, _ = QFileDialog.getOpenFileName(
            self, "Import Expert Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if name:
            self._start(
                lambda: load_reference_video(Path(name)),
                lambda asset: self._save(cast(ReferenceVideo, asset)),
            )

    def open_source(self) -> None:
        if self._asset is not None:
            path = self._asset.source_path
            if path.is_file():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            else:
                self.status.setText(
                    "Linked source is missing. The saved motion remains available; restore the source at its recorded location."
                )

    def closeEvent(self, event: QCloseEvent | None) -> None:
        if event is None:
            return
        if self._worker is not None:
            self.status.setText(
                "Please wait for the current reference operation to finish before closing."
            )
            event.ignore()
        elif self._leave():
            event.accept()
        else:
            event.ignore()

    def reject(self) -> None:
        self.close()
