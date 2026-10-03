"""OpenCap session import action and dialog for PyQt6 and OpenSim (#11409).

Lists the trials in an OpenCap session directory, allows user selection,
loads the chosen trial via :func:`load_opencap_session`, and hands the scaled
OpenSim model and kinematics to the target OpenSim engine or widget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_pipeline.sources.opencap_session import (
    OpenCapSession,
    OpenCapSessionMetadata,
    inspect_opencap_session,
    load_opencap_session,
)

logger = get_logger(__name__)


class OpenCapImportAction:
    """Action that inspects OpenCap sessions, loads a trial, and hands off to OpenSim.

    Parameters:
        target_engine: Target OpenSim engine or widget receiving the model and kinematics.
        parent: Optional parent QWidget for modal dialogs.
    """

    def __init__(
        self,
        target_engine: Any = None,
        parent: QWidget | None = None,
    ) -> None:
        self.target_engine = target_engine
        self.parent = parent

    def inspect_session(self, session_dir: Path | str) -> OpenCapSessionMetadata:
        """Inspect an OpenCap session directory and return its metadata."""
        return inspect_opencap_session(session_dir)

    def import_session(
        self,
        session_dir: Path | str,
        trial: str | None = None,
    ) -> OpenCapSession:
        """Load the specified trial from the OpenCap session."""
        return load_opencap_session(Path(session_dir), trial=trial)

    def hand_off_to_engine(
        self,
        session: OpenCapSession,
        target: Any = None,
    ) -> None:
        """Pass the scaled model and kinematics to the OpenSim engine target."""
        engine = target if target is not None else self.target_engine
        if engine is None:
            return

        if hasattr(engine, "load_opencap_session") and callable(
            engine.load_opencap_session
        ):
            engine.load_opencap_session(session)
            return

        self._apply_model_to_engine(engine, session.model_file)
        self._apply_kinematics_to_engine(engine, session.kinematics)

        try:
            engine.opencap_session = session
        except (AttributeError, TypeError):
            pass

    @staticmethod
    def _apply_model_to_engine(engine: Any, model_file: Path | None) -> None:
        if model_file is None:
            return
        model_str = str(model_file)
        if hasattr(engine, "load_from_path") and callable(engine.load_from_path):
            engine.load_from_path(model_str)
        elif hasattr(engine, "load_model") and callable(engine.load_model):
            engine.load_model(model_str)
        elif hasattr(engine, "model_path"):
            engine.model_path = model_str

    @staticmethod
    def _apply_kinematics_to_engine(engine: Any, kinematics: Any) -> None:
        if kinematics is None:
            return
        if hasattr(engine, "set_kinematics") and callable(engine.set_kinematics):
            engine.set_kinematics(kinematics)
        elif hasattr(engine, "kinematics"):
            engine.kinematics = kinematics

    def open_import_dialog(self) -> OpenCapSession | None:
        """Open the interactive PyQt6 import dialog."""
        dialog = OpenCapImportDialog(action=self, parent=self.parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.imported_session
        return None


class OpenCapImportDialog(QDialog):
    """PyQt6 dialog for selecting an OpenCap session and trial."""

    def __init__(
        self,
        action: OpenCapImportAction,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.action = action
        self.session_dir: Path | None = None
        self.metadata: OpenCapSessionMetadata | None = None
        self.imported_session: OpenCapSession | None = None

        self.setWindowTitle("Import OpenCap Session")
        self.resize(520, 420)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Directory selection
        dir_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Path to OpenCap session folder...")
        self.path_edit.textChanged.connect(self._on_path_edited)
        dir_layout.addWidget(self.path_edit)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_directory)
        dir_layout.addWidget(self.btn_browse)
        layout.addLayout(dir_layout)

        # Status / Error label
        self.lbl_status = QLabel("Select an OpenCap session directory.")
        self.lbl_status.setStyleSheet("color: gray;")
        layout.addWidget(self.lbl_status)

        # Subject metadata display
        self.lbl_subject = QLabel("")
        self.lbl_subject.setStyleSheet("font-size: 11px; color: #444;")
        layout.addWidget(self.lbl_subject)

        # Trial selection list
        trial_label = QLabel("Select Trial to Import:")
        layout.addWidget(trial_label)

        self.trial_list = QListWidget()
        layout.addWidget(self.trial_list)

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_import = QPushButton("Import Trial")
        self.btn_import.setEnabled(False)
        self.btn_import.setStyleSheet(
            "background-color: #007acc; color: white; padding: 6px 14px; font-weight: bold;"
        )
        self.btn_import.clicked.connect(self.accept_import)
        btn_layout.addWidget(self.btn_import)

        layout.addLayout(btn_layout)

    def _browse_directory(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Select OpenCap Session Directory"
        )
        if chosen:
            self.set_session_dir(Path(chosen))

    def _on_path_edited(self, text: str) -> None:
        path = Path(text.strip())
        if path.is_dir():
            self.set_session_dir(path)

    def set_session_dir(self, session_dir: Path | str) -> None:
        """Scan and populate the session directory."""
        path = Path(session_dir)
        self.session_dir = path
        if self.path_edit.text() != str(path):
            self.path_edit.setText(str(path))

        try:
            self.metadata = self.action.inspect_session(path)
            self._update_trial_list(self.metadata.trials)
            self._update_subject_info(self.metadata)
            self.lbl_status.setText(f"Found {len(self.metadata.trials)} trials.")
            self.lbl_status.setStyleSheet("color: green;")
            self.btn_import.setEnabled(True)
        except Exception as exc:
            self.metadata = None
            self.trial_list.clear()
            self.lbl_subject.setText("")
            self.lbl_status.setText(f"Error: {exc}")
            self.lbl_status.setStyleSheet("color: red;")
            self.btn_import.setEnabled(False)
            raise

    def _update_trial_list(self, trials: list[str]) -> None:
        self.trial_list.clear()
        for trial in trials:
            self.trial_list.addItem(trial)

        # Default selection: first non-neutral trial if available, else first item
        selected_index = 0
        for i, trial in enumerate(trials):
            if trial.lower() != "neutral":
                selected_index = i
                break
        if trials:
            self.trial_list.setCurrentRow(selected_index)

    def _update_subject_info(self, meta: OpenCapSessionMetadata) -> None:
        subj = meta.subject
        parts = []
        if subj.mass_kg is not None:
            parts.append(f"Mass: {subj.mass_kg:.1f} kg")
        if subj.height_m is not None:
            parts.append(f"Height: {subj.height_m:.2f} m")
        if subj.opensim_model:
            parts.append(f"Model: {subj.opensim_model}")
        if meta.model_file:
            parts.append(f"Scaled .osim: {meta.model_file.name}")
        self.lbl_subject.setText(" | ".join(parts) if parts else "No subject metadata")

    def selected_trial(self) -> str | None:
        """Return the currently selected trial name."""
        current = self.trial_list.currentItem()
        return current.text() if current is not None else None

    def accept_import(self) -> None:
        """Load selected trial and hand off to target engine."""
        trial = self.selected_trial()
        if self.session_dir is None or trial is None:
            QMessageBox.warning(
                self, "Selection Required", "Please select a trial to import."
            )
            return

        try:
            session = self.action.import_session(self.session_dir, trial=trial)
            self.action.hand_off_to_engine(session)
            self.imported_session = session
            self.accept()
        except Exception as exc:
            logger.exception("Failed to import OpenCap session: %s", exc)
            QMessageBox.critical(
                self, "Import Failed", f"Failed to import OpenCap session:\n\n{exc}"
            )
