"""Review named lens calibration revisions without losing earlier captures."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.rig.plan import CameraBinding, RigPlan

from .calibration_profiles import (
    CalibrationProfile,
    CameraSetup,
    ProfileAssignment,
    ProfileHistory,
    save_profile,
    write_profile_set,
)


class CameraProfilePanel(QWidget):
    """One camera's operator-confirmed lens settings and saved revisions."""

    def __init__(self, binding: CameraBinding, history_path: Path) -> None:
        super().__init__()
        self.binding, self.history_path = binding, history_path
        self.profiles = QComboBox()
        self.fields = {
            key: QLineEdit() for key in ("lens", "zoom", "focus", "sensor_mode")
        }
        self.name = QLineEdit()
        self.confirmed = QCheckBox("I Verified These Camera Settings")
        self.confirmed.setToolTip(
            "Check the physical lens settings used for the selected recording or setup."
        )
        self._build_form()
        self._reload()
        self.profiles.currentIndexChanged.connect(self._select)
        for field in self.fields.values():
            field.textChanged.connect(lambda: self.confirmed.setChecked(False))

    def _build_form(self) -> None:
        form = QFormLayout(self)
        form.addRow(
            "Camera / view", QLabel(f"{self.binding.identity} / {self.binding.view}")
        )
        mode = self.binding.mode
        form.addRow("Image size", QLabel(f"{mode.width} × {mode.height} pixels"))
        form.addRow("Saved revision", self.profiles)
        labels = {
            "lens": "Lens",
            "zoom": "Optical zoom",
            "focus": "Focus",
            "sensor_mode": "Sensor / crop mode",
        }
        for key, label in labels.items():
            field = self.fields[key]
            field.setPlaceholderText(
                f"Record the {label.lower()} used for this capture"
            )
            form.addRow(label, field)
        form.addRow("New revision name", self.name)
        form.addRow(self.confirmed)
        save = QPushButton("Save From Calibration File…")
        save.clicked.connect(self._save)
        form.addRow(save)

    def _reload(self, selected_id: str | None = None) -> None:
        history = (
            ProfileHistory.model_validate_json(self.history_path.read_bytes())
            if self.history_path.exists()
            else ProfileHistory()
        )
        self.profiles.blockSignals(True)
        self.profiles.clear()
        self.profiles.addItem("Select a saved revision…", None)
        for profile in history.profiles:
            if profile.setup.camera_identity == self.binding.identity:
                self.profiles.addItem(
                    f"{profile.name} · {profile.created_utc[:16]}", profile
                )
                if profile.profile_id == selected_id:
                    self.profiles.setCurrentIndex(self.profiles.count() - 1)
        self.profiles.blockSignals(False)
        self._select()

    def _select(self) -> None:
        profile = self.profiles.currentData()
        if isinstance(profile, CalibrationProfile):
            for key, field in self.fields.items():
                field.setText(getattr(profile.setup, key))
        self.confirmed.setChecked(False)

    def setup(self) -> CameraSetup:
        """Read declared settings, with camera identity and size from the rig."""
        mode = self.binding.mode
        return CameraSetup(
            camera_identity=self.binding.identity,
            image_size_px=(mode.width, mode.height),
            **{key: field.text() for key, field in self.fields.items()},
        )

    def assignment(self) -> ProfileAssignment:
        """Return the selection; export still validates its compatibility."""
        profile = self.profiles.currentData()
        if not isinstance(profile, CalibrationProfile):
            raise ValueError(f"Select a saved revision for {self.binding.view}")
        return ProfileAssignment(
            self.binding.view, profile, self.setup(), self.confirmed.isChecked()
        )

    def _save(self) -> None:
        try:
            if not self.confirmed.isChecked():
                raise ValueError(
                    "Confirm the lens settings used for this calibration first"
                )
            setup = self.setup()
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Intrinsic Calibration", "", "Calibration (*.json)"
            )
            if not path:
                return
            profile = CalibrationProfile.capture(
                name=self.name.text(),
                setup=setup,
                camera_id=self.binding.view,
                intrinsics_path=Path(path),
            )
            saved = save_profile(self.history_path, profile)
            self._reload(saved.profile_id)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Calibration Revision", str(exc))


class CalibrationDialog(QDialog):
    """A scrollable rig-wide review with an explicit recalibration exit."""

    def __init__(
        self,
        plan: RigPlan,
        root: Path,
        parent: QWidget | None = None,
        *,
        reference_available: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Camera Calibration")
        self.resize(720, 540)
        self.root, self.plan = root, plan
        self.output_path: Path | None = None
        self.recalibrate_requested = False
        self.reference_requested = False
        self.reuse_requested = False
        self._reference_available = reference_available
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Review each camera's lens settings before reusing calibration. Optical zoom and "
            "focus are entered manually; they are not detected automatically. For an existing "
            "recording, confirm the settings used when it was recorded. Moving a camera also "
            "requires updating its scene calibration."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        tabs = QTabWidget()
        self.panels = []
        for binding in plan.cameras:
            panel = CameraProfilePanel(binding, root / "profiles.json")
            self.panels.append(panel)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(panel)
            tabs.addTab(scroll, binding.view)
        layout.addWidget(tabs)
        self._buttons(layout)

    def _buttons(self, layout: QVBoxLayout) -> None:
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        apply = QPushButton("Use Reviewed Revisions")
        repeat = QPushButton("Recalibrate Again…")
        reference = QPushButton("Paper / Ruler References…")
        reuse = QPushButton("Reuse a Camera Layout…")
        reuse.setEnabled(self._reference_available)
        reuse.setToolTip(
            "Review a previous capture's camera layout for the selected swing."
        )
        reuse.clicked.connect(self._reuse)
        reference.setEnabled(self._reference_available)
        reference.setToolTip(
            "Mark common references in the selected capture. Open or record a capture first."
        )
        buttons.addButton(apply, QDialogButtonBox.ButtonRole.AcceptRole)
        actions = QGridLayout()
        actions.addWidget(reference, 0, 0)
        actions.addWidget(reuse, 0, 1)
        actions.addWidget(repeat, 1, 0, 1, 2)
        layout.addLayout(actions)
        reference.clicked.connect(self._reference)
        apply.clicked.connect(self._apply)
        repeat.clicked.connect(self._repeat)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _apply(self) -> None:
        try:
            target = self.root / f"intrinsics-selected-{uuid4()}.json"
            write_profile_set(
                target,
                [panel.assignment() for panel in self.panels],
                required_views=tuple(camera.view for camera in self.plan.cameras),
            )
            self.output_path = target
            self.accept()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Review Camera Settings", str(exc))

    def _repeat(self) -> None:
        self.recalibrate_requested = True
        self.accept()

    def _reference(self) -> None:
        self.reference_requested = True
        self.accept()

    def _reuse(self) -> None:
        self.reuse_requested = True
        self.accept()
