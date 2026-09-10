"""Native mapping step and existing video-reader adapter for reference imports."""

from pathlib import Path
from typing import Literal, cast

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.provenance import sha256_of
from src.motion_capture.reference import ReferenceSource, ReferenceVideo
from src.motion_capture.reference.importers import MotionDraft, MotionImportOptions
from src.motion_capture.reference.model import Axis
from src.shared.python.ui.qt import create_button

from .player import VideoReader
from . import styling


def load_reference_video(path: Path) -> ReferenceVideo:
    """Inspect a linked video without modifying it or inferring a 3D viewpoint."""
    path = path.expanduser().resolve()
    digest = sha256_of(path)
    with VideoReader(path) as reader:
        if reader.read(0) is None:
            raise ValueError("Reference video has no decodable first frame")
        asset = ReferenceVideo(
            title=path.stem,
            source=ReferenceSource(path=str(path), sha256=digest, format="video"),
            width=reader.width,
            height=reader.height,
            frames=reader.frame_count,
            fps=reader.fps,
        )
    if sha256_of(path) != digest:
        raise ValueError("Reference video changed during import; retry after saving it")
    return asset


class ReferenceMappingDialog(QDialog):
    """Explicit source-to-reference mapping, including unknown unit/axis choices."""

    def __init__(self, draft: MotionDraft, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.draft = draft
        self.setWindowTitle("Map Motion Reference")
        self.resize(640, 650)
        layout = QVBoxLayout(self)
        description = QLabel(
            "Confirm the source convention before importing. Reference coordinates use "
            "metres in a right-handed frame with Z up. Camera placement is configured "
            "separately. Missing samples stay missing."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        form = QFormLayout()
        self.title = QLineEdit(Path(draft.source.path).stem)
        self.title.setMaxLength(200)
        form.addRow("Reference title", self.title)
        self.units = QComboBox()
        self.units.addItems(["Choose units…", "m", "cm", "mm"])
        form.addRow(f"Source units (reader reports {draft.source_units})", self.units)
        self.axes: list[QComboBox] = []
        for target in "XYZ":
            box = QComboBox()
            box.addItems(["Choose source axis…", "+X", "-X", "+Y", "-Y", "+Z", "-Z"])
            form.addRow(f"Reference {target} from", box)
            self.axes.append(box)
        self.model_identity = QLineEdit()
        self.model_identity.setText(draft.model_identity or "")
        form.addRow("Model identity (optional)", self.model_identity)
        layout.addLayout(form)
        self.names = QTableWidget(len(draft.names), 2)
        self.names.setHorizontalHeaderLabels(["Source marker", "Reference joint name"])
        header = self.names.horizontalHeader()
        assert header is not None
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for row, name in enumerate(draft.names):
            source = QTableWidgetItem(name)
            source.setFlags(source.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.names.setItem(row, 0, source)
            self.names.setItem(row, 1, QTableWidgetItem(name))
        layout.addWidget(self.names, 1)
        layout.addWidget(
            QLabel(
                "Skeleton connections (optional): one joint name, joint name pair per line"
            )
        )
        self.edges = QPlainTextEdit()
        self.edges.setMaximumHeight(80)
        self.edges.setPlaceholderText("pelvis, spine\nspine, shoulder")
        layout.addWidget(self.edges)
        self.confirm = QCheckBox(
            "I confirm these units, axes and marker-to-joint names"
        )
        layout.addWidget(self.confirm)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(self)
        buttons.addButton(
            create_button("Import Reference", self.accept),
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        buttons.addButton(
            create_button("Cancel", self.reject), QDialogButtonBox.ButtonRole.RejectRole
        )
        layout.addWidget(buttons)
        if draft.canonical or draft.source.format == "simulation-trace/2":
            self.units.setCurrentText("m")
            self.units.setEnabled(False)
        if draft.canonical:
            for target, box in zip("XYZ", self.axes, strict=True):
                box.setCurrentText("+" + target)
                box.setEnabled(False)
        styling.apply_theme(self)

    def options(self) -> MotionImportOptions:
        if (
            not self.confirm.isChecked()
            or self.units.currentIndex() == 0
            or any(b.currentIndex() == 0 for b in self.axes)
        ):
            raise ValueError("Choose and confirm units, all three axes and joint names")
        names = tuple(
            cast(QTableWidgetItem, self.names.item(i, 1)).text().strip()
            for i in range(self.names.rowCount())
        )
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("Reference joint names must be non-empty and unique")
        axes = tuple(box.currentText() for box in self.axes)
        if len({axis[-1] for axis in axes}) != 3:
            raise ValueError("Assign each source axis exactly once")
        edges: list[tuple[int, int]] = []
        for line in self.edges.toPlainText().splitlines():
            if not line.strip():
                continue
            pair = [name.strip() for name in line.split(",")]
            if len(pair) != 2 or any(name not in names for name in pair):
                raise ValueError(
                    "Each connection needs two existing reference joint names separated by a comma"
                )
            edges.append((names.index(pair[0]), names.index(pair[1])))
        return {
            "title": self.title.text(),
            "units": cast(Literal["m", "cm", "mm"], self.units.currentText()),
            "axes": cast(tuple[Axis, Axis, Axis], axes),
            "joint_names": names,
            "edges": tuple(edges),
            "model_identity": self.model_identity.text().strip() or None,
        }

    def accept(self) -> None:
        try:
            self.options()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        super().accept()
