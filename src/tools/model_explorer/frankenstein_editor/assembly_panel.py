"""Drag-and-drop assembly tab: part library, canvas, validation, undo/redo."""

from __future__ import annotations

from PyQt6.QtCore import QByteArray, QMimeData, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QDrag, QKeySequence
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.tools.model_explorer.assembly_session import AssemblySession
from src.tools.model_explorer.frankenstein_editor.assembly_canvas import (
    PART_MIME,
    AssemblyCanvas,
)
from src.tools.model_explorer.part_catalog import PartCatalog

_ROLE_PART = Qt.ItemDataRole.UserRole
BASE_PARTS = ("humanoid_torso", "pedestal")


class PartLibraryTree(QTreeWidget):
    """Category tree of catalog parts; rows can be dragged onto the canvas."""

    def __init__(self, catalog: PartCatalog, parent: QWidget | None = None) -> None:
        if catalog is None:
            raise ValueError("catalog must be provided")
        super().__init__(parent)
        self.catalog = catalog
        self.setHeaderLabels(["Part Library"])
        self.setDragEnabled(True)
        self.populate("")

    def populate(self, query: str) -> None:
        """Rebuild the tree for a search query."""
        self.clear()
        for category, label in self.catalog.categories():
            parts = self.catalog.list_parts(category=category, query=query)
            if not parts:
                continue
            top = QTreeWidgetItem([label])
            top.setFlags(top.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)
            self.addTopLevelItem(top)
            for part in parts:
                child = QTreeWidgetItem([part.name])
                child.setData(0, _ROLE_PART, part.part_id)
                child.setToolTip(0, part.description)
                top.addChild(child)
            top.setExpanded(True)

    def part_id_at(self, item: QTreeWidgetItem) -> str | None:
        """Part id carried by a row, or None for category headers."""
        value = item.data(0, _ROLE_PART)
        return str(value) if value else None

    def startDrag(self, supported_actions) -> None:  # noqa: ANN001
        item = self.currentItem()
        part_id = self.part_id_at(item) if item is not None else None
        if part_id is None:
            return
        mime = QMimeData()
        mime.setData(PART_MIME, QByteArray(part_id.encode("utf-8")))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


class AssemblyPanel(QWidget):
    """Library + canvas + live validation for drag-and-drop composition."""

    assembly_changed = pyqtSignal(str)  # URDF xml, for the 3D preview

    def __init__(
        self, catalog: PartCatalog | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.catalog = catalog or PartCatalog.bundled()
        self.session = AssemblySession(self.catalog, BASE_PARTS[0])
        self._build_ui()
        self._bind_session()
        self._on_session_changed()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.base_combo = QComboBox()
        for part_id in BASE_PARTS:
            self.base_combo.addItem(self.catalog.get(part_id).name, part_id)
        self.new_btn = QPushButton("New Assembly")
        self.undo_btn = QPushButton("Undo")
        self.redo_btn = QPushButton("Redo")
        self.detach_btn = QPushButton("Detach Selected")
        for widget in (
            QLabel("Base:"),
            self.base_combo,
            self.new_btn,
            self.undo_btn,
            self.redo_btn,
            self.detach_btn,
        ):
            bar.addWidget(widget)
        bar.addStretch()
        root.addLayout(bar)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search parts")
        self.library = PartLibraryTree(self.catalog)
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.addWidget(self.search)
        left_layout.addWidget(self.library)
        self.canvas = AssemblyCanvas(self.session)
        self.validation_list = QListWidget()
        self.validation_list.setWordWrap(True)
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.addWidget(QLabel("Live Validation"))
        right_layout.addWidget(self.validation_list)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        for pane in (left_pane, self.canvas, right_pane):
            splitter.addWidget(pane)
        splitter.setSizes([220, 560, 260])
        root.addWidget(splitter, 1)
        self.status_label = QLabel("Drag a part from the library onto a socket")
        root.addWidget(self.status_label)
        self._add_shortcuts()
        self._connect_widgets()

    def _add_shortcuts(self) -> None:
        for text, keys, slot in (
            ("Undo", QKeySequence.StandardKey.Undo, self.undo),
            ("Redo", QKeySequence.StandardKey.Redo, self.redo),
        ):
            action = QAction(text, self)
            action.setShortcut(keys)
            action.triggered.connect(slot)
            self.addAction(action)

    def _connect_widgets(self) -> None:
        self.new_btn.clicked.connect(self.new_assembly)
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn.clicked.connect(self.redo)
        self.detach_btn.clicked.connect(self.detach_selected)
        self.search.textChanged.connect(self.library.populate)
        self.canvas.drop_rejected.connect(self._show_rejection)
        self.canvas.hover_reason.connect(self.status_label.setText)

    def _bind_session(self) -> None:
        self.session.subscribe(self._on_session_changed)

    # ----------------------------------------------------------------- actions
    def new_assembly(self) -> None:
        """Start over from the selected base part."""
        base = str(self.base_combo.currentData())
        self.session = AssemblySession(self.catalog, base)
        self._bind_session()
        self.canvas.set_session(self.session)
        self._on_session_changed()
        self.status_label.setText(f"New assembly on {self.base_combo.currentText()}")

    def undo(self) -> None:
        """Undo the last drop or detach."""
        if self.session.undo():
            self.status_label.setText("Undid last change")

    def redo(self) -> None:
        """Redo the last undone change."""
        if self.session.redo():
            self.status_label.setText("Redid change")

    def detach_selected(self) -> None:
        """Detach the part selected on the canvas, with anything mated to it."""
        instance = self.canvas.selected_instance()
        if not instance:
            self.status_label.setText("Click a part on the canvas first")
            return
        try:
            removed = self.session.detach(instance)
        except (ValueError, KeyError) as exc:
            self.status_label.setText(str(exc))
            return
        self.status_label.setText(f"Detached {len(removed)} part(s)")

    # --------------------------------------------------------------- reactions
    def _show_rejection(self, reason: str) -> None:
        self.status_label.setText(f"Rejected: {reason}")

    def _on_session_changed(self) -> None:
        self.canvas.refresh()
        self.undo_btn.setEnabled(self.session.can_undo)
        self.redo_btn.setEnabled(self.session.can_redo)
        self._refresh_validation()
        self.assembly_changed.emit(self.session.to_urdf(force=True))

    def _refresh_validation(self) -> None:
        self.validation_list.clear()
        findings = self.session.model.validate_composition().findings
        if not findings:
            self.validation_list.addItem("Valid: no findings")
        for finding in findings:
            self.validation_list.addItem(
                f"{finding.severity.upper()} {finding.code}: {finding.message}"
            )
