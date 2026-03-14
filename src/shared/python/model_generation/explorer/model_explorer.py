"""
Model Explorer PyQt6 Main Window.

Provides a GUI for browsing, loading, and previewing URDF and MJCF models
from the bundled library, local files, and remote repositories.

Display preview checkboxes (ordered):
  1. Segments   (checked by default)
  2. Joints     (checked by default)
  3. Collisions (checked by default)
  4. Inertias   (checked by default)
  5. Frames     (unchecked by default)
"""

from __future__ import annotations

import logging
import sys

from model_generation.explorer.display_config import DISPLAY_OPTIONS
from model_generation.library.unified_loader import (
    LoadResult,
    UnifiedModelLoader,
    UserPreferences,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# Catppuccin Mocha color palette
CATPPUCCIN_MOCHA = {
    "rosewater": "#f5e0dc",
    "flamingo": "#f2cdcd",
    "pink": "#f5c2e7",
    "mauve": "#cba6f7",
    "red": "#f38ba8",
    "maroon": "#eba0ac",
    "peach": "#fab387",
    "yellow": "#f9e2af",
    "green": "#a6e3a1",
    "teal": "#94e2d5",
    "sky": "#89dceb",
    "sapphire": "#74c7ec",
    "blue": "#89b4fa",
    "lavender": "#b4befe",
    "text": "#cdd6f4",
    "subtext1": "#bac2de",
    "subtext0": "#a6adc8",
    "overlay2": "#9399b2",
    "overlay1": "#7f849c",
    "overlay0": "#6c7086",
    "surface2": "#585b70",
    "surface1": "#45475a",
    "surface0": "#313244",
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
}

STYLESHEET = f"""
QMainWindow {{
    background-color: {CATPPUCCIN_MOCHA["base"]};
}}

QWidget {{
    background-color: {CATPPUCCIN_MOCHA["base"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    font-family: "Segoe UI", "Arial", sans-serif;
}}

QScrollArea {{
    border: none;
    background-color: {CATPPUCCIN_MOCHA["base"]};
}}

QTabWidget::pane {{
    border: 1px solid {CATPPUCCIN_MOCHA["surface1"]};
    background-color: {CATPPUCCIN_MOCHA["mantle"]};
    border-radius: 4px;
}}

QTabBar::tab {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["subtext1"]};
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}

QTabBar::tab:selected {{
    background-color: {CATPPUCCIN_MOCHA["surface1"]};
    color: {CATPPUCCIN_MOCHA["blue"]};
}}

QGroupBox {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface1"]};
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {CATPPUCCIN_MOCHA["mauve"]};
}}

QLabel {{
    color: {CATPPUCCIN_MOCHA["text"]};
    background-color: transparent;
}}

QListWidget {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    padding: 4px;
}}

QListWidget::item {{
    padding: 6px;
}}

QListWidget::item:selected {{
    background-color: {CATPPUCCIN_MOCHA["surface2"]};
    color: {CATPPUCCIN_MOCHA["blue"]};
}}

QTableWidget {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    gridline-color: {CATPPUCCIN_MOCHA["surface1"]};
}}

QTableWidget::item {{
    padding: 4px;
}}

QHeaderView::section {{
    background-color: {CATPPUCCIN_MOCHA["surface1"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    padding: 6px;
    border: none;
}}

QCheckBox {{
    color: {CATPPUCCIN_MOCHA["text"]};
    spacing: 8px;
    background-color: transparent;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 3px;
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
}}

QCheckBox::indicator:checked {{
    background-color: {CATPPUCCIN_MOCHA["blue"]};
    border-color: {CATPPUCCIN_MOCHA["blue"]};
}}

QComboBox {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    padding: 6px 10px;
}}

QTextEdit {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    padding: 8px;
    font-family: "Consolas", "Courier New", monospace;
}}

QPushButton {{
    background-color: {CATPPUCCIN_MOCHA["blue"]};
    color: {CATPPUCCIN_MOCHA["crust"]};
    border: none;
    border-radius: 4px;
    padding: 10px 24px;
    font-weight: bold;
}}

QPushButton:hover {{
    background-color: {CATPPUCCIN_MOCHA["sapphire"]};
}}

QPushButton:pressed {{
    background-color: {CATPPUCCIN_MOCHA["lavender"]};
}}

QPushButton#loadBtn {{
    background-color: {CATPPUCCIN_MOCHA["green"]};
}}

QPushButton#loadBtn:hover {{
    background-color: {CATPPUCCIN_MOCHA["teal"]};
}}

QPushButton#setDefaultBtn {{
    background-color: {CATPPUCCIN_MOCHA["peach"]};
}}

QPushButton#setDefaultBtn:hover {{
    background-color: {CATPPUCCIN_MOCHA["yellow"]};
}}
"""


class DisplayPreviewPanel(QGroupBox):
    """
    Display preview panel with checkboxes for controlling model visualization.

    Checkbox order:
      1. Segments   (checked)
      2. Joints     (checked)
      3. Collisions (checked)
      4. Inertias   (checked)
      5. Frames     (unchecked)
    """

    display_changed = pyqtSignal(str, bool)

    def __init__(self, preferences: UserPreferences, parent: QWidget | None = None):
        assert preferences is not None, "preferences must be provided"
        super().__init__("Display Preview", parent)
        self._preferences = preferences
        self._checkboxes: dict[str, QCheckBox] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        pref_map = {
            "segments": self._preferences.show_segments,
            "joints": self._preferences.show_joints,
            "collisions": self._preferences.show_collisions,
            "inertias": self._preferences.show_inertias,
            "frames": self._preferences.show_frames,
        }

        for key, label, _default in DISPLAY_OPTIONS:
            cb = QCheckBox(label)
            cb.setChecked(pref_map.get(key, _default))
            cb.toggled.connect(lambda checked, k=key: self._on_toggle(k, checked))
            layout.addWidget(cb)
            self._checkboxes[key] = cb

    def _on_toggle(self, key: str, checked: bool) -> None:
        """Handle checkbox toggle and update preferences."""
        assert key is not None, "key must be provided"
        attr_name = f"show_{key}"
        if hasattr(self._preferences, attr_name):
            setattr(self._preferences, attr_name, checked)
        self.display_changed.emit(key, checked)

    def get_display_state(self) -> dict[str, bool]:
        """Return current display checkbox states."""
        return {key: cb.isChecked() for key, cb in self._checkboxes.items()}


class ModelExplorerWindow(QMainWindow):
    """
    Main window for the Model Explorer application.

    Features:
    - Browse bundled model library (URDF + MJCF)
    - Load local URDF/MJCF files
    - Display preview checkboxes (Segments, Joints, Collisions, Inertias, Frames)
    - Set and persist default model preference
    - View model information (links, joints, DOF)
    """

    def __init__(self) -> None:
        super().__init__()
        self._loader = UnifiedModelLoader()
        self._current_result: LoadResult | None = None
        self._setup_ui()
        self._load_default_model()

    def _setup_ui(self) -> None:
        """Build the user interface."""
        self.setWindowTitle("Model Explorer")
        self.setMinimumSize(900, 700)
        self.setStyleSheet(STYLESHEET)

        # Central widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setCentralWidget(scroll_area)

        central = QWidget()
        scroll_area.setWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Title
        title = QLabel("Model Explorer")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['blue']};")
        main_layout.addWidget(title)

        # Top bar: load buttons + default model selector
        main_layout.addWidget(self._create_top_bar())

        # Splitter: library list | details panel
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: library model list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._create_library_panel())
        splitter.addWidget(left_panel)

        # Right: details + display preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Display preview checkboxes
        self.display_panel = DisplayPreviewPanel(self._loader.preferences)
        self.display_panel.display_changed.connect(self._on_display_changed)
        right_layout.addWidget(self.display_panel)

        # Details tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self._create_info_tab(), "Model Info")
        self.tab_widget.addTab(self._create_links_tab(), "Links")
        self.tab_widget.addTab(self._create_joints_tab(), "Joints")
        self.tab_widget.addTab(self._create_xml_tab(), "Source XML")
        right_layout.addWidget(self.tab_widget)

        splitter.addWidget(right_panel)
        splitter.setSizes([300, 600])

        main_layout.addWidget(splitter)

        # Status bar
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet(
            f"color: {CATPPUCCIN_MOCHA['subtext0']}; padding: 4px;"
        )
        main_layout.addWidget(self._status_label)

    # -- Top Bar --

    def _create_top_bar(self) -> QGroupBox:
        group = QGroupBox("Model Loading")
        layout = QHBoxLayout(group)
        layout.setSpacing(10)

        # Load from file
        load_file_btn = QPushButton("Load File...")
        load_file_btn.setObjectName("loadBtn")
        load_file_btn.clicked.connect(self._load_from_file)
        layout.addWidget(load_file_btn)

        # Default model selector
        layout.addWidget(QLabel("Default:"))
        self.default_combo = QComboBox()
        self._populate_default_combo()
        layout.addWidget(self.default_combo, stretch=1)

        # Set as default button
        set_default_btn = QPushButton("Set as Default")
        set_default_btn.setObjectName("setDefaultBtn")
        set_default_btn.clicked.connect(self._set_default_model)
        layout.addWidget(set_default_btn)

        # Load default button
        load_default_btn = QPushButton("Load Default")
        load_default_btn.clicked.connect(self._load_default_model)
        layout.addWidget(load_default_btn)

        return group

    def _populate_default_combo(self) -> None:
        """Populate the default model combo box from bundled library."""
        self.default_combo.clear()
        current_default = self._loader.preferences.default_model_id

        for entry in self._loader.list_bundled_models():
            model_id = entry["id"]
            display = f"{entry['name']} ({entry['format'].upper()})"
            self.default_combo.addItem(display, model_id)
            if model_id == current_default:
                self.default_combo.setCurrentIndex(self.default_combo.count() - 1)

    # -- Library Panel --

    def _create_library_panel(self) -> QGroupBox:
        group = QGroupBox("Model Library")
        layout = QVBoxLayout(group)

        # Category filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItem("All", "")
        self.category_combo.addItem("Humanoid", "humanoid")
        self.category_combo.addItem("Robot Arm", "robot_arm")
        self.category_combo.addItem("Quadruped", "quadruped")
        self.category_combo.currentIndexChanged.connect(self._filter_library)
        filter_layout.addWidget(self.category_combo, stretch=1)
        layout.addLayout(filter_layout)

        # Model list
        self.model_list = QListWidget()
        self.model_list.itemClicked.connect(self._on_model_selected)
        self.model_list.itemDoubleClicked.connect(self._on_model_double_clicked)
        layout.addWidget(self.model_list)

        # Populate
        self._populate_model_list()

        return group

    def _populate_model_list(self, category_filter: str = "") -> None:
        """Populate the model list from bundled library."""
        assert category_filter is not None, "category_filter must be provided"
        self.model_list.clear()
        for entry in self._loader.list_bundled_models():
            if category_filter and entry.get("category") != category_filter:
                continue
            item = QListWidgetItem(f"{entry['name']}  [{entry['format'].upper()}]")
            item.setData(Qt.ItemDataRole.UserRole, entry["id"])
            default_id = self._loader.preferences.default_model_id
            if entry["id"] == default_id:
                item.setForeground(item.foreground().color())
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.model_list.addItem(item)

    # -- Detail Tabs --

    def _create_info_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)

        info_group = QGroupBox("Model Information")
        info_layout = QGridLayout(info_group)
        info_layout.setSpacing(8)

        labels = [
            ("Name:", "name"),
            ("Format:", "format"),
            ("Links:", "links"),
            ("Joints:", "joints"),
            ("Total Mass:", "total_mass"),
            ("Root Link:", "root_link"),
            ("Source:", "source"),
        ]

        self.info_labels: dict[str, QLabel] = {}
        for row, (label_text, key) in enumerate(labels):
            info_layout.addWidget(QLabel(label_text), row, 0)
            value_label = QLabel("-")
            value_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['sapphire']};")
            self.info_labels[key] = value_label
            info_layout.addWidget(value_label, row, 1)

        layout.addWidget(info_group)
        layout.addStretch()
        return tab

    def _create_links_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)

        self.links_table = QTableWidget()
        self.links_table.setColumnCount(4)
        self.links_table.setHorizontalHeaderLabels(
            ["Name", "Mass (kg)", "Geometry", "Material"]
        )
        header = self.links_table.horizontalHeader()
        if header:
            header.setStretchLastSection(True)
        layout.addWidget(self.links_table)

        return tab

    def _create_joints_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)

        self.joints_table = QTableWidget()
        self.joints_table.setColumnCount(5)
        self.joints_table.setHorizontalHeaderLabels(
            ["Name", "Type", "Parent", "Child", "Axis"]
        )
        header = self.joints_table.horizontalHeader()
        if header:
            header.setStretchLastSection(True)
        layout.addWidget(self.joints_table)

        return tab

    def _create_xml_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)

        self.xml_view = QTextEdit()
        self.xml_view.setReadOnly(True)
        layout.addWidget(self.xml_view)

        return tab

    # -- Actions --

    def _load_from_file(self) -> None:
        """Open file dialog to load a URDF or MJCF file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Model File",
            "",
            "Model Files (*.urdf *.xml *.mjcf);;URDF Files (*.urdf);;MJCF Files (*.xml *.mjcf);;All Files (*)",
        )
        if not file_path:
            return

        result = self._loader.load_file(file_path)
        self._show_load_result(result)

    def _load_default_model(self) -> None:
        """Load the configured default model."""
        result = self._loader.load_default()
        self._show_load_result(result)

    def _set_default_model(self) -> None:
        """Set the currently selected combo item as the default model."""
        model_id = self.default_combo.currentData()
        if model_id:
            self._loader.set_default_model(model_id)
            self._status_label.setText(f"Default model set to: {model_id}")
            self._status_label.setStyleSheet(
                f"color: {CATPPUCCIN_MOCHA['green']}; padding: 4px;"
            )
            # Refresh list to update bold indicator
            self._populate_model_list(self.category_combo.currentData() or "")

    def _on_model_selected(self, item: QListWidgetItem) -> None:
        """Handle single click: show model info."""
        assert item is not None, "item must be provided"
        model_id = item.data(Qt.ItemDataRole.UserRole)
        if model_id:
            result = self._loader.load_bundled(model_id)
            self._show_load_result(result)

    def _on_model_double_clicked(self, item: QListWidgetItem) -> None:
        """Handle double click: load and set as current."""
        self._on_model_selected(item)

    def _filter_library(self) -> None:
        """Filter library list by category."""
        cat = self.category_combo.currentData() or ""
        self._populate_model_list(cat)

    def _on_display_changed(self, key: str, checked: bool) -> None:
        """Handle display checkbox change."""
        assert key is not None, "key must be provided"
        self._loader.save_preferences()
        self._status_label.setText(
            f"Display: {key} {'enabled' if checked else 'disabled'}"
        )

    # -- Display Updates --

    def _show_load_result(self, result: LoadResult) -> None:
        """Update all panels with a load result."""
        assert result is not None, "result must be provided"
        self._current_result = result

        if not result.success:
            self._status_label.setText(f"Load failed: {result.error}")
            self._status_label.setStyleSheet(
                f"color: {CATPPUCCIN_MOCHA['red']}; padding: 4px;"
            )
            return

        model = result.model
        if model is None:
            return

        # Info tab
        self.info_labels["name"].setText(model.name)
        self.info_labels["format"].setText(result.source_format.value.upper())
        self.info_labels["links"].setText(str(len(model.links)))
        self.info_labels["joints"].setText(str(len(model.joints)))

        total_mass = sum(link.inertia.mass for link in model.links)
        self.info_labels["total_mass"].setText(f"{total_mass:.3f} kg")

        root = model.get_root_link()
        self.info_labels["root_link"].setText(root.name if root else "N/A")
        self.info_labels["source"].setText(
            str(result.source_path) if result.source_path else "N/A"
        )

        # Links table
        self.links_table.setRowCount(len(model.links))
        for row, link in enumerate(model.links):
            self.links_table.setItem(row, 0, QTableWidgetItem(link.name))
            self.links_table.setItem(
                row, 1, QTableWidgetItem(f"{link.inertia.mass:.4f}")
            )
            geom_str = "-"
            if link.visual_geometry:
                geom_str = link.visual_geometry.geometry_type.value
            self.links_table.setItem(row, 2, QTableWidgetItem(geom_str))
            mat_str = "-"
            if link.visual_material:
                mat_str = link.visual_material.name
            self.links_table.setItem(row, 3, QTableWidgetItem(mat_str))

        # Joints table
        self.joints_table.setRowCount(len(model.joints))
        for row, joint in enumerate(model.joints):
            self.joints_table.setItem(row, 0, QTableWidgetItem(joint.name))
            self.joints_table.setItem(row, 1, QTableWidgetItem(joint.joint_type.value))
            self.joints_table.setItem(row, 2, QTableWidgetItem(joint.parent))
            self.joints_table.setItem(row, 3, QTableWidgetItem(joint.child))
            axis_str = f"{joint.axis[0]:.2f}, {joint.axis[1]:.2f}, {joint.axis[2]:.2f}"
            self.joints_table.setItem(row, 4, QTableWidgetItem(axis_str))

        # XML tab
        if result.source_path and result.source_path.exists():
            self.xml_view.setPlainText(result.source_path.read_text())
        elif hasattr(model, "original_xml") and model.original_xml:
            self.xml_view.setPlainText(model.original_xml)
        else:
            try:
                self.xml_view.setPlainText(model.to_urdf())
            except (RuntimeError, AttributeError):
                self.xml_view.setPlainText("(source XML not available)")

        # Status
        self._status_label.setText(
            f"Loaded: {model.name} "
            f"({len(model.links)} links, {len(model.joints)} joints)"
        )
        self._status_label.setStyleSheet(
            f"color: {CATPPUCCIN_MOCHA['green']}; padding: 4px;"
        )


def main() -> int:
    """Run the Model Explorer application."""
    app = QApplication(sys.argv)
    window = ModelExplorerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
