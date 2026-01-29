"""Model Loader Dialog for URDF Generator.

Provides a PyQt6 dialog for loading pre-configured URDF models from the library,
including human models and golf clubs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path for src imports
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from src.shared.python.logging_config import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


logger = get_logger(__name__)


class ModelLoaderDialog(QDialog):
    """Dialog for loading URDF models from the library."""

    model_selected = pyqtSignal(str, str)  # (category, model_key)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the model loader dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Load URDF Model from Library")
        self.setMinimumSize(600, 500)

        # Import here to avoid circular imports
        from .model_library import ModelLibrary

        self.library = ModelLibrary()
        self.selected_category: str | None = None
        self.selected_model: str | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        # Update imports needed for UI changes
        from PyQt6.QtWidgets import (
            QTabWidget, QTreeWidget, QTreeWidgetItem, QListWidget, QLineEdit, QHeaderView
        )
        
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Select a Model to Load")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 14pt; font-weight: bold; margin: 10px;")
        layout.addWidget(title)

        # Tabs
        self.tabs = QTabWidget()
        
        # Tab 1: Bundled Models (Original UI)
        bundled_tab = QWidget()
        bundled_layout = QVBoxLayout(bundled_tab)
        
        # Human Models Section
        human_group = self._create_human_models_group()
        bundled_layout.addWidget(human_group)

        # Golf Clubs Section
        golf_group = self._create_golf_clubs_group()
        bundled_layout.addWidget(golf_group)
        bundled_layout.addStretch()
        
        self.tabs.addTab(bundled_tab, "Bundled Assets")

        # Tab 2: Repository Models (Discovered)
        repo_tab = QWidget()
        self._setup_repo_tab(repo_tab)
        self.tabs.addTab(repo_tab, "Repository Models")

        # Tab 3: Embedded Models (Python defined)
        embedded_tab = QWidget()
        self._setup_embedded_tab(embedded_tab)
        self.tabs.addTab(embedded_tab, "Embedded Models")

        layout.addWidget(self.tabs)

        # Info display
        info_label = QLabel("Model Information:")
        info_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(info_label)

        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_display.setMaximumHeight(150)
        self.info_display.setPlainText(
            "Select a model above to see its specifications..."
        )
        layout.addWidget(self.info_display)

        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _setup_repo_tab(self, parent: QWidget) -> None:
        from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem, QHeaderView, QLineEdit
        
        layout = QVBoxLayout(parent)
        
        # Search
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.repo_search = QLineEdit()
        self.repo_search.setPlaceholderText("Filter models...")
        self.repo_search.textChanged.connect(self._filter_repo_list)
        search_layout.addWidget(self.repo_search)
        layout.addLayout(search_layout)
        
        # Tree
        self.repo_tree = QTreeWidget()
        self.repo_tree.setHeaderLabels(["Name", "Type", "Path"])
        self.repo_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.repo_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.repo_tree.itemSelectionChanged.connect(
            lambda: self._on_model_selected("discovered")
        )
        layout.addWidget(self.repo_tree)
        
        # Populate
        self.discovered_models = self.library.list_available_models()["discovered"]
        self._populate_repo_tree(self.discovered_models)
        
        # Load button for this tab
        load_btn = QPushButton("Load Selected Repository Model")
        load_btn.clicked.connect(lambda: self._load_selected_model("discovered"))
        layout.addWidget(load_btn)

    def _populate_repo_tree(self, models: list) -> None:
        from PyQt6.QtWidgets import QTreeWidgetItem
        self.repo_tree.clear()
        for model in models:
            item = QTreeWidgetItem([model["name"], model["type"].upper(), model["path"]])
            item.setData(0, Qt.ItemDataRole.UserRole, model["config_key"])
            self.repo_tree.addTopLevelItem(item)

    def _filter_repo_list(self, text: str) -> None:
        text = text.lower()
        filtered = [
            m for m in self.discovered_models 
            if text in m["name"].lower() or text in m["path"].lower()
        ]
        self._populate_repo_tree(filtered)

    def _setup_embedded_tab(self, parent: QWidget) -> None:
        from PyQt6.QtWidgets import QListWidget
        
        layout = QVBoxLayout(parent)
        
        layout.addWidget(QLabel("Pre-defined MuJoCo XML models found in Python code:"))
        
        self.embedded_list = QListWidget()
        self.embedded_list.itemSelectionChanged.connect(
            lambda: self._on_model_selected("embedded")
        )
        layout.addWidget(self.embedded_list)
        
        # Populate
        embedded_models = self.library.list_available_models()["embedded"]
        for key, model in embedded_models.items():
            from PyQt6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(f"{model['name']} (MJCF)")
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.embedded_list.addItem(item)
            
        # Load button
        load_btn = QPushButton("Load Selected Embedded Model")
        load_btn.clicked.connect(lambda: self._load_selected_model("embedded"))
        layout.addWidget(load_btn)

    def _on_accept(self) -> None:
        # Determine active tab and selected item
        idx = self.tabs.currentIndex()
        if idx == 0: # Bundled
            pass 
        
        if self.selected_model:
            self.accept()
        else:
             # Try to select from active tab
             if idx == 1:
                self._load_selected_model("discovered")
             elif idx == 2:
                self._load_selected_model("embedded")

    def _create_human_models_group(self) -> QGroupBox:
        """Create the human models selection group.

        Returns:
            QGroupBox containing human model controls
        """
        group = QGroupBox("Human Biomechanical Models")
        layout = QVBoxLayout(group)

        # Description
        desc = QLabel(
            "High-fidelity human models from the human-gazebo repository.\n"
            "Includes detailed STL meshes for realistic visualization."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 9pt; margin: 5px;")
        layout.addWidget(desc)

        # Model selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Model:"))

        self.human_combo = QComboBox()
        available_models = self.library.list_available_models()

        for model_key in available_models["human"]:
            model_info = self.library.get_model_info("human", model_key)
            if model_info:
                self.human_combo.addItem(model_info["name"], model_key)

        self.human_combo.currentIndexChanged.connect(
            lambda: self._on_model_selected("human")
        )
        selector_layout.addWidget(self.human_combo)

        load_btn = QPushButton("Load Human Model")
        load_btn.clicked.connect(lambda: self._load_selected_model("human"))
        selector_layout.addWidget(load_btn)

        layout.addLayout(selector_layout)

        # Set as default checkbox
        from PyQt6.QtWidgets import QCheckBox
        self.default_human_chk = QCheckBox("Set as default human model")
        self.default_human_chk.setToolTip("Automatically load this model on startup")
        layout.addWidget(self.default_human_chk)

        # Download button
        download_layout = QHBoxLayout()
        download_btn = QPushButton("Download from human-gazebo")
        download_btn.setToolTip(
            "Download URDF and mesh files from the human-gazebo repository"
        )
        download_btn.clicked.connect(self._download_human_model)
        download_layout.addWidget(download_btn)

        license_label = QLabel("License: CC-BY-SA 2.0")
        license_label.setStyleSheet("color: #888; font-size: 8pt;")
        download_layout.addWidget(license_label)
        download_layout.addStretch()

        layout.addLayout(download_layout)

        return group

    def _create_golf_clubs_group(self) -> QGroupBox:
        """Create the golf clubs selection group.

        Returns:
            QGroupBox containing golf club controls
        """
        group = QGroupBox("Golf Clubs")
        layout = QVBoxLayout(group)

        # Description
        desc = QLabel(
            "Select a golf club to add to your model.\n"
            "Clubs include realistic mass properties and geometry."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 9pt; margin: 5px;")
        layout.addWidget(desc)

        # Club selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Club Type:"))

        self.club_combo = QComboBox()
        available_models = self.library.list_available_models()

        for club_key in available_models["golf_clubs"]:
            club_info = self.library.get_model_info("golf_clubs", club_key)
            if club_info:
                self.club_combo.addItem(club_info["name"], club_key)

        self.club_combo.currentIndexChanged.connect(
            lambda: self._on_model_selected("golf_clubs")
        )
        selector_layout.addWidget(self.club_combo)

        generate_btn = QPushButton("Generate Club URDF")
        generate_btn.clicked.connect(lambda: self._load_selected_model("golf_clubs"))
        selector_layout.addWidget(generate_btn)

        layout.addLayout(selector_layout)

        return group

    def _on_model_selected(self, category: str) -> None:
        """Handle model selection change.

        Args:
            category: 'human', 'golf_clubs', 'discovered', or 'embedded'
        """
        model_key = None
        
        if category == "human":
            model_key = self.human_combo.currentData()
        elif category == "golf_clubs":
            model_key = self.club_combo.currentData()
        elif category == "discovered":
            item = self.repo_tree.currentItem()
            if item:
                model_key = item.data(0, Qt.ItemDataRole.UserRole)
        elif category == "embedded":
            item = self.embedded_list.currentItem()
            if item:
                model_key = item.data(Qt.ItemDataRole.UserRole)

        if model_key:
            model_info = self.library.get_model_info(category, model_key)
            if model_info:
                self._display_model_info(category, model_key, model_info)
            self.selected_category = category
            self.selected_model = model_key # Track selection lightly for tab changes

    def _display_model_info(
        self, category: str, model_key: str, model_info: dict[str, Any]
    ) -> None:
        """Display information about selected model.

        Args:
            category: Model category
            model_key: Model identifier
            model_info: Model information dictionary
        """
        if category == "human":
            info_text = f"""Name: {model_info["name"]}
Description: {model_info["description"]}
License: {model_info["license"]}

Repository: https://github.com/gbionics/human-gazebo
"""
        elif category == "golf_clubs":
            info_text = f"""Club: {model_info["name"]}
Loft: {model_info["loft"]}°
Length: {model_info["length"] * 100:.1f} cm ({model_info["length"] / 0.0254:.1f} inches)
Total Mass: {model_info["mass"] * 1000:.1f} g
  - Head: {model_info["head_mass"] * 1000:.1f} g
  - Shaft: {model_info["shaft_mass"] * 1000:.1f} g
  - Grip: {model_info["grip_mass"] * 1000:.1f} g

The URDF will be automatically generated with realistic geometry and inertial properties.
"""
        elif category == "discovered":
             info_text = f"""Name: {model_info["name"]}
Type: {model_info["type"].upper()}
Path: {model_info["description"]}

Click 'Load Selected Repository Model' to view.
"""
        elif category == "embedded":
            content_preview = model_info["content"][:200] + "..." if len(model_info["content"]) > 200 else model_info["content"]
            info_text = f"""Name: {model_info["name"]}
Type: Embedded MJCF
Description: {model_info["description"]}

Content Preview:
{content_preview}
"""
        else:
            info_text = "No information available."

        self.info_display.setPlainText(info_text)

    def _download_human_model(self) -> None:
        """Download the currently selected human model."""
        model_key = self.human_combo.currentData()
        if not model_key:
            return

        reply = QMessageBox.question(
            self,
            "Download Model",
            f"Download human model '{self.human_combo.currentText()}' "
            "from human-gazebo repository?\n\n"
            "This will download the URDF file and associated mesh files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                urdf_path = self.library.download_human_model(model_key)
                if urdf_path:
                    QMessageBox.information(
                        self,
                        "Download Complete",
                        f"Model downloaded successfully to:\n{urdf_path}",
                    )
                else:
                    QMessageBox.warning(
                        self, "Download Failed", "Failed to download model files."
                    )
            except Exception as e:
                logger.error(f"Download error: {e}")
                QMessageBox.critical(
                    self, "Error", f"Download failed with error:\n{str(e)}"
                )

    def _load_selected_model(self, category: str) -> None:
        """Load the selected model.

        Args:
            category: 'human' or 'golf_clubs'
        """
        if category == "human":
            model_key = self.human_combo.currentData()
        else:
            model_key = self.club_combo.currentData()

        if model_key:
            self.selected_category = category
            self.selected_model = model_key

            # Save as default if requested
            if category == "human" and hasattr(self, "default_human_chk") and self.default_human_chk.isChecked():
                from PyQt6.QtCore import QSettings
                settings = QSettings("GolfModelingSuite", "URDFGenerator")
                settings.setValue("default_human_model", model_key)
                logger.info(f"Set default human model to: {model_key}")

            self.model_selected.emit(category, model_key)
            self.accept()

    def get_selected_model(self) -> tuple[str, str] | None:
        """Get the selected model.

        Returns:
            Tuple of (category, model_key) or None if no selection
        """
        if self.selected_category and self.selected_model:
            return (self.selected_category, self.selected_model)
        return None
