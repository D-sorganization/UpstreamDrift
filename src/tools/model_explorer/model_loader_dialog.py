"""Model Loader Dialog for URDF Generator.



Provides a PyQt6 dialog for loading pre-configured URDF models from the library,

including human models and golf clubs.

"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
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

from src.shared.python.logging_pkg.logger_utils import get_logger  # noqa: E402

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

    def _setup_biomechanics_tab(self) -> QWidget:
        """Create the Biomechanics tab with human models."""
        biomech_tab = QWidget()
        biomech_layout = QVBoxLayout(biomech_tab)
        human_group = self._create_human_models_group()
        biomech_layout.addWidget(human_group)
        biomech_layout.addStretch()
        return biomech_tab

    def _setup_equipment_tab(self) -> QWidget:
        """Create the Equipment tab with golf clubs and components."""
        equipment_tab = QWidget()
        equip_layout = QVBoxLayout(equipment_tab)
        golf_group = self._create_golf_clubs_group()
        equip_layout.addWidget(golf_group)
        component_group = self._create_components_group()
        equip_layout.addWidget(component_group)
        equip_layout.addStretch()
        return equipment_tab

    def _setup_robotics_tab(self) -> QWidget:
        """Create the Robotics tab with pendulums and manipulators."""
        robotics_tab = QWidget()
        robotics_layout = QVBoxLayout(robotics_tab)

        pendulum_group = self._create_model_group(
            "Simplified Physics Models",
            "pendulum",
            "Simple pendulum models for understanding swing mechanics.",
            "Load Physics Model",
        )
        robotics_layout.addWidget(pendulum_group)

        robot_group = self._create_model_group(
            "Robotic Manipulators",
            "robotic",
            "Industrial robot arms with golf attachments.",
            "Load Robot Model",
        )
        robotics_layout.addWidget(robot_group)

        robotics_layout.addStretch()
        return robotics_tab

    def _setup_info_and_buttons(self, layout: QVBoxLayout) -> None:
        """Set up the model info display and OK/Cancel buttons."""
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

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _setup_ui(self) -> None:
        """Set up the user interface."""

        from PyQt6.QtWidgets import (
            QTabWidget,
        )

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Select a Model to Load")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 14pt; font-weight: bold; margin: 10px;")
        layout.addWidget(title)

        # Tabs
        self.tabs = QTabWidget()

        self.tabs.addTab(self._setup_biomechanics_tab(), "Biomechanics")
        self.tabs.addTab(self._setup_equipment_tab(), "Equipment")
        self.tabs.addTab(self._setup_robotics_tab(), "Robotics")

        repo_tab = QWidget()
        self._setup_repo_tab(repo_tab)
        self.tabs.addTab(repo_tab, "Repository")

        community_tab = QWidget()
        self._setup_community_tab(community_tab)
        self.tabs.addTab(community_tab, "Community")

        imported_tab = QWidget()
        self._setup_imported_tab(imported_tab)
        self.tabs.addTab(imported_tab, "Imported")

        embedded_tab = QWidget()
        self._setup_embedded_tab(embedded_tab)
        self.tabs.addTab(embedded_tab, "Embedded")

        layout.addWidget(self.tabs)

        # Info display and dialog buttons
        self._setup_info_and_buttons(layout)

    def _setup_repo_tab(self, parent: QWidget) -> None:
        from PyQt6.QtWidgets import QHeaderView, QLineEdit, QTreeWidget

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

        header = self.repo_tree.header()

        if header:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)

            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

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
            item = QTreeWidgetItem(
                [model["name"], model["type"].upper(), model["path"]]
            )

            item.setData(0, Qt.ItemDataRole.UserRole, model["config_key"])

            self.repo_tree.addTopLevelItem(item)

    def _filter_repo_list(self, text: str) -> None:
        text = text.lower()

        filtered = [
            m
            for m in self.discovered_models
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

    def _setup_community_tab(self, parent: QWidget) -> None:
        from PyQt6.QtWidgets import QListWidget, QListWidgetItem

        layout = QVBoxLayout(parent)

        layout.addWidget(QLabel("Community models from 'robot_descriptions' library:"))

        self.community_list = QListWidget()

        self.community_list.itemSelectionChanged.connect(
            lambda: self._on_model_selected("robot_descriptions")
        )

        layout.addWidget(self.community_list)

        # Populate

        community_models = self.library.list_available_models().get(
            "robot_descriptions", []
        )

        if not community_models:
            layout.addWidget(
                QLabel(
                    "No community models found.\nEnsure 'robot_descriptions' is installed."
                )
            )

        for model in community_models:
            item = QListWidgetItem(f"{model['name']} ({model['type'].upper()})")

            item.setData(Qt.ItemDataRole.UserRole, model["config_key"])

            self.community_list.addItem(item)

        load_btn = QPushButton("Load Selected Community Model")

        load_btn.clicked.connect(
            lambda: self._load_selected_model("robot_descriptions")
        )

        layout.addWidget(load_btn)

    def _setup_imported_tab(self, parent: QWidget) -> None:
        from PyQt6.QtWidgets import QHeaderView, QTreeWidget

        layout = QVBoxLayout(parent)

        layout.addWidget(QLabel("User Imported models (Right-click to manage):"))

        # Tree

        self.imported_tree = QTreeWidget()

        self.imported_tree.setHeaderLabels(["Name", "Type", "Path"])

        self.imported_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.imported_tree.customContextMenuRequested.connect(
            self._on_imported_context_menu
        )

        header = self.imported_tree.header()

        if header:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)

            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

        self.imported_tree.itemSelectionChanged.connect(
            lambda: self._on_model_selected("imported")
        )

        layout.addWidget(self.imported_tree)

        # Buttons

        btn_layout = QHBoxLayout()

        import_btn = QPushButton("Import URDF/MJCF...")

        import_btn.clicked.connect(self._on_import_button)

        btn_layout.addWidget(import_btn)

        reload_btn = QPushButton("Reload")

        reload_btn.clicked.connect(self._reload_imported_models)

        btn_layout.addWidget(reload_btn)

        layout.addLayout(btn_layout)

        load_btn = QPushButton("Load Selected Imported Model")

        load_btn.clicked.connect(lambda: self._load_selected_model("imported"))

        layout.addWidget(load_btn)

        self._reload_imported_models()

    def _reload_imported_models(self) -> None:
        from PyQt6.QtWidgets import QTreeWidgetItem

        self.imported_tree.clear()

        models = self.library.discover_imported_models()

        for model in models:
            item = QTreeWidgetItem(
                [model["name"], model["type"].upper(), model["path"]]
            )

            item.setData(0, Qt.ItemDataRole.UserRole, model["config_key"])

            item.setData(
                0, Qt.ItemDataRole.UserRole + 1, model["path"]
            )  # Store path for file ops

            self.imported_tree.addTopLevelItem(item)

    def _on_import_button(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Model File",
            "",
            "Model Files (*.urdf *.xml *.mjcf);;All Files (*)",
        )

        if path:
            if self.library.import_model(path):
                self._reload_imported_models()

                QMessageBox.information(self, "Success", "Model imported successfully.")

            else:
                QMessageBox.warning(self, "Error", "Failed to import model.")

    def _on_imported_context_menu(self, pos: Any) -> None:
        from PyQt6.QtWidgets import QInputDialog, QMenu

        item = self.imported_tree.itemAt(pos)

        if not item:
            return

        menu = QMenu(self)

        rename_action = menu.addAction("Rename")

        delete_action = menu.addAction("Delete")

        viewport = self.imported_tree.viewport()

        if viewport is None:
            return

        action = menu.exec(viewport.mapToGlobal(pos))

        model_path = item.data(0, Qt.ItemDataRole.UserRole + 1)

        if action == rename_action:
            old_name = item.text(0)

            new_name, ok = QInputDialog.getText(
                self, "Rename Model", "New name:", text=old_name
            )

            if ok and new_name and new_name != old_name:
                if self.library.rename_imported_model(model_path, new_name):
                    self._reload_imported_models()

                else:
                    QMessageBox.warning(self, "Error", "Failed to rename model.")

        elif action == delete_action:
            reply = QMessageBox.question(
                self,
                "Confirm Delete",
                f"Are you sure you want to delete '{item.text(0)}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                if self.library.delete_imported_model(model_path):
                    self._reload_imported_models()

                else:
                    QMessageBox.warning(self, "Error", "Failed to delete model.")

    def _on_accept(self) -> None:
        # Determine active tab and selected item

        idx = self.tabs.currentIndex()

        if idx == 0:  # Bundled
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

    def _create_components_group(self) -> QGroupBox:
        """Create the components selection group."""

        return self._create_model_group(
            "Components",
            "component",
            "Individual simulation elements like balls and flexible shafts.",
            "Load Component",
        )

    def _create_model_group(
        self, title: str, category: str, description: str, btn_text: str
    ) -> QGroupBox:
        """Generic helper to create a model selection group.



        Args:

            title: Group box title

            category: Model category key in library

            description: Description text

            btn_text: Button text



        Returns:

            Configured QGroupBox

        """

        group = QGroupBox(title)

        layout = QVBoxLayout(group)

        # Description

        desc = QLabel(description)

        desc.setWordWrap(True)

        desc.setStyleSheet("color: #666; font-size: 9pt; margin: 5px;")

        layout.addWidget(desc)

        # Selector

        selector_layout = QHBoxLayout()

        selector_layout.addWidget(QLabel("Model:"))

        combo = QComboBox()

        available_models = self.library.list_available_models()

        # Store reference to combo for this category

        setattr(self, f"{category}_combo", combo)

        for key in available_models.get(category, []):
            info = self.library.get_model_info(category, key)

            if info:
                combo.addItem(info["name"], key)

        combo.currentIndexChanged.connect(lambda: self._on_model_selected(category))

        selector_layout.addWidget(combo)

        load_btn = QPushButton(btn_text)

        load_btn.clicked.connect(lambda: self._load_selected_model(category))

        selector_layout.addWidget(load_btn)

        layout.addLayout(selector_layout)

        return group

    def _on_model_selected(self, category: str) -> None:
        """Handle model selection change.



        Args:

            category: 'human', 'golf_clubs', 'pendulum', 'robotic', 'component',

                     'discovered', or 'embedded'

        """

        model_key = None

        if category == "human":
            model_key = self.human_combo.currentData()

        elif category == "golf_clubs":
            model_key = self.club_combo.currentData()

        elif hasattr(self, f"{category}_combo"):
            # Generic handling for pendulum, robotic, component

            combo = getattr(self, f"{category}_combo")

            model_key = combo.currentData()

        elif category == "discovered":
            repo_item = self.repo_tree.currentItem()

            if repo_item:
                model_key = repo_item.data(0, Qt.ItemDataRole.UserRole)

        elif category == "embedded":
            embed_item = self.embedded_list.currentItem()

            if embed_item:
                model_key = embed_item.data(Qt.ItemDataRole.UserRole)

        elif category == "robot_descriptions":
            comm_item = self.community_list.currentItem()

            if comm_item:
                model_key = comm_item.data(Qt.ItemDataRole.UserRole)

        elif category == "imported":
            imp_item = self.imported_tree.currentItem()

            if imp_item:
                model_key = imp_item.data(0, Qt.ItemDataRole.UserRole)

        if model_key:
            model_info = self.library.get_model_info(category, model_key)

            if model_info:
                self._display_model_info(category, model_key, model_info)

            self.selected_category = category

            self.selected_model = model_key

    def _display_model_info(
        self, category: str, model_key: str, model_info: dict[str, Any]
    ) -> None:
        """Display information about selected model.



        Args:

            category: Model category

            model_key: Model identifier

            model_info: Model information dictionary

        """

        formatters: dict[str, Any] = {
            "human": self._format_human_info,
            "golf_clubs": self._format_golf_clubs_info,
            "pendulum": self._format_generic_model_info,
            "robotic": self._format_generic_model_info,
            "component": self._format_generic_model_info,
            "discovered": self._format_discovered_info,
            "embedded": self._format_embedded_info,
            "robot_descriptions": self._format_robot_descriptions_info,
            "imported": self._format_imported_info,
        }

        formatter = formatters.get(category)
        if formatter:
            info_text = formatter(model_info)
        else:
            info_text = "No information available."

        self.info_display.setPlainText(info_text)

    def _format_human_info(self, model_info: dict[str, Any]) -> str:
        return (
            f"Name: {model_info['name']}\n"
            f"Description: {model_info['description']}\n"
            f"License: {model_info['license']}\n\n"
            f"Repository: https://github.com/gbionics/human-gazebo\n"
        )

    def _format_golf_clubs_info(self, model_info: dict[str, Any]) -> str:
        return (
            f"Club: {model_info['name']}\n"
            f"Loft: {model_info['loft']}\u00b0\n"
            f"Length: {model_info['length'] * 100:.1f} cm "
            f"({model_info['length'] / 0.0254:.1f} inches)\n"
            f"Total Mass: {model_info['mass'] * 1000:.1f} g\n"
            f"  - Head: {model_info['head_mass'] * 1000:.1f} g\n"
            f"  - Shaft: {model_info['shaft_mass'] * 1000:.1f} g\n"
            f"  - Grip: {model_info['grip_mass'] * 1000:.1f} g\n\n"
            f"The URDF will be automatically generated with realistic "
            f"geometry and inertial properties.\n"
        )

    def _format_generic_model_info(self, model_info: dict[str, Any]) -> str:
        return (
            f"Name: {model_info['name']}\n"
            f"Type: {model_info.get('type', 'Unknown').upper()}\n"
            f"Description: {model_info['description']}\n"
            f"Path: {model_info.get('path', 'N/A')}\n\n"
            f"Click 'Load' to view this model.\n"
        )

    def _format_discovered_info(self, model_info: dict[str, Any]) -> str:
        return (
            f"Name: {model_info['name']}\n"
            f"Type: {model_info['type'].upper()}\n"
            f"Path: {model_info['description']}\n\n"
            f"Click 'Load Selected Repository Model' to view.\n"
        )

    def _format_embedded_info(self, model_info: dict[str, Any]) -> str:
        content_preview = (
            model_info["content"][:200] + "..."
            if len(model_info["content"]) > 200
            else model_info["content"]
        )
        return (
            f"Name: {model_info['name']}\n"
            f"Type: Embedded MJCF\n"
            f"Description: {model_info['description']}\n\n"
            f"Content Preview:\n{content_preview}\n"
        )

    def _format_robot_descriptions_info(self, model_info: dict[str, Any]) -> str:
        return (
            f"Name: {model_info['name']}\n"
            f"Type: {model_info['type'].upper()}\n"
            f"Package: {model_info.get('package', 'robot_descriptions')}\n"
            f"Path: {model_info['path']}\n\n"
            f"Description: {model_info['description']}\n"
        )

    def _format_imported_info(self, model_info: dict[str, Any]) -> str:
        return (
            f"Name: {model_info['name']}\n"
            f"Type: {model_info['type'].upper()}\n"
            f"Path: {model_info['path']}\n\n"
            f"User imported model. Right-click in the list to Rename or Delete.\n"
        )

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

            except (RuntimeError, ValueError, OSError) as e:
                logger.error(f"Download error: {e}")

                QMessageBox.critical(
                    self, "Error", f"Download failed with error:\n{str(e)}"
                )

    def _load_selected_model(self, category: str) -> None:
        """Load the selected model.



        Args:

            category: Model category key

        """

        if category == "human":
            model_key = self.human_combo.currentData()

        elif category == "golf_clubs":
            model_key = self.club_combo.currentData()

        elif hasattr(self, f"{category}_combo"):
            combo = getattr(self, f"{category}_combo")

            model_key = combo.currentData()

        else:
            # Fallback for manual calls, though generic way is better

            return

        if model_key:
            self.selected_category = category

            self.selected_model = model_key

            # Save as default if requested

            if (
                category == "human"
                and hasattr(self, "default_human_chk")
                and self.default_human_chk.isChecked()
            ):
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
