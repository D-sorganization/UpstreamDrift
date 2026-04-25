"""Model Loader Dialog for URDF Generator.

Provides a PyQt6 dialog for loading pre-configured URDF models from the library,
including human models and golf clubs.

The original monolithic file has been split into focused modules:

- model_loader_dialog.py  — this file; thin dialog orchestrator
- model_loader.py         — load/download actions and info-formatting helpers
                            (ModelLoaderMixin)
- model_tree_widget.py    — repository/embedded/community/imported tab builders
                            (ModelTreeMixin)
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_pkg.logger_utils import get_logger  # noqa: E402

from .model_loader import ModelLoaderMixin
from .model_tree_widget import ModelTreeMixin

logger = get_logger(__name__)


class ModelLoaderDialog(ModelLoaderMixin, ModelTreeMixin, QDialog):
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

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        from PyQt6.QtWidgets import QTabWidget

        layout = QVBoxLayout(self)

        title = QLabel("Select a Model to Load")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 14pt; font-weight: bold; margin: 10px;")
        layout.addWidget(title)

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
        self._setup_info_and_buttons(layout)

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
        if layout is None:
            raise ValueError("layout must be provided")
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

    # ------------------------------------------------------------------
    # Widget group builders
    # ------------------------------------------------------------------

    def _create_human_models_group(self) -> QGroupBox:
        """Create the human models selection group.

        Returns:
            QGroupBox containing human model controls

        """
        group = QGroupBox("Human Biomechanical Models")
        layout = QVBoxLayout(group)

        desc = QLabel(
            "High-fidelity human models from the human-gazebo repository.\n"
            "Includes detailed STL meshes for realistic visualization."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 9pt; margin: 5px;")
        layout.addWidget(desc)

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

        self.default_human_chk = QCheckBox("Set as default human model")
        self.default_human_chk.setToolTip("Automatically load this model on startup")
        layout.addWidget(self.default_human_chk)

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

        desc = QLabel(
            "Select a golf club to add to your model.\n"
            "Clubs include realistic mass properties and geometry."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 9pt; margin: 5px;")
        layout.addWidget(desc)

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
        if title is None:
            raise ValueError("title must be provided")
        group = QGroupBox(title)
        layout = QVBoxLayout(group)

        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 9pt; margin: 5px;")
        layout.addWidget(desc)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Model:"))
        combo = QComboBox()
        available_models = self.library.list_available_models()
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

    # ------------------------------------------------------------------
    # Selection and info display
    # ------------------------------------------------------------------

    def _on_model_selected(self, category: str) -> None:
        """Handle model selection change.

        Args:
            category: 'human', 'golf_clubs', 'pendulum', 'robotic', 'component',
                     'discovered', 'embedded', 'robot_descriptions', or 'imported'

        """
        if category is None:
            raise ValueError("category must be provided")
        model_key = None

        if category == "human":
            model_key = self.human_combo.currentData()
        elif category == "golf_clubs":
            model_key = self.club_combo.currentData()
        elif hasattr(self, f"{category}_combo"):
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
        """Display information about the selected model.

        Args:
            category: Model category
            model_key: Model identifier
            model_info: Model information dictionary

        """
        if category is None:
            raise ValueError("category must be provided")
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
        info_text = formatter(model_info) if formatter else "No information available."
        self.info_display.setPlainText(info_text)

    # ------------------------------------------------------------------
    # Dialog accept / public API
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        """Handle the OK button: accept if a model is selected."""
        idx = self.tabs.currentIndex()

        if idx == 0:  # Bundled / Biomechanics
            pass

        if self.selected_model:
            self.accept()
        else:
            if idx == 1:
                self._load_selected_model("discovered")
            elif idx == 2:
                self._load_selected_model("embedded")

    def get_selected_model(self) -> tuple[str, str] | None:
        """Get the selected model.

        Returns:
            Tuple of (category, model_key) or None if no selection

        """
        if self.selected_category and self.selected_model:
            return (self.selected_category, self.selected_model)
        return None
