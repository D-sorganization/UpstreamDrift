"""Model tree/list tab widget builders for ModelLoaderDialog.

Provides ModelTreeMixin, which supplies the Repository, Embedded, Community,
and Imported tab setup methods.  Intended to be mixed into ModelLoaderDialog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass


class ModelTreeMixin:
    """Mixin providing tree/list tab builders for the model loader dialog."""

    # ------------------------------------------------------------------
    # Repository tab
    # ------------------------------------------------------------------

    def _setup_repo_tab(self, parent: QWidget) -> None:
        """Set up the Repository tab with a searchable tree of discovered models."""
        if parent is None:
            raise ValueError("parent must be provided")
        from PyQt6.QtWidgets import QHeaderView, QLineEdit, QTreeWidget

        layout = QVBoxLayout(parent)

        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.repo_search = QLineEdit()
        self.repo_search.setPlaceholderText("Filter models...")
        self.repo_search.textChanged.connect(self._filter_repo_list)
        search_layout.addWidget(self.repo_search)
        layout.addLayout(search_layout)

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

        self.discovered_models = self.library.list_available_models()["discovered"]
        self._populate_repo_tree(self.discovered_models)

        load_btn = QPushButton("Load Selected Repository Model")
        load_btn.clicked.connect(lambda: self._load_selected_model("discovered"))
        layout.addWidget(load_btn)

    def _populate_repo_tree(self, models: list) -> None:
        """Populate the repository tree widget with *models*."""
        if models is None:
            raise ValueError("models must be provided")
        from PyQt6.QtWidgets import QTreeWidgetItem

        self.repo_tree.clear()
        for model in models:
            item = QTreeWidgetItem(
                [model["name"], model["type"].upper(), model["path"]]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, model["config_key"])
            self.repo_tree.addTopLevelItem(item)

    def _filter_repo_list(self, text: str) -> None:
        """Filter the repository tree to models matching *text*."""
        if text is None:
            raise ValueError("text must be provided")
        text = text.lower()
        filtered = [
            m
            for m in self.discovered_models
            if text in m["name"].lower() or text in m["path"].lower()
        ]
        self._populate_repo_tree(filtered)

    # ------------------------------------------------------------------
    # Embedded tab
    # ------------------------------------------------------------------

    def _setup_embedded_tab(self, parent: QWidget) -> None:
        """Set up the Embedded tab listing pre-defined MuJoCo XML models."""
        if parent is None:
            raise ValueError("parent must be provided")
        from PyQt6.QtWidgets import QListWidget

        layout = QVBoxLayout(parent)
        layout.addWidget(QLabel("Pre-defined MuJoCo XML models found in Python code:"))

        self.embedded_list = QListWidget()
        self.embedded_list.itemSelectionChanged.connect(
            lambda: self._on_model_selected("embedded")
        )
        layout.addWidget(self.embedded_list)

        embedded_models = self.library.list_available_models()["embedded"]
        for key, model in embedded_models.items():
            from PyQt6.QtWidgets import QListWidgetItem

            item = QListWidgetItem(f"{model['name']} (MJCF)")
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.embedded_list.addItem(item)

        load_btn = QPushButton("Load Selected Embedded Model")
        load_btn.clicked.connect(lambda: self._load_selected_model("embedded"))
        layout.addWidget(load_btn)

    # ------------------------------------------------------------------
    # Community tab
    # ------------------------------------------------------------------

    def _setup_community_tab(self, parent: QWidget) -> None:
        """Set up the Community tab listing robot_descriptions library models."""
        if parent is None:
            raise ValueError("parent must be provided")
        from PyQt6.QtWidgets import QListWidget, QListWidgetItem

        layout = QVBoxLayout(parent)
        layout.addWidget(QLabel("Community models from 'robot_descriptions' library:"))

        self.community_list = QListWidget()
        self.community_list.itemSelectionChanged.connect(
            lambda: self._on_model_selected("robot_descriptions")
        )
        layout.addWidget(self.community_list)

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

    # ------------------------------------------------------------------
    # Imported tab
    # ------------------------------------------------------------------

    def _setup_imported_tab(self, parent: QWidget) -> None:
        """Set up the Imported tab for user-managed model files."""
        if parent is None:
            raise ValueError("parent must be provided")
        from PyQt6.QtWidgets import QHeaderView, QTreeWidget

        layout = QVBoxLayout(parent)
        layout.addWidget(QLabel("User Imported models (Right-click to manage):"))

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
        """Refresh the imported-models tree from the library."""
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
