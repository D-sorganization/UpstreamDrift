from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class _EndEffectorManagerWidgetUIMixin:
    """UI construction helpers for EndEffectorManagerWidget."""

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)  # type: ignore[call-overload]

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side - current model end effectors
        splitter.addWidget(self._create_current_ee_widget())

        # Right side - library
        splitter.addWidget(self._create_library_widget())

        layout.addWidget(splitter)

        # Status
        self.status_label = QLabel("Load a URDF to manage end effectors")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

    def _create_current_ee_widget(self) -> QWidget:
        """Create the widget for managing current end effectors."""
        current_widget = QWidget()
        current_layout = QVBoxLayout(current_widget)

        current_group = QGroupBox("Current End Effectors")
        current_inner = QVBoxLayout(current_group)

        self.current_list = QListWidget()
        current_inner.addWidget(self.current_list)

        btn_layout = QHBoxLayout()
        self.identify_btn = QPushButton("Identify EEs")
        self.remove_ee_btn = QPushButton("Remove Selected")
        self.extract_btn = QPushButton("Extract to Library")
        btn_layout.addWidget(self.identify_btn)
        btn_layout.addWidget(self.remove_ee_btn)
        btn_layout.addWidget(self.extract_btn)
        current_inner.addLayout(btn_layout)

        current_layout.addWidget(current_group)

        # Selected EE info
        info_group = QGroupBox("Selected End Effector")
        info_layout = QVBoxLayout(info_group)
        self.ee_info_text = QTextEdit()
        self.ee_info_text.setReadOnly(True)
        self.ee_info_text.setMaximumHeight(100)
        info_layout.addWidget(self.ee_info_text)
        current_layout.addWidget(info_group)

        return current_widget

    def _create_library_widget(self) -> QWidget:
        """Create the widget for the end effector library."""
        library_widget = QWidget()
        library_layout = QVBoxLayout(library_widget)

        # Built-in end effectors
        builtin_group = QGroupBox("Built-in End Effectors")
        builtin_layout = QVBoxLayout(builtin_group)

        self.builtin_list = QListWidget()
        builtin_layout.addWidget(self.builtin_list)

        library_layout.addWidget(builtin_group)

        # Custom library
        custom_group = QGroupBox("Custom Library")
        custom_layout = QVBoxLayout(custom_group)

        self.custom_list = QListWidget()
        custom_layout.addWidget(self.custom_list)

        import_btn_layout = QHBoxLayout()
        self.import_from_file_btn = QPushButton("Import from URDF")
        import_btn_layout.addWidget(self.import_from_file_btn)
        custom_layout.addLayout(import_btn_layout)

        library_layout.addWidget(custom_group)

        # Attach button
        self.attach_btn = QPushButton("Attach Selected to Model")
        self.attach_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        library_layout.addWidget(self.attach_btn)

        return library_widget

    def _connect_signals(self) -> None:
        """Connect signals."""
        self.identify_btn.clicked.connect(self._on_identify_end_effectors)  # type: ignore[attr-defined]
        self.remove_ee_btn.clicked.connect(self._on_remove_end_effector)  # type: ignore[attr-defined]
        self.extract_btn.clicked.connect(self._on_extract_to_library)  # type: ignore[attr-defined]
        self.import_from_file_btn.clicked.connect(self._on_import_from_file)  # type: ignore[attr-defined]
        self.attach_btn.clicked.connect(self._on_attach_end_effector)  # type: ignore[attr-defined]

        self.current_list.itemSelectionChanged.connect(
            self._on_current_selection_changed  # type: ignore[attr-defined]
        )
        self.builtin_list.itemSelectionChanged.connect(
            self._on_library_selection_changed  # type: ignore[attr-defined]
        )
        self.custom_list.itemSelectionChanged.connect(
            self._on_library_selection_changed  # type: ignore[attr-defined]
        )

    def _populate_builtin_list(self) -> None:
        """Populate the built-in end effectors list."""
        self.builtin_list.clear()
        for key in self.library.get_builtin_names():  # type: ignore[attr-defined]
            info = self.library.get_builtin_info(key)  # type: ignore[attr-defined]
            if info:
                from PyQt6.QtCore import Qt
                from PyQt6.QtWidgets import QListWidgetItem

                item = QListWidgetItem(f"{info['name']}")
                item.setData(Qt.ItemDataRole.UserRole, key)
                item.setToolTip(info["description"])
                self.builtin_list.addItem(item)
