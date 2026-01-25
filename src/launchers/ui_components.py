"""UI Components for the Golf Modeling Suite Launcher.

This module provides specialized widgets and data containers to improve 
the modularity and maintainability of the launcher.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class StartupResults:
    """Container for asynchronous startup metrics and pre-loaded data."""

    def __init__(self) -> None:
        self.registry: Any = None
        self.engine_manager: Any = None
        self.available_engines: list = []
        self.ai_available: bool = False
        self.docker_available: bool = False
        self.startup_time_ms: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> StartupResults:
        """Create StartupResults from worker results dict."""
        results = cls()
        results.registry = data.get("registry")
        results.engine_manager = data.get("engine_manager")
        results.available_engines = data.get("available_engines", [])
        results.ai_available = data.get("ai_available", False)
        results.docker_available = data.get("docker_available", False)
        results.startup_time_ms = data.get("startup_time_ms", 0)
        return results


class DraggableModelCard(QFrame):
    """Draggable model card widget with reordering support.
    
    Orthogonality: Encapsulates the visual representation and drag-and-drop
    behavior of a single model entry.
    """

    def __init__(self, model: Any, parent_launcher: Any, assets_dir: Path, image_map: dict[str, str]):
        super().__init__(parent_launcher)
        self.model = model
        self.parent_launcher = parent_launcher
        self.assets_dir = assets_dir
        self.image_map = image_map
        
        # Match initial drag-and-drop state to the parent's mode
        self.setAcceptDrops(bool(getattr(parent_launcher, "layout_edit_mode", False)))
        self.setObjectName("ModelCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.drag_start_position = QPoint()
        
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the card UI (Decomposed for clarity)."""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 1. Component: Image
        self._add_image_section(layout)

        # 2. Component: Labels
        self._add_label_section(layout)

    def _add_image_section(self, layout: QVBoxLayout) -> None:
        img_name = self.image_map.get(self.model.name, "default_icon.png")
        img_path = self.assets_dir / img_name

        lbl_img = QLabel()
        lbl_img.setFixedSize(200, 200)
        lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_img.setStyleSheet("QLabel { border: none; background: transparent; }")

        if img_path.exists():
            pixmap = QPixmap(str(img_path))
            pixmap = pixmap.scaled(
                180, 180, 
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            lbl_img.setPixmap(pixmap)
        else:
            lbl_img.setText("No Image")
            lbl_img.setStyleSheet("color: #666; font-style: italic;")

        layout.addWidget(lbl_img)

    def _add_label_section(self, layout: QVBoxLayout) -> None:
        lbl_name = QLabel(self.model.name)
        lbl_name.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        lbl_name.setWordWrap(True)
        lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_name)

        lbl_desc = QLabel(self.model.description)
        lbl_desc.setFont(QFont("Segoe UI", 9))
        lbl_desc.setStyleSheet("color: #cccccc;")
        lbl_desc.setWordWrap(True)
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_desc)
