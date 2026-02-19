"""Draggable model card widget for the launcher grid.

Provides the tile component for each model/application in the launcher.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QMimeData, QPoint, Qt
from PyQt6.QtGui import QDrag, QDragEnterEvent, QDropEvent, QFont, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.style_constants import Styles

from .startup import ASSETS_DIR, _get_theme_colors

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Tile image file names
_IMG_SIMSCAPE = "simscape_multibody.png"
_IMG_MATLAB = "matlab_logo.png"

# Maps display names to tile image files in assets/
MODEL_IMAGES = {
    # Physics Engines - Current names from models.yaml
    "MuJoCo": "mujoco_humanoid.png",
    "Drake": "drake.png",
    "Pinocchio": "pinocchio.png",
    "OpenSim": "opensim.png",
    "MyoSuite": "myosim.png",
    # MATLAB/Simscape
    "Matlab Models": _IMG_MATLAB,
    # Tools
    "Motion Capture": "c3d_icon.png",
    "Model Explorer": "urdf_icon.png",
    "Putting Green": "putting_green.svg",
    "Video Analyzer": "video_analyzer.svg",
    "Data Explorer": "data_explorer.svg",
    # Legacy names (backward compatibility)
    "MuJoCo Humanoid": "mujoco_humanoid.png",
    "MuJoCo Dashboard": "mujoco_hand.png",
    "Drake Golf Model": "drake.png",
    "Pinocchio Golf Model": "pinocchio.png",
    "OpenSim Golf": "opensim.png",
    "MyoSim Suite": "myosim.png",
    "OpenPose Analysis": "openpose.jpg",
    "Matlab Simscape": _IMG_MATLAB,
    "Matlab Simscape 2D": _IMG_MATLAB,
    "Matlab Simscape 3D": _IMG_MATLAB,
    "Dataset Generator GUI": _IMG_MATLAB,
    "Golf Swing Analysis GUI": _IMG_MATLAB,
    "MATLAB Code Analyzer": _IMG_MATLAB,
    "URDF Generator": "urdf_icon.png",
    "C3D Motion Viewer": "c3d_icon.png",
    "Shot Tracer": "golf_icon.png",
}


class DraggableModelCard(QFrame):
    """Draggable model card widget with reordering support."""

    def __init__(self, model: Any, parent_launcher: Any) -> None:
        super().__init__(None)
        self.model = model
        self.parent_launcher = parent_launcher

        # Match initial drag-and-drop state to the parent's mode
        self.setAcceptDrops(bool(getattr(parent_launcher, "layout_edit_mode", False)))
        self.setObjectName("ModelCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.drag_start_position = QPoint()

        self.setup_ui()

    def _resolve_image_name(self) -> str | None:
        """Determine the image filename for this model card."""
        img_name = MODEL_IMAGES.get(self.model.name)
        if img_name:
            return img_name

        model_id = self.model.id.lower()
        if "mujoco" in model_id:
            return "mujoco_humanoid.png"
        if "drake" in model_id:
            return "drake.png"
        if "pinocchio" in model_id:
            return "pinocchio.png"
        if "opensim" in model_id:
            return "opensim.png"
        if "myosim" in model_id or "myosuite" in model_id:
            return "myosim.png"
        if "matlab" in model_id:
            return "matlab_logo.png"
        if "motion" in model_id or "capture" in model_id or "c3d" in model_id:
            return "c3d_icon.png"
        if "model_explorer" in model_id or "urdf" in model_id:
            return "urdf_icon.png"
        if (
            "engine_managed" in getattr(self.model, "type", "")
            and getattr(self.model, "engine_type", "") == "mujoco"
        ):
            return "mujoco_humanoid.png"
        return None

    @staticmethod
    def _find_image_path(img_name: str | None) -> Path | None:
        """Locate the image file in assets or SVG logos directories."""
        if not img_name:
            return None
        img_path = ASSETS_DIR / img_name
        if img_path.exists():
            return img_path
        svg_logos_dir = Path(__file__).parent.parent.parent / "assets" / "logos"
        img_path = svg_logos_dir / img_name
        if img_path.exists():
            return img_path
        return None

    def _create_image_widget(self, layout: QVBoxLayout) -> None:
        """Create and add the model image label to the layout."""
        img_name = self._resolve_image_name()
        img_path = self._find_image_path(img_name)

        lbl_img = QLabel()
        lbl_img.setObjectName("CardImage")
        lbl_img.setFixedSize(200, 200)
        lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setAlignment(lbl_img, Qt.AlignmentFlag.AlignCenter)
        lbl_img.setStyleSheet(Styles.LABEL_TRANSPARENT)

        if img_path and img_path.exists():
            pixmap = QPixmap(str(img_path))
            pixmap = pixmap.scaled(
                180,
                180,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            lbl_img.setPixmap(pixmap)
        else:
            c = _get_theme_colors()
            lbl_img.setText("No Image")
            lbl_img.setStyleSheet(Styles.no_image_label(c.text_quaternary))

        img_container = QWidget()
        img_layout = QHBoxLayout(img_container)
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.addStretch()
        img_layout.addWidget(lbl_img)
        img_layout.addStretch()
        layout.addWidget(img_container)

    def _create_status_chip(self, layout: QVBoxLayout) -> None:
        """Create and add the status chip to the layout."""
        status_text, status_color, text_color = self._get_status_info()
        lbl_status = QLabel(status_text)
        lbl_status.setObjectName("StatusChip")
        lbl_status.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        lbl_status.setStyleSheet(Styles.status_chip(status_color, text_color))
        lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_status.setMinimumWidth(80)

        chip_layout = QHBoxLayout()
        chip_layout.addStretch()
        chip_layout.addWidget(lbl_status)
        chip_layout.addStretch()
        layout.addLayout(chip_layout)

    def setup_ui(self) -> None:
        """Build the model card widget layout with image, labels, and status chip."""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._create_image_widget(layout)

        lbl_name = QLabel(self.model.name)
        lbl_name.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        lbl_name.setWordWrap(True)
        lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_name)

        lbl_desc = QLabel(self.model.description)
        lbl_desc.setFont(QFont("Segoe UI", 9))
        lbl_desc.setObjectName("CardDescription")
        lbl_desc.setWordWrap(True)
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_desc)

        self._create_status_chip(layout)

    def _get_status_info(self) -> tuple[str, str, str]:
        c = _get_theme_colors()
        t = getattr(self.model, "type", "").lower()
        if t in [
            "custom_humanoid",
            "custom_dashboard",
            "drake",
            "pinocchio",
            "openpose",
        ]:
            return "GUI Ready", c.success, "#000000"

        path_str = str(getattr(self.model, "path", ""))
        if t == "mjcf" or path_str.endswith(".xml"):
            return "Viewer", c.chart_cyan, "#000000"
        if t in ["opensim", "myosim"]:
            return "Engine Ready", c.success, "#000000"
        if t in ["matlab", "matlab_app"]:
            return "External", c.chart_purple, "#ffffff"
        if t in ["urdf_generator", "c3d_viewer"]:
            return "Utility", c.text_tertiary, "#ffffff"

        return "Unknown", c.text_tertiary, "#ffffff"

    def refresh_theme(self) -> None:
        """Refresh inline styles to match the current theme."""
        c = _get_theme_colors()
        # Update description label
        desc = self.findChild(QLabel, "CardDescription")
        if desc:
            desc.setStyleSheet(f"color: {c.text_secondary};")
        # Update status chip
        status_text, status_color, text_color = self._get_status_info()
        chip = self.findChild(QLabel, "StatusChip")
        if chip:
            chip.setStyleSheet(Styles.status_chip(status_color, text_color))
        # Update no-image fallback
        img = self.findChild(QLabel, "CardImage")
        if img and not img.pixmap():
            img.setStyleSheet(Styles.no_image_label(c.text_quaternary))

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        """Handle left-click to select this model card."""
        if event and event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_position = event.position().toPoint()
            if self.parent_launcher:
                self.parent_launcher.select_model(self.model.id)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        """Initiate drag-and-drop when in layout-edit mode."""
        if not event or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if not getattr(self.parent_launcher, "layout_edit_mode", False):
            return

        if (
            event.position().toPoint() - self.drag_start_position
        ).manhattanLength() < QApplication.startDragDistance():
            return

        drag = QDrag(self)
        mimeData = QMimeData()
        mimeData.setText(f"model_card:{self.model.id}")
        drag.setMimeData(mimeData)
        drag.setPixmap(self.grab())
        drag.setHotSpot(self.drag_start_position)
        drag.exec(Qt.DropAction.MoveAction)

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:
        """Launch the model directly on double-click."""
        if self.parent_launcher:
            self.parent_launcher.launch_model_direct(self.model.id)

    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        """Accept drag events carrying a model card identifier."""
        if not event:
            return

        mime_data = event.mimeData()
        if (
            mime_data
            and mime_data.hasText()
            and mime_data.text().startswith("model_card:")
        ):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent | None) -> None:
        """Swap model card positions on drop."""
        if not event:
            return

        mime_data = event.mimeData()
        if mime_data and mime_data.hasText():
            source_id = mime_data.text().split(":")[1]
            if self.parent_launcher and source_id != self.model.id:
                self.parent_launcher._swap_models(source_id, self.model.id)
            event.acceptProposedAction()
