"""Draggable model card widget for the launcher grid.

Provides the tile component for each model/application in the launcher.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import pyqtProperty  # type: ignore[attr-defined]
from PyQt6.QtCore import (
    QEasingCurve,
    QEvent,
    QMimeData,
    QPoint,
    QPropertyAnimation,
    Qt,
)
from PyQt6.QtGui import (
    QColor,
    QDrag,
    QDragEnterEvent,
    QDropEvent,
    QEnterEvent,
    QMouseEvent,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.style_constants import Styles
from src.shared.python.theme.typography import Weights, get_display_font, get_qfont

from .launcher_constants import (
    TILE_SCALE_DEFAULT,
    scaled_font_pt,
    scaled_image_px,
    scaled_padding_px,
    validate_tile_scale,
)
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
    "Motion Capture": "c3d_viewer_modern.png",
    "Model Explorer": "urdf_icon.png",
    "Putting Green": "putting_green_modern.png",
    "Video Analyzer": "video_analyzer_modern.png",
    "Data Explorer": "data_explorer_modern.png",
    "OpenPose": "openpose.png",
    "MediaPipe": "mediapipe.png",
    "Project Map": "project_map.png",
    "Movement Optimizer": "movement_optimizer.png",
    # Legacy names (backward compatibility)
    "MuJoCo Humanoid": "mujoco_humanoid.png",
    "MuJoCo Dashboard": "mujoco_hand.png",
    "Drake Dashboard": "drake.png",
    "Pinocchio Dashboard": "pinocchio.png",
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
    "C3D Motion Viewer": "c3d_viewer_modern.png",
    "Shot Tracer": "golf_icon.png",
    # New launcher tiles
    "Cross Engine": "cross_engine.svg",
    "Exercise Dashboard": "exercise_dashboard.svg",
    "Swing Optimizer": "swing_optimizer.svg",
    "Injury Analysis": "injury_analysis.svg",
    "Terrain Engine": "putting_green_modern.png",
    "BunkerShot 3D": "bunkershot3d.svg",
    "Pendulum": "pendulum.svg",
    "Chat Assistant": "golf_logo.png",
    "Character Builder": "urdf_icon.png",
    "Pose Studio": "pose_studio.svg",
    "Dataset Generator": "data_explorer_modern.png",
    "Golf Simulation Suite": "golf_logo.png",
    "Motion-Match Preview": "motion_target_preview.svg",
    "Video Processor": "video_analyzer.svg",
    "Data Processor": "data_explorer.svg",
    "Starting-Pose Matcher (legacy)": "motion_target_preview.svg",
}


class SkeletonCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SkeletonCard")
        self.setMinimumSize(180, 240)
        self.setStyleSheet("""
            #SkeletonCard {
                background-color: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
            }
        """)

        # Simple pulsing animation using opacity
        self.effect = QGraphicsDropShadowEffect(self)
        self.effect.setBlurRadius(20)
        self.effect.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(self.effect)

        self._anim = QPropertyAnimation(self, b"windowOpacity")
        self._anim.setDuration(1000)
        self._anim.setStartValue(0.5)
        self._anim.setEndValue(1.0)
        self._anim.setLoopCount(-1)
        self._anim.start()


class DraggableModelCard(QFrame):
    """Draggable model card widget with reordering support."""

    def __init__(
        self,
        model: Any,
        parent_launcher: Any,
        tile_scale: float = TILE_SCALE_DEFAULT,
        *,
        show_description: bool = True,
        list_mode: bool = False,
        list_compact: bool = False,
    ) -> None:
        super().__init__(None)
        self.model = model
        self.parent_launcher = parent_launcher
        self.tile_scale: float = validate_tile_scale(tile_scale)
        self._show_description: bool = bool(show_description)
        self._list_mode: bool = bool(list_mode)
        self._list_compact: bool = bool(list_compact)

        # Match initial drag-and-drop state to the parent's mode
        self.setAcceptDrops(bool(getattr(parent_launcher, "layout_edit_mode", False)))
        self.setObjectName("ModelCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.drag_start_position = QPoint()

        self.is_selected = False
        c = _get_theme_colors()

        def _color(c_obj: Any, ns_key: str, dict_key: str, default: str) -> str:
            if isinstance(c_obj, dict):
                return c_obj.get(dict_key, default)
            return getattr(c_obj, ns_key, getattr(c_obj, dict_key, default))

        self._base_style = f"""
            #ModelCard {{
                background-color: {_color(c, "surface_hover", "group_bg", "#2d2d2d")};
                border: 1px solid {_color(c, "border_light", "border", "#444444")};
                border-radius: 16px;
            }}
            #ModelCard:hover {{
                background-color: {_color(c, "surface_active", "input_bg", "#3a3a3a")};
                border: 1px solid {_color(c, "border_strong", "focus", "#666666")};
            }}
            #CardName {{
                color: {_color(c, "text_primary", "text", "#ffffff")};
            }}
            #CardDescription {{
                color: {_color(c, "text_secondary", "text_secondary", "#aaaaaa")};
            }}
        """
        self._selected_style = f"""
            #ModelCard {{
                background-color: {_color(c, "accent_muted", "title_bg", "#1a3a5a")};
                border: 2px solid {_color(c, "accent_primary", "accent", "#0a84ff")};
                border-radius: 16px;
            }}
            #ModelCard:hover {{
                background-color: {_color(c, "accent_muted", "table_alt", "#1a3a5a")};
                border: 2px solid {_color(c, "accent_hover", "focus", "#409cff")};
            }}
            #CardName {{
                color: {_color(c, "text_primary", "text", "#ffffff")};
            }}
            #CardDescription {{
                color: {_color(c, "text_secondary", "text_secondary", "#aaaaaa")};
            }}
        """
        # Glassmorphism styling - enhanced with translucent backgrounds and background-blur effect
        self.setStyleSheet(self._base_style)

        # Drop Shadow - soft elevated shadow that deepens on hover
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setOffset(0, 6)
        self.shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(self.shadow)

        # Micro-animations
        self._hover_offset = 0.0
        self._hover_anim = QPropertyAnimation(self, b"hoverOffset", self)
        self._hover_anim.setDuration(150)
        self._hover_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.setup_ui()

    @pyqtProperty(float)
    def hoverOffset(self) -> float:
        return self._hover_offset

    @hoverOffset.setter  # type: ignore[no-redef]
    def hoverOffset(self, value: float) -> None:
        self._hover_offset = value
        # Animate drop shadow - soft elevated shadow that deepens on hover
        # Base blur: 20, hover adds up to 8 more (at max hover_offset of 4.0)
        self.shadow.setBlurRadius(20 + value * 2)
        # Base offset: 6, hover adds up to 4 more for lifted effect
        self.shadow.setOffset(0, 6 + value)
        # Deepen shadow color on hover (alpha increases from 80 to 120)
        hover_alpha = int(80 + value * 10)
        self.shadow.setColor(QColor(0, 0, 0, hover_alpha))

        # Animate icon scale (scale up by 3%)
        scale_factor = 1.0 + (value / 4.0) * 0.03
        if hasattr(self, "lbl_img") and hasattr(self, "base_pixmap"):  # noqa: SIM102
            if self.base_pixmap and not self.base_pixmap.isNull():
                # In list mode, icon is fixed at 60x60 regardless of tile_scale.
                # Hover animation must use the same fixed base size to avoid
                # clipping/jitter when zoom is adjusted.
                base_px = 60 if self._list_mode else scaled_image_px(self.tile_scale)
                new_size = max(1, int(base_px * 0.9 * scale_factor))
                scaled = self.base_pixmap.scaled(
                    new_size,
                    new_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.lbl_img.setPixmap(scaled)

    def enterEvent(self, event: QEnterEvent | None) -> None:
        """Trigger micro-animation on hover enter."""
        self._hover_anim.setStartValue(self._hover_offset)
        self._hover_anim.setEndValue(4.0)
        self._hover_anim.start()
        # Reveal the per-tile launch button and info button.
        btn = getattr(self, "_btn_quick_launch", None)
        info_btn = getattr(self, "_btn_info", None)
        if btn is not None:
            btn.show()
            btn.raise_()
        if info_btn is not None:
            info_btn.show()
            info_btn.raise_()
        self._reposition_quick_launch_button()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent | None) -> None:
        """Reverse micro-animation on hover leave."""
        self._hover_anim.setStartValue(self._hover_offset)
        self._hover_anim.setEndValue(0.0)
        self._hover_anim.start()
        btn = getattr(self, "_btn_quick_launch", None)
        if btn is not None:
            btn.hide()
        info_btn = getattr(self, "_btn_info", None)
        if info_btn is not None and not self.is_selected:
            info_btn.hide()
        self._reposition_quick_launch_button()  # update info button position
        super().leaveEvent(event)

    def set_selected(self, is_selected: bool) -> None:
        """Update the glassmorphism styling to reflect selection state."""
        self.is_selected = is_selected
        self.setStyleSheet(self._selected_style if is_selected else self._base_style)
        info_btn = getattr(self, "_btn_info", None)
        if info_btn is not None:
            if is_selected:
                info_btn.show()
                info_btn.raise_()
            else:
                info_btn.hide()

    def resizeEvent(self, event: Any) -> None:  # type: ignore[override]
        """Keep the quick-launch button anchored to the top-right corner."""
        super().resizeEvent(event)
        self._reposition_quick_launch_button()

    def _reposition_quick_launch_button(self) -> None:
        btn = getattr(self, "_btn_quick_launch", None)
        info_btn = getattr(self, "_btn_info", None)
        margin = 6

        # Position launch button at top right
        if btn is not None:
            btn_w = btn.sizeHint().width()
            btn_h = btn.sizeHint().height()
            btn.setFixedSize(btn_w, btn_h)
            btn.move(self.width() - btn_w - margin, margin)

        # Position info button to the left of launch button
        if info_btn is not None:
            info_w = info_btn.sizeHint().width()
            info_h = info_btn.sizeHint().height()
            info_btn.setFixedSize(info_w, info_h)

            # If launch button is visible, put info button to its left
            if btn is not None and not btn.isHidden():
                info_btn.move(
                    self.width() - btn_w - info_w - margin * 2,
                    margin + (btn_h - info_h) // 2,
                )
            else:
                info_btn.move(self.width() - info_w - margin, margin)

    def _build_quick_launch_button(self) -> None:
        """Create the per-tile hover launch button (hidden until hover)."""
        btn = QPushButton("Launch ▶", self)
        btn.setObjectName("CardQuickLaunch")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(f"Launch {getattr(self.model, 'name', 'this model')} now")
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Visual: small pill with theme-friendly default styles.  Stylesheet
        # is intentionally light-touch so the global QSS can override it.
        btn.setStyleSheet(
            "QPushButton#CardQuickLaunch {"
            "  background: rgba(38, 110, 200, 220);"
            "  color: white;"
            "  border: none;"
            "  border-radius: 10px;"
            "  padding: 4px 10px;"
            "  font-weight: 600;"
            "  font-size: 10px;"
            "}"
            "QPushButton#CardQuickLaunch:hover {"
            "  background: rgba(64, 140, 230, 240);"
            "}"
            "QPushButton#CardQuickLaunch:pressed {"
            "  background: rgba(28, 90, 170, 240);"
            "}"
        )
        btn.clicked.connect(self._on_quick_launch_clicked)
        btn.hide()
        self._btn_quick_launch = btn

    def _on_quick_launch_clicked(self) -> None:
        """Click handler: dispatch directly to the launcher's launch path."""
        if self.parent_launcher and hasattr(
            self.parent_launcher, "launch_model_direct"
        ):
            try:
                self.parent_launcher.launch_model_direct(self.model.id)
            except Exception:  # pragma: no cover — defensive UI
                logger.exception("Quick-launch failed for model %s", self.model.id)

    def _resolve_image_name(self) -> str | None:  # noqa: C901
        """Determine the image filename for this model card."""
        # Use explicit launcher metadata if present (ensures Web App/PyQt parity)
        launcher = getattr(self.model, "launcher", None)
        if launcher and getattr(launcher, "logo", None):
            return Path(launcher.logo).name

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
            return "c3d_viewer_modern.png"
        if "model_explorer" in model_id or "urdf" in model_id:
            return "urdf_icon.png"
        if "openpose" in model_id:
            return "openpose.png"
        if "mediapipe" in model_id:
            return "mediapipe.png"
        if "project_map" in model_id:
            return "project_map.png"
        if "movement_optimizer" in model_id:
            return "movement_optimizer.png"
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

    def _create_image_widget(self, layout: QVBoxLayout | QHBoxLayout) -> None:
        """Create and add the model image label to the layout."""
        if layout is None:
            raise ValueError("layout must be provided")
        img_name = self._resolve_image_name()
        img_path = self._find_image_path(img_name)

        # LIST_SMALL → 32px, LIST_LARGE → 60px, grid → scaled
        if self._list_mode:
            img_size = 32 if self._list_compact else 60
        else:
            img_size = scaled_image_px(self.tile_scale)
        pixmap_target = max(1, int(img_size * 0.9))

        self.lbl_img = QLabel()
        self.lbl_img.setObjectName("CardImage")
        self.lbl_img.setFixedSize(img_size, img_size)
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img.setStyleSheet(Styles.LABEL_TRANSPARENT)
        self.base_pixmap = None

        if img_path and img_path.exists():
            self.base_pixmap = QPixmap(str(img_path))
            pixmap = self.base_pixmap.scaled(
                pixmap_target,
                pixmap_target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.lbl_img.setPixmap(pixmap)
        else:
            c = _get_theme_colors()
            self.lbl_img.setText("No Image")
            self.lbl_img.setStyleSheet(Styles.no_image_label(c.text_quaternary))

        if self._list_mode:
            # In list mode the icon sits to the left without centering frames.
            layout.addWidget(self.lbl_img)
            return

        layout.setAlignment(self.lbl_img, Qt.AlignmentFlag.AlignCenter)
        img_container = QWidget()
        img_layout = QHBoxLayout(img_container)
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.addStretch()
        img_layout.addWidget(self.lbl_img)
        img_layout.addStretch()
        layout.addWidget(img_container)

    def _create_status_chip(
        self, layout: QVBoxLayout | QHBoxLayout, *, embed_in_row: bool = False
    ) -> None:
        """Create and add the status chip to the layout.

        When ``embed_in_row`` is True (LIST mode) the chip is added directly
        to the supplied horizontal layout, on the right; otherwise it is
        centred in its own horizontal sub-layout (grid modes).
        """
        if layout is None:
            raise ValueError("layout must be provided")
        status_text, status_class = self._get_status_info()
        lbl_status = QLabel(status_text)
        lbl_status.setObjectName("StatusChip")
        chip_pt = max(8, scaled_font_pt(self.tile_scale, base_pt=8))
        lbl_status.setFont(get_qfont(size=chip_pt, weight=Weights.BOLD))
        lbl_status.setProperty("status_chip", status_class)
        style = lbl_status.style()
        if style:
            style.polish(lbl_status)
        lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_status.setMinimumWidth(80)

        if embed_in_row:
            layout.addWidget(lbl_status)
            return

        chip_layout = QHBoxLayout()
        chip_layout.addStretch()
        chip_layout.addWidget(lbl_status)
        chip_layout.addStretch()
        layout.addLayout(chip_layout)

    def setup_ui(self) -> None:
        """Build the model card widget layout with image, labels, and status chip."""
        if self._list_mode:
            self._setup_list_ui()
        else:
            self._setup_grid_ui()

        self._apply_card_padding()

        # Quick-launch button: small floating pill in the top-right corner,
        # revealed on hover. Hidden by default so it does not obscure the
        # tile content. Adds a one-click launch path that bypasses the
        # global header button for power users.
        self._build_quick_launch_button()

        # Info button, also floating top right
        self._build_info_button()

        self._reposition_quick_launch_button()

        # Tile-level help text.  Tooltip is a one-line preview; What's-this
        # shows the description and a usage hint.
        name = getattr(self.model, "name", "this model")
        desc = getattr(self.model, "description", "") or ""
        self.setToolTip(f"<b>{name}</b><br>{desc}<br><br><i>Double-click to launch</i>")
        self.setStatusTip(f"Selects {name}")
        self.setWhatsThis(
            f"<b>{name}</b><br>"
            f"{desc}<br><br>"
            "Double-click the tile to launch this model. "
            "Single-click selects it without launching. "
            "Recommended when you want to inspect the tile's status chip "
            "before opening the simulator."
        )

    def _build_info_button(self) -> QPushButton:
        """Create a small 'i' info button."""
        btn = QPushButton("ℹ", self)
        btn.setObjectName("CardInfoButton")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip("Click for details")
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setFixedSize(18, 18)
        btn.setStyleSheet(
            "QPushButton#CardInfoButton {"
            "  background: rgba(255, 255, 255, 0.1);"
            "  color: #aaaaaa;"
            "  border: none;"
            "  border-radius: 9px;"
            "  font-size: 10px;"
            "  font-weight: bold;"
            "}"
            "QPushButton#CardInfoButton:hover {"
            "  background: rgba(255, 255, 255, 0.2);"
            "  color: #ffffff;"
            "}"
        )
        btn.clicked.connect(self._show_info_dialog)
        btn.hide()
        self._btn_info = btn
        return btn

    def _show_info_dialog(self) -> None:
        """Show a dialog with full tile information."""
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.information(
            self,
            f"{self.model.name} Details",
            f"<b>{self.model.name}</b><br><br>{self.model.description}",
        )

    def _setup_grid_ui(self) -> None:
        """Build the vertical grid-mode layout (Comfortable/Compact/Dense)."""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._create_image_widget(layout)

        name_pt = scaled_font_pt(self.tile_scale)
        self.lbl_name = QLabel(self.model.name)
        self.lbl_name.setObjectName("CardName")
        self.lbl_name.setFont(get_display_font(size=name_pt, weight=Weights.BOLD))
        self.lbl_name.setWordWrap(True)
        self.lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Center the name
        title_layout = QHBoxLayout()
        title_layout.addStretch()
        title_layout.addWidget(self.lbl_name)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        desc_pt = max(scaled_font_pt(self.tile_scale, base_pt=9), 9)
        self.lbl_desc = QLabel(self.model.description)
        self.lbl_desc.setFont(get_qfont(size=desc_pt))
        self.lbl_desc.setObjectName("CardDescription")
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_desc.setVisible(self._show_description)
        layout.addWidget(self.lbl_desc)

        self._create_status_chip(layout)

    def _setup_list_ui(self) -> None:
        """Build the horizontal LIST-mode layout.

        LIST_LARGE: 60px icon | name + desc | status, 85px row height.
        LIST_SMALL: 32px icon | name only   | status, 40px row height.
        """
        outer = QHBoxLayout(self)

        if self._list_compact:
            # ── LIST_SMALL: compact single-line rows ──
            outer.setSpacing(8)
            outer.setContentsMargins(6, 2, 6, 2)
            self._create_image_widget(outer)

            name_pt = max(scaled_font_pt(self.tile_scale, base_pt=10), 9)
            self.lbl_name = QLabel(self.model.name)
            self.lbl_name.setObjectName("CardName")
            self.lbl_name.setFont(get_display_font(size=name_pt, weight=Weights.BOLD))

            self.lbl_desc = QLabel(self.model.description)
            self.lbl_desc.setObjectName("CardDescription")
            self.lbl_desc.setFont(get_qfont(size=8))
            self.lbl_desc.setVisible(False)

            outer.addWidget(self.lbl_name, 1)
            self._create_status_chip(outer, embed_in_row=True)
            self.setFixedHeight(40)
        else:
            # ── LIST_LARGE: original list with description ──
            outer.setSpacing(12)
            self._create_image_widget(outer)

            text_box = QVBoxLayout()
            text_box.setSpacing(2)
            name_pt = max(scaled_font_pt(self.tile_scale, base_pt=12), 10)
            self.lbl_name = QLabel(self.model.name)
            self.lbl_name.setObjectName("CardName")
            self.lbl_name.setFont(get_display_font(size=name_pt, weight=Weights.BOLD))

            name_layout = QHBoxLayout()
            name_layout.addWidget(self.lbl_name)
            name_layout.addStretch()
            text_box.addLayout(name_layout)

            self.lbl_desc = QLabel(self.model.description)
            self.lbl_desc.setObjectName("CardDescription")
            self.lbl_desc.setFont(get_qfont(size=9))
            self.lbl_desc.setVisible(self._show_description)
            text_box.addWidget(self.lbl_desc)

            outer.addLayout(text_box, 1)
            self._create_status_chip(outer, embed_in_row=True)
            self.setMinimumHeight(85)

    def _apply_card_padding(self) -> None:
        """Set contents margins on the active layout based on tile_scale."""
        pad = scaled_padding_px(self.tile_scale)
        active = self.layout()
        if active is not None:
            active.setContentsMargins(pad, pad, pad, pad)

    def set_tile_scale(
        self,
        scale: float,
        *,
        show_description: bool | None = None,
        list_mode: bool | None = None,
        list_compact: bool | None = None,
    ) -> None:
        """Resize this card in place using the supplied tile scale.

        Existing labels/pixmap are reused — no disk reload — but the layout
        is rebuilt when ``list_mode`` or ``list_compact`` changes.
        """
        scale = validate_tile_scale(scale)
        if show_description is not None:
            self._show_description = bool(show_description)
        new_list_mode = self._list_mode if list_mode is None else bool(list_mode)
        new_list_compact = (
            self._list_compact if list_compact is None else bool(list_compact)
        )
        full_rebuild = (new_list_mode != self._list_mode) or (
            new_list_compact != self._list_compact
        )
        self.tile_scale = scale
        self._list_mode = new_list_mode
        self._list_compact = new_list_compact

        if full_rebuild:
            # Clear children + layout, then rebuild from scratch.
            old_layout = self.layout()
            if old_layout is not None:
                while old_layout.count():
                    item = old_layout.takeAt(0)
                    w = item.widget() if item else None
                    if w is not None:
                        w.setParent(None)
                        w.deleteLater()
                # PyQt6: detach the old layout by reparenting to a temp QWidget.
                QWidget().setLayout(old_layout)
            self.setup_ui()
            return

        # In-place update: image, fonts, padding, description visibility.
        new_img = 60 if self._list_mode else scaled_image_px(self.tile_scale)
        if hasattr(self, "lbl_img"):
            self.lbl_img.setFixedSize(new_img, new_img)
            if self.base_pixmap and not self.base_pixmap.isNull():
                target = max(1, int(new_img * 0.9))
                self.lbl_img.setPixmap(
                    self.base_pixmap.scaled(
                        target,
                        target,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        if hasattr(self, "lbl_name"):
            base_pt = 12 if self._list_mode else 11
            self.lbl_name.setFont(
                get_display_font(
                    size=scaled_font_pt(self.tile_scale, base_pt=base_pt),
                    weight=Weights.BOLD,
                )
            )
        if hasattr(self, "lbl_desc"):
            self.lbl_desc.setVisible(self._show_description)
            self.lbl_desc.setFont(
                get_qfont(size=max(scaled_font_pt(self.tile_scale, base_pt=9), 9))
            )
        chip = self.findChild(QLabel, "StatusChip")
        if chip is not None:
            chip.setFont(
                get_qfont(
                    size=max(8, scaled_font_pt(self.tile_scale, base_pt=8)),
                    weight=Weights.BOLD,
                )
            )
        self._apply_card_padding()

    # Status string -> (display_text, css_class) used when the YAML supplies
    # an explicit launcher.status. Anything else falls back to type-based
    # detection so older entries without a launcher block still render
    # something meaningful (no more "Unknown" chips).
    _STATUS_STRINGS: dict[str, tuple[str, str]] = {
        "ready": ("Ready", "success"),
        "available": ("Ready", "success"),
        "stable": ("Ready", "success"),
        "beta": ("Beta", "info"),
        "experimental": ("Experimental", "info"),
        "alpha": ("Alpha", "warning"),
        "broken": ("Broken", "error"),
        "deprecated": ("Deprecated", "warning"),
        "external": ("External", "external"),
    }

    def _get_status_info(self) -> tuple[str, str]:
        # 1. Prefer an explicit launcher.status from the YAML — that is the
        #    canonical declaration. Most tiles already set it.
        launcher = getattr(self.model, "launcher", None)
        if isinstance(launcher, dict):
            yaml_status = launcher.get("status")
        else:
            yaml_status = getattr(launcher, "status", None) if launcher else None
        if isinstance(yaml_status, str):
            mapped = self._STATUS_STRINGS.get(yaml_status.strip().lower())
            if mapped is not None:
                return mapped

        # 2. Fall back to type-based detection for legacy entries.
        t = getattr(self.model, "type", "").lower()
        if t in [
            "custom_humanoid",
            "custom_dashboard",
            "drake",
            "pinocchio",
            "openpose",
        ]:
            return "GUI Ready", "success"

        path_str = str(getattr(self.model, "path", ""))
        if t == "mjcf" or path_str.endswith(".xml"):
            return "Viewer", "info"
        if t in ["opensim", "myosim"]:
            return "Engine Ready", "success"
        if t in ["matlab", "matlab_app", "matlab_suite"]:
            return "External", "external"
        if t in ["urdf_generator", "c3d_viewer"]:
            return "Utility", "utility"
        if t == "putting_green":
            return "Ready", "success"
        if t == "special_app":
            return "Ready", "success"
        if t == "document":
            return "Reference", "info"

        return "Ready", "success"

    def refresh_theme(self) -> None:
        """Refresh inline styles to match the current theme."""
        c = _get_theme_colors()
        # Update description label
        desc = self.findChild(QLabel, "CardDescription")
        if desc:
            desc.setStyleSheet(f"color: {c.text_secondary};")
        # Update status chip
        status_text, status_class = self._get_status_info()
        chip = self.findChild(QLabel, "StatusChip")
        if chip:
            chip.setProperty("status_chip", status_class)
            style = chip.style()
            if style:
                style.polish(chip)

            # Compute a text color based on background

            bg_color = chip.palette().color(chip.backgroundRole())  # noqa: F841
            # For QLabels styled with QSS, we need to extract from the computed styles or just use heuristics.
            # To be safe, we will apply an explicit style.
            # If the background is bright, use black. If dark, use white.
            # Usually 'warning' (yellow) and 'success' (light green) are bright, while 'error' (red) and 'info' (blue) are dark.
            if status_class in ("warning", "success"):
                chip.setStyleSheet("color: black;")
            else:
                chip.setStyleSheet("color: white;")
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

    def keyPressEvent(self, event: Any) -> None:
        """Handle keyboard navigation and activation.

        Supports:
        - Enter/Return: Launch the model
        - Space: Select the model
        - Arrow keys: Navigate to adjacent cards (handled by parent grid)
        """
        if event and event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            if self.parent_launcher:
                self.parent_launcher.launch_model_direct(self.model.id)
        elif event and event.key() == Qt.Key.Key_Space:
            if self.parent_launcher:
                self.parent_launcher.select_model(self.model.id)
        else:
            super().keyPressEvent(event)

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
