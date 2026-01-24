"""Recent models quick access panel for Golf Modeling Suite.

Provides a collapsible panel showing recently used models for quick access.

Usage:
    from shared.python.ui.recent_models import RecentModelsPanel

    panel = RecentModelsPanel()
    panel.model_selected.connect(on_model_selected)
    panel.add_recent("mujoco_humanoid", "MuJoCo Humanoid")
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_config import get_logger

if TYPE_CHECKING:
    pass

# Import theme if available
try:
    from shared.python.theme import Colors, Sizes, Weights, get_qfont

    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False

logger = get_logger(__name__)

# Recent models storage
RECENT_FILE = Path.home() / ".golf_modeling_suite" / "recent_models.json"
MAX_RECENT = 10


class RecentModelItem(QFrame):
    """Individual recent model item widget."""

    clicked = pyqtSignal(str)  # Emits model_id

    def __init__(
        self,
        model_id: str,
        display_name: str,
        parent: QWidget | None = None,
    ) -> None:
        """Create a recent model item.

        Args:
            model_id: Unique model identifier
            display_name: Human-readable name
            parent: Parent widget
        """
        super().__init__(parent)
        self.model_id = model_id
        self.display_name = display_name

        self._setup_ui()
        self._setup_style()

    def _setup_ui(self) -> None:
        """Set up the item UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # Model name
        name_label = QLabel(self.display_name)
        if THEME_AVAILABLE:
            name_label.setFont(get_qfont(size=Sizes.SM, weight=Weights.MEDIUM))
        else:
            font = QFont("Segoe UI", 9)
            font.setWeight(QFont.Weight.Medium)
            name_label.setFont(font)
        layout.addWidget(name_label)

        layout.addStretch()

        # Launch button (appears on hover)
        self.launch_btn = QPushButton("\u25b6")  # Play symbol
        self.launch_btn.setFixedSize(24, 24)
        self.launch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.launch_btn.clicked.connect(lambda: self.clicked.emit(self.model_id))
        self.launch_btn.setVisible(False)
        layout.addWidget(self.launch_btn)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _setup_style(self) -> None:
        """Apply styling."""
        if THEME_AVAILABLE:
            self.setStyleSheet(
                f"QFrame {{"
                f"  background-color: transparent;"
                f"  border-radius: 6px;"
                f"}}"
                f"QFrame:hover {{"
                f"  background-color: {Colors.BG_ELEVATED};"
                f"}}"
                f"QLabel {{"
                f"  color: {Colors.TEXT_SECONDARY};"
                f"  background: transparent;"
                f"}}"
                f"QPushButton {{"
                f"  background-color: {Colors.PRIMARY};"
                f"  color: white;"
                f"  border: none;"
                f"  border-radius: 12px;"
                f"}}"
                f"QPushButton:hover {{"
                f"  background-color: {Colors.PRIMARY_HOVER};"
                f"}}"
            )
        else:
            self.setStyleSheet(
                "QFrame {"
                "  background-color: transparent;"
                "  border-radius: 6px;"
                "}"
                "QFrame:hover {"
                "  background-color: #2D2D2D;"
                "}"
                "QLabel {"
                "  color: #E0E0E0;"
                "  background: transparent;"
                "}"
                "QPushButton {"
                "  background-color: #0A84FF;"
                "  color: white;"
                "  border: none;"
                "  border-radius: 12px;"
                "}"
            )

    def enterEvent(self, event: Any) -> None:
        """Show launch button on hover."""
        self.launch_btn.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:
        """Hide launch button when not hovering."""
        self.launch_btn.setVisible(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event: Any) -> None:
        """Handle click on entire item."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.model_id)
        super().mousePressEvent(event)


class RecentModelsPanel(QFrame):
    """Collapsible panel showing recently used models."""

    model_selected = pyqtSignal(str)  # Emits model_id when user selects

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the recent models panel.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._collapsed = False
        self._recent_models: list[tuple[str, str]] = []  # [(id, name), ...]

        self._setup_ui()
        self._load_recent()

    def _setup_ui(self) -> None:
        """Set up the panel UI."""
        self.setObjectName("recentModelsPanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setObjectName("recentHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 10, 12, 10)

        title = QLabel("Recent Models")
        if THEME_AVAILABLE:
            title.setFont(get_qfont(size=Sizes.SM, weight=Weights.SEMIBOLD))
        else:
            font = QFont("Segoe UI", 9)
            font.setWeight(QFont.Weight.DemiBold)
            title.setFont(font)
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Collapse button
        self.collapse_btn = QPushButton("\u25bc")  # Down arrow
        self.collapse_btn.setFixedSize(20, 20)
        self.collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.collapse_btn.clicked.connect(self._toggle_collapse)
        header_layout.addWidget(self.collapse_btn)

        layout.addWidget(header)

        # Content area (scrollable)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(4, 4, 4, 4)
        self.content_layout.setSpacing(2)

        scroll = QScrollArea()
        scroll.setWidget(self.content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMaximumHeight(200)

        layout.addWidget(scroll)

        # Empty state label
        self.empty_label = QLabel("No recent models")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if THEME_AVAILABLE:
            self.empty_label.setStyleSheet(f"color: {Colors.TEXT_QUATERNARY};")
        else:
            self.empty_label.setStyleSheet("color: #666666;")
        self.content_layout.addWidget(self.empty_label)

        self._apply_style()

    def _apply_style(self) -> None:
        """Apply panel styling."""
        if THEME_AVAILABLE:
            self.setStyleSheet(
                f"QFrame#recentModelsPanel {{"
                f"  background-color: {Colors.BG_SURFACE};"
                f"  border: 1px solid {Colors.BORDER_SUBTLE};"
                f"  border-radius: 8px;"
                f"}}"
                f"QFrame#recentHeader {{"
                f"  background-color: transparent;"
                f"  border-bottom: 1px solid {Colors.BORDER_SUBTLE};"
                f"}}"
                f"QLabel {{"
                f"  color: {Colors.TEXT_SECONDARY};"
                f"}}"
                f"QPushButton {{"
                f"  background-color: transparent;"
                f"  color: {Colors.TEXT_TERTIARY};"
                f"  border: none;"
                f"}}"
                f"QPushButton:hover {{"
                f"  color: {Colors.TEXT_PRIMARY};"
                f"}}"
            )
        else:
            self.setStyleSheet(
                "QFrame#recentModelsPanel {"
                "  background-color: #242424;"
                "  border: 1px solid #2D2D2D;"
                "  border-radius: 8px;"
                "}"
                "QFrame#recentHeader {"
                "  background-color: transparent;"
                "  border-bottom: 1px solid #2D2D2D;"
                "}"
                "QLabel {"
                "  color: #E0E0E0;"
                "}"
                "QPushButton {"
                "  background-color: transparent;"
                "  color: #A0A0A0;"
                "  border: none;"
                "}"
            )

    def _toggle_collapse(self) -> None:
        """Toggle collapsed state."""
        self._collapsed = not self._collapsed
        self.content.setVisible(not self._collapsed)
        self.collapse_btn.setText("\u25b6" if self._collapsed else "\u25bc")

    def _load_recent(self) -> None:
        """Load recent models from storage."""
        if RECENT_FILE.exists():
            try:
                data = json.loads(RECENT_FILE.read_text())
                self._recent_models = [(m["id"], m["name"]) for m in data[:MAX_RECENT]]
                self._refresh_list()
            except Exception as e:
                logger.warning(f"Failed to load recent models: {e}")

    def _save_recent(self) -> None:
        """Save recent models to storage."""
        try:
            RECENT_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = [{"id": m[0], "name": m[1]} for m in self._recent_models]
            RECENT_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save recent models: {e}")

    def _refresh_list(self) -> None:
        """Refresh the displayed list."""
        # Clear existing items (except empty label)
        for i in reversed(range(self.content_layout.count())):
            item = self.content_layout.itemAt(i)
            widget = item.widget() if item else None
            if widget and widget != self.empty_label:
                widget.deleteLater()

        # Show/hide empty state
        self.empty_label.setVisible(len(self._recent_models) == 0)

        # Add items
        for model_id, display_name in self._recent_models:
            model_item = RecentModelItem(model_id, display_name)
            model_item.clicked.connect(self._on_item_clicked)
            self.content_layout.insertWidget(
                self.content_layout.count() - 1, model_item
            )

    def _on_item_clicked(self, model_id: str) -> None:
        """Handle item click."""
        self.model_selected.emit(model_id)

    def add_recent(self, model_id: str, display_name: str) -> None:
        """Add a model to the recent list.

        Args:
            model_id: Unique model identifier
            display_name: Human-readable name
        """
        # Remove if already exists
        self._recent_models = [
            (mid, name) for mid, name in self._recent_models if mid != model_id
        ]

        # Add to front
        self._recent_models.insert(0, (model_id, display_name))

        # Trim to max
        self._recent_models = self._recent_models[:MAX_RECENT]

        # Save and refresh
        self._save_recent()
        self._refresh_list()

    def clear_recent(self) -> None:
        """Clear all recent models with confirmation dialog.

        UX FIX: Added confirmation to prevent accidental history loss.
        """
        if not self._recent_models:
            return  # Nothing to clear

        # UX FIX: Confirm before clearing recent models
        reply = QMessageBox.question(
            self,
            "Clear Recent Models",
            f"This will clear {len(self._recent_models)} recent model(s) from history.\n\n"
            "This action cannot be undone. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._recent_models = []
            self._save_recent()
            self._refresh_list()
            logger.info("Recent models history cleared")


__all__ = ["RecentModelItem", "RecentModelsPanel"]
