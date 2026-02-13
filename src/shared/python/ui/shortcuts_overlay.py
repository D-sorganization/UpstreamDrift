"""Keyboard shortcuts overlay for Golf Modeling Suite.

Displays a modal overlay showing all available keyboard shortcuts
when the user presses Ctrl+? or F1.

Usage:
    from shared.python.ui.shortcuts_overlay import ShortcutsOverlay

    # Create and show
    overlay = ShortcutsOverlay(parent_window)
    overlay.show()
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QKeyEvent, QPainter
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow

# Import theme if available
try:
    from src.shared.python.theme import Colors, Sizes, Weights, get_qfont

    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False

from src.shared.python.theme.style_constants import Styles


@dataclass
class Shortcut:
    """Represents a keyboard shortcut."""

    keys: str  # e.g., "Ctrl+S"
    description: str  # e.g., "Save file"
    category: str = "General"  # Grouping category


# Default shortcuts for the Golf Modeling Suite
DEFAULT_SHORTCUTS = [
    # General
    Shortcut("Ctrl+?", "Show keyboard shortcuts", "General"),
    Shortcut("F1", "Show keyboard shortcuts", "General"),
    Shortcut("Escape", "Close dialog/overlay", "General"),
    Shortcut("Ctrl+Q", "Quit application", "General"),
    # Navigation
    Shortcut("Ctrl+F", "Search/filter models", "Navigation"),
    Shortcut("Tab", "Move to next element", "Navigation"),
    Shortcut("Shift+Tab", "Move to previous element", "Navigation"),
    Shortcut("Enter", "Launch selected model", "Navigation"),
    # Simulation
    Shortcut("Space", "Pause/Resume simulation", "Simulation"),
    Shortcut("Ctrl+R", "Start simulation", "Simulation"),
    Shortcut("S", "Stop simulation", "Simulation"),
    Shortcut("R", "Reset simulation", "Simulation"),
    # Layout
    Shortcut("Ctrl+L", "Toggle layout edit mode", "Layout"),
    Shortcut("Ctrl+S", "Save current layout", "Layout"),
    # Help
    Shortcut("Ctrl+H", "Toggle help panel", "Help"),
    Shortcut("Ctrl+,", "Open preferences", "Help"),
]


class ShortcutBadge(QFrame):
    """Styled keyboard shortcut badge (e.g., displays "Ctrl+S")."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.text = text
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the badge UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # Split by + and create individual key badges
        keys = self.text.split("+")
        for i, key in enumerate(keys):
            if i > 0:
                plus = QLabel("+")
                if THEME_AVAILABLE:
                    plus.setStyleSheet(f"color: {Colors.TEXT_TERTIARY};")
                else:
                    plus.setStyleSheet(Styles.SHORTCUT_PLUS)
                layout.addWidget(plus)

            key_label = QLabel(key.strip())
            if THEME_AVAILABLE:
                font = get_qfont(size=Sizes.SM, weight=Weights.SEMIBOLD)
                key_label.setStyleSheet(
                    f"background-color: {Colors.BG_SURFACE}; "
                    f"color: {Colors.TEXT_PRIMARY}; "
                    f"border: 1px solid {Colors.BORDER_DEFAULT}; "
                    "border-radius: 4px; "
                    "padding: 2px 6px;"
                )
            else:
                font = QFont("Segoe UI", 9)
                font.setWeight(QFont.Weight.DemiBold)
                key_label.setStyleSheet(Styles.SHORTCUT_KEY_BADGE)
            key_label.setFont(font)
            layout.addWidget(key_label)

        self.setStyleSheet("background: transparent;")


class ShortcutsOverlay(QWidget):
    """Modal overlay displaying keyboard shortcuts.

    Appears centered over the parent window with a semi-transparent
    backdrop. Dismissed by clicking outside or pressing Escape.
    """

    def __init__(
        self,
        parent: "QMainWindow",
        shortcuts: list[Shortcut] | None = None,
    ) -> None:
        """Create the shortcuts overlay.

        Args:
            parent: Parent window
            shortcuts: List of shortcuts to display (uses defaults if None)
        """
        super().__init__(parent)
        self.shortcuts = shortcuts or DEFAULT_SHORTCUTS
        self._setup_ui()

        # Make overlay cover entire parent
        self.setGeometry(parent.rect())

        # Set up for modal behavior
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _setup_ui(self) -> None:
        """Set up the overlay UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        content = self._create_content_frame()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 20, 24, 20)
        content_layout.setSpacing(16)

        self._create_header(content_layout)
        self._create_shortcuts_scroll(content_layout)
        self._create_footer_hint(content_layout)

        layout.addStretch()
        h_layout = QHBoxLayout()
        h_layout.addStretch()
        h_layout.addWidget(content)
        h_layout.addStretch()
        layout.addLayout(h_layout)
        layout.addStretch()

    def _create_content_frame(self) -> QFrame:
        content = QFrame()
        content.setObjectName("shortcutsContent")
        content.setFixedSize(600, 500)

        if THEME_AVAILABLE:
            content.setStyleSheet(
                f"QFrame#shortcutsContent {{"
                f"  background-color: {Colors.BG_BASE};"
                f"  border: 1px solid {Colors.BORDER_DEFAULT};"
                f"  border-radius: 12px;"
                f"}}"
            )
        else:
            content.setStyleSheet(Styles.SHORTCUT_CONTENT)
        return content

    def _create_header(self, content_layout: QVBoxLayout) -> None:
        header = QHBoxLayout()
        title = QLabel("Keyboard Shortcuts")
        if THEME_AVAILABLE:
            title.setFont(get_qfont(size=Sizes.LG, weight=Weights.BOLD))
            title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
        else:
            font = QFont("Segoe UI", 13)
            font.setWeight(QFont.Weight.Bold)
            title.setFont(font)
            title.setStyleSheet(Styles.SHORTCUT_TITLE)

        close_btn = QPushButton("\u2715")
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        if THEME_AVAILABLE:
            close_btn.setStyleSheet(
                f"QPushButton {{"
                f"  background-color: transparent;"
                f"  color: {Colors.TEXT_TERTIARY};"
                f"  border: none;"
                f"  border-radius: 14px;"
                f"  font-size: 14px;"
                f"}}"
                f"QPushButton:hover {{"
                f"  background-color: {Colors.BG_ELEVATED};"
                f"  color: {Colors.TEXT_PRIMARY};"
                f"}}"
            )
        else:
            close_btn.setStyleSheet(Styles.SHORTCUT_CLOSE_BTN)

        header.addWidget(title)
        header.addStretch()
        header.addWidget(close_btn)
        content_layout.addLayout(header)

    def _create_shortcuts_scroll(self, content_layout: QVBoxLayout) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(Styles.TRANSPARENT_BG)

        shortcuts_widget = QWidget()
        shortcuts_layout = QVBoxLayout(shortcuts_widget)
        shortcuts_layout.setContentsMargins(0, 0, 0, 0)
        shortcuts_layout.setSpacing(20)

        categories: dict[str, list[Shortcut]] = {}
        for shortcut in self.shortcuts:
            if shortcut.category not in categories:
                categories[shortcut.category] = []
            categories[shortcut.category].append(shortcut)

        for category, cat_shortcuts in categories.items():
            section = self._create_category_section(category, cat_shortcuts)
            shortcuts_layout.addWidget(section)

        shortcuts_layout.addStretch()
        scroll.setWidget(shortcuts_widget)
        content_layout.addWidget(scroll)

    @staticmethod
    def _create_footer_hint(content_layout: QVBoxLayout) -> None:
        hint = QLabel("Press Escape or click outside to close")
        if THEME_AVAILABLE:
            hint.setFont(get_qfont(size=Sizes.XS))
            hint.setStyleSheet(f"color: {Colors.TEXT_QUATERNARY};")
        else:
            hint.setFont(QFont("Segoe UI", 8))
            hint.setStyleSheet(Styles.SHORTCUT_HINT)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(hint)

    def _create_category_section(
        self, category: str, shortcuts: list[Shortcut]
    ) -> QWidget:
        """Create a section for a category of shortcuts."""
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Category header
        header = QLabel(category)
        if THEME_AVAILABLE:
            header.setFont(get_qfont(size=Sizes.SM, weight=Weights.SEMIBOLD))
            header.setStyleSheet(f"color: {Colors.PRIMARY};")
        else:
            font = QFont("Segoe UI", 9)
            font.setWeight(QFont.Weight.DemiBold)
            header.setFont(font)
            header.setStyleSheet(Styles.SHORTCUT_CATEGORY_HEADER)
        layout.addWidget(header)

        # Shortcuts grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)

        for i, shortcut in enumerate(shortcuts):
            # Shortcut badge
            badge = ShortcutBadge(shortcut.keys)
            grid.addWidget(badge, i, 0, Qt.AlignmentFlag.AlignLeft)

            # Description
            desc = QLabel(shortcut.description)
            if THEME_AVAILABLE:
                desc.setFont(get_qfont(size=Sizes.SM))
                desc.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
            else:
                desc.setFont(QFont("Segoe UI", 9))
                desc.setStyleSheet(Styles.SHORTCUT_DESC)
            grid.addWidget(desc, i, 1, Qt.AlignmentFlag.AlignLeft)

        layout.addLayout(grid)
        return section

    def paintEvent(self, event: Any) -> None:
        """Paint semi-transparent backdrop."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Semi-transparent dark backdrop
        painter.fillRect(self.rect(), QColor(0, 0, 0, 180))
        painter.end()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        """Handle key press - close on Escape."""
        if event is not None and event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event: Any) -> None:
        """Handle mouse press - close if clicking backdrop."""
        # Find the content frame
        content = self.findChild(QFrame, "shortcutsContent")
        if content and not content.geometry().contains(event.pos()):
            self.close()


__all__ = ["Shortcut", "ShortcutBadge", "ShortcutsOverlay", "DEFAULT_SHORTCUTS"]
