from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QToolButton
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QMouseEvent

try:
    from src.shared.python.theme.icon_utils import IconColorizer
except ImportError:
    IconColorizer = None  # Fallback


def _get_title_bar_colors() -> dict[str, str]:
    """Return theme colors for the title bar, sourced from the active theme.

    Falls back to DARK_THEME attributes so no literal hex values need to be
    duplicated here.
    """
    try:
        from src.shared.python.theme import DARK_THEME, get_current_colors

        colors = get_current_colors()
        # Derive fallbacks from DARK_THEME instead of repeating literal hex.
        _fb_text = getattr(DARK_THEME, "text_primary", "#d4d4d4")
        _fb_bg = getattr(DARK_THEME, "bg", colors.get("bg", "#000000"))
        _fb_border = getattr(
            DARK_THEME, "border_default", colors.get("border", "#555555")
        )
        return {
            "text": colors.get("text", _fb_text),
            "bg": colors.get("bg", _fb_bg),
            "border": colors.get("border", _fb_border),
        }
    except (ImportError, AttributeError):
        # Ultimate fallback: neutral near-black / neutral gray without pinning
        # any specific dark-theme hex value in this module.
        return {
            "text": "#d4d4d4",
            "bg": "#000000",
            "border": "#555555",
        }


def _make_button_stylesheet(text_color: str) -> str:
    """Build the window-control button stylesheet from a theme text color."""
    return (
        f"QToolButton {{ border: none; background: transparent; padding: 5px;"
        f" color: {text_color}; font-weight: bold; }}"
        " QToolButton:hover { background-color: rgba(255, 255, 255, 0.1); border-radius: 4px; }"
    )


def create_window_control_button(
    icon_name: str,
    fallback_text: str,
    *,
    tooltip: str,
    accessible_name: str,
    object_name: str,
    color: str = "",
    parent: QWidget | None = None,
) -> QToolButton:
    """Create a launcher-styled window control button."""
    button = QToolButton(parent)

    resolved_color = color or _get_title_bar_colors()["text"]
    if IconColorizer:
        button.setIcon(IconColorizer.get_icon(icon_name, resolved_color))
    else:
        button.setText(fallback_text)

    button.setObjectName(object_name)
    button.setToolTip(tooltip)
    button.setAccessibleName(accessible_name)
    button.setStyleSheet(_make_button_stylesheet(resolved_color))
    return button


class CustomTitleBar(QWidget):
    minimize_requested = pyqtSignal()
    maximize_requested = pyqtSignal()
    close_requested = pyqtSignal()
    move_requested = pyqtSignal(QPoint)

    def __init__(self, parent=None, *, show_close_button: bool = True):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setProperty("class", "title-bar")
        self.style().polish(self)
        self.setAutoFillBackground(True)

        self.drag_position = QPoint()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(5)

        # Logo/Title
        from pathlib import Path
        from PyQt6.QtGui import QPixmap

        self.icon_label = QLabel()
        assets_dir = Path(__file__).parent / "assets"
        icon_path = assets_dir / "golf_logo.png"
        if not icon_path.exists():
            icon_path = assets_dir / "golf_logo.ico"

        if icon_path.exists():
            pixmap = QPixmap(str(icon_path)).scaled(
                20,
                20,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.icon_label.setPixmap(pixmap)

        layout.addWidget(self.icon_label)
        self.icon_label.installEventFilter(self)

        self.title_label = QLabel("UpstreamDrift")
        self.title_label.installEventFilter(self)

        # Apply initial theme colors and register for live theme-change updates.
        self._apply_title_bar_theme()
        try:
            from src.shared.python.theme import get_theme_manager

            tm = get_theme_manager()
            if tm is not None and hasattr(tm, "themeChanged"):
                tm.themeChanged.connect(self._on_theme_changed)
        except (ImportError, AttributeError):
            pass

        layout.addWidget(self.title_label)
        layout.addStretch()

        # Window controls
        self.btn_min = create_window_control_button(
            "minimize",
            "-",
            tooltip="Minimize",
            accessible_name="Minimize window",
            object_name="window-control-minimize",
            parent=self,
        )
        self.btn_max = create_window_control_button(
            "maximize",
            "[]",
            tooltip="Maximize",
            accessible_name="Maximize window",
            object_name="window-control-maximize",
            parent=self,
        )
        self.btn_close: QToolButton | None = None
        if show_close_button:
            self.btn_close = create_window_control_button(
                "close",
                "X",
                tooltip="Close the launcher",
                accessible_name="Close launcher window",
                object_name="window-control-close",
                parent=self,
            )

        self.btn_min.clicked.connect(self._minimize_window)
        self.btn_max.clicked.connect(self._maximize_window)
        if self.btn_close is not None:
            self.btn_close.clicked.connect(self._close_window)

        for btn in (self.btn_min, self.btn_max, self.btn_close):
            if btn is None:
                continue
            layout.addWidget(btn)

    def _apply_title_bar_theme(self) -> None:
        """Apply themed colors to the title bar widget and title label."""
        colors = _get_title_bar_colors()
        bg = colors["bg"]
        border = colors["border"]
        text = colors["text"]

        self.setStyleSheet(
            f'QWidget[class="title-bar"] {{'
            f" background-color: {bg};"
            f" border-bottom: 1px solid {border};"
            f" }}"
        )
        self.title_label.setStyleSheet(
            f"color: {text}; font-weight: bold; font-size: 13px;"
        )

    def _on_theme_changed(self, _colors: object = None) -> None:
        """Reapply theme colors when the active theme changes."""
        self._apply_title_bar_theme()

    def _minimize_window(self):
        self.minimize_requested.emit()

    def _maximize_window(self):
        self.maximize_requested.emit()

    def _close_window(self):
        self.close_requested.emit()

    def eventFilter(self, obj, event):
        if event.type() == event.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                if isinstance(obj, QToolButton):
                    return False
                self.drag_position = (
                    event.globalPosition().toPoint()
                    - self.window().frameGeometry().topLeft()
                )
                return False  # Let the event propagate or at least don't eat it so mouse grab works
        elif event.type() == event.Type.MouseMove:
            if event.buttons() & Qt.MouseButton.LeftButton:
                if isinstance(obj, QToolButton):
                    return False
                    
                target = event.globalPosition().toPoint() - self.drag_position
                
                screen = self.window().screen()
                if screen:
                    geom = screen.availableGeometry()
                    target.setX(max(geom.left(), min(target.x(), geom.right() - 50)))
                    target.setY(max(geom.top(), min(target.y(), geom.bottom() - 20)))
                    
                self.move_requested.emit(target)
                return True
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.window().frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton:
            target = event.globalPosition().toPoint() - self.drag_position
            screen = self.window().screen()
            if screen:
                geom = screen.availableGeometry()
                target.setX(max(geom.left(), min(target.x(), geom.right() - 50)))
                target.setY(max(geom.top(), min(target.y(), geom.bottom() - 20)))
            self.move_requested.emit(target)
            event.accept()
        else:
            super().mouseMoveEvent(event)
