from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QToolButton
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QMouseEvent

try:
    from src.shared.python.theme.icon_utils import IconColorizer
except ImportError:
    IconColorizer = None  # Fallback


_WINDOW_CONTROL_BUTTON_STYLESHEET = """
    QToolButton { border: none; background: transparent; padding: 5px; color: #d4d4d4; font-weight: bold; }
    QToolButton:hover { background-color: rgba(255, 255, 255, 0.1); border-radius: 4px; }
"""


def create_window_control_button(
    icon_name: str,
    fallback_text: str,
    *,
    tooltip: str,
    accessible_name: str,
    object_name: str,
    color: str = "#d4d4d4",
    parent: QWidget | None = None,
) -> QToolButton:
    """Create a launcher-styled window control button."""
    button = QToolButton(parent)

    if IconColorizer:
        button.setIcon(IconColorizer.get_icon(icon_name, color))
    else:
        button.setText(fallback_text)

    button.setObjectName(object_name)
    button.setToolTip(tooltip)
    button.setAccessibleName(accessible_name)
    button.setStyleSheet(_WINDOW_CONTROL_BUTTON_STYLESHEET)
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

        # Use dark background for title bar to blend with main UI
        self.setStyleSheet(
            'QWidget[class="title-bar"] { background-color: #1e1e1e; border-bottom: 1px solid #3a3f4a; }'
        )

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

        self.title_label = QLabel("UpstreamDrift")
        self.title_label.setStyleSheet(
            "color: #d4d4d4; font-weight: bold; font-size: 13px;"
        )

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

    def _minimize_window(self):
        self.minimize_requested.emit()

    def _maximize_window(self):
        self.maximize_requested.emit()

    def _close_window(self):
        self.close_requested.emit()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = (
                event.globalPosition().toPoint()
                - self.window().frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.move_requested.emit(
                event.globalPosition().toPoint() - self.drag_position
            )
            event.accept()
