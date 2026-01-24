"""GUI utilities for eliminating duplication in PyQt6 applications.

This module provides reusable GUI patterns to eliminate repeated code
across launcher and dashboard implementations.

Usage:
    from src.shared.python.gui_utils import (
        get_qapp,
        BaseApplicationWindow,
        create_dialog,
        setup_window_geometry,
    )

    # Get or create QApplication instance
    app = get_qapp()

    # Create window with standard setup
    window = BaseApplicationWindow("My App", (1400, 900))
    window.show()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_config import get_logger
from src.shared.python.path_utils import get_repo_root

logger = get_logger(__name__)


def get_qapp(args: list[str] | None = None) -> QApplication:
    """Get or create QApplication instance (singleton pattern).

    Args:
        args: Command line arguments for QApplication

    Returns:
        QApplication instance

    Example:
        app = get_qapp()
        window = QMainWindow()
        window.show()
        app.exec()
    """
    app = get_qapp()
    logger.debug("Created new QApplication instance")
    return app


def get_default_icon() -> QIcon | None:
    """Get default application icon.

    Returns:
        QIcon if icon file exists, None otherwise
    """
    icon_paths = [
        get_repo_root() / "launchers" / "assets" / "golf_robot.png",
        get_repo_root() / "src" / "launchers" / "assets" / "golf_robot.png",
        get_repo_root() / "assets" / "golf_robot.png",
    ]

    for icon_path in icon_paths:
        if icon_path.exists():
            return QIcon(str(icon_path))

    logger.warning("Default icon not found")
    return None


def setup_window_geometry(
    window: QWidget,
    size: tuple[int, int] = (1400, 900),
    center: bool = True,
) -> None:
    """Setup window geometry with standard defaults.

    Args:
        window: Window widget to configure
        size: (width, height) tuple
        center: Whether to center window on screen
    """
    window.resize(*size)

    if center:
        # Center window on screen
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.geometry()
            x = (screen_geometry.width() - size[0]) // 2
            y = (screen_geometry.height() - size[1]) // 2
            window.move(x, y)


class BaseApplicationWindow(QMainWindow):
    """Base class for application windows with standard setup.

    Provides:
    - Standard window geometry
    - Icon setup
    - Title configuration
    - Status bar
    - Common styling

    Example:
        class MyWindow(BaseApplicationWindow):
            def __init__(self):
                super().__init__("My Application", (1200, 800))
                self.setup_ui()

            def setup_ui(self):
                # Add your widgets here
                pass
    """

    def __init__(
        self,
        title: str,
        size: tuple[int, int] = (1400, 900),
        icon_path: Path | None = None,
    ):
        """Initialize base application window.

        Args:
            title: Window title
            size: (width, height) tuple
            icon_path: Optional custom icon path
        """
        super().__init__()

        # Set title
        self.setWindowTitle(title)

        # Set geometry
        setup_window_geometry(self, size, center=True)

        # Set icon
        if icon_path and icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            icon = get_default_icon()
            if icon:
                self.setWindowIcon(icon)

        # Create status bar
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage("Ready")

        logger.debug(f"Initialized {self.__class__.__name__}: {title}")

    def show_status(self, message: str, timeout: int = 0) -> None:
        """Show message in status bar.

        Args:
            message: Message to display
            timeout: Timeout in milliseconds (0 = no timeout)
        """
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(message, timeout)

    def show_error(self, title: str, message: str) -> None:
        """Show error dialog.

        Args:
            title: Dialog title
            message: Error message
        """
        QMessageBox.critical(self, title, message)
        logger.error(f"{title}: {message}")

    def show_warning(self, title: str, message: str) -> None:
        """Show warning dialog.

        Args:
            title: Dialog title
            message: Warning message
        """
        QMessageBox.warning(self, title, message)
        logger.warning(f"{title}: {message}")

    def show_info(self, title: str, message: str) -> None:
        """Show information dialog.

        Args:
            title: Dialog title
            message: Information message
        """
        QMessageBox.information(self, title, message)
        logger.info(f"{title}: {message}")

    def confirm(self, title: str, message: str) -> bool:
        """Show confirmation dialog.

        Args:
            title: Dialog title
            message: Confirmation message

        Returns:
            True if user confirmed, False otherwise
        """
        reply = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes


def create_dialog(
    parent: QWidget | None,
    title: str,
    message: str,
    buttons: QDialogButtonBox.StandardButton = QDialogButtonBox.StandardButton.Ok,
) -> QDialog:
    """Create a standard dialog with message and buttons.

    Args:
        parent: Parent widget
        title: Dialog title
        message: Dialog message
        buttons: Standard buttons to include

    Returns:
        Configured QDialog

    Example:
        dialog = create_dialog(
            parent=self,
            title="Confirm Action",
            message="Are you sure?",
            buttons=QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # User clicked Yes
            pass
    """
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)

    layout = QVBoxLayout(dialog)

    # Add message label
    label = QLabel(message)
    label.setWordWrap(True)
    layout.addWidget(label)

    # Add buttons
    button_box = QDialogButtonBox(buttons)
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addWidget(button_box)

    return dialog


def create_button(
    text: str,
    callback: Any | None = None,
    tooltip: str | None = None,
    icon: QIcon | None = None,
    size: tuple[int, int] | None = None,
) -> QPushButton:
    """Create a configured button.

    Args:
        text: Button text
        callback: Click callback function
        tooltip: Tooltip text
        icon: Button icon
        size: (width, height) tuple

    Returns:
        Configured QPushButton

    Example:
        button = create_button(
            "Load Model",
            callback=self.load_model,
            tooltip="Load a physics model"
        )
    """
    button = QPushButton(text)

    if callback:
        button.clicked.connect(callback)

    if tooltip:
        button.setToolTip(tooltip)

    if icon:
        button.setIcon(icon)

    if size:
        button.setFixedSize(QSize(*size))

    return button


def create_label(
    text: str,
    bold: bool = False,
    alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft,
) -> QLabel:
    """Create a configured label.

    Args:
        text: Label text
        bold: Whether to make text bold
        alignment: Text alignment

    Returns:
        Configured QLabel

    Example:
        label = create_label("Model Name:", bold=True)
    """
    label = QLabel(text)
    label.setAlignment(alignment)

    if bold:
        font = label.font()
        font.setBold(True)
        label.setFont(font)

    return label


class LayoutBuilder:
    """Builder for creating layouts with fluent interface.

    Example:
        layout = (LayoutBuilder(QVBoxLayout())
                  .add_widget(QLabel("Title"))
                  .add_stretch()
                  .add_widget(QPushButton("OK"))
                  .build())
    """

    def __init__(self, layout: Any):
        """Initialize layout builder.

        Args:
            layout: Layout to build (QVBoxLayout, QHBoxLayout, etc.)
        """
        self.layout = layout

    def add_widget(self, widget: QWidget, stretch: int = 0) -> LayoutBuilder:
        """Add widget to layout.

        Args:
            widget: Widget to add
            stretch: Stretch factor

        Returns:
            Self for chaining
        """
        self.layout.addWidget(widget, stretch)
        return self

    def add_layout(self, layout: Any, stretch: int = 0) -> LayoutBuilder:
        """Add sub-layout to layout.

        Args:
            layout: Layout to add
            stretch: Stretch factor

        Returns:
            Self for chaining
        """
        self.layout.addLayout(layout, stretch)
        return self

    def add_stretch(self, stretch: int = 1) -> LayoutBuilder:
        """Add stretch to layout.

        Args:
            stretch: Stretch factor

        Returns:
            Self for chaining
        """
        self.layout.addStretch(stretch)
        return self

    def add_spacing(self, spacing: int) -> LayoutBuilder:
        """Add spacing to layout.

        Args:
            spacing: Spacing in pixels

        Returns:
            Self for chaining
        """
        self.layout.addSpacing(spacing)
        return self

    def set_margins(
        self, left: int, top: int, right: int, bottom: int
    ) -> LayoutBuilder:
        """Set layout margins.

        Args:
            left: Left margin
            top: Top margin
            right: Right margin
            bottom: Bottom margin

        Returns:
            Self for chaining
        """
        self.layout.setContentsMargins(left, top, right, bottom)
        return self

    def set_spacing(self, spacing: int) -> LayoutBuilder:
        """Set layout spacing.

        Args:
            spacing: Spacing between widgets

        Returns:
            Self for chaining
        """
        self.layout.setSpacing(spacing)
        return self

    def build(self) -> Any:
        """Build and return the layout.

        Returns:
            Configured layout
        """
        return self.layout


def apply_stylesheet(widget: QWidget, stylesheet: str) -> None:
    """Apply stylesheet to widget with error handling.

    Args:
        widget: Widget to style
        stylesheet: CSS stylesheet string
    """
    try:
        widget.setStyleSheet(stylesheet)
    except Exception as e:
        logger.warning(f"Failed to apply stylesheet: {e}")
