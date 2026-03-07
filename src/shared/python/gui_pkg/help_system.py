"""Help System Module for UpstreamDrift.

This module provides a reusable help system that can be integrated across
all UI components in the application. It includes:

- HelpDialog: A dialog for displaying markdown documentation
- HelpButton: A small help button widget that opens context-specific help
- TooltipManager: Manages context-sensitive tooltips for UI elements
- Functions to load and parse help content from markdown files

The help system supports:
- Rendering markdown documentation
- Topic-based help navigation
- Integration with the main USER_MANUAL.md
- Context-sensitive help based on selected UI elements
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QSize, Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass

# Get paths
REPOS_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DOCS_DIR = REPOS_ROOT / "docs"
HELP_DIR = DOCS_DIR / "help"
USER_MANUAL_PATH = DOCS_DIR / "USER_MANUAL.md"


def get_user_manual_content() -> str:
    """Load the main USER_MANUAL.md content.

    Returns:
        The content of USER_MANUAL.md, or an error message if not found.
    """
    if USER_MANUAL_PATH.exists():
        try:
            return USER_MANUAL_PATH.read_text(encoding="utf-8")
        except (RuntimeError, ValueError, OSError) as e:
            return f"# Error Loading Manual\n\nFailed to load USER_MANUAL.md: {e}"
    return "# User Manual Not Found\n\nThe USER_MANUAL.md file could not be found."


def get_help_topic_content(topic: str) -> str:
    """Load help content for a specific topic.

    Args:
        topic: The topic name (corresponds to a file in docs/help/).

    Returns:
        The markdown content for the topic, or a fallback message.
    """
    # Try topic-specific file first
    topic_file = HELP_DIR / f"{topic}.md"
    if topic_file.exists():
        try:
            content = topic_file.read_text(encoding="utf-8")
            # Add link back to main manual
            content += "\n\n---\n\n*See also: [Full User Manual](../USER_MANUAL.md)*"
            return content
        except (RuntimeError, ValueError, OSError) as e:
            return f"# Error Loading Topic\n\nFailed to load {topic}.md: {e}"

    # Fallback: try to extract section from USER_MANUAL.md
    manual_content = get_user_manual_content()
    section = _extract_section_from_manual(manual_content, topic)
    if section:
        return section

    return f"# Topic Not Found\n\nNo help content found for topic: {topic}\n\nPlease refer to the main User Manual."


def _extract_section_from_manual(content: str, topic: str) -> str | None:
    """Extract a relevant section from the USER_MANUAL.md based on topic.

    Args:
        content: The full manual content.
        topic: The topic to search for.

    Returns:
        The extracted section content, or None if not found.
    """
    # Map topics to section headers in the manual
    topic_mapping = {
        "engine_selection": "Physics Engines Guide",
        "simulation_controls": "Core Features",
        "motion_capture": "Motion Capture Integration",
        "visualization": "Visualization and Analysis",
        "analysis_tools": "Analysis",
    }

    search_term = topic_mapping.get(topic, topic.replace("_", " ").title())

    # Find the section in the manual
    pattern = rf"(##+ .*{re.escape(search_term)}.*?\n)(.*?)(?=\n##+ |\Z)"
    match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)

    if match:
        return match.group(1) + match.group(2)
    return None


def list_help_topics() -> list[tuple[str, str]]:
    """List all available help topics.

    Returns:
        A list of tuples (topic_id, topic_title) for all available help files.
    """
    topics = []

    if HELP_DIR.exists():
        for md_file in HELP_DIR.glob("*.md"):
            topic_id = md_file.stem
            # Read first line as title
            try:
                first_line = md_file.read_text(encoding="utf-8").split("\n")[0]
                title = first_line.lstrip("#").strip()
            except (RuntimeError, ValueError, OSError):
                title = topic_id.replace("_", " ").title()
            topics.append((topic_id, title))

    return sorted(topics, key=lambda x: x[1])


class HelpDialog(QDialog):
    """A comprehensive help dialog that displays markdown documentation.

    Features:
    - Sidebar with topic navigation
    - Main content area with markdown rendering
    - Search functionality
    - Links to external documentation
    - Back to USER_MANUAL.md link
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        initial_topic: str | None = None,
    ) -> None:
        """Initialize the HelpDialog.

        Args:
            parent: The parent widget.
            initial_topic: Optional topic to display initially.
        """
        super().__init__(parent)
        self.setWindowTitle("UpstreamDrift - Help")
        self.resize(1000, 700)
        self.setMinimumSize(800, 500)

        self._setup_ui()
        self._load_topics()
        self._setup_shortcuts()

        # Load initial content
        if initial_topic:
            self._load_topic(initial_topic)
        else:
            self._load_user_manual()

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Top toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)

        # Main content area with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Sidebar
        sidebar = self._create_sidebar()
        splitter.addWidget(sidebar)

        # Content area
        content_frame = self._create_content_area()
        splitter.addWidget(content_frame)

        # Set splitter proportions
        splitter.setSizes([250, 750])

        layout.addWidget(splitter, 1)

        # Bottom bar
        bottom_bar = self._create_bottom_bar()
        layout.addWidget(bottom_bar)

        # Apply styling
        self._apply_styles()

    def _create_toolbar(self) -> QFrame:
        """Create the top toolbar."""
        toolbar = QFrame()
        toolbar.setObjectName("HelpToolbar")
        toolbar.setFixedHeight(50)

        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(10, 5, 10, 5)

        # Home button
        btn_home = QPushButton("User Manual")
        btn_home.setToolTip("Go to main User Manual")
        btn_home.clicked.connect(self._load_user_manual)
        layout.addWidget(btn_home)

        layout.addStretch()

        # Search
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search help...")
        self.search_input.setFixedWidth(250)
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self._on_search_changed)
        layout.addWidget(self.search_input)

        layout.addStretch()

        # Topic selector
        layout.addWidget(QLabel("Topic:"))
        self.topic_combo = QComboBox()
        self.topic_combo.setMinimumWidth(200)
        self.topic_combo.currentIndexChanged.connect(self._on_topic_selected)
        layout.addWidget(self.topic_combo)

        return toolbar

    def _create_sidebar(self) -> QFrame:
        """Create the sidebar with topic list."""
        sidebar = QFrame()
        sidebar.setObjectName("HelpSidebar")
        sidebar.setMinimumWidth(200)
        sidebar.setMaximumWidth(350)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(5, 5, 5, 5)

        # Section header
        header = QLabel("Help Topics")
        header.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(header)

        # Topic list
        self.topic_list = QListWidget()
        self.topic_list.itemClicked.connect(self._on_list_item_clicked)
        layout.addWidget(self.topic_list)

        return sidebar

    def _create_content_area(self) -> QFrame:
        """Create the main content area."""
        frame = QFrame()
        frame.setObjectName("HelpContent")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)

        # Text browser for markdown content
        self.content_browser = QTextBrowser()
        self.content_browser.setOpenExternalLinks(True)
        self.content_browser.setReadOnly(True)
        layout.addWidget(self.content_browser)

        return frame

    def _create_bottom_bar(self) -> QFrame:
        """Create the bottom bar with close button."""
        bar = QFrame()
        bar.setObjectName("HelpBottomBar")
        bar.setFixedHeight(50)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 5, 10, 5)

        # Links
        lbl_links = QLabel(
            '<a href="https://github.com/dieterolson/UpstreamDrift">GitHub</a> | '
            '<a href="https://github.com/dieterolson/UpstreamDrift/issues">Report Issue</a>'
        )
        lbl_links.setOpenExternalLinks(True)
        layout.addWidget(lbl_links)

        # Report a Bug button
        btn_bug = QPushButton("Report a Bug")
        btn_bug.setStyleSheet(
            "background-color: #d32f2f; color: white; padding: 6px 14px;"
            " border: none; border-radius: 4px;"
        )
        btn_bug.setToolTip("Report a bug via email")
        btn_bug.clicked.connect(self._report_bug)
        layout.addWidget(btn_bug)

        layout.addStretch()

        # Close button
        btn_close = QPushButton("Close")
        btn_close.setFixedWidth(100)
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

        return bar

    def _setup_shortcuts(self) -> None:
        """Set up keyboard shortcuts."""
        # Escape to close
        QShortcut(QKeySequence("Escape"), self, self.close)

        # Ctrl+F to focus search
        QShortcut(QKeySequence("Ctrl+F"), self, self._focus_search)

    def _focus_search(self) -> None:
        """Focus the search input."""
        self.search_input.setFocus()
        self.search_input.selectAll()

    def _report_bug(self) -> None:
        """Open default mail client to report a bug."""
        from urllib.parse import quote

        subject = "Bug Report: Golf Modeling Suite"
        body = "Please describe the issue you encountered:\n\n"
        email = "support@golfmodelingsuite.com"
        mailto_url = f"mailto:{email}?subject={quote(subject)}&body={quote(body)}"
        QDesktopServices.openUrl(QUrl(mailto_url))

    def _apply_styles(self) -> None:
        """Apply CSS styles to the dialog."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1E1E1E;
            }
            #HelpToolbar {
                background-color: #252526;
                border-bottom: 1px solid #3E3E42;
            }
            #HelpSidebar {
                background-color: #252526;
                border-right: 1px solid #3E3E42;
            }
            #HelpContent {
                background-color: #1E1E1E;
            }
            #HelpBottomBar {
                background-color: #252526;
                border-top: 1px solid #3E3E42;
            }
            QLabel {
                color: #CCCCCC;
            }
            QLabel a {
                color: #0A84FF;
            }
            QPushButton {
                background-color: #0A84FF;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0077E6;
            }
            QPushButton:pressed {
                background-color: #0066CC;
            }
            QLineEdit {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: 1px solid #3E3E42;
                border-radius: 4px;
                padding: 6px;
            }
            QLineEdit:focus {
                border: 1px solid #0A84FF;
            }
            QComboBox {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: 1px solid #3E3E42;
                border-radius: 4px;
                padding: 6px;
            }
            QListWidget {
                background-color: #1E1E1E;
                color: #CCCCCC;
                border: none;
                outline: none;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #094771;
            }
            QListWidget::item:hover:!selected {
                background-color: #2A2D2E;
            }
            QTextBrowser {
                background-color: #1E1E1E;
                color: #CCCCCC;
                border: none;
                padding: 20px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
            }
        """)

    def _load_topics(self) -> None:
        """Load available topics into the UI."""
        # Clear existing
        self.topic_list.clear()
        self.topic_combo.clear()

        # Add User Manual as first item
        item = QListWidgetItem("User Manual (Full)")
        item.setData(Qt.ItemDataRole.UserRole, "__user_manual__")
        self.topic_list.addItem(item)
        self.topic_combo.addItem("User Manual", "__user_manual__")

        # Add separator in combo
        self.topic_combo.insertSeparator(1)

        # Add help topics
        topics = list_help_topics()
        for topic_id, title in topics:
            # List widget
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, topic_id)
            self.topic_list.addItem(item)

            # Combo box
            self.topic_combo.addItem(title, topic_id)

    def _load_user_manual(self) -> None:
        """Load the main USER_MANUAL.md."""
        content = get_user_manual_content()
        self.content_browser.setMarkdown(content)
        self._select_topic_in_list("__user_manual__")

    def _load_topic(self, topic: str) -> None:
        """Load content for a specific topic.

        Args:
            topic: The topic identifier.
        """
        if topic == "__user_manual__":
            self._load_user_manual()
            return

        content = get_help_topic_content(topic)
        self.content_browser.setMarkdown(content)
        self._select_topic_in_list(topic)

    def _select_topic_in_list(self, topic: str) -> None:
        """Select a topic in the sidebar list.

        Args:
            topic: The topic identifier.
        """
        for i in range(self.topic_list.count()):
            item = self.topic_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == topic:
                self.topic_list.setCurrentItem(item)
                break

    def _on_list_item_clicked(self, item: QListWidgetItem) -> None:
        """Handle topic list item click.

        Args:
            item: The clicked list item.
        """
        topic = item.data(Qt.ItemDataRole.UserRole)
        if topic:
            self._load_topic(topic)

    def _on_topic_selected(self, index: int) -> None:
        """Handle topic combo box selection.

        Args:
            index: The selected index.
        """
        topic = self.topic_combo.itemData(index)
        if topic:
            self._load_topic(topic)

    def _on_search_changed(self, text: str) -> None:
        """Handle search text changes.

        Args:
            text: The search text.
        """
        # Filter topic list
        search_lower = text.lower()
        for i in range(self.topic_list.count()):
            item = self.topic_list.item(i)
            if item:
                item.setHidden(
                    search_lower not in item.text().lower() and bool(search_lower)
                )

        # Search within content
        if text:
            self.content_browser.find(text)


class HelpButton(QToolButton):
    """A small help button that opens topic-specific help.

    This button can be placed next to any UI element to provide
    context-sensitive help.
    """

    def __init__(
        self,
        topic: str,
        tooltip: str = "Click for help",
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the HelpButton.

        Args:
            topic: The help topic to display when clicked.
            tooltip: The tooltip text.
            parent: The parent widget.
        """
        super().__init__(parent)
        self.topic = topic

        # Setup appearance
        self.setText("?")
        self.setToolTip(tooltip)
        self.setFixedSize(QSize(20, 20))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Style
        self.setStyleSheet("""
            QToolButton {
                background-color: #3C3C3C;
                color: #0A84FF;
                border: 1px solid #555555;
                border-radius: 10px;
                font-weight: bold;
                font-size: 12px;
            }
            QToolButton:hover {
                background-color: #0A84FF;
                color: white;
                border: 1px solid #0A84FF;
            }
        """)

        # Connect click
        self.clicked.connect(self._on_clicked)

    def _on_clicked(self) -> None:
        """Handle button click to open help dialog."""
        dialog = HelpDialog(self.window(), initial_topic=self.topic)
        dialog.exec()


class TooltipManager:
    """Manages context-sensitive tooltips for UI elements.

    This class provides enhanced tooltips that can include:
    - Rich text formatting
    - Links to help topics
    - Delayed display
    """

    # Default delay before showing tooltip (ms)
    DEFAULT_DELAY_MS = 500

    # Mapping of widget names to help content
    _tooltip_content: dict[int, dict[str, Any]] = {}

    @classmethod
    def register_tooltip(
        cls,
        widget: QWidget,
        short_text: str,
        long_text: str | None = None,
        help_topic: str | None = None,
    ) -> None:
        """Register a tooltip for a widget.

        Args:
            widget: The widget to register.
            short_text: The short tooltip text (shown on hover).
            long_text: Optional longer description.
            help_topic: Optional help topic for "more info" link.
        """
        # Build tooltip HTML
        tooltip_html = f"<b>{short_text}</b>"
        if long_text:
            tooltip_html += f"<br/><br/>{long_text}"
        if help_topic:
            tooltip_html += "<br/><br/><i>Press F1 for more help</i>"

        widget.setToolTip(tooltip_html)

        # Store for later reference
        widget_id = id(widget)
        cls._tooltip_content[widget_id] = {
            "short": short_text,
            "long": long_text,
            "topic": help_topic,
        }

    @classmethod
    def get_help_topic(cls, widget: QWidget) -> str | None:
        """Get the help topic associated with a widget.

        Args:
            widget: The widget to look up.

        Returns:
            The help topic, or None if not registered.
        """
        widget_id = id(widget)
        content = cls._tooltip_content.get(widget_id)
        return content.get("topic") if content else None

    @classmethod
    def show_extended_tooltip(
        cls,
        widget: QWidget,
        position: Any = None,
    ) -> None:
        """Show an extended tooltip for a widget.

        Args:
            widget: The widget.
            position: Optional position to show tooltip.
        """
        widget_id = id(widget)
        content = cls._tooltip_content.get(widget_id)

        if not content:
            return

        # Build extended content
        html = f"<b>{content['short']}</b>"
        if content.get("long"):
            html += f"<br/><br/>{content['long']}"
        if content.get("topic"):
            html += f"<br/><br/><i>Topic: {content['topic']}</i>"

        # Show tooltip
        pos = position or widget.mapToGlobal(widget.rect().bottomLeft())
        QToolTip.showText(pos, html, widget)


def create_help_menu_actions(
    parent: QWidget,
    open_manual_callback: Callable[[], None] | None = None,
    open_about_callback: Callable[[], None] | None = None,
) -> list[tuple[str, str, Callable[[], None]]]:
    """Create standard help menu actions.

    Args:
        parent: The parent widget (main window).
        open_manual_callback: Optional callback for "User Manual" action.
        open_about_callback: Optional callback for "About" action.

    Returns:
        A list of tuples (name, shortcut, callback) for menu actions.
    """

    def default_open_manual() -> None:
        """Open the help dialog as a user manual."""
        dialog = HelpDialog(parent)
        dialog.exec()

    def default_open_about() -> None:
        """Show the About dialog with version and project information."""
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.about(
            parent,
            "About UpstreamDrift",
            "<h2>UpstreamDrift</h2>"
            "<p>Version 2.1</p>"
            "<p>Biomechanical Golf Swing Analysis Platform</p>"
            "<p>Copyright 2024-2026 UpstreamDrift Contributors</p>"
            '<p><a href="https://github.com/dieterolson/UpstreamDrift">GitHub</a></p>',
        )

    actions = [
        ("User Manual", "F1", open_manual_callback or default_open_manual),
        ("About UpstreamDrift", "", open_about_callback or default_open_about),
    ]

    return actions


def add_help_button_to_widget(
    layout: QHBoxLayout | QVBoxLayout,
    topic: str,
    tooltip: str = "Click for help",
) -> HelpButton:
    """Convenience function to add a help button to a layout.

    Args:
        layout: The layout to add the button to.
        topic: The help topic.
        tooltip: The tooltip text.

    Returns:
        The created HelpButton.
    """
    button = HelpButton(topic, tooltip)
    layout.addWidget(button)
    return button
