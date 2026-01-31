"""URDF Code Editor with XML syntax highlighting and validation.

Provides a code editor experience for viewing and editing URDF/XML files
with syntax highlighting, line numbers, validation, and auto-completion.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

from PyQt6.QtCore import QRect, QRegularExpression, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextDocument,
)
from PyQt6.QtWidgets import (
    QCompleter,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


class XMLHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for XML/URDF content."""

    def __init__(self, parent: QTextDocument | None = None) -> None:
        """Initialize the highlighter."""
        super().__init__(parent)
        self._setup_highlighting_rules()

    def _setup_highlighting_rules(self) -> None:
        """Set up highlighting rules for XML."""
        self.highlighting_rules: list[tuple[QRegularExpression, QTextCharFormat]] = []

        # Tag names (blue)
        tag_format = QTextCharFormat()
        tag_format.setForeground(QColor("#0000FF"))
        tag_format.setFontWeight(QFont.Weight.Bold)
        self.highlighting_rules.append((QRegularExpression(r"</?[\w:-]+"), tag_format))

        # Attribute names (dark cyan)
        attr_format = QTextCharFormat()
        attr_format.setForeground(QColor("#008B8B"))
        self.highlighting_rules.append(
            (QRegularExpression(r'\b[\w:-]+(?==")'), attr_format)
        )

        # Attribute values (dark red/maroon)
        value_format = QTextCharFormat()
        value_format.setForeground(QColor("#8B0000"))
        self.highlighting_rules.append((QRegularExpression(r'"[^"]*"'), value_format))

        # Comments (gray)
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#808080"))
        comment_format.setFontItalic(True)
        self.highlighting_rules.append(
            (QRegularExpression(r"<!--.*?-->"), comment_format)
        )

        # XML declaration (purple)
        decl_format = QTextCharFormat()
        decl_format.setForeground(QColor("#800080"))
        self.highlighting_rules.append(
            (QRegularExpression(r"<\?xml.*?\?>"), decl_format)
        )

        # URDF-specific keywords (green)
        urdf_format = QTextCharFormat()
        urdf_format.setForeground(QColor("#006400"))
        urdf_format.setFontWeight(QFont.Weight.Bold)
        urdf_keywords = [
            "robot",
            "link",
            "joint",
            "visual",
            "collision",
            "inertial",
            "geometry",
            "origin",
            "mass",
            "inertia",
            "material",
            "color",
            "parent",
            "child",
            "axis",
            "limit",
            "dynamics",
            "transmission",
            "box",
            "cylinder",
            "sphere",
            "mesh",
            "capsule",
        ]
        for keyword in urdf_keywords:
            self.highlighting_rules.append(
                (QRegularExpression(rf"</?{keyword}\b"), urdf_format)
            )

        # Numbers (orange)
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#FF8C00"))
        self.highlighting_rules.append(
            (QRegularExpression(r'"[\d\s\.\-e+]+(?=")|[\d\.]+'), number_format)
        )

    def highlightBlock(self, text: str | None) -> None:
        """Apply highlighting rules to a block of text."""
        if text is None:
            return
        for pattern, fmt in self.highlighting_rules:
            match_iterator = pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)


class LineNumberArea(QWidget):
    """Widget displaying line numbers for the code editor."""

    def __init__(self, editor: URDFCodeEditor) -> None:
        """Initialize the line number area."""
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self) -> QSize:
        """Return the recommended size."""
        return QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, event: Any) -> None:
        """Paint the line numbers."""
        self.editor.line_number_area_paint_event(event)


class URDFCodeEditor(QPlainTextEdit):
    """Code editor for URDF/XML with syntax highlighting and line numbers."""

    content_changed = pyqtSignal(str)  # Emitted when content changes
    validation_result = pyqtSignal(bool, list)  # (is_valid, errors)
    cursor_position_changed = pyqtSignal(int, int)  # (line, column)

    # URDF tag completions
    URDF_COMPLETIONS = [
        "robot",
        "link",
        "joint",
        "visual",
        "collision",
        "inertial",
        "geometry",
        "origin",
        "mass",
        "inertia",
        "material",
        "color",
        "parent",
        "child",
        "axis",
        "limit",
        "dynamics",
        "transmission",
        "box",
        "cylinder",
        "sphere",
        "mesh",
        "capsule",
        "name",
        "type",
        "xyz",
        "rpy",
        "value",
        "filename",
        "scale",
        "rgba",
        "ixx",
        "iyy",
        "izz",
        "ixy",
        "ixz",
        "iyz",
        "lower",
        "upper",
        "effort",
        "velocity",
        "fixed",
        "revolute",
        "prismatic",
        "continuous",
        "floating",
        "planar",
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the code editor."""
        super().__init__(parent)
        self._setup_editor()
        self._setup_highlighter()
        self._setup_line_numbers()
        self._setup_completer()
        self._connect_signals()

    def _setup_editor(self) -> None:
        """Set up editor appearance."""
        # Set monospace font
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)

        # Set tab width to 4 spaces
        metrics = QFontMetrics(font)
        self.setTabStopDistance(4 * metrics.horizontalAdvance(" "))

        # Enable word wrap
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        # Set placeholder
        self.setPlaceholderText("Enter URDF/XML content here...")

    def _setup_highlighter(self) -> None:
        """Set up syntax highlighter."""
        self.highlighter = XMLHighlighter(self.document())

    def _setup_line_numbers(self) -> None:
        """Set up line number area."""
        self.line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.update_line_number_area_width(0)

    def _setup_completer(self) -> None:
        """Set up auto-completion."""
        self.completer = QCompleter(self.URDF_COMPLETIONS, self)
        self.completer.setWidget(self)
        self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.activated.connect(self.insert_completion)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self.textChanged.connect(self._on_text_changed)
        self.cursorPositionChanged.connect(self._on_cursor_position_changed)

    def _on_text_changed(self) -> None:
        """Handle text changes."""
        content = self.toPlainText()
        self.content_changed.emit(content)

    def _on_cursor_position_changed(self) -> None:
        """Handle cursor position changes."""
        cursor = self.textCursor()
        line = cursor.blockNumber() + 1
        column = cursor.columnNumber() + 1
        self.cursor_position_changed.emit(line, column)

    def line_number_area_width(self) -> int:
        """Calculate the width needed for line numbers."""
        digits = 1
        max_num = max(1, self.blockCount())
        while max_num >= 10:
            max_num //= 10
            digits += 1

        space = 10 + self.fontMetrics().horizontalAdvance("9") * digits
        return space

    def update_line_number_area_width(self, _: int) -> None:
        """Update the margins to account for line number area."""
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect: QRect, dy: int) -> None:
        """Update the line number area when scrolling."""
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(
                0, rect.y(), self.line_number_area.width(), rect.height()
            )

        viewport = self.viewport()
        if viewport and rect.contains(viewport.rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event: Any) -> None:
        """Handle resize events."""
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(
            QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height())
        )

    def line_number_area_paint_event(self, event: Any) -> None:
        """Paint line numbers."""
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor("#F0F0F0"))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = round(
            self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        )
        bottom = top + round(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QColor("#808080"))
                painter.drawText(
                    0,
                    top,
                    self.line_number_area.width() - 5,
                    self.fontMetrics().height(),
                    Qt.AlignmentFlag.AlignRight,
                    number,
                )

            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            block_number += 1

    def keyPressEvent(self, event: Any) -> None:
        """Handle key press events for auto-completion."""
        popup = self.completer.popup()
        if popup and popup.isVisible():
            if event.key() in (
                Qt.Key.Key_Enter,
                Qt.Key.Key_Return,
                Qt.Key.Key_Escape,
                Qt.Key.Key_Tab,
                Qt.Key.Key_Backtab,
            ):
                event.ignore()
                return

        super().keyPressEvent(event)

        # Trigger completion on typing
        completion_prefix = self._get_completion_prefix()
        popup = self.completer.popup()
        model = self.completer.completionModel()
        if len(completion_prefix) >= 2:
            if completion_prefix != self.completer.completionPrefix():
                self.completer.setCompletionPrefix(completion_prefix)
                if popup and model:
                    popup.setCurrentIndex(model.index(0, 0))

            cr = self.cursorRect()
            if popup:
                scrollbar = popup.verticalScrollBar()
                scrollbar_width = scrollbar.sizeHint().width() if scrollbar else 0
                cr.setWidth(popup.sizeHintForColumn(0) + scrollbar_width)
            self.completer.complete(cr)
        elif popup:
            popup.hide()

    def _get_completion_prefix(self) -> str:
        """Get the current word being typed for completion."""
        cursor = self.textCursor()
        cursor.select(cursor.SelectionType.WordUnderCursor)
        return cursor.selectedText()

    def insert_completion(self, completion: str) -> None:
        """Insert the selected completion."""
        cursor = self.textCursor()
        extra = len(completion) - len(self.completer.completionPrefix())
        cursor.movePosition(cursor.MoveOperation.Left)
        cursor.movePosition(cursor.MoveOperation.EndOfWord)
        cursor.insertText(completion[-extra:])
        self.setTextCursor(cursor)

    def validate_xml(self) -> tuple[bool, list[str]]:
        """Validate the XML content.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        content = self.toPlainText()
        if not content.strip():
            return True, []

        errors = []
        try:
            ET.fromstring(content)
            self.validation_result.emit(True, [])
            return True, []
        except ET.ParseError as e:
            error_msg = str(e)
            # Extract line and column info
            match = re.search(r"line (\d+), column (\d+)", error_msg)
            if match:
                line, col = match.groups()
                errors.append(f"XML Error at line {line}, column {col}: {error_msg}")
            else:
                errors.append(f"XML Error: {error_msg}")

            self.validation_result.emit(False, errors)
            return False, errors

    def validate_urdf(self) -> tuple[bool, list[str]]:
        """Validate URDF-specific structure.

        Returns:
            Tuple of (is_valid, list of error/warning messages)
        """
        content = self.toPlainText()
        if not content.strip():
            return True, []

        errors: list[str] = []

        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            # XML parsing errors handled by validate_xml
            return False, ["Invalid XML - cannot validate URDF structure"]

        # Check for robot root element
        if root.tag != "robot":
            errors.append("Root element must be 'robot'")

        # Check for robot name
        if not root.get("name"):
            errors.append("Robot element should have a 'name' attribute")

        # Collect links and joints
        links = {link.get("name") for link in root.findall("link")}
        joints = root.findall("joint")

        # Check for at least one link
        if not links:
            errors.append("URDF must have at least one link")

        # Validate joints
        for joint in joints:
            joint_name = joint.get("name", "unnamed")
            joint_type = joint.get("type")

            # Check joint type
            valid_types = [
                "fixed",
                "revolute",
                "prismatic",
                "continuous",
                "floating",
                "planar",
            ]
            if joint_type not in valid_types:
                errors.append(
                    f"Joint '{joint_name}' has invalid type '{joint_type}'. "
                    f"Must be one of: {', '.join(valid_types)}"
                )

            # Check parent/child links exist
            parent = joint.find("parent")
            child = joint.find("child")

            if parent is not None:
                parent_link = parent.get("link")
                if parent_link and parent_link not in links:
                    errors.append(
                        f"Joint '{joint_name}' references non-existent parent link '{parent_link}'"
                    )

            if child is not None:
                child_link = child.get("link")
                if child_link and child_link not in links:
                    errors.append(
                        f"Joint '{joint_name}' references non-existent child link '{child_link}'"
                    )

            # Check limits for revolute/prismatic
            if joint_type in ["revolute", "prismatic"]:
                limit = joint.find("limit")
                if limit is None:
                    errors.append(
                        f"Joint '{joint_name}' ({joint_type}) must have limits"
                    )

        is_valid = len(errors) == 0
        self.validation_result.emit(is_valid, errors)
        return is_valid, errors

    def set_content(self, content: str) -> None:
        """Set the editor content."""
        self.setPlainText(content)

    def get_content(self) -> str:
        """Get the editor content."""
        return self.toPlainText()

    def go_to_line(self, line: int) -> None:
        """Move cursor to a specific line."""
        doc = self.document()
        if doc is None:
            return
        block = doc.findBlockByLineNumber(line - 1)
        cursor = self.textCursor()
        cursor.setPosition(block.position())
        self.setTextCursor(cursor)
        self.centerCursor()

    def find_text(self, text: str, case_sensitive: bool = False) -> bool:
        """Find text in the editor.

        Args:
            text: Text to find
            case_sensitive: Whether to match case

        Returns:
            True if found
        """
        flags = QTextDocument.FindFlag(0)
        if case_sensitive:
            flags |= QTextDocument.FindFlag.FindCaseSensitively

        return self.find(text, flags)

    def replace_text(
        self, find: str, replace: str, all_occurrences: bool = False
    ) -> int:
        """Replace text in the editor.

        Args:
            find: Text to find
            replace: Replacement text
            all_occurrences: Replace all if True

        Returns:
            Number of replacements made
        """
        count = 0
        content = self.toPlainText()

        if all_occurrences:
            new_content = content.replace(find, replace)
            count = content.count(find)
            if count > 0:
                self.setPlainText(new_content)
        else:
            cursor = self.textCursor()
            if cursor.hasSelection() and cursor.selectedText() == find:
                cursor.insertText(replace)
                count = 1
            else:
                if self.find_text(find):
                    cursor = self.textCursor()
                    cursor.insertText(replace)
                    count = 1

        return count


class URDFCodeEditorWidget(QWidget):
    """Complete code editor widget with toolbar and status."""

    content_saved = pyqtSignal(str)  # Emitted when content is saved
    validation_changed = pyqtSignal(bool, list)  # Validation status

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the code editor widget."""
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        self._current_file: str | None = None

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QHBoxLayout()

        self.validate_btn = QPushButton("Validate")
        self.format_btn = QPushButton("Format")
        self.find_btn = QPushButton("Find/Replace")

        toolbar.addWidget(self.validate_btn)
        toolbar.addWidget(self.format_btn)
        toolbar.addWidget(self.find_btn)
        toolbar.addStretch()

        layout.addLayout(toolbar)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Code editor
        self.editor = URDFCodeEditor()
        splitter.addWidget(self.editor)

        # Error/output panel
        self.output_panel = QTextEdit()
        self.output_panel.setReadOnly(True)
        self.output_panel.setMaximumHeight(100)
        self.output_panel.setPlaceholderText("Validation messages will appear here...")
        splitter.addWidget(self.output_panel)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # Status bar
        status_layout = QHBoxLayout()
        self.position_label = QLabel("Line 1, Col 1")
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.position_label)
        status_layout.addStretch()
        status_layout.addWidget(self.status_label)
        layout.addLayout(status_layout)

    def _connect_signals(self) -> None:
        """Connect signals to slots."""
        self.validate_btn.clicked.connect(self._on_validate)
        self.format_btn.clicked.connect(self._on_format)
        self.find_btn.clicked.connect(self._on_find_replace)

        self.editor.cursor_position_changed.connect(self._on_cursor_position_changed)
        self.editor.validation_result.connect(self._on_validation_result)

    def _on_validate(self) -> None:
        """Handle validate button click."""
        # First validate XML
        xml_valid, xml_errors = self.editor.validate_xml()
        if not xml_valid:
            self._show_errors(xml_errors)
            return

        # Then validate URDF structure
        urdf_valid, urdf_errors = self.editor.validate_urdf()
        if urdf_valid:
            self.output_panel.setHtml(
                '<span style="color: green;">Validation successful - URDF is valid!</span>'
            )
            self.status_label.setText("Valid URDF")
        else:
            self._show_errors(urdf_errors)

    def _show_errors(self, errors: list[str]) -> None:
        """Display validation errors."""
        html = '<span style="color: red;">Validation errors:</span><br>'
        for error in errors:
            html += f'<span style="color: red;">- {error}</span><br>'
        self.output_panel.setHtml(html)
        self.status_label.setText(f"{len(errors)} error(s)")

    def _on_format(self) -> None:
        """Handle format button click."""
        content = self.editor.get_content()
        if not content.strip():
            return

        try:
            # Parse and re-format XML
            root = ET.fromstring(content)
            ET.indent(root, space="  ")
            formatted = ET.tostring(root, encoding="unicode", xml_declaration=True)
            self.editor.set_content(formatted)
            self.status_label.setText("Formatted")
        except ET.ParseError as e:
            self.output_panel.setHtml(
                f'<span style="color: red;">Cannot format - invalid XML: {e}</span>'
            )

    def _on_find_replace(self) -> None:
        """Handle find/replace button click."""
        dialog = FindReplaceDialog(self.editor, self)
        dialog.exec()

    def _on_cursor_position_changed(self, line: int, col: int) -> None:
        """Handle cursor position changes."""
        self.position_label.setText(f"Line {line}, Col {col}")

    def _on_validation_result(self, is_valid: bool, errors: list[str]) -> None:
        """Handle validation results."""
        self.validation_changed.emit(is_valid, errors)

    def set_content(self, content: str, file_path: str | None = None) -> None:
        """Set editor content."""
        self.editor.set_content(content)
        self._current_file = file_path
        if file_path:
            self.status_label.setText(f"Loaded: {file_path}")

    def get_content(self) -> str:
        """Get editor content."""
        return self.editor.get_content()


class FindReplaceDialog(QDialog):
    """Dialog for find and replace functionality."""

    def __init__(self, editor: URDFCodeEditor, parent: QWidget | None = None) -> None:
        """Initialize the dialog."""
        super().__init__(parent)
        self.editor = editor
        self.setWindowTitle("Find and Replace")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Find row
        find_layout = QHBoxLayout()
        find_layout.addWidget(QLabel("Find:"))
        self.find_edit = QLineEdit()
        find_layout.addWidget(self.find_edit)
        self.find_btn = QPushButton("Find Next")
        find_layout.addWidget(self.find_btn)
        layout.addLayout(find_layout)

        # Replace row
        replace_layout = QHBoxLayout()
        replace_layout.addWidget(QLabel("Replace:"))
        self.replace_edit = QLineEdit()
        replace_layout.addWidget(self.replace_edit)
        self.replace_btn = QPushButton("Replace")
        self.replace_all_btn = QPushButton("Replace All")
        replace_layout.addWidget(self.replace_btn)
        replace_layout.addWidget(self.replace_all_btn)
        layout.addLayout(replace_layout)

        # Status
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        # Connect signals
        self.find_btn.clicked.connect(self._on_find)
        self.replace_btn.clicked.connect(self._on_replace)
        self.replace_all_btn.clicked.connect(self._on_replace_all)

    def _on_find(self) -> None:
        """Handle find button click."""
        text = self.find_edit.text()
        if text:
            found = self.editor.find_text(text)
            if found:
                self.status_label.setText("Found")
            else:
                self.status_label.setText("Not found")

    def _on_replace(self) -> None:
        """Handle replace button click."""
        find = self.find_edit.text()
        replace = self.replace_edit.text()
        if find:
            count = self.editor.replace_text(find, replace)
            self.status_label.setText(f"Replaced {count} occurrence(s)")

    def _on_replace_all(self) -> None:
        """Handle replace all button click."""
        find = self.find_edit.text()
        replace = self.replace_edit.text()
        if find:
            count = self.editor.replace_text(find, replace, all_occurrences=True)
            self.status_label.setText(f"Replaced {count} occurrence(s)")
