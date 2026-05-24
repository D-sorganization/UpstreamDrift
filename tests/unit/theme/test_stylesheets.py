"""Tests for src.shared.python.theme.stylesheets (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.theme.stylesheets import (
    generate_minimal_stylesheet,
    generate_stylesheet,
)

_THEME: dict[str, str] = {
    "bg": "#1e1e2e",
    "group_bg": "#2a2a3e",
    "border": "#3d3d5c",
    "text": "#cdd6f4",
    "text_secondary": "#a6adc8",
    "label": "#6c7086",
    "focus": "#94e2d5",
    "input_bg": "#181825",
    "accent": "#89b4fa",
    "title_bg": "#313244",
    "title_border": "#45475a",
    "table_header": "#313244",
    "table_alt": "#252535",
    "button_hover": "#7487c8",
}


class TestGenerateStylesheet:
    def test_stylesheets_returns_string(self) -> None:
        result = generate_stylesheet(_THEME)
        assert isinstance(result, str)

    def test_stylesheets_nonempty(self) -> None:
        result = generate_stylesheet(_THEME)
        assert len(result) > 0

    def test_contains_bg_color(self) -> None:
        result = generate_stylesheet(_THEME)
        assert _THEME["bg"] in result

    def test_contains_text_color(self) -> None:
        result = generate_stylesheet(_THEME)
        assert _THEME["text"] in result

    def test_contains_qwidget(self) -> None:
        result = generate_stylesheet(_THEME)
        assert "QWidget" in result

    def test_contains_qpushbutton(self) -> None:
        result = generate_stylesheet(_THEME)
        assert "QPushButton" in result

    def test_contains_qlineedit(self) -> None:
        result = generate_stylesheet(_THEME)
        assert "QLineEdit" in result

    def test_contains_accent_color(self) -> None:
        result = generate_stylesheet(_THEME)
        assert _THEME["accent"] in result

    def test_full_stylesheet_includes_each_theme_token(self) -> None:
        result = generate_stylesheet(_THEME)

        for color in _THEME.values():
            assert color in result

    def test_full_stylesheet_includes_major_widget_sections(self) -> None:
        result = generate_stylesheet(_THEME)

        expected_selectors = [
            "QMainWindow, QWidget",
            "QGroupBox::title",
            "QComboBox QAbstractItemView",
            "QMenu::separator",
            "QTabBar::tab:selected",
            "QTableWidget, QTableView",
            "QTreeWidget::item:selected, QTreeView::item:selected",
            "QProgressBar::chunk",
            "QCheckBox::indicator:checked",
            "QToolBar::separator",
            "QDockWidget::title",
            "QScrollBar::handle:horizontal:hover",
            'QFrame[frameShape="4"], QFrame[frameShape="5"]',
            "#launchButton:pressed",
        ]
        for selector in expected_selectors:
            assert selector in result

    def test_full_stylesheet_preserves_section_order(self) -> None:
        result = generate_stylesheet(_THEME)

        ordered_markers = [
            "QMainWindow, QWidget",
            "QGroupBox",
            "QScrollArea",
            "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox",
            "QLabel",
            "QPushButton",
            "QMenuBar",
            "QTabWidget::pane",
            "QTableWidget, QTableView",
            "QTextEdit, QPlainTextEdit",
            "QProgressBar",
            "QSlider::groove:horizontal",
            "QToolTip",
            "QDockWidget",
            "QSplitter::handle",
            "ToolCard",
        ]
        positions = [result.index(marker) for marker in ordered_markers]

        assert positions == sorted(positions)

    def test_full_stylesheet_raises_for_missing_required_color(self) -> None:
        incomplete_theme = dict(_THEME)
        del incomplete_theme["button_hover"]

        try:
            generate_stylesheet(incomplete_theme)
        except KeyError as exc:
            assert exc.args == ("button_hover",)
        else:
            raise AssertionError("Expected KeyError for missing theme color")


class TestGenerateMinimalStylesheet:
    def test_stylesheets_returns_string(self) -> None:
        result = generate_minimal_stylesheet(_THEME)
        assert isinstance(result, str)

    def test_stylesheets_nonempty(self) -> None:
        result = generate_minimal_stylesheet(_THEME)
        assert len(result) > 0

    def test_contains_bg_color(self) -> None:
        result = generate_minimal_stylesheet(_THEME)
        assert _THEME["bg"] in result

    def test_minimal_shorter_than_full(self) -> None:
        full = generate_stylesheet(_THEME)
        minimal = generate_minimal_stylesheet(_THEME)
        assert len(minimal) < len(full)

    def test_contains_qwidget(self) -> None:
        result = generate_minimal_stylesheet(_THEME)
        assert "QWidget" in result

    def test_minimal_stylesheet_only_uses_embedding_selectors(self) -> None:
        result = generate_minimal_stylesheet(_THEME)

        assert "QWidget" in result
        assert "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox" in result
        assert "QPushButton:hover" in result
        assert "QMenuBar" not in result
        assert "QDockWidget" not in result
        assert "ToolCard" not in result

    def test_minimal_stylesheet_uses_expected_theme_colors(self) -> None:
        result = generate_minimal_stylesheet(_THEME)

        used_keys = {"bg", "text", "input_bg", "border", "group_bg", "accent"}
        unused_keys = set(_THEME) - used_keys
        for key in used_keys:
            assert _THEME[key] in result
        for key in unused_keys:
            assert _THEME[key] not in result

    def test_minimal_stylesheet_raises_for_missing_required_color(self) -> None:
        incomplete_theme = dict(_THEME)
        del incomplete_theme["group_bg"]

        try:
            generate_minimal_stylesheet(incomplete_theme)
        except KeyError as exc:
            assert exc.args == ("group_bg",)
        else:
            raise AssertionError("Expected KeyError for missing theme color")
