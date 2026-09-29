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
    "focus": "#89b4fa",
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
