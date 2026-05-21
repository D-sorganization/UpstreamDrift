"""Tests for non-GUI logic in src.launchers.custom_title_bar."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.launchers import custom_title_bar as ctb


@pytest.fixture(scope="module")
def _qapp() -> object:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


class TestGetTitleBarColors:
    def test_returns_required_keys(self) -> None:
        colors = ctb._get_title_bar_colors()
        for key in ("text", "bg", "border"):
            assert key in colors
            assert isinstance(colors[key], str)

    def test_fallback_on_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        original_import = builtins.__import__

        def fake_import(name: str, *a: object, **kw: object) -> object:
            if "theme" in name and "shared" in name:
                raise ImportError(name)
            return original_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        colors = ctb._get_title_bar_colors()
        assert colors == {"text": "#E0E0E0", "bg": "#1A1A1A", "border": "#555555"}


class TestMakeButtonStylesheet:
    def test_basic(self) -> None:
        s = ctb._make_button_stylesheet("#FFFFFF")
        assert "color: #FFFFFF" in s
        assert "QToolButton" in s
        assert "rgba(255, 255, 255, 0.1)" in s

    def test_with_hover_text(self) -> None:
        s = ctb._make_button_stylesheet(
            "#FFF", hover_bg="#E81123", hover_text_color="#000"
        )
        assert "#E81123" in s
        assert "color: #000" in s

    def test_no_hover_text_when_empty(self) -> None:
        s = ctb._make_button_stylesheet("#FFF", hover_text_color="")
        # No 'color: ;' literal — but should still have color: #FFF
        assert "color: #FFF" in s


class TestClampToVisibleScreen:
    def test_returns_qpoint(self, _qapp: object) -> None:
        from PyQt6.QtCore import QPoint

        out = ctb.clamp_to_visible_screen(QPoint(100, 100))
        assert isinstance(out, QPoint)

    def test_no_screen_returns_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from PyQt6.QtCore import QPoint
        from PyQt6.QtWidgets import QApplication

        monkeypatch.setattr(QApplication, "primaryScreen", staticmethod(lambda: None))
        target = QPoint(50, 75)
        out = ctb.clamp_to_visible_screen(target)
        assert out.x() == 50
        assert out.y() == 75

    def test_clamps_to_screen_bounds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from PyQt6.QtCore import QPoint, QRect
        from PyQt6.QtWidgets import QApplication

        fake_screen = MagicMock()
        fake_screen.availableGeometry.return_value = QRect(0, 0, 1000, 800)
        monkeypatch.setattr(
            QApplication, "primaryScreen", staticmethod(lambda: fake_screen)
        )

        # Far off-screen should be clamped
        out = ctb.clamp_to_visible_screen(QPoint(5000, 5000))
        assert out.x() < 5000
        assert out.y() < 5000

        # Negative should be clamped to >= 0
        out2 = ctb.clamp_to_visible_screen(QPoint(-500, -500))
        assert out2.x() >= 0
        assert out2.y() >= 0


class TestCreateWindowControlButton:
    def test_creates_button(self, _qapp: object) -> None:
        btn = ctb.create_window_control_button(
            "minimize",
            "-",
            tooltip="tip",
            accessible_name="acc",
            object_name="obj",
        )
        assert btn.toolTip() == "tip"
        assert btn.accessibleName() == "acc"
        assert btn.objectName() == "obj"

    def test_fallback_text_when_no_colorizer(
        self, _qapp: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ctb, "IconColorizer", None)
        btn = ctb.create_window_control_button(
            "minimize",
            "FB",
            tooltip="t",
            accessible_name="a",
            object_name="o",
        )
        assert btn.text() == "FB"

    def test_uses_icon_when_colorizer_available(
        self, _qapp: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from PyQt6.QtGui import QIcon

        fake_colorizer = MagicMock()
        fake_colorizer.get_icon.return_value = QIcon()
        monkeypatch.setattr(ctb, "IconColorizer", fake_colorizer)

        btn = ctb.create_window_control_button(
            "close",
            "X",
            tooltip="t",
            accessible_name="a",
            object_name="o",
            color="#ABCDEF",
        )
        fake_colorizer.get_icon.assert_called_once_with("close", "#ABCDEF")
        # When icon set, text shouldn't be the fallback
        assert btn.text() == ""

    def test_color_defaults_to_theme(
        self, _qapp: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[str] = []
        fake_colorizer = MagicMock()

        def get_icon(name: str, color: str):  # noqa: ANN202
            captured.append(color)
            from PyQt6.QtGui import QIcon

            return QIcon()

        fake_colorizer.get_icon.side_effect = get_icon
        monkeypatch.setattr(ctb, "IconColorizer", fake_colorizer)

        ctb.create_window_control_button(
            "minimize",
            "-",
            tooltip="t",
            accessible_name="a",
            object_name="o",
        )
        # Should have used the title-bar text color
        assert captured and captured[0] == ctb._get_title_bar_colors()["text"]
