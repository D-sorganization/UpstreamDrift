from pathlib import Path

import pytest
from PyQt6.QtGui import QIcon

from src.shared.python.theme import icon_utils
from src.shared.python.theme.icon_utils import SVG_REGISTRY, IconColorizer


def test_icon_colorizer_get_icon_success(qapp):
    """Test that a valid registered icon returns a QIcon."""
    icon = IconColorizer.get_icon("home", "#ff0000")
    assert isinstance(icon, QIcon)
    assert not icon.isNull()


@pytest.mark.parametrize(
    "icon_name",
    [
        "accessibility",
        "sports_golf",
        "directions_run",
        "videocam",
        "build",
        "book",
        "chat",
        "menu",
    ],
)
def test_launcher_sidebar_icons_are_registered_and_render(icon_name, qapp):
    """Launcher sidebar fallback glyphs should stay available."""
    icon = IconColorizer.get_icon(icon_name, "navy")

    assert icon_name in SVG_REGISTRY
    assert isinstance(icon, QIcon)
    assert not icon.isNull()


def test_icon_colorizer_get_icon_invalid_name():
    """Test that an invalid icon name raises ValueError."""
    with pytest.raises(ValueError, match="is not registered"):
        IconColorizer.get_icon("non_existent_icon", "#ffffff")


def test_icon_colorizer_type_assertions():
    """Test that invalid types raise AssertionError."""
    with pytest.raises(AssertionError, match="name must be a string"):
        IconColorizer.get_icon(123, "#ffffff")

    with pytest.raises(AssertionError, match="color must be a string"):
        IconColorizer.get_icon("home", None)


def test_get_icon_injects_requested_color_into_svg(monkeypatch):
    """Registered SVG placeholders are replaced before pixmap loading."""
    loaded_svg: dict[str, str] = {}

    class _Pixmap:
        def loadFromData(self, payload):  # noqa: N802
            loaded_svg["content"] = bytes(payload).decode("utf-8")
            return True

    class _Icon:
        def __init__(self, pixmap) -> None:
            self.pixmap = pixmap

    monkeypatch.setattr(icon_utils, "QPixmap", _Pixmap)
    monkeypatch.setattr(icon_utils, "QIcon", _Icon)

    icon = IconColorizer.get_icon("search", "#123abc")

    assert isinstance(icon, _Icon)
    assert 'stroke="#123abc"' in loaded_svg["content"]
    assert "{color}" not in loaded_svg["content"]


def test_icon_colorizer_colorize_svg_file(tmp_path, qapp):
    """Test dynamic recoloring of an external SVG file."""
    svg_file = tmp_path / "test.svg"
    svg_file.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000">'
        '<path d="M12 2L2 22h20L12 2z"/>'
        "</svg>"
    )

    icon = IconColorizer.colorize_svg_file(svg_file, "#ff0000")
    assert isinstance(icon, QIcon)
    assert not icon.isNull()


def test_colorize_svg_file_rewrites_fill_and_stroke_attributes(
    tmp_path,
    monkeypatch,
):
    """External SVG recoloring should update every quoted fill/stroke value."""
    svg_file = tmp_path / "test.svg"
    svg_file.write_text(
        '<svg fill="none" stroke="#111">'
        '<path fill="#222" stroke="currentColor" d="M1 1h2v2z"/>'
        "</svg>",
        encoding="utf-8",
    )
    loaded_svg: dict[str, str] = {}

    class _Pixmap:
        def loadFromData(self, payload):  # noqa: N802
            loaded_svg["content"] = bytes(payload).decode("utf-8")
            return True

    class _Icon:
        def __init__(self, pixmap) -> None:
            self.pixmap = pixmap

    monkeypatch.setattr(icon_utils, "QPixmap", _Pixmap)
    monkeypatch.setattr(icon_utils, "QIcon", _Icon)

    icon = IconColorizer.colorize_svg_file(Path(svg_file), "#00ff88")

    assert isinstance(icon, _Icon)
    assert loaded_svg["content"].count('fill="#00ff88"') == 2
    assert loaded_svg["content"].count('stroke="#00ff88"') == 2
    assert "currentColor" not in loaded_svg["content"]


def test_icon_colorizer_colorize_svg_file_not_found():
    """Test behavior when SVG file does not exist."""
    with pytest.raises(FileNotFoundError):
        IconColorizer.colorize_svg_file("nonexistent.svg", "#000000")


def test_colorize_svg_file_requires_string_color(tmp_path):
    svg_file = tmp_path / "test.svg"
    svg_file.write_text("<svg />", encoding="utf-8")

    with pytest.raises(AssertionError, match="color must be a string"):
        IconColorizer.colorize_svg_file(svg_file, None)
