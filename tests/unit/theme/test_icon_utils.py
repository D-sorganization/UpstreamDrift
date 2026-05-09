import pytest
from pathlib import Path
from PyQt6.QtGui import QIcon
from src.shared.python.theme.icon_utils import IconColorizer

def test_icon_colorizer_get_icon_success(qapp):
    """Test that a valid registered icon returns a QIcon."""
    icon = IconColorizer.get_icon("home", "#ff0000")
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

def test_icon_colorizer_colorize_svg_file(tmp_path, qapp):
    """Test dynamic recoloring of an external SVG file."""
    svg_file = tmp_path / "test.svg"
    svg_file.write_text('<svg fill="none" stroke="#000"></svg>')
    
    icon = IconColorizer.colorize_svg_file(svg_file, "#ff0000")
    assert isinstance(icon, QIcon)
    assert not icon.isNull()

def test_icon_colorizer_colorize_svg_file_not_found():
    """Test behavior when SVG file does not exist."""
    with pytest.raises(FileNotFoundError):
        IconColorizer.colorize_svg_file("nonexistent.svg", "#000000")
