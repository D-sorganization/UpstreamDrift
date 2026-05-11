import re
from pathlib import Path
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import QByteArray

_SVG_HOME = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>"""
_SVG_COMPUTER = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" x2="16" y1="21" y2="21"/><line x1="12" x2="12" y1="17" y2="21"/></svg>"""
_SVG_SETTINGS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>"""
_SVG_HELP = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>"""
_SVG_SEARCH = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>"""
_SVG_CLOSE = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>"""
_SVG_MINIMIZE = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/></svg>"""
_SVG_MAXIMIZE = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/></svg>"""
_SVG_BIOMECHANICS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="4" r="2"/><path d="M12 6v6"/><path d="M12 12 8 17"/><path d="M12 12l4 5"/><path d="M9 8H6"/><path d="M15 8h3"/><path d="M8 17v4"/><path d="M16 17v4"/></svg>"""

SVG_REGISTRY = {
    "home": _SVG_HOME,
    "computer": _SVG_COMPUTER,
    "settings": _SVG_SETTINGS,
    "help": _SVG_HELP,
    "search": _SVG_SEARCH,
    "close": _SVG_CLOSE,
    "minimize": _SVG_MINIMIZE,
    "maximize": _SVG_MAXIMIZE,
    "biomechanics": _SVG_BIOMECHANICS,
}


class IconColorizer:
    @staticmethod
    def get_icon(name: str, color: str) -> QIcon:
        """
        Get a QIcon generated from an SVG string with the specified stroke color.

        Preconditions:
            - name must be a registered icon string.
            - color must be a valid hex string or color name.
        """
        assert isinstance(name, str), "name must be a string"
        assert isinstance(color, str), "color must be a string"

        svg_content = SVG_REGISTRY.get(name)
        if not svg_content:
            raise ValueError(f"Icon '{name}' is not registered in SVG_REGISTRY.")

        colored_svg = svg_content.replace("{color}", color)
        pixmap = QPixmap()
        pixmap.loadFromData(QByteArray(colored_svg.encode("utf-8")))
        return QIcon(pixmap)

    @staticmethod
    def colorize_svg_file(path: str | Path, color: str) -> QIcon:
        """
        Dynamically recolor an external SVG file's fill/stroke.

        Preconditions:
            - path must be a valid, existing file path.
            - color must be a valid hex string or color name.
        """
        assert isinstance(color, str), "color must be a string"

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SVG file not found at {path}")

        with open(path, encoding="utf-8") as f:
            svg_content = f.read()

        svg_content = re.sub(r'fill="[^"]+"', f'fill="{color}"', svg_content)
        svg_content = re.sub(r'stroke="[^"]+"', f'stroke="{color}"', svg_content)

        pixmap = QPixmap()
        pixmap.loadFromData(QByteArray(svg_content.encode("utf-8")))
        return QIcon(pixmap)
