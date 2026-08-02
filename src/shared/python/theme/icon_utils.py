# ruff: noqa: E501
# This module is a static catalogue of Lucide SVG glyphs stored as raw string
# literals. Each glyph is naturally one long line; wrapping breaks the SVG
# templating contract used downstream. Lint suppressed at module level.

import re
from pathlib import Path

from PyQt6.QtCore import QByteArray
from PyQt6.QtGui import QIcon, QPixmap

_SVG_HOME = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>"""
_SVG_COMPUTER = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" x2="16" y1="21" y2="21"/><line x1="12" x2="12" y1="17" y2="21"/></svg>"""
_SVG_SETTINGS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>"""
_SVG_HELP = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>"""
_SVG_SEARCH = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>"""
_SVG_CLOSE = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>"""
_SVG_MINIMIZE = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/></svg>"""
_SVG_MAXIMIZE = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/></svg>"""
_SVG_BIOMECHANICS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="4" r="2"/><path d="M12 6v6"/><path d="M12 12 8 17"/><path d="M12 12l4 5"/><path d="M9 8H6"/><path d="M15 8h3"/><path d="M8 17v4"/><path d="M16 17v4"/></svg>"""
# ---------------------------------------------------------------------------
# Launcher sidebar icons (#5624): every sidebar nav button needs a
# registered SVG glyph here, otherwise ``IconColorizer.get_icon`` raises
# ``ValueError`` and the launcher's ``_build_sidebar_button`` swallows it,
# leaving a text-only button.  Adding the missing fallbacks below closes
# that gap with simple Lucide-style line glyphs.
# ---------------------------------------------------------------------------
_SVG_ACCESSIBILITY = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="4" r="2"/><path d="M19 13V8h-3l-2 4-4-1-4 1L4 8H1v5"/><path d="m8 22 4-5 4 5"/><path d="M12 13v4"/></svg>"""
_SVG_SPORTS_GOLF = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="6" r="4"/><path d="M9 10v8"/><path d="M5 22h14"/><path d="M9 18l8-3"/></svg>"""
_SVG_DIRECTIONS_RUN = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="15" cy="4" r="2"/><path d="m7 21 3-7 4 3v6"/><path d="m7 14 2-4 4 1 3-3"/><path d="M16 13h4"/></svg>"""
_SVG_VIDEOCAM = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m22 8-6 4 6 4z"/><rect width="14" height="12" x="2" y="6" rx="2"/></svg>"""
_SVG_BUILD = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>"""
_SVG_CHAT = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>"""
_SVG_MENU = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" x2="21" y1="12" y2="12"/><line x1="3" x2="21" y1="6" y2="6"/><line x1="3" x2="21" y1="18" y2="18"/></svg>"""

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
    # Launcher sidebar additions (#5624)
    "accessibility": _SVG_ACCESSIBILITY,
    "sports_golf": _SVG_SPORTS_GOLF,
    "directions_run": _SVG_DIRECTIONS_RUN,
    "videocam": _SVG_VIDEOCAM,
    "build": _SVG_BUILD,
    "chat": _SVG_CHAT,
    "menu": _SVG_MENU,
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
