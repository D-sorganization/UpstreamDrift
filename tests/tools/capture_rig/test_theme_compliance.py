"""Theme compliance for the Capture Rig package (#9816).

Static: no module in ``src/tools/capture_rig`` names a colour or passes a
string straight to ``setStyleSheet``; every style comes from
:mod:`src.tools.capture_rig.styling`, which composes it from the active
palette and the fleet ``Styles`` constants. (BGR tuples for OpenCV drawing
on video frames are not UI colours and are outside this rule.)

Dynamic: the tile builds under the built-in Dark and Light themes, its
computed styles differ between them, and it restyles on ``themeChanged``.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("cv2")

from PyQt6.QtWidgets import QApplication

from src.shared.python.theme.colors import BUILTIN_THEMES
from src.shared.python.theme.style_constants import Styles
from src.shared.python.theme.theme_manager import ThemeManager
from src.tools.capture_rig import styling
from src.tools.capture_rig.gui import CaptureRigWidget, CaptureRigWindow
from src.tools.capture_rig.preview import stamp_badge

pytestmark = [pytest.mark.unit, pytest.mark.ui]

PACKAGE = Path(__file__).resolve().parents[3] / "src" / "tools" / "capture_rig"
HEX_LITERAL = re.compile(r"#[0-9a-fA-F]{6}\b")
INLINE_STYLESHEET = re.compile(r"setStyleSheet\(\s*f?[\"']")
NAMED_QCOLOR = re.compile(r"QColor\(\s*[\"']")
#: ``{file name: literals}`` a module may keep; empty today and meant to stay so.
ALLOWED_LITERALS: dict[str, frozenset[str]] = {}

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # type: ignore[assignment]
    return _APP  # type: ignore[return-value]


def _sources() -> list[Path]:
    files = sorted(PACKAGE.glob("*.py"))
    assert files, PACKAGE
    return files


@pytest.fixture
def theme_manager() -> Iterator[ThemeManager]:
    """The singleton, with the operator's theme put back afterwards."""
    _app()
    manager = ThemeManager.instance()
    before = manager.get_current_theme_name()
    try:
        yield manager
    finally:
        manager.change_theme(before)


# -- static --------------------------------------------------------------------
def test_no_literal_colours_in_the_package() -> None:
    offenders = []
    for path in _sources():
        allowed = ALLOWED_LITERALS.get(path.name, frozenset())
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for literal in HEX_LITERAL.findall(line):
                if literal not in allowed:
                    offenders.append(f"{path.name}:{number}: {literal}")
            if NAMED_QCOLOR.search(line):
                offenders.append(f"{path.name}:{number}: named QColor")
    assert offenders == []


def test_no_inline_stylesheet_strings() -> None:
    offenders = [
        f"{path.name}:{number}: {line.strip()}"
        for path in _sources()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if INLINE_STYLESHEET.search(line)
    ]
    assert offenders == []


# -- styling helpers -------------------------------------------------------------
def test_signal_colours_come_from_the_fleet_styles() -> None:
    signals = styling.signal_colors()
    assert signals.record == styling.color_of(Styles.STATUS_ERROR)
    assert signals.warning == styling.color_of(Styles.STATUS_WARNING)
    assert signals.ok == styling.color_of(Styles.STATUS_SUCCESS)
    with pytest.raises(ValueError):
        styling.color_of("font-weight: bold;")


def test_contrast_text_reads_on_light_and_dark_pills() -> None:
    dark, light = BUILTIN_THEMES["Dark"], BUILTIN_THEMES["Light"]
    red = styling.color_of(Styles.STATUS_ERROR)
    yellow = styling.color_of(Styles.STATUS_WARNING)
    assert styling.contrast_text(red, dark) == dark["text"]
    assert styling.contrast_text(red, light) == light["bg"]
    assert styling.contrast_text(yellow, dark) == dark["bg"]
    assert styling.contrast_text(yellow, light) == light["text"]


def test_chip_style_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        styling.chip_style("loud")
    for kind in styling.CHIP_KINDS:
        assert "border-radius" in styling.chip_style(kind)


# -- dynamic ---------------------------------------------------------------------
@pytest.mark.parametrize("theme", ["Dark", "Light"])
def test_tile_builds_under_builtin_theme(
    theme: str, theme_manager: ThemeManager
) -> None:
    theme_manager.change_theme(theme)
    window = CaptureRigWindow()
    widget = window.widget
    assert BUILTIN_THEMES[theme]["bg"] in window.styleSheet()
    assert BUILTIN_THEMES[theme]["title_bg"] in widget.header.styleSheet()
    assert widget.status_strip.cameras.text().startswith("cameras")
    widget.shutdown()
    window.close()


def test_computed_styles_differ_between_themes(theme_manager: ThemeManager) -> None:
    theme_manager.change_theme("Dark")
    widget = CaptureRigWidget()
    dark = (
        widget.header.styleSheet(),
        styling.tile_style(),
        widget.action_grid.section_style(),
    )
    theme_manager.change_theme("Light")  # themeChanged → widget.restyle()
    light = (
        widget.header.styleSheet(),
        styling.tile_style(),
        widget.action_grid.section_style(),
    )
    assert all(a != b for a, b in zip(dark, light, strict=True))
    assert BUILTIN_THEMES["Light"]["title_bg"] in light[0]
    widget.shutdown()


def test_badge_uses_the_signal_colour_not_a_literal(
    theme_manager: ThemeManager,
) -> None:
    from PyQt6.QtGui import QColor, QPixmap

    theme_manager.change_theme("Dark")
    pixmap = QPixmap(240, 120)
    pixmap.fill(QColor(BUILTIN_THEMES["Dark"]["bg"]))
    stamped = stamp_badge(pixmap, "REC")
    image = stamped.toImage()
    inset = styling.LayoutMetrics.SPACING_SM + 2
    pill = QColor(image.pixel(inset + 2, inset + 2)).name()
    assert pill == styling.signal_colors().record


def test_embed_adapter_applies_the_theme(theme_manager: ThemeManager) -> None:
    from src.tools.capture_rig._embed_adapter import CaptureRigAdapter

    theme_manager.change_theme("Light")
    adapter = CaptureRigAdapter()
    widget = adapter.create_main_widget(None)
    assert BUILTIN_THEMES["Light"]["bg"] in widget.styleSheet()
    adapter.cleanup()
