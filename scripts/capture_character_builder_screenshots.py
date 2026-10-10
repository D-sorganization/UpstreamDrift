"""Capture the desktop Character Builder walkthrough screenshots (CMB-12, #11663).

Drives the PyQt6 ``CharacterBuilderWidget`` offscreen through the tutorial's
desktop steps (pick a preset, override the mass, swap the club, export the
spec) and saves one PNG per step. The images in
``docs/user_guide/images/character_builder/`` are this script's output::

    python3 scripts/capture_character_builder_screenshots.py

No window is shown: ``QT_QPA_PLATFORM`` defaults to ``offscreen``.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from contextlib import chdir
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# The builder's import chain needs both roots (see the tutorial prerequisites).
for _root in (REPO_ROOT / "src", REPO_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

DEFAULT_OUT = REPO_ROOT / "docs/user_guide/images/character_builder"
WIDGET_SIZE = (960, 600)
PRESET_ID = "tour_average_male"
MASS_KG = 90.0
CLUB = "iron7"
STEP_FILES = (
    "desktop_01_preset.png",
    "desktop_02_mass_override.png",
    "desktop_03_club_swap.png",
    "desktop_04_export.png",
)


def _use_bundled_font(app: Any) -> None:
    """Use matplotlib's DejaVu Sans so text renders on font-less offscreen hosts."""
    import matplotlib
    from PyQt6 import QtGui

    ttf = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    font_id = QtGui.QFontDatabase.addApplicationFont(str(ttf))
    if font_id < 0:
        raise OSError(f"could not load font {ttf}")
    family = QtGui.QFontDatabase.applicationFontFamilies(font_id)[0]
    app.setFont(QtGui.QFont(family, 9))


def capture(out_dir: Path) -> list[Path]:
    """Render the four walkthrough steps into ``out_dir``.

    Postcondition: returns the four written PNG paths in ``STEP_FILES`` order,
    each ``WIDGET_SIZE`` pixels. The export step writes into a temporary
    directory so its status line shows a bare filename, not a host path.
    """
    if not isinstance(out_dir, Path):
        raise TypeError(f"out_dir must be a Path, got {type(out_dir).__name__}")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets

    from src.tools.character_builder.gui import CharacterBuilderWidget

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _use_bundled_font(app)
    out_dir.mkdir(parents=True, exist_ok=True)
    widget = CharacterBuilderWidget()
    widget.resize(*WIDGET_SIZE)
    widget.show()
    paths = [out_dir / name for name in STEP_FILES]

    def grab(path: Path) -> None:
        app.processEvents()
        if not widget.grab().save(str(path), "PNG"):
            raise OSError(f"could not write screenshot {path}")

    try:
        widget.preset_combo.setCurrentIndex(widget.preset_combo.findData(PRESET_ID))
        grab(paths[0])
        widget.spins["mass_kg"].setValue(MASS_KG)
        grab(paths[1])
        widget.club_combo.setCurrentText(CLUB)
        grab(paths[2])
        with tempfile.TemporaryDirectory() as tmp, chdir(tmp):
            if widget.export("spec", ".") is None:
                raise RuntimeError(f"spec export failed: {widget.status.text()}")
            grab(paths[3])
    finally:
        widget.close()
        widget.cleanup()
    return paths


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    for path in capture(args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
