"""Tests for honest messaging in sim GUIs — issue #6641 F3/F4.

Verifies that the bunker shot GUI and putting green GUI do not claim a
physics engine ran when only a procedural preview was generated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

BUNKER_GUI = (
    Path(__file__).parent.parent.parent / "src" / "tools" / "bunker_shot_gui" / "gui.py"
)
PUTTING_GUI = (
    Path(__file__).parent.parent.parent
    / "src"
    / "tools"
    / "putting_green_gui"
    / "gui.py"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_bunker_gui_no_false_dem_claim() -> None:
    """Bunker shot GUI must not claim 'Chrono DEM simulation ... completed'."""
    text = _read(BUNKER_GUI)
    assert "Chrono DEM simulation mock completed" not in text, (
        "Bunker shot GUI still claims a DEM engine completed — use honest preview messaging"
    )


def test_bunker_gui_uses_preview_label() -> None:
    """Bunker shot GUI group box should indicate it's a preview."""
    text = _read(BUNKER_GUI)
    assert "Preview" in text or "not yet wired" in text, (
        "Bunker shot GUI should label results as a preview or indicate engine is not wired"
    )


def test_putting_gui_no_false_loaded_claim() -> None:
    """Putting green GUI must not claim 'Simulator loaded successfully'."""
    text = _read(PUTTING_GUI)
    assert "Simulator loaded successfully" not in text, (
        "Putting green GUI still claims the simulator loaded — use honest preview messaging"
    )


def test_putting_gui_uses_preview_label() -> None:
    """Putting green GUI result text should indicate it is a preview."""
    text = _read(PUTTING_GUI)
    assert "Preview" in text or "not yet wired" in text, (
        "Putting green GUI should label results as a preview or indicate engine is not wired"
    )
