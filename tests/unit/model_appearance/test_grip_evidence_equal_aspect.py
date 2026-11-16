"""Grip evidence scripts must crop preview axes with equal half-widths (#15141)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = (
    _ROOT
    / "docs/development/full_body_models/evidence/grip_kinetics/render_grip_clip.py",
    _ROOT
    / "docs/development/full_body_models/evidence/grip_kinetics/run_grip_kinetics.py",
)


@pytest.mark.parametrize("path", _SCRIPTS)
def test_grip_kinetics_renderers_use_equal_axis_helper(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "apply_equal_axis_limits" in text, path.name
    assert (
        "set_box_aspect((1, 1, 1))" not in text.split("apply_equal_axis_limits")[0]
    ), path.name
