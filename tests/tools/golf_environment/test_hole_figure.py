"""Headless hole figure tests (matplotlib Agg, epic #15150)."""

from __future__ import annotations

import pytest

from src.tools.golf_environment import shots as sh
from tests.tools.golf_environment.test_shots import _Hole, _flight, _place

pytestmark = pytest.mark.unit


def _figure_module():
    try:
        import matplotlib  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"matplotlib not loadable: {exc}")
    from src.tools.golf_environment import hole_figure

    return hole_figure


def test_write_hole_figure_makes_png(tmp_path) -> None:
    hf = _figure_module()
    settings = sh.load_shots_settings()
    hole = _Hole(pin_position=(settings["hole"]["length_m"], 0.0, 0.0))
    shots = _place([_flight(166.0, -11.0), _flight(207.0, 15.0)], hole)
    out = hf.write_hole_figure(shots, hole, tmp_path / "hole.png", settings)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_empty_shot_list_raises(tmp_path) -> None:
    hf = _figure_module()
    with pytest.raises(ValueError):
        hf.write_hole_figure([], _Hole(), tmp_path / "x.png", sh.load_shots_settings())


def test_side_text_is_signed_without_zero_side() -> None:
    hf = _figure_module()
    assert hf.side_text(-11.4, 0) == "11 m left"
    assert hf.side_text(15.0, 0) == "15 m right"
    assert hf.side_text(0.2, 0) == "on line"
