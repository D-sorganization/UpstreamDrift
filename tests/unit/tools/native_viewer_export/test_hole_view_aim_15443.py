"""The hole view honours the aim line of the shot-shape views (#15443)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.tools.native_viewer_export import (
    aim_line,
    flight_tracer as tr,
    hole_view as hv,
)
from src.tools.native_viewer_export.core import ClipOverlays
from src.tools.native_viewer_export.rollout import Rollout
from tests.unit.tools.native_viewer_export import test_flight_tracer as base

pytestmark = pytest.mark.unit

CFG = hv.load_hole_view_settings()
AIM = aim_line.load_aim_settings()


def _flight(mode: str, roll: bool = True):
    settings = replace(base.DEFAULTS, hole_view=CFG, aim=replace(AIM, aim_line=mode))
    play = base._playback(settings)
    if roll:
        end = (18.0, -5.0)
        play = replace(
            play,
            rollout=Rollout(19.0, play.carry_m + 19.0, end, "driver", "f", "flat", "x"),
        )
    return tr.with_aim(play, base._flight())


def _rest_left_m(shot) -> float:
    return float(shot.path_m[-1][1] + shot.roll_offset_m[1])


def test_finish_aim_puts_the_finish_on_the_centre_line() -> None:
    pytest.importorskip("PyQt6.QtCore")
    shot = hv.hole_shot(_flight("finish"), CFG)
    assert _rest_left_m(shot) == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(shot.path_m[0], [0.0, 0.0, 0.0])


def test_finish_aim_reports_the_side_against_the_aim_line() -> None:
    pytest.importorskip("PyQt6.QtCore")
    play = _flight("finish")
    shot = hv.hole_shot(play, CFG)
    assert shot.side_m == pytest.approx(play.aim_lateral_m)
    assert shot.carry_m == play.carry_m


def test_address_aim_leaves_the_geometry_unchanged() -> None:
    pytest.importorskip("PyQt6.QtCore")
    play = _flight("address")
    plain = hv.hole_shot(replace(play, aim_yaw_rad=None), CFG)
    shot = hv.hole_shot(play, CFG)
    assert np.allclose(shot.path_m, plain.path_m)
    assert shot.side_m == plain.side_m
    assert shot.roll_offset_m == plain.roll_offset_m


def test_aim_text_comes_from_the_aim_settings() -> None:
    assert hv.aim_note(_flight("finish")) == AIM.finish_text
    assert hv.aim_note(_flight("address")) == AIM.address_text
    assert hv.aim_note(base._playback()) is None


def test_still_names_the_aim_in_the_figure(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("PyQt6.QtCore")
    for mode in ("finish", "address"):
        out = hv.write_still(ClipOverlays(flight=_flight(mode)), tmp_path, mode, "m")
        assert out["hole_view"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_figure_title_carries_the_subtitle() -> None:
    pytest.importorskip("matplotlib")
    from src.tools.golf_environment import hole_figure as hf
    from src.tools.golf_environment.shots import (
        hole_from_settings,
        load_shots_settings,
    )

    settings = load_shots_settings()
    hole = hole_from_settings(settings, CFG.preset)
    shot = hv.hole_shot(_flight("finish"), CFG)
    fig, plan, _ = hf.build_hole_figure([shot], hole, settings, subtitle="Aim: x")
    assert plan.get_title().endswith("Aim: x")
    plain = hf.build_hole_figure([shot], hole, settings)[1]
    assert "Aim" not in plain.get_title()
    with pytest.raises(ValueError, match="subtitle"):
        hf.build_hole_figure([shot], hole, settings, subtitle="")
