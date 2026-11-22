"""Opt-in overhead hole view of a native export flight (#15150, track A4)."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.tools.native_viewer_export import cli, flight_tracer as tr, hole_view as hv
from src.tools.native_viewer_export.core import ClipOverlays
from tests.unit.tools.native_viewer_export import test_flight_tracer as base

pytestmark = pytest.mark.unit

CFG = hv.load_hole_view_settings()


def _overlays(valid: bool = True, settings=None) -> ClipOverlays:
    flight = base._playback(settings or replace(base.DEFAULTS, hole_view=CFG), valid)
    return ClipOverlays(flight=flight)


def test_settings_are_documented_and_validated(tmp_path: Path) -> None:
    raw = json.loads(hv.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert set(raw["sources"]) == set(raw) - {"sources"}
    assert CFG.preset == "par4"
    with pytest.raises(ValueError, match="preset"):
        hv.load_hole_view_settings(preset="par9")
    with pytest.raises(ValueError, match="label"):
        replace(CFG, label="")
    raw["figure"]["dpi"] = 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="dpi"):
        hv.load_hole_view_settings(bad)


def test_flight_in_hole_coordinates_keeps_carry_and_side() -> None:
    pytest.importorskip("PyQt6.QtCore")
    flight = base._playback()
    shot = hv.hole_shot(flight, CFG)
    assert shot.carry_m == flight.carry_m and shot.side_m == flight.lateral_m
    assert np.allclose(shot.path_m[0], [0.0, 0.0, 0.0])
    assert shot.path_m[-1][0] == pytest.approx(160.0, abs=1.0)
    assert shot.path_m[-1][1] < 0.0  # right of the line is -left


def test_hole_shot_rejects_a_flight_without_a_path() -> None:
    with pytest.raises(ValueError, match="no drawable flight"):
        hv.hole_shot(base._playback(valid=False), CFG)


def test_still_is_written_only_when_enabled_and_drawable(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("PyQt6.QtCore")
    assert hv.write_still(None, tmp_path, "s", "mujoco") == {}
    off = _overlays(settings=base.DEFAULTS)
    assert hv.write_still(off, tmp_path, "s", "mujoco") == {}
    assert hv.write_still(_overlays(valid=False), tmp_path, "s", "mujoco") == {}
    written = hv.write_still(_overlays(), tmp_path, "s", "mujoco")
    png = written["hole_view"]
    assert png.name == "s_mujoco_hole_view.png"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_flag_is_off_by_default_and_turns_the_flight_on() -> None:
    parser = cli.build_parser()
    required = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]
    assert cli.build_settings(parser.parse_args(required)).ball_flight is None
    args = parser.parse_args([*required, "--hole-view", "--hole-preset", "par5"])
    flight = cli.build_settings(args).ball_flight
    assert isinstance(flight, tr.TracerSettings)
    assert flight.hole_view is not None and flight.hole_view.preset == "par5"
