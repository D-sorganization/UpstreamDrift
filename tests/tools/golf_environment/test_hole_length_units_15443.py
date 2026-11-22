"""Hole length presets and display units of the shots figure (#15443)."""

from __future__ import annotations

import json
from types import SimpleNamespace
import pytest

from src.tools.golf_environment import shots as sh
from src.tools.native_viewer_export.display_units import METRIC, load_display_units
from tests.tools.golf_environment.test_shots import _flight, _Hole, _place

pytestmark = pytest.mark.unit


def _qt_hole(**kwargs):
    try:
        return sh.hole_from_settings(sh.load_shots_settings(), **kwargs)
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")


def test_par3_preset_exists_and_is_shorter_than_par4() -> None:
    settings = sh.load_shots_settings()
    presets = settings["hole"]["presets"]
    assert {"par3", "par4", "par5"} <= set(presets)
    assert presets["par3"]["length_m"] < presets["par4"]["length_m"]
    assert presets["par3"]["par"] == 3


def test_hole_from_settings_preset_and_length_override() -> None:
    presets = sh.load_shots_settings()["hole"]["presets"]
    assert _qt_hole(preset="par3").pin_position[0] == presets["par3"]["length_m"]
    assert _qt_hole().pin_position[0] == presets["par4"]["length_m"]
    assert _qt_hole(preset="par3", length_m=200.0).pin_position[0] == 200.0
    with pytest.raises(ValueError):
        _qt_hole(preset="par9")
    with pytest.raises(ValueError):
        _qt_hole(length_m=-1.0)


def test_cli_hole_length_is_in_display_unit_and_moves_green() -> None:
    from src.tools.golf_environment import shots_cli

    args = shots_cli.build_parser().parse_args(
        ["--bundle", "b", "--swing", "s", "--club", "c", "--out", "o"]
        + ["--hole", "par3", "--hole-length", "100", "--distance-unit", "yd"]
    )
    units = shots_cli.resolve_units(args.distance_unit)
    _qt_hole()
    hole = shots_cli.resolve_hole(args, sh.load_shots_settings(), units)
    assert hole.pin_position[0] == pytest.approx(91.44)
    assert shots_cli.resolve_units(None).distance_unit == "yd"
    assert shots_cli.resolve_units("m").distance_unit == "m"
    with pytest.raises(SystemExit):
        shots_cli.build_parser().parse_args(["--distance-unit", "ft"])


def _hole_figure():
    pytest.importorskip("matplotlib")
    from src.tools.golf_environment import hole_figure

    return hole_figure


def test_figure_labels_default_to_yards_and_m_restores_metres(tmp_path) -> None:
    hf = _hole_figure()
    yd = load_display_units()
    (shot,) = _place([_flight(140.0, -11.0)])
    assert "carry 153 yd, 12 yd left" in hf._legend_label(shot, yd)
    assert "carry 140 m, 11 m left" in hf._legend_label(shot, METRIC)
    assert all("(yd)" in label for label in hf.axis_labels(yd))
    assert all("(m)" in label for label in hf.axis_labels(METRIC))
    path = hf.write_hole_figure(
        [shot],
        _Hole(pin_position=(160.0, 0.0, 0.0)),
        tmp_path / "h.png",
        sh.load_shots_settings(),
        yd,
    )
    assert path.stat().st_size > 1000


def test_shots_json_keeps_si_and_names_units() -> None:
    from src.tools.golf_environment import shots_cli

    row = {"carry_m": 143.1, "side_m_right_positive": 3.0}
    hole = SimpleNamespace(name="H", par=4, **vars(_Hole()))
    doc = shots_cli.shots_table(
        ["A"], [(None, row)], hole, load_display_units(), sh.load_shots_settings()
    )
    assert doc["shots"][0]["carry_m"] == 143.1
    assert doc["units"]["distance_values"] == "m"
    assert doc["units"]["display_distance_unit"] == "yd"
    assert doc["hole"]["length_m"] == 380.0
    json.dumps(doc)
