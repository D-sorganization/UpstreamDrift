"""Camera look-at, height and distance are settings, not driver literals (#15150)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.native_viewer_export import camera_framing, cli
from src.tools.native_viewer_export.core import ExportSettings
from src.tools.native_viewer_export.runner import ExportJob, run_export
from tests.unit.tools.native_viewer_export.test_runner_cli import (
    _Backend,
    _bundle,
    _Writer,
)

pytestmark = pytest.mark.unit

BASE = ["--bundle", "b.npz", "--out", "out", "--swing", "s"]


def _settings(*extra: str) -> ExportSettings:
    return cli.build_settings(cli.build_parser().parse_args([*BASE, *extra]))


def test_default_settings_are_unchanged() -> None:
    s, d = _settings(), ExportSettings()
    assert (s.lookat_m, s.distance_m) == (d.lookat_m, d.distance_m)
    assert s.lookat_explicit is False and s.lookat_height_m is None


def test_default_height_comes_from_the_settings_file() -> None:
    raw = json.loads(camera_framing.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert camera_framing.default_lookat_height_m() == raw["default_lookat_height_m"]
    assert camera_framing.default_lookat_height_m() == 0.9
    assert "default_lookat_height_m" in raw["sources"]


def test_lookat_distance_and_height_populate_settings() -> None:
    s = _settings("--camera-lookat", "1.2,0.1,0.6", "--camera-distance", "3.5")
    assert s.lookat_m == (1.2, 0.1, 0.6) and s.lookat_explicit
    assert s.distance_m == 3.5
    assert _settings("--camera-height", "0.7").lookat_height_m == 0.7


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--camera-lookat", "1,2"),
        ("--camera-lookat", "1,2,nan"),
        ("--camera-lookat", "a,b,c"),
        ("--camera-height", "inf"),
        ("--camera-distance", "0"),
        ("--camera-distance", "-2"),
        ("--camera-distance", "nan"),
    ],
)
def test_bad_values_are_rejected(flag: str, value: str) -> None:
    with pytest.raises(ValueError, match="camera"):
        _settings(flag, value)


def test_resolve_lookat_precedence() -> None:
    auto = (0.4, 0.2, 0.9)
    assert camera_framing.resolve_lookat(auto, ExportSettings()) == auto
    height = ExportSettings(lookat_height_m=0.5)
    assert camera_framing.resolve_lookat(auto, height) == (0.4, 0.2, 0.5)
    both = ExportSettings(
        lookat_m=(9.0, 8.0, 7.0), lookat_explicit=True, lookat_height_m=0.5
    )
    assert camera_framing.resolve_lookat(auto, both) == (9.0, 8.0, 7.0)


def _run(tmp_path: Path, settings: ExportSettings) -> ExportSettings:
    b = _bundle()
    bpath = tmp_path / "b.npz"
    b.save(bpath)
    backend = _Backend("drake")
    run_export(
        ExportJob(bpath, tmp_path / "o", "d", "Driver", ("drake",)),
        settings,
        lambda e: backend,
        lambda swing, engine: (None, (0.5, 0.0, 0.9)),
        _Writer,
    )
    return backend.swings[0][1]


def _base(**kw: object) -> ExportSettings:
    return ExportSettings(width=32, height=32, fps=20, speeds=(1.0,), **kw)  # type: ignore[arg-type]


def test_runner_default_keeps_automatic_lookat(tmp_path: Path) -> None:
    assert _run(tmp_path, _base()).lookat_m == (0.5, 0.0, 0.9)


def test_runner_height_overrides_only_z(tmp_path: Path) -> None:
    assert _run(tmp_path, _base(lookat_height_m=0.4)).lookat_m == (0.5, 0.0, 0.4)


def test_runner_explicit_lookat_beats_automatic(tmp_path: Path) -> None:
    got = _run(tmp_path, _base(lookat_m=(2.0, 1.0, 0.3), lookat_explicit=True))
    assert got.lookat_m == (2.0, 1.0, 0.3)
