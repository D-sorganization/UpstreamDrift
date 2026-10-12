"""Tests for the public-explorer putt export (issue #12319, mirror pattern).

The exporter writes a byte-deterministic JSON artifact that the public
explorer's Putting tab vendors verbatim.  Every test round-trips through the
filesystem: write, read back, compare.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tools.putting_green_gui._scene_builder import (
    PuttConfig,
    PuttScene,
    build_putt_scene,
)
from src.tools.putting_green_gui._scene_export import (
    SCHEMA_ID,
    PuttExportError,
    export_putt_scene,
    read_putt_export,
    scene_to_record,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def scene() -> PuttScene:
    return build_putt_scene(PuttConfig(grid_resolution=16))


def _assert_scenes_equal(a: PuttScene, b: PuttScene) -> None:
    for name in (
        "grid_x",
        "grid_y",
        "grid_z",
        "trajectory_xyz",
        "start_xyz",
        "cup_xyz",
        "aim_line_xyz",
        "target_line_xyz",
    ):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert a.roll_modes == b.roll_modes
    assert a.green_size == b.green_size
    assert a.holed is b.holed
    assert a.roll_model == b.roll_model
    for name in (
        "hole_radius_m",
        "final_distance_to_cup_m",
        "total_roll_m",
        "duration_s",
        "peak_break_m",
        "launch_speed_ms",
    ):
        assert getattr(a, name) == getattr(b, name), name


def test_round_trip_write_read_compare(scene: PuttScene, tmp_path: Path) -> None:
    path = tmp_path / "putt_scene.json"
    export_putt_scene(scene, path)
    _assert_scenes_equal(read_putt_export(path), scene)


def test_export_is_byte_deterministic(scene: PuttScene, tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    export_putt_scene(scene, a)
    export_putt_scene(scene, b)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes().endswith(b"\n")


def test_record_declares_schema_units_and_provenance(scene: PuttScene) -> None:
    record = scene_to_record(scene)
    assert record["schema"] == SCHEMA_ID == "ud.putt_scene/1"
    assert record["units"] == {"length": "m", "time": "s", "speed": "m/s"}
    assert record["frame"]["x"] == "down the nominal ball-to-cup line"
    assert record["roll_model"] == scene.roll_model
    assert record["roll_model"]


def test_artifact_is_plain_json_with_expected_shape(
    scene: PuttScene, tmp_path: Path
) -> None:
    path = tmp_path / "x.json"
    export_putt_scene(scene, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    n = len(data["trajectory"]["xyz"])
    assert n == len(data["trajectory"]["roll_modes"]) == scene.trajectory_xyz.shape[0]
    assert len(data["terrain"]["z"]) == scene.grid_y.size
    assert len(data["terrain"]["z"][0]) == scene.grid_x.size


def test_export_creates_parent_directories(scene: PuttScene, tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "putt.json"
    export_putt_scene(scene, path)
    assert path.is_file()


def test_export_rejects_non_finite_values(scene: PuttScene, tmp_path: Path) -> None:
    bad_traj = scene.trajectory_xyz.copy()
    bad_traj[0, 0] = np.nan
    bad = PuttScene(**{**scene.__dict__, "trajectory_xyz": bad_traj})
    with pytest.raises(PuttExportError, match="finite"):
        export_putt_scene(bad, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


def test_export_rejects_inconsistent_scene(scene: PuttScene, tmp_path: Path) -> None:
    bad = PuttScene(**{**scene.__dict__, "roll_modes": scene.roll_modes[:-1]})
    with pytest.raises(PuttExportError, match="roll_modes"):
        export_putt_scene(bad, tmp_path / "bad.json")


def test_export_rejects_empty_roll_model(scene: PuttScene, tmp_path: Path) -> None:
    bad = PuttScene(**{**scene.__dict__, "roll_model": ""})
    with pytest.raises(PuttExportError, match="roll_model"):
        export_putt_scene(bad, tmp_path / "bad.json")


def test_export_rejects_wrong_path_type(scene: PuttScene) -> None:
    with pytest.raises(PuttExportError, match="path"):
        export_putt_scene(scene, "putt.json")  # type: ignore[arg-type]


def test_read_rejects_unknown_schema(scene: PuttScene, tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    export_putt_scene(scene, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema"] = "ud.putt_scene/2"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PuttExportError, match="schema"):
        read_putt_export(path)


def test_read_rejects_missing_field(scene: PuttScene, tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    export_putt_scene(scene, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["terrain"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PuttExportError, match="terrain"):
        read_putt_export(path)


def test_read_rejects_malformed_json_and_missing_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(PuttExportError, match="JSON"):
        read_putt_export(bad)
    with pytest.raises(PuttExportError, match="not found"):
        read_putt_export(tmp_path / "missing.json")


def test_cli_writes_artifact_that_reads_back(tmp_path: Path) -> None:
    from scripts.export_putt_scene import main

    out = tmp_path / "cli.json"
    assert main(["--out", str(out), "--speed", "2.0", "--stimp", "11"]) == 0
    expected = build_putt_scene(PuttConfig(putter_speed_ms=2.0, stimp=11.0))
    _assert_scenes_equal(read_putt_export(out), expected)


def test_cli_rejects_out_of_range_control(tmp_path: Path) -> None:
    from scripts.export_putt_scene import main

    out = tmp_path / "cli.json"
    assert main(["--out", str(out), "--speed", "99"]) == 2
    assert not out.exists()
