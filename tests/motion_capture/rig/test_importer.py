"""Bundles from existing files (#9659) and typed estimator options (#9661)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.motion_capture.rig import __main__ as rig_cli
from src.motion_capture.rig.bundle import load_bundle
from src.motion_capture.rig.importer import import_videos, parse_view_spec
from src.motion_capture.rig.probe import RecordingProbe

pytestmark = pytest.mark.unit


def _prober(path: Path) -> RecordingProbe:
    return RecordingProbe(240, 4.0, 1280, 720, 60.0)


def test_import_single_file_is_a_valid_single_camera_bundle(tmp_path: Path) -> None:
    video = tmp_path / "face_on.mp4"
    video.write_bytes(b"x" * 100)
    out = tmp_path / "take"
    manifest = import_videos({"face_on": video}, out, prober=_prober)
    plan, index, loaded = load_bundle(out)
    assert manifest.outcome.value != "blocked" and loaded.tools_schema["imported"]
    assert [c.view for c in plan.cameras] == ["face_on"]
    assert plan.cameras[0].identity == "file:face_on.mp4"
    assert plan.cameras[0].mode.model_dump() == {
        "width": 1280,
        "height": 720,
        "fps": 60,
        "fourcc": "FILE",
    }
    entry = index.recordings[0]
    assert entry.ok and Path(entry.file) == video.resolve() and entry.frames == 240
    assert index.duration_s == pytest.approx(4.0)


def test_import_preconditions(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="at least one"):
        import_videos({}, tmp_path / "o", prober=_prober)
    with pytest.raises(Exception, match="must exist"):
        import_videos({"a": tmp_path / "nope.mp4"}, tmp_path / "o", prober=_prober)
    assert parse_view_spec("cam_b=C:/x/y.mp4") == ("cam_b", Path("C:/x/y.mp4"))
    with pytest.raises(Exception, match="NAME=PATH"):
        parse_view_spec("nope")


def test_import_command_builds_multi_view_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.motion_capture.rig.importer as importer

    monkeypatch.setattr(importer, "probe_recording", _prober)
    for name in ("a.avi", "b.avi"):
        (tmp_path / name).write_bytes(b"y" * 10)
    out = tmp_path / "multi"
    code = rig_cli.main(
        [
            "import",
            "--out",
            str(out),
            "--view",
            f"cam_a={tmp_path / 'a.avi'}",
            "--view",
            f"cam_b={tmp_path / 'b.avi'}",
            "--name",
            "bay",
        ]
    )
    assert code == 0
    plan, index, _ = load_bundle(out)
    assert plan.name == "bay" and len(index.recordings) == 2


def test_estimator_options_are_typed() -> None:
    assert rig_cli.parse_options(
        ["min_detection_confidence=0.6", "input_height=368", "smooth=false", "v=full"]
    ) == {
        "min_detection_confidence": 0.6,
        "input_height": 368,
        "smooth": False,
        "v": "full",
    }
    with pytest.raises(SystemExit):
        rig_cli.parse_options(["broken"])
