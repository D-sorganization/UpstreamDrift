"""Owner showcase driver (GCV-19, #11725): validation, layout, manifest, smoke."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import render_forces_showcase as rfs

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


def _tiny_bundle(path: Path, steps: int = 40) -> Path:
    import json as _json

    from src.shared.python.motion_matching.same_input import InputBundle

    spec_bytes = SPEC.read_bytes()
    names = tuple(_json.loads(spec_bytes)["coordinate_order"])
    q = np.zeros((steps + 1, len(names)))
    bundle = InputBundle(
        spec_bytes=spec_bytes,
        coordinate_order=names,
        dt_s=0.001,
        q0=q[0],
        v0=q[0],
        efforts=np.zeros((steps, len(names))),
        reference_q=q,
        reference_v=q.copy(),
        reference_engine="mujoco",
    )
    bundle.save(path)
    return path


def test_default_output_dir_is_the_owner_folder() -> None:
    args = rfs.build_parser().parse_args(["--swing", "driver=x.npz"])
    assert args.out == Path("~/Videos/Parity Audit/forces_and_impact/")
    assert rfs.resolve_out_dir(args.out).is_absolute()
    assert "~" not in str(rfs.resolve_out_dir(args.out))


@pytest.mark.parametrize(
    "argv, fragment",
    [
        (["--capture", "capture-Z", "--swing", "d=x.npz"], "capture"),
        (["--swing", "nonsense"], "NAME=PATH"),
        (["--swing", "d=x.npz", "--engines", "bogus"], "unknown engine"),
        (["--swing", "d=x.npz", "--clips", "nope"], "unknown clip"),
        (["--swing", "d=x.npz", "--size", "big"], "size"),
        (["--swing", "d=x.npz", "--speeds", "0"], "speeds"),
    ],
)
def test_invalid_arguments_are_rejected(
    argv: list[str], fragment: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "x.npz").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=fragment):
        rfs.plan_from_args(rfs.build_parser().parse_args(argv))


def test_missing_bundle_fails_clearly(tmp_path: Path) -> None:
    argv = ["--swing", f"d={tmp_path / 'absent.npz'}", "--out", str(tmp_path)]
    with pytest.raises(FileNotFoundError, match="absent.npz"):
        rfs.plan_from_args(rfs.build_parser().parse_args(argv))


def test_main_reports_missing_bundle_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = rfs.main(["--swing", f"d={tmp_path / 'absent.npz'}", "--out", str(tmp_path)])
    assert code == 2
    assert "absent.npz" in capsys.readouterr().err


def test_clip_kinds_follow_the_issue_layout() -> None:
    kinds = rfs.CLIP_KINDS
    assert list(kinds) == ["clean", "club_forces", "grf"]
    assert kinds["clean"].overlays is False and kinds["clean"].grip is False
    assert kinds["club_forces"].grip is True
    assert kinds["club_forces"].views == ("hands_closeup",)
    assert kinds["grf"].views == ("face_on", "down_the_line")
    assert rfs.HQ_SIZE == (1280, 720) and rfs.FPS == 60 and rfs.CRF <= 18


def test_unavailable_engine_fails_unless_skipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "b.npz"
    bundle.write_bytes(b"x")
    monkeypatch.setattr(
        rfs, "engine_unavailable_reason", lambda e: "no such engine here"
    )
    argv = [
        "--swing",
        f"d={bundle}",
        "--out",
        str(tmp_path / "o"),
        "--engines",
        "mujoco",
    ]
    plan = rfs.plan_from_args(rfs.build_parser().parse_args(argv))
    with pytest.raises(rfs.EngineUnavailableError, match="no such engine here"):
        rfs.check_engines(plan)
    assert rfs.main(argv) == 1
    skip = rfs.plan_from_args(
        rfs.build_parser().parse_args([*argv, "--skip-unavailable"])
    )
    assert rfs.check_engines(skip) == {"mujoco": "no such engine here"}


def test_run_lays_out_directories_and_manifest(tmp_path: Path) -> None:
    from src.tools.native_viewer_export.core import ExportResult

    bundle = tmp_path / "b.npz"
    bundle.write_bytes(b"x")
    calls = []

    def fake_runner(job, settings, **_kw):  # noqa: ANN001, ANN202
        calls.append((job, settings))
        job.out_dir.mkdir(parents=True, exist_ok=True)
        clip = job.out_dir / f"{job.swing}_mujoco_face_on_1x.mp4"
        clip.write_bytes(b"mp4")
        return [ExportResult("mujoco", {"face_on_1x": clip}, 3, None, (), {"_1x": 3})]

    out = tmp_path / "o"
    argv = ["--swing", f"driver={bundle}", "--out", str(out), "--engines", "mujoco"]
    plan = rfs.plan_from_args(rfs.build_parser().parse_args(argv))
    manifest_path = rfs.run_showcase(plan, runner=fake_runner)

    for sub in ("clean", "club_forces", "grf"):
        assert (out / sub).is_dir()
    assert {c[0].out_dir.name for c in calls} == {"clean", "club_forces", "grf"}
    by_kind = {c[0].out_dir.name: c[1] for c in calls}
    assert by_kind["clean"].overlays is False
    assert by_kind["club_forces"].grip is True
    assert by_kind["grf"].overlays is True and by_kind["grf"].grip is False
    assert all(
        s.width == 1280 and s.height == 720 and s.fps == 60 for s in by_kind.values()
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == rfs.MANIFEST_SCHEMA
    assert manifest["capture"] == "capture-A"
    assert manifest["swings"] == ["driver"]
    assert manifest["encoding"]["crf"] <= 18
    assert str(tmp_path) not in json.dumps(manifest["swings"])
    clip = manifest["clips"][0]
    assert clip["kind"] in rfs.CLIP_KINDS and clip["status"] == "rendered"
    assert clip["files"] == [f"{clip['kind']}/capture-A_driver_mujoco_face_on_1x.mp4"]


def test_skipped_engine_is_recorded_not_zero(tmp_path: Path) -> None:
    from src.tools.native_viewer_export.core import ExportResult

    bundle = tmp_path / "b.npz"
    bundle.write_bytes(b"x")

    def fake_runner(job, settings, **_kw):  # noqa: ANN001, ANN202
        return [ExportResult("mujoco", {}, 0, skipped_reason="gl missing")]

    argv = ["--swing", f"d={bundle}", "--out", str(tmp_path / "o"), "--clips", "clean"]
    plan = rfs.plan_from_args(rfs.build_parser().parse_args(argv))
    manifest = json.loads(rfs.run_showcase(plan, runner=fake_runner).read_text())
    assert manifest["clips"][0]["status"] == "unavailable"
    assert manifest["clips"][0]["reason"] == "gl missing"
    assert manifest["clips"][0]["files"] == []


def test_smoke_renders_a_tiny_clean_clip(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL", "egl")
    bundle = _tiny_bundle(tmp_path / "b.npz")
    out = tmp_path / "o"
    argv = [
        "--swing", f"driver={bundle}", "--out", str(out), "--engines", "mujoco",
        "--clips", "clean", "--size", "160x120", "--speeds", "1",
        "--no-impact-clip", "--skip-unavailable",
    ]  # fmt: skip
    plan = rfs.plan_from_args(rfs.build_parser().parse_args(argv))
    if "mujoco" in rfs.check_engines(plan):
        pytest.skip("mujoco backend unavailable")
    manifest_path = rfs.run_showcase(plan)
    manifest = json.loads(manifest_path.read_text())
    clip = manifest["clips"][0]
    if clip["status"] == "unavailable":
        pytest.skip(f"no usable GL context: {clip['reason']}")
    assert clip["status"] == "rendered"
    files = [out / f for f in clip["files"]]
    assert files and all(f.is_file() and f.stat().st_size > 0 for f in files)
    assert all(f.parent.name == "clean" for f in files)
