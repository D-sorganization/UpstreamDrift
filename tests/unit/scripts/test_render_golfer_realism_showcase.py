"""Tests for the golfer realism showcase render script (OSV-5, #11731).

Tests CLI parsing, input validation, manifest generation for 3 views x 5 engines x 2 clubs,
handling of unavailable engines, and headless smoke execution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from scripts import render_golfer_realism_showcase as rgrs

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_default_output_dir_is_owner_golfer_realism_folder() -> None:
    args = rgrs.build_parser().parse_args(["--swing", "driver=x.npz"])
    assert args.out == Path("~/Videos/Parity Audit/golfer_realism/")
    resolved = rgrs.resolve_out_dir(args.out)
    assert resolved.is_absolute()
    assert "~" not in str(resolved)


def test_views_and_engine_defaults() -> None:
    assert rgrs.DEFAULT_ENGINES == (
        "mujoco",
        "myosuite",
        "drake",
        "pinocchio",
        "opensim",
    )
    assert rgrs.DEFAULT_VIEWS == ("face_on", "dtl", "hands_closeup")
    assert rgrs.VIEW_CANONICAL["dtl"] == "down_the_line"
    assert rgrs.VIEW_CANONICAL["face_on"] == "face_on"
    assert rgrs.VIEW_CANONICAL["hands_closeup"] == "hands_closeup"
    assert rgrs.FPS == 60
    assert rgrs.HQ_SIZE == (1280, 720)
    assert rgrs.REALISM_FEATURES["club"] is True
    assert rgrs.REALISM_FEATURES["wrapped_grip"] is True
    assert rgrs.REALISM_FEATURES["gaze_stabilized_head"] is True
    assert rgrs.REALISM_FEATURES["neutral_address_toe_out"] is True
    assert rgrs.REALISM_FEATURES["decorative_ball"] is True


@pytest.mark.parametrize(
    "argv, fragment",
    [
        (["--capture", "capture-Z", "--swing", "d=x.npz"], "capture"),
        (["--swing", "malformed"], "NAME=PATH"),
        (["--swing", "d=x.npz", "--engines", "unknown_sim"], "unknown engine"),
        (["--swing", "d=x.npz", "--views", "invalid_view"], "unknown view"),
        (["--swing", "d=x.npz", "--size", "1000"], "size"),
        (["--swing", "d=x.npz", "--speeds", "fast"], "speeds"),
        (["--swing", "d=x.npz", "--fps", "0"], "fps"),
        ([], "at least one swing"),
    ],
)
def test_invalid_arguments_are_rejected(
    argv: list[str], fragment: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "x.npz").write_bytes(b"dummy")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=fragment):
        rgrs.plan_from_args(rgrs.build_parser().parse_args(argv))


def test_duplicate_swing_name_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "b.npz"
    bundle.write_bytes(b"dummy")
    argv = ["--swing", f"driver={bundle}", "--swing", f"driver={bundle}"]
    with pytest.raises(ValueError, match="duplicate swing name"):
        rgrs.plan_from_args(rgrs.build_parser().parse_args(argv))


def test_convenience_flags_driver_and_iron7(tmp_path: Path) -> None:
    b1 = tmp_path / "b1.npz"
    b2 = tmp_path / "b2.npz"
    b1.write_bytes(b"dummy")
    b2.write_bytes(b"dummy")
    argv = ["--driver", str(b1), "--iron7", str(b2)]
    plan = rgrs.plan_from_args(rgrs.build_parser().parse_args(argv))
    assert set(plan.swings.keys()) == {"driver", "7_iron"}
    assert plan.swings["driver"] == b1
    assert plan.swings["7_iron"] == b2


def test_missing_bundle_fails_clearly(tmp_path: Path) -> None:
    argv = ["--swing", f"driver={tmp_path / 'absent.npz'}", "--out", str(tmp_path)]
    with pytest.raises(FileNotFoundError, match="absent.npz"):
        rgrs.plan_from_args(rgrs.build_parser().parse_args(argv))


def test_main_reports_missing_bundle_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = rgrs.main(
        ["--swing", f"driver={tmp_path / 'absent.npz'}", "--out", str(tmp_path)]
    )
    assert code == 2
    assert "absent.npz" in capsys.readouterr().err


def test_unavailable_engine_fails_unless_skipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "b.npz"
    bundle.write_bytes(b"dummy")
    monkeypatch.setattr(
        rgrs, "engine_unavailable_reason", lambda e: "test engine not installed"
    )
    argv = [
        "--swing",
        f"driver={bundle}",
        "--out",
        str(tmp_path / "out"),
        "--engines",
        "mujoco",
    ]
    plan = rgrs.plan_from_args(rgrs.build_parser().parse_args(argv))
    with pytest.raises(rgrs.EngineUnavailableError, match="test engine not installed"):
        rgrs.check_engines(plan)
    assert rgrs.main(argv) == 1

    skip_plan = rgrs.plan_from_args(
        rgrs.build_parser().parse_args([*argv, "--skip-unavailable"])
    )
    assert rgrs.check_engines(skip_plan) == {"mujoco": "test engine not installed"}


def test_manifest_generation_3_views_5_engines_2_clubs(tmp_path: Path) -> None:
    """Validate clip manifest generation across 3 views x 5 engines x 2 clubs (30 clips)."""
    from src.tools.native_viewer_export.core import ExportResult

    driver_bundle = tmp_path / "driver.npz"
    iron_bundle = tmp_path / "7_iron.npz"
    driver_bundle.write_bytes(b"driver_data")
    iron_bundle.write_bytes(b"iron_data")

    calls: list[tuple[Any, Any]] = []

    def fake_runner(job: Any, settings: Any, **_kw: Any) -> list[ExportResult]:
        calls.append((job, settings))
        results = []
        for engine in job.engines:
            paths: dict[str, Path] = {}
            for view in ("face_on", "down_the_line", "hands_closeup"):
                for speed_sfx in ("_1x", "_0p5x"):
                    clip_path = (
                        job.out_dir / f"{job.swing}_{engine}_{view}{speed_sfx}.mp4"
                    )
                    clip_path.write_bytes(b"mp4_content")
                    paths[f"{view}{speed_sfx}"] = clip_path
            results.append(
                ExportResult(
                    engine=engine,
                    paths=paths,
                    frames=100,
                    skipped_reason=None,
                    frames_by_suffix={"_1x": 100, "_0p5x": 200},
                )
            )
        return results

    out_dir = tmp_path / "showcase_out"
    argv = [
        "--swing",
        f"driver={driver_bundle}",
        "--swing",
        f"7_iron={iron_bundle}",
        "--engines",
        "mujoco,myosuite,drake,pinocchio,opensim",
        "--views",
        "face_on,dtl,hands_closeup",
        "--speeds",
        "1,0.5",
        "--fps",
        "60",
        "--out",
        str(out_dir),
    ]
    plan = rgrs.plan_from_args(rgrs.build_parser().parse_args(argv))
    manifest_path = rgrs.run_showcase(plan, runner=fake_runner)

    assert manifest_path.is_file()
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert doc["schema"] == "golfer-realism-showcase-manifest/v1"
    assert doc["capture"] == "capture-A"
    assert doc["swings"] == ["driver", "7_iron"]
    assert doc["engines"] == ["mujoco", "myosuite", "drake", "pinocchio", "opensim"]
    assert doc["views"] == ["face_on", "dtl", "hands_closeup"]
    assert doc["speeds"] == [1.0, 0.5]
    assert doc["fps"] == 60
    assert doc["encoding"]["fps"] == 60
    assert doc["encoding"]["width"] == 1280
    assert doc["encoding"]["height"] == 720
    assert doc["realism_features"]["club"] is True
    assert doc["realism_features"]["wrapped_grip"] is True
    assert doc["realism_features"]["gaze_stabilized_head"] is True
    assert doc["realism_features"]["neutral_address_toe_out"] is True
    assert doc["realism_features"]["decorative_ball"] is True

    # 3 views x 5 engines x 2 clubs = 30 clips
    assert len(doc["clips"]) == 30
    assert doc["summary"]["total_clips"] == 30
    assert doc["summary"]["rendered"] == 30
    assert doc["summary"]["unavailable"] == 0

    for clip in doc["clips"]:
        assert clip["status"] == "rendered"
        assert clip["reason"] is None
        assert clip["fps"] == 60
        assert clip["speeds"] == [1.0, 0.5]
        assert len(clip["files"]) == 2  # 1x and 0.5x
        assert any("_1x" in f for f in clip["files"])
        assert any("_0p5x" in f for f in clip["files"])
        assert clip["swing"] in ("driver", "7_iron")
        assert clip["engine"] in doc["engines"]
        assert clip["view"] in ("face_on", "dtl", "hands_closeup")

    # Output subdirectories exist
    assert (out_dir / "driver").is_dir()
    assert (out_dir / "7_iron").is_dir()


def test_manifest_records_unavailable_engines_cleanly(tmp_path: Path) -> None:
    bundle = tmp_path / "driver.npz"
    bundle.write_bytes(b"driver_data")

    def fake_runner(_job: Any, _settings: Any, **_kw: Any) -> list[Any]:
        return []

    out_dir = tmp_path / "showcase_out_unavail"
    plan = rgrs.ShowcasePlan(
        capture="capture-A",
        swings={"driver": bundle},
        engines=("mujoco", "opensim"),
        views=("face_on", "dtl", "hands_closeup"),
        out_dir=out_dir,
        size=(1280, 720),
        speeds=(1.0, 0.5),
        fps=60,
        skip_unavailable=True,
    )
    unavailable = {"opensim": "simbody visualizer missing xvfb"}
    manifest_path = rgrs.run_showcase(plan, runner=fake_runner, unavailable=unavailable)

    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    unavail_clips = [c for c in doc["clips"] if c["engine"] == "opensim"]
    assert len(unavail_clips) == 3  # 3 views for opensim driver
    for c in unavail_clips:
        assert c["status"] == "unavailable"
        assert c["reason"] == "simbody visualizer missing xvfb"
        assert c["files"] == []


def test_headless_smoke_execution(tmp_path: Path) -> None:
    """Smoke test ensuring headless environment variables and clean CLI execution."""
    assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"
    assert os.environ.get("MPLBACKEND") == "Agg"

    bundle = tmp_path / "driver.npz"
    bundle.write_bytes(b"dummy")

    def fake_runner(job: Any, _settings: Any, **_kw: Any) -> list[Any]:
        from src.tools.native_viewer_export.core import ExportResult

        paths = {}
        for v in ("face_on", "down_the_line", "hands_closeup"):
            p = job.out_dir / f"{job.swing}_mujoco_{v}_1x.mp4"
            p.write_bytes(b"video")
            paths[f"{v}_1x"] = p
        return [ExportResult("mujoco", paths, 10, None, (), {"_1x": 10})]

    out_dir = tmp_path / "smoke_out"
    plan = rgrs.ShowcasePlan(
        capture="capture-A",
        swings={"driver": bundle},
        engines=("mujoco",),
        views=("face_on", "dtl", "hands_closeup"),
        out_dir=out_dir,
        size=(640, 360),
        speeds=(1.0,),
        fps=60,
    )
    manifest = rgrs.run_showcase(plan, runner=fake_runner)
    assert manifest.is_file()
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    assert doc["summary"]["rendered"] == 3
