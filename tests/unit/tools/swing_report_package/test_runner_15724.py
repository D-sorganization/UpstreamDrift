"""Swing report package executor (#15724, SRP-5).

Each video renders in its own temporary folder behind a free-disk guard, then
grids and sheets are composed and ``manifest.json`` written; a rerun resumes.
A video the exporter does not accept yet is ``unavailable`` before any render.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.tools.native_viewer_export.frame_times import write_frame_times
from src.tools.swing_report_package import runner
from src.tools.swing_report_package.catalog import (
    Job,
    PackageRequest,
    load_catalog,
    plan_package,
)
from src.tools.swing_report_package.phases import KeyPhase

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
SMALL = {"size": "64x48"}
MIDHANDS = {"midhands_ztcf", "midhands_zvcf", "midhands_total"}


def _fake_render(calls):
    """A stand-in exporter that writes a tiny clip and its frame times."""
    imageio = pytest.importorskip("imageio.v2")

    def render(job, clip, tmp_dir):
        assert tmp_dir.is_dir() and clip.parent.is_dir()
        calls.append(job.key)
        with imageio.get_writer(str(clip), fps=10, macro_block_size=8) as w:
            for k in range(4):
                w.append_data(np.full((48, 64, 3), 40 * k, np.uint8))
        write_frame_times({"x": clip}, [0.0, 0.1, 0.2, 0.3])

    return render


def _phases(_swing):
    return (KeyPhase("top", "Top", 1, 0.1), KeyPhase("impact", "Impact", 2, 0.2))


def _jobs(*items, view="face_on"):
    request = PackageRequest(
        include=items, default=False, views=(view,), render_overrides=SMALL
    )
    return plan_package(load_catalog(), request)


def _kwargs(calls):
    from src.tools.swing_report_package.composite import load_composite_settings

    composite = load_composite_settings().scaled(tile_width_px=64, cell_width_px=64)
    return {
        "swing": "driver",
        "render": _fake_render(calls),
        "phases_of": _phases,
        "composite": composite,
        "min_free_gb": 0.0,
        "preflight": lambda jobs: {},  # the fake renderer takes every option
    }


def test_a_package_renders_composes_and_records_a_manifest(tmp_path) -> None:
    calls: list[str] = []
    jobs = _jobs("case_grid", "phase_sheet")
    manifest = runner.run_package(jobs, tmp_path, **_kwargs(calls))
    assert len(calls) == 3  # the three case renders, once each
    doc = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert doc == manifest
    assert {j["status"] for j in doc["jobs"]} == {"done"}
    grid = next(j for j in doc["jobs"] if j["item"] == "case_grid")
    assert (tmp_path / grid["outputs"][0]).is_file()
    sheet = next(j for j in doc["jobs"] if j["item"] == "phase_sheet")
    assert any(o.endswith("phase_sheet.png") for o in sheet["outputs"])
    assert [p["id"] for p in doc["phases"]] == ["top", "impact"]
    assert not (tmp_path / ".tmp").exists()


def test_a_rerun_skips_finished_renders(tmp_path) -> None:
    calls: list[str] = []
    jobs = _jobs("case_grid", "phase_sheet")
    runner.run_package(jobs, tmp_path, **_kwargs(calls))
    runner.run_package(jobs, tmp_path, **_kwargs(calls))
    assert len(calls) == 3


def test_the_disk_guard_stops_before_a_render(tmp_path) -> None:
    kwargs = {**_kwargs([]), "min_free_gb": 1e9}
    with pytest.raises(runner.DiskSpaceError, match="GB"):
        runner.run_package(_jobs("case_grid"), tmp_path, **kwargs)


def test_a_dry_run_lists_the_plan_and_renders_nothing(tmp_path) -> None:
    manifest = runner.plan_manifest(
        _jobs("case_grid"), swing="driver", preflight=lambda jobs: {}
    )
    assert {j["status"] for j in manifest["jobs"]} == {"planned"}
    assert not any(tmp_path.iterdir())


def test_the_exporter_command_carries_the_job_options(tmp_path) -> None:
    (job,) = _jobs("case_total")
    argv = runner.exporter_command(
        job, tmp_path / "b.npz", tmp_path, "driver", "Driver"
    )
    assert argv[1:3] == ["-m", "src.tools.native_viewer_export"]
    assert argv[argv.index("--bundle") + 1] == str(tmp_path / "b.npz")
    assert argv[argv.index("--ztcf-zvcf-hands") + 1] == "both"
    assert argv[argv.index("--size") + 1] == "64x48"


def test_an_unsupported_video_is_unavailable_and_blocks_its_composites(
    tmp_path,
) -> None:
    """A video whose options the exporter rejects fails closed before any
    render instead of hours into a package, and blocks what composes it."""
    calls: list[str] = []
    kwargs = {**_kwargs(calls), "preflight": None}
    args = {"ztcf_zvcf_view": "total", "ztcf_zvcf_hands": "elbows", "no_grid": True}
    video = Job("elbows", "video", "Elbows", "face_on", 0.25, args)
    grid = Job("elbows_grid", "grid", "Grid", "face_on", 0.25, needs=(video.key,))
    manifest = runner.run_package([video, grid], tmp_path, **kwargs)
    status = {j["item"]: j for j in manifest["jobs"]}
    assert status["elbows"]["status"] == "unavailable"
    assert "not available yet" in status["elbows"]["error"]
    assert "elbows" in status["elbows"]["error"]
    assert status["elbows_grid"]["status"] == "blocked"
    assert calls == []


def test_the_midhands_videos_pass_the_preflight() -> None:
    """The midhands exporter (#15720) accepts the catalog's midhands videos."""
    jobs = [j for j in _jobs("midhands_grid") if j.item in MIDHANDS]
    assert jobs
    assert runner.unavailable_jobs(jobs) == {}


def test_a_video_the_exporter_accepts_passes_the_preflight() -> None:
    args = {"ztcf_zvcf_view": "total", "ztcf_zvcf_hands": "both", "no_grid": True}
    job = Job("x", "video", "X", "face_on", 0.25, args)
    assert runner.unavailable_jobs([job]) == {}
    bad = Job("y", "video", "Y", "face_on", 0.25, {"no_such_option": 1})
    assert "--no-such-option" in runner.unavailable_jobs([bad])["y@face_on@0.25x"]


def test_a_second_run_into_the_folder_keeps_the_first_runs_jobs(tmp_path) -> None:
    """Rendering one view per run (to bound temporary disk) must not drop
    the earlier view from the manifest."""
    for view in ("face_on", "down_the_line"):
        manifest = runner.run_package(
            _jobs("case_grid", view=view), tmp_path, **_kwargs([])
        )
    views = {j["key"].split("@")[1] for j in manifest["jobs"]}
    assert views == {"face_on", "down_the_line"}
    assert len(manifest["jobs"]) == 8


def test_a_failed_render_reports_the_exporters_own_error(tmp_path, monkeypatch) -> None:
    """The manifest names why the exporter stopped, not only its exit status,
    and the full stderr is kept beside the clip."""
    import sys

    script = (
        "import sys; sys.stderr.write('INFO noise\\n"
        "error: ZTCF/ZVCF shading needs a dynamics bundle with efforts\\n');"
        " sys.exit(2)"
    )
    monkeypatch.setattr(
        runner, "exporter_command", lambda *a, **k: [sys.executable, "-c", script]
    )
    render = runner.subprocess_render(tmp_path / "b.npz", "driver", "Driver")
    (job,) = _jobs("case_total")
    clip = tmp_path / "out" / "clip.mp4"
    clip.parent.mkdir()
    with pytest.raises(runner.RenderError, match="needs a dynamics bundle") as err:
        render(job, clip, tmp_path)
    assert "exit 2" in str(err.value)
    log = clip.parent / runner.load_runner_settings()["render_log"]
    assert "INFO noise" in log.read_text(encoding="utf-8")
    assert issubclass(runner.RenderError, runner.subprocess.SubprocessError)
