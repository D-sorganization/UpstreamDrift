"""Swing report package generator (#15724, SRP-5).

The runner executes a planned package: each video through the native export
(one subprocess per job, its own temporary folder, a free-disk guard), then
the grids and phase sheets from those clips, and writes ``manifest.json``.
A rerun skips finished jobs, so a long package resumes after a failure.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.tools.native_viewer_export.frame_times import write_frame_times
from src.tools.swing_report_package import runner
from src.tools.swing_report_package.catalog import (
    PackageRequest,
    load_catalog,
    plan_package,
)
from src.tools.swing_report_package.phases import KeyPhase

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
SMALL = {"size": "64x48"}


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


def _request():
    return PackageRequest(
        include=("case_grid", "phase_sheet"),
        default=False,
        views=("face_on",),
        render_overrides=SMALL,
    )


def _settings():
    from src.tools.swing_report_package.composite import load_composite_settings

    return load_composite_settings().scaled(tile_width_px=64, cell_width_px=64)


def test_a_package_renders_composes_and_records_a_manifest(tmp_path) -> None:
    calls: list[str] = []
    jobs = plan_package(load_catalog(), _request())
    manifest = runner.run_package(
        jobs,
        tmp_path,
        swing="driver",
        render=_fake_render(calls),
        phases_of=_phases,
        composite=_settings(),
        min_free_gb=0.0,
    )
    assert len(calls) == 3  # the three case renders, once each
    doc = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert doc == manifest
    status = {j["key"]: j["status"] for j in doc["jobs"]}
    assert set(status.values()) == {"done"}
    grid = next(j for j in doc["jobs"] if j["item"] == "case_grid")
    assert (tmp_path / grid["outputs"][0]).is_file()
    sheet = next(j for j in doc["jobs"] if j["item"] == "phase_sheet")
    assert any(o.endswith("phase_sheet.png") for o in sheet["outputs"])
    assert [p["id"] for p in doc["phases"]] == ["top", "impact"]


def test_a_rerun_skips_finished_renders(tmp_path) -> None:
    calls: list[str] = []
    jobs = plan_package(load_catalog(), _request())
    kwargs = {
        "swing": "driver",
        "render": _fake_render(calls),
        "phases_of": _phases,
        "composite": _settings(),
        "min_free_gb": 0.0,
    }
    runner.run_package(jobs, tmp_path, **kwargs)
    runner.run_package(jobs, tmp_path, **kwargs)
    assert len(calls) == 3


def test_the_disk_guard_stops_before_a_render(tmp_path) -> None:
    jobs = plan_package(load_catalog(), _request())
    with pytest.raises(runner.DiskSpaceError, match="GB"):
        runner.run_package(
            jobs,
            tmp_path,
            swing="driver",
            render=_fake_render([]),
            phases_of=_phases,
            min_free_gb=1e9,
        )


def test_a_dry_run_lists_the_plan_and_renders_nothing(tmp_path) -> None:
    jobs = plan_package(load_catalog(), _request())
    manifest = runner.run_package(
        jobs, tmp_path, swing="driver", render=None, phases_of=None, dry_run=True
    )
    assert {j["status"] for j in manifest["jobs"]} == {"planned"}
    assert not any(tmp_path.iterdir())


def test_the_exporter_command_carries_the_job_options(tmp_path) -> None:
    (job,) = plan_package(
        load_catalog(),
        PackageRequest(include=("midhands_total",), default=False, views=("face_on",)),
    )
    argv = runner.exporter_command(
        job, tmp_path / "b.npz", tmp_path, "driver", "Driver"
    )
    assert argv[1:3] == ["-m", "src.tools.native_viewer_export"]
    assert argv[argv.index("--bundle") + 1] == str(tmp_path / "b.npz")
    assert argv[argv.index("--ztcf-zvcf-hands") + 1] == "midhands"


def test_the_cli_plans_from_flags(tmp_path, capsys) -> None:
    code = runner.main(
        [
            "--bundle",
            str(tmp_path / "b.npz"),
            "--out",
            str(tmp_path / "pkg"),
            "--include",
            "midhands_total",
            "--no-default",
            "--views",
            "face_on",
            "--set",
            "size=3840x2160",
            "--dry-run",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "midhands_total@face_on@0.25x" in out and "3840x2160" in out


def test_every_catalog_video_parses_in_the_exporter_cli() -> None:
    """A catalog option the exporter does not know would fail hours into a
    package; every video item's command line is parsed here instead."""
    from src.tools.native_viewer_export import cli

    catalog = load_catalog()
    parser = cli.build_parser()
    request = PackageRequest(include=tuple(catalog.items), default=False)
    for job in plan_package(catalog, request):
        if job.kind == "video":
            parser.parse_args(
                ["--bundle", "b", "--out", "o", "--swing", "s", *job.exporter_argv()]
            )
