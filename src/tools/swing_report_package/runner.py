"""Swing report package generator (#15724, SRP-5).

Runs a planned package (:func:`~.catalog.plan_package`): each video job is
one ``native_viewer_export`` subprocess with its own temporary folder (wiped
after the run) and a free-disk check first; then the grids and phase sheets
are composed from those clips. ``manifest.json`` records every job, its
status and outputs. A rerun reuses finished clips, so a long package resumes
after a failure. Settings live in ``runner_settings.json``.

Usage::

    python3 -m src.tools.swing_report_package --bundle B.npz --out OUT \\
        [--package request.json] [--include ITEM] [--exclude ITEM] \\
        [--no-default] [--views V ...] [--speeds S ...] [--set KEY=VALUE] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from src.shared.python.video_timing.frame_schedule import speed_suffix
from src.tools.native_viewer_export.frame_times import frame_times_path
from src.tools.swing_report_package.catalog import (
    Job,
    PackageRequest,
    load_catalog,
    plan_package,
)

__all__ = [
    "RUNNER_SETTINGS_PATH",
    "DiskSpaceError",
    "bundle_phases",
    "exporter_command",
    "job_dir",
    "load_runner_settings",
    "main",
    "run_package",
    "subprocess_render",
]

RUNNER_SETTINGS_PATH = Path(__file__).with_name("runner_settings.json")
MANIFEST = "manifest.json"
Render = Callable[[Job, Path, Path], None]


class DiskSpaceError(RuntimeError):
    """Too little free disk to start a render."""


def load_runner_settings(path: Path | None = None) -> dict[str, Any]:
    """The runner settings; ``ValueError`` when a required key is missing."""
    doc = json.loads((path or RUNNER_SETTINGS_PATH).read_text(encoding="utf-8"))
    need = ("exporter_module", "min_free_gb", "render_timeout_s", "phase_bodies")
    missing = [k for k in need if k not in doc]
    if missing:
        raise ValueError(f"runner settings miss {missing}")
    return doc


def job_dir(root: Path, job: Job) -> Path:
    """Folder of a job's outputs (a video job: the exporter's ``--out``)."""
    if job.kind == "video":
        return root / job.item / job.view
    return root / job.item / f"{job.view}{speed_suffix(job.speed)}"


def _clip(root: Path, job: Job, swing: str) -> Path:
    engine = str(job.args.get("engines", "mujoco"))
    name = f"{swing}_{engine}_{job.view}{speed_suffix(job.speed)}.mp4"
    return job_dir(root, job) / name


def exporter_command(
    job: Job,
    bundle: Path,
    out_dir: Path,
    swing: str,
    club: str,
    *,
    impact_time_s: float | None = None,
    settings: Mapping[str, Any] | None = None,
) -> list[str]:
    """The ``native_viewer_export`` command line of video ``job``.

    Raises:
        ValueError: ``job`` is not a video job.
    """
    if job.kind != "video":
        raise ValueError(f"{job.key} is a {job.kind}, not a video job")
    module = (settings or load_runner_settings())["exporter_module"]
    argv = [sys.executable, "-m", module, "--bundle", str(bundle)]
    argv += ["--out", str(out_dir), "--swing", swing, "--club", club]
    if impact_time_s is not None:
        argv += ["--impact-time", f"{impact_time_s:.6f}"]
    return argv + job.exporter_argv()


def subprocess_render(
    bundle: Path,
    swing: str,
    club: str,
    *,
    impact_time_s: float | None = None,
    settings: Mapping[str, Any] | None = None,
) -> Render:
    """A render that runs the exporter headless with ``TMPDIR`` = the job's
    temporary folder. Raises ``subprocess.CalledProcessError`` on failure."""
    s = settings or load_runner_settings()

    def render(job: Job, clip: Path, tmp_dir: Path) -> None:
        argv = exporter_command(
            job,
            bundle,
            clip.parent,
            swing,
            club,
            impact_time_s=impact_time_s,
            settings=s,
        )
        env = {
            **os.environ,
            "TMPDIR": str(tmp_dir),
            "MUJOCO_GL": "egl",
            "QT_QPA_PLATFORM": "offscreen",
            "MPLBACKEND": "Agg",
        }
        subprocess.run(argv, check=True, env=env, timeout=s["render_timeout_s"])

    return render


def bundle_phases(
    bundle: Path,
    swing: str,
    club: str,
    *,
    impact_time_s: float | None = None,
    settings: Mapping[str, Any] | None = None,
) -> Callable[[str], tuple[Any, ...]]:
    """Key phases of the bundle's swing (MuJoCo body positions, lazy)."""

    def phases_of(_swing: str) -> tuple[Any, ...]:
        import numpy as np

        from src.engines.physics_engines.mujoco.python.overlay_source import (
            MujocoOverlaySource,
        )
        from src.shared.python.model_appearance.club_face import (
            top_of_backswing_index,
        )
        from src.tools.native_viewer_export.core import load_swing_input
        from src.tools.native_viewer_export.overlay import detect_impact_time_s
        from src.tools.swing_report_package.phases import detect_phases

        bodies = (settings or load_runner_settings())["phase_bodies"]
        sw = load_swing_input(bundle, swing=swing, club=club)
        source = MujocoOverlaySource(sw.bundle.spec_bytes)
        order = sw.bundle.coordinate_order
        rows = [
            source.body_positions_m(
                dict(zip(order, map(float, q), strict=True)), list(bodies.values())
            )
            for q in sw.q
        ]
        points = {k: np.asarray([r[b] for r in rows]) for k, b in bodies.items()}
        t = np.arange(len(sw.q)) * float(sw.bundle.dt_s)
        impact = detect_impact_time_s(sw) if impact_time_s is None else impact_time_s
        top = float(t[top_of_backswing_index(t, points["clubhead"])])
        return detect_phases(t, points, impact, top)

    return phases_of


def _check_disk(root: Path, min_free_gb: float) -> None:
    free_gb = shutil.disk_usage(root).free / 1e9
    if free_gb < min_free_gb:
        raise DiskSpaceError(
            f"{free_gb:.1f} GB free under {root}; a render needs {min_free_gb} GB"
        )


def _render_job(job: Job, root: Path, swing: str, render: Render, min_gb: float):
    clip = _clip(root, job, swing)
    if clip.is_file() and frame_times_path(clip).is_file():
        return "done", [clip]
    clip.parent.mkdir(parents=True, exist_ok=True)
    _check_disk(root, min_gb)
    tmp_root = root / ".tmp"
    tmp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        render(job, clip, Path(tmp))
    if not clip.is_file():
        raise FileNotFoundError(f"{job.key}: the exporter wrote no {clip.name}")
    return "done", [clip]


def _case_clips(job: Job, clips: Mapping[str, Path], by_key: Mapping[str, Job]):
    return {str(by_key[n].args["force_part"]): clips[n] for n in job.needs}


def _compose_job(job, root, clips, by_key, phases, composite):
    from src.tools.swing_report_package import composite as comp

    cases = _case_clips(job, clips, by_key)
    folder = job_dir(root, job)
    folder.mkdir(parents=True, exist_ok=True)
    if job.kind == "grid":
        out = folder / f"{job.item}.mp4"
        info = (job.title, f"{job.view.replace('_', ' ')}, {job.speed:g}x speed")
        comp.compose_grid_video(cases, out, composite, info_lines=info)
        return "done", [out]
    out = folder / f"{job.item}.png"
    stills = comp.phase_sheet(cases, phases, out, composite)
    return "done", [out, *stills]


def run_package(
    jobs: Sequence[Job],
    root: Path,
    *,
    swing: str,
    render: Render | None,
    phases_of: Callable[[str], Sequence[Any]] | None,
    composite: Any = None,
    min_free_gb: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run ``jobs`` into ``root`` and write ``root/manifest.json``.

    A job whose input failed is ``blocked``; a failed render is ``failed``
    and the rest still run. A dry run writes nothing.

    Raises:
        DiskSpaceError: less than ``min_free_gb`` free before a render.
    """
    entries = [
        {
            "key": j.key,
            "item": j.item,
            "kind": j.kind,
            "title": j.title,
            "status": "planned",
            "outputs": [],
        }
        for j in jobs
    ]
    manifest: dict[str, Any] = {"swing": swing, "jobs": entries, "phases": []}
    if dry_run:
        return manifest
    root.mkdir(parents=True, exist_ok=True)
    gb = load_runner_settings()["min_free_gb"] if min_free_gb is None else min_free_gb
    by_key = {j.key: j for j in jobs}
    clips: dict[str, Path] = {}
    try:
        for job, entry in zip(jobs, entries, strict=True):
            _run_one(
                job,
                entry,
                root,
                swing,
                render,
                gb,
                clips,
                by_key,
                phases_of,
                composite,
                manifest,
            )
    finally:
        shutil.rmtree(root / ".tmp", ignore_errors=True)
        (root / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def _run_one(
    job, entry, root, swing, render, gb, clips, by_key, phases_of, composite, manifest
) -> None:
    if any(n not in clips for n in job.needs):
        entry["status"] = "blocked"
        return
    try:
        if job.kind == "video":
            status, outs = _render_job(job, root, swing, render, gb)
            clips[job.key] = outs[0]
        else:
            if job.kind == "phase_sheet" and not manifest["phases"]:
                manifest["phases"] = [p.to_dict() for p in phases_of(swing)]
            phases = [_Phase(**p) for p in manifest["phases"]]
            status, outs = _compose_job(job, root, clips, by_key, phases, composite)
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        entry.update(status="failed", error=str(exc))
        return
    entry.update(status=status, outputs=[str(p.relative_to(root)) for p in outs])


def _Phase(**kw: Any) -> Any:  # noqa: N802 - KeyPhase from its manifest dict
    from src.tools.swing_report_package.phases import KeyPhase

    return KeyPhase(**kw)


def _value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swing_report_package", description=__doc__)
    p.add_argument("--bundle", type=Path, required=True, help="swing bundle .npz")
    p.add_argument("--out", type=Path, required=True, help="package folder")
    p.add_argument("--swing", default="swing", help="swing name in file names")
    p.add_argument("--club", default="Driver", help="club label for the HUD")
    p.add_argument("--impact-time", type=float, default=None, help="impact (s)")
    p.add_argument("--catalog", type=Path, default=None, help="catalog JSON")
    p.add_argument("--package", type=Path, default=None, help="request JSON")
    p.add_argument("--include", action="append", default=[], help="add an item")
    p.add_argument("--exclude", action="append", default=[], help="drop an item")
    p.add_argument("--no-default", action="store_true", help="skip the default set")
    p.add_argument("--views", nargs="+", default=None, help="camera views")
    p.add_argument("--speeds", nargs="+", type=float, default=None, help="speeds")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a render option for every video",
    )
    p.add_argument("--dry-run", action="store_true", help="print the plan only")
    return p


def _request(args: argparse.Namespace) -> PackageRequest:
    base = PackageRequest.from_file(args.package) if args.package else PackageRequest()
    sets = dict(s.split("=", 1) for s in args.set)
    return replace(
        base,
        include=(*base.include, *args.include),
        exclude=(*base.exclude, *args.exclude),
        default=base.default and not args.no_default,
        views=tuple(args.views) if args.views else base.views,
        speeds=tuple(args.speeds) if args.speeds else base.speeds,
        render_overrides={
            **base.render_overrides,
            **{k.replace("-", "_"): _value(v) for k, v in sets.items()},
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Plan (and unless ``--dry-run``, run) a package; 1 when a job failed."""
    args = build_parser().parse_args(argv)
    jobs = plan_package(load_catalog(args.catalog), _request(args))
    if args.dry_run:
        for job in jobs:
            opts = (
                " ".join(job.exporter_argv())
                if job.kind == "video"
                else ("from " + ", ".join(job.needs))
            )
            sys.stdout.write(f"{job.key}  [{job.kind}]  {opts}\n")
        return 0
    common = {"impact_time_s": args.impact_time}
    manifest = run_package(
        jobs,
        args.out,
        swing=args.swing,
        render=subprocess_render(args.bundle, args.swing, args.club, **common),
        phases_of=bundle_phases(args.bundle, args.swing, args.club, **common),
    )
    for entry in manifest["jobs"]:
        sys.stdout.write(f"{entry['status']:8s} {entry['key']}\n")
    return int(any(e["status"] != "done" for e in manifest["jobs"]))
