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
from dataclasses import dataclass, field, replace
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


def _render_job(
    job: Job, root: Path, swing: str, render: Render | None, min_gb: float
) -> Path:
    if render is None:
        raise ValueError(f"{job.key}: no renderer")
    clip = _clip(root, job, swing)
    if clip.is_file() and frame_times_path(clip).is_file():
        return clip
    clip.parent.mkdir(parents=True, exist_ok=True)
    _check_disk(root, min_gb)
    tmp_root = root / ".tmp"
    tmp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        render(job, clip, Path(tmp))
    if not clip.is_file():
        raise FileNotFoundError(f"{job.key}: the exporter wrote no {clip.name}")
    return clip


@dataclass
class _Run:
    """State shared by the jobs of one package run."""

    root: Path
    swing: str
    render: Render | None
    phases_of: Callable[[str], Sequence[Any]] | None
    composite: Any
    min_free_gb: float
    manifest: dict[str, Any]
    by_key: dict[str, Job] = field(default_factory=dict)
    clips: dict[str, Path] = field(default_factory=dict)

    def phases(self) -> list[Any]:
        """Key phases, detected once per run (none without a detector)."""
        from src.tools.swing_report_package.phases import KeyPhase

        if self.phases_of is not None and not self.manifest["phases"]:
            found = self.phases_of(self.swing)
            self.manifest["phases"] = [p.to_dict() for p in found]
        return [KeyPhase(**p) for p in self.manifest["phases"]]

    def case_clips(self, job: Job) -> dict[str, Path]:
        """The clips ``job`` is built from, by their ZTCF/ZVCF case."""
        return {
            str(self.by_key[n].args["force_part"]): self.clips[n] for n in job.needs
        }


def _compose_job(job: Job, run: _Run) -> list[Path]:
    from src.tools.swing_report_package import composite as comp

    cases = run.case_clips(job)
    folder = job_dir(run.root, job)
    folder.mkdir(parents=True, exist_ok=True)
    phases = run.phases()
    if job.kind == "grid":
        out = folder / f"{job.item}.mp4"
        info = (job.title, f"{job.view.replace('_', ' ')}, {job.speed:g}x speed")
        comp.compose_grid_video(
            cases, out, run.composite, info_lines=info, phases=phases
        )
        return [out]
    out = folder / f"{job.item}.png"
    return [out, *comp.phase_sheet(cases, phases, out, run.composite)]


def _entry(job: Job) -> dict[str, Any]:
    return {
        "key": job.key,
        "item": job.item,
        "kind": job.kind,
        "title": job.title,
        "status": "planned",
        "outputs": [],
    }


def _merged(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """``manifest`` plus the jobs of an earlier run into the same folder."""
    if not path.is_file():
        return manifest
    old = json.loads(path.read_text(encoding="utf-8"))
    keys = {j["key"] for j in manifest["jobs"]}
    kept = [j for j in old.get("jobs", ()) if j["key"] not in keys]
    phases = manifest["phases"] or old.get("phases", [])
    return {**manifest, "jobs": [*kept, *manifest["jobs"]], "phases": phases}


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
    and the rest still run. A dry run writes nothing. Jobs recorded by an
    earlier run into ``root`` (another view, say) stay in the manifest.

    Raises:
        DiskSpaceError: less than ``min_free_gb`` free before a render.
    """
    entries = [_entry(j) for j in jobs]
    manifest: dict[str, Any] = {"swing": swing, "jobs": entries, "phases": []}
    if dry_run:
        return manifest
    root.mkdir(parents=True, exist_ok=True)
    gb = load_runner_settings()["min_free_gb"] if min_free_gb is None else min_free_gb
    run = _Run(root, swing, render, phases_of, composite, gb, manifest)
    run.by_key = {j.key: j for j in jobs}
    try:
        for job, entry in zip(jobs, entries, strict=True):
            _run_one(job, entry, run)
    finally:
        shutil.rmtree(root / ".tmp", ignore_errors=True)
        manifest = _merged(root / MANIFEST, manifest)
        (root / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def _run_one(job: Job, entry: dict[str, Any], run: _Run) -> None:
    if any(n not in run.clips for n in job.needs):
        entry["status"] = "blocked"
        return
    try:
        if job.kind == "video":
            outs = [_render_job(job, run.root, run.swing, run.render, run.min_free_gb)]
            run.clips[job.key] = outs[0]
        else:
            outs = _compose_job(job, run)
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        entry.update(status="failed", error=str(exc))
        return
    entry.update(status="done", outputs=[str(p.relative_to(run.root)) for p in outs])


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
