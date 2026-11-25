"""Swing report package executor (#15724, SRP-5).

Runs a planned package (:func:`~.catalog.plan_package`): each video job is
one ``native_viewer_export`` subprocess with its own temporary folder (wiped
after the run) and a free-disk check first; then the grids and phase sheets
are composed from those clips. ``manifest.json`` records every job, its
status and outputs. A rerun reuses finished clips, so a long package resumes
after a failure. Before any render, every video's options are parsed by the
exporter's own command line: a job it rejects (an option it does not support
yet) is ``unavailable`` and its composites are ``blocked``. Settings live in
``runner_settings.json``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import contextlib
from dataclasses import dataclass, field
import io
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
from src.tools.swing_report_package.catalog import Job

__all__ = [
    "RUNNER_SETTINGS_PATH",
    "DiskSpaceError",
    "exporter_command",
    "job_dir",
    "load_runner_settings",
    "plan_manifest",
    "run_package",
    "subprocess_render",
    "unavailable_jobs",
]

RUNNER_SETTINGS_PATH = Path(__file__).with_name("runner_settings.json")
MANIFEST = "manifest.json"
_HEADLESS_ENV = {
    "MUJOCO_GL": "egl",
    "QT_QPA_PLATFORM": "offscreen",
    "MPLBACKEND": "Agg",
}
Render = Callable[[Job, Path, Path], None]
Preflight = Callable[[Sequence[Job]], Mapping[str, str]]


class DiskSpaceError(RuntimeError):
    """Too little free disk to start a render."""


def load_runner_settings(path: Path | None = None) -> dict[str, Any]:
    """The runner settings; ``ValueError`` when a required key is missing."""
    doc = json.loads((path or RUNNER_SETTINGS_PATH).read_text(encoding="utf-8"))
    missing = [
        k
        for k in ("exporter_module", "min_free_gb", "render_timeout_s")
        if k not in doc
    ]
    if missing:
        raise ValueError(f"runner settings miss {missing}")
    return dict(doc)


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


def unavailable_jobs(jobs: Sequence[Job]) -> dict[str, str]:
    """Video jobs the exporter's command line rejects, by key, with why.

    Postcondition: every other video job's options parse in the exporter.
    """
    from src.tools.native_viewer_export.cli import build_parser

    parser = build_parser()
    reasons: dict[str, str] = {}
    for job in jobs:
        if job.kind != "video":
            continue
        err = io.StringIO()
        argv = ["--bundle", "b", "--out", "o", "--swing", "s", *job.exporter_argv()]
        try:
            with contextlib.redirect_stderr(err):
                parser.parse_args(argv)
        except SystemExit:
            why = err.getvalue().strip().splitlines()[-1:] or ["rejected"]
            reasons[job.key] = f"not available yet: the exporter says {why[0]}"
    return reasons


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
    opts: dict[str, Any] = {"impact_time_s": impact_time_s, "settings": s}

    def render(job: Job, clip: Path, tmp_dir: Path) -> None:
        argv = exporter_command(job, bundle, clip.parent, swing, club, **opts)
        env = {**os.environ, **_HEADLESS_ENV, "TMPDIR": str(tmp_dir)}
        subprocess.run(argv, check=True, env=env, timeout=s["render_timeout_s"])

    return render


def _check_disk(root: Path, min_free_gb: float) -> None:
    free_gb = shutil.disk_usage(root).free / 1e9
    if free_gb < min_free_gb:
        raise DiskSpaceError(
            f"{free_gb:.1f} GB free under {root}; a render needs {min_free_gb} GB"
        )


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

    def render_clip(self, job: Job) -> Path:
        """The clip of video ``job``, rendered unless a finished one exists."""
        if self.render is None:
            raise ValueError(f"{job.key}: no renderer")
        clip = _clip(self.root, job, self.swing)
        if clip.is_file() and frame_times_path(clip).is_file():
            return clip
        clip.parent.mkdir(parents=True, exist_ok=True)
        _check_disk(self.root, self.min_free_gb)
        tmp_root = self.root / ".tmp"
        tmp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            self.render(job, clip, Path(tmp))
        if not clip.is_file():
            raise FileNotFoundError(f"{job.key}: the exporter wrote no {clip.name}")
        return clip


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


def _entry(job: Job, reason: str | None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "key": job.key,
        "item": job.item,
        "kind": job.kind,
        "title": job.title,
        "status": "planned" if reason is None else "unavailable",
        "outputs": [],
    }
    if reason is not None:
        entry["error"] = reason
    return entry


def _merged(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """``manifest`` plus the jobs of an earlier run into the same folder."""
    if not path.is_file():
        return manifest
    old = json.loads(path.read_text(encoding="utf-8"))
    keys = {j["key"] for j in manifest["jobs"]}
    kept = [j for j in old.get("jobs", ()) if j["key"] not in keys]
    phases = manifest["phases"] or old.get("phases", [])
    return {**manifest, "jobs": [*kept, *manifest["jobs"]], "phases": phases}


def plan_manifest(
    jobs: Sequence[Job], *, swing: str, preflight: Preflight | None = None
) -> dict[str, Any]:
    """The manifest of ``jobs`` before any render (a dry run); writes nothing.

    Each job is ``planned``, or ``unavailable`` with the reason ``preflight``
    (default :func:`unavailable_jobs`) gives.
    """
    reasons = (preflight or unavailable_jobs)(jobs)
    entries = [_entry(j, reasons.get(j.key)) for j in jobs]
    return {"swing": swing, "jobs": entries, "phases": []}


def run_package(
    jobs: Sequence[Job],
    root: Path,
    *,
    swing: str,
    render: Render | None,
    phases_of: Callable[[str], Sequence[Any]] | None,
    composite: Any = None,
    min_free_gb: float | None = None,
    preflight: Preflight | None = None,
) -> dict[str, Any]:
    """Run ``jobs`` into ``root`` and write ``root/manifest.json``.

    A video the exporter rejects is ``unavailable``; a job whose input did
    not finish is ``blocked``; a failed render is ``failed`` and the rest
    still run (:func:`plan_manifest` is the dry run). Jobs recorded by an earlier run
    into ``root`` (another view, say) stay in the manifest. ``preflight``
    (default :func:`unavailable_jobs`) names the videos that cannot run.

    Raises:
        DiskSpaceError: less than ``min_free_gb`` free before a render.
    """
    manifest = plan_manifest(jobs, swing=swing, preflight=preflight)
    entries = manifest["jobs"]
    root.mkdir(parents=True, exist_ok=True)
    gb = load_runner_settings()["min_free_gb"] if min_free_gb is None else min_free_gb
    run = _Run(root, swing, render, phases_of, composite, float(gb), manifest)
    run.by_key = {j.key: j for j in jobs}
    try:
        for job, entry in zip(jobs, entries, strict=True):
            if entry["status"] == "planned":
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
            outs = [run.render_clip(job)]
            run.clips[job.key] = outs[0]
        else:
            outs = _compose_job(job, run)
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        entry.update(status="failed", error=str(exc))
        return
    entry.update(status="done", outputs=[str(p.relative_to(run.root)) for p in outs])
