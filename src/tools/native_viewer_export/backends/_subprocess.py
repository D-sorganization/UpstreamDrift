"""Run a render worker in a subprocess and read its frames back."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from src.shared.python.force_overlay.glyphs import GlyphSet
from src.tools.native_viewer_export.backends._worker_job import WorkerJob
from src.tools.native_viewer_export.core import (
    ExportSettings,
    Image8,
    OverlayFeed,
    SwingInput,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKER_TIMEOUT_S = 3600


def worker_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Child environment: repo on ``PYTHONPATH``, headless-safe GL settings."""
    env = {**os.environ}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")])
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.update(extra or {})
    return env


def stage_job(
    swing: SwingInput,
    settings: ExportSettings,
    indices: Sequence[int],
    overlay: OverlayFeed | None,
    work: Path,
) -> WorkerJob:
    """Write the bundle, rollout and optional glyph sets a worker reads."""
    bundle_path, q_path = work / "bundle.npz", work / "q.npy"
    swing.bundle.save(bundle_path)
    np.save(q_path, swing.q)
    glyphs_path = None
    if overlay is not None:
        glyphs_path = work / "glyphs.json"
        sets: list[GlyphSet] = [overlay.glyphs_at(k) for k in indices]
        glyphs_path.write_text(
            json.dumps([g.to_dict() for g in sets]), encoding="utf-8"
        )
    frames = work / "frames"
    frames.mkdir()
    return WorkerJob(
        bundle_path=str(bundle_path),
        q_path=str(q_path),
        indices=list(indices),
        views=list(settings.views),
        width=settings.width,
        height=settings.height,
        lookat_m=list(settings.lookat_m),
        distance_m=settings.distance_m,
        out_dir=str(frames),
        glyphs_path=str(glyphs_path) if glyphs_path else None,
    )


def run_worker(
    command: Sequence[str], job: WorkerJob, work: Path, env: dict[str, str]
) -> None:
    """Run ``command`` with the job file; raise ``RuntimeError`` with its log on failure."""
    job_file = work / "job.json"
    job.dump(job_file)
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [*command, str(job_file)],
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_S,
        check=False,
        cwd=REPO_ROOT,
        env=env,
    )
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr)[-2000:]
        raise RuntimeError(f"render worker failed (rc={proc.returncode}):\n{tail}")


def read_frames(job: WorkerJob) -> Iterator[dict[str, Image8]]:
    """Yield the worker's frames in index order."""
    for pos in range(len(job.indices)):
        yield {v: np.load(job.frame_path(v, pos)) for v in job.views}


def render_in_worker(
    command: Sequence[str],
    swing: SwingInput,
    settings: ExportSettings,
    indices: Sequence[int],
    overlay: OverlayFeed | None,
    env: dict[str, str],
) -> Iterator[dict[str, Image8]]:
    """Stage a job, run the worker, then stream its frames."""
    with tempfile.TemporaryDirectory(prefix="native_viewer_") as tmp:
        work = Path(tmp)
        job = stage_job(swing, settings, indices, overlay, work)
        run_worker(command, job, work, env)
        yield from read_frames(job)


_URDF_SNIPPET = (
    "import json, sys\n"
    "from src.engines.physics_engines.drake.python.full_body_urdf import "
    "export_full_body_urdf\n"
    "xml, meta = export_full_body_urdf(sys.stdin.buffer.read())\n"
    "sys.stdout.write(json.dumps({'xml': xml, 'meta': meta}))\n"
)


def export_urdf(spec_bytes: bytes) -> tuple[str, dict[str, object]]:
    """Full-body URDF text and sidecar, built in a child with the repo import paths.

    The legacy ``shared.python`` aliases only resolve when ``src`` leads
    ``PYTHONPATH`` from interpreter start, so the export runs out of process.
    """
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", _URDF_SNIPPET],
        input=spec_bytes,
        capture_output=True,
        timeout=600,
        check=False,
        cwd=REPO_ROOT,
        env=worker_env(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"URDF export failed:\n{proc.stderr.decode()[-2000:]}")
    doc = json.loads(proc.stdout.decode())
    return str(doc["xml"]), dict(doc["meta"])
