"""Asynchronous native viewer export jobs, shared by REST and MCP (#15150 G8).

Jobs are ``queued``, ``running``, ``succeeded`` (with the clip manifest) or
``failed``; each render is its own exporter subprocess (``managed_popen``).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from typing import Any
import uuid

from src.shared.python.core.process_safety import managed_popen

logger = logging.getLogger(__name__)
_SETTINGS_PATH = Path(__file__).with_name("native_export_jobs_settings.json")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DONE = ("succeeded", "failed")

ExportRunner = Callable[[list[str]], dict[str, Any]]


def load_job_settings() -> dict[str, Any]:
    """Job settings; raises ``ValueError`` unless every bound is positive."""
    settings = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    for key in ("max_concurrency", "timeout_seconds", "max_retained_jobs"):
        if not settings[key] > 0:
            raise ValueError(f"{key} must be positive, got {settings[key]!r}")
    return settings


def manifest_of(results: Sequence[Any]) -> dict[str, Any]:
    """JSON manifest of ``ExportResult`` rows: clips, frames, grip, flight, replay mode
    and model card."""
    return {
        "engines": [
            {
                "engine": r.engine,
                "skipped_reason": r.skipped_reason,
                "clips": {name: str(path) for name, path in r.paths.items()},
                "frames_by_suffix": dict(r.frames_by_suffix),
                "grip_json": None if r.grip_json is None else str(r.grip_json),
                "flight": r.flight,
                "replay_mode": r.replay_mode,
                "model_card": r.model_card,
            }
            for r in results
        ]
    }


def run_export_subprocess(argv: list[str]) -> dict[str, Any]:
    """Render validated exporter ``argv`` in a subprocess (stdout discarded,
    keeping the MCP stdio wire clean); ``RuntimeError`` if it fails."""
    timeout = float(load_job_settings()["timeout_seconds"])
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.json"
        cmd = [sys.executable, "-m", __name__, "--manifest", str(manifest), *argv]
        pipes = {"stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE}
        try:
            with managed_popen(
                cmd, timeout=timeout, cwd=_REPO_ROOT, text=True, **pipes
            ) as proc:
                _, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"export exceeded {timeout:g} s; stopped") from exc
        if proc.returncode != 0 or not manifest.is_file():
            tail = ((stderr or "").strip().splitlines() or ["no output"])[-1]
            raise RuntimeError(f"exporter exited with code {proc.returncode}: {tail}")
        return json.loads(manifest.read_text(encoding="utf-8"))


class ExportJobQueue:
    """Bounded, thread-safe export jobs keyed by job id."""

    def __init__(self, runner: ExportRunner, *, max_concurrency: int, retain: int):
        if max_concurrency < 1 or retain < 1:
            raise ValueError("max_concurrency and retain must be >= 1")
        self._runner, self._retain = runner, retain
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future[None]] = {}
        self._pool = ThreadPoolExecutor(max_concurrency, "native-export")

    def submit(self, argv: list[str], command_line: str) -> dict[str, Any]:
        """Queue ``argv`` (dropping old finished jobs); the job is ``queued``."""
        job_id = f"export_{uuid.uuid4().hex}"
        job: dict[str, Any] = {"job_id": job_id, "state": "queued"}
        job |= {"command_line": command_line, "manifest": None, "error": None}
        with self._lock:
            done = [k for k, j in self._jobs.items() if j["state"] in _DONE]
            for old in done[: max(0, len(self._jobs) + 1 - self._retain)]:
                del self._jobs[old], self._futures[old]
            self._jobs[job_id] = dict(job)
            self._futures[job_id] = self._pool.submit(self._run, job_id, list(argv))
        return job

    def status(self, job_id: str) -> dict[str, Any]:
        """A copy of the job; raises ``ValueError`` for an unknown id."""
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"unknown export job id {job_id!r}")
            return dict(self._jobs[job_id])

    def wait(self, job_id: str, timeout: float) -> dict[str, Any]:
        """Block up to ``timeout`` seconds for ``job_id``; return its status."""
        with self._lock:
            future = self._futures.get(job_id)
        if future is not None:
            wait_futures([future], timeout=timeout)
        return self.status(job_id)

    def job_ids(self) -> list[str]:
        with self._lock:
            return list(self._jobs)

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(fields)

    def _run(self, job_id: str, argv: list[str]) -> None:
        self._update(job_id, state="running")
        try:
            manifest = self._runner(argv)
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.warning("native export %s failed: %s", job_id, exc)
            self._update(job_id, state="failed", error=str(exc))
            return
        self._update(job_id, state="succeeded", manifest=manifest)


_QUEUE: ExportJobQueue | None = None
_QUEUE_LOCK = threading.Lock()


def _queue() -> ExportJobQueue:
    global _QUEUE
    with _QUEUE_LOCK:
        if _QUEUE is None:
            settings = load_job_settings()
            _QUEUE = ExportJobQueue(
                run_export_subprocess,
                max_concurrency=int(settings["max_concurrency"]),
                retain=int(settings["max_retained_jobs"]),
            )
        return _QUEUE


def shared_queue() -> ExportJobQueue:
    """The process-wide queue REST, MCP and the desktop dialog all submit to."""
    return _queue()


def submit_export(request: dict[str, Any] | None) -> dict[str, Any]:
    """Validate ``request`` (``TypeError``/``ValueError`` before queuing),
    then queue its export and return the queued job."""
    from src.tools.native_viewer_export.request_schema import validate

    checked = validate(request or {})
    return _queue().submit(checked["argv"], checked["command_line"])


def export_status(request: dict[str, Any] | None) -> dict[str, Any]:
    """Status of ``request["job_id"]``; raises ``ValueError`` if unknown."""
    job_id = (request or {}).get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise TypeError("request must carry a non-empty string 'job_id'")
    return _queue().status(job_id)


def main(argv: Sequence[str] | None = None) -> int:
    """Subprocess entry: run the exporter CLI and write the manifest JSON."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    known, export_argv = parser.parse_known_args(argv)
    from src.tools.native_viewer_export import cli

    def _write(results: list[Any]) -> None:
        text = json.dumps(manifest_of(results), default=str)
        known.manifest.write_text(text, encoding="utf-8")

    return cli.main(export_argv, on_results=_write)


if __name__ == "__main__":
    sys.exit(main())
