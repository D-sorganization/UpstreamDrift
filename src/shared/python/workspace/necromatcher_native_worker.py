"""Fixed clean-process transport shared by two reviewed research operations."""

from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Literal
from src.shared.python.motion_matching.jobs import JobCancelledError, ProcessGuard
from src.shared.python.security import secure_popen
from src.shared.python.core import repo_python_environment
from src.shared.python.version_info import get_repo_root

_WORKERS = {
    "refit": "src.shared.python.workspace.necromatcher_fit_worker",
    "hypothesis": "src.shared.python.workspace.necromatcher_hypothesis_worker",
    "impact": "src.shared.python.workspace.necromatcher_impact_worker",
}


def execute_native_research_worker(
    request_path: Path,
    budget: float,
    cancelled: Callable[[], bool],
    *,
    operation: Literal["refit", "hypothesis", "impact"] = "refit",
) -> dict[str, Any]:
    if operation not in _WORKERS:
        raise ValueError("Unknown native research worker operation")
    guard = ProcessGuard()
    env = repo_python_environment(get_repo_root())
    if sys.platform.startswith("linux"):
        env.setdefault("MUJOCO_GL", "osmesa")
    process = secure_popen(
        [
            sys.executable,
            "-u",
            "-m",
            _WORKERS[operation],
            str(request_path),
        ],
        cwd=get_repo_root(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    guard.register(process)
    deadline = time.monotonic() + budget
    try:
        while True:
            if cancelled():
                raise JobCancelledError("Native refit cancelled before publication")
            if time.monotonic() >= deadline:
                raise RuntimeError("Native refit exceeded its wall execution budget")
            try:
                output, errors = process.communicate(
                    timeout=max(0.001, min(0.2, deadline - time.monotonic()))
                )
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise RuntimeError(f"Native refit worker failed: {errors[-2000:]}")
        return dict(json.loads(output))
    finally:
        guard.terminate_all(reason="refit complete or interrupted")
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
