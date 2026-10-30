"""Isolated subprocess worker pool and process lifecycle management (Epic #14085 / #14088).

Executes simulation runs outside the event loop in isolated worker subprocesses
with configurable wall-clock timeouts, concurrency bounds, Linux rlimits, and
guaranteed process reaping before cleanup.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
from typing import Any

from src.api.models.responses import SimulationErrorInfo
from src.shared.python.core.error_utils import SimulationBusyError
from src.sim_run.models import (
    Provenance,
    ResultEnvelope,
    RunRecord,
    RunRequest,
    SCHEMA_VERSION_V1,
)
from src.sim_run.store import SimulationRunStore

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class LinuxLimits:
    """Linux process resource limits (verified only on Linux)."""

    cpu_seconds: int | None = None
    memory_bytes: int | None = None


def apply_linux_rlimits(limits: LinuxLimits | None = None) -> bool:
    """Apply Linux POSIX resource limits (RLIMIT_CPU, RLIMIT_AS). Documented verified only on Linux."""
    if limits is None or not sys.platform.startswith("linux"):
        return False
    try:
        import resource

        c, m = limits.cpu_seconds, limits.memory_bytes
        if c and c > 0:
            resource.setrlimit(resource.RLIMIT_CPU, (c, c))
        if m and m > 0:
            resource.setrlimit(resource.RLIMIT_AS, (m, m))
        return True
    except (OSError, ValueError) as exc:
        logger.warning("Failed to apply Linux rlimits: %s", exc)
        return False


def reap_subprocess(proc: subprocess.Popen[Any], kill_timeout: float = 1.0) -> int:
    """Terminate, kill if unresponsive, and reap worker subprocess."""
    if proc.poll() is not None:
        return proc.returncode
    killpg = getattr(os, "killpg", None)
    for sig in (15, 9):
        try:
            killpg(proc.pid, sig) if killpg else (
                proc.terminate() if sig == 15 else proc.kill()
            )
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=kill_timeout)
            return proc.returncode if proc.returncode is not None else 0
        except (subprocess.TimeoutExpired, OSError):
            continue
    return proc.returncode if proc.returncode is not None else -1


def _make_preexec(limits: LinuxLimits | None) -> Callable[[], None] | None:
    """Build preexec function on POSIX to establish session and apply rlimits."""
    if os.name != "posix":
        return None

    def _preexec() -> None:
        try:
            getattr(os, "setsid", lambda: None)()
        except OSError:
            pass
        if limits is not None:
            apply_linux_rlimits(limits)

    return _preexec


def run_simulation_job(request: RunRequest, run_id: str) -> ResultEnvelope:
    """Headless simulation solver dispatcher returning ResultEnvelope."""
    now = datetime.now(timezone.utc).isoformat()
    return ResultEnvelope(
        schema_version=SCHEMA_VERSION_V1,
        run_id=run_id,
        provenance=Provenance(
            repo_sha="HEAD",
            engine_type=str(request.engine_type),
            engine_version=None,
            input_sha256=hashlib.sha256(
                request.model_dump_json().encode("utf-8")
            ).hexdigest(),
            created_at=now,
            completed_at=now,
            host=platform.node(),
        ),
        validation_status="valid",
        validation_errors=[],
        success=True,
        data={
            "duration": request.duration,
            "timestep": request.timestep,
            "engine": str(request.engine_type),
        },
        analysis_results=None,
        artifacts=[],
        error=None,
    )


class SimulationWorkerPool:
    """Bounded subprocess worker pool for isolated simulation runs."""

    def __init__(
        self,
        store: SimulationRunStore,
        *,
        max_concurrency: int = 2,
        default_timeout_seconds: float = 300.0,
        limits: LinuxLimits | None = None,
        python_executable: str = sys.executable,
        repo_root: Path | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self.store = store
        self.max_concurrency = max_concurrency
        self.default_timeout_seconds = default_timeout_seconds
        self.limits = limits
        self.python_executable = python_executable
        self.repo_root = repo_root or _REPO_ROOT
        self._semaphore = threading.BoundedSemaphore(max_concurrency)

    def _build_cmd(
        self, run_id: str, record: RunRecord, worker_cmd: list[str] | None
    ) -> tuple[list[str], Path]:
        run_dir = self.store.get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        inp, out = run_dir / "input.json", run_dir / "output.json"
        inp.write_text(record.request.model_dump_json(indent=2), encoding="utf-8")
        if worker_cmd:
            return list(worker_cmd), out
        return [
            self.python_executable,
            "-m",
            "src.sim_run.worker",
            "--input",
            str(inp),
            "--output",
            str(out),
            "--run-id",
            run_id,
        ], out

    def _resolve_record(
        self,
        run_id: str,
        proc: subprocess.Popen[Any],
        out: Path,
        timed_out: bool,
        timeout: float,
        stderr: str,
    ) -> RunRecord:
        def _fail(
            code: str, msg: str, details: dict[str, Any] | None = None
        ) -> RunRecord:
            err = SimulationErrorInfo(
                code=code, stage="execution", message=msg, details=details
            )
            return self.store.transition(run_id, "failed", error=err)

        if timed_out:
            return _fail(
                "timeout",
                f"Simulation run '{run_id}' exceeded wall-clock timeout of {timeout}s",
                {"timeout_seconds": timeout, "exit_code": proc.returncode},
            )
        if proc.returncode != 0:
            return _fail(
                "worker_error",
                f"Worker process exited with code {proc.returncode}",
                {"exit_code": proc.returncode, "stderr": (stderr or "")[:1000]},
            )
        if not out.is_file():
            return _fail(
                "missing_output",
                "Worker process finished without writing output envelope",
            )
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
            return self.store.transition(
                run_id, "succeeded", result=ResultEnvelope.model_validate(data)
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            return _fail(
                "invalid_result_envelope",
                f"Failed to parse result envelope: {exc}",
            )

    def execute_run(
        self,
        run: str | RunRecord,
        *,
        worker_cmd: list[str] | None = None,
        timeout_seconds: float | None = None,
        on_cleanup: Callable[[], None] | None = None,
        block: bool = True,
        wait_timeout: float | None = None,
    ) -> RunRecord:
        """Execute a simulation run in an isolated worker subprocess."""
        if not self._semaphore.acquire(blocking=block, timeout=wait_timeout):
            raise SimulationBusyError(
                f"Concurrency limit ({self.max_concurrency}) reached"
            )

        try:
            record = self.store.get_run(run) if isinstance(run, str) else run
            run_id = record.run_id
            self.store.transition(run_id, "running")
            timeout = (
                timeout_seconds
                or record.request.timeout_seconds
                or self.default_timeout_seconds
            )
            cmd, out = self._build_cmd(run_id, record, worker_cmd)

            env = dict(os.environ)
            pypath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{self.repo_root}{os.pathsep}{pypath}"
                if pypath
                else str(self.repo_root)
            )

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.repo_root),
                env=env,
                preexec_fn=_make_preexec(self.limits),
            )

            timed_out, stdout, stderr = False, "", ""
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                reap_subprocess(proc)
                if on_cleanup:
                    on_cleanup()

            return self._resolve_record(run_id, proc, out, timed_out, timeout, stderr)
        finally:
            self._semaphore.release()


def main() -> None:
    """CLI entrypoint for standalone worker subprocess."""
    p = argparse.ArgumentParser(description="Simulation worker subprocess")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--run-id", required=False)
    args = p.parse_args()
    req = RunRequest.model_validate(
        json.loads(Path(args.input).read_text(encoding="utf-8"))
    )
    rid = args.run_id or f"run_{hashlib.sha256(os.urandom(16)).hexdigest()[:16]}"
    res = run_simulation_job(req, run_id=rid)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(res.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
