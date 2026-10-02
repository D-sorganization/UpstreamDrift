"""Reusable native projection interpreter isolated from the Qt runtime."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, TextIO

from src.shared.python.security import secure_popen
from src.shared.python.version_info import get_repo_root

PROJECTION_TIMEOUT_S = 45.0
WORKER_MODULE = "src.shared.python.workspace.necromatcher_projection_worker"


def _read_responses(stream: TextIO, responses: Queue[str | None]) -> None:
    try:
        for line in stream:
            responses.put(line)
    finally:
        responses.put(None)


class NativeFitProjectionProcess:
    """Reuse a clean interpreter without multiprocessing's Qt main imports.

    Requests run serially in the canonical background worker. The owned process
    is terminated on timeout; subsequent requests can start a fresh runtime.
    """

    def __init__(self, library_root: Path | str) -> None:
        self._root = str(Path(library_root).resolve())
        self._process: subprocess.Popen[str] | None = None
        self._responses: Queue[str | None] = Queue()
        self._reader: Thread | None = None
        self._lock = Lock()
        self._closed = False

    def _start(self) -> subprocess.Popen[str]:
        process = secure_popen(
            [sys.executable, "-u", "-m", WORKER_MODULE],
            cwd=get_repo_root(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        self._process = process
        self._responses = Queue()
        assert process.stdout is not None
        self._reader = Thread(
            target=_read_responses,
            args=(process.stdout, self._responses),
            daemon=True,
        )
        self._reader.start()
        return process

    def project(self, fit_id: str, frame_index: int) -> dict[str, Any]:
        """Project a verified saved frame; release failed or timed-out runtimes."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Native projection process is closed")
            process = self._process or self._start()
            assert process.stdin is not None
            try:
                process.stdin.write(
                    json.dumps([self._root, fit_id, frame_index]) + "\n"
                )
                process.stdin.flush()
                response = self._responses.get(timeout=PROJECTION_TIMEOUT_S)
                if response is None:
                    self._stop()
                    raise RuntimeError("Native projection runtime exited")
                record = json.loads(response)
            except (OSError, Empty, json.JSONDecodeError) as exc:
                self._stop()
                raise RuntimeError(
                    "Native projection runtime failed or timed out"
                ) from exc
            if "error" in record:
                error_type = {
                    "ValueError": ValueError,
                    "TypeError": ValueError,
                    "KeyError": ValueError,
                    "IndexError": IndexError,
                }.get(record["type"], RuntimeError)
                raise error_type(record["error"])
            return dict(record["result"])

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._reader is not None:
            self._reader.join(timeout=5)
            self._reader = None
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()

    def close(self) -> None:
        """Release the owned interpreter and pipes exactly once."""
        with self._lock:
            self._closed = True
            self._stop()

    def __enter__(self) -> NativeFitProjectionProcess:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
