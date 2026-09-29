"""Asynchronous Qt adapter for the isolated canonical reference provider."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QWidget

from src.motion_capture.rig.documents import write_document

from ..process_runner import RigProcessRunner

MAX_MESSAGE_BYTES = 4 * 1024 * 1024
SCHEMA = "capture-reference-worker/1"


class ReferenceWorkerClient(QObject):
    """One bounded request at a time, using the existing QProcess lifecycle.

    The GUI never imports the provider's Sidekick family. Request/response files
    belong to this adapter's temporary directory and are read only after exit.
    A dialog must call cancel when closing while a request is active.
    """

    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)
    activity = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget, *, executable: str | None = None) -> None:
        super().__init__(parent)
        self.runner = RigProcessRunner(parent)
        self.runner.hide()
        self.runner.output.connect(self.activity.emit)
        self.runner.finished.connect(self._finished)
        self._executable = executable or sys.executable
        self._directory: TemporaryDirectory[str] | None = None
        self._cancelled = False

    @property
    def busy(self) -> bool:
        return self._directory is not None

    def request(self, payload: dict[str, Any]) -> None:
        """Start work without waiting or touching camera hardware in this thread."""
        if self.busy:
            raise ValueError("A calibration operation is already running")
        request = {**payload, "schema_version": SCHEMA}
        encoded = json.dumps(request, allow_nan=False, ensure_ascii=False, indent=2)
        if len(encoded.encode()) + 1 > MAX_MESSAGE_BYTES:
            raise ValueError("Calibration request exceeds the workspace limit")
        directory = TemporaryDirectory(prefix="capture-reference-request-")
        self._directory = directory
        self._cancelled = False
        root = Path(directory.name)
        request_path, response_path = root / "request.json", root / "response.json"
        try:
            write_document(request_path, request)
            worker = Path(__file__).with_name("worker.py")
            self.busy_changed.emit(True)
            self.runner.run(
                [
                    self._executable,
                    "-I",
                    str(worker),
                    "--request",
                    str(request_path),
                    "--response",
                    str(response_path),
                ]
            )
        except (ValueError, OSError):
            self._cleanup()
            raise

    def cancel(self) -> None:
        """Stop this operation; completed immutable revisions remain available."""
        if self.busy:
            self._cancelled = True
            self.runner.stop()

    def _response(self, root: Path) -> dict[str, Any]:
        with (root / "response.json").open("rb") as stream:
            data = stream.read(MAX_MESSAGE_BYTES + 1)
        if len(data) > MAX_MESSAGE_BYTES:
            raise ValueError("Calibration response exceeds the workspace limit")
        response = json.loads(data)
        if not isinstance(response, dict) or not isinstance(response.get("ok"), bool):
            raise ValueError("Calibration provider returned an invalid response")
        return response

    def _finished(self, code: int) -> None:
        directory = self._directory
        if directory is None:
            return
        result: dict[str, Any] | None = None
        error: str | None = None
        try:
            if self._cancelled:
                error = (
                    "Calibration operation stopped. Saved revisions remain available."
                )
            elif code == -1:
                error = "Could not start the calibration runtime. Check its installation and retry."
            else:
                response = self._response(Path(directory.name))
                if code != 0 or response["ok"] is not True:
                    error = str(
                        response.get(
                            "error",
                            "Calibration operation failed; review the activity log.",
                        )
                    )
                elif isinstance(response.get("result"), dict):
                    result = response["result"]
                else:
                    error = "Calibration provider returned no usable result."
        except (ValueError, OSError) as exc:
            error = f"Could not read calibration results: {exc}. Review the activity log and retry."
        finally:
            self._cleanup()
        if result is not None:
            self.completed.emit(result)
        else:
            self.failed.emit(error or "Calibration operation failed.")

    def _cleanup(self) -> None:
        directory, self._directory = self._directory, None
        if directory is not None:
            directory.cleanup()
        self.busy_changed.emit(False)
