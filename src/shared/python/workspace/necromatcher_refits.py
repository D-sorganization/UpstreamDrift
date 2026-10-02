"""Native/web refit controls over the canonical matching executor and manifests."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import re
from threading import Lock
from typing import Any

from src.shared.python.motion_matching.jobs import (
    AcceptanceState,
    JobStatus,
    MatchingJobService,
    RunManifest,
)
from src.shared.python.motion_matching.jobs.service import JobHandle
from .necromatcher import NecromatcherLibrary
from .necromatcher_fit_jobs import NativeRefitOptions, start_native_refit


class NativeRefitSession:
    """Own live handles, bounded admission and shutdown; never schedule work twice.

    The matching service owns execution and durable status. This session keeps
    only live/control handles; historical views reopen its canonical manifests.
    One active job avoids queued duplicate identities and unbounded submissions.
    """

    def __init__(self, library: NecromatcherLibrary) -> None:
        self.library = library
        self._service = MatchingJobService()
        self._handles: OrderedDict[str, tuple[JobHandle, str, str]] = OrderedDict()
        self._lock = Lock()
        self._closed = False

    def submit(
        self, source_fit_id: str, new_fit_id: str, options: NativeRefitOptions
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Refit session is closed")
            for handle, _, _ in self._handles.values():
                try:
                    handle.join(timeout=0)
                except TimeoutError as exc:
                    raise RuntimeError("A research refit is already running") from exc
            handle, root = start_native_refit(
                self.library, source_fit_id, new_fit_id, options, self._service
            )
            self._handles[root.name] = (handle, source_fit_id, new_fit_id)
            while len(self._handles) > 32:
                self._handles.popitem(last=False)
            return self._view(root.name)

    def _root(self, run_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("Invalid refit run identity")
        root = self.library.root / "runs" / run_id
        if not root.is_dir():
            raise KeyError(run_id)
        return root

    def _view(self, run_id: str) -> dict[str, Any]:
        live = self._handles.get(run_id)
        root = self.library.root / "runs" / run_id if live else self._root(run_id)
        if live:
            handle, source_id, new_id = live
            try:
                result = handle.join(timeout=0)
            except TimeoutError:
                return {
                    "run_id": run_id,
                    "source_fit_id": source_id,
                    "new_fit_id": new_id,
                    "status": "running",
                    "acceptance": "partial",
                    "blockers": [],
                    "message": "Computing source-bound research refit",
                    "fraction": None,
                    "control_available": True,
                }
            return {
                "run_id": run_id,
                "source_fit_id": source_id,
                "new_fit_id": new_id,
                "status": result.status.value,
                "acceptance": result.acceptance.value,
                "blockers": list(result.blockers),
                "message": result.message,
                "fraction": result.last_progress.fraction
                if result.last_progress
                else None,
                "control_available": True,
            }
        manifest = RunManifest.from_dict(
            json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        )
        request = json.loads((root / "request.json").read_text(encoding="utf-8"))
        if manifest.run_id != run_id:
            raise ValueError("Refit manifest identity mismatch")
        unowned_active = manifest.status in {JobStatus.PENDING, JobStatus.RUNNING}
        message = (
            "Execution state unverified; this host has no live control handle"
            if unowned_active
            else "Reopened research run"
        )
        diagnostics = root / "diagnostics.json"
        if diagnostics.is_file() and not unowned_active:
            message = str(
                json.loads(diagnostics.read_text(encoding="utf-8"))["message"]
            )
        return {
            "run_id": run_id,
            "source_fit_id": request["source_fit_id"],
            "new_fit_id": request["new_fit_id"],
            "status": manifest.status.value,
            "acceptance": manifest.acceptance.value,
            "blockers": list(manifest.blockers),
            "message": message,
            "fraction": None,
            "control_available": False,
        }

    def view(self, run_id: str) -> dict[str, Any]:
        """Poll without waiting; reopen canonical research status after restart."""
        with self._lock:
            return self._view(run_id)

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Request cancellation of an owned live job; never relabel terminal work."""
        with self._lock:
            if run_id in self._handles:
                self._handles[run_id][0].request_cancel()
            return self._view(run_id)

    def close(self) -> None:
        """Cancel owned jobs and drain the canonical executor on host shutdown."""
        with self._lock:
            self._closed = True
            for handle, _, _ in self._handles.values():
                handle.request_cancel()
        self._service.close()


def refit_plan(library: NecromatcherLibrary, fit_id: str) -> dict[str, Any]:
    """Expose verified source choices and recorded priors, without inventing scales."""
    fit = library.load_fit(fit_id)
    return {
        "source_fit_id": fit_id,
        "frame_indices": fit["frame_indices"],
        "coordinate_order": fit["coordinate_order"],
        "coordinate_units": fit["coordinate_units"],
        "recorded_options": fit["provenance"].get("request_options"),
        "qualification": fit["qualification"],
    }
