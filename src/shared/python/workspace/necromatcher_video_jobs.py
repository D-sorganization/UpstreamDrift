"""Owned async source overlays using canonical matching jobs and checked downloads."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Lock
import time
from typing import Any
from uuid import uuid4
from zipfile import ZIP_STORED, ZipFile

from src.shared.python.core import repo_python_environment
from src.shared.python.motion_matching.jobs import (
    AcceptanceState,
    HashBundle,
    JobCancelledError,
    JobProgress,
    JobStage,
    JobStatus,
    MatchingJobService,
    MatchingJobSpec,
    MatchingWorkOutcome,
    ProcessGuard,
    RunManifest,
)
from src.shared.python.motion_matching.jobs.io_atomic import atomic_write_json
from src.shared.python.motion_matching.jobs.service import JobHandle
from src.shared.python.security import secure_popen
from src.shared.python.version_info import get_repo_root
from .artifact_handoff import compute_file_sha256
from .necromatcher import NecromatcherLibrary
from .necromatcher_fit_jobs import fit_execution_stamp

_BUDGET_WALL_S = 600.0
_BLOCKERS = (
    "monocular_research_only",
    "physical_clock_unknown",
    "anatomy_camera_unqualified",
)
_KIND = "necromatcher/video-job/1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Video job record must be a JSON object")
    return value


def _digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
    )


def _execute_worker(
    request_path: Path, budget: float, cancelled: Callable[[], bool]
) -> dict[str, Any]:
    """Run one clean SDK process and terminate only that owned process."""
    if cancelled():
        raise JobCancelledError("Video export cancelled before worker launch")
    guard = ProcessGuard()
    env = repo_python_environment(get_repo_root())
    if sys.platform.startswith("linux"):
        env.setdefault("MUJOCO_GL", "osmesa")
    process = secure_popen(
        [
            sys.executable,
            "-u",
            "-m",
            "src.shared.python.workspace.necromatcher_video_worker",
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
                raise JobCancelledError("Owned video export cancelled")
            if time.monotonic() >= deadline:
                raise RuntimeError("Video export exceeded its wall budget")
            try:
                output, errors = process.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise RuntimeError(f"Native video worker failed: {errors[-2000:]}")
        response = json.loads(output)
        if not isinstance(response, dict):
            raise ValueError("Native video worker response must be an object")
        return response
    finally:
        guard.terminate_all(reason="video export complete or interrupted")
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def _parents(library: NecromatcherLibrary, request: dict[str, Any]) -> dict[str, Any]:
    fit = library.load_fit(request["source_fit_id"])
    if (
        library.load_asset(request["source_fit_id"]).metadata["hash"]
        != request["source_fit_hash"]
    ):
        raise ValueError("Video source fit hash changed")
    for kind in ("model", "capture"):
        if (
            fit[f"{kind}_id"] != request[f"{kind}_id"]
            or fit[f"{kind}_hash"] != request[f"{kind}_hash"]
        ):
            raise ValueError("Video parent identity or hash changed")
        if (
            library.load_asset(fit[f"{kind}_id"]).metadata["hash"]
            != fit[f"{kind}_hash"]
        ):
            raise ValueError("Video parent bytes changed")
    return fit


def _outputs(root: Path, request: dict[str, Any]) -> tuple[dict[str, Any], list[Path]]:
    directory = root / "overlay"
    manifest_path = directory / "manifest.json"
    if directory.is_symlink() or manifest_path.is_symlink():
        raise ValueError("Video output may not use symlink paths")
    manifest = _read(manifest_path)
    if manifest.get("schema") != "necromatcher/source-overlay-video/1":
        raise ValueError("Unsupported overlay manifest")
    for name, expected in (
        ("fit_id", request["source_fit_id"]),
        ("fit_hash", request["source_fit_hash"]),
        ("model_id", request["model_id"]),
        ("model_hash", request["model_hash"]),
        ("capture_id", request["capture_id"]),
        ("capture_hash", request["capture_hash"]),
        ("qualification", "monocular_research_hypothesis"),
        ("physical_time_qualified", False),
        ("camera_qualified", False),
        ("anatomy_qualified", False),
    ):
        if manifest.get(name) != expected or (
            expected is False and manifest.get(name) is not False
        ):
            raise ValueError("Overlay manifest identity or qualification mismatch")
    requested = {f"frame-{index:06d}.png" for index in request["selected_frames"]}
    if {record["path"] for record in manifest["pngs"]} != requested:
        raise ValueError("Overlay selected stills differ from the request")
    records = [manifest["video"], *manifest["pngs"]]
    paths = [manifest_path]
    names = {"manifest.json"}
    for index, record in enumerate(records):
        name = record["path"]
        if (
            not isinstance(name, str)
            or name in names
            or (index == 0 and name != "overlay.mp4")
            or (index > 0 and not re.fullmatch(r"frame-[0-9]+\.png", name))
        ):
            raise ValueError("Unsafe or duplicate overlay output filename")
        path = directory / name
        if (
            path.is_symlink()
            or not path.is_file()
            or compute_file_sha256(path) != "sha256:" + record["sha256"]
        ):
            raise ValueError("Overlay output hash mismatch")
        names.add(name)
        paths.append(path)
    if {path.name for path in directory.iterdir()} != names:
        raise ValueError("Overlay output contains undeclared files")
    return manifest, paths


class NativeVideoSession:
    """Own bounded job handles; durable manifests never invent worker termination."""

    def __init__(self, library: NecromatcherLibrary) -> None:
        self.library = library
        self._service = MatchingJobService()
        self._handles: OrderedDict[str, JobHandle] = OrderedDict()
        self._lock = Lock()
        self._closed = False

    def _root(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("Invalid video run identity")
        root = self.library.root / "video-runs" / run_id
        if root.parent.is_symlink() or root.is_symlink() or not root.is_dir():
            raise KeyError(run_id)
        return root

    def submit(self, fit_id: str) -> dict[str, Any]:
        """Schedule a new overlay with owned paths and first/middle/last stills."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Video session is closed")
            for handle in self._handles.values():
                try:
                    handle.join(timeout=0)
                except TimeoutError as exc:
                    raise RuntimeError("An overlay export is already running") from exc
            fit = self.library.load_fit(fit_id)
            stamp = fit_execution_stamp()
            directory = self.library.root / "video-runs"
            if directory.is_symlink():
                raise ValueError("Video run directory may not be a symlink")
            root = directory / uuid4().hex
            indices = fit["frame_indices"]
            request = {
                "kind": _KIND,
                "run_id": root.name,
                "library_root": str(self.library.root),
                "source_fit_id": fit_id,
                "source_fit_hash": self.library.load_asset(fit_id).metadata["hash"],
                "model_id": fit["model_id"],
                "model_hash": fit["model_hash"],
                "capture_id": fit["capture_id"],
                "capture_hash": fit["capture_hash"],
                "selected_frames": sorted(
                    {indices[0], indices[len(indices) // 2], indices[-1]}
                ),
                "execution_stamp": stamp,
                "execution_started": False,
            }
            spec = MatchingJobSpec(
                root.name,
                "mujoco",
                JobStage.CANDIDATE,
                root,
                HashBundle(
                    fit["capture_hash"],
                    fit["model_hash"],
                    stamp["runtime_sha256"],
                    _digest(request["selected_frames"]),
                    stamp["source_sha256"],
                ),
                budget_wall_s=_BUDGET_WALL_S,
                blockers=_BLOCKERS,
            )
            atomic_write_json(root / "request.json", request)
            self._handles[root.name] = self._service.start(
                spec, work=self._work(root, request, spec)
            )
            while len(self._handles) > 32:
                self._handles.popitem(last=False)
            return self._view(root.name)

    def _work(
        self, root: Path, request: dict[str, Any], spec: MatchingJobSpec
    ) -> Callable[
        [Callable[[JobProgress], None], Callable[[], bool]], MatchingWorkOutcome
    ]:
        def work(
            progress: Callable[[JobProgress], None], cancelled: Callable[[], bool]
        ) -> MatchingWorkOutcome:
            stamp = fit_execution_stamp()
            if (
                stamp["source_sha256"] != spec.hashes.solver_hash
                or stamp["runtime_sha256"] != spec.hashes.runtime_hash
            ):
                raise ValueError(
                    "Video implementation or runtime changed before execution"
                )
            _parents(self.library, request)
            request["execution_started"] = True
            request["execution_stamp"] = stamp
            path = atomic_write_json(root / "request.json", request)
            progress(
                JobProgress(
                    JobStage.CANDIDATE, None, "Rendering source-bound research overlay"
                )
            )
            response = _execute_worker(path, _BUDGET_WALL_S, cancelled)
            if cancelled():
                raise JobCancelledError("Video cancelled before verified publication")
            current = fit_execution_stamp()
            if (
                current["source_sha256"] != spec.hashes.solver_hash
                or current["runtime_sha256"] != spec.hashes.runtime_hash
            ):
                raise ValueError(
                    "Video implementation or runtime changed during execution"
                )
            _parents(self.library, request)
            _outputs(root, request)
            if (
                compute_file_sha256(root / "overlay" / "manifest.json")
                != "sha256:" + response["manifest_sha256"]
            ):
                raise ValueError("Worker overlay manifest hash mismatch")

            def publish() -> None:
                atomic_write_json(
                    root / "complete.json",
                    {
                        "kind": _KIND,
                        "run_id": root.name,
                        "request_hash": compute_file_sha256(root / "request.json"),
                        "manifest_hash": compute_file_sha256(
                            root / "overlay" / "manifest.json"
                        ),
                        "source_sha256": spec.hashes.solver_hash,
                        "runtime_sha256": spec.hashes.runtime_hash,
                    },
                )
                progress(
                    JobProgress(
                        JobStage.CANDIDATE,
                        1.0,
                        "Verified overlay ready; research remains unqualified",
                    )
                )

            return MatchingWorkOutcome(
                AcceptanceState.REJECTED,
                _BLOCKERS,
                "Overlay exported; scientific qualification remains rejected",
                publish,
            )

        return work

    def _record(self, run_id: str) -> tuple[Path, dict[str, Any]]:
        root = self._root(run_id)
        if (root / "request.json").is_symlink():
            raise ValueError("Video request may not be a symlink")
        request = _read(root / "request.json")
        if request.get("kind") != _KIND or request.get("run_id") != run_id:
            raise ValueError("Video request identity mismatch")
        return root, request

    def _verified(
        self, root: Path, request: dict[str, Any], manifest: RunManifest
    ) -> bool:
        complete_path = root / "complete.json"
        if (
            manifest.status != JobStatus.SUCCEEDED
            or complete_path.is_symlink()
            or not complete_path.is_file()
        ):
            return False
        complete = _read(complete_path)
        return bool(
            complete.get("kind") == _KIND
            and complete.get("run_id") == root.name
            and complete.get("request_hash")
            == compute_file_sha256(root / "request.json")
            and complete.get("manifest_hash")
            == compute_file_sha256(root / "overlay" / "manifest.json")
            and complete.get("source_sha256") == manifest.hashes.solver_hash
            and complete.get("runtime_sha256") == manifest.hashes.runtime_hash
            and manifest.hashes.model_hash == request.get("model_hash")
            and manifest.hashes.data_hash == request.get("capture_hash")
            and request.get("execution_started") is True
        )

    def _view(self, run_id: str) -> dict[str, Any]:
        root, request = self._record(run_id)
        handle = self._handles.get(run_id)
        result = None
        if handle is not None:
            try:
                result = handle.join(timeout=0)
            except TimeoutError:
                pass
        manifest_path = root / "run_manifest.json"
        manifest = (
            RunManifest.from_dict(_read(manifest_path))
            if manifest_path.exists()
            else None
        )
        if manifest is not None and manifest.run_id != run_id:
            raise ValueError("Video manifest identity mismatch")
        status = (
            result.status
            if result
            else manifest.status
            if manifest
            else JobStatus.PENDING
        )
        acceptance = (
            result.acceptance
            if result
            else manifest.acceptance
            if manifest
            else AcceptanceState.PARTIAL
        )
        verified = self._verified(root, request, manifest) if manifest else False
        message = result.message if result else "Source-bound research overlay export"
        if handle is None and status in {JobStatus.PENDING, JobStatus.RUNNING}:
            message = "Execution state unverified; this host has no live control handle"
        return {
            "run_id": run_id,
            "source_fit_id": request["source_fit_id"],
            "status": status.value,
            "acceptance": acceptance.value,
            "qualification": "monocular_research_hypothesis",
            "blockers": list(
                result.blockers
                if result
                else manifest.blockers
                if manifest
                else _BLOCKERS
            ),
            "message": message,
            "fraction": result.last_progress.fraction
            if result and result.last_progress
            else None,
            "control_available": handle is not None,
            "execution_started": request.get("execution_started") is True,
            "execution_verified": verified,
            "download_available": verified,
        }

    def view(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            return self._view(run_id)

    def cancel(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            if run_id in self._handles:
                self._handles[run_id].request_cancel()
            return self._view(run_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for handle in self._handles.values():
                handle.request_cancel()
        self._service.close()

    def download(self, run_id: str) -> Path:
        """Revalidate immutable parents and exact output bytes before packaging."""
        with self._lock:
            if not self._view(run_id)["execution_verified"]:
                raise ValueError(
                    "Download requires terminal verified successful execution"
                )
            root, request = self._record(run_id)
            _parents(self.library, request)
            _, files = _outputs(root, request)
            expected = {path.name: compute_file_sha256(path) for path in files}
            with TemporaryDirectory(prefix="video-download-", dir=root) as temporary:
                archive_path = Path(temporary) / "overlay.zip"
                with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
                    for path in files:
                        archive.write(path, path.name)
                with ZipFile(archive_path) as archive:
                    for name, identity in expected.items():
                        with archive.open(name) as stream:
                            digest = hashlib.sha256()
                            while chunk := stream.read(1024 * 1024):
                                digest.update(chunk)
                            if "sha256:" + digest.hexdigest() != identity:
                                raise ValueError(
                                    "Download archive output hash mismatch"
                                )
                _parents(self.library, request)
                _, current_files = _outputs(root, request)
                if {
                    path.name: compute_file_sha256(path) for path in current_files
                } != expected or not self._view(run_id)["execution_verified"]:
                    raise ValueError("Overlay changed during download packaging")
                destination = root / ("overlay-" + uuid4().hex + ".zip")
                os.link(archive_path, destination)
            return destination
