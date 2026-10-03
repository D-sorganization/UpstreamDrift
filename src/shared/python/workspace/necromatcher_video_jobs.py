"""Owned async source overlays using canonical matching jobs and checked downloads."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import stat
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
from src.shared.python.motion_matching.jobs.io_atomic import (
    atomic_write_json,
    read_text,
)
from src.shared.python.motion_matching.jobs.service import JobHandle
from src.shared.python.security import secure_popen
from src.shared.python.version_info import get_repo_root
from .artifact_handoff import compute_file_sha256
from .necromatcher import NecromatcherLibrary
from .necromatcher_fit_jobs import fit_execution_stamp
from .necromatcher_shaft_evidence import bind_fit_shaft_evidence
from src.shared.python.motion_matching.historical_fit import ShaftAxisEvidence

_BUDGET_WALL_S = 600.0
_BLOCKERS = (
    "monocular_research_only",
    "physical_clock_unknown",
    "anatomy_camera_unqualified",
)
_KIND = "necromatcher/video-job/1"
logger = logging.getLogger(__name__)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(read_text(path))
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


def video_shaft_evidence(
    library: NecromatcherLibrary, fit: dict[str, Any], request: dict[str, Any]
) -> ShaftAxisEvidence | None:
    """Reopen reviewed PNGs and source clock; a request receipt is not admission."""
    if "shaft_overlay" not in request:
        return None
    recipe = request["shaft_overlay"]
    if not isinstance(recipe, dict) or set(recipe) != {
        "evidence",
        "evidence_sha256",
        "source_clock_sha256",
    }:
        raise ValueError("Malformed shaft overlay request")
    evidence = ShaftAxisEvidence.from_record(recipe["evidence"])
    if evidence.sha256 != recipe["evidence_sha256"]:
        raise ValueError("Video shaft evidence hash differs from request")
    bound = bind_fit_shaft_evidence(library, fit, evidence)
    if bound.source_clock_sha256 != recipe["source_clock_sha256"]:
        raise ValueError("Video shaft source clock differs from request")
    return evidence


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
    video_shaft_evidence(library, fit, request)
    return fit


def _outputs(root: Path, request: dict[str, Any]) -> tuple[dict[str, Any], list[Path]]:
    directory = root / "overlay"
    manifest_path = directory / "manifest.json"
    if directory.is_symlink() or manifest_path.is_symlink():
        raise ValueError("Video output may not use symlink paths")
    manifest = _read(manifest_path)
    if manifest.get("schema") != "necromatcher/source-overlay-video/1":
        raise ValueError("Unsupported overlay manifest")
    if "shaft_overlay" in request:
        shaft = manifest.get("shaft_overlay")
        if not isinstance(shaft, dict) or any(
            shaft.get(key) != expected
            for key, expected in request["shaft_overlay"].items()
        ):
            raise ValueError("Overlay shaft evidence differs from the request")
        if (
            shaft.get("schema") != "necromatcher/shaft-video-overlay/1"
            or shaft.get("uncertainty_calibrated") is not False
            or shaft.get("physical_geometry_qualified") is not False
        ):
            raise ValueError("Overlay shaft qualification is malformed")
    elif "shaft_overlay" in manifest:
        raise ValueError("Disabled shaft overlay cannot include evidence")
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


def _file_identity(library_root: Path, path: Path) -> dict[str, Any]:
    """Bound a regular file to its owned path without reading its contents."""
    relative = path.absolute().relative_to(library_root.absolute())
    if ".." in relative.parts:
        raise ValueError("Artifact baseline path must stay inside the library")
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError("Artifact baseline may not use symlink paths")
        if ancestor == library_root:
            break
    value = path.stat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("Artifact baseline requires regular files")
    return {
        "path": relative.as_posix(),
        "identity": [
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        ],
    }


def _asset_bindings(
    library: NecromatcherLibrary, swing: str, identities: list[str]
) -> list[dict[str, Any]]:
    assets = {asset.dataset_id: asset for asset in library.assets(swing)}
    return [
        {
            "id": identity,
            "path": assets[identity].path,
            "kind": assets[identity].kind,
            "hash": assets[identity].metadata["hash"],
        }
        for identity in identities
    ]


def _artifact_baseline(
    library: NecromatcherLibrary, root: Path, request: dict[str, Any], swing: str
) -> dict[str, Any]:
    """Capture bounded metadata around an authoritative owned hash check."""
    identities = [request["source_fit_id"], request["model_id"], request["capture_id"]]
    bindings = _asset_bindings(library, swing, identities)
    names = sorted(path.name for path in (root / "overlay").iterdir())
    if len(names) > 5:
        raise ValueError("Overlay contains too many artifact files")
    paths = [root / "request.json", *(root / "overlay" / name for name in names)]
    for binding in bindings:
        path = Path(binding["path"])
        paths.append(path if path.is_absolute() else library.root / path)
    return {
        "swing_id": swing,
        "bindings": bindings,
        "output_names": names,
        "files": [_file_identity(library.root, path) for path in paths],
    }


def _artifact_available(
    library: NecromatcherLibrary, root: Path, request: dict[str, Any]
) -> bool:
    """Fail closed on legacy or changed artifacts using only bounded metadata."""
    try:
        baseline = _read(root / "complete.json").get("artifact_baseline")
        if not isinstance(baseline, dict) or len(baseline.get("files", [])) > 9:
            return False
        return baseline == _artifact_baseline(
            library, root, request, baseline["swing_id"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


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

    def submit(
        self, fit_id: str, shaft_evidence: ShaftAxisEvidence | None = None
    ) -> dict[str, Any]:
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
            hash_options: Any = request["selected_frames"]
            if shaft_evidence is not None:
                bound = bind_fit_shaft_evidence(self.library, fit, shaft_evidence)
                request["shaft_overlay"] = {
                    "evidence": shaft_evidence.to_record(),
                    "evidence_sha256": shaft_evidence.sha256,
                    "source_clock_sha256": bound.source_clock_sha256,
                }
                hash_options = {
                    "selected_frames": request["selected_frames"],
                    "shaft_overlay": request["shaft_overlay"],
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
                    _digest(hash_options),
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
            swing = self.library.load_asset(request["source_fit_id"]).session_id
            baseline = _artifact_baseline(self.library, root, request, swing)
            _parents(self.library, request)
            _outputs(root, request)
            if (
                compute_file_sha256(root / "overlay" / "manifest.json")
                != "sha256:" + response["manifest_sha256"]
            ):
                raise ValueError("Worker overlay manifest hash mismatch")
            if baseline != _artifact_baseline(self.library, root, request, swing):
                raise ValueError("Overlay metadata changed during verification")

            def publish() -> None:
                if "shaft_overlay" in request:
                    _parents(self.library, request)
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
                        "artifact_baseline": baseline,
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
        if manifest_path.is_symlink() or (
            manifest_path.exists() and not manifest_path.is_file()
        ):
            raise ValueError("Video manifest must be a regular file without symlink")
        manifest = (
            RunManifest.from_dict(_read(manifest_path))
            if manifest_path.exists()
            else None
        )
        if manifest is not None and manifest.run_id != run_id:
            raise ValueError("Video manifest identity mismatch")
        if (
            manifest is not None
            and manifest.status == JobStatus.SUCCEEDED
            and manifest.acceptance != AcceptanceState.REJECTED
        ):
            raise ValueError("Research overlay cannot claim scientific acceptance")
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
        available = verified and _artifact_available(self.library, root, request)
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
            "download_available": available,
            "artifact_state": "verified_stat_baseline"
            if available
            else "changed_or_unverified",
            "producer_source_commit": request.get("execution_stamp", {}).get(
                "source_commit"
            ),
        }

    def view(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            return self._view(run_id)

    def _view_for_fit(self, fit_id: str, run_id: str) -> dict[str, Any]:
        _, request = self._record(run_id)
        if request.get("source_fit_id") != fit_id:
            raise ValueError("Stored overlay belongs to a different source fit")
        _parents(self.library, request)
        return self._view(run_id)

    def view_for_fit(self, fit_id: str, run_id: str) -> dict[str, Any]:
        """Recall one source-bound run without granting foreign fit controls."""
        with self._lock:
            return self._view_for_fit(fit_id, run_id)

    def stored_runs(self, fit_id: str) -> list[dict[str, Any]]:
        """List durable fit-scoped statuses without scheduling or hashing video."""
        with self._lock:
            self.library.load_fit(fit_id)
            directory = self.library.root / "video-runs"
            if directory.is_symlink():
                raise ValueError("Video run directory may not be a symlink")
            if not directory.exists():
                return []
            runs = []
            for path in sorted(directory.iterdir()):
                if not re.fullmatch(r"[a-f0-9]{32}", path.name):
                    continue
                try:
                    _, request = self._record(path.name)
                except (OSError, ValueError, KeyError) as exc:
                    logger.warning(
                        "Skipping unreadable stored overlay %s: %s", path.name, exc
                    )
                    continue
                if request.get("source_fit_id") == fit_id:
                    runs.append(self._view_for_fit(fit_id, path.name))
            return runs

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
            swing = self.library.load_asset(request["source_fit_id"]).session_id
            baseline = _artifact_baseline(self.library, root, request, swing)
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
                if baseline != _artifact_baseline(self.library, root, request, swing):
                    raise ValueError(
                        "Overlay metadata changed during download packaging"
                    )
                destination = root / ("overlay-" + uuid4().hex + ".zip")
                os.link(archive_path, destination)
            complete = _read(root / "complete.json")
            complete["artifact_baseline"] = baseline
            atomic_write_json(root / "complete.json", complete)
            return destination
