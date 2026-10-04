"""Owned, cancellable authored-replay impact jobs with authenticated downloads."""

from __future__ import annotations
from collections.abc import Callable
import hashlib
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any, cast
from uuid import uuid4
from zipfile import ZipFile, ZIP_STORED
import os

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
    RunManifest,
)
from src.shared.python.motion_matching.jobs.io_atomic import (
    atomic_write_json,
    read_text,
)
from .artifact_handoff import compute_file_sha256
from .necromatcher import NecromatcherLibrary
from .necromatcher_impact_contracts import ReplayImpactGeometry, ReplayImpactSelection
from .necromatcher_native_worker import execute_native_research_worker

_KIND = "necromatcher/impact-job/1"
_FILES = ("trajectory.json", "impact-receipt.json", "result.json")
_BLOCKERS = (
    "authored_impact_assumptions",
    "physical_source_time_unknown",
    "scientific_unqualified",
)


def impact_execution_stamp() -> dict[str, Any]:
    """Resolve the canonical impact implementation/runtime owner lazily."""
    from .necromatcher_impact_execution import impact_execution_stamp as stamp

    return stamp()


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Impact record must be a regular link-free file")
    value = json.loads(read_text(path))
    if not isinstance(value, dict):
        raise ValueError("Impact record must be an object")
    json.dumps(value, allow_nan=False)
    return value


def _digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
    )


def impact_job_parents(library: NecromatcherLibrary, replay_id: str) -> dict[str, Any]:
    """Revalidate the canonical trace and exact same-session immutable parents."""
    with library.authenticated_read():
        trace = library.load_replay(replay_id)
        replay = library.load_asset(replay_id)
        if replay.kind != "authored_replay":
            raise ValueError("Impact jobs require an authored replay")
        parents = {"replay_id": replay_id, "replay_hash": replay.metadata["hash"]}
        for name, kind in (
            ("profile", "torque_profile"),
            ("fit", "kinematic_fit"),
            ("model", "native_model"),
            ("capture", "image_capture"),
        ):
            identity, digest = trace.meta[name + "_id"], trace.meta[name + "_hash"]
            if not isinstance(identity, str) or not isinstance(digest, str):
                raise ValueError("Replay parent declarations must be strings")
            asset = library.load_asset(identity)
            if (
                asset.kind != kind
                or asset.session_id != replay.session_id
                or asset.metadata["hash"] != digest
            ):
                raise ValueError("Replay parent kind, session or hash differs")
            parents.update({name + "_id": identity, name + "_hash": digest})
        return parents


def _check_request(
    library: NecromatcherLibrary, root: Path, request: dict[str, Any]
) -> None:
    if (
        request.get("kind") != _KIND
        or request.get("run_id") != root.name
        or Path(request.get("library_root", "")).resolve() != library.root.resolve()
        or request.get("parents") != impact_job_parents(library, request["replay_id"])
    ):
        raise ValueError("Impact request identity or parents differ")
    ReplayImpactGeometry.from_record(request["geometry"])
    ReplayImpactSelection.from_record(request["selection"])
    _budget(request["budget_wall_s"])


def _budget(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("Impact wall budget must be a real number")
    number = cast("int | float", value)
    if not math.isfinite(number) or not 0 < number <= 600:
        raise ValueError("Impact wall budget must be finite and in (0,600]")
    return float(number)


def _check_stamp(expected: dict[str, Any]) -> None:
    current = impact_execution_stamp()
    if any(
        current.get(key) != expected.get(key)
        for key in ("source_commit", "source_sha256", "runtime_sha256")
    ):
        raise ValueError("Impact implementation or runtime changed")


def _artifact_summary(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    result = _read(root / "output" / "result.json")
    summary = result.get("summary")
    keys = {"carry_m", "max_height_m", "flight_time_s", "landing_angle_deg"}
    if (
        not isinstance(summary, dict)
        or set(summary) != keys
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in summary.values()
        )
        or result.get("replay_clock_policy") != "authored_simulation_seconds"
    ):
        raise ValueError("Malformed impact result summary or clock")
    return {
        "files": [*_FILES, "request.json"],
        "summary": summary,
        "geometry": request["geometry"],
        "selection": request["selection"],
        "clockpolicy": result["replay_clock_policy"],
    }


def _outputs(root: Path, request: dict[str, Any]) -> dict[str, str]:
    output = root / "output"
    if (
        output.is_symlink()
        or not output.is_dir()
        or {p.name for p in output.iterdir()} != set(_FILES)
    ):
        raise ValueError("Impact output contains missing or undeclared files")
    records = {name: _read(output / name) for name in _FILES}
    from .necromatcher_impact_receipt import load_replay_impact_receipt

    state = load_replay_impact_receipt(
        output / "impact-receipt.json", output / "trajectory.json"
    )
    metadata = state.metadata
    for key, expected in request["parents"].items():
        if _digest(metadata.get(key)) != _digest(expected):
            raise ValueError("Impact receipt parent identity differs")
    for key in ("geometry", "selection"):
        if _digest(metadata.get(key)) != _digest(request[key]):
            raise ValueError("Impact receipt assumptions differ")
    result = records["result.json"]
    if result.get("schema") != "necromatcher/impact-result/1":
        raise ValueError("Unsupported impact result schema")
    if (
        result.get("scientific_qualified") is not False
        or result.get("physical_source_time_qualified") is not False
    ):
        raise ValueError(
            "Impact output cannot claim scientific or physical qualification"
        )
    for key in ("parents", "geometry", "selection", "execution_stamp"):
        if _digest(result.get(key)) != _digest(request[key]):
            raise ValueError("Impact output assumptions or identities differ")
    _artifact_summary(root, request)
    return {name: compute_file_sha256(output / name) for name in _FILES}


class NativeImpactSession:
    """Own canonical matching jobs; restarted hosts never invent live controls."""

    def __init__(self, library: NecromatcherLibrary) -> None:
        self.library = library
        self._service = MatchingJobService()
        self._handles: dict[str, Any] = {}
        self._lock = Lock()
        self._closed = False

    def _root(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("Invalid impact run identity")
        root = self.library.root / "impact-runs" / run_id
        if (
            self.library.root.is_symlink()
            or root.parent.is_symlink()
            or root.is_symlink()
            or not root.is_dir()
        ):
            raise KeyError(run_id)
        return root

    def _record(self, replay_id: str, run_id: str) -> tuple[Path, dict[str, Any]]:
        root = self._root(run_id)
        request = _read(root / "request.json")
        if request.get("replay_id") != replay_id:
            raise ValueError("Impact run belongs to a different replay")
        _check_request(self.library, root, request)
        return root, request

    def submit(
        self,
        replay_id: str,
        geometry: ReplayImpactGeometry,
        selection: ReplayImpactSelection,
        budget_wall_s: float,
    ) -> dict[str, Any]:
        """Admit exact declarations before creating a run or starting any SDK work."""
        if not isinstance(geometry, ReplayImpactGeometry) or not isinstance(
            selection, ReplayImpactSelection
        ):
            raise ValueError("Impact jobs require typed geometry and selection")
        budget = _budget(budget_wall_s)
        with self._lock:
            if self._closed:
                raise RuntimeError("Impact session is closed")
            for handle in self._handles.values():
                try:
                    handle.join(timeout=0)
                except TimeoutError as exc:
                    raise RuntimeError("An impact job is already running") from exc
            parents = impact_job_parents(self.library, replay_id)
            with self.library.authenticated_read():
                if selection.recorded_sample_index >= len(
                    self.library.load_replay(replay_id).t
                ):
                    raise IndexError("Selected sample is outside the replay")
            directory = self.library.root / "impact-runs"
            if self.library.root.is_symlink() or directory.is_symlink():
                raise ValueError("Impact run directory may not be a symlink")
            root = directory / uuid4().hex
            stamp = impact_execution_stamp()
            request = {
                "kind": _KIND,
                "run_id": root.name,
                "library_root": str(self.library.root),
                "replay_id": replay_id,
                "parents": parents,
                "geometry": geometry.to_record(),
                "selection": selection.to_record(),
                "budget_wall_s": budget,
                "execution_stamp": stamp,
            }
            spec = MatchingJobSpec(
                root.name,
                "mujoco",
                JobStage.CANDIDATE,
                root,
                HashBundle(
                    parents["capture_hash"],
                    parents["model_hash"],
                    stamp["runtime_sha256"],
                    _digest(request),
                    stamp["source_sha256"],
                ),
                budget_wall_s=budget,
                blockers=_BLOCKERS,
            )
            atomic_write_json(root / "request.json", request)
            self._handles[root.name] = self._service.start(
                spec, work=self._work(root, request)
            )
            while len(self._handles) > 32:
                self._handles.pop(next(iter(self._handles)))
            return self._view(replay_id, root.name)

    def _work(self, root: Path, request: dict[str, Any]) -> Callable:
        def work(progress: Callable, cancelled: Callable) -> MatchingWorkOutcome:
            _check_stamp(request["execution_stamp"])
            _check_request(self.library, root, request)
            progress(
                JobProgress(
                    JobStage.CANDIDATE,
                    None,
                    "Computing explicitly authored research impact",
                )
            )
            response = execute_native_research_worker(
                root / "request.json",
                request["budget_wall_s"],
                cancelled,
                operation="impact",
            )
            if cancelled():
                raise JobCancelledError("Impact cancelled before publication")
            _check_stamp(request["execution_stamp"])
            _check_request(self.library, root, request)
            hashes = _outputs(root, request)
            if response.get("artifact_hashes") != hashes:
                raise ValueError("Worker artifact hashes differ")

            def publish() -> None:
                _check_stamp(request["execution_stamp"])
                _check_request(self.library, root, request)
                if _outputs(root, request) != hashes:
                    raise ValueError("Impact artifacts changed before publication")
                atomic_write_json(
                    root / "complete.json",
                    {
                        "kind": _KIND,
                        "run_id": root.name,
                        "request_hash": compute_file_sha256(root / "request.json"),
                        "artifact_hashes": hashes,
                        "execution_stamp": request["execution_stamp"],
                    },
                )

            return MatchingWorkOutcome(
                AcceptanceState.REJECTED,
                _BLOCKERS,
                "Research impact computed; scientific qualification rejected",
                publish,
            )

        return work

    def _verified(
        self, root: Path, request: dict[str, Any], manifest: RunManifest | None
    ) -> bool:
        if manifest is None or manifest.status != JobStatus.SUCCEEDED:
            return False
        complete = _read(root / "complete.json")
        expected = HashBundle(
            request["parents"]["capture_hash"],
            request["parents"]["model_hash"],
            request["execution_stamp"]["runtime_sha256"],
            _digest(request),
            request["execution_stamp"]["source_sha256"],
        )
        return (
            manifest.acceptance == AcceptanceState.REJECTED
            and manifest.hashes.matches(expected)
            and complete.get("kind") == _KIND
            and complete.get("run_id") == root.name
            and complete.get("request_hash")
            == compute_file_sha256(root / "request.json")
            and complete.get("execution_stamp") == request["execution_stamp"]
            and complete.get("artifact_hashes") == _outputs(root, request)
        )

    def _view(self, replay_id: str, run_id: str) -> dict[str, Any]:
        root, request = self._record(replay_id, run_id)
        handle = self._handles.get(run_id)
        result = None
        if handle is not None:
            try:
                result = handle.join(timeout=0)
            except TimeoutError:
                pass
        path = root / "run_manifest.json"
        manifest = RunManifest.from_dict(_read(path)) if path.exists() else None
        if manifest is not None and manifest.run_id != run_id:
            raise ValueError("Impact manifest identity differs")
        if (
            manifest is not None
            and manifest.status == JobStatus.SUCCEEDED
            and manifest.acceptance != AcceptanceState.REJECTED
        ):
            raise ValueError("Research impact cannot claim scientific acceptance")
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
        verified = self._verified(root, request, manifest)
        return {
            "run_id": run_id,
            "replay_id": replay_id,
            "status": status.value,
            "acceptance": acceptance.value,
            "message": result.message
            if result
            else "Execution unverified without a live handle"
            if handle is None and status in (JobStatus.PENDING, JobStatus.RUNNING)
            else "Authored research impact",
            "blockers": list(_BLOCKERS),
            "control_available": handle is not None
            and status in (JobStatus.PENDING, JobStatus.RUNNING),
            "execution_verified": verified,
            "download_available": verified,
            "fraction": None,
            "scientific_qualified": False,
            "physical_source_time_qualified": False,
            "artifactsummary": _artifact_summary(root, request) if verified else None,
        }

    def view(self, replay_id: str, run_id: str) -> dict[str, Any]:
        with self._lock:
            return self._view(replay_id, run_id)

    def cancel(self, replay_id: str, run_id: str) -> dict[str, Any]:
        with self._lock:
            self._record(replay_id, run_id)
            if run_id not in self._handles:
                raise RuntimeError("This host has no live impact control handle")
            self._handles[run_id].request_cancel()
            return self._view(replay_id, run_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for handle in self._handles.values():
                handle.request_cancel()
        self._service.close()

    def download(self, replay_id: str, run_id: str) -> Path:
        """Revalidate parents and exact file bytes before and after ZIP creation."""
        with self._lock:
            if not self._view(replay_id, run_id)["execution_verified"]:
                raise RuntimeError("Download requires verified terminal success")
            root, request = self._record(replay_id, run_id)
            hashes = {
                **_outputs(root, request),
                "request.json": compute_file_sha256(root / "request.json"),
            }
            with TemporaryDirectory(prefix="impact-download-", dir=root) as temporary:
                archive_path = Path(temporary) / "review.zip"
                with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
                    for name in hashes:
                        archive.write(
                            root / name
                            if name == "request.json"
                            else root / "output" / name,
                            name,
                        )
                with ZipFile(archive_path) as archive:
                    if any(
                        "sha256:" + hashlib.sha256(archive.read(name)).hexdigest()
                        != digest
                        for name, digest in hashes.items()
                    ):
                        raise ValueError("Impact ZIP differs from authenticated files")
                self._record(replay_id, run_id)
                if not self._view(replay_id, run_id)["execution_verified"]:
                    raise ValueError("Impact changed during download")
                destination = root / ("review-" + uuid4().hex + ".zip")
                os.link(archive_path, destination)
            return destination
