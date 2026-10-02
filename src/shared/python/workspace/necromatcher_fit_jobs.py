"""Source-stamped native research refits on the canonical matching job service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from src.shared.python.motion_matching.historical_fit import ImageFitConfig
from src.shared.python.motion_matching.jobs import (
    AcceptanceState,
    HashBundle,
    JobCancelledError,
    JobProgress,
    JobStage,
    MatchingJobService,
    MatchingJobSpec,
    MatchingWorkOutcome,
    ProcessGuard,
)
from src.shared.python.motion_matching.jobs.service import JobHandle
from src.shared.python.motion_matching.jobs.io_atomic import atomic_write_json
from src.shared.python.security import secure_popen
from src.shared.python.core import repo_python_environment
from src.shared.python.version_info import get_repo_root, read_git_commit
from .artifact_handoff import compute_file_sha256
from .necromatcher import NecromatcherLibrary
from .project_store import validate_workspace_id

_SOURCE_DIRECTORIES = (
    "src/shared/python/core",
    "src/shared/python/workspace",
    "src/shared/python/motion_matching",
    "src/shared/python/estimation",
    "src/shared/python/numerical_methods",
    "src/engines/physics_engines/mujoco/python",
)
_BLOCKERS = (
    "monocular_research_only",
    "physical_clock_unknown",
    "camera_unqualified",
    "independent_dynamics_not_replayed",
)


def _digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()
    )


@dataclass(frozen=True)
class NativeRefitOptions:
    """Explicit sampling and prior scales; all times remain source PTS."""

    frame_indices: tuple[int, ...]
    knot_count: int
    coordinate_scales: tuple[float, ...]
    config: ImageFitConfig = field(default_factory=ImageFitConfig)
    unknown_visibility_weight: float = 0.5
    budget_wall_s: float = 600.0

    def __post_init__(self) -> None:
        indices = tuple(self.frame_indices)
        if (
            len(indices) < 2
            or any(type(i) is not int or i < 0 for i in indices)
            or any(b <= a for a, b in zip(indices, indices[1:], strict=False))
        ):
            raise ValueError("Refit samples require increasing source frame indices")
        if type(self.knot_count) is not int or not 2 <= self.knot_count <= len(indices):
            raise ValueError("Refit knot count must be between two and sample count")
        scales = tuple(self.coordinate_scales)
        numeric = np.asarray(scales)
        if (
            numeric.ndim != 1
            or numeric.dtype.kind not in "ifu"
            or not scales
            or not np.isfinite(numeric).all()
            or np.any(numeric <= 0)
        ):
            raise ValueError("Refit coordinate scales must be finite and positive")
        if not isinstance(self.config, ImageFitConfig):
            raise ValueError("Refit requires validated image-fit configuration")
        if not np.isfinite(self.budget_wall_s) or self.budget_wall_s <= 0:
            raise ValueError("Refit wall budget must be finite and positive")
        if (
            not np.isfinite(self.unknown_visibility_weight)
            or not 0 <= self.unknown_visibility_weight <= 1
        ):
            raise ValueError("Unknown visibility weight must be in [0, 1]")
        object.__setattr__(self, "frame_indices", indices)
        object.__setattr__(self, "coordinate_scales", scales)


def fit_execution_stamp() -> dict[str, Any]:
    """Fingerprint reviewed source files and installed runtime at execution time."""
    root = get_repo_root()
    sources = {
        path.relative_to(root).as_posix(): compute_file_sha256(path)
        for directory in _SOURCE_DIRECTORIES
        for path in sorted((root / directory).rglob("*.py"))
    }
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tools_commit": read_git_commit(root / "vendor/ud-tools") or "unresolved",
    }
    for package in ("numpy", "scipy", "mujoco", "opencv-python"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            runtime[package] = "not-installed"
    return {
        "source_commit": read_git_commit(root),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": _digest(sources),
        "source_files": sources,
        "runtime": runtime,
        "runtime_sha256": _digest(runtime),
    }


def _execute_worker(
    request_path: Path, budget: float, cancelled: Callable[[], bool]
) -> dict[str, Any]:
    guard = ProcessGuard()
    process = secure_popen(
        [
            sys.executable,
            "-u",
            "-m",
            "src.shared.python.workspace.necromatcher_fit_worker",
            str(request_path),
        ],
        cwd=get_repo_root(),
        env=repo_python_environment(get_repo_root()),
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


def start_native_refit(
    library: NecromatcherLibrary,
    source_fit_id: str,
    new_fit_id: str,
    options: NativeRefitOptions,
    service: MatchingJobService,
) -> tuple[JobHandle, Path]:
    """Run a new immutable research version, preserving the original fit.

    The service owns scheduling. A clean worker owns native SDK execution;
    cancellation, changed inputs/code or failed computation prevent publication.
    """
    validate_workspace_id(new_fit_id, "new_fit_id")
    source = library.load_fit(source_fit_id)
    asset = library.load_asset(source_fit_id)
    if new_fit_id in {item.dataset_id for item in library.assets(asset.session_id)}:
        raise ValueError("Refit requires a new immutable fit identity")
    if len(options.coordinate_scales) != len(source["coordinate_order"]):
        raise ValueError("Refit scales must match bound native coordinate order")
    if any(index not in source["frame_indices"] for index in options.frame_indices):
        raise ValueError("Warm-start samples must exist in the source fit")
    run_root = library.root / "runs" / uuid4().hex
    queued_stamp = fit_execution_stamp()
    request = {
        "library_root": str(library.root),
        "source_fit_id": source_fit_id,
        "source_fit_hash": asset.metadata["hash"],
        "new_fit_id": new_fit_id,
        "options": asdict(options),
        "execution_stamp": queued_stamp,
        "execution_started": False,
    }
    spec = MatchingJobSpec(
        run_root.name,
        "mujoco",
        JobStage.IK,
        run_root,
        HashBundle(
            source["capture_hash"],
            source["model_hash"],
            queued_stamp["runtime_sha256"],
            _digest(asdict(options)),
            queued_stamp["source_sha256"],
        ),
        budget_wall_s=options.budget_wall_s,
        blockers=_BLOCKERS,
    )

    def work(
        progress: Callable[[JobProgress], None], cancelled: Callable[[], bool]
    ) -> MatchingWorkOutcome:
        stamp = fit_execution_stamp()
        if stamp["source_sha256"] != spec.hashes.solver_hash:
            raise ValueError("Fit implementation changed before execution")
        request["execution_stamp"] = stamp
        request["execution_started"] = True
        request_path = atomic_write_json(run_root / "request.json", request)
        progress(
            JobProgress(
                JobStage.IK, None, "Computing source-bound native research refit"
            )
        )
        response = _execute_worker(request_path, options.budget_wall_s, cancelled)
        if cancelled():
            raise JobCancelledError("Native refit cancelled before publication")
        if fit_execution_stamp()["source_sha256"] != spec.hashes.solver_hash:
            raise ValueError("Fit implementation changed during execution")
        if (
            library.load_asset(source_fit_id).metadata["hash"]
            != request["source_fit_hash"]
        ):
            raise ValueError("Warm-start fit changed during execution")
        candidate = run_root / "candidate.json"
        atomic_write_json(candidate, response["fit"])

        def publish() -> None:
            library.add_fit(new_fit_id, asset.session_id, candidate)
            progress(
                JobProgress(
                    JobStage.IK, 1.0, "Research fit stored; dynamics remain unqualified"
                )
            )

        return MatchingWorkOutcome(
            AcceptanceState.REJECTED,
            tuple(response["fit"]["evidence"]["rejection_reasons"]),
            "Research computation completed; dynamics not qualified",
            publish=publish,
        )

    atomic_write_json(run_root / "request.json", request)
    return service.start(spec, work=work), run_root
