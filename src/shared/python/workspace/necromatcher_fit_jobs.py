"""Source-stamped native research refits on the canonical matching job service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
from typing import Any, Callable, Literal, TYPE_CHECKING
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
)
from src.shared.python.motion_matching.jobs.service import JobHandle
from src.shared.python.motion_matching.jobs.io_atomic import atomic_write_json
from src.shared.python.version_info import get_repo_root, read_git_commit
from .necromatcher_native_worker import execute_native_research_worker
from .artifact_handoff import compute_file_sha256
from .necromatcher import NecromatcherLibrary
from .project_store import DatasetMetadata, validate_workspace_id
from .necromatcher_shaft_evidence import BoundShaftEvidence, bind_fit_shaft_evidence
from src.shared.python.motion_matching.historical_fit.shaft_observations import (
    ShaftAxisEvidence,
)
from .necromatcher_fit_telemetry import (
    record_missing_worker_telemetry,
    read_worker_telemetry,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

_SOURCE_DIRECTORIES = (
    "src/shared/python/body_part_viz",
    "src/shared/python/core",
    "src/shared/python/workspace",
    "src/shared/python/shadow_tracker",
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


if TYPE_CHECKING:
    from .necromatcher_source_scope import BoundSourceFitScope, SourceFitScope


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
    operation: Literal["fit", "author_initialization"] = "fit"
    initialization_source: Literal["sampled_parent", "preserved_spline"] = (
        "sampled_parent"
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.initialization_source, str
        ) or self.initialization_source not in ("sampled_parent", "preserved_spline"):
            raise ValueError("Unknown refit initialization source")
        if not isinstance(self.operation, str) or self.operation not in (
            "fit",
            "author_initialization",
        ):
            raise ValueError("Refit operation must be fit or author_initialization")
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
        if self.initialization_source == "preserved_spline":
            if self.operation != "fit":
                raise ValueError("Preserved spline requires fit operation")
            if self.config.initialization_policy != "strict":
                raise ValueError(
                    "Preserved spline requires strict initialization policy"
                )
        if (
            self.operation == "author_initialization"
            and self.config.initialization_policy
            != "authored_range_project_zero_slopes"
        ):
            raise ValueError(
                "Author operation requires explicit authored initialization policy"
            )
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
    """Retain the existing three-argument monkeypatch/cancellation boundary."""
    return execute_native_research_worker(request_path, budget, cancelled)


def _shaft_record(bound: BoundShaftEvidence, weight: float) -> dict[str, Any]:
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not np.isfinite(weight)
        or not 0 <= weight <= 1
    ):
        raise ValueError("Shaft unknown visibility weight must be finite in [0,1]")
    return {
        "schema": "necromatcher/shaft-image-recipe/1",
        "evidence": bound.evidence.to_record(),
        "evidence_sha256": bound.evidence.sha256,
        "binding": {
            "source_clock_sha256": bound.source_clock_sha256,
            "reviewed_frame_count": bound.reviewed_frame_count,
            "observed_segment_count": bound.observed_segment_count,
        },
        "unknown_visibility_weight": weight,
    }


def _queue_shaft_recipe(
    library: NecromatcherLibrary,
    source: dict[str, Any],
    evidence: ShaftAxisEvidence,
    weight: float,
) -> dict[str, Any]:
    return _shaft_record(bind_fit_shaft_evidence(library, source, evidence), weight)


def _parse_shaft_recipe(recipe: dict[str, Any]) -> tuple[BoundShaftEvidence, float]:
    """Validate a declared receipt before native/source I/O; never authenticate it."""
    if not isinstance(recipe, dict) or set(recipe) != {
        "schema",
        "evidence",
        "evidence_sha256",
        "binding",
        "unknown_visibility_weight",
    }:
        raise ValueError("Malformed shaft image recipe")
    evidence = ShaftAxisEvidence.from_record(recipe["evidence"])
    if not isinstance(recipe["binding"], dict):
        raise ValueError("Malformed shaft recipe binding")
    try:
        bound = BoundShaftEvidence(evidence, **recipe["binding"])
    except TypeError as exc:
        raise ValueError("Malformed shaft recipe binding") from exc
    weight = recipe["unknown_visibility_weight"]
    if _shaft_record(bound, weight) != recipe:
        raise ValueError("Shaft image recipe differs from declared binding identities")
    return bound, weight


def _verify_shaft_request(
    library: NecromatcherLibrary, source: dict[str, Any], recipe: dict[str, Any]
) -> None:
    declared, weight = _parse_shaft_recipe(recipe)
    if _queue_shaft_recipe(library, source, declared.evidence, weight) != recipe:
        raise ValueError("Shaft image recipe binding changed since queue admission")


def _verify_scope_request(
    library: NecromatcherLibrary,
    request: dict[str, Any],
    candidate: dict[str, Any] | None = None,
) -> None:
    from .necromatcher_fit import (
        admit_refit_scope,
        scope_binding_record,
        validate_scope_payload,
        validate_scope_binding_record,
    )

    source = library.load_fit(request["source_fit_id"])
    if "source_scope" not in request:
        from .necromatcher_fit import scope_record

        if scope_record(source) is not None:
            raise ValueError("Request cannot delete inherited source scope")
        return
    recipe = request.get("shaft_images")
    shaft = _parse_shaft_recipe(recipe)[0].evidence if recipe is not None else None
    bound = admit_refit_scope(
        library,
        source,
        tuple(request["options"]["frame_indices"]),
        ImageFitConfig.from_record(request["options"]["config"]),
        shaft,
        request.get("source_scope"),
    )
    if bound is not None:
        validate_scope_binding_record(
            request.get("source_scope_binding"),
            scope_binding_record(bound, tuple(request["options"]["frame_indices"])),
        )
        if candidate is not None:
            if json.dumps(
                candidate["provenance"].get("request_options"),
                sort_keys=True,
                allow_nan=False,
            ) != json.dumps(request["options"], sort_keys=True, allow_nan=False):
                raise ValueError("Candidate options differ from admitted request")
            if (
                candidate["provenance"].get("source_fit_scope")
                != bound.scope.to_record()
            ):
                raise ValueError("Candidate differs from admitted source scope")
            validate_scope_binding_record(
                candidate["provenance"].get("source_fit_scope_binding"),
                request["source_scope_binding"],
            )
            if json.dumps(
                candidate.get("evidence", {})
                .get("original_fit", {})
                .get("frame_indices"),
                allow_nan=False,
            ) != json.dumps(request["options"]["frame_indices"], allow_nan=False):
                raise ValueError("Candidate training differs from admitted selection")
            validate_scope_payload(library, candidate, source)


def _verify_candidate_reads(
    library: NecromatcherLibrary, request: dict[str, Any], candidate: dict[str, Any]
) -> None:
    """Close fresh input authentication before candidate or fit publication."""
    with library.authenticated_read():
        if (
            library.load_asset(request["source_fit_id"]).metadata["hash"]
            != request["source_fit_hash"]
        ):
            raise ValueError("Warm-start fit changed during execution")
        recipe = request.get("shaft_images")
        if recipe is not None:
            _verify_shaft_request(
                library, library.load_fit(request["source_fit_id"]), recipe
            )
            if (
                candidate.get("evidence", {}).get("shaft_axis", {}).get("recipe")
                != recipe
            ):
                raise ValueError("Worker shaft recipe differs from admitted request")
        _verify_scope_request(library, request, candidate)


def _refit_work(
    library: NecromatcherLibrary,
    options: NativeRefitOptions,
    request: dict[str, Any],
    spec: MatchingJobSpec,
    session_id: str,
) -> Callable[[Callable[[JobProgress], None], Callable[[], bool]], MatchingWorkOutcome]:
    run_root = spec.run_root
    new_fit_id = request["new_fit_id"]

    def execute(
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
        telemetry = read_worker_telemetry(run_root, request)
        if (
            telemetry is not None
            and response["fit"]
            .get("evidence", {})
            .get("original_fit", {})
            .get("solver_telemetry")
            != telemetry.to_record()
        ):
            raise ValueError("Worker telemetry differs from candidate result")
        if cancelled():
            raise JobCancelledError("Native refit cancelled before publication")
        if fit_execution_stamp()["source_sha256"] != spec.hashes.solver_hash:
            raise ValueError("Fit implementation changed during execution")
        _verify_candidate_reads(library, request, response["fit"])
        candidate = run_root / "candidate.json"
        atomic_write_json(candidate, response["fit"])

        def publish() -> None:
            _verify_candidate_reads(library, request, response["fit"])
            library.add_fit(new_fit_id, session_id, candidate)
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

    def work(
        progress: Callable[[JobProgress], None], cancelled: Callable[[], bool]
    ) -> MatchingWorkOutcome:
        try:
            return execute(progress, cancelled)
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
            JobCancelledError,
        ) as exc:
            try:
                record_missing_worker_telemetry(run_root, request, str(exc))
            except (OSError, ValueError, TypeError, KeyError):
                logger.exception(
                    "Telemetry publication failed; original job fault retained"
                )
            raise

    return work


def _admit_refit_inputs(
    library: NecromatcherLibrary,
    source_fit_id: str,
    new_fit_id: str,
    options: NativeRefitOptions,
    shaft_evidence: ShaftAxisEvidence | None,
    source_scope: SourceFitScope | None,
) -> tuple[
    dict[str, Any], DatasetMetadata, BoundSourceFitScope | None, dict[str, Any] | None
]:
    """Authenticate a read group completely before scheduling or request writes."""
    from .necromatcher_fit import admit_refit_scope

    with library.authenticated_read():
        source = library.load_fit(source_fit_id)
        asset = library.load_asset(source_fit_id)
        if new_fit_id in {item.dataset_id for item in library.assets(asset.session_id)}:
            raise ValueError("Refit requires a new immutable fit identity")
        if len(options.coordinate_scales) != len(source["coordinate_order"]):
            raise ValueError("Refit scales must match bound native coordinate order")
        if any(index not in source["frame_indices"] for index in options.frame_indices):
            raise ValueError("Warm-start samples must exist in the source fit")
        bound_scope = admit_refit_scope(
            library,
            source,
            options.frame_indices,
            options.config,
            shaft_evidence,
            source_scope,
        )
        if (
            bound_scope is not None
            and options.initialization_source == "preserved_spline"
        ):
            from .necromatcher_spline import preserved_fit_spline

            start = preserved_fit_spline(source)
            domain = bound_scope.selected_domain(options.frame_indices)
            if start is None or (start.knot_times[0], start.knot_times[-1]) != (
                float(domain.first_pts),
                float(domain.last_pts),
            ):
                raise ValueError(
                    "Scoped preserved spline requires qualified exact interval restriction"
                )
        shaft_recipe = (
            _queue_shaft_recipe(
                library, source, shaft_evidence, options.unknown_visibility_weight
            )
            if shaft_evidence is not None
            else None
        )
    return source, asset, bound_scope, shaft_recipe


def start_native_refit(
    library: NecromatcherLibrary,
    source_fit_id: str,
    new_fit_id: str,
    options: NativeRefitOptions,
    service: MatchingJobService,
    shaft_evidence: ShaftAxisEvidence | None = None,
    source_scope: SourceFitScope | None = None,
) -> tuple[JobHandle, Path]:
    """Run a new immutable research version, preserving the original fit.

    The service owns scheduling. A clean worker owns native SDK execution;
    cancellation, changed inputs/code or failed computation prevent publication.
    """
    validate_workspace_id(new_fit_id, "new_fit_id")
    source, asset, bound_scope, shaft_recipe = _admit_refit_inputs(
        library, source_fit_id, new_fit_id, options, shaft_evidence, source_scope
    )
    from .necromatcher_fit import scope_binding_record

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
    hash_options: Any = asdict(options)
    if shaft_recipe is not None:
        request["shaft_images"] = shaft_recipe
        hash_options = {"options": hash_options, "shaft_images": shaft_recipe}
    if bound_scope is not None:
        request["source_scope"] = bound_scope.scope.to_record()
        request["source_scope_binding"] = scope_binding_record(
            bound_scope, options.frame_indices
        )
        hash_options = {
            "options": hash_options,
            "source_scope": request["source_scope"],
            "source_scope_binding": request["source_scope_binding"],
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
            _digest(hash_options),
            queued_stamp["source_sha256"],
        ),
        budget_wall_s=options.budget_wall_s,
        blockers=_BLOCKERS,
    )

    atomic_write_json(run_root / "request.json", request)
    return service.start(
        spec, work=_refit_work(library, options, request, spec, asset.session_id)
    ), run_root
