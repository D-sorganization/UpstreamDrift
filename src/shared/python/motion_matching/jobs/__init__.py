"""MS-105 (#10379): reliable motion-matching jobs, recovery, portable results.

Reuses the shared worker/progress/cancel mechanism from PR #9472 (#8880) and
training subprocess process-guard patterns. This package is **not** a second
scheduler or message store — it owns contracts, atomic I/O, checkpoints,
portable packages, fault recovery, and PF-08 benchmark schemas only.

Software-contract layer: measured service targets are budgets, never invented
universal solve-time guarantees. Native long-run host recovery remains a
named blocker until DeskComputer evidence lands.
"""

from __future__ import annotations

from .benchmarks import (
    SERVICE_TARGETS,
    StageBenchmarkSample,
    TimeToAcceptedReport,
    measure_stage_sample,
    publish_service_targets,
)
from .checkpoint import load_checkpoint, write_checkpoint
from .contracts import (
    JOBS_SCHEMA,
    AcceptanceState,
    CheckpointCompatibilityError,
    CorruptPackageError,
    DiskFullError,
    EngineUnavailableError,
    FaultKind,
    HashBundle,
    IncompatibleResumeError,
    JobCancelledError,
    JobProgress,
    JobStage,
    JobStatus,
    MatchingJobSpec,
    PartialOutputRejectedError,
    PortablePackageError,
    RecoveryDecision,
    RunManifest,
    UnsupportedHostError,
)
from .portable import export_portable_package, import_portable_package
from .process_guard import ProcessGuard
from .recovery import classify_fault, decide_recovery
from .service import MatchingJobService

__all__ = [
    "JOBS_SCHEMA",
    "SERVICE_TARGETS",
    "AcceptanceState",
    "CheckpointCompatibilityError",
    "CorruptPackageError",
    "DiskFullError",
    "EngineUnavailableError",
    "FaultKind",
    "HashBundle",
    "IncompatibleResumeError",
    "JobCancelledError",
    "JobProgress",
    "JobStage",
    "JobStatus",
    "MatchingJobService",
    "MatchingJobSpec",
    "PartialOutputRejectedError",
    "PortablePackageError",
    "ProcessGuard",
    "RecoveryDecision",
    "RunManifest",
    "StageBenchmarkSample",
    "TimeToAcceptedReport",
    "UnsupportedHostError",
    "classify_fault",
    "decide_recovery",
    "export_portable_package",
    "import_portable_package",
    "load_checkpoint",
    "measure_stage_sample",
    "publish_service_targets",
    "write_checkpoint",
    "write_run_manifest",
]

# Re-export write_run_manifest from io_atomic without circular import noise.
from .io_atomic import write_run_manifest as write_run_manifest  # noqa: E402
