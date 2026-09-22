"""Immutable contracts for MS-105 matching jobs (#10379)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

JOBS_SCHEMA = "motion-matching-jobs/1.0.0"
_GOVERNING_ISSUE = 10379

_DEFAULT_BLOCKERS = (
    "native_long_run_host_recovery_requires_desk_receipt",
    "software_contract_jobs_are_not_native_timing_evidence",
    "no_universal_solve_time_guarantee",
)


class JobStatus(str, Enum):
    """Lifecycle status for one matching run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStage(str, Enum):
    """Pipeline stages used for progress and PF-08 benchmarks."""

    IK = "ik"
    REALLOCATION = "reallocation"
    CANDIDATE = "candidate"
    FIT_REPLAY = "fit_replay"


class AcceptanceState(str, Enum):
    """Whether output may be treated as an accepted swing result.

    Partial and interrupted states must never be advertised as accepted.
    """

    PARTIAL = "partial"
    INTERRUPTED = "interrupted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class FaultKind(str, Enum):
    """Named fault classes for recovery decisions."""

    CANCEL = "cancel"
    WORKER_CRASH = "worker_crash"
    APP_RESTART = "app_restart"
    DISK_FULL = "disk_full"
    ENGINE_ABSENT = "engine_absent"
    REMOTE_DISCONNECT = "remote_disconnect"
    HOST_UNAVAILABLE = "host_unavailable"
    CORRUPT_ARTIFACT = "corrupt_artifact"
    INCOMPATIBLE_CHECKPOINT = "incompatible_checkpoint"
    UNKNOWN = "unknown"


class RecoveryDecision(str, Enum):
    """What the recovery layer may do after a classified fault."""

    RESUME_COMPATIBLE = "resume_compatible"
    RESTART_FRESH = "restart_fresh"
    PRESERVE_AND_FAIL = "preserve_and_fail"
    FAIL_CLOSED = "fail_closed"


class JobCancelledError(RuntimeError):
    """Cooperative cancellation requested by the caller or shell."""


class PartialOutputRejectedError(ValueError):
    """Raised when code attempts to advertise partial output as accepted."""


class IncompatibleResumeError(ValueError):
    """Checkpoint hashes do not match the current run identity."""


class CheckpointCompatibilityError(IncompatibleResumeError):
    """Alias retained for call sites that name the compatibility gate."""


class CorruptPackageError(ValueError):
    """Manifest or artifact failed checksum / schema validation."""


class PortablePackageError(ValueError):
    """Portable package export/import contract violation."""


class EngineUnavailableError(RuntimeError):
    """Required native engine is not importable on this host."""


class DiskFullError(OSError):
    """Disk capacity exhausted while writing artifacts."""


class UnsupportedHostError(RuntimeError):
    """Host is offline or otherwise cannot run the matching job."""


@dataclass(frozen=True, slots=True)
class HashBundle:
    """Content hashes that gate checkpoint resume compatibility."""

    data_hash: str
    model_hash: str
    runtime_hash: str
    controller_hash: str
    solver_hash: str

    def __post_init__(self) -> None:
        for name in (
            "data_hash",
            "model_hash",
            "runtime_hash",
            "controller_hash",
            "solver_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {
            "data_hash": self.data_hash,
            "model_hash": self.model_hash,
            "runtime_hash": self.runtime_hash,
            "controller_hash": self.controller_hash,
            "solver_hash": self.solver_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HashBundle:
        return cls(
            data_hash=str(data["data_hash"]),
            model_hash=str(data["model_hash"]),
            runtime_hash=str(data["runtime_hash"]),
            controller_hash=str(data["controller_hash"]),
            solver_hash=str(data["solver_hash"]),
        )

    def matches(self, other: HashBundle) -> bool:
        return self.to_dict() == other.to_dict()


@dataclass(frozen=True, slots=True)
class JobProgress:
    """One progress report for desktop and web shells."""

    stage: JobStage
    fraction: float | None
    message: str

    def __post_init__(self) -> None:
        if self.fraction is not None and not 0.0 <= self.fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {self.fraction}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "fraction": self.fraction,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class MatchingJobSpec:
    """Inputs for one matching job run.

    ``use_external_scheduler`` is rejected: MS-105 reuses existing job APIs
    and must not create a second scheduler.
    """

    run_id: str
    engine: str
    stage: JobStage
    run_root: Path
    hashes: HashBundle
    budget_wall_s: float = 30.0
    max_workers: int = 1
    use_external_scheduler: bool = False
    blockers: tuple[str, ...] = _DEFAULT_BLOCKERS

    def __post_init__(self) -> None:
        if self.use_external_scheduler:
            raise ValueError(
                "MS-105 forbids a second scheduler; reuse async_action / "
                "training CancelToken infrastructure"
            )
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if not self.engine.strip():
            raise ValueError("engine must be non-empty")
        if self.budget_wall_s <= 0:
            raise ValueError("budget_wall_s must be > 0")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        object.__setattr__(self, "run_root", Path(self.run_root))
        object.__setattr__(self, "blockers", tuple(self.blockers))


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Atomic run manifest persisted beside artifacts."""

    run_id: str
    status: JobStatus
    stage: JobStage
    hashes: HashBundle
    acceptance: AcceptanceState
    provenance: str
    resume_reason: str | None = None
    fault: FaultKind | None = None
    blockers: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_BLOCKERS)
    diagnostics_relpath: str | None = None

    def __post_init__(self) -> None:
        if (
            self.acceptance == AcceptanceState.ACCEPTED
            and self.status != JobStatus.SUCCEEDED
        ):
            raise ValueError("accepted manifests require succeeded status")
        object.__setattr__(self, "blockers", tuple(self.blockers))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": JOBS_SCHEMA,
            "governing_issue": _GOVERNING_ISSUE,
            "run_id": self.run_id,
            "status": self.status.value,
            "stage": self.stage.value,
            "hashes": self.hashes.to_dict(),
            "acceptance": self.acceptance.value,
            "provenance": self.provenance,
            "blockers": list(self.blockers),
        }
        if self.resume_reason is not None:
            payload["resume_reason"] = self.resume_reason
        if self.fault is not None:
            payload["fault"] = self.fault.value
        if self.diagnostics_relpath is not None:
            payload["diagnostics_relpath"] = self.diagnostics_relpath
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunManifest:
        schema = data.get("schema_version")
        if schema != JOBS_SCHEMA:
            raise CorruptPackageError(
                f"manifest schema_version must be {JOBS_SCHEMA!r}; got {schema!r}"
            )
        fault_raw = data.get("fault")
        return cls(
            run_id=str(data["run_id"]),
            status=JobStatus(str(data["status"])),
            stage=JobStage(str(data["stage"])),
            hashes=HashBundle.from_dict(data["hashes"]),
            acceptance=AcceptanceState(str(data["acceptance"])),
            provenance=str(data["provenance"]),
            resume_reason=data.get("resume_reason"),
            fault=FaultKind(str(fault_raw)) if fault_raw else None,
            blockers=tuple(data.get("blockers", _DEFAULT_BLOCKERS)),
            diagnostics_relpath=data.get("diagnostics_relpath"),
        )
