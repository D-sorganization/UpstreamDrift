"""Matching job service — progress, cancel, recovery, shell views (#10379).

Reuses cooperative cancel (training :class:`ThreadingCancelToken` pattern)
and does **not** introduce a second scheduler or message store. Bounded
concurrency is a local thread pool only for owned work callables.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from src.shared.python.training.contracts import ThreadingCancelToken

from .checkpoint import load_checkpoint
from .contracts import (
    JOBS_SCHEMA,
    AcceptanceState,
    CheckpointCompatibilityError,
    CorruptPackageError,
    DiskFullError,
    EngineUnavailableError,
    FaultKind,
    JobCancelledError,
    JobProgress,
    JobStage,
    JobStatus,
    MatchingJobSpec,
    PartialOutputRejectedError,
    RunManifest,
    UnsupportedHostError,
)
from .io_atomic import atomic_write_json, write_run_manifest
from .process_guard import ProcessGuard
from .recovery import classify_fault, decide_recovery

__all__ = ["JobHandle", "JobResult", "MatchingJobService", "ResumeOutcome"]

WorkCallable = Callable[
    [Callable[[JobProgress], None], Callable[[], bool]],
    Any,
]
EngineProbe = Callable[[str], bool]

# Named worker/trust-boundary faults — never bare ``except Exception``.
_JOB_WORK_ERRORS: tuple[type[BaseException], ...] = (
    DiskFullError,
    UnsupportedHostError,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
)
_CHECKPOINT_LOAD_ERRORS: tuple[type[BaseException], ...] = (
    CorruptPackageError,
    CheckpointCompatibilityError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
)


@dataclass(frozen=True, slots=True)
class JobResult:
    """Terminal outcome of one matching job attempt."""

    run_id: str
    status: JobStatus
    acceptance: AcceptanceState
    stage: JobStage
    provenance: str
    fault: FaultKind | None = None
    blockers: tuple[str, ...] = ()
    diagnostics_path: Path | None = None
    last_progress: JobProgress | None = None
    resume_reason: str | None = None
    message: str = ""

    def to_manifest(self, hashes: Any) -> RunManifest:
        return RunManifest(
            run_id=self.run_id,
            status=self.status,
            stage=self.stage,
            hashes=hashes,
            acceptance=self.acceptance,
            provenance=self.provenance,
            resume_reason=self.resume_reason,
            fault=self.fault,
            blockers=self.blockers,
            diagnostics_relpath=(
                self.diagnostics_path.name if self.diagnostics_path else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ResumeOutcome:
    """Result of resume-or-restart after crash / app restart."""

    provenance: str
    acceptance: AcceptanceState
    resume_reason: str
    stage: JobStage | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)


class JobHandle:
    """Opaque handle for cancel + join (not a scheduler ticket)."""

    def __init__(
        self,
        *,
        cancel_token: ThreadingCancelToken,
        future: Future[JobResult],
        process_guard: ProcessGuard,
    ) -> None:
        self._cancel_token = cancel_token
        self._future = future
        self._process_guard = process_guard

    def request_cancel(self) -> None:
        self._cancel_token.request_cancel()
        self._process_guard.terminate_all(reason="cancel")

    def join(self, timeout: float | None = None) -> JobResult:
        return self._future.result(timeout=timeout)


class MatchingJobService:
    """Run matching work with progress, cancel, recovery, and shell DTOs.

    Args:
        engine_available: Optional probe; ``False`` fails closed as engine absent.
        max_workers: Bounded concurrency for owned work (default 1).
    """

    def __init__(
        self,
        *,
        engine_available: EngineProbe | None = None,
        max_workers: int = 1,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._engine_available = engine_available or (lambda _engine: True)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="mm-job"
        )

    def start(self, spec: MatchingJobSpec, *, work: WorkCallable) -> JobHandle:
        """Start owned work under the shared cancel/progress contract."""
        cancel_token = ThreadingCancelToken()
        process_guard = ProcessGuard()
        future = self._executor.submit(
            self._run, spec, work, cancel_token, process_guard
        )
        return JobHandle(
            cancel_token=cancel_token,
            future=future,
            process_guard=process_guard,
        )

    def _run(
        self,
        spec: MatchingJobSpec,
        work: WorkCallable,
        cancel_token: ThreadingCancelToken,
        process_guard: ProcessGuard,
    ) -> JobResult:
        del process_guard  # reserved for owned subprocess registration by callers
        run_root = Path(spec.run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        last_progress: JobProgress | None = None

        def progress_cb(update: JobProgress) -> None:
            nonlocal last_progress
            last_progress = update

        def cancel_check() -> bool:
            return cancel_token.is_cancelled

        if not self._engine_available(spec.engine):
            result = self._fail_engine_absent(spec, last_progress)
            self._persist_result(spec, result)
            return result

        write_run_manifest(
            run_root,
            RunManifest(
                run_id=spec.run_id,
                status=JobStatus.RUNNING,
                stage=spec.stage,
                hashes=spec.hashes,
                acceptance=AcceptanceState.PARTIAL,
                provenance="fresh",
                blockers=spec.blockers,
            ),
        )

        try:
            if cancel_check():
                raise JobCancelledError("cancelled before start")
            work(progress_cb, cancel_check)
            if cancel_check():
                raise JobCancelledError("cancelled after work returned")
            result = JobResult(
                run_id=spec.run_id,
                status=JobStatus.SUCCEEDED,
                acceptance=AcceptanceState.ACCEPTED,
                stage=spec.stage,
                provenance="fresh",
                blockers=spec.blockers,
                last_progress=last_progress,
                message="succeeded",
            )
        except JobCancelledError as exc:
            diagnostics = self._write_diagnostics(run_root, exc, FaultKind.CANCEL)
            result = JobResult(
                run_id=spec.run_id,
                status=JobStatus.CANCELLED,
                acceptance=AcceptanceState.INTERRUPTED,
                stage=last_progress.stage if last_progress else spec.stage,
                provenance="interrupted",
                fault=FaultKind.CANCEL,
                blockers=spec.blockers,
                diagnostics_path=diagnostics,
                last_progress=last_progress,
                message=str(exc),
            )
        except EngineUnavailableError as exc:
            result = self._fail_engine_absent(spec, last_progress, message=str(exc))
        except _JOB_WORK_ERRORS as exc:
            fault = classify_fault(exc)
            diagnostics = self._write_diagnostics(run_root, exc, fault)
            result = JobResult(
                run_id=spec.run_id,
                status=JobStatus.FAILED,
                acceptance=AcceptanceState.REJECTED,
                stage=last_progress.stage if last_progress else spec.stage,
                provenance="interrupted",
                fault=fault,
                blockers=(*spec.blockers, f"fault:{fault.value}"),
                diagnostics_path=diagnostics,
                last_progress=last_progress,
                message=str(exc),
            )

        self._persist_result(spec, result)
        return result

    def _fail_engine_absent(
        self,
        spec: MatchingJobSpec,
        last_progress: JobProgress | None,
        *,
        message: str = "native engine unavailable",
    ) -> JobResult:
        diagnostics = self._write_diagnostics(
            Path(spec.run_root),
            EngineUnavailableError(message),
            FaultKind.ENGINE_ABSENT,
        )
        return JobResult(
            run_id=spec.run_id,
            status=JobStatus.FAILED,
            acceptance=AcceptanceState.REJECTED,
            stage=last_progress.stage if last_progress else spec.stage,
            provenance="interrupted",
            fault=FaultKind.ENGINE_ABSENT,
            blockers=(*spec.blockers, "native_engine_unavailable"),
            diagnostics_path=diagnostics,
            last_progress=last_progress,
            message=message,
        )

    def _write_diagnostics(
        self,
        run_root: Path,
        exc: BaseException,
        fault: FaultKind,
    ) -> Path:
        path = Path(run_root) / "diagnostics.json"
        atomic_write_json(
            path,
            {
                "fault": fault.value,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "recovery": decide_recovery(
                    fault, has_compatible_checkpoint=False
                ).value,
            },
        )
        return path

    def _persist_result(self, spec: MatchingJobSpec, result: JobResult) -> None:
        write_run_manifest(Path(spec.run_root), result.to_manifest(spec.hashes))

    def resume_or_restart(
        self,
        spec: MatchingJobSpec,
        *,
        reason: str,
    ) -> ResumeOutcome:
        """Resume a compatible checkpoint or restart with an explicit reason."""
        if not reason.strip():
            raise ValueError("resume/restart reason must be non-empty")
        ckpt_path = Path(spec.run_root) / "checkpoint.json"
        if not ckpt_path.exists():
            return ResumeOutcome(
                provenance="restarted",
                acceptance=AcceptanceState.INTERRUPTED,
                resume_reason=reason,
            )
        try:
            record = load_checkpoint(ckpt_path, expected_hashes=spec.hashes)
        except _CHECKPOINT_LOAD_ERRORS:
            return ResumeOutcome(
                provenance="restarted",
                acceptance=AcceptanceState.INTERRUPTED,
                resume_reason=reason,
            )
        return ResumeOutcome(
            provenance="resumed_numerical",
            acceptance=AcceptanceState.PARTIAL,
            resume_reason=reason,
            stage=record.stage,
            payload=record.payload,
        )

    def advertise_as_accepted(self, manifest: RunManifest) -> None:
        """Fail closed when partial/interrupted output is treated as accepted."""
        if manifest.acceptance != AcceptanceState.ACCEPTED:
            raise PartialOutputRejectedError(
                f"cannot advertise acceptance={manifest.acceptance.value} as accepted"
            )
        if manifest.status != JobStatus.SUCCEEDED:
            raise PartialOutputRejectedError(
                f"cannot advertise status={manifest.status.value} as accepted"
            )

    def shell_view(
        self,
        result: JobResult,
        *,
        shell: str,
    ) -> dict[str, Any]:
        """Desktop/web progress and failure payload (no fabricated success)."""
        if shell not in {"desktop", "web"}:
            raise ValueError(f"unsupported shell: {shell!r}")
        progress = (
            result.last_progress.to_dict()
            if result.last_progress is not None
            else {
                "stage": result.stage.value,
                "fraction": None,
                "message": result.message or result.status.value,
            }
        )
        failure: dict[str, Any] | None = None
        if result.status in {JobStatus.FAILED, JobStatus.CANCELLED}:
            failure = {
                "fault": (result.fault or FaultKind.UNKNOWN).value,
                "message": result.message or result.status.value,
                "blockers": list(result.blockers),
                "diagnostics_path": (
                    str(result.diagnostics_path) if result.diagnostics_path else None
                ),
            }
        return {
            "shell": shell,
            "run_id": result.run_id,
            "status": result.status.value,
            "acceptance": result.acceptance.value,
            "progress": progress,
            "failure": failure,
            "can_reopen_package": True,
            "schema_version": JOBS_SCHEMA,
        }
