"""Durable simulation run storage, lifecycle state machine, and quota pruning (Issue #14087 / Epic #14085)."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.api.models.responses import SimulationErrorInfo
from src.sim_run.models import (
    ResultEnvelope,
    RunRecord,
    RunRequest,
    RunStatus,
    SCHEMA_VERSION_V1,
)

_RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")
ALLOWED_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    "queued": {"running", "failed", "expired"},
    "running": {"succeeded", "failed", "expired"},
    "succeeded": {"expired"},
    "failed": {"expired"},
    "expired": set(),
}


class RunStoreError(RuntimeError):
    """Base exception for simulation run store operations."""


class RunNotFoundError(KeyError, RunStoreError):
    """Raised when a requested run ID does not exist in the store."""


class InvalidStateTransitionError(ValueError, RunStoreError):
    """Raised when an illegal lifecycle state transition is attempted."""


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SimulationRunStore:
    """Durable run store for headless simulation runs with quota pruning."""

    def __init__(
        self,
        results_root: Path | str,
        max_runs: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        self.results_root = Path(results_root).resolve()
        self.max_runs = max_runs
        self.max_bytes = max_bytes
        self.results_root.mkdir(parents=True, exist_ok=True)

    def _validate_run_id(self, run_id: str) -> None:
        if (
            not run_id
            or not _RUN_ID_PATTERN.match(run_id)
            or ".." in run_id
            or len(run_id) > 128
        ):
            raise ValueError(
                f"Invalid run_id '{run_id}': must match [a-zA-Z0-9_-]+ up to 128 chars"
            )

    def _generate_run_id(self) -> str:
        return f"run_{uuid.uuid4().hex}"

    def get_run_dir(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        return self.results_root / run_id

    def _get_record_path(self, run_id: str) -> Path:
        return self.get_run_dir(run_id) / "record.json"

    def _atomic_write_record(self, run_id: str, record: RunRecord) -> None:
        run_dir = self.get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        record_path = self._get_record_path(run_id)
        temp_path = record_path.with_suffix(f".tmp.{uuid.uuid4().hex}")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(record.model_dump_json(indent=2))
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, record_path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def create_run(self, request: RunRequest, run_id: str | None = None) -> RunRecord:
        """Create and persist a new simulation run in queued state."""
        assigned_id = run_id or self._generate_run_id()
        self._validate_run_id(assigned_id)
        if self._get_record_path(assigned_id).exists():
            raise RunStoreError(f"Run '{assigned_id}' already exists")
        record = RunRecord(
            schema_version=SCHEMA_VERSION_V1,
            run_id=assigned_id,
            status="queued",
            request=request,
            created_at=_iso_now(),
            started_at=None,
            completed_at=None,
            expires_at=None,
            result=None,
            error=None,
        )
        self._atomic_write_record(assigned_id, record)
        return record

    def get_run(self, run_id: str) -> RunRecord:
        """Load and return RunRecord by run ID."""
        self._validate_run_id(run_id)
        record_path = self._get_record_path(run_id)
        if not record_path.is_file():
            raise RunNotFoundError(
                f"Run '{run_id}' not found under {self.results_root}"
            )
        try:
            with open(record_path, encoding="utf-8") as f:
                return RunRecord.model_validate(json.load(f))
        except Exception as exc:
            raise RunStoreError(
                f"Failed to read record for run '{run_id}': {exc}"
            ) from exc

    def list_runs(self, status: RunStatus | None = None) -> list[RunRecord]:
        """List all runs in the store, optionally filtered by status, ordered by created_at."""
        runs: list[RunRecord] = []
        if not self.results_root.exists():
            return runs
        for entry in self.results_root.iterdir():
            if entry.is_dir() and (entry / "record.json").is_file():
                try:
                    record = self.get_run(entry.name)
                    if status is None or record.status == status:
                        runs.append(record)
                except (RunStoreError, ValueError):
                    continue
        runs.sort(key=lambda r: (r.created_at, r.run_id))
        return runs

    def transition(
        self,
        run_id: str,
        new_status: RunStatus,
        *,
        started_at: str | None = None,
        completed_at: str | None = None,
        expires_at: str | None = None,
        result: ResultEnvelope | None = None,
        error: SimulationErrorInfo | None = None,
    ) -> RunRecord:
        """Validate and atomically apply a lifecycle state transition."""
        record = self.get_run(run_id)
        if new_status == record.status:
            return record
        if new_status not in ALLOWED_TRANSITIONS.get(record.status, set()):
            raise InvalidStateTransitionError(
                f"Illegal state transition from '{record.status}' to '{new_status}' for run '{run_id}'"
            )
        now = _iso_now()
        updates: dict[str, Any] = {"status": new_status}
        if new_status == "running":
            updates["started_at"] = started_at or record.started_at or now
        elif new_status in ("succeeded", "failed"):
            updates["completed_at"] = completed_at or record.completed_at or now
            if result is not None:
                updates["result"] = result
            if error is not None:
                updates["error"] = error
        elif new_status == "expired":
            updates["expires_at"] = expires_at or now
            updates["result"] = None
        updated_record = record.model_copy(update=updates)
        self._atomic_write_record(run_id, updated_record)
        return updated_record

    def start_run(self, run_id: str, started_at: str | None = None) -> RunRecord:
        return self.transition(run_id, "running", started_at=started_at)

    def complete_run(
        self, run_id: str, result: ResultEnvelope, completed_at: str | None = None
    ) -> RunRecord:
        return self.transition(
            run_id, "succeeded", result=result, completed_at=completed_at
        )

    def fail_run(
        self,
        run_id: str,
        error: SimulationErrorInfo | None = None,
        completed_at: str | None = None,
    ) -> RunRecord:
        return self.transition(run_id, "failed", error=error, completed_at=completed_at)

    def prune_run(self, run_id: str, reason: str = "quota_exceeded") -> RunRecord:
        """Prune run storage and mark status visibly as expired."""
        run_dir = self.get_run_dir(run_id)
        shutil.rmtree(run_dir / "artifacts", ignore_errors=True)
        for item in run_dir.iterdir():
            if item.is_file() and item.name != "record.json":
                try:
                    item.unlink()
                except OSError:
                    pass
        return self.transition(run_id, "expired")

    def save_artifact(self, run_id: str, relative_path: str, data: bytes | str) -> Path:
        """Save a file artifact associated with the run under its artifacts directory."""
        if ".." in relative_path or Path(relative_path).is_absolute():
            raise ValueError(f"Invalid artifact relative path: '{relative_path}'")
        artifact_path = self.get_run_dir(run_id) / "artifacts" / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = artifact_path.with_suffix(f".tmp.{uuid.uuid4().hex}")
        mode = "w" if isinstance(data, str) else "wb"
        encoding = "utf-8" if isinstance(data, str) else None
        with open(temp_path, mode, encoding=encoding) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, artifact_path)
        return artifact_path

    def get_total_disk_usage(self) -> int:
        """Calculate total disk bytes consumed across all runs in results root."""
        if not self.results_root.exists():
            return 0
        return sum(
            e.stat().st_size for e in self.results_root.rglob("*") if e.is_file()
        )

    def apply_quota(
        self, max_runs: int | None = None, max_bytes: int | None = None
    ) -> list[str]:
        """Prune oldest active runs until constraints are satisfied, marking them expired."""
        lim_runs = max_runs if max_runs is not None else self.max_runs
        lim_bytes = max_bytes if max_bytes is not None else self.max_bytes
        pruned: list[str] = []
        if lim_runs is not None and lim_runs >= 0:
            active = [r for r in self.list_runs() if r.status != "expired"]
            if len(active) > lim_runs:
                for run in active[: len(active) - lim_runs]:
                    self.prune_run(run.run_id, reason="max_runs_exceeded")
                    pruned.append(run.run_id)
        if (
            lim_bytes is not None
            and lim_bytes >= 0
            and self.get_total_disk_usage() > lim_bytes
        ):
            for run in [r for r in self.list_runs() if r.status != "expired"]:
                if self.get_total_disk_usage() <= lim_bytes:
                    break
                self.prune_run(run.run_id, reason="max_bytes_exceeded")
                if run.run_id not in pruned:
                    pruned.append(run.run_id)
        return pruned
