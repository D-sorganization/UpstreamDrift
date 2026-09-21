"""Guided workflow coordinator and transition engine across workspaces (#10518).

Coordinates the 7-stage pipeline:
Capture/Import -> Inspect Targets -> Configure Model -> Fit -> Dynamics -> Compare -> Export.

Enforces:
- Dynamic artifact and disk integrity verification (cryptographic hashes)
- Capabilities & engine requirements (single-view vs 3-D physics)
- Contract distinctions: dynamics cannot inherit a purely kinematic pass
- Reopening, cancellation, retry attempt tracking, and UI state projection parity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
from pathlib import Path
from typing import Any, Final

from src.shared.python.core.contracts.exceptions import StateError
from .artifact_handoff import (
    ArtifactKind,
    ArtifactReference,
)
from .project_store import SessionProjectStore

logger = logging.getLogger(__name__)

PHYSICS_ENGINES: Final[frozenset[str]] = frozenset(
    {"mujoco", "pinocchio", "simscape", "drake"}
)


class WorkflowStepId(str, Enum):
    """The 7 canonical stages of the unified workflow pipeline."""

    CAPTURE_IMPORT = "capture_import"
    INSPECT_TARGETS = "inspect_targets"
    CONFIGURE_MODEL = "configure_model"
    FIT = "fit"
    DYNAMICS = "dynamics"
    COMPARE = "compare"
    EXPORT = "export"


class WorkflowMode(str, Enum):
    """Workflow execution mode tailoring required stages and capabilities."""

    FULL_BODY_3D = "full_body_3d"
    SINGLE_VIEW_COACHING = "single_view_coaching"
    BALL_FLIGHT_ANALYSIS = "ball_flight_analysis"


class StepStatus(str, Enum):
    """Execution status of an individual workflow step."""

    READY = "ready"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELED = "canceled"


STEP_ORDER: Final[tuple[WorkflowStepId, ...]] = (
    WorkflowStepId.CAPTURE_IMPORT,
    WorkflowStepId.INSPECT_TARGETS,
    WorkflowStepId.CONFIGURE_MODEL,
    WorkflowStepId.FIT,
    WorkflowStepId.DYNAMICS,
    WorkflowStepId.COMPARE,
    WorkflowStepId.EXPORT,
)

STEP_TITLES: Final[dict[WorkflowStepId, str]] = {
    WorkflowStepId.CAPTURE_IMPORT: "Capture / Import",
    WorkflowStepId.INSPECT_TARGETS: "Inspect Targets",
    WorkflowStepId.CONFIGURE_MODEL: "Configure Model",
    WorkflowStepId.FIT: "Fit",
    WorkflowStepId.DYNAMICS: "Dynamics",
    WorkflowStepId.COMPARE: "Compare",
    WorkflowStepId.EXPORT: "Export",
}


def _artifact_to_dict(art: ArtifactReference) -> dict[str, Any]:
    kind_val = art.kind.value if isinstance(art.kind, ArtifactKind) else str(art.kind)
    return {
        "artifact_id": art.artifact_id,
        "path": str(art.path),
        "hash": art.hash,
        "schema": art.schema,
        "kind": kind_val,
        "metadata": dict(art.metadata),
    }


def _artifact_from_dict(data: dict[str, Any]) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=data["artifact_id"],
        path=data["path"],
        hash=data["hash"],
        schema=data["schema"],
        kind=ArtifactKind(data["kind"]),
        metadata=dict(data.get("metadata", {})),
    )


@dataclass(frozen=True)
class StepProjection:
    """Read-only projection of a step's live state and readiness."""

    key: str
    title: str
    status: StepStatus
    reason: str = ""
    attempt: int = 1
    available_actions: tuple[str, ...] = ()
    inputs: tuple[ArtifactReference, ...] = ()
    outputs: tuple[ArtifactReference, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Dictionary representation matching Qt and React/Tauri client contracts."""
        return {
            "key": self.key,
            "title": self.title,
            "status": (
                self.status.value
                if isinstance(self.status, StepStatus)
                else str(self.status)
            ),
            "reason": self.reason,
            "attempt": self.attempt,
            "available_actions": list(self.available_actions),
            "inputs": [_artifact_to_dict(a) for a in self.inputs],
            "outputs": [_artifact_to_dict(a) for a in self.outputs],
        }


@dataclass(frozen=True)
class WorkflowProjection:
    """Full projection of the workflow pipeline state and next recommended action."""

    current_step: WorkflowStepId
    mode: WorkflowMode
    steps: tuple[StepProjection, ...]
    next_action: str
    can_advance: bool
    diagnostics: tuple[str, ...] = ()

    def get_step(self, step_id: WorkflowStepId | str) -> StepProjection:
        """Look up a step projection by step ID or string key."""
        target_key = (
            step_id.value if isinstance(step_id, WorkflowStepId) else str(step_id)
        )
        for s in self.steps:
            if s.key == target_key:
                return s
        raise KeyError(f"Step '{target_key}' not found in workflow projection")

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete projection to dictionary."""
        return {
            "current_step": (
                self.current_step.value
                if isinstance(self.current_step, WorkflowStepId)
                else str(self.current_step)
            ),
            "mode": (
                self.mode.value
                if isinstance(self.mode, WorkflowMode)
                else str(self.mode)
            ),
            "steps": [s.to_dict() for s in self.steps],
            "next_action": self.next_action,
            "can_advance": self.can_advance,
            "diagnostics": list(self.diagnostics),
        }


@dataclass
class _StepState:
    step_id: WorkflowStepId
    status: StepStatus
    reason: str = ""
    attempt: int = 1
    inputs: list[ArtifactReference] = field(default_factory=list)
    outputs: list[ArtifactReference] = field(default_factory=list)


def _coerce_step_id(step_id: WorkflowStepId | str) -> WorkflowStepId:
    return (
        step_id if isinstance(step_id, WorkflowStepId) else WorkflowStepId(str(step_id))
    )


class WorkflowCoordinator:
    """Coordinates guided workflow progression across unified workspace environments and readiness checks."""

    def __init__(
        self,
        store: SessionProjectStore,
        session_id: str,
        run_id: str,
        mode: WorkflowMode = WorkflowMode.FULL_BODY_3D,
        available_engines: frozenset[str] | None = None,
        *,
        _is_loading: bool = False,
    ) -> None:
        self._store = store
        self._session_id = session_id
        self._run_id = run_id
        self._mode = mode if isinstance(mode, WorkflowMode) else WorkflowMode(str(mode))
        self._available_engines = (
            PHYSICS_ENGINES
            if available_engines is None
            else frozenset(available_engines)
        )
        self._steps: dict[WorkflowStepId, _StepState] = {}

        if not _is_loading:
            for idx, sid in enumerate(STEP_ORDER):
                if (
                    self._mode == WorkflowMode.SINGLE_VIEW_COACHING
                    and sid == WorkflowStepId.DYNAMICS
                ):
                    st = StepStatus.SKIPPED
                    reason = (
                        "Single-view coaching does not require 3-D physics/dynamics"
                    )
                elif idx == 0:
                    st = StepStatus.READY
                    reason = ""
                else:
                    st = StepStatus.BLOCKED
                    reason = "Waiting for preceding step completion"

                self._steps[sid] = _StepState(
                    step_id=sid,
                    status=st,
                    reason=reason,
                    attempt=1,
                )
            self._save()

    @property
    def run_id(self) -> str:
        """Run identifier."""
        return self._run_id

    @property
    def session_id(self) -> str:
        """Session identifier."""
        return self._session_id

    @property
    def mode(self) -> WorkflowMode:
        """Workflow execution mode."""
        return self._mode

    def _storage_path(self) -> Path:
        return self._store.root / ".workflows" / f"{self._run_id}.json"

    def _save(self) -> None:
        path = self._storage_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self._run_id,
            "session_id": self._session_id,
            "mode": self._mode.value,
            "available_engines": sorted(self._available_engines),
            "steps": {
                k.value: {
                    "step_id": v.step_id.value,
                    "status": v.status.value,
                    "reason": v.reason,
                    "attempt": v.attempt,
                    "inputs": [_artifact_to_dict(a) for a in v.inputs],
                    "outputs": [_artifact_to_dict(a) for a in v.outputs],
                }
                for k, v in self._steps.items()
            },
        }
        tmp = path.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, store: SessionProjectStore, run_id: str) -> WorkflowCoordinator:
        """Load an existing workflow coordinator from the project store."""
        path = store.root / ".workflows" / f"{run_id}.json"
        if not path.exists():
            raise KeyError(f"no workflow found for run_id: {run_id}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise StateError(f"corrupt workflow file at {path}") from exc

        session_id = data["session_id"]
        mode = WorkflowMode(data["mode"])
        engines = frozenset(data.get("available_engines", ()))
        coord = cls(
            store=store,
            session_id=session_id,
            run_id=run_id,
            mode=mode,
            available_engines=engines,
            _is_loading=True,
        )
        for step_key, sdata in data.get("steps", {}).items():
            sid = WorkflowStepId(step_key)
            coord._steps[sid] = _StepState(
                step_id=sid,
                status=StepStatus(sdata["status"]),
                reason=sdata.get("reason", ""),
                attempt=sdata.get("attempt", 1),
                inputs=[_artifact_from_dict(d) for d in sdata.get("inputs", [])],
                outputs=[_artifact_from_dict(d) for d in sdata.get("outputs", [])],
            )
        return coord

    def _get_step(self, step_id: WorkflowStepId | str) -> _StepState:
        return self._steps[_coerce_step_id(step_id)]

    @classmethod
    def entry_from_artifacts(
        cls,
        store: SessionProjectStore,
        session_id: str,
        run_id: str,
        entry_step: WorkflowStepId | str,
        upstream_artifacts: tuple[ArtifactReference, ...] | list[ArtifactReference],
        mode: WorkflowMode = WorkflowMode.FULL_BODY_3D,
        available_engines: frozenset[str] | None = None,
    ) -> WorkflowCoordinator:
        """Initialize workflow entering at a later stage with pre-existing upstream artifacts."""
        target_step = _coerce_step_id(entry_step)
        coord = cls(
            store=store,
            session_id=session_id,
            run_id=run_id,
            mode=mode,
            available_engines=available_engines,
        )
        entry_idx = STEP_ORDER.index(target_step)
        for i in range(entry_idx):
            sid = STEP_ORDER[i]
            if i == entry_idx - 1:
                coord._steps[sid].status = StepStatus.DONE
                coord._steps[sid].outputs = list(upstream_artifacts)
                coord._steps[sid].reason = "Satisfied by imported upstream artifacts"
            else:
                coord._steps[sid].status = StepStatus.SKIPPED
                coord._steps[sid].reason = "Skipped prior to imported entry stage"

        coord._steps[target_step].status = StepStatus.READY
        coord._steps[target_step].inputs = list(upstream_artifacts)
        coord._steps[target_step].reason = ""

        for i in range(entry_idx + 1, len(STEP_ORDER)):
            sid = STEP_ORDER[i]
            if (
                coord._mode == WorkflowMode.SINGLE_VIEW_COACHING
                and sid == WorkflowStepId.DYNAMICS
            ):
                coord._steps[sid].status = StepStatus.SKIPPED
                coord._steps[
                    sid
                ].reason = "Single-view coaching does not require 3-D dynamics"
            else:
                coord._steps[sid].status = StepStatus.BLOCKED
                coord._steps[sid].reason = "Waiting for preceding step completion"

        coord._save()
        return coord

    enter_at = entry_from_artifacts

    def advance_step(
        self,
        step_id: WorkflowStepId | str,
        outputs: tuple[ArtifactReference, ...] | list[ArtifactReference],
    ) -> None:
        """Mark a step DONE with satisfying artifact outputs, advancing downstream flow."""
        sid = _coerce_step_id(step_id)
        if not outputs:
            raise ValueError("cannot advance step without valid outputs")

        # Contract distinction: dynamics cannot inherit a purely kinematic pass
        if sid == WorkflowStepId.DYNAMICS:
            for art in outputs:
                raw_kind = (
                    art.kind.value
                    if isinstance(art.kind, ArtifactKind)
                    else str(art.kind)
                )
                if (
                    raw_kind == ArtifactKind.STATIC_POSE.value
                    or "canonical" in art.schema.lower()
                    or "kinematics" in art.schema.lower()
                ):
                    raise ValueError(
                        f"dynamics cannot inherit a kinematic pass (schema: {art.schema}, kind: {raw_kind})"
                    )

        # Precondition verification: outputs must exist and match hash on disk
        for art in outputs:
            art.verify_on_disk(self._store.root)

        curr = self._steps[sid]
        curr.status = StepStatus.DONE
        curr.outputs = list(outputs)
        curr.reason = ""

        # Flow outputs to the next non-skipped step
        curr_idx = STEP_ORDER.index(sid)
        for next_id in STEP_ORDER[curr_idx + 1 :]:
            next_step = self._steps[next_id]
            if next_step.status == StepStatus.SKIPPED:
                continue
            next_step.inputs = list(outputs)
            if next_step.status == StepStatus.BLOCKED:
                next_step.status = StepStatus.READY
                next_step.reason = ""
            break

        self._save()

    def cancel_step(self, step_id: WorkflowStepId | str, reason: str = "") -> None:
        """Cancel a step with diagnostic justification."""
        step = self._get_step(step_id)
        step.status = StepStatus.CANCELED
        step.reason = reason
        self._save()

    def retry_step(self, step_id: WorkflowStepId | str) -> None:
        """Retry a canceled or failed step, incrementing attempt counter."""
        step = self._get_step(step_id)
        step.attempt += 1
        step.status = StepStatus.READY
        step.reason = ""
        step.outputs.clear()
        self._save()

    def invalidate_from(self, step_id: WorkflowStepId | str) -> None:
        """Invalidate step and downstream pipeline due to upstream input changes."""
        sid = _coerce_step_id(step_id)
        target_idx = STEP_ORDER.index(sid)
        for i in range(target_idx, len(STEP_ORDER)):
            step_key = STEP_ORDER[i]
            step = self._steps[step_key]
            step.outputs.clear()
            if i == target_idx:
                prec_done = i == 0 or self._steps[STEP_ORDER[i - 1]].status in (
                    StepStatus.DONE,
                    StepStatus.SKIPPED,
                )
                step.status = StepStatus.READY if prec_done else StepStatus.BLOCKED
                step.reason = (
                    "Invalidated by upstream input change" if not prec_done else ""
                )
            else:
                if (
                    self._mode == WorkflowMode.SINGLE_VIEW_COACHING
                    and step_key == WorkflowStepId.DYNAMICS
                ):
                    step.status = StepStatus.SKIPPED
                    step.reason = "Single-view coaching does not require 3-D dynamics"
                else:
                    step.status = StepStatus.BLOCKED
                    step.reason = "Waiting for preceding step completion"
                    step.inputs.clear()
        self._save()

    def _project_step(
        self,
        sid: WorkflowStepId,
    ) -> tuple[StepProjection, str | None]:
        st = self._steps[sid]
        status = st.status
        reason = st.reason
        attempt = st.attempt
        inputs = tuple(st.inputs)
        outputs = tuple(st.outputs)

        if status == StepStatus.SKIPPED:
            return (
                StepProjection(
                    key=sid.value,
                    title=STEP_TITLES[sid],
                    status=StepStatus.SKIPPED,
                    reason=reason or "Skipped in current workflow mode",
                    attempt=attempt,
                    available_actions=(),
                    inputs=inputs,
                    outputs=outputs,
                ),
                None,
            )

        if sid == WorkflowStepId.DYNAMICS and self._mode == WorkflowMode.FULL_BODY_3D:
            if not (self._available_engines & PHYSICS_ENGINES):
                if status not in (StepStatus.DONE, StepStatus.CANCELED):
                    status = StepStatus.BLOCKED
                    reason = (
                        "No physics engine available for 3-D dynamics "
                        "(requires mujoco, pinocchio, simscape, or drake)"
                    )

        if inputs and status != StepStatus.CANCELED:
            for art in inputs:
                try:
                    art.verify_on_disk(self._store.root)
                except (FileNotFoundError, ValueError) as exc:
                    status = StepStatus.BLOCKED
                    reason = (
                        f"Missing input file: {exc}"
                        if isinstance(exc, FileNotFoundError)
                        else f"Stale input artifact (hash mismatch): {exc}"
                    )
                    break

        if status == StepStatus.DONE:
            if not outputs:
                status = StepStatus.BLOCKED
                reason = "Missing output artifacts for completed step"
            else:
                for art in outputs:
                    try:
                        art.verify_on_disk(self._store.root)
                    except (FileNotFoundError, ValueError) as exc:
                        status = StepStatus.BLOCKED
                        reason = (
                            f"Missing output file: {exc}"
                            if isinstance(exc, FileNotFoundError)
                            else f"Stale output artifact (hash mismatch): {exc}"
                        )
                        break

        actions: list[str] = []
        if status in (StepStatus.READY, StepStatus.IN_PROGRESS):
            actions = ["advance", "cancel"]
        elif status == StepStatus.DONE:
            actions = ["inspect", "retry"]
        elif status == StepStatus.BLOCKED:
            actions = ["retry", "cancel"]
        elif status in (StepStatus.CANCELED, StepStatus.FAILED):
            actions = ["retry"]

        diagnostic: str | None = None
        if (
            status in (StepStatus.BLOCKED, StepStatus.FAILED, StepStatus.CANCELED)
            and reason
        ):
            diagnostic = f"[{sid.value}] {reason}"

        proj = StepProjection(
            key=sid.value,
            title=STEP_TITLES[sid],
            status=status,
            reason=reason,
            attempt=attempt,
            available_actions=tuple(actions),
            inputs=inputs,
            outputs=outputs,
        )
        return proj, diagnostic

    @staticmethod
    def _compute_next_action(curr: StepProjection) -> str:
        if curr.status == StepStatus.READY:
            return f"Run {curr.title}"
        if curr.status == StepStatus.BLOCKED:
            return f"Resolve blockage in {curr.title}: {curr.reason}"
        if curr.status == StepStatus.CANCELED:
            return f"Retry {curr.title}"
        if curr.status == StepStatus.FAILED:
            return f"Investigate failure in {curr.title}: {curr.reason}"
        if curr.status == StepStatus.DONE:
            return "Workflow complete"
        return f"Continue {curr.title}"

    def get_projection(self) -> WorkflowProjection:
        """Compute live projection and step readiness based on actual disk artifacts."""
        diagnostics: list[str] = []
        step_projections: list[StepProjection] = []

        for sid in STEP_ORDER:
            proj, diag = self._project_step(sid)
            step_projections.append(proj)
            if diag:
                diagnostics.append(diag)

        current_step_id: WorkflowStepId | None = None
        for s in step_projections:
            if s.status not in (StepStatus.DONE, StepStatus.SKIPPED):
                current_step_id = WorkflowStepId(s.key)
                break
        if current_step_id is None:
            current_step_id = WorkflowStepId.EXPORT

        curr_step_proj = next(
            s for s in step_projections if s.key == current_step_id.value
        )
        can_advance = curr_step_proj.status == StepStatus.READY
        next_action = self._compute_next_action(curr_step_proj)

        return WorkflowProjection(
            current_step=current_step_id,
            mode=self._mode,
            steps=tuple(step_projections),
            next_action=next_action,
            can_advance=can_advance,
            diagnostics=tuple(diagnostics),
        )
