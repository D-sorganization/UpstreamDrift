"""Multi-step workflow runner for Sidekick (epic #5967 / S7 / #5976).

Composes :class:`SidekickActionService` invocations into ordered
workflows with the four standard recovery strategies (abort, retry,
skip, ask_user). The action_step builder validates inputs at
construction so workflow authors get fast feedback.

Why a focused mini-runner rather than reusing the existing
``shared.python.ai.workflow_engine.WorkflowEngine``: the AI engine is a
600-line module with its own tool-registry coupling, currently in flux
during the multi-agent review (#5907). A 200-line bridge keeps Sidekick
workflows running cleanly today; when the AI engine stabilises and
adopts the action-service shape, swapping the runner is a small,
isolated change because the surface (:class:`SidekickWorkflow`,
:class:`action_step`, :class:`PendingUserDecision`) is what authors
already use.

Design contracts:

* **DbC.** :func:`action_step` validates ``on_failure`` against a closed
  set; :class:`SidekickWorkflow` and :class:`WorkflowStep` are frozen.
* **LOD.** The runner sees ``SidekickActionService.invoke`` and
  ``SidekickActionService.list_actions`` only — never reaches into
  handlers.
* **DRY.** :class:`WorkflowStepStatus` reuses the same vocabulary
  (COMPLETED / FAILED / SKIPPED) that the AI engine's ``StepStatus``
  uses, so the migration path is mechanical.
* **Headless-safe.** No PyQt6 imports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from .action_service import ActionResult, SidekickActionService

__all__ = [
    "PendingUserDecision",
    "RecoveryStrategy",
    "SidekickWorkflow",
    "WorkflowOutcome",
    "WorkflowStep",
    "WorkflowStepResult",
    "WorkflowStepStatus",
    "action_step",
    "run_sidekick_workflow",
]


RecoveryStrategy = Literal["abort", "retry", "skip", "ask_user"]
_VALID_STRATEGIES: frozenset[str] = frozenset({"abort", "retry", "skip", "ask_user"})


class WorkflowStepStatus(Enum):
    """Outcome of one step. Mirrors ``ai.workflow_engine.StepStatus``."""

    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Frozen wire types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    """One step in a Sidekick workflow."""

    action_id: str
    params: Mapping[str, Any]
    on_failure: RecoveryStrategy = "abort"
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        if self.on_failure not in _VALID_STRATEGIES:
            raise ValueError(
                f"on_failure={self.on_failure!r} not in {sorted(_VALID_STRATEGIES)}"
            )


@dataclass(frozen=True, slots=True)
class SidekickWorkflow:
    """A named, ordered sequence of :class:`WorkflowStep`."""

    name: str
    steps: Sequence[WorkflowStep]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("workflow name must be non-empty")


@dataclass(frozen=True, slots=True)
class WorkflowStepResult:
    """Outcome of one step run."""

    action_id: str
    status: WorkflowStepStatus
    value: Any = None
    error_message: str | None = None
    attempts: int = 1


@dataclass(frozen=True, slots=True)
class WorkflowOutcome:
    """Aggregate result of a workflow run."""

    workflow_name: str
    completed: bool
    step_results: tuple[WorkflowStepResult, ...]
    outputs: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PendingUserDecision(Exception):
    """Raised when a step with ``on_failure="ask_user"`` fails.

    The chat layer catches this, surfaces the question to the user, and
    decides whether to retry, skip, or abort by resuming the workflow.
    """

    def __init__(
        self,
        *,
        workflow_name: str,
        action_id: str,
        error_message: str,
    ) -> None:
        super().__init__(
            f"workflow {workflow_name!r} paused at {action_id!r}: {error_message}"
        )
        self.workflow_name = workflow_name
        self.action_id = action_id
        self.error_message = error_message


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def action_step(
    action_id: str,
    params: Mapping[str, Any],
    *,
    on_failure: RecoveryStrategy = "abort",
    rationale: str = "",
) -> WorkflowStep:
    """Build one step that invokes ``action_id`` via the action service."""
    return WorkflowStep(
        action_id=action_id,
        params=dict(params),
        on_failure=on_failure,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_sidekick_workflow(
    workflow: SidekickWorkflow,
    *,
    service: SidekickActionService,
) -> WorkflowOutcome:
    """Run every step in order, applying each step's recovery strategy.

    Returns a :class:`WorkflowOutcome` even when the workflow aborts;
    :attr:`WorkflowOutcome.completed` distinguishes success from
    partial completion. The single exception is ``ask_user``: that
    raises :class:`PendingUserDecision` so the chat layer can intervene.

    Args:
        workflow: The :class:`SidekickWorkflow` to run.
        service: The :class:`SidekickActionService` to dispatch through.

    Returns:
        A :class:`WorkflowOutcome` summarising every step's status.

    Raises:
        PendingUserDecision: When an ``ask_user`` step fails.
    """
    results: list[WorkflowStepResult] = []
    outputs: dict[str, Any] = {}
    completed = True

    for step in workflow.steps:
        step_result, raise_pending = _run_one_step(step, service=service)
        results.append(step_result)
        if step_result.status == WorkflowStepStatus.COMPLETED:
            outputs[step.action_id] = step_result.value
            continue
        if raise_pending:
            raise PendingUserDecision(
                workflow_name=workflow.name,
                action_id=step.action_id,
                error_message=step_result.error_message or "no message",
            )
        if step_result.status == WorkflowStepStatus.SKIPPED:
            continue
        # FAILED + abort: stop the workflow without raising.
        completed = False
        break

    return WorkflowOutcome(
        workflow_name=workflow.name,
        completed=completed,
        step_results=tuple(results),
        outputs=outputs,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _run_one_step(
    step: WorkflowStep, *, service: SidekickActionService
) -> tuple[WorkflowStepResult, bool]:
    """Run one step, applying its retry policy. Returns
    ``(result, raise_pending_user_decision)``."""
    attempt_results: list[ActionResult] = [service.invoke(step.action_id, step.params)]
    if step.on_failure == "retry" and not attempt_results[-1].ok:
        attempt_results.append(service.invoke(step.action_id, step.params))

    last = attempt_results[-1]
    attempts = len(attempt_results)

    if last.ok:
        return (
            WorkflowStepResult(
                action_id=step.action_id,
                status=WorkflowStepStatus.COMPLETED,
                value=last.value,
                attempts=attempts,
            ),
            False,
        )

    if step.on_failure == "skip":
        return (
            WorkflowStepResult(
                action_id=step.action_id,
                status=WorkflowStepStatus.SKIPPED,
                error_message=last.error,
                attempts=attempts,
            ),
            False,
        )

    if step.on_failure == "ask_user":
        return (
            WorkflowStepResult(
                action_id=step.action_id,
                status=WorkflowStepStatus.FAILED,
                error_message=last.error,
                attempts=attempts,
            ),
            True,
        )

    # abort or retry-exhausted
    return (
        WorkflowStepResult(
            action_id=step.action_id,
            status=WorkflowStepStatus.FAILED,
            error_message=last.error,
            attempts=attempts,
        ),
        False,
    )
