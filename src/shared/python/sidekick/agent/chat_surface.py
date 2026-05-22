"""Chat-side surface for the Sidekick agent layer (epic #5967 / S8 / #5977).

Both the PyQt6 assistant panel and the React/Tauri ChatPanel render
action chips — small interactive widgets that let the user inspect,
preview, and run each :class:`PlannedStep` the planner emitted. This
module owns the chip's data model and wire format. Per-surface widget
code (PyQt :class:`QWidget`, React component) consumes
:func:`serialize_envelope` output and is intentionally implemented
elsewhere; that keeps this module headless-testable and shields it
from drift in the larger UI packages.

Design contracts:

* **DbC.** :class:`ActionChipModel` is frozen. State transitions return
  new instances; the only "edit" allowed is
  :meth:`with_confirmation` (which is a no-op on non-destructive
  chips). Error chips refuse confirmation.
* **LOD.** The model carries only what the surface needs to render and
  to call back; the underlying :class:`SidekickActionService` and
  handlers are never exposed to UI code.
* **DRY.** Param redaction reuses
  :func:`sidekick.agent.action_audit.redact_secrets` rather than
  rolling its own list. Side-effects labelling mirrors the same closed
  set the descriptors use.
* **Headless-safe.** No PyQt6, no React. Pure data + helpers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .action_audit import redact_secrets
from .action_service import SidekickActionService
from .planner import PlannedStep

__all__ = [
    "ActionChipModel",
    "ActionChipState",
    "ChatActionEnvelope",
    "build_chip",
    "serialize_envelope",
]


class ActionChipState(Enum):
    """User-visible state of one chip.

    ``READY`` — runnable now (user has either no confirmation needed or
    has already confirmed).
    ``LOCKED`` — destructive, awaiting user confirmation.
    ``RUNNING`` — execution in progress (reserved for the UI loop).
    ``COMPLETED`` — execution finished successfully.
    ``ERROR`` — invalid step or runtime failure; rendered with the
    ``error_message`` shown to the user.
    """

    READY = "ready"
    LOCKED = "locked"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ActionChipModel:
    """Wire shape consumed by every chat-side surface."""

    action_id: str
    summary: str
    params: Mapping[str, Any]
    side_effects: str
    reversible: bool
    rationale: str
    state: ActionChipState
    error_message: str = ""

    def with_confirmation(self) -> ActionChipModel:
        """Promote a locked (destructive) chip to ready.

        Idempotent for non-destructive chips (they're already ready).
        Refuses on error chips — the chat layer should render the error
        message and let the user fix the underlying step instead.
        """
        if self.state == ActionChipState.ERROR:
            raise RuntimeError(
                f"cannot confirm an error chip for {self.action_id!r}: "
                f"{self.error_message}"
            )
        if self.state != ActionChipState.LOCKED:
            return self
        new_params = dict(self.params)
        new_params["_confirmed"] = True
        return ActionChipModel(
            action_id=self.action_id,
            summary=self.summary,
            params=new_params,
            side_effects=self.side_effects,
            reversible=self.reversible,
            rationale=self.rationale,
            state=ActionChipState.READY,
            error_message="",
        )


@dataclass(frozen=True, slots=True)
class ChatActionEnvelope:
    """One chat-side delivery: planned steps + matching chips.

    Sent over the WebSocket (Tauri side) and emitted as a Qt signal
    payload (PyQt side). Both surfaces consume the same JSON via
    :func:`serialize_envelope`.
    """

    steps: Sequence[PlannedStep]
    chips: Sequence[ActionChipModel]


def build_chip(
    *,
    step: PlannedStep,
    service: SidekickActionService,
) -> ActionChipModel:
    """Render one :class:`PlannedStep` into its UI model."""
    # Error step → render the error chip directly without consulting
    # the catalog. The planner already attached the message; we trust
    # it.
    if step.is_error:
        return ActionChipModel(
            action_id=step.action_id,
            summary="(invalid step)",
            params=dict(step.params),
            side_effects="read",
            reversible=False,
            rationale=step.rationale,
            state=ActionChipState.ERROR,
            error_message=step.error_message,
        )
    descriptor = {d.action_id: d for d in service.list_actions()}.get(step.action_id)
    if descriptor is None:
        return ActionChipModel(
            action_id=step.action_id,
            summary="(unknown action)",
            params=dict(step.params),
            side_effects="read",
            reversible=False,
            rationale=step.rationale,
            state=ActionChipState.ERROR,
            error_message=f"action {step.action_id!r} is not registered",
        )
    initial_state = (
        ActionChipState.LOCKED
        if descriptor.side_effects == "destructive"
        else ActionChipState.READY
    )
    return ActionChipModel(
        action_id=descriptor.action_id,
        summary=descriptor.summary,
        params=dict(step.params),
        side_effects=descriptor.side_effects,
        reversible=descriptor.reversible,
        rationale=step.rationale,
        state=initial_state,
    )


def serialize_envelope(envelope: ChatActionEnvelope) -> dict[str, Any]:
    """JSON-friendly projection of one envelope.

    Params are redacted via :func:`redact_secrets` so the wire payload
    never carries plaintext secrets, even if a planner step mistakenly
    captured one.
    """
    return {
        "chips": [_serialize_chip(c) for c in envelope.chips],
    }


def _serialize_chip(chip: ActionChipModel) -> dict[str, Any]:
    return {
        "action_id": chip.action_id,
        "summary": chip.summary,
        "params": redact_secrets(chip.params),
        "side_effects": chip.side_effects,
        "reversible": chip.reversible,
        "rationale": chip.rationale,
        "state": chip.state.value,
        "error_message": chip.error_message,
    }
