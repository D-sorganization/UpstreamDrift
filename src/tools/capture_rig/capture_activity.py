"""Bounded, durable per-capture command outcomes; not scientific approval (#9913)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.motion_capture.rig.documents import write_document

ACTIVITY_FILE = "capture_activity.json"
MAX_HISTORY = 100
MAX_BYTES = 1024 * 1024


class CaptureAction(BaseModel):
    """The selected processing context and outcome of one command invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=100)
    action: str = Field(min_length=1, max_length=100)
    context: str = Field(default="", max_length=2000)
    started: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished: datetime | None = None
    status: Literal["running", "finished", "failed", "cancelled"] = "running"
    exit_code: int | None = None

    def complete(self, code: int, *, cancelled: bool = False) -> CaptureAction:
        return self.model_copy(
            update={
                "status": "cancelled"
                if cancelled
                else "failed"
                if code
                else "finished",
                "exit_code": code,
                "finished": datetime.now(UTC),
            }
        )


class CaptureHistory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["capture-activity/1.0"] = "capture-activity/1.0"
    actions: list[CaptureAction] = Field(default_factory=list, max_length=MAX_HISTORY)


def read_activity(root: Path) -> CaptureHistory:
    """Missing history is empty; malformed history must never be overwritten."""
    path = root / ACTIVITY_FILE
    try:
        with path.open("rb") as stream:
            content = stream.read(MAX_BYTES + 1)
    except FileNotFoundError:
        return CaptureHistory()
    if len(content) > MAX_BYTES:
        raise ValueError("Capture activity exceeds the supported size")
    return CaptureHistory.model_validate_json(content)


def save_action(root: Path, action: CaptureAction) -> bool:
    """Upsert atomically, without creating a recording's destination early."""
    if not root.is_dir():
        return False
    history = read_activity(root)
    history.actions = [item for item in history.actions if item.id != action.id]
    history.actions = [*history.actions, action][-MAX_HISTORY:]
    write_document(root / ACTIVITY_FILE, history.model_dump(mode="json"))
    return True


def display_status(action: CaptureAction, active_id: str | None = None) -> str:
    """Persisted running records alone cannot establish that a process is alive."""
    if action.status == "running" and action.id != active_id:
        return "Unconfirmed — May Have Been Interrupted"
    return action.status.title()
