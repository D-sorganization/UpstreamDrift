"""Fault classification and recovery decisions for MS-105."""

from __future__ import annotations

from .contracts import FaultKind, RecoveryDecision

__all__ = ["classify_fault", "decide_recovery"]


def classify_fault(exc: BaseException) -> FaultKind:
    """Map an exception to a named fault kind (fail-closed on unknowns)."""
    from .contracts import (
        DiskFullError,
        EngineUnavailableError,
        JobCancelledError,
        UnsupportedHostError,
    )

    if isinstance(exc, JobCancelledError):
        return FaultKind.CANCEL
    if isinstance(exc, DiskFullError):
        return FaultKind.DISK_FULL
    if isinstance(exc, EngineUnavailableError):
        return FaultKind.ENGINE_ABSENT
    if isinstance(exc, UnsupportedHostError):
        return FaultKind.HOST_UNAVAILABLE
    if isinstance(exc, OSError):
        errno = getattr(exc, "errno", None)
        message = str(exc).lower()
        if errno in {28, 112} or "no space" in message or "enospc" in message:
            return FaultKind.DISK_FULL
    return FaultKind.UNKNOWN


def decide_recovery(
    fault: FaultKind,
    *,
    has_compatible_checkpoint: bool,
) -> RecoveryDecision:
    """Choose recovery without inventing silent success."""
    if fault == FaultKind.CANCEL:
        return RecoveryDecision.PRESERVE_AND_FAIL
    if fault == FaultKind.DISK_FULL:
        return RecoveryDecision.PRESERVE_AND_FAIL
    if fault == FaultKind.HOST_UNAVAILABLE:
        return RecoveryDecision.FAIL_CLOSED
    if fault == FaultKind.ENGINE_ABSENT:
        return RecoveryDecision.FAIL_CLOSED
    if fault == FaultKind.CORRUPT_ARTIFACT:
        return RecoveryDecision.FAIL_CLOSED
    if fault == FaultKind.INCOMPATIBLE_CHECKPOINT:
        return RecoveryDecision.RESTART_FRESH
    if fault in {
        FaultKind.WORKER_CRASH,
        FaultKind.APP_RESTART,
        FaultKind.REMOTE_DISCONNECT,
    }:
        if has_compatible_checkpoint:
            return RecoveryDecision.RESUME_COMPATIBLE
        return RecoveryDecision.RESTART_FRESH
    return RecoveryDecision.FAIL_CLOSED
