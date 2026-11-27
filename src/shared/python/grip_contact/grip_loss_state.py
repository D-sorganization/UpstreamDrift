"""Per-patch grip breakaway state machine with margin hysteresis (#15128)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from src.shared.python.grip_contact.grip_loss_config import GripBreakawayThresholds

__all__ = ["GripPatchState", "PatchTransition", "advance_patch_state"]


class GripPatchState(StrEnum):
    HELD = "HELD"
    SLIPPING = "SLIPPING"
    RELEASED = "RELEASED"


@dataclass
class PatchTransition:
    """One frame advance for a single patch."""

    state: GripPatchState
    entered_slip: bool = False
    entered_release: bool = False


@dataclass
class _PatchRuntime:
    state: GripPatchState = GripPatchState.HELD
    slip_dwell_s: float = 0.0
    release_dwell_s: float = 0.0
    recover_dwell_s: float = 0.0


def advance_patch_state(
    runtime: _PatchRuntime,
    margin: float,
    *,
    dt_s: float,
    thresholds: GripBreakawayThresholds,
) -> PatchTransition:
    """Apply one hysteretic step for a loaded patch margin."""
    if not (dt_s > 0.0 and margin == margin):
        raise ValueError("dt_s must be positive and margin must be finite")

    entered_slip = False
    entered_release = False
    state = runtime.state
    t = thresholds

    if state == GripPatchState.HELD:
        if margin <= t.slip_enter_margin:
            runtime.slip_dwell_s += dt_s
            if runtime.slip_dwell_s >= t.min_slip_dwell_s:
                runtime.state = GripPatchState.SLIPPING
                runtime.slip_dwell_s = 0.0
                entered_slip = True
        else:
            runtime.slip_dwell_s = 0.0
    elif state == GripPatchState.SLIPPING:
        if margin <= t.release_enter_margin:
            runtime.release_dwell_s += dt_s
            runtime.recover_dwell_s = 0.0
            if runtime.release_dwell_s >= t.min_release_dwell_s:
                runtime.state = GripPatchState.RELEASED
                runtime.release_dwell_s = 0.0
                entered_release = True
        elif margin >= t.slip_exit_margin:
            runtime.recover_dwell_s += dt_s
            runtime.release_dwell_s = 0.0
            if runtime.recover_dwell_s >= t.min_slip_dwell_s:
                runtime.state = GripPatchState.HELD
                runtime.recover_dwell_s = 0.0
        else:
            runtime.recover_dwell_s = 0.0
            runtime.release_dwell_s = 0.0
    else:
        runtime.slip_dwell_s = 0.0
        runtime.release_dwell_s = 0.0
        runtime.recover_dwell_s = 0.0

    return PatchTransition(
        runtime.state, entered_slip=entered_slip, entered_release=entered_release
    )
