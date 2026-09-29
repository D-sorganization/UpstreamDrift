"""Parameterized downswing torque programs for the timing experiments.

Every program shares the same constant proximal (shoulder) torque and
differs only in the wrist channel, so sweeps isolate the *timing and sign*
of the distal handoff:

- ``passive``: zero wrist torque throughout (free hinge).
- ``drive_only``: zero wrist torque until ``onset_s``, then a constant
  positive (opening) torque.
- ``restrain_then_drive``: a constant negative (cock-retaining) torque
  until ``onset_s``, then the same positive drive torque.

Signs follow the model convention: positive wrist torque accelerates the
relative wrist angle ``theta2`` toward opening (``theta2 -> 0`` from the
cocked negative angle).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TorqueProgram:
    """A named open-loop torque program for one downswing rollout."""

    name: str
    shoulder_torque_nm: float
    wrist_drive_nm: float
    wrist_restrain_nm: float
    onset_s: float

    def controls(self, horizon: int, dt: float) -> np.ndarray:
        """Render the program to a ``(horizon, 2)`` control array."""
        u = np.zeros((horizon, 2), dtype=float)
        u[:, 0] = self.shoulder_torque_nm
        times = np.arange(horizon, dtype=float) * dt
        before = times < self.onset_s
        u[before, 1] = -abs(self.wrist_restrain_nm)
        u[~before, 1] = self.wrist_drive_nm
        return u


def passive_program(shoulder_torque_nm: float) -> TorqueProgram:
    """Shoulder-driven swing with a completely free wrist hinge."""
    return TorqueProgram(
        name="passive",
        shoulder_torque_nm=shoulder_torque_nm,
        wrist_drive_nm=0.0,
        wrist_restrain_nm=0.0,
        onset_s=float("inf"),
    )


def drive_only_program(
    shoulder_torque_nm: float, wrist_drive_nm: float, onset_s: float
) -> TorqueProgram:
    """Free hinge until ``onset_s``, then constant positive wrist drive."""
    return TorqueProgram(
        name=f"drive_only@{onset_s:.3f}s",
        shoulder_torque_nm=shoulder_torque_nm,
        wrist_drive_nm=wrist_drive_nm,
        wrist_restrain_nm=0.0,
        onset_s=onset_s,
    )


def restrain_then_drive_program(
    shoulder_torque_nm: float,
    wrist_drive_nm: float,
    wrist_restrain_nm: float,
    onset_s: float,
) -> TorqueProgram:
    """Cock-retaining negative torque until ``onset_s``, then drive."""
    return TorqueProgram(
        name=f"restrain_then_drive@{onset_s:.3f}s",
        shoulder_torque_nm=shoulder_torque_nm,
        wrist_drive_nm=wrist_drive_nm,
        wrist_restrain_nm=wrist_restrain_nm,
        onset_s=onset_s,
    )
