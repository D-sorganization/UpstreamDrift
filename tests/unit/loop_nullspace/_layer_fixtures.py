"""Shared small models for the NSX-93 impedance and delayed-feedback tests."""

from __future__ import annotations

import math

import numpy as np

from src.shared.python.loop_nullspace.allocation_spec import AllocationResult
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
)
from src.shared.python.loop_nullspace.replay import (
    ReferenceTrajectory,
    ReplaySetup,
)
from src.shared.python.loop_nullspace.solvers import SolveStatus


class PointMass:
    """One free coordinate, no closure rows: ``m x'' = u`` (plant of the margin).

    The delay margin of the PD law below has a closed form, so this is the
    independent oracle for the discrete margin and for the replay instability.
    """

    m = 2.0

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        return np.array([[self.m]])

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.zeros(1)

    def closure_jacobian(self, q: np.ndarray) -> np.ndarray:
        return np.zeros((0, 1))

    def closure_drift(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.zeros(0)


def analytic_pd_delay_margin(m: float, kp: float, kd: float) -> float:
    """Delay margin [s] of ``u = -(kp x + kd x')(t - tau)`` on ``m x'' = u``.

    Loop ``L(s) = (kd s + kp) e^{-s tau} / (m s^2)``.  The gain crossover solves
    ``m^2 w^4 = kd^2 w^2 + kp^2``; the phase of the delay-free loop there is
    ``atan(kd w / kp) - pi`` so the loop reaches ``-pi`` at ``tau = atan(kd w_c
    / kp) / w_c`` (textbook phase-margin / delay-margin relation).
    """
    a, b = (kd / m) ** 2, (kp / m) ** 2
    wc = math.sqrt(0.5 * (a + math.sqrt(a * a + 4.0 * b)))
    return math.atan(kd * wc / kp) / wc


def zero_torque_allocation(t_end: float, frames: int, n: int) -> AllocationResult:
    """A hand-built all-zero allocation (test fixture only)."""
    times = np.linspace(0.0, t_end, frames)
    torque = np.zeros((frames, n))
    record = QuantityRecord(
        values=torque,
        units="N m",
        allocation_dependence=AllocationDependence.FAMILY_DEPENDENT,
        model_provenance=ModelProvenance("test_point_mass", "1"),
        input_provenance=InputProvenance("test", "zero torque"),
        hypotheses=("H-TORQUE-CAPACITY",),
    )
    return AllocationResult(
        SolveStatus.OPTIMAL, times_s=times, torque=torque, record=record
    )


def hold_reference(t_end: float, n: int) -> ReferenceTrajectory:
    """Rest at the origin: the trajectory the tracking error is measured against."""
    times = np.linspace(0.0, t_end, 5)
    return ReferenceTrajectory(times, np.zeros((5, n)), np.zeros((5, n)))


def point_mass_setup(t_end: float, x0: float = 0.1) -> ReplaySetup:
    return ReplaySetup(
        actuation=np.eye(1),
        q0=np.array([x0]),
        v0=np.zeros(1),
        reference=hold_reference(t_end, 1),
    )
