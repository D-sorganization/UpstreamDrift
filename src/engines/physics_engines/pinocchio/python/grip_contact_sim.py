"""Pinocchio contact grip on the same input as OpenSim (issue #11739, OSV-7).

Pinocchio has no native contact dynamics for this problem, so the pad contact
is the shared law
(:mod:`src.shared.python.grip_contact.pad_contact`, which applies
:mod:`src.shared.python.motion_matching.contact_law` to each sphere against
the grip cylinder) evaluated at every right-hand side.  The club is the same
free-flyer ``aba`` model as the bushing run, and the hand frames come from
Pinocchio's own forward kinematics of the weld model.  Integration is SciPy
``solve_ivp`` with an implicit method (Radau): the regularised friction is stiff.

Pad contacts that never touch the grip carry no force; the gradient of the
law is continuous at first contact (penetration starts from zero), so no event
handling is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from src.engines.physics_engines.pinocchio.python import grip_bushing as bushing
from src.shared.python.grip_contact import (
    CoordinateSpline,
    CoordinateSwing,
    GripInterface,
    hand_frame_states,
)
from src.shared.python.grip_contact.contact_run import ContactRun, slip_from_frames
from src.shared.python.grip_contact.pad_contact import PadContactModel, pad_wrench
from src.shared.python.grip_contact.parity import GripKineticsSeries

ENGINE = "pinocchio_contact"
SIDES = bushing.SIDES
DEFAULT_RTOL = 1.0e-6
DEFAULT_ATOL = 1.0e-9
DEFAULT_MAX_STEP_S = 2.0e-4


@dataclass(frozen=True)
class IntegratorTolerances:
    """Radau tolerances and step cap for the contact integration."""

    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL
    max_step_s: float = DEFAULT_MAX_STEP_S


class ClubInHands(bushing.ClubOnBushings):
    """Free-flyer club held by pads: the bushing model with the pad law."""

    def __init__(
        self,
        spec: dict[str, Any],
        names: list[str],
        interface: GripInterface,
        pads: PadContactModel,
    ) -> None:
        super().__init__(spec, names, interface)
        self.pads = pads
        self._pad_positions = {s: pads.layout.positions_grip_frame(s) for s in SIDES}
        self.last_pads: dict[str, Any] = {}

    def wrenches(
        self, t: float, y: np.ndarray, hand_source: Any
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Hand frames, club frames and pad wrenches at ``(t, y)``."""
        q, qdot = hand_source(t)
        hands = hand_frame_states(self.kinematics.state(q, qdot), self.interface)
        club = self.club_state(y)
        frames = {s: club.frame(self._offsets[s]) for s in SIDES}
        law = {
            s: pad_wrench(self.pads, s, hands[s], frames[s], self._pad_positions[s])
            for s in SIDES
        }
        self.last_pads = law
        return hands, frames, law


def hold_state(sim: ClubInHands, q: np.ndarray) -> tuple[Any, np.ndarray]:
    """Hand source at rest and the club state at the weld pose for ``q``."""
    still = np.zeros_like(q)
    weld = sim.kinematics.state(q, still)
    y = bushing._initial_state(weld, sim._pin)  # noqa: SLF001
    return (lambda _t: (q, still)), y


def simulate_grip_contact(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    pads: PadContactModel,
    interface: GripInterface | None = None,
    tolerances: IntegratorTolerances | None = None,
    t_end_s: float | None = None,
    hold: bool = False,
) -> ContactRun:
    """Integrate the free club held by pads over the prescribed swing.

    With ``hold`` the hands stay at the first sample's pose (static hold).  The
    club starts with the weld velocity of the first sample, as in the MuJoCo
    contact run (a stiff frictional contact would otherwise see an artificial
    impulse).

    Raises:
        RuntimeError: if the integrator fails.
    """
    tol = tolerances or IntegratorTolerances()
    spec = json.loads(spec_bytes)
    interface = interface or GripInterface.from_spec(spec)
    sim = ClubInHands(spec, list(swing.names), interface, pads)
    times = swing.time_s
    if t_end_s is not None:
        times = times[times <= t_end_s + 1e-12]
    if hold:
        source, y0 = hold_state(sim, np.asarray(swing.q[0], float))
    else:
        spline = CoordinateSpline(swing.time_s, swing.q)
        source = spline.evaluate
        weld0 = sim.kinematics.state(*source(float(times[0])))
        y0 = bushing._initial_state(weld0, sim._pin)  # noqa: SLF001
        y0[7:10] = weld0.rotation.T @ weld0.velocity_m_s
        y0[10:13] = weld0.rotation.T @ weld0.omega_rad_s
    sol = solve_ivp(
        lambda t, y: sim.derivatives(t, y, source),
        (float(times[0]), float(times[-1])),
        y0,
        method="Radau",
        t_eval=times,
        rtol=tol.rtol,
        atol=tol.atol,
        max_step=tol.max_step_s,
    )
    if not sol.success:
        raise RuntimeError(f"pinocchio contact integration failed: {sol.message}")
    wrench: dict[str, tuple[list, list]] = {s: ([], []) for s in SIDES}
    hand_pose: dict[str, tuple[list, list]] = {s: ([], []) for s in SIDES}
    club_pose: dict[str, tuple[list, list]] = {s: ([], []) for s in SIDES}
    normal: dict[str, list] = {s: [] for s in SIDES}
    club_rot = []
    for k, t in enumerate(times):
        y = sol.y[:, k]
        hands, frames, law = sim.wrenches(float(t), y, source)
        for s in SIDES:
            wrench[s][0].append(law[s].force_n)
            wrench[s][1].append(law[s].moment_nm)
            normal[s].append(law[s].normal_force_n)
            hand_pose[s][0].append(hands[s].rotation)
            hand_pose[s][1].append(hands[s].position_m)
            club_pose[s][0].append(frames[s].rotation)
            club_pose[s][1].append(frames[s].position_m)
        club_rot.append(sim.club_state(y).rotation)
    arrays = {
        name: {s: (np.array(d[s][0]), np.array(d[s][1])) for s in SIDES}
        for name, d in (("w", wrench), ("h", hand_pose), ("c", club_pose))
    }
    series = GripKineticsSeries.from_frames(
        ENGINE,
        times,
        arrays["w"],
        arrays["h"],
        arrays["c"],
        np.array(club_rot),
        metadata={
            "integrator": "SciPy solve_ivp Radau, error controlled",
            "rtol": tol.rtol,
            "atol": tol.atol,
            "rhs_evaluations": int(sol.nfev),
            "force_law": "shared pad contact law, Pinocchio aba",
        },
    )
    axial, roll = slip_from_frames(arrays["h"], arrays["c"])
    return ContactRun(series, {s: np.array(normal[s]) for s in SIDES}, roll, axial)


__all__ = ["ClubInHands", "IntegratorTolerances", "simulate_grip_contact"]
