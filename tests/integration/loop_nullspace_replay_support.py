"""Support for the anthro open-loop versus feedback-assisted replay (NSX-31).

Chain: the ground-supported anthro record -> frames that decompose
(``AnthroBench.frame``, NSX-60) -> the longest run of consecutive strided
frames whose free feasible set is non-empty under the ASSUMED capacity prior
-> ``allocate_trajectory`` (NSX-23) over that window -> replay of the allocated
torques plus the allocated foot-ground loads through the Pinocchio constrained
provider (NSX-30), open loop and feedback-assisted.

Window rule (declared, not tuned): frames ``0, 3, ..., 597`` are the NSX-22
benchmark set; a frame is *usable* when its free feasible set is non-empty
(75 of 199 are not: the recorded efforts exceed the ASSUMED capacities).  The
window is the longest run of strided frames that are all usable and adjacent
(3 record steps apart); ties go to the earlier run.  The initial state is the
closure-projected measured state of the first window frame and is the only
state the replay is given.

The "feedback" is the state-dependent acceleration-space PD law of
:class:`AccelerationFeedbackLaw` on the 35 actuated coordinates; the six root
coordinates are unactuated and receive no feedback.  The foot-ground load is
the allocated ``f_g`` held on the contact set of the allocation frame,
re-projected through ``J_g(q)^T`` at the replayed configuration: it is a known
external load, not a contact model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.engines.feedback_comparison import ReplayPolicy
from src.shared.python.loop_nullspace.allocation import (
    ObjectiveSpec,
    allocate_trajectory,
    torque_scale_from_capacity,
)
from src.shared.python.loop_nullspace.allocation_spec import AllocationResult
from src.shared.python.loop_nullspace.constrained_dynamics import (
    constrained_acceleration,
    kkt_projector,
)
from src.shared.python.loop_nullspace.provider import AdapterConstrainedProvider
from src.shared.python.loop_nullspace.replay import (
    AccelerationFeedbackLaw,
    ReferenceTrajectory,
    ReplayReport,
    ReplaySettings,
    ReplaySetup,
    replay,
)
from src.shared.python.loop_nullspace.replay_ground import (
    GroundLoadSchedule,
    ground_forces_of_allocation,
)
from tests.integration.loop_nullspace_lower_body_support import (
    AnthroBench,
    Frame,
)

STRIDE = 3
RATE_WEIGHT = 0.1
DIVERGENCE_TOL = 0.05
PROBE_TIMES_S = (0.01, 0.03, 0.05, 0.1)
#: (natural frequency [rad/s], RK4 sub-steps per 9 ms allocation interval).
FEEDBACK_SWEEP = ((50.0, 3), (100.0, 6), (150.0, 6), (300.0, 12))


def usable_runs(frames: Sequence[Frame], empty: Sequence[bool]) -> list[list[int]]:
    """Runs of adjacent (``STRIDE`` apart) frames whose feasible set is non-empty.

    Returns lists of positions into ``frames``.
    """
    runs: list[list[int]] = []
    current: list[int] = []
    for pos, (frame, is_empty) in enumerate(zip(frames, empty, strict=True)):
        if is_empty:
            current = []
            continue
        if current and frame.index - frames[current[-1]].index != STRIDE:
            current = []
        if not current:
            runs.append(current)
        current.append(pos)
    return runs


def longest_run(runs: Sequence[list[int]]) -> list[int]:
    """The longest run (earliest on ties)."""
    if not runs:
        raise ValueError("no usable frames")
    return max(runs, key=len) if len({len(r) for r in runs}) > 1 else runs[0]


@dataclass
class ReplayCase:
    """Everything the two replays share."""

    bench: AnthroBench
    window: list[Frame]
    allocation: AllocationResult
    provider: AdapterConstrainedProvider
    selection: np.ndarray
    times: np.ndarray
    q_ref: np.ndarray
    v_ref: np.ndarray
    setup: ReplaySetup
    capacity_scale: np.ndarray
    runs_summary: list[tuple[int, int, int]]
    n_usable: int

    def response(self, q: np.ndarray) -> np.ndarray:
        """Actuated acceleration response ``S P_c S^T`` at ``q``."""
        mass = self.provider.mass_matrix(q)
        proj = kkt_projector(mass, self.provider.closure_jacobian(q))
        return np.asarray(self.selection @ proj @ self.selection.T)


def build_case(
    bench: AnthroBench, frames: Sequence[Frame], free_sets: Sequence[Any]
) -> ReplayCase:
    """Select the window, allocate it and assemble the replay inputs."""
    empty = [fs.is_empty() for fs in free_sets]
    runs = usable_runs(frames, empty)
    chosen = longest_run(runs)
    window = [frames[p] for p in chosen]
    sets = [free_sets[p] for p in chosen]
    times = np.asarray(bench.record["time_s"])[[f.index for f in window]]
    scale = torque_scale_from_capacity(bench.capacity)
    allocation = allocate_trajectory(
        sets,
        ObjectiveSpec(
            times_s=times,
            torque_scale=scale,
            effort_weight=1.0,
            rate_weight=RATE_WEIGHT,
        ),
    )
    if allocation.status.name != "OPTIMAL":
        raise RuntimeError(f"window allocation not OPTIMAL: {allocation.status}")
    topo = window[0].snapshot.topology
    selection = np.eye(len(bench.order))[list(topo.actuated_indices)]
    q = np.array([[f.coordinates[n] for n in bench.order] for f in window])
    v = np.array([[f.rates[n] for n in bench.order] for f in window])
    provider = AdapterConstrainedProvider(bench.adapter)
    forces = ground_forces_of_allocation(
        allocation.x, topo.n_actuated, window[0].snapshot.J_c.shape[0]
    )
    ground = GroundLoadSchedule(
        bench.adapter, times, [f.contact_names for f in window], forces
    )
    reference = ReferenceTrajectory(times, q, v)
    return ReplayCase(
        bench,
        window,
        allocation,
        provider,
        selection,
        times,
        q,
        v,
        ReplaySetup(selection, q[0], v[0], reference=reference, external_load=ground),
        scale,
        [(frames[r[0]].index, frames[r[-1]].index, len(r)) for r in runs],
        len(frames) - sum(empty),
    )


def acceleration_consistency(case: ReplayCase, every: int = 9) -> dict[str, float]:
    """Instantaneous check: allocated loads at measured states give measured ``a``.

    Per sampled frame, ``max |a_replay - a_measured|`` where ``a_replay`` is the
    KKT-projector acceleration under ``S^T tau + J_g^T f_g`` and ``a_measured``
    the snapshot's.  It isolates the load map from the time integration.
    """
    ground = case.setup.external_load
    worst_abs, worst_rel = 0.0, 0.0
    for k in range(0, len(case.window), every):
        q, v = case.q_ref[k], case.v_ref[k]
        load = case.selection.T @ case.allocation.torque[k] + ground(case.times[k], q)
        acc, _ = constrained_acceleration(
            case.provider.mass_matrix(q),
            case.provider.bias_forces(q, v),
            load,
            case.provider.closure_jacobian(q),
            case.provider.closure_drift(q, v),
        )
        measured = case.window[k].snapshot.a
        diff = float(np.max(np.abs(acc - measured)))
        worst_abs = max(worst_abs, diff)
        worst_rel = max(worst_rel, diff / float(np.max(np.abs(measured))))
    return {"max_abs": worst_abs, "max_relative_to_peak_a": worst_rel}


def _error_curve(report: ReplayReport) -> tuple[np.ndarray, np.ndarray]:
    return report.times_s - report.times_s[0], np.max(
        np.abs(report.q - report.reference_q), axis=1
    )


def _probe(report: ReplayReport) -> dict[str, float | None]:
    t, err = _error_curve(report)
    return {
        f"{1000 * p:g}ms": float(np.interp(p, t, err)) if p <= t[-1] else None
        for p in PROBE_TIMES_S
    }


def _growth_rate(report: ReplayReport) -> float | None:
    """Exponential rate [1/s] of the error between 6 and 30 ms (log-linear fit)."""
    t, err = _error_curve(report)
    keep = (t >= 0.006) & (t <= 0.030)
    if keep.sum() < 3 or np.any(err[keep] <= 0.0):
        return None
    return float(np.polyfit(t[keep], np.log(err[keep]), 1)[0])


def _worst_coordinates(report: ReplayReport, order: Sequence[str], at_s: float):
    t, _ = _error_curve(report)
    k = int(np.argmin(np.abs(t - at_s)))
    err = np.abs(report.q[k] - report.reference_q[k])
    return [(order[j], float(err[j])) for j in np.argsort(-err)[:3]]


def summarize(case: ReplayCase, report: ReplayReport) -> dict[str, Any]:
    """JSON-ready record of one replay (no value is hidden when it diverges)."""
    t0 = float(report.times_s[0])
    ttd = report.time_to_divergence_s
    out: dict[str, Any] = {
        "mode": report.mode.value,
        "policy": report.policy.value,
        "evidence_label": report.evidence_label,
        "settings": {
            "substeps": report.settings.substeps,
            "interpolation": report.settings.interpolation,
        },
        "nodes": int(report.times_s.size),
        "horizon_s": float(report.times_s[-1] - t0),
        "terminated_reason": report.terminated_reason,
        "divergence_tolerance": report.divergence_tolerance,
        "time_to_divergence_s": None if ttd is None else float(ttd - t0),
        "max_state_error": report.max_state_error,
        "state_error_at": _probe(report),
        "growth_rate_per_s_6_to_30ms": _growth_rate(report),
        "worst_coordinates_at_50ms": _worst_coordinates(report, case.bench.order, 0.05),
        "closure_velocity_drift_max": report.closure_drift.velocity_max,
        "closure_position_drift_max": report.closure_drift.position_max,
        "stabilisation": None,
    }
    if ttd is not None:
        before = report.times_s <= ttd
        drift = report.closure_drift
        out["closure_drift_max_until_divergence"] = {
            "velocity": float(np.max(drift.velocity[before])),
            "position": None
            if drift.position is None
            else float(np.max(drift.position[before])),
        }
    budget = report.feedback_budget
    if budget is not None:
        ratio = np.abs(report.feedback_torque) / case.capacity_scale
        order = [
            case.bench.order[i]
            for i in case.window[0].snapshot.topology.actuated_indices
        ]
        top = np.argsort(-ratio.max(axis=0))[:3]
        out["feedback"] = {
            "law": report.feedback_law.name,
            "peak_norm_ratio": budget.peak_norm_ratio,
            "energy_ratio": budget.energy_ratio,
            "rms_ratio": budget.rms_ratio,
            "abs_work_ratio": budget.abs_work_ratio,
            "abs_work_feedback_j": budget.abs_work_feedback_j,
            "abs_work_allocated_j": budget.abs_work_allocated_j,
            "peak_over_assumed_capacity_max": float(ratio.max()),
            "joints_above_capacity": int(np.sum(ratio.max(axis=0) > 1.0)),
            "worst_joints_over_capacity": [
                (order[j], float(ratio.max(axis=0)[j])) for j in top
            ],
        }
    return out


def run_open_loop(case: ReplayCase, **settings: Any) -> ReplayReport:
    """Open-loop replay (``INDEPENDENT_TIME_ONLY``) from the window's first state."""
    return replay(
        case.allocation,
        case.provider,
        policy=ReplayPolicy.INDEPENDENT_TIME_ONLY,
        setup=case.setup,
        settings=ReplaySettings(divergence_tolerance=DIVERGENCE_TOL, **settings),
    )


def run_feedback(
    case: ReplayCase, omega: float, substeps: int, interpolation: str = "linear"
) -> ReplayReport:
    """Feedback-assisted replay (``STATE_FEEDBACK``) with the acceleration PD law."""
    law = AccelerationFeedbackLaw(
        case.response, case.selection.shape[0], omega, zeta=1.0
    )
    return replay(
        case.allocation,
        case.provider,
        policy=ReplayPolicy.STATE_FEEDBACK,
        setup=case.setup,
        feedback=law,
        settings=ReplaySettings(
            substeps=substeps,
            interpolation=interpolation,
            divergence_tolerance=DIVERGENCE_TOL,
        ),
    )
