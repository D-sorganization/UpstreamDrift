"""Kinematic and contact admission gate (NSX-14, R7).

``admit`` refuses allocation on a trajectory that violates the loop at the
position, velocity or acceleration level, whose time base is not usable, whose
derivatives are not consistent with the sampled states, or whose declared stance
feet move.  The three closure levels are::

    Phi       = closure_pose_residual(q)           (needs a ``ClosurePlant``)
    Phi_dot   = J_c v
    Phi_ddot  = J_c a + Jdot_v

and are scaled by characteristic length ``L`` and time ``T`` (``Phi/L``,
``Phi_dot T/L``, ``Phi_ddot T^2/L``) before they meet the gate.  Repair happens
only on request (``repair="project_to_closure"``): it reuses
``same_input.closure.project_to_closure`` on every frame that fails at the
position or velocity level, projects the acceleration onto the tangent
acceleration set of the projected state, re-evaluates, and records every
correction so the original state can be recovered.  Admission is not validation
of the realism of the kinematics.  No engine is imported.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.contracts import ClosedLoopSnapshot
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    QuantityRecord,
)
from src.shared.python.motion_matching.same_input.closure import (
    ClosurePlant,
    project_to_closure,
)

__all__ = [
    "AdmissionGates",
    "AdmissionReport",
    "AdmittedTrajectory",
    "ContactCheck",
    "FrameResiduals",
    "RepairCorrection",
    "RepairRecord",
    "TimeSyncCheck",
    "Verdict",
    "VerdictKind",
    "admit",
]

Rebuild = Callable[[Array, Array, Array], ClosedLoopSnapshot]
_LEVELS = ("phi", "phi_dot", "phi_ddot")
_LEVEL_NAME = {"phi": "position", "phi_dot": "velocity", "phi_ddot": "acceleration"}
_SCALE_CONVENTION = "C-SCALE-ADMISSION: Phi/L, Phi_dot*T/L, Phi_ddot*T^2/L"
_TOKEN = object()


class VerdictKind(Enum):
    """Outcome of an admission run."""

    ADMITTED = "ADMITTED"
    REJECTED = "REJECTED"
    REPAIRED = "REPAIRED"


@dataclass(frozen=True)
class AdmissionGates:
    """Admission thresholds (scaled, dimensionless) and the schedule inputs.

    ``phi_tol`` etc. default to the NSX-00 gate ``1e-6``.  ``length_scale_m``
    and ``time_scale_s`` are the characteristic scales.  ``require_position``
    (default ``True``) makes a missing ``ClosurePlant`` a rejection; setting it
    ``False`` is an explicit, recorded skip of the position level.
    ``derivative_method`` records how velocities relate to positions:
    ``"analytic"`` (the snapshot's ``v`` and ``a`` are engine outputs, no
    sampling noise) or ``"central_difference"`` (``v`` is compared with the
    central difference of ``q``; the worst scaled mismatch is the noise floor
    and must not exceed ``derivative_tol``).  ``max_dt_jitter`` (relative
    spread of the time step) is checked only when given.  ``contact_schedule``
    names the stance contacts of each frame.
    """

    length_scale_m: float = 1.0
    time_scale_s: float = 1.0
    phi_tol: float = 1e-6
    phi_dot_tol: float = 1e-6
    phi_ddot_tol: float = 1e-6
    require_position: bool = True
    derivative_method: Literal["analytic", "central_difference"] = "analytic"
    derivative_tol: float = 1e-6
    max_dt_jitter: float | None = None
    stance_velocity_tol: float = 1e-6
    contact_schedule: tuple[frozenset[str], ...] | None = None

    def __post_init__(self) -> None:
        for name in (
            "length_scale_m",
            "time_scale_s",
            "phi_tol",
            "phi_dot_tol",
            "phi_ddot_tol",
            "derivative_tol",
            "stance_velocity_tol",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"AdmissionGates.{name} must be positive, got {value}")
        if self.derivative_method not in ("analytic", "central_difference"):
            raise ValueError(
                "derivative_method must be 'analytic' or 'central_difference'"
            )
        if self.max_dt_jitter is not None and not self.max_dt_jitter >= 0.0:
            raise ValueError("max_dt_jitter must be non-negative")
        if self.contact_schedule is not None:
            object.__setattr__(
                self,
                "contact_schedule",
                tuple(frozenset(s) for s in self.contact_schedule),
            )

    def tolerance(self, level: str) -> float:
        """Gate for ``level`` in ``("phi", "phi_dot", "phi_ddot")``."""
        return {
            "phi": self.phi_tol,
            "phi_dot": self.phi_dot_tol,
            "phi_ddot": self.phi_ddot_tol,
        }[level]


@dataclass(frozen=True)
class FrameResiduals:
    """Scaled max-abs closure residuals of one frame (``nan`` = not evaluated)."""

    index: int
    phi: float
    phi_dot: float
    phi_ddot: float


@dataclass(frozen=True)
class TimeSyncCheck:
    """Time-base check: ``dt`` range and relative jitter."""

    ok: bool
    dt_min: float
    dt_max: float
    max_dt_jitter: float


@dataclass(frozen=True)
class ContactCheck:
    """Stance-foot consistency against the contact schedule."""

    evaluated: bool
    stance_velocity: tuple[float, ...]
    failing_frames: tuple[int, ...]
    unknown: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class RepairCorrection:
    """One repaired frame: before/after states and correction magnitudes."""

    frame: int
    q_before: Array
    v_before: Array
    a_before: Array
    q_after: Array
    v_after: Array
    a_after: Array
    dq_norm: float
    dv_norm: float
    da_norm: float


@dataclass(frozen=True)
class RepairRecord:
    """Audit of a ``project_to_closure`` repair; the originals are retained."""

    method: str
    corrections: tuple[RepairCorrection, ...]

    def original_state(self, frame: int) -> tuple[Array, Array, Array]:
        """Return the pre-repair ``(q, v, a)`` of ``frame``."""
        for c in self.corrections:
            if c.frame == frame:
                return c.q_before, c.v_before, c.a_before
        raise ValueError(f"frame {frame} was not repaired")


@dataclass(frozen=True)
class Verdict:
    """``ADMITTED``, ``REJECTED(reasons)`` or ``REPAIRED(record)``."""

    kind: VerdictKind
    reasons: tuple[str, ...] = ()
    failing_frames: tuple[int, ...] = ()
    repair: RepairRecord | None = None


@dataclass(frozen=True)
class AdmissionReport:
    """Everything the gate measured, with the (possibly repaired) frames."""

    verdict: Verdict
    frames: tuple[FrameResiduals, ...]
    snapshots: tuple[ClosedLoopSnapshot, ...]
    times: tuple[float, ...]
    derivative_method: str
    noise_floor: float
    time_sync: TimeSyncCheck
    contact: ContactCheck
    position_evaluated: bool
    gates: AdmissionGates

    def residual_record(self, level: str) -> QuantityRecord:
        """Per-frame scaled residual of ``level`` as a :class:`QuantityRecord`."""
        if level not in _LEVELS:
            raise ValueError(f"level must be one of {_LEVELS}, got {level!r}")
        values = np.array([getattr(f, level) for f in self.frames])
        first = self.snapshots[0]
        common = {
            "units": "1",
            "model_provenance": first.model_provenance,
            "input_provenance": first.input_provenance,
            "conventions": (_SCALE_CONVENTION,),
        }
        if np.isnan(values).any():
            return QuantityRecord(
                values=np.full(values.shape, np.nan),
                allocation_dependence=AllocationDependence.UNAVAILABLE,
                reason=f"{_LEVEL_NAME[level]} residual not evaluated: no plant",
                **common,
            )
        return QuantityRecord(
            values=values,
            allocation_dependence=AllocationDependence.INVARIANT,
            **common,
        )


@dataclass(frozen=True)
class AdmittedTrajectory:
    """A trajectory that passed admission; only :meth:`from_report` builds one."""

    report: AdmissionReport
    snapshots: tuple[ClosedLoopSnapshot, ...]
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _TOKEN:
            raise TypeError("AdmittedTrajectory is built with from_report()")

    @classmethod
    def from_report(cls, report: AdmissionReport) -> AdmittedTrajectory:
        """Wrap an ``ADMITTED`` or ``REPAIRED`` report.

        Raises:
            ValueError: If the verdict is ``REJECTED``.
        """
        kind = report.verdict.kind
        if kind is VerdictKind.REJECTED:
            raise ValueError(
                "cannot admit a REJECTED trajectory: "
                + "; ".join(report.verdict.reasons)
            )
        return cls(report=report, snapshots=report.snapshots, _token=_TOKEN)


# ------------------------------------------------------------ measurements --


def _scaled_closure(
    snap: ClosedLoopSnapshot, gates: AdmissionGates, plant: ClosurePlant | None
) -> tuple[float, float, float]:
    length, time = gates.length_scale_m, gates.time_scale_s
    phi = float("nan")
    if plant is not None:
        phi = float(np.max(np.abs(plant.closure_pose_residual(snap.q)))) / length
    jac = snap.closure_rows()
    phi_dot = float(np.max(np.abs(jac @ snap.v))) * time / length
    phi_ddot = float(np.max(np.abs(jac @ snap.a + snap.Jdot_v))) * time**2 / length
    return phi, phi_dot, phi_ddot


def _check_time(times: Array, gates: AdmissionGates) -> TimeSyncCheck:
    dt = np.diff(times)
    jitter = float(np.max(np.abs(dt - np.median(dt))) / np.median(dt))
    ok = gates.max_dt_jitter is None or jitter <= gates.max_dt_jitter
    return TimeSyncCheck(ok, float(dt.min()), float(dt.max()), jitter)


def _derivative_mismatch(
    snaps: Sequence[ClosedLoopSnapshot], times: Array, gates: AdmissionGates
) -> tuple[float, tuple[int, ...]]:
    worst, failing = 0.0, []
    for k in range(1, len(snaps) - 1):
        v_fd = (snaps[k + 1].q - snaps[k - 1].q) / (times[k + 1] - times[k - 1])
        err = float(np.max(np.abs(snaps[k].v - v_fd)))
        err *= gates.time_scale_s / gates.length_scale_m
        worst = max(worst, err)
        if err > gates.derivative_tol:
            failing.append(k)
    return worst, tuple(failing)


def _check_contacts(
    snaps: Sequence[ClosedLoopSnapshot], gates: AdmissionGates
) -> ContactCheck:
    schedule = gates.contact_schedule
    if schedule is None:
        return ContactCheck(False, (), ())
    if len(schedule) != len(snaps):
        raise ValueError(
            f"contact_schedule has {len(schedule)} entries for {len(snaps)} frames"
        )
    speeds, failing, unknown = [], [], []
    for k, (snap, stance) in enumerate(zip(snaps, schedule, strict=True)):
        names = [c.name for c in snap.foot_contacts]
        worst = 0.0
        for name in sorted(stance):
            if name not in names:
                unknown.append((k, name))
                continue
            row = 3 * names.index(name)
            if snap.J_g is None or snap.J_g.shape[0] < row + 3:
                raise ValueError(f"frame {k}: J_g has no rows for contact {name!r}")
            vel = snap.J_g[row : row + 3] @ snap.v
            worst = max(worst, float(np.max(np.abs(vel))))
        worst *= gates.time_scale_s / gates.length_scale_m
        speeds.append(worst)
        if worst > gates.stance_velocity_tol:
            failing.append(k)
    return ContactCheck(True, tuple(speeds), tuple(failing), tuple(unknown))


# ------------------------------------------------------------------ repair --


def _repair_frame(
    k: int, snap: ClosedLoopSnapshot, plant: ClosurePlant, rebuild: Rebuild
) -> tuple[ClosedLoopSnapshot, RepairCorrection]:
    proj = project_to_closure(plant, snap.q, snap.v)
    probe = rebuild(proj.q, proj.v, snap.a)
    jac = probe.closure_rows()
    a_new = snap.a - np.linalg.pinv(jac) @ (jac @ snap.a + probe.Jdot_v)
    fixed = rebuild(proj.q, proj.v, a_new)
    return fixed, RepairCorrection(
        frame=k,
        q_before=snap.q,
        v_before=snap.v,
        a_before=snap.a,
        q_after=fixed.q,
        v_after=fixed.v,
        a_after=fixed.a,
        dq_norm=float(np.linalg.norm(fixed.q - snap.q)),
        dv_norm=float(np.linalg.norm(fixed.v - snap.v)),
        da_norm=float(np.linalg.norm(fixed.a - snap.a)),
    )


def _failing(
    frames: Sequence[FrameResiduals], gates: AdmissionGates
) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for f in frames:
        for level in _LEVELS:
            value = getattr(f, level)
            if np.isnan(value):
                continue
            if value > gates.tolerance(level):
                msg = (
                    f"frame {f.index}: {_LEVEL_NAME[level]} closure residual "
                    f"{value:.3e} exceeds {gates.tolerance(level):.1e}"
                )
                out.setdefault(f.index, []).append(msg)
    return out


# ------------------------------------------------------------------- admit --


def _validate_inputs(
    trajectory: Sequence[ClosedLoopSnapshot],
    times: ArrayLike,
    repair: str,
    plant: ClosurePlant | None,
    rebuild: Rebuild | None,
) -> Array:
    if repair not in ("none", "project_to_closure"):
        raise ValueError(
            f"repair must be 'none' or 'project_to_closure', got {repair!r}"
        )
    if len(trajectory) < 3:
        raise ValueError(f"admission needs at least 3 frames, got {len(trajectory)}")
    if not all(isinstance(s, ClosedLoopSnapshot) for s in trajectory):
        raise TypeError("trajectory must contain ClosedLoopSnapshot objects")
    t = np.asarray(times, dtype=np.float64)
    if t.shape != (len(trajectory),):
        raise ValueError("one time per frame is required")
    if not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0.0):
        raise ValueError("times must be finite and strictly increasing")
    if repair == "project_to_closure" and (plant is None or rebuild is None):
        raise ValueError("repair='project_to_closure' needs both plant and rebuild")
    return t


def _measure(
    snaps: Sequence[ClosedLoopSnapshot],
    gates: AdmissionGates,
    plant: ClosurePlant | None,
) -> tuple[FrameResiduals, ...]:
    return tuple(
        FrameResiduals(k, *_scaled_closure(s, gates, plant))
        for k, s in enumerate(snaps)
    )


def admit(
    trajectory: Sequence[ClosedLoopSnapshot],
    *,
    gates: AdmissionGates,
    times: ArrayLike,
    plant: ClosurePlant | None = None,
    repair: Literal["none", "project_to_closure"] = "none",
    rebuild: Rebuild | None = None,
) -> AdmissionReport:
    """Decide whether ``trajectory`` may be used for allocation.

    ``times`` are the frame time stamps; ``plant`` supplies the position-level
    residual and the repair projection; ``rebuild(q, v, a)`` re-creates a
    snapshot at a repaired state (needed only for repair).

    Preconditions (``ValueError``): at least 3 frames, one finite, strictly
    increasing time per frame, ``repair`` known, repair has ``plant`` and
    ``rebuild``, a contact schedule matches the frame count.

    Postconditions: the report names every failing frame by index; verdict
    ``ADMITTED`` only if every level, the time base, the derivative check and
    the stance feet pass; ``REPAIRED`` only when ``repair`` was requested,
    corrections were needed, and the repaired frames pass; otherwise
    ``REJECTED`` with reasons.  A missing plant rejects unless
    ``gates.require_position`` is ``False``.
    """
    t = _validate_inputs(trajectory, times, repair, plant, rebuild)
    snaps = tuple(trajectory)
    frames = _measure(snaps, gates, plant)
    time_sync = _check_time(t, gates)
    contact = _check_contacts(snaps, gates)
    noise, deriv_bad = (0.0, ())
    if gates.derivative_method == "central_difference":
        noise, deriv_bad = _derivative_mismatch(snaps, t, gates)

    failing = _failing(frames, gates)
    other: list[str] = []
    if plant is None and gates.require_position:
        other.append("position closure residual unavailable: no plant supplied")
    if not time_sync.ok:
        other.append(
            f"time base: dt jitter {time_sync.max_dt_jitter:.3e} exceeds "
            f"{gates.max_dt_jitter}"
        )
    other += [
        f"frame {k}: derivative mismatch exceeds {gates.derivative_tol:.1e} "
        f"(noise floor {noise:.3e})"
        for k in deriv_bad
    ]
    other += [
        f"frame {k}: stance contact speed {contact.stance_velocity[k]:.3e} "
        f"exceeds {gates.stance_velocity_tol:.1e}"
        for k in contact.failing_frames
    ]
    other += [
        f"frame {k}: schedule names unknown contact {n!r}" for k, n in contact.unknown
    ]

    verdict_frames = set(failing) | set(deriv_bad) | set(contact.failing_frames)
    verdict_frames |= {k for k, _ in contact.unknown}
    record = None
    if repair == "project_to_closure" and failing and not other:
        snaps, frames, failing, record = _try_repair(
            snaps, frames, failing, gates, plant, rebuild
        )
        verdict_frames = set(failing)

    reasons = tuple(m for msgs in failing.values() for m in msgs) + tuple(other)
    if reasons:
        verdict = Verdict(VerdictKind.REJECTED, reasons, tuple(sorted(verdict_frames)))
    elif record is not None:
        verdict = Verdict(VerdictKind.REPAIRED, repair=record)
    else:
        verdict = Verdict(VerdictKind.ADMITTED)
    return AdmissionReport(
        verdict=verdict,
        frames=frames,
        snapshots=snaps,
        times=tuple(float(x) for x in t),
        derivative_method=gates.derivative_method,
        noise_floor=noise,
        time_sync=time_sync,
        contact=contact,
        position_evaluated=plant is not None,
        gates=gates,
    )


def _try_repair(
    snaps: tuple[ClosedLoopSnapshot, ...],
    frames: tuple[FrameResiduals, ...],
    failing: dict[int, list[str]],
    gates: AdmissionGates,
    plant: ClosurePlant | None,
    rebuild: Rebuild | None,
) -> tuple[
    tuple[ClosedLoopSnapshot, ...],
    tuple[FrameResiduals, ...],
    dict,
    RepairRecord | None,
]:
    """Project the position/velocity-failing frames; leave the rest untouched."""
    assert plant is not None and rebuild is not None  # validated by admit()
    fixable = [
        k
        for k, f in enumerate(frames)
        if f.phi > gates.phi_tol or f.phi_dot > gates.phi_dot_tol
    ]
    out, corrections = list(snaps), []
    for k in fixable:
        out[k], corr = _repair_frame(k, snaps[k], plant, rebuild)
        corrections.append(corr)
    new_frames = _measure(out, gates, plant)
    still = _failing(new_frames, gates)
    for k in still:
        still[k] = [m + " (not repairable or repair insufficient)" for m in still[k]]
    return (
        tuple(out),
        new_frames,
        still,
        RepairRecord("project_to_closure", tuple(corrections)),
    )
