"""True closure alternatives with kinematics re-derived per grip model (NSX-72).

NSX-26 (``restriction``) keeps the welded kinematics and only removes reaction
rows.  A *grip model* is a different closure: it keeps its own rows, so it
admits a different set of motions and has its own pose, velocity and
acceleration residuals.  On the spatial fixture the four models are (the
closure rows are in the shaft-first grip frame, ``x`` along the club):

* ``WELD``                  both hands rigid on the club, ``fx fy fz tx ty tz``;
* ``SHAFT_TWIST_REVOLUTE``  the trail hand turns about the shaft, rows
  ``fx fy fz ty tz`` (the pose residual drops the twist component);
* ``TRAIL_BALL``            the trail hand is a ball joint on the club, rows
  ``fx fy fz`` (the pose residual is the position error of the grip point);
* ``LEAD_BALL``             the club hangs on a 3-DOF ball at the lead hand
  (three extra unactuated coordinates) and the trail hand welds to the club.

The first three share the 14 arm (+6 root) coordinates and differ only in the
rows kept; their Jacobians are row selections of the weld rows, which the tests
confirm independently against a finite difference of each model's own pose
residual.  ``LEAD_BALL`` is a different coordinate set.  Motions are sampled in
the null space of a model's own rows (``Motion.OWN``) or in the null space of
the rows of the welded-coordinate classes (``WELDED``, ``TWIST``, ``TILT``) so
the same trajectory can be put to every model.  All frames sit at the closed
posture of the fixture (rates differ); :func:`position_probe` takes small
finite steps off it to exercise the position level.

Nothing here is a measurement of a real grip or a statement about which grip a
golfer uses; a model that needs the user's hand-club contact data is stated as
a limitation, not filled in.  No engine other than Pinocchio is imported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

from src.shared.python.loop_nullspace.admission import (
    AdmissionGates,
    AdmissionReport,
    VerdictKind,
    admit,
)
from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    ClosureKind,
    InputProvenance,
    SpatialInertia,
)
from src.shared.python.loop_nullspace.spatial_fixture import (
    closed_spatial_plant,
    snapshot_at,
)

Array = NDArray[np.float64]

__all__ = [
    "GRIP_MODEL_SPECS",
    "GripModel",
    "GripModelSpec",
    "ModelAdmission",
    "Motion",
    "SpatialGripPlant",
    "admission_outcome",
    "admit_model",
    "motion_frames",
    "position_probe",
]

#: Order of the fixture's ``[lin; ang]`` rows with the club (grip z) first.
_SHAFT_FIRST = [2, 0, 1, 5, 3, 4]
_P = np.eye(3)[:, [2, 0, 1]]  # columns: new x = old z, new y = old x, new z = old y
_FRAME_DT_S = 0.01
_MIN_FRAMES = 3


class GripModel(Enum):
    """A grip model with its own closure kinematics."""

    WELD = "WELD"
    SHAFT_TWIST_REVOLUTE = "SHAFT_TWIST_REVOLUTE"
    TRAIL_BALL = "TRAIL_BALL"
    LEAD_BALL = "LEAD_BALL"


class Motion(Enum):
    """Motion classes sampled at the closed posture.

    ``WELDED`` satisfies all six weld rows; ``TWIST`` satisfies the five
    ``SHAFT_FREE5`` rows but turns the trail hand about the shaft; ``TILT``
    satisfies only the three force rows.  All three live in the welded
    coordinates.  ``OWN`` lies in the null space of the model's own rows.
    """

    WELDED = "WELDED"
    TWIST = "TWIST"
    TILT = "TILT"
    OWN = "OWN"


@dataclass(frozen=True)
class GripModelSpec:
    """Closure kind, kept rows, extra coordinates and what the model assumes."""

    model: GripModel
    closure_kind: ClosureKind
    rows: tuple[int, ...]
    components: tuple[str, ...]
    extra_coordinates: int
    summary: str
    limitation: str


_ALL6 = (0, 1, 2, 3, 4, 5)
_NAMES6 = ("fx", "fy", "fz", "tx", "ty", "tz")
GRIP_MODEL_SPECS: dict[GripModel, GripModelSpec] = {
    GripModel.WELD: GripModelSpec(
        GripModel.WELD,
        ClosureKind.WELD6,
        _ALL6,
        _NAMES6,
        0,
        "both hands rigid on the club: six closure rows",
        "assumes no hand-club slip or compliance; the grip is not measured",
    ),
    GripModel.SHAFT_TWIST_REVOLUTE: GripModelSpec(
        GripModel.SHAFT_TWIST_REVOLUTE,
        ClosureKind.SHAFT_FREE5,
        (0, 1, 2, 4, 5),
        ("fx", "fy", "fz", "ty", "tz"),
        0,
        "the trail hand turns freely about the shaft: five closure rows",
        "free twist with no friction or limit; whether a real grip turns about "
        "the shaft needs the golfer's hand-club torque data, which is not "
        "available",
    ),
    GripModel.TRAIL_BALL: GripModelSpec(
        GripModel.TRAIL_BALL,
        ClosureKind.BALL3,
        (0, 1, 2),
        ("fx", "fy", "fz"),
        0,
        "the trail hand is a ball joint on the club: three force rows",
        "a ball joint transmits no moment; a real palm and fingers do, and the "
        "size of the moment is not measured",
    ),
    GripModel.LEAD_BALL: GripModelSpec(
        GripModel.LEAD_BALL,
        ClosureKind.WELD6,
        _ALL6,
        _NAMES6,
        3,
        "the club hangs on a 3-DOF ball at the lead hand; the trail hand welds "
        "to the club: six rows, three extra unactuated coordinates",
        "adds a topology (three coordinates) the full-body spec builder does "
        "not support, so it exists only on the synthetic spatial fixture",
    ),
}


@dataclass(frozen=True)
class ModelAdmission:
    """Admission of one trajectory under one grip model's own residuals.

    ``max_phi``, ``max_phi_dot`` and ``max_phi_ddot`` are the scaled max-abs
    position, velocity and acceleration residuals of the model's own rows over
    all frames (``NaN`` for a level that was not evaluated).
    """

    model: GripModel
    admitted: bool
    n_frames: int
    max_phi: float
    max_phi_dot: float
    max_phi_ddot: float
    reasons: tuple[str, ...]
    failing_frames: tuple[int, ...]
    report: AdmissionReport


def _check_model(model: object) -> GripModel:
    if not isinstance(model, GripModel):
        raise TypeError(f"model must be a GripModel, got {type(model).__name__}")
    return model


def _check_motion(motion: object) -> Motion:
    if not isinstance(motion, Motion):
        raise TypeError(f"motion must be a Motion, got {type(motion).__name__}")
    return motion


class SpatialGripPlant:
    """One grip model on the spatial two-arm fixture (a ``ClosurePlant``).

    ``n`` coordinates ordered ``[root (6) |] lead arm (7) | trail arm (7) |
    [ball (3)]``.  Satisfies the ``ClosurePlant`` protocol of admission
    (``closure_pose_residual``, ``closure_rate_matrix``).  Raises ``TypeError``
    for a ``model`` that is not a :class:`GripModel`.
    """

    def __init__(self, model: GripModel, *, floating_root: bool = False) -> None:
        self.model = _check_model(model)
        self.spec = GRIP_MODEL_SPECS[self.model]
        self.floating_root = bool(floating_root)
        ball = self.spec.extra_coordinates > 0
        self._sp = closed_spatial_plant(floating_root=floating_root, lead_ball=ball)
        self._weld: SpatialGripPlant = (
            SpatialGripPlant(GripModel.WELD, floating_root=floating_root)
            if ball
            else self
        )
        self.q_star: Array = np.array(self._sp.q_star)
        self.n: int = int(self._sp.model.nv)

    # ------------------------------------------------------------ kinematics
    def integrate(self, q: Array, dq: Array) -> Array:
        """``q`` advanced by the tangent step ``dq``."""
        return np.asarray(pin.integrate(self._sp.model, q, dq))

    def closure_pose_residual(self, q: Array) -> Array:
        """The model's own pose residual (kept components, shaft-first)."""
        sp = self._sp
        pin.forwardKinematics(sp.model, sp.data, q)
        pin.updateFramePlacements(sp.model, sp.data)
        rel = sp.data.oMf[sp.lead_grip].inverse() * sp.data.oMf[sp.trail_grip]
        err = np.asarray(pin.log6(rel).vector)[_SHAFT_FIRST]
        return np.asarray(err[list(self.spec.rows)])

    def _rows6(self, q: Array) -> Array:
        return np.asarray(self._sp.closure_jacobian(q)[_SHAFT_FIRST])

    def _drift6(self, q: Array, v: Array) -> Array:
        return np.asarray(self._sp.closure_drift(q, v)[_SHAFT_FIRST])

    def closure_rate_matrix(self, q: Array) -> Array:
        """Kept closure rows ``J_c`` (m x n) in the shaft-first grip frame."""
        return self._rows6(q)[list(self.spec.rows)]

    def kinematic_nullity(self) -> int:
        """``n - rank(J_c)`` at the closed posture: the admitted-motion dimension."""
        sv = np.linalg.svd(self.closure_rate_matrix(self.q_star), compute_uv=False)
        return self.n - int(np.count_nonzero(sv > 1e-9 * max(float(sv[0]), 1.0)))

    # --------------------------------------------------------------- motions
    def _null_state(self, rows: Sequence[int], rng: np.random.Generator) -> tuple:
        jac = self._rows6(self.q_star)[list(rows)]
        pinv = np.linalg.pinv(jac, rcond=1e-12)
        v = rng.uniform(-1.0, 1.0, self.n)
        v = v - pinv @ (jac @ v)
        drift = self._drift6(self.q_star, v)[list(rows)]
        a = rng.uniform(-1.0, 1.0, self.n)
        return v, a - pinv @ (jac @ a + drift)

    def _embed(self, vec: Array) -> Array:
        return np.concatenate([vec, np.zeros(self.n - vec.size)])

    def motion_state(self, motion: Motion, rng: np.random.Generator) -> tuple:
        """Rates ``(v, a)`` of a motion class at the closed posture.

        ``OWN`` uses this plant's own rows.  The welded-coordinate classes are
        sampled on the weld plant and embedded with zero ball rates.
        """
        _check_motion(motion)
        if motion is Motion.OWN:
            return self._null_state(self.spec.rows, rng)
        rows = {
            Motion.WELDED: GRIP_MODEL_SPECS[GripModel.WELD].rows,
            Motion.TWIST: GRIP_MODEL_SPECS[GripModel.SHAFT_TWIST_REVOLUTE].rows,
            Motion.TILT: GRIP_MODEL_SPECS[GripModel.TRAIL_BALL].rows,
        }[motion]
        v, a = self._weld._null_state(rows, rng)
        return self._embed(v), self._embed(a)

    # -------------------------------------------------------------- snapshot
    def snapshot(self, q: Array, v: Array, a: Array) -> ClosedLoopSnapshot:
        """Snapshot at ``(q, v, a)`` with this model's rows, shaft along grip x.

        Postcondition: ``J_c`` and ``Jdot_v`` are the kept rows; the topology's
        closure kind is the model's.
        """
        note = f"NSX-72 grip model {self.model.value}: rows {list(self.spec.rows)}"
        base = snapshot_at(
            self._sp,
            q,
            v,
            a,
            input_provenance=InputProvenance("synthetic", "nsx-72", note),
        )
        rows = list(self.spec.rows)
        club = base.bare_club
        rotated = SpatialInertia(club.mass, _P.T @ club.com, _P.T @ club.inertia @ _P)
        grip = replace(base.grip_frame, rotation=base.grip_frame.rotation @ _P)
        return replace(
            base,
            J_c=base.J_c[_SHAFT_FIRST][rows],
            Jdot_v=base.Jdot_v[_SHAFT_FIRST][rows],
            grip_frame=grip,
            bare_club=rotated,
            topology=replace(base.topology, closure_kind=self.spec.closure_kind),
        )


def _times(n_frames: int) -> Array:
    return np.arange(n_frames) * _FRAME_DT_S


def motion_frames(
    plant: SpatialGripPlant,
    motion: Motion,
    *,
    n_frames: int,
    seed: int,
    source: SpatialGripPlant | None = None,
) -> tuple[list[ClosedLoopSnapshot], Array]:
    """Snapshots of ``n_frames`` rate samples of ``motion`` and their times.

    ``source`` is the plant whose coordinates the motion is sampled in (default
    ``plant``); a motion sampled in a different coordinate count is
    ``not representable`` in ``plant`` (``ValueError``).  Raises ``ValueError``
    for fewer than 3 frames (admission needs three) and ``TypeError`` for a
    ``motion`` that is not a :class:`Motion`.
    """
    _check_motion(motion)
    if n_frames < _MIN_FRAMES:
        raise ValueError(f"n_frames must be at least {_MIN_FRAMES}, got {n_frames}")
    origin = plant if source is None else source
    if origin.n != plant.n:
        raise ValueError(
            f"a {origin.model.value} motion ({origin.n} coordinates) is not "
            f"representable in the {plant.model.value} model ({plant.n})"
        )
    rng = np.random.default_rng(seed)
    snaps = []
    for _ in range(n_frames):
        v, a = origin.motion_state(motion, rng)
        snaps.append(plant.snapshot(plant.q_star, v, a))
    return snaps, _times(n_frames)


def position_probe(
    plant: SpatialGripPlant, motion: Motion, *, steps: Sequence[float]
) -> tuple[list[ClosedLoopSnapshot], Array]:
    """Rate-free frames at small finite steps off the closed posture.

    The step direction ``d`` is a ``motion`` direction scaled so the weld rows
    see a unit rate (``|J_weld d| = 1``); frame ``k`` sits at ``q* + h_k d``.
    A model whose rows ``d`` satisfies sees a residual that is second order in
    ``h``; a model whose rows ``d`` violates sees it at first order, which is
    the position-level difference between the closures.  Raises ``ValueError``
    for fewer than three steps or a non-positive step.
    """
    _check_motion(motion)
    hs = [float(h) for h in steps]
    if len(hs) < _MIN_FRAMES:
        raise ValueError(f"steps needs at least {_MIN_FRAMES} entries, got {len(hs)}")
    if any(not np.isfinite(h) or h <= 0.0 for h in hs):
        raise ValueError(f"steps must be positive and finite, got {hs}")
    direction, _ = plant.motion_state(motion, np.random.default_rng(0))
    weld_rows = plant._weld._rows6(plant._weld.q_star)
    rate = np.linalg.norm(weld_rows @ direction[: weld_rows.shape[1]])
    direction = direction / (rate if rate > 1e-9 else np.linalg.norm(direction))
    zero = np.zeros(plant.n)
    snaps = [
        plant.snapshot(plant.integrate(plant.q_star, h * direction), zero, zero)
        for h in hs
    ]
    return snaps, _times(len(hs))


def _level_max(report: AdmissionReport, level: str) -> float:
    values = [getattr(f, level) for f in report.frames]
    return float(np.nanmax(values)) if not np.isnan(values).all() else float("nan")


def admit_model(
    model: GripModel,
    snapshots: Sequence[ClosedLoopSnapshot],
    times: Array,
    gates: AdmissionGates,
    *,
    plant: SpatialGripPlant | None = None,
) -> ModelAdmission:
    """Admit ``snapshots`` under ``model``'s own pose, velocity and acceleration rows.

    ``plant`` supplies the model's own pose residual; without it the position
    level is not evaluated (admission then needs ``gates.require_position`` to
    be ``False``, which records the skip).  The gates are the caller's (the
    NSX-00 default is ``1e-6``); nothing is repaired.  The result carries the
    full :class:`AdmissionReport`.
    """
    _check_model(model)
    report = admit(list(snapshots), gates=gates, times=times, plant=plant)
    verdict = report.verdict
    return ModelAdmission(
        model=model,
        admitted=verdict.kind is VerdictKind.ADMITTED,
        n_frames=len(snapshots),
        max_phi=_level_max(report, "phi"),
        max_phi_dot=_level_max(report, "phi_dot"),
        max_phi_ddot=_level_max(report, "phi_ddot"),
        reasons=verdict.reasons,
        failing_frames=verdict.failing_frames,
        report=report,
    )


def admission_outcome(
    plant: SpatialGripPlant,
    snapshots: Sequence[ClosedLoopSnapshot],
    times: Array,
    gates: AdmissionGates,
) -> ModelAdmission:
    """:func:`admit_model` for a spatial plant (its own pose residual included)."""
    return admit_model(plant.model, snapshots, times, gates, plant=plant)
