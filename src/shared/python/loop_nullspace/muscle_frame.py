"""Muscle frame: the muscle-to-joint-torque map at one pose (NSX-94, #12723).

Models the pullback of ``docs/research/loop_nullspace/muscle_level_view.md``:
the joint torque produced by ``m`` muscles on ``n`` coordinates at a fixed
state is

    tau = R^T f(a),       f(a) = f_p + diag(F_a) a,       0 <= a <= 1,

where ``R = -dL/dq`` is the ``(m, n)`` moment-arm matrix (the repository
convention of ``biomechanics.muscle_allocation`` and
``myofullbody.redundancy``), ``f_p`` the tension at zero activation and ``F_a``
the tension per unit activation.  A :class:`MuscleFrame` stores the engine's
tension *table* ``force_grid`` on an activation grid.  The fibre algebra
(:mod:`muscle_fiber`) always uses the secant between the first and last grid
point, ``f_p = force_grid[0]`` and ``F_a = force_grid[-1] - force_grid[0]``;
:meth:`MuscleFrame.linearity_residual` measures how far the engine table is
from that linear law, and :func:`qualify_frame` refuses a frame whose
nonlinearity is as large as a reserve actuator would be.

Qualification (fail closed).  A frame is ``QUALIFIED`` for a target torque only
if both

* the *reserve fraction* ``||rho|| / ||tau||`` is below ``reserve_limit``, with
  ``rho`` the smallest torque the muscles cannot supply
  (``min ||tau - tau_p - G a||`` over ``0 <= a <= 1``); and
* the *linearity fraction* ``||R^T (f_table - f_secant)||_max / ||tau||`` is
  below ``linearity_limit``.

Both limits default to ``RESERVE_RMS_FRACTION_LIMIT = 0.10`` of
:mod:`myofullbody.redundancy` -- the repository's existing criterion that a
frame whose reserves carry more than 10 percent of the effort is
``NOT_QUALIFIED``.  The same number bounds the linearity residual because a
muscle model whose force law departs from the linear map by torque ``e`` acts
on the fibre analysis exactly as an unmodelled reserve of size ``e`` would; the
two failure modes are the same unmodelled torque and share a limit.

Non-claims: the frame is a model of one pose, not of a golfer; moment arms,
forces and stiffnesses are whatever the engine reports.  Nothing here infers a
co-contraction level from a net torque.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import lsq_linear

from src.shared.python.loop_nullspace.evidence import (
    Array,
    InputProvenance,
    ModelProvenance,
)
from src.shared.python.myofullbody.redundancy import RESERVE_RMS_FRACTION_LIMIT

__all__ = [
    "RESERVE_FRACTION_LIMIT",
    "FrameQualification",
    "MuscleFrame",
    "NotQualifiedError",
    "QualificationStatus",
    "TendonKind",
    "qualify_frame",
    "require_qualified",
]

RESERVE_FRACTION_LIMIT = RESERVE_RMS_FRACTION_LIMIT
"""Largest reserve (and linearity) fraction of a qualified frame; see module doc."""

_BALANCE_RTOL = 1e-9
"""Reserve fraction below which a target counts as reachable (ForceNullSpace's
``balance_tolerance`` default: a dimensionless per-row residual)."""


class TendonKind(str, Enum):
    """Whether the engine's muscle-tendon unit has a series-compliant tendon."""

    RIGID = "rigid"
    COMPLIANT = "compliant"


class QualificationStatus(str, Enum):
    """Outcome of :func:`qualify_frame`."""

    QUALIFIED = "QUALIFIED"
    NOT_QUALIFIED = "NOT_QUALIFIED"


class NotQualifiedError(ValueError):
    """A fail-closed refusal; carries the :class:`FrameQualification`."""

    def __init__(self, message: str, qualification: FrameQualification) -> None:
        super().__init__(message)
        self.qualification = qualification


def _readonly(value: ArrayLike, name: str, ndim: int) -> Array:
    arr = np.array(value, dtype=np.float64, copy=True)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True)
class MuscleFrame:
    """Muscle tensions, moment arms and optional stiffness data at one pose.

    Attributes:
        names: Muscle names, unique, length ``m``.
        coordinates: Coordinate names, unique, length ``n`` (velocity dofs).
        moment_arms: ``R = -dL/dq`` in metres, shape ``(m, n)``.
        activation_grid: Strictly increasing, starts at 0 and ends at 1.
        force_grid: Tension in N at each grid activation, ``(g, m)``, ``>= 0``.
        max_isometric_force: ``F_max`` in N, ``(m,)``, ``> 0``.
        tendon: :class:`TendonKind` of the engine's muscle-tendon unit.
        engine: Engine identifier (``opensim``, ``myosuite``, ``fixture``).
        fiber_stiffness_grid: Fibre stiffness along the tendon in N/m,
            ``(g, m)``, or ``None`` when the engine does not expose it.
        tendon_stiffness_grid: Tendon stiffness in N/m, ``(g, m)``, ``> 0``;
            required for a compliant tendon's series stiffness, ``None`` for a
            rigid one.
        moment_arm_jacobian: ``dR[k, i]/dq_j``, ``(m, n, n)``, or ``None``.

    Design by Contract: finite arrays of the stated shapes; unique names;
    ``activation_grid`` in ``[0, 1]`` from 0 to 1; tensions non-negative.
    Postcondition: every array is read-only.
    """

    names: tuple[str, ...]
    coordinates: tuple[str, ...]
    moment_arms: Array
    activation_grid: Array
    force_grid: Array
    max_isometric_force: Array
    tendon: TendonKind
    engine: str
    model_provenance: ModelProvenance
    input_provenance: InputProvenance
    fiber_stiffness_grid: Array | None = None
    tendon_stiffness_grid: Array | None = None
    moment_arm_jacobian: Array | None = None
    _gain: Array = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        m, n = len(self.names), len(self.coordinates)
        if m < 1 or len(set(self.names)) != m:
            raise ValueError("names must be unique and non-empty")
        if n < 1 or len(set(self.coordinates)) != n:
            raise ValueError("coordinates must be unique and non-empty")
        if not self.engine:
            raise ValueError("engine must be a non-empty string")
        arms = _readonly(self.moment_arms, "moment_arms", 2)
        grid = _readonly(self.activation_grid, "activation_grid", 1)
        forces = _readonly(self.force_grid, "force_grid", 2)
        f_max = _readonly(self.max_isometric_force, "max_isometric_force", 1)
        if arms.shape != (m, n):
            raise ValueError(
                f"moment_arms must have shape ({m}, {n}), got {arms.shape}"
            )
        ok_grid = (
            grid.size >= 2
            and grid[0] == 0.0
            and grid[-1] == 1.0
            and bool(np.all(np.diff(grid) > 0.0))
        )
        if not ok_grid:
            raise ValueError("activation_grid must increase strictly from 0 to 1")
        if forces.shape != (grid.size, m) or bool(np.any(forces < 0.0)):
            raise ValueError(
                f"force_grid must be non-negative with shape ({grid.size}, {m})"
            )
        if f_max.shape != (m,) or bool(np.any(f_max <= 0.0)):
            raise ValueError(f"max_isometric_force must be positive with shape ({m},)")
        stiff = {
            "fiber_stiffness_grid": self.fiber_stiffness_grid,
            "tendon_stiffness_grid": self.tendon_stiffness_grid,
        }
        for key, value in stiff.items():
            if value is not None:
                checked = _readonly(value, key, 2)
                if checked.shape != (grid.size, m):
                    raise ValueError(f"{key} must have shape ({grid.size}, {m})")
                object.__setattr__(self, key, checked)
        if self.moment_arm_jacobian is not None:
            jac = _readonly(self.moment_arm_jacobian, "moment_arm_jacobian", 3)
            if jac.shape != (m, n, n):
                raise ValueError(f"moment_arm_jacobian must have shape ({m}, {n}, {n})")
            object.__setattr__(self, "moment_arm_jacobian", jac)
        object.__setattr__(self, "moment_arms", arms)
        object.__setattr__(self, "activation_grid", grid)
        object.__setattr__(self, "force_grid", forces)
        object.__setattr__(self, "max_isometric_force", f_max)
        gain = arms.T * (forces[-1] - forces[0])[None, :]
        gain.flags.writeable = False
        object.__setattr__(self, "_gain", gain)

    # ------------------------------------------------------------- accessors
    @property
    def n_muscles(self) -> int:
        """Number of muscles ``m``."""
        return len(self.names)

    @property
    def n_coordinates(self) -> int:
        """Number of coordinates ``n``."""
        return len(self.coordinates)

    @property
    def passive_force(self) -> Array:
        """``f_p``: tension at zero activation, ``(m,)``."""
        return np.asarray(self.force_grid[0])

    @property
    def active_force(self) -> Array:
        """``F_a``: secant tension per unit activation, ``(m,)``."""
        return np.asarray(self.force_grid[-1] - self.force_grid[0])

    @property
    def gain_matrix(self) -> Array:
        """``G = R^T diag(F_a)``, ``(n, m)``: torque per unit activation."""
        return np.asarray(self._gain)

    @property
    def passive_torque(self) -> Array:
        """``R^T f_p``, ``(n,)``."""
        return np.asarray(self.moment_arms.T @ self.passive_force)

    def _activation(self, activation: ArrayLike) -> Array:
        a = np.asarray(activation, dtype=np.float64)
        if a.shape != (self.n_muscles,) or not np.all(np.isfinite(a)):
            raise ValueError(
                f"activation must be finite with shape ({self.n_muscles},)"
            )
        return a

    # --------------------------------------------------------------- forces
    def linear_force(self, activation: ArrayLike) -> Array:
        """Secant tension ``f_p + F_a a``, ``(m,)``."""
        return np.asarray(
            self.passive_force + self.active_force * self._activation(activation)
        )

    def torque(self, activation: ArrayLike) -> Array:
        """Joint torque ``R^T (f_p + F_a a)`` of the linear law, ``(n,)``."""
        return np.asarray(self.moment_arms.T @ self.linear_force(activation))

    def engine_force(self, activation: ArrayLike) -> Array:
        """The engine's tension table interpolated per muscle, ``(m,)``."""
        a = self._activation(activation)
        grid = self.activation_grid
        return np.array(
            [np.interp(a[k], grid, self.force_grid[:, k]) for k in range(a.size)]
        )

    def engine_torque(self, activation: ArrayLike) -> Array:
        """Joint torque of the interpolated engine table, ``(n,)``."""
        return np.asarray(self.moment_arms.T @ self.engine_force(activation))

    def linearity_residual(self) -> float:
        """Largest grid-point torque gap ``||R^T (f_table - f_secant)||_2`` in N m."""
        secant = self.passive_force[None, :] + np.outer(
            self.activation_grid, self.active_force
        )
        gap = (self.force_grid - secant) @ self.moment_arms
        return float(np.max(np.linalg.norm(gap, axis=1)))


@dataclass(frozen=True)
class FrameQualification:
    """Reserve and linearity gate outcome for one target torque."""

    status: QualificationStatus
    reserve_fraction: float
    reserve_limit: float
    linearity_fraction: float
    linearity_limit: float
    reachable_torque: Array
    reasons: tuple[str, ...]


def _norm_positive(torque: ArrayLike, n: int) -> tuple[Array, float]:
    tau = np.asarray(torque, dtype=np.float64)
    if tau.shape != (n,) or not np.all(np.isfinite(tau)):
        raise ValueError(f"torque must be finite with shape ({n},)")
    norm = float(np.linalg.norm(tau))
    if norm == 0.0:
        raise ValueError(
            "torque must be non-zero: a reserve fraction is undefined at 0"
        )
    return tau, norm


def closest_reachable(frame: MuscleFrame, torque: Array) -> tuple[Array, Array]:
    """Muscle-reachable torque nearest ``torque`` and the activations producing it.

    Bounded least squares (Stark-Parker BVLS) of ``min ||tau - tau_p - G a||``
    over ``0 <= a <= 1``.  The nearest point of a convex set (the zonotope of
    reachable torques) is unique, so the returned torque is unambiguous even
    though the activations are not.
    """
    scale = float(np.linalg.norm(torque))
    gain = frame.gain_matrix / scale
    rhs = (torque - frame.passive_torque) / scale
    sol = lsq_linear(gain, rhs, bounds=(0.0, 1.0), method="bvls", tol=1e-15)
    act = np.clip(sol.x, 0.0, 1.0)
    return frame.passive_torque + frame.gain_matrix @ act, act


def qualify_frame(
    frame: MuscleFrame,
    torque: ArrayLike,
    *,
    reserve_limit: float = RESERVE_FRACTION_LIMIT,
    linearity_limit: float = RESERVE_FRACTION_LIMIT,
) -> FrameQualification:
    """Reserve and linearity gate of ``frame`` for the target ``torque`` (N m).

    Preconditions (``ValueError``): ``torque`` finite, non-zero, length ``n``;
    both limits in ``(0, 1]``.  Postconditions: ``reachable_torque`` is the
    nearest muscle-reachable torque; ``status`` is ``QUALIFIED`` iff both
    fractions are at most their limits; every failure adds a reason.
    """
    for name, limit in (
        ("reserve_limit", reserve_limit),
        ("linearity_limit", linearity_limit),
    ):
        if not 0.0 < limit <= 1.0:
            raise ValueError(f"{name} must lie in (0, 1], got {limit}")
    tau, norm = _norm_positive(torque, frame.n_coordinates)
    reach, _ = closest_reachable(frame, tau)
    reserve = float(np.linalg.norm(tau - reach)) / norm
    lin = frame.linearity_residual() / norm
    reasons: list[str] = []
    if reserve > reserve_limit:
        reasons.append(
            f"reserve fraction {reserve:.4g} exceeds {reserve_limit:.4g}: reserves "
            "carry the load, so the muscle set does not close the torque"
        )
    if lin > linearity_limit:
        reasons.append(
            f"engine force table departs from the linear activation law by "
            f"{lin:.4g} of the torque (limit {linearity_limit:.4g})"
        )
    status = (
        QualificationStatus.NOT_QUALIFIED if reasons else QualificationStatus.QUALIFIED
    )
    out = np.array(reach)
    out.flags.writeable = False
    return FrameQualification(
        status=status,
        reserve_fraction=reserve,
        reserve_limit=float(reserve_limit),
        linearity_fraction=lin,
        linearity_limit=float(linearity_limit),
        reachable_torque=out,
        reasons=tuple(reasons),
    )


def require_qualified(qualification: FrameQualification) -> FrameQualification:
    """Return ``qualification`` or raise :class:`NotQualifiedError` with the reasons."""
    if qualification.status is not QualificationStatus.QUALIFIED:
        raise NotQualifiedError(
            "frame is NOT_QUALIFIED: " + "; ".join(qualification.reasons), qualification
        )
    return qualification
