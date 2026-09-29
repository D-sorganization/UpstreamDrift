"""The ``GranularSolver`` protocol and its value objects (issue #8611).

ADR-0032 decides that *every* fidelity tier implements one protocol and
that **every result carries its fidelity tier plus a validity verdict**.
This module is that contract.

Two design rules are load-bearing and are enforced by the types here:

* **Array-granular, never per-element-object.** A solver receives one
  :class:`~bunkershot3d.solvers.elements.SurfaceElements` structure of
  arrays and returns one :class:`Wrench`.  There is no ``Element`` class
  to iterate, so a 1000-point design of experiments stays a NumPy
  problem rather than a Python-loop problem.
* **A result is never a bare number.**  :class:`SolverResult` cannot be
  constructed without a :class:`~bunkershot3d.solvers.envelope.ValidityVerdict`,
  so a force can never be reported without the statement of how far
  outside the calibrated envelope it was produced.

SI units throughout, with unit-suffixed names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .elements import SurfaceElements
from .envelope import ValidityVerdict
from .exceptions import SolverInputError

__all__ = [
    "FidelityTier",
    "GranularSolver",
    "IntrusionState",
    "SolverResult",
    "Wrench",
]


class FidelityTier(StrEnum):
    """The ADR-0032 fidelity ladder.

    ==========  ==================================================
    Tier        Solver
    ==========  ==================================================
    ``F0``      Dynamic Resistive Force Theory, analytic (default)
    ``F1``      Reduced-order / 2-D plane-strain continuum
    ``F2``      Material Point Method, GPU reference truth
    ``F3``      Discrete Element Method, grain-scale studies only
    ==========  ==================================================
    """

    F0 = "F0"
    F1 = "F1"
    F2 = "F2"
    F3 = "F3"


def _as_vector(name: str, value: ArrayLike) -> NDArray[np.float64]:
    """Coerce ``value`` to an immutable finite ``(3,)`` float array.

    The finiteness check is a scalar sum rather than ``np.isfinite`` over
    the array: NaN and both infinities propagate through the sum, and
    this runs once per timestep per vector on the shot path, where a
    NumPy reduction over three floats is pure overhead.
    """
    array = np.array(value, dtype=np.float64, copy=True).reshape(-1)
    if array.shape != (3,):
        raise SolverInputError(
            f"{name} must be a 3-vector, got shape {np.shape(value)!r}"
        )
    if not math.isfinite(float(array[0]) + float(array[1]) + float(array[2])):
        raise SolverInputError(f"{name} contains non-finite values: {array!r}")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class Wrench:
    """A resultant force and torque about a stated reference point.

    The reference point is part of the value: a torque without the point
    it is taken about is not a physical quantity, and silently defaulting
    it is how sign errors survive review.

    Attributes:
        force_n: Resultant force in newtons, world frame.
        torque_n_m: Resultant torque in newton-metres about
            ``reference_point_m``, world frame.
        reference_point_m: Where the torque is taken.
    """

    force_n: NDArray[np.float64]
    torque_n_m: NDArray[np.float64]
    reference_point_m: NDArray[np.float64]

    def __init__(
        self,
        force_n: ArrayLike,
        torque_n_m: ArrayLike,
        reference_point_m: ArrayLike = (0.0, 0.0, 0.0),
    ) -> None:
        object.__setattr__(self, "force_n", _as_vector("force_n", force_n))
        object.__setattr__(self, "torque_n_m", _as_vector("torque_n_m", torque_n_m))
        object.__setattr__(
            self,
            "reference_point_m",
            _as_vector("reference_point_m", reference_point_m),
        )

    @property
    def force_magnitude_n(self) -> float:
        """Magnitude of the resultant force."""
        return float(np.linalg.norm(self.force_n))

    @property
    def torque_magnitude_n_m(self) -> float:
        """Magnitude of the resultant torque about the reference point."""
        return float(np.linalg.norm(self.torque_n_m))

    def about(self, point_m: ArrayLike) -> Wrench:
        """Return the same wrench referred to a different point.

        Args:
            point_m: New reference point in world coordinates.

        Returns:
            A wrench with the identical force and the torque shifted by
            ``(reference - point) x force``.
        """
        new_point = _as_vector("point_m", point_m)
        shifted = self.torque_n_m + np.cross(
            self.reference_point_m - new_point, self.force_n
        )
        return Wrench(self.force_n, shifted, new_point)

    def __add__(self, other: Wrench) -> Wrench:
        """Add two wrenches taken about the same reference point."""
        if not isinstance(other, Wrench):
            return NotImplemented
        if not np.array_equal(self.reference_point_m, other.reference_point_m):
            raise SolverInputError(
                "cannot add wrenches taken about different reference points: "
                f"{self.reference_point_m!r} vs {other.reference_point_m!r}; "
                "move one with Wrench.about() first"
            )
        return Wrench(
            self.force_n + other.force_n,
            self.torque_n_m + other.torque_n_m,
            self.reference_point_m,
        )

    @classmethod
    def zero(cls, reference_point_m: ArrayLike = (0.0, 0.0, 0.0)) -> Wrench:
        """The null wrench about ``reference_point_m``."""
        return cls((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), reference_point_m)


@dataclass(frozen=True, eq=False)
class IntrusionState:
    """One instantaneous query: a body, its motion, and the free surface.

    The body is supplied as a structure of arrays rather than a mesh so
    that a time-marching caller can transform the element arrays once per
    step instead of re-deriving them from vertices.

    Attributes:
        elements: Surface discretisation in world coordinates.
        velocity_m_s: Linear velocity of ``reference_point_m``.
        angular_velocity_rad_s: Body angular velocity, world frame.
        reference_point_m: Point the velocity and torque refer to.
        free_surface_height_m: World ``z`` of the undisturbed sand
            surface.  The bed is the half space below it.
    """

    elements: SurfaceElements
    velocity_m_s: NDArray[np.float64]
    angular_velocity_rad_s: NDArray[np.float64]
    reference_point_m: NDArray[np.float64]
    free_surface_height_m: float

    def __init__(
        self,
        elements: SurfaceElements,
        velocity_m_s: ArrayLike,
        *,
        angular_velocity_rad_s: ArrayLike = (0.0, 0.0, 0.0),
        reference_point_m: ArrayLike = (0.0, 0.0, 0.0),
        free_surface_height_m: float = 0.0,
    ) -> None:
        if not isinstance(elements, SurfaceElements):
            raise SolverInputError(
                f"elements must be a SurfaceElements, got {type(elements).__name__}"
            )
        height = float(free_surface_height_m)
        if not np.isfinite(height):
            raise SolverInputError(
                f"free_surface_height_m must be finite, got {free_surface_height_m!r}"
            )
        object.__setattr__(self, "elements", elements)
        object.__setattr__(
            self, "velocity_m_s", _as_vector("velocity_m_s", velocity_m_s)
        )
        object.__setattr__(
            self,
            "angular_velocity_rad_s",
            _as_vector("angular_velocity_rad_s", angular_velocity_rad_s),
        )
        object.__setattr__(
            self,
            "reference_point_m",
            _as_vector("reference_point_m", reference_point_m),
        )
        object.__setattr__(self, "free_surface_height_m", height)

    @property
    def speed_m_s(self) -> float:
        """Magnitude of the reference-point velocity."""
        return float(np.linalg.norm(self.velocity_m_s))

    @property
    def has_uniform_velocity(self) -> bool:
        """True when every element shares one velocity.

        The common case, and worth naming: it lets a solver skip
        per-element velocity work entirely on the hot path.
        """
        return not bool(self.angular_velocity_rad_s.any())

    def element_velocities_m_s(self) -> NDArray[np.float64]:
        """Per-element velocity ``v + omega x (c - reference)``, ``(m, 3)``.

        The rigid-rotation term is skipped entirely when the body is not
        rotating, which is the common case and the hot path.
        """
        centroids = self.elements.centroids_m
        if not self.angular_velocity_rad_s.any():
            return np.broadcast_to(self.velocity_m_s, centroids.shape)
        lever = centroids - self.reference_point_m
        return self.velocity_m_s + np.cross(self.angular_velocity_rad_s, lever)

    def element_depths_m(self) -> NDArray[np.float64]:
        """Signed element depth: negative below the free surface, ``(m,)``."""
        centroids = self.elements.centroids_m
        return centroids[:, 2] - self.free_surface_height_m


@dataclass(frozen=True)
class SolverResult:
    """What a :class:`GranularSolver` returns, verdict included.

    Attributes:
        wrench: Resultant force and torque from the medium on the body.
        fidelity_tier: Which rung of the ADR-0032 ladder produced it.
        verdict: The validity statement for this query.
        depth_force_n: The depth-linear (``alpha * |z_tilde|``) part of
            the resultant force, reported separately so the
            depth/inertia crossover is observable rather than asserted.
        inertial_force_n: The dynamic (``lambda * rho * v_n^2``) part.
        n_active_elements: Elements that were both leading-edge and
            below the effective free surface.
        active_area_m2: Their total area.
        max_depth_m: Deepest submerged element, positive below surface.
    """

    wrench: Wrench
    fidelity_tier: FidelityTier
    verdict: ValidityVerdict
    depth_force_n: NDArray[np.float64]
    inertial_force_n: NDArray[np.float64]
    n_active_elements: int
    active_area_m2: float
    max_depth_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, ValidityVerdict):
            raise SolverInputError(
                "a solver result must carry a ValidityVerdict; ADR-0032 forbids "
                "returning a force without the statement of how far outside the "
                f"calibrated envelope it was produced (got {type(self.verdict).__name__})"
            )

    @property
    def force_magnitude_n(self) -> float:
        """Magnitude of the resultant force."""
        return self.wrench.force_magnitude_n

    @property
    def depth_force_magnitude_n(self) -> float:
        """Magnitude of the depth-linear part."""
        return float(np.linalg.norm(self.depth_force_n))

    @property
    def inertial_force_magnitude_n(self) -> float:
        """Magnitude of the dynamic part."""
        return float(np.linalg.norm(self.inertial_force_n))

    @property
    def inertial_fraction(self) -> float:
        """Share of the force magnitude carried by the dynamic term.

        Returns ``0.0`` for a null result rather than raising, because a
        query with nothing submerged is a legitimate answer.
        """
        total = self.depth_force_magnitude_n + self.inertial_force_magnitude_n
        if total <= 0.0:
            return 0.0
        return self.inertial_force_magnitude_n / total

    def summary(self) -> str:
        """A one-paragraph statement fit for a run manifest."""
        return (
            f"tier={self.fidelity_tier.value} "
            f"|F|={self.force_magnitude_n:.4g} N "
            f"(depth {self.depth_force_magnitude_n:.4g} N, "
            f"inertial {self.inertial_force_magnitude_n:.4g} N) "
            f"over {self.n_active_elements} element(s), "
            f"{self.active_area_m2:.4g} m^2\n" + self.verdict.summary()
        )


@runtime_checkable
class GranularSolver(Protocol):
    """The one interface every fidelity tier implements (ADR-0032).

    Implementations must be **array-granular**: ``solve`` receives one
    structure of arrays and returns one resultant, and must not allocate
    a Python object per surface element.
    """

    @property
    def fidelity_tier(self) -> FidelityTier:
        """Which rung of the fidelity ladder this solver occupies."""
        ...

    def envelope(self, state: IntrusionState) -> ValidityVerdict:
        """Judge a query *without* computing forces.

        A caller sweeping a design space uses this to discard
        out-of-envelope points cheaply before paying for the solve.
        """
        ...

    def solve(self, state: IntrusionState) -> SolverResult:
        """Return the resultant wrench, its tier and its validity verdict.

        Raises:
            OutOfEnvelopeError: If the query is outside the solver's
                envelope and the solver's refusal policy is strict.
        """
        ...
