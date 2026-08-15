"""Strike trace: the result-artifact view every designer metric is computed from.

Issue #8614 (W7, epic #8607), baseline finding B24.

**Why this module exists.** ADR-0032 makes BunkerShot3D multi-fidelity -- F0 DRFT,
F1 continuum, F2 MPM, F3 DEM. A metric computed from solver internals would work
for exactly one tier and would have to be rewritten for the next. Every metric in
this package is therefore computed from the **result artifact**: the contiguous
clubhead and wrench arrays of result schema v2
(:class:`bunkershot3d.io.schema.BunkerShotResultReader`), plus a small explicit
description of the head and of the scene it was swung in. Any solver that can
write the artifact gets the metrics for free, and two tiers can be compared
because they were measured the same way.

Conventions
-----------
* **SI throughout**, and names carry the unit: ``time_s``, ``sand_force_N``,
  ``max_depth_m``, ``shaft_axis_moment_Nm``.
* **Structure of arrays.** ``head_position_m`` is ``(T, 3)``. There is no
  ``Sample`` object, so there is no ``trace.samples[i].position.x`` to reach
  through (research digest section 7).
* **Quaternions are scalar-first** ``(w, x, y, z)`` and rotate body -> world,
  matching the v2 schema.
* **World +z is up**, and depth is measured **positive downward** from the
  undisturbed sand surface, so "deeper" is a larger number.
* The wrench is the action of the sand **on the head**, in world coordinates.
  Schema v2 carries no statement of the point the moment is taken about, so
  :class:`WrenchReference` makes the caller say. The merged W5 backend work
  (#8612) reports it about the head centre of mass, which is the default here.

Limitation, stated rather than hidden: the scene is a single flat horizontal sand
surface. A bunker face is inclined, and an incline drops granular drag by ~50 %
(research digest addendum section 1). These metrics describe a strike on a
locally flat lie; they do not model a face shot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.core.contracts import (
    check_positive_definite,
    check_symmetric,
    require,
)

from .enums import WrenchReference

__all__ = [
    "STANDARD_GRAVITY_MPS2",
    "WORLD_UP",
    "HeadModel",
    "StrikeScene",
    "StrikeTrace",
    "WrenchReference",
    "angular_velocity_world_radps",
    "centre_of_mass_moment_Nm",
    "rotate_body_to_world",
    "rotate_world_to_body",
]

#: World "up" direction. Fixed by the result schema, not a free parameter.
WORLD_UP = np.array([0.0, 0.0, 1.0])

#: Standard gravity [m/s^2]. Used for impulse balances and for reporting head
#: deceleration in g, the unit a designer reads it in.
STANDARD_GRAVITY_MPS2 = 9.80665

#: Largest vertical component a travel axis may have and still count as
#: horizontal. Divot length is a ground-plane measurement, so a travel axis that
#: dips is a caller error rather than a modelling choice.
_HORIZONTAL_ATOL = 1e-9

#: Tolerance on ``|q| - 1`` before an orientation is refused as non-unit.
_QUAT_NORM_ATOL = 1e-6


def _finite_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Return ``value`` as a float array of ``shape``, proving it is usable.

    The finiteness check is an explicit ``raise`` and not an ``assert``: a NaN
    reaching a metric is a silent wrong answer, and ``python -O`` strips
    ``assert`` (research digest section 6).

    Args:
        name: Field name, for diagnostics.
        value: Array-like to validate.
        shape: Required shape.

    Returns:
        The validated array as float64.

    Raises:
        ValueError: If the shape is wrong or any element is not finite.
    """
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite; found NaN or inf")
    return array


def _unit_vector(name: str, value: Any) -> np.ndarray:
    """Return ``value`` as a unit 3-vector.

    Args:
        name: Field name, for diagnostics.
        value: Array-like of length 3.

    Returns:
        The normalised vector.

    Raises:
        ValueError: If the shape is wrong, an element is not finite, or the
            vector has (near) zero length so its direction is undefined.
    """
    vector = _finite_array(name, value, (3,))
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError(f"{name} must have a direction; got a zero-length vector")
    return vector / norm


def rotate_body_to_world(
    quaternions: np.ndarray, vector_body: np.ndarray
) -> np.ndarray:
    """Rotate body-frame vectors into world coordinates at every sample.

    Uses ``v' = v + 2 w (u x v) + 2 u x (u x v)`` for ``q = (w, u)``, which needs
    no rotation-matrix construction and stays vectorised over time.

    Args:
        quaternions: ``(T, 4)`` unit quaternions, scalar-first, body -> world.
        vector_body: ``(3,)`` body-frame vector, broadcast over time, or a
            ``(T, 3)`` per-sample body-frame vector.

    Returns:
        ``(T, 3)`` world-frame vectors.

    Raises:
        ValueError: If the shapes are wrong or a value is not finite.
    """
    quats = np.asarray(quaternions, dtype=float)
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(f"quaternions must have shape (T, 4), got {quats.shape}")
    vector = np.asarray(vector_body, dtype=float)
    shape = (3,) if vector.ndim == 1 else (quats.shape[0], 3)
    vector = _finite_array("vector_body", vector, shape)
    scalar = quats[:, 0:1]
    axis = quats[:, 1:4]
    cross_one = np.cross(axis, np.broadcast_to(vector, axis.shape))
    cross_two = np.cross(axis, cross_one)
    return vector + 2.0 * scalar * cross_one + 2.0 * cross_two


def rotate_world_to_body(
    quaternions: np.ndarray, vector_world: np.ndarray
) -> np.ndarray:
    """Rotate world-frame vectors into body coordinates at every sample.

    Args:
        quaternions: ``(T, 4)`` unit quaternions, scalar-first, body -> world.
        vector_world: ``(3,)`` or ``(T, 3)`` world-frame vectors.

    Returns:
        ``(T, 3)`` body-frame vectors.
    """
    quats = np.asarray(quaternions, dtype=float)
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(f"quaternions must have shape (T, 4), got {quats.shape}")
    conjugate = quats * np.array([1.0, -1.0, -1.0, -1.0])
    return rotate_body_to_world(conjugate, vector_world)


def _canonical_quaternion_signs(quaternions: np.ndarray) -> np.ndarray:
    """Return ``quaternions`` with sign flips removed along the time axis.

    ``q`` and ``-q`` are the same rotation, so a solver may emit either. A raw
    time derivative across a flip reports an enormous spurious angular velocity;
    forcing consecutive samples into the same hemisphere removes it.

    Args:
        quaternions: ``(T, 4)`` quaternions, scalar-first.

    Returns:
        ``(T, 4)`` quaternions describing the same rotations, sign-continuous.
    """
    quats = np.array(quaternions, dtype=float, copy=True)
    for index in range(1, quats.shape[0]):
        if float(np.dot(quats[index], quats[index - 1])) < 0.0:
            quats[index] = -quats[index]
    return quats


def angular_velocity_world_radps(
    time_s: np.ndarray, quaternions: np.ndarray
) -> np.ndarray:
    """Return the world-frame angular velocity of a quaternion history.

    For a body -> world quaternion, ``dq/dt = 0.5 * omega_world (x) q``, so
    ``omega_world = 2 * (dq/dt) (x) q*`` for unit ``q``.

    Args:
        time_s: ``(T,)`` strictly increasing sample times [s].
        quaternions: ``(T, 4)`` unit quaternions, scalar-first.

    Returns:
        ``(T, 3)`` angular velocity [rad/s].

    Raises:
        ValueError: If the arrays disagree in length or are too short.
    """
    times = np.asarray(time_s, dtype=float).reshape(-1)
    quats = _canonical_quaternion_signs(quaternions)
    if quats.shape != (times.size, 4):
        raise ValueError(
            f"quaternions must have shape {(times.size, 4)}, got {quats.shape}"
        )
    if times.size < 3:
        raise ValueError(
            "angular velocity needs at least 3 samples for a second-order "
            f"edge difference, got {times.size}"
        )
    rates = np.gradient(quats, times, axis=0, edge_order=2)
    # Hamilton product rates (x) conj(q), vector part only.
    scalar_rate, vector_rate = rates[:, 0:1], rates[:, 1:4]
    scalar_q, vector_q = quats[:, 0:1], -quats[:, 1:4]
    vector = (
        scalar_rate * vector_q
        + scalar_q * vector_rate
        + np.cross(vector_rate, vector_q)
    )
    return 2.0 * vector


@dataclass(frozen=True)
class HeadModel:
    """Static properties of the clubhead the trace was recorded for.

    Only what the metrics actually need: enough to place the centre of mass and
    the sole in world coordinates, and to convert a moment into a twist.

    Attributes:
        mass_kg: Head mass [kg]. Tour wedges are 0.290-0.310 kg.
        centre_of_mass_body_m: ``(3,)`` CG offset from the recorded clubhead
            origin, in the head body frame [m].
        sole_reference_body_m: ``(3,)`` the sole point whose depth defines the
            divot -- normally the lowest point of the sole at address, or the
            leading edge [m].
        shaft_axis_body: ``(3,)`` unit shaft axis in the body frame, pointing
            **from the head up toward the grip**.
        inertia_body_kg_m2: Optional ``(3, 3)`` inertia tensor about the CG in
            body axes. Without it, rotational energy and free-head face rotation
            are not reported rather than being guessed.
    """

    mass_kg: float
    centre_of_mass_body_m: np.ndarray
    sole_reference_body_m: np.ndarray
    shaft_axis_body: np.ndarray
    inertia_body_kg_m2: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate and normalise the head description.

        Raises:
            ValueError: If the mass is not positive, a vector is malformed, or
                the inertia tensor is not symmetric positive definite.
        """
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ValueError(f"mass_kg must be positive and finite, got {self.mass_kg}")
        object.__setattr__(
            self,
            "centre_of_mass_body_m",
            _finite_array("centre_of_mass_body_m", self.centre_of_mass_body_m, (3,)),
        )
        object.__setattr__(
            self,
            "sole_reference_body_m",
            _finite_array("sole_reference_body_m", self.sole_reference_body_m, (3,)),
        )
        object.__setattr__(
            self,
            "shaft_axis_body",
            _unit_vector("shaft_axis_body", self.shaft_axis_body),
        )
        if self.inertia_body_kg_m2 is None:
            return
        inertia = _finite_array("inertia_body_kg_m2", self.inertia_body_kg_m2, (3, 3))
        if not check_symmetric(inertia):
            raise ValueError("inertia_body_kg_m2 must be symmetric")
        if not check_positive_definite(inertia):
            raise ValueError("inertia_body_kg_m2 must be positive definite")
        object.__setattr__(self, "inertia_body_kg_m2", inertia)

    def shaft_axis_moment_of_inertia(
        self, axis_body: np.ndarray | None = None
    ) -> float:
        """Return the moment of inertia about the shaft axis through the CG.

        This is the number that turns a sand-induced shaft-axis moment into an
        actual face rotation, and the research digest names it as what matters
        for a wedge (gear effect is negligible at a ~2 mm CG depth).

        Args:
            axis_body: Optional body-frame axis; defaults to the shaft axis.

        Returns:
            ``a . I . a`` [kg.m^2].

        Raises:
            ValueError: If no inertia tensor was supplied.
        """
        if self.inertia_body_kg_m2 is None:
            raise ValueError(
                "inertia_body_kg_m2 was not supplied, so the moment of inertia "
                "about the shaft axis is unknown; supply it rather than assuming one"
            )
        axis = (
            self.shaft_axis_body
            if axis_body is None
            else _unit_vector("axis_body", axis_body)
        )
        return float(axis @ self.inertia_body_kg_m2 @ axis)


@dataclass(frozen=True)
class StrikeScene:
    """Where the strike happened: the sand surface, the ball, and the target line.

    Attributes:
        sand_surface_height_m: World ``z`` of the undisturbed sand surface [m].
        ball_position_m: ``(3,)`` ball centre at address [m]. Entry distance is
            measured **behind the ball**, which is how the delivery data is
            reported (Wivou et al. 2016: 25-150 mm).
        travel_axis: ``(3,)`` unit horizontal direction of head travel, toward
            the target.
    """

    sand_surface_height_m: float
    ball_position_m: np.ndarray
    travel_axis: np.ndarray

    def __post_init__(self) -> None:
        """Validate the scene.

        Raises:
            ValueError: If a value is not finite or the travel axis is not
                horizontal (divot length is a ground-plane measurement).
        """
        if not np.isfinite(self.sand_surface_height_m):
            raise ValueError(
                f"sand_surface_height_m must be finite, got {self.sand_surface_height_m}"
            )
        object.__setattr__(
            self,
            "ball_position_m",
            _finite_array("ball_position_m", self.ball_position_m, (3,)),
        )
        axis = _unit_vector("travel_axis", self.travel_axis)
        if abs(float(axis @ WORLD_UP)) > _HORIZONTAL_ATOL:
            raise ValueError(
                "travel_axis must be horizontal (its vertical component is the "
                f"attack angle, which belongs to the trace); got {axis.tolist()}"
            )
        object.__setattr__(self, "travel_axis", axis)

    def depth_m(self, points_m: np.ndarray) -> np.ndarray:
        """Return depth below the undisturbed surface, positive downward.

        Args:
            points_m: ``(..., 3)`` world points [m].

        Returns:
            ``(...,)`` depth [m]; negative above the sand.
        """
        points = np.asarray(points_m, dtype=float)
        return self.sand_surface_height_m - points[..., 2]

    def along_travel_m(self, points_m: np.ndarray) -> np.ndarray:
        """Return signed distance along the travel axis, measured from the ball.

        Negative is behind the ball (where the club enters), positive is past it.

        Args:
            points_m: ``(..., 3)`` world points [m].

        Returns:
            ``(...,)`` signed along-track distance [m].
        """
        points = np.asarray(points_m, dtype=float)
        return (points - self.ball_position_m) @ self.travel_axis

    def translated(self, offset_m: np.ndarray) -> StrikeScene:
        """Return the scene rigidly translated by ``offset_m``.

        Args:
            offset_m: ``(3,)`` world translation [m].

        Returns:
            The translated scene. The surface rises with the vertical component
            of the offset, which is what makes the metrics translation-invariant
            when the trace is moved with it.
        """
        offset = _finite_array("offset_m", offset_m, (3,))
        return StrikeScene(
            sand_surface_height_m=self.sand_surface_height_m + float(offset @ WORLD_UP),
            ball_position_m=self.ball_position_m + offset,
            travel_axis=self.travel_axis,
        )


@dataclass(frozen=True)
class StrikeTrace:
    """One strike, as recorded in a result artifact.

    Attributes:
        time_s: ``(T,)`` strictly increasing sample times [s].
        head_position_m: ``(T, 3)`` recorded clubhead origin [m].
        head_orientation_quat: ``(T, 4)`` unit quaternions, scalar-first,
            body -> world.
        sand_force_N: ``(T, 3)`` world force of the sand **on the head** [N].
        sand_moment_Nm: ``(T, 3)`` world moment of the sand on the head [N.m],
            about the point named by ``moment_reference``.
        moment_reference: Point the recorded moment is taken about.
    """

    time_s: np.ndarray
    head_position_m: np.ndarray
    head_orientation_quat: np.ndarray
    sand_force_N: np.ndarray
    sand_moment_Nm: np.ndarray
    moment_reference: WrenchReference = WrenchReference.CENTRE_OF_MASS

    def __post_init__(self) -> None:
        """Validate the trace.

        Raises:
            ValueError: If shapes disagree, a value is not finite, time is not
                strictly increasing, or an orientation is not a unit quaternion.
        """
        times = np.asarray(self.time_s, dtype=float).reshape(-1)
        count = times.size
        if count < 3:
            raise ValueError(
                "a strike trace needs at least 3 samples: velocities are "
                "differentiated with second-order edge differences so that a "
                f"uniformly accelerating head is exact at both ends, got {count}"
            )
        object.__setattr__(self, "time_s", _finite_array("time_s", times, (count,)))
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        for name, width in (
            ("head_position_m", 3),
            ("head_orientation_quat", 4),
            ("sand_force_N", 3),
            ("sand_moment_Nm", 3),
        ):
            object.__setattr__(
                self, name, _finite_array(name, getattr(self, name), (count, width))
            )
        norms = np.linalg.norm(self.head_orientation_quat, axis=1)
        if not np.allclose(norms, 1.0, atol=_QUAT_NORM_ATOL, rtol=0.0):
            raise ValueError(
                "head_orientation_quat must hold unit quaternions; norms span "
                f"[{norms.min():.6f}, {norms.max():.6f}]"
            )
        object.__setattr__(
            self, "moment_reference", WrenchReference(self.moment_reference)
        )

    @property
    def n_samples(self) -> int:
        """Number of samples in the trace."""
        return int(self.time_s.size)

    @property
    def duration_s(self) -> float:
        """Span of the trace [s]."""
        return float(self.time_s[-1] - self.time_s[0])

    def point_path_m(self, offset_body_m: np.ndarray) -> np.ndarray:
        """Return the world path of a body-fixed point.

        Args:
            offset_body_m: ``(3,)`` offset from the recorded clubhead origin, in
                body axes [m].

        Returns:
            ``(T, 3)`` world positions [m].
        """
        return self.head_position_m + rotate_body_to_world(
            self.head_orientation_quat, offset_body_m
        )

    def point_velocity_mps(self, offset_body_m: np.ndarray) -> np.ndarray:
        """Return the world velocity of a body-fixed point.

        Differentiated from the recorded path rather than reconstructed from
        ``v + omega x r``, so it stays consistent with the positions the divot
        metrics are measured on. ``edge_order=2`` keeps the first and last
        samples exact for a uniformly accelerating head, which is what the
        energy balance is taken across.

        Args:
            offset_body_m: ``(3,)`` offset from the clubhead origin [m].

        Returns:
            ``(T, 3)`` world velocity [m/s].
        """
        return np.gradient(
            self.point_path_m(offset_body_m), self.time_s, axis=0, edge_order=2
        )

    def angular_velocity_radps(self) -> np.ndarray:
        """Return the head's world-frame angular velocity, ``(T, 3)`` [rad/s]."""
        return angular_velocity_world_radps(self.time_s, self.head_orientation_quat)

    def body_axis_world(self, axis_body: np.ndarray) -> np.ndarray:
        """Return a body-fixed unit axis expressed in world axes at each sample.

        Args:
            axis_body: ``(3,)`` body-frame axis; normalised before use.

        Returns:
            ``(T, 3)`` unit world vectors.
        """
        return rotate_body_to_world(
            self.head_orientation_quat, _unit_vector("axis_body", axis_body)
        )

    def translated(self, offset_m: np.ndarray) -> StrikeTrace:
        """Return the trace rigidly translated by ``offset_m``.

        Orientation and wrench are unchanged: a translation does not rotate the
        head, and the recorded moment is taken about a body-fixed point that
        moves with it.

        Args:
            offset_m: ``(3,)`` world translation [m].

        Returns:
            The translated trace.
        """
        offset = _finite_array("offset_m", offset_m, (3,))
        return StrikeTrace(
            time_s=self.time_s,
            head_position_m=self.head_position_m + offset,
            head_orientation_quat=self.head_orientation_quat,
            sand_force_N=self.sand_force_N,
            sand_moment_Nm=self.sand_moment_Nm,
            moment_reference=self.moment_reference,
        )

    @classmethod
    def from_result_file(
        cls,
        filepath: Path | str,
        *,
        moment_reference: WrenchReference = WrenchReference.CENTRE_OF_MASS,
        time_atol_s: float = 1e-12,
    ) -> StrikeTrace:
        """Read a strike trace from a BunkerShot3D result artifact.

        Schema v1 and v2 are both accepted -- the reader migrates v1 on read --
        so a metric computed here is comparable across every fidelity tier that
        writes the artifact.

        Args:
            filepath: Result file to read.
            moment_reference: Point the file's recorded moment is taken about.
            time_atol_s: Tolerance when checking that the clubhead and wrench
                streams share a time base.

        Returns:
            The trace.

        Raises:
            ValueError: If the two streams have different sample times, so the
                force at a sample cannot be attributed to the pose at it.
        """
        from ..io.schema import BunkerShotResultReader

        with BunkerShotResultReader(filepath) as reader:
            times, positions, quaternions = reader.read_clubhead_states()
            wrench_times, forces, moments = reader.read_contact_wrenches()
        if wrench_times.shape != times.shape or not np.allclose(
            wrench_times, times, atol=time_atol_s, rtol=0.0
        ):
            raise ValueError(
                "the clubhead and wrench streams do not share a time base, so a "
                "force cannot be attributed to a pose; resample them first"
            )
        return cls(
            time_s=times,
            head_position_m=positions,
            head_orientation_quat=quaternions,
            sand_force_N=forces,
            sand_moment_Nm=moments,
            moment_reference=moment_reference,
        )


def centre_of_mass_moment_Nm(trace: StrikeTrace, head: HeadModel) -> np.ndarray:
    """Return the sand moment about the head centre of mass, ``(T, 3)`` [N.m].

    When the artifact already reports the moment about the CG this is the
    recorded moment. When it reports it about the recorded clubhead origin the
    ``(r_origin - r_cg) x F`` transport term is added -- the term whose absence
    was baseline finding B5b, and the entire reason a sole twists.

    Args:
        trace: Strike trace.
        head: Head the trace was recorded for.

    Returns:
        ``(T, 3)`` moment about the centre of mass [N.m].
    """
    if trace.moment_reference is WrenchReference.CENTRE_OF_MASS:
        return trace.sand_moment_Nm
    require(
        trace.moment_reference is WrenchReference.HEAD_ORIGIN,
        "unhandled wrench reference",
        value=trace.moment_reference,
    )
    lever = trace.head_position_m - trace.point_path_m(head.centre_of_mass_body_m)
    return trace.sand_moment_Nm + np.cross(lever, trace.sand_force_N)
