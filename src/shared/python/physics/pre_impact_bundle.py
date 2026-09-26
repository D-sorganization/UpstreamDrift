"""Versioned immutable pre-impact bundle for the Tools impact kernels (#9703).

Placement: this is a *consumer-side* export record. It composes the pinned
Tools conventions instead of redefining them - ``golf_club.types.RigidTransform``
point convention, ``swing_sim.delivery_interchange`` grip frame and ``(w, x, y,
z)`` quaternions, ``golf_club.impact_mobility.RigidContactBody`` head mass and
COM inertia, and the ``golf_club.grip_impedance`` linear-then-angular twist
order. Tools does not yet define a pre-impact bundle, a modal-state record or a
per-field origin wire; those are candidates to upstream (see
``docs/development/impact_acoustics_program.md``, "Pre-Impact Bundle Version 1 -
Placement"). No Tools module is imported here; adapters take Tools records.

Postconditions: every constructed bundle holds finite SI values, proper
rotations, strictly increasing samples, an event inside a sampled interpolation
interval, a physically realizable COM inertia and modal arrays matching their
declared basis. Absent fields are explicit and raise
:class:`AbsentFieldError` on numeric use; they are never zero.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.physics._pre_impact_contracts import (
    AbsentFieldError,
    FieldOrigin,
    FloatArray,
    PreImpactBundleError,
    Quantity,
    com_inertia,
    fail,
    finite_array,
    finite_scalar,
    identifier,
    nonnegative,
    parse_quantity,
    positive,
    positive_definite,
    positive_semidefinite,
    reject_unknown,
    sha256_hex,
    strictly_increasing,
    symmetric,
    unit_vector,
)
from src.shared.python.physics._pre_impact_frames import (
    Pose,
    shift_twist_reference,
    shift_wrench_origin,
    twist_to_parent,
    wrench_to_parent,
)

PRE_IMPACT_BUNDLE_SCHEMA = "upstreamdrift.pre_impact_bundle"
PRE_IMPACT_BUNDLE_VERSION = 1
UNIT_SYSTEM = "SI"
WORLD_FRAME_ID = "world"
HEAD_FRAME_ID = "head"
GRIP_FRAME_ID = "grip"
#: Declared conventions; a payload carrying different ones is refused.
CONVENTIONS: Mapping[str, str] = {
    "pose": "p_parent = rotation @ p_child + translation_m",
    "quaternion": "wxyz, unit norm, never normalized",
    "head_frame": "origin at head COM; inertia, contact offset and normal in head",
    "grip_frame": "origin at butt, +z along shaft to head (Tools delivery wire)",
    "twist": "(linear, angular) of the head COM, expressed in world",
    "wrench": "(force, moment) in frame_id, moment about origin_m",
    "hand_impedance": "6x6 symmetric PSD, translations then small rotations",
    "modal_energy": "0.5 qd^T M_r qd + 0.5 q^T K_r q with basis matrices",
}
_INTERPOLANTS = frozenset({"linear", "cubic_hermite"})
_NORMALIZATIONS = frozenset({"mass", "none"})
_HANDS = frozenset({"lead", "trail"})
_MASS_NORMALIZATION_TOLERANCE = 1e-9


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        fail("type", f"{name} must be a nonempty list of strings")
    return tuple(identifier(item, name) for item in value)


@dataclass(frozen=True, eq=False)
class Provenance:
    """Equipment/ball/calibration identity, source digests and model tier."""

    equipment_id: str
    ball_id: str
    calibration_id: str
    model_tier: str
    source_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name in ("equipment_id", "ball_id", "calibration_id", "model_tier"):
            identifier(getattr(self, name), name)
        if not self.source_hashes:
            fail("hash", "at least one source hash is required")
        for key, digest in self.source_hashes:
            identifier(key, "source_hashes key")
            sha256_hex(digest, f"source_hashes[{key}]")


@dataclass(frozen=True, eq=False)
class TimeBase:
    """Sampled source times, event time and bounded interpolation interval.

    The interval must lie inside the sampled span and contain the event, so no
    state is extrapolated through contact.
    """

    sample_times_s: FloatArray
    event_time_s: float
    interval_s: tuple[float, float]
    interpolant: str
    time_uncertainty_s: Quantity
    interpolation_error_m: Quantity

    def __post_init__(self) -> None:
        times = self.sample_times_s
        if times.ndim != 1 or times.size < 2:
            fail("shape", "sample_times_s needs at least two samples")
        strictly_increasing(times, "sample_times_s")
        start, end = self.interval_s
        if not times[0] <= start < end <= times[-1]:
            fail("extrapolation", "interval_s must lie within the sampled span")
        if not start <= self.event_time_s <= end:
            fail("event_outside_interval", "event_time_s must lie in interval_s")
        if self.interpolant not in _INTERPOLANTS:
            fail("interpolant", f"interpolant must be one of {sorted(_INTERPOLANTS)}")
        nonnegative(self.time_uncertainty_s, "time_uncertainty_s")
        nonnegative(self.interpolation_error_m, "interpolation_error_m")


@dataclass(frozen=True, eq=False)
class HeadState:
    """Rigid head COM state; vectors other than velocities are in ``head``.

    ``face_curvature_1_m`` is an optional symmetric 2x2 face curvature tensor
    [1/m] in face-tangent coordinates; no pinned Tools provider supplies it, so
    it is usually ``absent``. Mass, inertia, velocities, contact offset and
    normal are required; direct construction with them absent raises.
    """

    mass_kg: Quantity
    inertia_com_kg_m2: Quantity
    linear_velocity_mps: Quantity
    angular_velocity_rad_s: Quantity
    contact_offset_m: Quantity
    contact_normal: Quantity
    face_curvature_1_m: Quantity

    def __post_init__(self) -> None:
        positive(self.mass_kg, "head.mass_kg")
        com_inertia(self.inertia_com_kg_m2.value, "head.inertia_com_kg_m2")
        unit_vector(self.contact_normal.value, "head.contact_normal")
        if not self.face_curvature_1_m.is_absent:
            symmetric(self.face_curvature_1_m.value, 2, "head.face_curvature_1_m")

    def tools_contact_body_fields(self) -> dict[str, Any]:
        """Keyword arguments for Tools ``golf_club.impact_mobility.RigidContactBody``."""
        return {
            "mass_kg": float(self.mass_kg),
            "inertia_at_com_kg_m2": tuple(
                tuple(float(item) for item in row)
                for row in self.inertia_com_kg_m2.value
            ),
            "contact_offset_m": tuple(self.contact_offset_m.value),
        }


@dataclass(frozen=True, eq=False)
class BallState:
    """Ball COM position/velocity and spin, all in ``world``."""

    position_m: Quantity
    velocity_mps: Quantity
    spin_rad_s: Quantity


@dataclass(frozen=True, eq=False)
class ModalBasis:
    """Identity and reduced quadratic forms of a shaft mode basis."""

    basis_id: str
    version: str
    normalization: str
    dimension: int
    generalized_mass: FloatArray
    generalized_stiffness: FloatArray

    def __post_init__(self) -> None:
        identifier(self.basis_id, "basis_id")
        identifier(self.version, "basis version")
        if self.normalization not in _NORMALIZATIONS:
            fail("normalization", f"normalization must be in {sorted(_NORMALIZATIONS)}")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            fail("type", "dimension must be an integer")
        if self.dimension < 1:
            fail("out_of_range", "dimension must be >= 1")
        size = self.dimension
        mass = positive_definite(self.generalized_mass, size, "generalized_mass")
        stiffness = positive_semidefinite(
            self.generalized_stiffness, size, "generalized_stiffness"
        )
        if self.normalization == "mass" and not np.allclose(
            mass, np.eye(size), rtol=0.0, atol=_MASS_NORMALIZATION_TOLERANCE
        ):
            fail("normalization", "mass-normalized basis needs identity modal mass")
        object.__setattr__(self, "generalized_mass", mass)
        object.__setattr__(self, "generalized_stiffness", stiffness)

    def energy_j(self, amplitudes: FloatArray, velocities: FloatArray) -> float:
        """Represented energy ``0.5 qd^T M qd + 0.5 q^T K q`` in joules."""
        kinetic, potential = self._energies(amplitudes, velocities)
        return kinetic + potential

    def _energies(
        self, amplitudes: FloatArray, velocities: FloatArray
    ) -> tuple[float, float]:
        q = finite_array(amplitudes, (self.dimension,), "amplitudes")
        qd = finite_array(velocities, (self.dimension,), "velocities")
        kinetic = 0.5 * float(qd @ self.generalized_mass @ qd)
        return kinetic, 0.5 * float(q @ self.generalized_stiffness @ q)

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis_id": self.basis_id,
            "version": self.version,
            "normalization": self.normalization,
            "dimension": self.dimension,
            "generalized_mass": self.generalized_mass.tolist(),
            "generalized_stiffness": self.generalized_stiffness.tolist(),
        }


@dataclass(frozen=True, eq=False)
class ShaftState:
    """Reduced modal state and prescribed axial-force field of the shaft."""

    basis: ModalBasis
    basis_id: str
    basis_version: str
    amplitudes: Quantity
    velocities: Quantity
    axial_stations_m: FloatArray
    axial_force_n: Quantity

    def __post_init__(self) -> None:
        if (self.basis_id, self.basis_version) != (
            self.basis.basis_id,
            self.basis.version,
        ):
            fail("basis_mismatch", "modal state basis id/version must match basis")
        for name in ("amplitudes", "velocities"):
            quantity = getattr(self, name)
            if not quantity.is_absent and quantity.value.shape != (
                self.basis.dimension,
            ):
                fail("basis_mismatch", f"{name} must match basis dimension")
        stations = self.axial_stations_m
        if stations.ndim != 1 or stations.size < 1 or float(stations[0]) < 0.0:
            fail("shape", "axial_stations_m must be nonnegative stations")
        strictly_increasing(stations, "axial_stations_m")
        if not self.axial_force_n.is_absent and (
            self.axial_force_n.value.shape != stations.shape
        ):
            fail("shape", "axial_force_n must have one value per station")

    def modal_energy_j(self) -> float:
        """Represented modal energy; absent elastic state raises, never zero."""
        return self.basis.energy_j(self.amplitudes.value, self.velocities.value)


@dataclass(frozen=True, eq=False)
class HandWrench:
    """One hand's applied wrench at ``origin_m`` in ``frame_id``, plus impedance."""

    hand: str
    frame_id: str
    origin_m: FloatArray
    force_n: Quantity
    moment_n_m: Quantity
    stiffness: Quantity
    damping: Quantity
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.hand not in _HANDS:
            fail("hand", f"hand must be one of {sorted(_HANDS)}")
        if self.frame_id not in (WORLD_FRAME_ID, HEAD_FRAME_ID, GRIP_FRAME_ID):
            fail("frame_mismatch", f"unknown wrench frame {self.frame_id!r}")
        for name in ("stiffness", "damping"):
            quantity = getattr(self, name)
            if not quantity.is_absent:
                positive_semidefinite(quantity.value, 6, f"{self.hand}.{name}")


@dataclass(frozen=True, eq=False)
class ModalProjection:
    """Mass-orthogonal projection result with residuals and energy ledger."""

    amplitudes: FloatArray
    velocities: FloatArray
    displacement_residual: float
    velocity_residual: float
    full_kinetic_energy_j: float
    full_potential_energy_j: float
    represented_kinetic_energy_j: float
    represented_potential_energy_j: float


def _relative_residual(
    residual: FloatArray, full: FloatArray, mass: FloatArray
) -> float:
    full_norm = float(np.sqrt(full @ mass @ full))
    residual_norm = float(np.sqrt(max(float(residual @ mass @ residual), 0.0)))
    return residual_norm / full_norm if full_norm > 0.0 else residual_norm


def project_onto_basis(
    basis: ModalBasis,
    shapes: object,
    full_mass: object,
    full_stiffness: object,
    displacement: object,
    velocity: object,
) -> ModalProjection:
    """Project a full-coordinate state onto ``basis`` without resetting energy.

    Preconditions: ``shapes`` (n, dim) reproduce the basis quadratic forms,
    ``Phi^T M Phi == M_r`` and ``Phi^T K Phi == K_r``; ``M`` is SPD, ``K`` PSD.
    Postconditions: ``q = (Phi^T M Phi)^-1 Phi^T M u`` (M-orthogonal), relative
    M-norm residuals, and full versus represented energies. An in-span state has
    zero residual and identical energy; out-of-span energy is reported, not
    silently discarded.
    """
    phi = np.asarray(shapes, dtype=float)
    if phi.ndim != 2 or phi.shape[1] != basis.dimension:
        fail("basis_mismatch", "shapes must have one column per basis mode")
    size = phi.shape[0]
    phi = finite_array(phi, (size, basis.dimension), "shapes")
    mass = positive_definite(full_mass, size, "full_mass")
    stiffness = positive_semidefinite(full_stiffness, size, "full_stiffness")
    for reduced, expected, name in (
        (phi.T @ mass @ phi, basis.generalized_mass, "mass"),
        (phi.T @ stiffness @ phi, basis.generalized_stiffness, "stiffness"),
    ):
        scale = max(float(np.max(np.abs(expected))), 1.0)
        if not np.allclose(reduced, expected, rtol=0.0, atol=1e-9 * scale):
            fail("basis_mismatch", f"shapes do not reproduce the basis {name}")
    u = finite_array(displacement, (size,), "displacement")
    ud = finite_array(velocity, (size,), "velocity")
    projector = np.linalg.solve(basis.generalized_mass, phi.T @ mass)
    q, qd = projector @ u, projector @ ud
    kinetic, potential = basis._energies(q, qd)
    for array in (q, qd):
        array.setflags(write=False)
    return ModalProjection(
        amplitudes=q,
        velocities=qd,
        displacement_residual=_relative_residual(u - phi @ q, u, mass),
        velocity_residual=_relative_residual(ud - phi @ qd, ud, mass),
        full_kinetic_energy_j=0.5 * float(ud @ mass @ ud),
        full_potential_energy_j=0.5 * float(u @ stiffness @ u),
        represented_kinetic_energy_j=kinetic,
        represented_potential_energy_j=potential,
    )


def grip_pose_from_delivery_sample(
    sample: Any, *, world_frame_id: str = WORLD_FRAME_ID
) -> Pose:
    """Adapt a Tools ``swing_sim.delivery_interchange.TrajectorySample``.

    The Tools wire already maps grip into world with a ``(w, x, y, z)``
    quaternion; this only re-validates it against the bundle's stricter
    unit-norm tolerance and never normalizes.
    """
    return Pose.from_quaternion(
        world_frame_id, GRIP_FRAME_ID, sample.quaternion_wxyz, sample.position_m
    )


@dataclass(frozen=True, eq=False)
class PreImpactBundle:
    """Versioned immutable pre-impact state; see module postconditions."""

    provenance: Provenance
    timebase: TimeBase
    head_pose: Pose
    grip_pose: Pose
    head: HeadState
    ball: BallState
    shaft: ShaftState
    hands: tuple[HandWrench, ...]
    constraints: tuple[str, ...]
    schema: str = PRE_IMPACT_BUNDLE_SCHEMA
    version: int = PRE_IMPACT_BUNDLE_VERSION
    units: str = UNIT_SYSTEM

    def __post_init__(self) -> None:
        if self.schema != PRE_IMPACT_BUNDLE_SCHEMA:
            fail("unsupported_schema", f"schema must be {PRE_IMPACT_BUNDLE_SCHEMA!r}")
        if self.version != PRE_IMPACT_BUNDLE_VERSION or isinstance(self.version, bool):
            fail("unsupported_version", f"version {self.version!r} is not supported")
        if self.units != UNIT_SYSTEM:
            fail("units", f"units must be {UNIT_SYSTEM!r}")
        for pose, frame in (
            (self.head_pose, HEAD_FRAME_ID),
            (self.grip_pose, GRIP_FRAME_ID),
        ):
            if (pose.parent_frame_id, pose.frame_id) != (WORLD_FRAME_ID, frame):
                fail("frame_mismatch", f"{frame} pose must be expressed in world")
        names = [hand.hand for hand in self.hands]
        if len(set(names)) != len(names):
            fail("hand", "each hand may appear at most once")

    # --- frames ---------------------------------------------------------------

    def _world_from(self, frame_id: str) -> Pose:
        poses = {HEAD_FRAME_ID: self.head_pose, GRIP_FRAME_ID: self.grip_pose}
        if frame_id == WORLD_FRAME_ID:
            return Pose.identity(WORLD_FRAME_ID)
        if frame_id not in poses:
            fail("frame_mismatch", f"unknown frame {frame_id!r}")
        return poses[frame_id]

    def pose_between(self, target_frame_id: str, source_frame_id: str) -> Pose:
        """Pose of ``source`` expressed in ``target`` (maps source -> target)."""
        return (
            self._world_from(target_frame_id)
            .inverse()
            .compose(self._world_from(source_frame_id))
        )

    def hand(self, name: str) -> HandWrench:
        for hand in self.hands:
            if hand.hand == name:
                return hand
        raise PreImpactBundleError("hand", f"no {name!r} hand wrench declared")

    def hand_wrench_in(self, name: str, frame_id: str) -> tuple[FloatArray, FloatArray]:
        """Hand force and moment about ``frame_id``'s origin, in ``frame_id``.

        Raises :class:`AbsentFieldError` when the hand wrench is absent.
        """
        hand = self.hand(name)
        force, moment = shift_wrench_origin(
            hand.force_n.value, hand.moment_n_m.value, hand.origin_m, np.zeros(3)
        )
        return wrench_to_parent(
            self.pose_between(frame_id, hand.frame_id), force, moment
        )

    def field_origins(self) -> dict[str, FieldOrigin]:
        """Origin of every quantity, keyed by dotted field path."""
        groups: list[tuple[str, object]] = [
            ("timebase", self.timebase),
            ("head", self.head),
            ("ball", self.ball),
            ("shaft", self.shaft),
        ]
        groups += [(f"hands.{hand.hand}", hand) for hand in self.hands]
        return {
            f"{prefix}.{name}": value.origin
            for prefix, record in groups
            for name, value in vars(record).items()
            if isinstance(value, Quantity)
        }

    # --- serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return _bundle_to_dict(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True
        )

    @classmethod
    def from_dict(cls, payload: object) -> PreImpactBundle:
        return _bundle_from_dict(payload)

    @classmethod
    def from_json(cls, text: str) -> PreImpactBundle:
        if not isinstance(text, str):
            fail("type", "text must be str")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise PreImpactBundleError("json", str(error)) from error
        return cls.from_dict(payload)


# --- wire mapping -------------------------------------------------------------

_TOP = frozenset(
    {"schema", "version", "units", "provenance", "timebase", "frames", "head"}
    | {"ball", "shaft", "hands", "constraints"}
)
_HEAD_SHAPES = {
    "mass_kg": (),
    "inertia_com_kg_m2": (3, 3),
    "linear_velocity_mps": (3,),
    "angular_velocity_rad_s": (3,),
    "contact_offset_m": (3,),
    "contact_normal": (3,),
    "face_curvature_1_m": (2, 2),
}
_HEAD_REQUIRED = frozenset(_HEAD_SHAPES) - {"face_curvature_1_m"}
_BALL_SHAPES = {"position_m": (3,), "velocity_mps": (3,), "spin_rad_s": (3,)}
_HAND_FIELDS = frozenset(
    {"hand", "frame_id", "origin_m", "force_n", "moment_n_m", "stiffness"}
    | {"damping", "assumptions"}
)


def _quantities(
    raw: Mapping[str, Any],
    shapes: Mapping[str, tuple[int, ...]],
    prefix: str,
    required: frozenset[str] = frozenset(),
) -> dict[str, Quantity]:
    return {
        name: parse_quantity(
            raw[name], f"{prefix}.{name}", shape, required=name in required
        )
        for name, shape in shapes.items()
    }


def _pose_from_dict(raw: object, name: str) -> Pose:
    keys = frozenset(raw) if isinstance(raw, Mapping) else frozenset()
    rotation_key = "quaternion_wxyz" if "quaternion_wxyz" in keys else "rotation"
    data = reject_unknown(
        raw,
        frozenset({"parent_frame_id", "frame_id", rotation_key, "translation_m"}),
        name,
    )
    if rotation_key == "quaternion_wxyz":
        return Pose.from_quaternion(
            data["parent_frame_id"],
            data["frame_id"],
            data[rotation_key],
            data["translation_m"],
        )
    return Pose(
        data["parent_frame_id"],
        data["frame_id"],
        finite_array(data["rotation"], (3, 3), f"{name}.rotation"),
        finite_array(data["translation_m"], (3,), f"{name}.translation_m"),
    )


def _provenance_from_dict(raw: object) -> Provenance:
    data = reject_unknown(
        raw,
        frozenset(
            {"equipment_id", "ball_id", "calibration_id", "model_tier", "source_hashes"}
        ),
        "provenance",
    )
    hashes = data["source_hashes"]
    if not isinstance(hashes, Mapping):
        fail("type", "source_hashes must be an object")
    return Provenance(
        data["equipment_id"],
        data["ball_id"],
        data["calibration_id"],
        data["model_tier"],
        tuple(sorted((str(key), value) for key, value in hashes.items())),
    )


def _timebase_from_dict(raw: object) -> TimeBase:
    fields = {"sample_times_s", "event_time_s", "interval_s", "interpolant"}
    data = reject_unknown(
        raw,
        frozenset(fields | {"time_uncertainty_s", "interpolation_error_m"}),
        "timebase",
    )
    times = data["sample_times_s"]
    count = len(times) if isinstance(times, (list, tuple, np.ndarray)) else -1
    interval = finite_array(data["interval_s"], (2,), "interval_s")
    return TimeBase(
        sample_times_s=finite_array(times, (count,), "sample_times_s"),
        event_time_s=finite_scalar(data["event_time_s"], "event_time_s"),
        interval_s=(float(interval[0]), float(interval[1])),
        interpolant=identifier(data["interpolant"], "interpolant"),
        time_uncertainty_s=parse_quantity(
            data["time_uncertainty_s"], "time_uncertainty_s", ()
        ),
        interpolation_error_m=parse_quantity(
            data["interpolation_error_m"], "interpolation_error_m", ()
        ),
    )


def _basis_from_dict(raw: object) -> ModalBasis:
    fields = {"basis_id", "version", "normalization", "dimension"}
    data = reject_unknown(
        raw, frozenset(fields | {"generalized_mass", "generalized_stiffness"}), "basis"
    )
    size = data["dimension"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        fail("type", "basis dimension must be a positive integer")
    return ModalBasis(
        basis_id=data["basis_id"],
        version=data["version"],
        normalization=data["normalization"],
        dimension=size,
        generalized_mass=finite_array(
            data["generalized_mass"], (size, size), "generalized_mass"
        ),
        generalized_stiffness=finite_array(
            data["generalized_stiffness"], (size, size), "generalized_stiffness"
        ),
    )


def _vector_length(raw: object) -> int:
    value = raw.get("value") if isinstance(raw, Mapping) else None
    return len(value) if isinstance(value, (list, tuple, np.ndarray)) else -1


def _shaft_from_dict(raw: object) -> ShaftState:
    fields = {"basis", "basis_id", "basis_version", "amplitudes", "velocities"}
    data = reject_unknown(
        raw, frozenset(fields | {"axial_stations_m", "axial_force_n"}), "shaft"
    )
    stations = data["axial_stations_m"]
    count = len(stations) if isinstance(stations, (list, tuple, np.ndarray)) else -1
    return ShaftState(
        basis=_basis_from_dict(data["basis"]),
        basis_id=identifier(data["basis_id"], "basis_id"),
        basis_version=identifier(data["basis_version"], "basis_version"),
        amplitudes=parse_quantity(
            data["amplitudes"],
            "shaft.amplitudes",
            (_vector_length(data["amplitudes"]),),
        ),
        velocities=parse_quantity(
            data["velocities"],
            "shaft.velocities",
            (_vector_length(data["velocities"]),),
        ),
        axial_stations_m=finite_array(stations, (count,), "axial_stations_m"),
        axial_force_n=parse_quantity(
            data["axial_force_n"],
            "shaft.axial_force_n",
            (_vector_length(data["axial_force_n"]),),
        ),
    )


def _hand_from_dict(raw: object) -> HandWrench:
    data = reject_unknown(raw, _HAND_FIELDS, "hand")
    prefix = f"hands.{data['hand']}"
    shapes = {
        "force_n": (3,),
        "moment_n_m": (3,),
        "stiffness": (6, 6),
        "damping": (6, 6),
    }
    return HandWrench(
        hand=identifier(data["hand"], "hand"),
        frame_id=identifier(data["frame_id"], "frame_id"),
        origin_m=finite_array(data["origin_m"], (3,), f"{prefix}.origin_m"),
        assumptions=_strings(data["assumptions"], f"{prefix}.assumptions"),
        **_quantities(data, shapes, prefix),
    )


def _bundle_from_dict(payload: object) -> PreImpactBundle:
    allowed = (
        _TOP | {"conventions"}
        if isinstance(payload, Mapping) and "conventions" in payload
        else _TOP
    )
    data = reject_unknown(payload, frozenset(allowed), "bundle")
    if data["schema"] != PRE_IMPACT_BUNDLE_SCHEMA:
        fail("unsupported_schema", f"schema {data['schema']!r} is not supported")
    if data["version"] != PRE_IMPACT_BUNDLE_VERSION or isinstance(
        data["version"], bool
    ):
        fail("unsupported_version", f"version {data['version']!r} is not supported")
    if "conventions" in data and dict(data["conventions"]) != dict(CONVENTIONS):
        fail("conventions", "declared conventions differ from v1")
    frames = reject_unknown(data["frames"], frozenset({"head", "grip"}), "frames")
    head = reject_unknown(data["head"], frozenset(_HEAD_SHAPES), "head")
    ball = reject_unknown(data["ball"], frozenset(_BALL_SHAPES), "ball")
    hands = data["hands"]
    if not isinstance(hands, (list, tuple)):
        fail("type", "hands must be a list")
    return PreImpactBundle(
        schema=data["schema"],
        version=data["version"],
        units=data["units"],
        provenance=_provenance_from_dict(data["provenance"]),
        timebase=_timebase_from_dict(data["timebase"]),
        head_pose=_pose_from_dict(frames["head"], "frames.head"),
        grip_pose=_pose_from_dict(frames["grip"], "frames.grip"),
        head=HeadState(**_quantities(head, _HEAD_SHAPES, "head", _HEAD_REQUIRED)),
        ball=BallState(**_quantities(ball, _BALL_SHAPES, "ball")),
        shaft=_shaft_from_dict(data["shaft"]),
        hands=tuple(_hand_from_dict(hand) for hand in hands),
        constraints=_strings(data["constraints"], "constraints"),
    )


def _record_quantities(record: object) -> dict[str, Any]:
    return {
        name: value.to_dict()
        for name, value in vars(record).items()
        if isinstance(value, Quantity)
    }


def _bundle_to_dict(bundle: PreImpactBundle) -> dict[str, Any]:
    provenance, timebase, shaft = bundle.provenance, bundle.timebase, bundle.shaft
    return {
        "schema": bundle.schema,
        "version": bundle.version,
        "units": bundle.units,
        "conventions": dict(CONVENTIONS),
        "provenance": {
            "equipment_id": provenance.equipment_id,
            "ball_id": provenance.ball_id,
            "calibration_id": provenance.calibration_id,
            "model_tier": provenance.model_tier,
            "source_hashes": dict(provenance.source_hashes),
        },
        "timebase": {
            "sample_times_s": timebase.sample_times_s.tolist(),
            "event_time_s": timebase.event_time_s,
            "interval_s": list(timebase.interval_s),
            "interpolant": timebase.interpolant,
            **_record_quantities(timebase),
        },
        "frames": {
            "head": bundle.head_pose.to_dict(),
            "grip": bundle.grip_pose.to_dict(),
        },
        "head": _record_quantities(bundle.head),
        "ball": _record_quantities(bundle.ball),
        "shaft": {
            "basis": shaft.basis.to_dict(),
            "basis_id": shaft.basis_id,
            "basis_version": shaft.basis_version,
            "axial_stations_m": shaft.axial_stations_m.tolist(),
            **_record_quantities(shaft),
        },
        "hands": [
            {
                "hand": hand.hand,
                "frame_id": hand.frame_id,
                "origin_m": hand.origin_m.tolist(),
                "assumptions": list(hand.assumptions),
                **_record_quantities(hand),
            }
            for hand in bundle.hands
        ],
        "constraints": list(bundle.constraints),
    }


__all__ = [
    "CONVENTIONS",
    "PRE_IMPACT_BUNDLE_SCHEMA",
    "PRE_IMPACT_BUNDLE_VERSION",
    "AbsentFieldError",
    "BallState",
    "FieldOrigin",
    "HandWrench",
    "HeadState",
    "ModalBasis",
    "ModalProjection",
    "Pose",
    "PreImpactBundle",
    "PreImpactBundleError",
    "Provenance",
    "Quantity",
    "ShaftState",
    "TimeBase",
    "grip_pose_from_delivery_sample",
    "project_onto_basis",
    "shift_twist_reference",
    "shift_wrench_origin",
    "twist_to_parent",
    "wrench_to_parent",
]
