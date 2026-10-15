"""Snapshot, topology, and six-axis evidence contracts for NSX (#12555, NSX-10).

Provides frozen value objects with read-only NumPy arrays and Design-by-Contract
(DbC) validation enforcing Rules R1 and R8 of Epic #12593.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
LimbSide: TypeAlias = Literal["left", "right"]


class ClosureKind(str, Enum):
    """Supported kinematic loop-closure constraint models."""

    WELD6 = "WELD6"
    SHAFT_FREE5 = "SHAFT_FREE5"
    BALL3 = "BALL3"
    PLANAR_POINT2 = "PLANAR_POINT2"
    PLANAR_WELD3 = "PLANAR_WELD3"

    def expected_rows(self) -> int:
        """Return the canonical scalar row count for this closure kind."""
        return CLOSURE_KIND_ROWS[self]


CLOSURE_KIND_ROWS: dict[ClosureKind, int] = {
    ClosureKind.WELD6: 6,
    ClosureKind.SHAFT_FREE5: 5,
    ClosureKind.BALL3: 3,
    ClosureKind.PLANAR_POINT2: 2,
    ClosureKind.PLANAR_WELD3: 3,
}


class AllocationDependence(str, Enum):
    """Evidence axis 1: algebraic dependence on null-space coordinates."""

    INVARIANT = "INVARIANT"
    FAMILY_DEPENDENT = "FAMILY_DEPENDENT"
    UNAVAILABLE = "UNAVAILABLE"


class ValidationStatus(str, Enum):
    """Evidence axis 6: empirical validation status against held-out sensors."""

    UNVALIDATED = "UNVALIDATED"
    VALIDATED_INSTRUMENTED_GRIP = "VALIDATED_INSTRUMENTED_GRIP"
    VALIDATED_EMG_TIMING = "VALIDATED_EMG_TIMING"
    FALSIFIED = "FALSIFIED"
    DEFERRED_EXTERNAL = "DEFERRED_EXTERNAL"


def _readonly_float_array(
    value: ArrayLike,
    name: str,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
    require_finite: bool = True,
) -> FloatArray:
    """Convert input to a fresh read-only float64 ndarray and validate shape/finiteness."""
    arr = np.array(value, dtype=np.float64, copy=True)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {arr.shape}")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if require_finite and not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    arr.setflags(write=False)
    return arr


def _validate_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_str_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings, not a single string")
    result = tuple(_validate_nonempty_str(item, f"{name} entry") for item in values)
    return result


def _validate_index_tuple(
    indices: Sequence[int], n_dof: int, name: str
) -> tuple[int, ...]:
    result = tuple(int(idx) for idx in indices)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate indices: {result}")
    for idx in result:
        if idx < 0 or idx >= n_dof:
            raise ValueError(f"{name} index {idx} out of bounds for n_dof={n_dof}")
    return result


@dataclass(frozen=True)
class LoopTopology:
    """Topological metadata and index partitions for a cut kinematic loop."""

    closure_kind: ClosureKind
    closing_side: LimbSide
    holding_side: LimbSide
    actuated_indices: tuple[int, ...]
    unactuated_indices: tuple[int, ...]
    loop_support_indices: tuple[int, ...]
    coordinate_order: tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            kind = ClosureKind(self.closure_kind)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported closure_kind: {self.closure_kind!r}"
            ) from exc
        object.__setattr__(self, "closure_kind", kind)

        if self.closing_side not in ("left", "right"):
            raise ValueError(
                f"closing_side must be 'left' or 'right', got {self.closing_side!r}"
            )
        if self.holding_side not in ("left", "right"):
            raise ValueError(
                f"holding_side must be 'left' or 'right', got {self.holding_side!r}"
            )
        if self.closing_side == self.holding_side:
            raise ValueError(
                "closing_side and holding_side must be distinct ('left' vs 'right')"
            )

        coords = _validate_str_tuple(self.coordinate_order, "coordinate_order")
        if len(coords) == 0:
            raise ValueError("coordinate_order must be non-empty")
        if len(set(coords)) != len(coords):
            raise ValueError("coordinate_order must not contain duplicate names")
        object.__setattr__(self, "coordinate_order", coords)

        n_dof = len(coords)
        actuated = _validate_index_tuple(
            self.actuated_indices, n_dof, "actuated_indices"
        )
        unactuated = _validate_index_tuple(
            self.unactuated_indices, n_dof, "unactuated_indices"
        )
        support = _validate_index_tuple(
            self.loop_support_indices, n_dof, "loop_support_indices"
        )
        if set(actuated) & set(unactuated):
            raise ValueError("actuated_indices and unactuated_indices must be disjoint")
        if set(actuated) | set(unactuated) != set(range(n_dof)):
            raise ValueError(
                "actuated_indices and unactuated_indices must partition range(n_v)"
            )
        object.__setattr__(self, "actuated_indices", actuated)
        object.__setattr__(self, "unactuated_indices", unactuated)
        object.__setattr__(self, "loop_support_indices", support)

    def expected_closure_rows(self) -> int:
        """Return the expected closure Jacobian row count for this topology."""
        return CLOSURE_KIND_ROWS[self.closure_kind]


@dataclass(frozen=True)
class GripFrame:
    """Canonical grip midpoint frame R_G in SO(3) and world-frame origin r_M."""

    rotation: FloatArray
    origin: FloatArray

    def __post_init__(self) -> None:
        rot = _readonly_float_array(self.rotation, "rotation", shape=(3, 3))
        orig = _readonly_float_array(self.origin, "origin", shape=(3,))
        ortho_err = float(np.linalg.norm(rot.T @ rot - np.eye(3), ord="fro"))
        if ortho_err > 1e-6:
            raise ValueError(
                f"GripFrame rotation must be orthonormal (||R^T R - I||_F={ortho_err:.3e} > 1e-6)"
            )
        det_val = float(np.linalg.det(rot))
        if abs(det_val - 1.0) > 1e-6:
            raise ValueError(
                f"GripFrame rotation must have det == +1 in SO(3), got det={det_val:.6f}"
            )
        object.__setattr__(self, "rotation", rot)
        object.__setattr__(self, "origin", orig)


@dataclass(frozen=True)
class SpatialInertia:
    """Rigid-body spatial inertia parameters about body center of mass."""

    mass_kg: float
    com_m: FloatArray
    inertia_com_kg_m2: FloatArray
    solid_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        mass = float(self.mass_kg)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"mass_kg must be finite and > 0, got {self.mass_kg}")
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(
            self, "com_m", _readonly_float_array(self.com_m, "com_m", shape=(3,))
        )
        inertia = _readonly_float_array(
            self.inertia_com_kg_m2, "inertia_com_kg_m2", shape=(3, 3)
        )
        if not np.allclose(inertia, inertia.T, atol=1e-10):
            raise ValueError("inertia_com_kg_m2 must be symmetric")
        sym_inertia = 0.5 * (inertia + inertia.T)
        min_eig = float(np.linalg.eigvalsh(sym_inertia).min())
        if min_eig < -1e-10:
            raise ValueError(
                f"inertia_com_kg_m2 must be positive semidefinite (min eig={min_eig:.3e})"
            )
        object.__setattr__(self, "inertia_com_kg_m2", inertia)
        object.__setattr__(
            self, "solid_names", _validate_str_tuple(self.solid_names, "solid_names")
        )

    @property
    def com_local_m(self) -> FloatArray:
        """Alias for center of mass vector."""
        return self.com_m

    @property
    def inertia_com_kgm2(self) -> FloatArray:
        """Alias for rotational inertia tensor about center of mass."""
        return self.inertia_com_kg_m2


SpatialInertiaPartition: TypeAlias = SpatialInertia


@dataclass(frozen=True)
class FootContact:
    """Single foot contact patch/point constraint and friction metadata."""

    foot_id: str
    point_world_m: FloatArray
    normal_world: FloatArray
    jacobian: FloatArray
    friction_coeff: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "foot_id", _validate_nonempty_str(self.foot_id, "foot_id")
        )
        object.__setattr__(
            self,
            "point_world_m",
            _readonly_float_array(self.point_world_m, "point_world_m", shape=(3,)),
        )
        normal = _readonly_float_array(self.normal_world, "normal_world", shape=(3,))
        norm_len = float(np.linalg.norm(normal))
        if abs(norm_len - 1.0) > 1e-6:
            raise ValueError(
                f"normal_world must be a unit vector, got norm={norm_len:.6f}"
            )
        object.__setattr__(self, "normal_world", normal)
        jac = _readonly_float_array(self.jacobian, "jacobian", ndim=2)
        if jac.shape[0] == 0 or jac.shape[1] == 0:
            raise ValueError(f"jacobian must be non-empty, got shape {jac.shape}")
        object.__setattr__(self, "jacobian", jac)
        mu = float(self.friction_coeff)
        if not np.isfinite(mu) or mu < 0.0:
            raise ValueError(
                f"friction_coeff must be finite and >= 0, got {self.friction_coeff}"
            )
        object.__setattr__(self, "friction_coeff", mu)


@dataclass(frozen=True)
class ModelProvenance:
    """Evidence axis 2: biomechanical model and engine provenance."""

    spec_sha256: str
    inertial_source: str
    bare_club_bench_id: str | None = None
    engine_name: str = "analytical"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "spec_sha256", _validate_nonempty_str(self.spec_sha256, "spec_sha256")
        )
        object.__setattr__(
            self,
            "inertial_source",
            _validate_nonempty_str(self.inertial_source, "inertial_source"),
        )
        if self.bare_club_bench_id is not None:
            object.__setattr__(
                self,
                "bare_club_bench_id",
                _validate_nonempty_str(self.bare_club_bench_id, "bare_club_bench_id"),
            )
        object.__setattr__(
            self, "engine_name", _validate_nonempty_str(self.engine_name, "engine_name")
        )


@dataclass(frozen=True)
class InputProvenance:
    """Evidence axis 3: kinematic capture and differentiation provenance."""

    kinematic_source: str
    derivative_method: str
    time_sync_status: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kinematic_source",
            _validate_nonempty_str(self.kinematic_source, "kinematic_source"),
        )
        object.__setattr__(
            self,
            "derivative_method",
            _validate_nonempty_str(self.derivative_method, "derivative_method"),
        )
        object.__setattr__(
            self,
            "time_sync_status",
            _validate_nonempty_str(self.time_sync_status, "time_sync_status"),
        )


def _validate_spd_mass_matrix(M: FloatArray, n_v: int) -> FloatArray:
    """Validate that M has shape (n_v, n_v), is symmetric, and is positive definite."""
    arr = _readonly_float_array(M, "M", shape=(n_v, n_v))
    if not np.allclose(arr, arr.T, atol=1e-10):
        raise ValueError("M must be symmetric (||M - M^T|| exceeds 1e-10)")
    sym_m = 0.5 * (arr + arr.T)
    try:
        np.linalg.cholesky(sym_m)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "M must be symmetric positive definite (Cholesky factorization failed)"
        ) from exc
    return arr


@dataclass(frozen=True)
class ClosedLoopSnapshot:
    """Single-frame canonical closed-loop dynamical snapshot (R1, R8)."""

    q: FloatArray
    v: FloatArray
    a: FloatArray
    M: FloatArray
    h: FloatArray
    S: FloatArray
    J_c: FloatArray
    Jdot_v: FloatArray
    grip_frame: GripFrame
    bare_club: SpatialInertia
    model_provenance: ModelProvenance
    input_provenance: InputProvenance
    topology: LoopTopology
    welded_hand_inertias: tuple[SpatialInertia, ...] = ()
    J_g: FloatArray | None = None
    foot_contacts: tuple[FootContact, ...] = ()
    grf_observed: FloatArray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.topology, LoopTopology):
            raise TypeError("topology must be a LoopTopology instance")
        if not isinstance(self.grip_frame, GripFrame):
            raise TypeError("grip_frame must be a GripFrame instance")
        if not isinstance(self.bare_club, SpatialInertia):
            raise TypeError("bare_club must be a SpatialInertia instance")
        if not isinstance(self.model_provenance, ModelProvenance):
            raise TypeError("model_provenance must be a ModelProvenance instance")
        if not isinstance(self.input_provenance, InputProvenance):
            raise TypeError("input_provenance must be an InputProvenance instance")

        n_v = len(self.topology.coordinate_order)
        n_tau = len(self.topology.actuated_indices)
        m_c = self.topology.expected_closure_rows()

        q_arr = _readonly_float_array(self.q, "q", ndim=1)
        if q_arr.shape[0] < n_v:
            raise ValueError(f"q length {q_arr.shape[0]} must be >= n_v ({n_v})")
        object.__setattr__(self, "q", q_arr)
        object.__setattr__(self, "v", _readonly_float_array(self.v, "v", shape=(n_v,)))
        object.__setattr__(self, "a", _readonly_float_array(self.a, "a", shape=(n_v,)))
        object.__setattr__(self, "M", _validate_spd_mass_matrix(self.M, n_v))
        object.__setattr__(self, "h", _readonly_float_array(self.h, "h", shape=(n_v,)))
        object.__setattr__(
            self, "S", _readonly_float_array(self.S, "S", shape=(n_tau, n_v))
        )
        object.__setattr__(
            self, "J_c", _readonly_float_array(self.J_c, "J_c", shape=(m_c, n_v))
        )
        object.__setattr__(
            self,
            "Jdot_v",
            _readonly_float_array(self.Jdot_v, "Jdot_v", shape=(m_c,)),
        )

        hands = tuple(self.welded_hand_inertias)
        for hand in hands:
            if not isinstance(hand, SpatialInertia):
                raise TypeError(
                    "welded_hand_inertias entries must be SpatialInertia instances"
                )
        object.__setattr__(self, "welded_hand_inertias", hands)

        contacts = tuple(self.foot_contacts)
        for contact in contacts:
            if not isinstance(contact, FootContact):
                raise TypeError("foot_contacts entries must be FootContact instances")
        object.__setattr__(self, "foot_contacts", contacts)

        if self.J_g is not None:
            jg = _readonly_float_array(self.J_g, "J_g", ndim=2)
            if jg.shape[0] == 0 or jg.shape[1] != n_v:
                raise ValueError(
                    f"J_g must have shape (m_g > 0, {n_v}), got {jg.shape}"
                )
            object.__setattr__(self, "J_g", jg)

        if self.grf_observed is not None:
            grf = _readonly_float_array(self.grf_observed, "grf_observed", ndim=1)
            if self.J_g is not None and grf.shape != (self.J_g.shape[0],):
                raise ValueError(
                    f"grf_observed length {grf.shape[0]} must match J_g rows ({self.J_g.shape[0]})"
                )
            object.__setattr__(self, "grf_observed", grf)

    def generalized_load(self) -> FloatArray:
        """Return generalized inverse-dynamics load b = M @ a + h (read-only)."""
        load = np.asarray(self.M @ self.a + self.h, dtype=np.float64)
        load.setflags(write=False)
        return load

    def closure_rows(self) -> int:
        """Return the number of scalar closure rows m_c."""
        return int(self.J_c.shape[0])

    def actuated_rows(self) -> IntArray:
        """Return read-only index array of actuated generalized velocity rows."""
        idx = np.asarray(self.topology.actuated_indices, dtype=np.int64)
        idx.setflags(write=False)
        return idx

    def unactuated_rows(self) -> IntArray:
        """Return read-only index array of unactuated generalized velocity rows."""
        idx = np.asarray(self.topology.unactuated_indices, dtype=np.int64)
        idx.setflags(write=False)
        return idx

    def nv(self) -> int:
        """Return generalized velocity dimension n_v."""
        return int(self.v.shape[0])

    def nc(self) -> int:
        """Return closure constraint dimension m_c."""
        return int(self.J_c.shape[0])

    def nu(self) -> int:
        """Return unactuated generalized velocity dimension n_u."""
        return int(len(self.topology.unactuated_indices))

    @property
    def grip_frame_rotation(self) -> FloatArray:
        """LoD convenience accessor for grip frame rotation matrix."""
        return self.grip_frame.rotation

    @property
    def grip_origin(self) -> FloatArray:
        """LoD convenience accessor for grip frame origin vector."""
        return self.grip_frame.origin


@dataclass(frozen=True)
class QuantityRecord:
    """Value object binding a numeric output to all six NSX evidence axes (R1)."""

    values: FloatArray
    units: str
    allocation_dependence: AllocationDependence
    conventions: tuple[str, ...]
    hypotheses: tuple[str, ...]
    model_provenance: ModelProvenance
    input_provenance: InputProvenance
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "units", _validate_nonempty_str(self.units, "units"))
        try:
            dep = AllocationDependence(self.allocation_dependence)
        except ValueError as exc:
            raise ValueError(
                f"Invalid allocation_dependence: {self.allocation_dependence!r}"
            ) from exc
        object.__setattr__(self, "allocation_dependence", dep)

        try:
            status = ValidationStatus(self.validation_status)
        except ValueError as exc:
            raise ValueError(
                f"Invalid validation_status: {self.validation_status!r}"
            ) from exc
        object.__setattr__(self, "validation_status", status)

        if not isinstance(self.model_provenance, ModelProvenance):
            raise TypeError("model_provenance must be a ModelProvenance instance")
        if not isinstance(self.input_provenance, InputProvenance):
            raise TypeError("input_provenance must be an InputProvenance instance")

        conv = _validate_str_tuple(self.conventions, "conventions")
        hyps = _validate_str_tuple(self.hypotheses, "hypotheses")
        object.__setattr__(self, "conventions", conv)
        object.__setattr__(self, "hypotheses", hyps)

        arr = _readonly_float_array(self.values, "values", require_finite=False)
        if arr.size == 0:
            raise ValueError("QuantityRecord values must be non-empty")
        object.__setattr__(self, "values", arr)

        if dep is AllocationDependence.FAMILY_DEPENDENT:
            if len(hyps) == 0:
                raise ValueError(
                    "FAMILY_DEPENDENT QuantityRecord requires at least one hypothesis"
                )
            if not np.isfinite(arr).all():
                raise ValueError(
                    "FAMILY_DEPENDENT QuantityRecord values must be finite"
                )
        elif dep is AllocationDependence.INVARIANT:
            if len(hyps) > 0:
                raise ValueError(
                    f"INVARIANT QuantityRecord forbids hypotheses, got {hyps}"
                )
            if not np.isfinite(arr).all():
                raise ValueError("INVARIANT QuantityRecord values must be finite")
        elif dep is AllocationDependence.UNAVAILABLE:
            if self.reason is None or not self.reason.strip():
                raise ValueError(
                    "UNAVAILABLE QuantityRecord requires a non-empty reason"
                )
            if not np.isnan(arr).all():
                raise ValueError("UNAVAILABLE QuantityRecord requires all-NaN values")
