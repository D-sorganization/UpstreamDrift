"""Explicit authored feasibility initialization, never output acceptance.

Positions are range-projected once and all initial velocities are zeroed.
No solver, image observation, prior, or finished candidate is modified here.
"""

from dataclasses import dataclass, field
import hashlib

import numpy as np

from .hermite_bounds import HermiteBoundsDomain


@dataclass(frozen=True)
class AuthoredKnotChange:
    """One changed physical knot, retaining its original values and source time."""

    coordinate_name: str
    knot_index: int
    source_time: float
    original_position: float
    initialized_position: float
    original_velocity: float
    initialized_velocity: float


@dataclass(frozen=True)
class CoordinateDisplacement:
    """Position displacement in this coordinate's units, never mixed SI units."""

    coordinate_name: str
    maximum_absolute_position_displacement: float


@dataclass(frozen=True)
class AuthoredHermiteInitialization:
    """Copied immutable canonical coefficients and a reproducible change receipt.

    Hashes use SHA256 over canonical little-endian float64 coefficient bytes,
    in ordinary knot-major q then v packing; byte order of the input is ignored.
    Coordinate names/units/domain identity must be bound separately by callers.
    """

    coefficients: tuple[float, ...]
    changes: tuple[AuthoredKnotChange, ...]
    maximum_position_displacements: tuple[CoordinateDisplacement, ...]
    original_coefficient_sha256: str
    initialized_coefficient_sha256: str
    policy: str = field(default="authored_range_project_zero_slopes", init=False)


def _coefficient_hash(coefficients: np.ndarray) -> str:
    canonical = np.asarray(coefficients, dtype="<f8")
    return "sha256:" + hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _validate_candidate(
    domain: HermiteBoundsDomain,
    coefficients: np.ndarray,
    coordinate_names: tuple[str, ...],
) -> np.ndarray:
    if (
        not isinstance(coordinate_names, tuple)
        or len(coordinate_names) != domain.n_dof
        or any(
            not isinstance(name, str) or not name.strip() for name in coordinate_names
        )
        or len(set(coordinate_names)) != len(coordinate_names)
    ):
        raise ValueError(
            "Coordinate names must be immutable, unique, and match the domain"
        )
    candidate = np.asarray(coefficients)
    if (
        candidate.shape != (domain.physical_size,)
        or candidate.dtype.kind not in "ifu"
        or not np.isfinite(candidate).all()
    ):
        raise ValueError("Coefficients must be a finite real numeric physical vector")
    return np.array(candidate, dtype=float, copy=True)


def _changes(
    domain: HermiteBoundsDomain,
    coordinate_names: tuple[str, ...],
    original: np.ndarray,
    initialized: np.ndarray,
) -> tuple[AuthoredKnotChange, ...]:
    old_q, old_v = original.reshape(2, domain.n_knots, domain.n_dof)
    new_q, new_v = initialized.reshape(2, domain.n_knots, domain.n_dof)
    changes = []
    for knot, time in enumerate(domain.knot_times):
        for coordinate, name in enumerate(coordinate_names):
            if (
                old_q[knot, coordinate] == new_q[knot, coordinate]
                and old_v[knot, coordinate] == new_v[knot, coordinate]
            ):
                continue
            changes.append(
                AuthoredKnotChange(
                    name,
                    knot,
                    time,
                    float(old_q[knot, coordinate]),
                    float(new_q[knot, coordinate]),
                    float(old_v[knot, coordinate]),
                    float(new_v[knot, coordinate]),
                )
            )
    return tuple(changes)


def initialize_authored_hermite(
    domain: HermiteBoundsDomain,
    coefficients: np.ndarray,
    coordinate_names: tuple[str, ...],
) -> AuthoredHermiteInitialization:
    """Create an explicitly authored feasible start from physical coefficients.

    This opt-in operation projects finite bounded positions, including equal
    limits, retains unbounded positions, and zeroes ALL domain velocities.
    It is never an implicit repair of a finished fitted trajectory. Coordinates
    absent from this domain (e.g. locked native seed coordinates) are outside
    its scope; callers must independently reject locked-bound conflicts.
    Position displacement is reported separately in each native coordinate's
    own units. The operation does not qualify physical source time or contact.
    """
    original = _validate_candidate(domain, coefficients, coordinate_names)
    initialized = original.copy()
    q, v = initialized.reshape(2, domain.n_knots, domain.n_dof)
    for coordinate, bound in enumerate(domain.coordinate_bounds):
        if bound is not None:
            q[:, coordinate] = np.clip(q[:, coordinate], bound[0], bound[1])
    v.fill(0.0)
    # Public-domain verification is a postcondition, not another initialization.
    domain.encode(initialized)
    original_q = original[: domain.physical_size // 2].reshape(
        domain.n_knots, domain.n_dof
    )
    with np.errstate(over="ignore"):
        displacement = np.max(np.abs(q - original_q), axis=0)
    if not np.isfinite(displacement).all():
        raise ValueError("Per-coordinate position displacement must be representable")
    return AuthoredHermiteInitialization(
        tuple(float(value) for value in initialized),
        _changes(domain, coordinate_names, original, initialized),
        tuple(
            CoordinateDisplacement(name, float(value))
            for name, value in zip(coordinate_names, displacement, strict=True)
        ),
        _coefficient_hash(original),
        _coefficient_hash(initialized),
    )
