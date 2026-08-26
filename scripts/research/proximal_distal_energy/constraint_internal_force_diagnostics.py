"""Cross-tier contracts for kinematic and force-allocation nullspaces.

Kinematic closure, individual-force allocation, and net-wrench sensing are
separate linear maps. A right nullspace has meaning only relative to its
declared map; equal dimensions do not make the physical quantities
interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.bilateral_wrench_identifiability import (
    audit_linear_map,
    full_hand_wrench_map,
    point_force_wrench_map,
)
from scripts.research.proximal_distal_energy.mechanism_ladder import (
    closed_loop_grip_jacobian,
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PlanarClosedLoopAudit:
    """Dimensionally scaled rank evidence for one planar closure map.

    Singular values and conditioning refer to ``J @ S``, where ``S`` maps a
    dimensionless normalized coordinate increment to the declared angular and
    translational coordinate increments. Rank and nullity are invariant to
    positive diagonal ``S``; singular values and conditioning are not.
    """

    rank: int
    nullity: int
    scaled_singular_values_m: tuple[float, ...]
    smallest_scaled_singular_value_m: float
    scaled_condition_number: float
    maximum_scaled_nullspace_residual_m: float
    angular_coordinate_scale_rad: float
    translation_coordinate_scale_m: float


def _positive_finite(value: float, *, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return scalar


def normalized_point_force_wrench_map(
    contact_positions_m: npt.ArrayLike,
    *,
    reference_length_m: float,
    reference_position_m: npt.ArrayLike = (0.0, 0.0, 0.0),
) -> FloatArray:
    """Return a point-force wrench map with moment rows divided by a length.

    The raw map mixes force and moment rows. Dividing moment rows by one fixed,
    declared reference length puts every output row in force units. Numerical
    conditioning remains conditional on that reference length.
    """

    length = _positive_finite(reference_length_m, name="reference_length_m")
    matrix = point_force_wrench_map(contact_positions_m, reference_position_m)
    normalized = matrix.copy()
    normalized[3:, :] /= length
    return normalized


def normalized_full_hand_wrench_map(
    contact_positions_m: npt.ArrayLike,
    *,
    reference_length_m: float,
    reference_position_m: npt.ArrayLike = (0.0, 0.0, 0.0),
) -> FloatArray:
    """Return a dimensionless two-hand wrench allocation map.

    Force inputs use an arbitrary common force scale. Moment inputs use that
    force scale times ``reference_length_m``; moment outputs are divided by the
    same length. The common force scale cancels, leaving a dimensionless map
    whose conditioning is still conditional on the declared length.
    """

    length = _positive_finite(reference_length_m, name="reference_length_m")
    matrix = full_hand_wrench_map(contact_positions_m, reference_position_m)
    output_scale = np.diag((1.0, 1.0, 1.0, 1.0 / length, 1.0 / length, 1.0 / length))
    input_scale = np.diag(
        (
            1.0,
            1.0,
            1.0,
            length,
            length,
            length,
            1.0,
            1.0,
            1.0,
            length,
            length,
            length,
        )
    )
    return output_scale @ matrix @ input_scale


def planar_closed_loop_audit(
    *,
    lead_angle_rad: float,
    trail_angle_rad: float,
    grip_angle_rad: float,
    angular_coordinate_scale_rad: float,
    translation_coordinate_scale_m: float,
    lead_arm_length_m: float = 0.75,
    trail_arm_length_m: float = 0.78,
    grip_separation_m: float = 0.25,
) -> PlanarClosedLoopAudit:
    """Audit one planar closure map under declared coordinate scales.

    Postconditions: rank and nullity describe the raw and positively scaled
    Jacobians identically. Reported singular values, residual, and condition
    number describe only the scaled map and therefore carry an explicit scale
    contract.
    """

    angular_scale = _positive_finite(
        angular_coordinate_scale_rad,
        name="angular_coordinate_scale_rad coordinate_scale",
    )
    translation_scale = _positive_finite(
        translation_coordinate_scale_m,
        name="translation_coordinate_scale_m coordinate_scale",
    )
    jacobian = closed_loop_grip_jacobian(
        lead_angle_rad=lead_angle_rad,
        trail_angle_rad=trail_angle_rad,
        grip_angle_rad=grip_angle_rad,
        lead_arm_length_m=lead_arm_length_m,
        trail_arm_length_m=trail_arm_length_m,
        grip_separation_m=grip_separation_m,
    )
    coordinate_scales = np.array(
        (
            angular_scale,
            angular_scale,
            translation_scale,
            translation_scale,
            angular_scale,
        )
    )
    scaled_jacobian = jacobian @ np.diag(coordinate_scales)
    audit = audit_linear_map(
        scaled_jacobian,
        relative_tolerance=1e-12,
    )
    raw_rank = audit_linear_map(
        jacobian,
        relative_tolerance=1e-12,
    ).rank
    if raw_rank != audit.rank:
        raise ValueError(
            "positive coordinate scaling changed the numerical rank decision; "
            "review scales and tolerance"
        )
    residual = (
        float(np.max(np.abs(scaled_jacobian @ audit.right_null_basis)))
        if audit.right_null_basis.size
        else 0.0
    )
    full_row_rank = audit.rank == scaled_jacobian.shape[0]
    return PlanarClosedLoopAudit(
        rank=audit.rank,
        nullity=audit.nullity,
        scaled_singular_values_m=tuple(float(value) for value in audit.singular_values),
        smallest_scaled_singular_value_m=float(audit.singular_values[-1]),
        scaled_condition_number=(
            audit.nonzero_condition_number if full_row_rank else float("inf")
        ),
        maximum_scaled_nullspace_residual_m=residual,
        angular_coordinate_scale_rad=angular_scale,
        translation_coordinate_scale_m=translation_scale,
    )
