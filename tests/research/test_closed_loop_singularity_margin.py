"""Exact-closure and scaled singular-margin contracts for issue #9113."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.constraint_internal_force_diagnostics import (
    audit_scaled_planar_closure_jacobian,
)
from scripts.research.proximal_distal_energy.closed_loop_singularity_margin import (
    PlanarClosedGeometry,
    PlanarCoordinateScale,
    audit_closed_loop_orbit,
    audit_feasible_configuration,
    audit_triangle_degeneracies,
    feasible_closed_loop_configuration,
    planar_closure_residual,
)
from scripts.research.proximal_distal_energy.mechanism_ladder import (
    closed_loop_grip_jacobian,
)

pytestmark = pytest.mark.unit

GEOMETRY = PlanarClosedGeometry(
    lead_arm_length_m=0.75,
    trail_arm_length_m=0.78,
    grip_separation_m=0.25,
)
SCALE = PlanarCoordinateScale(
    angular_coordinate_scale_rad=1.0,
    translation_coordinate_scale_m=0.75,
)


@pytest.mark.parametrize("branch", (-1, 1))
@pytest.mark.parametrize("phase_rad", (-2.1, -0.3, 0.0, 1.7))
def test_feasible_configuration_closes_both_contacts(
    branch: int,
    phase_rad: float,
) -> None:
    configuration = feasible_closed_loop_configuration(
        GEOMETRY,
        phase_rad=phase_rad,
        branch=branch,
    )

    residual = planar_closure_residual(configuration, GEOMETRY)

    assert np.max(np.abs(residual)) < 1e-12
    assert configuration.triangle_sine_margin == pytest.approx(0.3201986380342093)
    assert configuration.lower_degeneracy_distance_m == pytest.approx(0.22)
    assert configuration.upper_degeneracy_distance_m == pytest.approx(1.28)


def test_nominal_closed_orbit_is_regular_and_phase_invariant() -> None:
    audit = audit_closed_loop_orbit(
        GEOMETRY,
        SCALE,
        phase_sample_count=181,
        relative_tolerance=1e-12,
    )

    assert audit.sample_count == 362
    assert audit.minimum_rank == audit.maximum_rank == 4
    assert audit.minimum_nullity == audit.maximum_nullity == 1
    assert audit.maximum_closure_residual_m < 1e-12
    assert audit.maximum_scaled_nullspace_residual_m < 1e-12
    assert audit.maximum_scaled_condition_number == pytest.approx(
        audit.minimum_scaled_condition_number,
        rel=1e-12,
    )
    assert audit.maximum_smallest_scaled_singular_value_m == pytest.approx(
        audit.minimum_smallest_scaled_singular_value_m,
        rel=1e-12,
    )
    assert audit.maximum_scaled_singular_value_spread_m < 1e-12
    assert audit.maximum_scaled_singular_value_spread_m == 2e-15


def test_exact_triangle_degeneracies_add_one_velocity_null_mode() -> None:
    audit = audit_triangle_degeneracies(
        GEOMETRY,
        SCALE,
        relative_tolerance=1e-12,
    )

    assert audit.lower_geometry.grip_separation_m == pytest.approx(0.03)
    assert audit.upper_geometry.grip_separation_m == pytest.approx(1.53)
    assert audit.lower.rank == audit.upper.rank == 3
    assert audit.lower.nullity == audit.upper.nullity == 2
    assert audit.lower_position_closure_residual_m < 1e-12
    assert audit.upper_position_closure_residual_m < 1e-12
    assert audit.lower.maximum_scaled_nullspace_residual_m < 1e-12
    assert audit.upper.maximum_scaled_nullspace_residual_m < 1e-12


def test_conditioning_is_explicitly_coordinate_scale_dependent() -> None:
    audits = [
        audit_closed_loop_orbit(
            GEOMETRY,
            PlanarCoordinateScale(1.0, translation_scale_m),
            phase_sample_count=41,
            relative_tolerance=1e-12,
        )
        for translation_scale_m in (0.50, 0.75, 1.00)
    ]

    assert [audit.minimum_rank for audit in audits] == [4, 4, 4]
    assert [audit.scale.translation_coordinate_scale_m for audit in audits] == [
        0.50,
        0.75,
        1.00,
    ]
    conditions = [audit.maximum_scaled_condition_number for audit in audits]
    assert conditions[0] < conditions[1] < conditions[2]


def test_equivalent_length_units_preserve_rank_and_condition() -> None:
    configuration_m = feasible_closed_loop_configuration(
        GEOMETRY, phase_rad=0.4, branch=1
    )
    audit_m = audit_feasible_configuration(
        configuration_m,
        GEOMETRY,
        SCALE,
        relative_tolerance=1e-12,
    )
    jacobian_m = closed_loop_grip_jacobian(
        lead_angle_rad=configuration_m.lead_angle_rad,
        trail_angle_rad=configuration_m.trail_angle_rad,
        grip_angle_rad=configuration_m.grip_angle_rad,
        lead_arm_length_m=GEOMETRY.lead_arm_length_m,
        trail_arm_length_m=GEOMETRY.trail_arm_length_m,
        grip_separation_m=GEOMETRY.grip_separation_m,
    )
    centimetre_jacobian = 100.0 * jacobian_m
    centimetre_jacobian[:, 2:4] /= 100.0
    audit_cm = audit_scaled_planar_closure_jacobian(
        centimetre_jacobian,
        (1.0, 75.0),
        relative_tolerance=1e-12,
    )

    assert audit_cm.rank == audit_m.rank
    assert audit_cm.nullity == audit_m.nullity
    assert audit_cm.scaled_condition_number == pytest.approx(
        audit_m.scaled_condition_number,
        rel=1e-12,
    )
    np.testing.assert_allclose(
        audit_cm.scaled_singular_values_m,
        100.0 * np.asarray(audit_m.scaled_singular_values_m),
        rtol=1e-12,
        atol=1e-12,
    )


def test_manufactured_row_dependency_is_detected() -> None:
    configuration = feasible_closed_loop_configuration(
        GEOMETRY, phase_rad=0.4, branch=1
    )
    jacobian = closed_loop_grip_jacobian(
        lead_angle_rad=configuration.lead_angle_rad,
        trail_angle_rad=configuration.trail_angle_rad,
        grip_angle_rad=configuration.grip_angle_rad,
        lead_arm_length_m=GEOMETRY.lead_arm_length_m,
        trail_arm_length_m=GEOMETRY.trail_arm_length_m,
        grip_separation_m=GEOMETRY.grip_separation_m,
    )
    manufactured = jacobian.copy()
    manufactured[3] = manufactured[2]

    audit = audit_scaled_planar_closure_jacobian(
        manufactured,
        (1.0, 0.75),
        relative_tolerance=1e-12,
    )

    assert audit.rank == 3
    assert audit.nullity == 2


def test_near_boundary_rank_is_tolerance_sensitive_without_changing_closure() -> None:
    lower_span_m = abs(GEOMETRY.lead_arm_length_m - GEOMETRY.trail_arm_length_m)
    near_geometry = PlanarClosedGeometry(0.75, 0.78, lower_span_m + 1e-8)
    configuration = feasible_closed_loop_configuration(
        near_geometry,
        phase_rad=0.0,
        branch=1,
    )
    audits = [
        audit_feasible_configuration(
            configuration,
            near_geometry,
            SCALE,
            relative_tolerance=tolerance,
        )
        for tolerance in (1e-12, 1e-8, 1e-6, 1e-4)
    ]

    assert [audit.rank for audit in audits] == [4, 4, 4, 3]
    assert np.max(np.abs(planar_closure_residual(configuration, near_geometry))) < 1e-12


@pytest.mark.parametrize(
    "geometry",
    (
        PlanarClosedGeometry(0.0, 0.78, 0.25),
        PlanarClosedGeometry(0.75, -0.1, 0.25),
        PlanarClosedGeometry(0.75, 0.78, 2.0),
    ),
)
def test_feasible_configuration_rejects_invalid_or_impossible_geometry(
    geometry: PlanarClosedGeometry,
) -> None:
    with pytest.raises(ValueError):
        feasible_closed_loop_configuration(geometry, phase_rad=0.0, branch=1)


@pytest.mark.parametrize("branch", (-2, 0, 2))
def test_feasible_configuration_rejects_invalid_branch(branch: int) -> None:
    with pytest.raises(ValueError, match="branch"):
        feasible_closed_loop_configuration(GEOMETRY, phase_rad=0.0, branch=branch)


def test_orbit_rejects_insufficient_phase_grid() -> None:
    with pytest.raises(ValueError, match="at least three"):
        audit_closed_loop_orbit(
            GEOMETRY,
            SCALE,
            phase_sample_count=2,
            relative_tolerance=1e-12,
        )


def test_equal_arm_lengths_make_lower_degeneracy_undefined() -> None:
    with pytest.raises(ValueError, match="coincident"):
        audit_triangle_degeneracies(
            PlanarClosedGeometry(0.75, 0.75, 0.25),
            SCALE,
            relative_tolerance=1e-12,
        )


@pytest.mark.parametrize(
    "scale",
    (
        PlanarCoordinateScale(0.0, 0.75),
        PlanarCoordinateScale(1.0, -0.75),
        PlanarCoordinateScale(float("nan"), 0.75),
    ),
)
def test_invalid_coordinate_scales_fail_closed(scale: PlanarCoordinateScale) -> None:
    configuration = feasible_closed_loop_configuration(
        GEOMETRY, phase_rad=0.0, branch=1
    )
    with pytest.raises(ValueError, match="coordinate_scale"):
        audit_feasible_configuration(
            configuration,
            GEOMETRY,
            scale,
            relative_tolerance=1e-12,
        )
