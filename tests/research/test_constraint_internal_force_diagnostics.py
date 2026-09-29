"""Analytical contracts for constraint and internal-force nullspaces."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.bilateral_wrench_identifiability import (
    audit_linear_map,
    full_hand_wrench_map,
    point_force_wrench_map,
)
from scripts.research.proximal_distal_energy.constraint_internal_force_diagnostics import (
    normalized_full_hand_wrench_map,
    normalized_point_force_wrench_map,
    planar_closed_loop_audit,
)

pytestmark = pytest.mark.scientific


def _contacts(span_m: float) -> np.ndarray:
    return np.array(((-0.5 * span_m, 0.0, 0.0), (0.5 * span_m, 0.0, 0.0)))


def _planar_audit(grip_angle_rad: float, *, translation_scale_m: float = 0.75):
    return planar_closed_loop_audit(
        lead_angle_rad=0.0,
        trail_angle_rad=0.0,
        grip_angle_rad=grip_angle_rad,
        angular_coordinate_scale_rad=1.0,
        translation_coordinate_scale_m=translation_scale_m,
    )


def test_normalized_wrench_map_uses_declared_reference_length() -> None:
    contacts = _contacts(0.2)
    raw = point_force_wrench_map(contacts)
    normalized = normalized_point_force_wrench_map(contacts, reference_length_m=0.1)

    np.testing.assert_allclose(normalized[:3], raw[:3])
    np.testing.assert_allclose(normalized[3:], raw[3:] / 0.1)
    audit = audit_linear_map(normalized)
    assert audit.rank == 5
    assert audit.nullity == 1
    assert audit.nonzero_condition_number == pytest.approx(1.0)


def test_coincident_contacts_create_three_invisible_force_modes() -> None:
    audit = audit_linear_map(
        normalized_point_force_wrench_map(_contacts(0.0), reference_length_m=0.1)
    )

    assert audit.rank == 3
    assert audit.nullity == 3


def test_full_hand_wrench_map_normalizes_input_and_output_moments() -> None:
    contacts = _contacts(0.2)
    raw = full_hand_wrench_map(contacts)
    normalized = normalized_full_hand_wrench_map(
        contacts,
        reference_length_m=0.1,
    )
    output_scale = np.diag((1.0, 1.0, 1.0, 10.0, 10.0, 10.0))
    input_scale = np.diag((1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1))

    np.testing.assert_allclose(normalized, output_scale @ raw @ input_scale)
    audit = audit_linear_map(normalized)
    assert audit.rank == 6
    assert audit.nullity == 6


def test_near_coincident_rank_is_tolerance_sensitive() -> None:
    matrix = normalized_point_force_wrench_map(_contacts(1e-6), reference_length_m=0.1)

    assert audit_linear_map(matrix, relative_tolerance=1e-12).rank == 5
    assert audit_linear_map(matrix, relative_tolerance=1e-8).rank == 5
    assert audit_linear_map(matrix, relative_tolerance=1e-6).rank == 3


def test_planar_closed_loop_singular_geometry_adds_velocity_mode() -> None:
    regular = _planar_audit(0.0)
    singular = _planar_audit(np.pi / 2.0)

    assert regular.rank == 4
    assert regular.nullity == 1
    assert singular.rank == 3
    assert singular.nullity == 2
    assert singular.smallest_scaled_singular_value_m < 1e-15


def test_planar_conditioning_declares_coordinate_scale_dependence() -> None:
    narrow_scale = _planar_audit(np.pi / 2.0 - 1e-3, translation_scale_m=0.25)
    broad_scale = _planar_audit(np.pi / 2.0 - 1e-3, translation_scale_m=1.0)

    assert narrow_scale.rank == broad_scale.rank == 4
    assert narrow_scale.nullity == broad_scale.nullity == 1
    assert narrow_scale.scaled_condition_number != pytest.approx(
        broad_scale.scaled_condition_number
    )
    assert narrow_scale.translation_coordinate_scale_m == pytest.approx(0.25)
    assert broad_scale.translation_coordinate_scale_m == pytest.approx(1.0)


def test_planar_near_singular_conditioning_diverges_at_fixed_scale() -> None:
    conditions = np.array(
        [
            _planar_audit(np.pi / 2.0 - offset).scaled_condition_number
            for offset in (1e-2, 1e-4, 1e-6)
        ]
    )

    assert np.all(np.isfinite(conditions))
    assert np.all(np.diff(conditions) > 0.0)


@pytest.mark.parametrize("reference_length", [0.0, -1.0, float("nan")])
def test_normalized_wrench_map_rejects_invalid_reference_length(
    reference_length: float,
) -> None:
    with pytest.raises(ValueError, match="reference_length_m must be positive"):
        normalized_point_force_wrench_map(
            _contacts(0.2), reference_length_m=reference_length
        )


@pytest.mark.parametrize(
    ("angular_scale", "translation_scale"),
    [(0.0, 0.75), (1.0, -0.75), (float("nan"), 0.75)],
)
def test_planar_audit_rejects_invalid_coordinate_scales(
    angular_scale: float, translation_scale: float
) -> None:
    with pytest.raises(ValueError, match="coordinate_scale"):
        planar_closed_loop_audit(
            lead_angle_rad=0.0,
            trail_angle_rad=0.0,
            grip_angle_rad=0.0,
            angular_coordinate_scale_rad=angular_scale,
            translation_coordinate_scale_m=translation_scale,
        )
