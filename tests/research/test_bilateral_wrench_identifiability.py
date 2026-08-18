from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.bilateral_wrench_identifiability import (
    audit_linear_map,
    full_hand_wrench_map,
    internal_axial_measurement,
    point_force_wrench_map,
)


def _contacts(span_m: float = 0.20) -> np.ndarray:
    return np.array(((-0.5 * span_m, 0.0, 0.0), (0.5 * span_m, 0.0, 0.0)))


def test_separated_point_contacts_have_one_invisible_internal_force_mode() -> None:
    positions = _contacts()
    matrix = point_force_wrench_map(positions)
    audit = audit_linear_map(matrix)
    direction = (positions[1] - positions[0]) / np.linalg.norm(
        positions[1] - positions[0]
    )
    expected_null = np.concatenate((direction, -direction)) / np.sqrt(2.0)

    assert matrix.shape == (6, 6)
    assert audit.rank == 5
    assert audit.nullity == 1
    assert np.linalg.norm(matrix @ expected_null) < 1e-12
    assert abs(np.dot(audit.right_null_basis[:, 0], expected_null)) > 1.0 - 1e-12


def test_axial_internal_measurement_closes_point_force_rank_gap() -> None:
    positions = _contacts()
    matrix = point_force_wrench_map(positions)
    augmented = np.vstack((matrix, internal_axial_measurement(positions)))

    assert audit_linear_map(augmented).rank == 6
    assert audit_linear_map(augmented).nullity == 0


def test_net_wrench_cannot_identify_individual_six_axis_hand_wrenches() -> None:
    audit = audit_linear_map(full_hand_wrench_map(_contacts()))

    assert audit.matrix_shape == (6, 12)
    assert audit.rank == 6
    assert audit.nullity == 6


def test_grip_span_controls_nonzero_conditioning_but_not_rank() -> None:
    spans = np.array((0.08, 0.12, 0.18, 0.24))
    audits = [
        audit_linear_map(point_force_wrench_map(_contacts(span))) for span in spans
    ]
    minimum_nonzero = np.array(
        [audit.minimum_nonzero_singular_value for audit in audits]
    )

    assert all(audit.rank == 5 for audit in audits)
    assert np.all(np.diff(minimum_nonzero) > 0.0)
    assert audits[0].nonzero_condition_number > audits[-1].nonzero_condition_number


def test_proper_rotation_preserves_rank_and_singular_values() -> None:
    positions = _contacts(0.17)
    angle = np.deg2rad(37.0)
    rotation = np.array(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    base = audit_linear_map(point_force_wrench_map(positions))
    rotated = audit_linear_map(point_force_wrench_map(positions @ rotation.T))

    assert rotated.rank == base.rank
    assert np.allclose(rotated.singular_values, base.singular_values, atol=1e-12)


def test_contracts_reject_invalid_or_degenerate_contact_geometry() -> None:
    with pytest.raises(ValueError, match="shape"):
        point_force_wrench_map(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="distinct"):
        internal_axial_measurement(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        point_force_wrench_map(np.array(((0.0, 0.0, 0.0), (np.nan, 0.0, 0.0))))
