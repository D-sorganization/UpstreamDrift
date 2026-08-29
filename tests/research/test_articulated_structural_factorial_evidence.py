"""Fail-closed tests for the retained structural-factorial time histories."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_evidence import (
    EVIDENCE_SIDECAR_SCHEMA,
    validate_structural_evidence_arrays,
)

pytestmark = pytest.mark.scientific


def _complete_arrays() -> dict[str, np.ndarray]:
    time = np.array([0.0, 0.1, 0.2])
    contact_force = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    contact_power = np.array([2.0, 4.0, 6.0])
    cumulative_dissipation = np.array([0.0, -0.1, -0.3])
    total_energy = np.array([10.0, 9.9, 9.7])
    active_station = np.array(
        [
            [[False, False], [False, False]],
            [[True, False], [False, False]],
            [[False, True], [False, False]],
        ]
    )
    return {
        "time_s": time,
        "q": np.zeros((3, 20)),
        "qd": np.zeros((3, 20)),
        "elastic_coordinates": np.zeros((3, 3)),
        "elastic_velocities": np.zeros((3, 3)),
        "base_coordinates": np.zeros((3, 3)),
        "base_velocities": np.zeros((3, 3)),
        "station_force_on_club_n": np.zeros((3, 2, 2, 3)),
        "active_station": active_station,
        "active_set_transition": np.array([False, True, True]),
        "net_club_force_n": contact_force,
        "contact_power_w": contact_power,
        "cumulative_contact_impulse_n_s": np.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.6, 0.0, 0.0]]
        ),
        "cumulative_contact_work_j": np.array([0.0, 0.3, 0.8]),
        "maximum_station_force_n": np.zeros(3),
        "active_station_count": np.array([0, 1, 1]),
        "force_couple_vector_nm": np.zeros((3, 3)),
        "grip_strain_energy_j": np.zeros(3),
        "grip_dissipation_power_w": np.zeros(3),
        "virtual_power_residual_w": np.zeros(3),
        "shaft_strain_energy_j": np.zeros(3),
        "shaft_damping_power_w": np.zeros(3),
        "shaft_power_residual_w": np.zeros(3),
        "ground_force_n": np.zeros((3, 3)),
        "ground_intrinsic_free_moment_nm": np.zeros(3),
        "ground_transported_moment_nm": np.zeros(3),
        "ground_strain_energy_j": np.zeros(3),
        "ground_damping_power_w": np.zeros(3),
        "ground_power_residual_w": np.zeros(3),
        "total_mechanical_energy_j": total_energy,
        "total_energy_j": total_energy,
        "cumulative_dissipation_j": cumulative_dissipation,
        "work_energy_residual_j": np.zeros(3),
        "tip_bending_m": np.zeros((3, 2)),
        "twist_angle_rad": np.zeros(3),
        "base_translation_m": np.zeros((3, 2)),
        "base_pitch_rad": np.zeros(3),
    }


def test_complete_evidence_sidecar_closes_integral_and_energy_identities() -> None:
    assert EVIDENCE_SIDECAR_SCHEMA == "articulated-structural-factorial-evidence/1.0.0"
    validate_structural_evidence_arrays(_complete_arrays())


def test_evidence_sidecar_rejects_the_legacy_minimal_array_set() -> None:
    arrays = _complete_arrays()
    del arrays["contact_power_w"]

    with pytest.raises(ValueError, match="missing required arrays: contact_power_w"):
        validate_structural_evidence_arrays(arrays)


def test_evidence_sidecar_rejects_an_unregistered_station_transition() -> None:
    arrays = _complete_arrays()
    arrays["active_set_transition"] = np.array([False, True, False])

    with pytest.raises(ValueError, match="active-set transition history"):
        validate_structural_evidence_arrays(arrays)


def test_evidence_sidecar_rejects_corrupt_cumulative_contact_work() -> None:
    arrays = _complete_arrays()
    arrays["cumulative_contact_work_j"][-1] += 0.01

    with pytest.raises(ValueError, match="cumulative contact work"):
        validate_structural_evidence_arrays(arrays)
