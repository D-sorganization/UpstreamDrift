"""Scientific contracts for the advanced frame and biological bridge."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.advanced_biological_bridge import (
    build_frame_invariance_audit,
    build_pose_adapter_audit,
    build_redundancy_surface,
    simulate_biological_programs,
)

pytestmark = pytest.mark.scientific


def test_frame_audit_preserves_power_and_jacobian_virtual_work() -> None:
    audit = build_frame_invariance_audit()

    assert audit["maximum_rotation_power_residual_w"] < 1e-11
    assert audit["maximum_transport_power_residual_w"] < 1e-11
    assert audit["maximum_virtual_work_residual_w"] < 1e-11
    assert audit["wrench_order"] == ["force_xyz_n", "couple_xyz_nm"]
    assert audit["twist_order"] == ["linear_xyz_m_s", "angular_xyz_rad_s"]


def test_optional_engine_pose_adapters_preserve_one_canonical_pose() -> None:
    audit = build_pose_adapter_audit()

    assert set(audit) == {"mujoco", "pinocchio", "drake", "opensim", "myosuite"}
    for record in audit.values():
        assert record["status"] == "executed_adapter_round_trip"
        assert record["maximum_translation_residual_m"] < 1e-12
        assert record["maximum_rotation_residual_deg"] < 1e-9
        assert record["maximum_joint_residual_deg"] < 1e-9


def test_redundancy_surface_closes_the_same_joint_moment() -> None:
    surface = build_redundancy_surface(target_torque_nm=10.0, sample_count=31)

    np.testing.assert_allclose(surface.net_torque_nm, 10.0, atol=1e-10)
    assert np.all(np.diff(surface.coactivation) > 0.0)
    assert surface.activation_sum[-1] > surface.activation_sum[0]
    assert surface.stiffness_proxy_nm_rad[-1] > surface.stiffness_proxy_nm_rad[0]
    assert surface.series_elastic_energy_j[-1] > surface.series_elastic_energy_j[0]


def test_persistent_direction_has_smaller_transition_error_for_declared_model() -> None:
    study = simulate_biological_programs(step_s=0.0002)
    persistent = study.programs["persistent_direction"]
    reversal = study.programs["complete_role_reversal"]

    assert persistent.preparation_duration_s == pytest.approx(0.18)
    assert persistent.post_target_torque_nm == pytest.approx(10.0)
    assert reversal.post_target_torque_nm == pytest.approx(10.0)
    assert persistent.post_transition_error_impulse_nms < (
        reversal.post_transition_error_impulse_nms
    )
    assert persistent.minimum_tendon_force_n > 0.0
    assert reversal.minimum_tendon_force_n >= 0.0
    assert study.claim_boundary.startswith("Reduced Hill-type")


def test_invalid_bridge_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="sample_count"):
        build_redundancy_surface(target_torque_nm=10.0, sample_count=1)
    with pytest.raises(ValueError, match="step_s"):
        simulate_biological_programs(step_s=0.0)
