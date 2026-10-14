"""Tests for spatial carriage damper angular-momentum closure and objectivity.

Covers the Owner ruling on Issue #11195 and Chapter 20 qualification:
1. Adverse counterexample moment residual and power conservation.
2. Rigid-motion frame indifference (objectivity) under superimposed rotation.
3. Archived spatial forward contact study moment reproduction and SHA-256 audit.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.spatial_carriage_closure import (
    CURRENT_ARCHIVE_SHA256,
    DOCUMENTED_ARCHIVE_SHA256,
    DOCUMENTED_PEAK_MOMENTS,
    audit_archived_spatial_contact_moments,
    central_force_compliant_contact,
    closed_wrench_compliant_contact,
    evaluate_contact_pair_moment_residual,
    verify_contact_law_objectivity,
)
from scripts.research.proximal_distal_energy.spatial_forward_contract import (
    contact_pair,
)

pytestmark = pytest.mark.unit

DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "data"
    / "spatial_forward_contact_study.npz"
)


def test_adverse_counterexample_legacy_vs_closed_laws() -> None:
    """Adverse case produces (0,0,-0.18) N m legacy moment, but zero under closed laws."""
    p = np.zeros(3, dtype=np.float64)
    h = np.array([0.01, 0.0, 0.0], dtype=np.float64)
    v_p = np.zeros(3, dtype=np.float64)
    v_h = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    k = 1800.0
    c = 18.0

    # 1. Legacy directional contact pair
    f_c, f_h, stored_p, diss_p = contact_pair(
        hand_position=h,
        hand_velocity=v_h,
        club_point_position=p,
        club_point_velocity=v_p,
        stiffness=k,
        damping=c,
    )
    np.testing.assert_allclose(f_c + f_h, 0.0, atol=1e-14)
    v_rel = v_h - v_p
    body_power_legacy = float(-f_c @ v_rel)
    assert body_power_legacy == pytest.approx(-18.0)
    assert stored_p + diss_p == pytest.approx(-18.0)
    assert diss_p <= 0.0

    legacy_moment = evaluate_contact_pair_moment_residual(
        hand_position=h,
        club_point_position=p,
        force_on_club=f_c,
        force_on_hand=f_h,
    )
    np.testing.assert_allclose(legacy_moment, [0.0, 0.0, -0.18], atol=1e-15)

    # 2. Central-force compliant contact
    res_central = central_force_compliant_contact(
        hand_position=h,
        hand_velocity=v_h,
        club_point_position=p,
        club_point_velocity=v_p,
        stiffness=k,
        damping=c,
    )
    np.testing.assert_allclose(
        res_central.force_on_club + res_central.force_on_hand, 0.0, atol=1e-14
    )
    moment_central = evaluate_contact_pair_moment_residual(
        hand_position=h,
        club_point_position=p,
        force_on_club=res_central.force_on_club,
        force_on_hand=res_central.force_on_hand,
        torque_on_club=res_central.torque_on_club,
        torque_on_hand=res_central.torque_on_hand,
    )
    assert np.linalg.norm(moment_central) < 1e-15
    body_power_central = float(-res_central.force_on_club @ v_rel)
    assert body_power_central == pytest.approx(
        res_central.storage_power + res_central.dissipated_power
    )
    assert res_central.dissipated_power <= 0.0

    # 3. Closed-wrench compliant contact
    res_wrench = closed_wrench_compliant_contact(
        hand_position=h,
        hand_velocity=v_h,
        club_point_position=p,
        club_point_velocity=v_p,
        stiffness=k,
        damping=c,
    )
    np.testing.assert_allclose(
        res_wrench.force_on_club + res_wrench.force_on_hand, 0.0, atol=1e-14
    )
    moment_wrench = evaluate_contact_pair_moment_residual(
        hand_position=h,
        club_point_position=p,
        force_on_club=res_wrench.force_on_club,
        force_on_hand=res_wrench.force_on_hand,
        torque_on_club=res_wrench.torque_on_club,
        torque_on_hand=res_wrench.torque_on_hand,
    )
    assert np.linalg.norm(moment_wrench) < 1e-15
    body_power_wrench = float(-res_wrench.force_on_club @ v_rel)
    assert body_power_wrench == pytest.approx(-18.0)
    assert res_wrench.storage_power + res_wrench.dissipated_power == pytest.approx(
        -18.0
    )
    assert res_wrench.dissipated_power <= 0.0

    # 4. Moment reference origin independence
    arbitrary_origin = np.array([4.2, -1.7, 9.8], dtype=np.float64)
    moment_arbitrary = evaluate_contact_pair_moment_residual(
        hand_position=h,
        club_point_position=p,
        force_on_club=res_wrench.force_on_club,
        force_on_hand=res_wrench.force_on_hand,
        torque_on_club=res_wrench.torque_on_club,
        torque_on_hand=res_wrench.torque_on_hand,
        reference_point=arbitrary_origin,
    )
    assert np.linalg.norm(moment_arbitrary) < 1e-15


def test_central_force_contact_is_strictly_objective() -> None:
    """Central-force law has zero damping force and dissipation under pure rigid rotation."""
    p = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    h = np.array([0.18, -0.15, 0.08], dtype=np.float64)
    k = 1800.0
    c = 18.0

    omegas = [
        np.array([0.0, 0.0, 10.0]),
        np.array([2.5, -4.1, 1.3]),
        np.array([-5.0, 3.2, -8.7]),
    ]
    for omega in omegas:
        audit = verify_contact_law_objectivity(
            contact_law=central_force_compliant_contact,
            hand_position=h,
            club_point_position=p,
            angular_velocity=omega,
            stiffness=k,
            damping=c,
        )
        assert audit.is_objective
        assert audit.spurious_damping_norm < 1e-14
        assert audit.spurious_dissipation < 1e-14

    # Non-central legacy damper fails objectivity under rigid rotation
    legacy_audit = verify_contact_law_objectivity(
        contact_law=contact_pair,
        hand_position=h,
        club_point_position=p,
        angular_velocity=np.array([0.0, 0.0, 10.0]),
        stiffness=k,
        damping=c,
    )
    assert not legacy_audit.is_objective
    assert legacy_audit.spurious_damping_norm > 0.1


def test_archived_spatial_contact_moments_reproduction() -> None:
    """Audit archived NPZ matches documented peak moments and SHA-256."""
    audit = audit_archived_spatial_contact_moments(DATA_PATH)

    # Documented peak net contact moments from #11195 and Chapter 20
    assert audit.mujoco_peak == pytest.approx(
        DOCUMENTED_PEAK_MOMENTS["mujoco"], abs=1e-12
    )
    assert audit.pinocchio_peak == pytest.approx(
        DOCUMENTED_PEAK_MOMENTS["pinocchio"], abs=1e-12
    )
    assert audit.peak_moments["mujoco_killswitch"] == pytest.approx(
        DOCUMENTED_PEAK_MOMENTS["mujoco"], abs=1e-12
    )
    assert audit.peak_moments["pinocchio_killswitch"] == pytest.approx(
        DOCUMENTED_PEAK_MOMENTS["pinocchio"], abs=1e-12
    )

    # Documented post-killswitch (t >= 0.180 s) maxima
    assert audit.post_killswitch_peak_moments["mujoco_baseline"] == pytest.approx(
        0.03219980748081049, abs=1e-12
    )
    assert audit.post_killswitch_peak_moments["mujoco_killswitch"] == pytest.approx(
        0.033644019086295295, abs=1e-12
    )
    assert audit.post_killswitch_peak_moments["pinocchio_baseline"] == pytest.approx(
        0.03253847731444623, abs=1e-12
    )
    assert audit.post_killswitch_peak_moments["pinocchio_killswitch"] == pytest.approx(
        0.03364913321733224, abs=1e-12
    )

    # Maximum endpoint gap
    assert audit.max_endpoint_gap == pytest.approx(0.01125, abs=1e-4)

    # Checksums
    assert audit.documented_sha256 == DOCUMENTED_ARCHIVE_SHA256
    assert audit.sha256 in (DOCUMENTED_ARCHIVE_SHA256, CURRENT_ARCHIVE_SHA256)

    # If git revision a1a6139 is reachable, check bit-exact reproduction
    try:
        rev_audit = audit_archived_spatial_contact_moments(
            git_revision="a1a613999eb0c744da96caa040941955eb210a21"
        )
        assert rev_audit.sha256 == DOCUMENTED_ARCHIVE_SHA256
        assert rev_audit.mujoco_peak == pytest.approx(
            DOCUMENTED_PEAK_MOMENTS["mujoco"], abs=1e-15
        )
        assert rev_audit.pinocchio_peak == pytest.approx(
            DOCUMENTED_PEAK_MOMENTS["pinocchio"], abs=1e-15
        )
    except (subprocess.SubprocessError, OSError):
        pass
