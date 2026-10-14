"""Tests for the seven named MOSAIC qualification gates."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.qualification_gates import (
    GateStatus,
    GateName,
    GateResult,
    evaluate_qualification_gates,
    check_kinematic_whiteness,
    check_declared_discrepancy,
    check_physical_consistency,
    check_observability,
    check_open_loop_replay,
    check_closed_loop_effort,
    check_cross_trial_consistency,
)

pytestmark = pytest.mark.unit


def test_gate_names_and_individually_reportable_checks() -> None:
    expected_names = {
        GateName.KINEMATIC_WHITENESS,
        GateName.DECLARED_DISCREPANCY,
        GateName.PHYSICAL_CONSISTENCY,
        GateName.OBSERVABILITY,
        GateName.OPEN_LOOP_REPLAY,
        GateName.CLOSED_LOOP_EFFORT,
        GateName.CROSS_TRIAL_CONSISTENCY,
    }
    assert len(expected_names) == 7


def test_missing_input_reports_unqualified_never_pass() -> None:
    # All checks with None or missing input must return UNQUALIFIED
    res1 = check_kinematic_whiteness(residual_autocorr=None, whiteness_metric=None)
    assert res1.status == GateStatus.UNQUALIFIED
    assert (
        "unavailable" in res1.message.lower() or "unqualified" in res1.message.lower()
    )

    res2 = check_declared_discrepancy(
        sigma_tau=None, basis_accel_error=None, mass_scale=None
    )
    assert res2.status == GateStatus.UNQUALIFIED

    res3 = check_physical_consistency(parameters=None)
    assert res3.status == GateStatus.UNQUALIFIED

    res4 = check_observability(observability_report=None)
    assert res4.status == GateStatus.UNQUALIFIED

    res5 = check_open_loop_replay(replays=None)
    assert res5.status == GateStatus.UNQUALIFIED

    res6 = check_closed_loop_effort(replays=None)
    assert res6.status == GateStatus.UNQUALIFIED

    res7 = check_cross_trial_consistency(template=None, modes=None)
    assert res7.status == GateStatus.UNQUALIFIED


def test_physical_consistency_gate_checks_cone() -> None:
    # Consistent parameters: m > 0, Ic > 0
    pi_valid = np.array(
        [1.3, 0.585, 0.0, 0.343, 0.7, 0.175, 0.0, 0.064, 0.3, 0.09, 0.0, 0.03]
    )
    res_pass = check_physical_consistency(pi_valid)
    assert res_pass.status == GateStatus.PASSED

    # Inconsistent parameters: negative mass or negative central inertia
    pi_invalid = pi_valid.copy()
    pi_invalid[0] = -1.0
    res_fail = check_physical_consistency(pi_invalid)
    assert res_fail.status == GateStatus.FAILED
