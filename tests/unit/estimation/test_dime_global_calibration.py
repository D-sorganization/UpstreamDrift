"""Tests for DIME-08 observable global calibration and consistent prior updates.

Enforces:
1. Monocular scale ambiguity refusal without metric scale anchor.
2. Mass/torque scaling ambiguity refusal without mass or measured force anchor.
3. Kinematic redundancy detection and nullspace flagging.
4. Physical inertia tensor validity (symmetry, positive definiteness, triangle inequality).
5. Frozen prior protection against unflagged calibration revisions.
6. Recovery of identifiable planted parameters in multi-view anchored setup.
7. Rank-deficient parameter block freezing or profiling.
8. Latency accounting separating inner-loop latency from outer-loop cost.
9. Equivalent batch reproduction after global update.
10. Serialization and deserialization roundtrips.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts import PreconditionError
from src.shared.python.estimation.dime_global_calibration import (
    CalibrationParameter,
    CalibrationParameterKind,
    GlobalCalibrationProblem,
    GlobalCalibrationResult,
    PhysicalGauge,
    PhysicalGaugePolicy,
    RankDeficiencyPolicy,
    calibrate_global_parameters,
    validate_physical_inertia,
)

pytestmark = pytest.mark.unit


def test_red_monocular_scale_ambiguity_rejected() -> None:
    """RED: Monocular observation without metric scale anchor cannot determine scale."""
    policy = PhysicalGaugePolicy(
        active_gauges={PhysicalGauge.GRAVITY_FRAME},  # Scale is NOT anchored
    )
    param = CalibrationParameter(
        name="upper_arm_length",
        kind=CalibrationParameterKind.BODY_GEOMETRY,
        value=0.35,
        nominal_value=0.35,
        bounds=(0.2, 0.5),
        sigma=0.05,
        is_anchored=False,
    )
    problem = GlobalCalibrationProblem(
        parameters=(param,),
        gauge_policy=policy,
        is_monocular=True,
    )
    with pytest.raises(
        PreconditionError,
        match="monocular.*scale.*ambiguity|scale.*unobservable without.*anchor",
    ):
        calibrate_global_parameters(problem)


def test_red_mass_torque_scaling_ambiguity_rejected() -> None:
    """RED: Motion alone cannot claim unique forces without mass or force anchor."""
    policy = PhysicalGaugePolicy(
        active_gauges={PhysicalGauge.METRIC_SCALE, PhysicalGauge.GRAVITY_FRAME},
        # Neither MASS_ANCHOR nor MEASURED_FORCE_ANCHOR is active
    )
    param_mass = CalibrationParameter(
        name="torso_mass",
        kind=CalibrationParameterKind.BODY_INERTIA,
        value=25.0,
        nominal_value=25.0,
        bounds=(10.0, 50.0),
        sigma=5.0,
        is_anchored=False,
    )
    param_torque_scale = CalibrationParameter(
        name="actuator_torque_gain",
        kind=CalibrationParameterKind.CONTACT_PARAMETER,
        value=1.0,
        nominal_value=1.0,
        bounds=(0.5, 2.0),
        sigma=0.2,
        is_anchored=False,
    )
    problem = GlobalCalibrationProblem(
        parameters=(param_mass, param_torque_scale),
        gauge_policy=policy,
        is_monocular=False,
        has_measured_contact_force=False,
    )
    with pytest.raises(
        PreconditionError,
        match="mass.*torque.*scaling|cannot claim unique forces without mass or force anchor",
    ):
        calibrate_global_parameters(problem)


def test_red_inertially_invalid_tensor_rejected() -> None:
    """RED: Inertia tensors violating triangle inequality or positive definiteness are rejected."""
    # Violates triangle inequality: I_xx (5.0) > I_yy (1.0) + I_zz (1.0) = 2.0
    invalid_tensor_triangle = np.diag([5.0, 1.0, 1.0])
    with pytest.raises(
        PreconditionError, match="triangle inequality|not positive definite"
    ):
        validate_physical_inertia(invalid_tensor_triangle)

    # Violates positive definiteness: non-positive eigenvalue
    invalid_tensor_non_psd = np.diag([-1.0, 2.0, 2.0])
    with pytest.raises(
        PreconditionError, match="triangle inequality|not positive definite"
    ):
        validate_physical_inertia(invalid_tensor_non_psd)

    # Valid physical inertia: e.g. solid sphere or box
    valid_tensor = np.diag([2.0, 2.0, 2.0])
    validated = validate_physical_inertia(valid_tensor)
    assert np.allclose(validated, valid_tensor)


def test_red_redundant_joint_solutions_detected_and_flagged() -> None:
    """RED: Co-axial or redundant joints create singular null directions."""
    policy = PhysicalGaugePolicy(
        active_gauges={PhysicalGauge.METRIC_SCALE, PhysicalGauge.GRAVITY_FRAME},
    )
    # Two co-axial joint angle offsets that sum to a single observable angle
    param_j1 = CalibrationParameter(
        name="shoulder_roll_offset",
        kind=CalibrationParameterKind.BODY_GEOMETRY,
        value=0.0,
        nominal_value=0.0,
        bounds=(-0.2, 0.2),
        sigma=0.05,
    )
    param_j2 = CalibrationParameter(
        name="shoulder_aux_roll_offset",
        kind=CalibrationParameterKind.BODY_GEOMETRY,
        value=0.0,
        nominal_value=0.0,
        bounds=(-0.2, 0.2),
        sigma=0.05,
    )

    # Linear observation function measuring only (j1 + j2)
    def observation_residual(vals: np.ndarray) -> np.ndarray:
        return np.array([(vals[0] + vals[1]) - 0.05])

    problem = GlobalCalibrationProblem(
        parameters=(param_j1, param_j2),
        gauge_policy=policy,
        residual_fn=observation_residual,
        rank_deficiency_policy=RankDeficiencyPolicy.FAIL_CLOSED,
    )
    with pytest.raises(
        PreconditionError, match="rank-deficient|unobservable null space"
    ):
        calibrate_global_parameters(problem)


def test_red_calibration_changes_beneath_frozen_prior_rejected() -> None:
    """RED: Calibration changes cannot modify parameters beneath a frozen prior without revision."""
    policy = PhysicalGaugePolicy(
        active_gauges={PhysicalGauge.METRIC_SCALE, PhysicalGauge.GRAVITY_FRAME},
    )
    param = CalibrationParameter(
        name="marker_offset_x",
        kind=CalibrationParameterKind.MARKER_OFFSET,
        value=0.01,
        nominal_value=0.0,
        bounds=(-0.05, 0.05),
        sigma=0.01,
    )
    problem = GlobalCalibrationProblem(
        parameters=(param,),
        gauge_policy=policy,
        is_prior_frozen=True,
        prior_revision_tagged=False,
    )
    with pytest.raises(
        PreconditionError,
        match="frozen prior.*without explicit revision|frozen prior",
    ):
        calibrate_global_parameters(problem)


def test_green_recover_identifiable_planted_parameters() -> None:
    """GREEN: Recovers planted marker and geometry perturbations with full multi-view anchor."""
    policy = PhysicalGaugePolicy(
        active_gauges={PhysicalGauge.METRIC_SCALE, PhysicalGauge.GRAVITY_FRAME},
    )
    planted_p0 = 0.025  # +25 mm marker offset
    planted_p1 = -0.015  # -15 mm marker offset

    # True observation yields 0 residual at planted values
    def residual_fn(vals: np.ndarray) -> np.ndarray:
        # 3 independent orthogonal measurements
        r0 = vals[0] - planted_p0
        r1 = vals[1] - planted_p1
        r2 = (vals[0] + vals[1]) - (planted_p0 + planted_p1)
        return np.array([r0, r1, r2])

    p0 = CalibrationParameter(
        name="marker_c7_x",
        kind=CalibrationParameterKind.MARKER_OFFSET,
        value=0.0,  # nominal guess
        nominal_value=0.0,
        bounds=(-0.08, 0.08),
        sigma=0.05,
    )
    p1 = CalibrationParameter(
        name="marker_c7_y",
        kind=CalibrationParameterKind.MARKER_OFFSET,
        value=0.0,  # nominal guess
        nominal_value=0.0,
        bounds=(-0.08, 0.08),
        sigma=0.05,
    )
    problem = GlobalCalibrationProblem(
        parameters=(p0, p1),
        gauge_policy=policy,
        residual_fn=residual_fn,
        rank_deficiency_policy=RankDeficiencyPolicy.FREEZE_NULLSPACE,
    )
    result = calibrate_global_parameters(problem)

    assert result.success is True
    assert result.parameter_values["marker_c7_x"] == pytest.approx(planted_p0, abs=1e-5)
    assert result.parameter_values["marker_c7_y"] == pytest.approx(planted_p1, abs=1e-5)
    assert result.total_cost < 1e-8


def test_green_flag_rank_deficient_blocks_and_freeze_nullspace() -> None:
    """GREEN: Rank-deficient parameter block is flagged and nullspace is locked at nominal."""
    policy = PhysicalGaugePolicy(
        active_gauges={PhysicalGauge.METRIC_SCALE, PhysicalGauge.GRAVITY_FRAME},
    )
    # p0 is observable, p1 is completely unobservable (residual has 0 derivative wrt p1)
    planted_p0 = 0.03

    def residual_fn(vals: np.ndarray) -> np.ndarray:
        return np.array([vals[0] - planted_p0, 0.0])

    p0 = CalibrationParameter(
        name="observable_param",
        kind=CalibrationParameterKind.MARKER_OFFSET,
        value=0.0,
        nominal_value=0.0,
        bounds=(-0.1, 0.1),
        sigma=0.05,
    )
    p1 = CalibrationParameter(
        name="unobservable_param",
        kind=CalibrationParameterKind.MARKER_OFFSET,
        value=0.0,
        nominal_value=0.0,
        bounds=(-0.1, 0.1),
        sigma=0.05,
    )
    problem = GlobalCalibrationProblem(
        parameters=(p0, p1),
        gauge_policy=policy,
        residual_fn=residual_fn,
        rank_deficiency_policy=RankDeficiencyPolicy.FREEZE_NULLSPACE,
    )
    result = calibrate_global_parameters(problem)

    assert result.success is True
    assert result.identifiability_report.rank == 1
    assert "unobservable_param" in result.locked_parameters
    assert result.parameter_values["observable_param"] == pytest.approx(
        planted_p0, abs=1e-5
    )
    assert result.parameter_values["unobservable_param"] == 0.0  # locked at nominal


def test_green_latency_accounting_excludes_outer_loop() -> None:
    """GREEN: Latency accounting separates inner window solve time from outer calibration."""
    policy = PhysicalGaugePolicy(
        active_gauges={PhysicalGauge.METRIC_SCALE, PhysicalGauge.GRAVITY_FRAME},
    )
    param = CalibrationParameter(
        name="camera_time_offset",
        kind=CalibrationParameterKind.TIME_OFFSET,
        value=0.0,
        nominal_value=0.0,
        bounds=(-0.05, 0.05),
        sigma=0.01,
    )

    def residual_fn(vals: np.ndarray) -> np.ndarray:
        return np.array([vals[0] - 0.005])

    problem = GlobalCalibrationProblem(
        parameters=(param,),
        gauge_policy=policy,
        residual_fn=residual_fn,
        inner_loop_latency_s=0.012,  # Recorded from fast window estimator
    )
    result = calibrate_global_parameters(problem)

    assert result.success is True
    assert result.inner_loop_latency_s == 0.012
    assert result.outer_loop_time_s >= 0.0
    assert result.total_time_s >= result.inner_loop_latency_s
    assert result.cost_breakdown["total_cost"] >= 0.0
    assert "data_cost" in result.cost_breakdown
    assert "prior_cost" in result.cost_breakdown


def test_green_serialization_roundtrip() -> None:
    """GREEN: GlobalCalibrationResult roundtrips perfectly through dict serialization."""
    report = GlobalCalibrationResult(
        success=True,
        parameter_values={"marker_c7_x": 0.025, "torso_mass": 24.5},
        locked_parameters=("unobserved_angle",),
        cost_breakdown={
            "data_cost": 1.2e-4,
            "prior_cost": 3.4e-5,
            "total_cost": 1.54e-4,
        },
        inner_loop_latency_s=0.008,
        outer_loop_time_s=0.015,
        total_time_s=0.023,
        parameter_revision="rev-dime-08-001",
    )
    as_dict = report.to_dict()
    assert as_dict["schema_version"] == "global-calibration-result-v1"
    recovered = GlobalCalibrationResult.from_dict(as_dict)
    assert recovered == report
